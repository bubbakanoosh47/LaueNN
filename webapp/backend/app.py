import copy
import importlib.util
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from webapp.backend.job_runner import JobManager
from webapp.backend.schemas import RunRequest, WorkflowRunRequest

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = WEBAPP_ROOT / "frontend"
PROJECT_DIR = REPO_ROOT / "project"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

STEP_SPECS = {
    "step1": {
        "title": "Step 1 - Generate Dataset",
        "description": "Generate synthetic Laue dataset for the selected metal.",
        "script": "project/Step1/Step1_Generation_dataset_LaueNN.py",
    },
    "step2": {
        "title": "Step 2 - Train Model",
        "description": "Train neural network using generated dataset.",
        "script": "project/Step2/Step2_Training_LaueNN.py",
    },
    "step3": {
        "title": "Step 3 - Simulated Prediction",
        "description": "Evaluate trained model on simulated patterns.",
        "script": "project/Step3/Step3_Prediction_LaueNN.py",
    },
    "step5": {
        "title": "Step 5 - Real Image Metrics",
        "description": "Measure spot/quality metrics on real X-ray Laue images.",
        "script": "project/Step5/Step5_Analyze_Experimental_Patterns_LaueNN.py",
    },
    "step6": {
        "title": "Step 6 - Real Image Prediction",
        "description": "Predict HKL classes on real experimental X-ray images.",
        "script": "project/Step6/Step6_Predict_Experimental_HKL_LaueNN.py",
    },
}

ARTIFACT_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".csv",
    ".json",
    ".txt",
    ".log",
    ".h5",
    ".npz",
    ".pickle",
    ".cor",
}

SKIP_DIR_NAMES = {
    "training_data",
    "testing_data",
    "__pycache__",
}


def _load_all_step_defaults() -> Dict[str, Dict[str, Any]]:
    step_defaults_path = PROJECT_DIR / "step_defaults.py"
    spec = importlib.util.spec_from_file_location("_lauenn_step_defaults", step_defaults_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load defaults from {step_defaults_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.ALL_STEPS)


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _apply_material(config: Dict[str, Any], material: str | None) -> Dict[str, Any]:
    if not material:
        return config
    if "material_" in config:
        config["material_"] = material
    if "material1_" in config:
        config["material1_"] = material
    return config


def _discover_materials(default_material: str | None) -> list[str]:
    materials = set()
    if default_material:
        materials.add(default_material)
    if PROJECT_DIR.exists():
        for item in PROJECT_DIR.iterdir():
            if item.is_dir() and not item.name.lower().startswith("step") and not item.name.startswith("__"):
                materials.add(item.name)
    return sorted(materials) or ["Ni"]


def _build_cli_args(
    step: str,
    config_path: Path,
    output_root: str,
    data_root: str,
    model_root: str | None = None,
) -> list[str]:
    spec = STEP_SPECS[step]
    script = spec["script"]
    if step in {"step1", "step5"}:
        args = [script, "--config", str(config_path), "--output-root", output_root]
        if step == "step1":
            args.append("--skip-preflight")
        return args
    if step in {"step2", "step3"}:
        args = [script, "--config", str(config_path), "--data-root", data_root, "--output-root", output_root]
        if step == "step3" and model_root:
            args.extend(["--model-root", model_root])
        return args
    if step == "step6":
        return [
            script,
            "--config",
            str(config_path),
            "--output-root",
            output_root,
            "--data-root",
            data_root,
        ]
    raise ValueError(f"Unsupported workflow step: {step}")


def _build_workflow_meta(material: str | None = None) -> Dict[str, Any]:
    all_steps = _load_all_step_defaults()
    default_material = all_steps.get("step1", {}).get("material_", "Ni")
    selected_material = material or default_material
    materials = _discover_materials(default_material)
    defaults = {}
    for step_key in STEP_SPECS.keys():
        step_defaults = copy.deepcopy(all_steps.get(step_key, {}))
        defaults[step_key] = _apply_material(step_defaults, selected_material)
    return {
        "default_material": default_material,
        "selected_material": selected_material,
        "materials": materials,
        "steps": [{"key": k, **v} for k, v in STEP_SPECS.items()],
        "defaults": defaults,
    }


def _to_rel_repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _read_job_config(job: Dict[str, Any]) -> Dict[str, Any]:
    config_path = job.get("config_path")
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def _folder_name_from_config(config: Dict[str, Any]) -> Optional[str]:
    material_ = config.get("material_")
    if not material_:
        return None
    material1_ = config.get("material1_", material_)
    prefix = config.get("prefix", "")
    if material_ != material1_:
        return f"{material_}_{material1_}{prefix}"
    return f"{material_}{prefix}"


def _resolve_step_output_dir(job: Dict[str, Any]) -> Optional[Path]:
    step = str(job.get("step", "")).lower()
    if not step:
        return None

    config = _read_job_config(job)
    folder_name = _folder_name_from_config(config)
    if not folder_name:
        return None

    output_root = REPO_ROOT / "project"
    data_root = REPO_ROOT / "project"
    cmd = job.get("cmd", []) or []
    for idx, part in enumerate(cmd):
        if part == "--output-root" and idx + 1 < len(cmd):
            output_root = (REPO_ROOT / cmd[idx + 1]).resolve()
        if part == "--data-root" and idx + 1 < len(cmd):
            data_root = (REPO_ROOT / cmd[idx + 1]).resolve()

    material_root_out = output_root / folder_name
    material_root_data = data_root / folder_name

    if step == "step1":
        return material_root_out
    if step in {"step2", "step3"}:
        return material_root_out
    if step in {"step5", "step6"}:
        subdir = str(config.get("output_subdir", "")).strip()
        if not subdir:
            subdir = "step5_experimental_metrics" if step == "step5" else "step6_prediction_results"
        return material_root_out / subdir
    return material_root_out


def _classify_artifact(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        return "images"
    if suffix == ".pdf":
        return "documents"
    if suffix in {".csv", ".json", ".txt", ".log", ".cor"}:
        return "tables_and_reports"
    if suffix in {".h5", ".npz", ".pickle"}:
        return "models_and_data"
    return "other"


def _discover_artifacts(output_dir: Optional[Path]) -> Dict[str, Any]:
    if output_dir is None or not output_dir.exists():
        return {
            "output_dir": str(output_dir) if output_dir else None,
            "exists": False,
            "summary": None,
            "artifacts": [],
            "grouped": {},
        }

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    artifacts: List[Dict[str, Any]] = []

    for file_path in sorted(output_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in file_path.parts):
            continue
        if file_path.suffix.lower() not in ARTIFACT_EXTENSIONS:
            continue
        category = _classify_artifact(file_path)
        rel_path = _to_rel_repo_path(file_path)
        entry = {
            "name": file_path.name,
            "relative_path": rel_path,
            "size": file_path.stat().st_size,
            "category": category,
            "url": f"/api/artifacts/{rel_path}",
        }
        artifacts.append(entry)
        grouped.setdefault(category, []).append(entry)

    summary = None
    for candidate_name in [
        "step5_summary.json",
        "step6_summary.json",
        "summary.json",
    ]:
        candidate = output_dir / candidate_name
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as file:
                    summary = json.load(file)
                break
            except Exception:
                pass

    return {
        "output_dir": _to_rel_repo_path(output_dir),
        "exists": True,
        "summary": summary,
        "artifacts": artifacts,
        "grouped": grouped,
    }


def _build_job_report(job: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = _resolve_step_output_dir(job)
    discovered = _discover_artifacts(output_dir)
    report = {
        "job": job,
        "step": job.get("step"),
        "material": job.get("material"),
        "status": job.get("status"),
        "returncode": job.get("returncode"),
        "discovery": discovered,
    }
    return report


def _workflow_roots(workflow_id: str, output_root: str) -> Dict[str, str]:
    base = (REPO_ROOT / output_root / "web_runs" / workflow_id).resolve()
    roots = {
        "workflow_base": str(base),
        "step1": str(base / "step1"),
        "step2": str(base / "step2"),
        "step3": str(base / "step3"),
        "step5": str(base / "step5"),
        "step6": str(base / "step6"),
    }
    for _, path in roots.items():
        Path(path).mkdir(parents=True, exist_ok=True)
    return roots

app = FastAPI(title="LaueNN Web API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = JobManager(root=(Path(__file__).resolve().parent / "jobs"))


@app.post("/api/run", response_model=dict)
async def run_job(req: RunRequest):
    """Start a job by providing a CLI-style command list.
    Example body: {"cli_args": ["project/Step1/Step1_Generation_dataset_LaueNN.py", "--out", "project/tmp"]}
    The backend will run: `python <cli_args...>` as a subprocess.
    """
    if not req.cli_args:
        raise HTTPException(status_code=400, detail="cli_args must be a non-empty list")
    job = jobs.start_job(req.cli_args, cwd=str(REPO_ROOT))
    return {"job_id": job["id"], "status": job["status"]}


@app.get("/api/workflow/meta")
async def workflow_meta(material: str | None = Query(default=None)):
    return _build_workflow_meta(material=material)


@app.get("/api/workflow/defaults/{step}")
async def workflow_step_defaults(step: str, material: str | None = Query(default=None)):
    meta = _build_workflow_meta(material=material)
    if step not in meta["defaults"]:
        raise HTTPException(status_code=404, detail=f"Unknown step: {step}")
    return meta["defaults"][step]


@app.post("/api/workflow/run")
async def workflow_run(req: WorkflowRunRequest):
    step = req.step.lower()
    if step not in STEP_SPECS:
        raise HTTPException(status_code=400, detail=f"Unsupported step: {req.step}")

    workflow_id = req.workflow_id or uuid.uuid4().hex[:12]
    roots = _workflow_roots(workflow_id, req.output_root)

    all_steps = _load_all_step_defaults()
    base_defaults = copy.deepcopy(all_steps.get(step, {}))
    merged = _apply_material(base_defaults, req.material)
    merged = _deep_merge(merged, req.overrides or {})

    material_name = req.material or merged.get("material_", "material")
    config_path = CONFIG_DIR / f"{step}_{material_name}.json"
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(merged, config_file, indent=2)

    if step == "step1":
        output_root = roots["step1"]
        data_root = roots["step1"]
        model_root = None
    elif step == "step2":
        output_root = roots["step2"]
        data_root = roots["step1"]
        model_root = None
    elif step == "step3":
        output_root = roots["step3"]
        data_root = roots["step1"]
        model_root = roots["step2"]
    elif step == "step5":
        output_root = roots["step5"]
        data_root = roots["step5"]
        model_root = None
    elif step == "step6":
        output_root = roots["step6"]
        data_root = roots["step2"]
        model_root = None
    else:
        output_root = req.output_root
        data_root = req.data_root
        model_root = None

    cli_args = _build_cli_args(step, config_path, output_root, data_root, model_root=model_root)
    job = jobs.start_job(cli_args, cwd=str(REPO_ROOT))
    jobs.update_job(
        job["id"],
        {
            "step": step,
            "material": material_name,
            "config_path": str(config_path),
            "workflow_id": workflow_id,
            "workflow_roots": roots,
        },
    )
    return {
        "job_id": job["id"],
        "status": job["status"],
        "step": step,
        "material": material_name,
        "workflow_id": workflow_id,
    }


@app.get("/api/jobs")
async def list_jobs():
    all_jobs = jobs.list_jobs()
    return sorted(all_jobs, key=lambda item: item.get("start_time", ""), reverse=True)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/api/jobs/{job_id}/report")
async def get_job_report(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    report = _build_job_report(job)
    discovered = report.get("discovery", {})
    jobs.update_job(
        job_id,
        {
            "output_dir": discovered.get("output_dir"),
            "artifacts": [item.get("relative_path") for item in discovered.get("artifacts", [])],
        },
    )
    return report


@app.get("/api/jobs/{job_id}/logs")
async def get_logs(job_id: str):
    log = jobs.get_job_log(job_id)
    if log is None:
        raise HTTPException(status_code=404, detail="job or log not found")
    return PlainTextResponse(content=log)


@app.get("/api/artifacts/{path:path}")
async def get_artifact(path: str):
    p = (REPO_ROOT / path).resolve()
    if not str(p).startswith(str(REPO_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="invalid artifact path")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(str(p))


@app.get("/health")
async def health():
    return {"status": "ok"}


# IMPORTANT: mount frontend after API routes so /api/* paths are not shadowed by static files.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
