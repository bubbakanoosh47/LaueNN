import json
import subprocess
import sys
from pathlib import Path
import importlib.util
import types


def _load_module_from_path(name: str, path: Path) -> types.ModuleType:
    """Load a module from a filesystem path and return the module object."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step0_generates_timestamp_and_step1_accepts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script0 = repo_root / "project" / "Step0" / "Step0_Detector_Preflight_LaueNN.py"
    config = repo_root / "project" / "Step0" / "Step0_config.example.json"

    # Run Step0 to generate a fresh preflight report into tmp_path
    subprocess.run(
        [sys.executable, str(script0), "--config", str(config), "--output-root", str(tmp_path)],
        check=True,
    )

    report_path = tmp_path / "Ni" / "detector_preflight" / "detector_preflight_report.json"
    assert report_path.exists(), f"Report not found at {report_path}"

    with report_path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)

    # Step0 should include the generated timestamp key used by Step1
    assert "generated_at_utc" in report, "generated_at_utc missing from Step0 report"
    assert "validation" in report, "validation block missing from report"
    assert report["validation"].get("is_valid") is True

    # Import Step1 file and call the checker directly to ensure it accepts the report
    step1_path = repo_root / "project" / "Step1" / "Step1_Generation_dataset_LaueNN.py"
    step1 = _load_module_from_path("step1_check", step1_path)

    ok, msg = step1._check_step0_preflight("Ni", tmp_path, 30)
    assert ok, f"Step1 preflight checker rejected report: {msg}"
