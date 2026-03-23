import subprocess
import sys
import threading
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

JOBS_FILE = "jobs.json"

class JobManager:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs_file = self.root / JOBS_FILE
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, "r") as f:
                    self.jobs = json.load(f)
            except Exception:
                self.jobs = {}
        else:
            self.jobs = {}

    def _save(self):
        with open(self.jobs_file, "w") as f:
            json.dump(self.jobs, f, indent=2, default=str)

    def _job_dir(self, job_id: str) -> Path:
        d = self.root / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def start_job(self, cli_args: list, cwd: Optional[str] = None, env: Optional[dict] = None):
        job_id = uuid.uuid4().hex
        job_dir = self._job_dir(job_id)
        log_path = job_dir / "output.log"
        # Prefer the project's .venv Python if it exists (so spawned jobs use the project's virtualenv)
        venv_python = Path.cwd() / '.venv' / 'bin' / 'python'
        python_exec = str(venv_python) if venv_python.exists() else sys.executable
        cmd = [python_exec] + cli_args
        entry = {
            "id": job_id,
            "cmd": cmd,
            "cwd": cwd or str(Path.cwd()),
            "status": "running",
            "start_time": datetime.utcnow().isoformat() + "Z",
            "end_time": None,
            "log": str(log_path),
            "returncode": None,
            "artifacts": [],
        }
        with self._lock:
            self.jobs[job_id] = entry
            self._save()

        def target():
            with open(log_path, "w") as lf:
                try:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=cwd or None,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        env=env,
                        text=True,
                    )
                    proc.wait()
                    rc = proc.returncode
                    entry['returncode'] = rc
                    entry['status'] = 'finished' if rc == 0 else 'failed'
                except Exception as e:
                    lf.write(f"Exception while running job: {e}\n")
                    entry['status'] = 'failed'
                finally:
                    entry['end_time'] = datetime.utcnow().isoformat() + "Z"
                    with self._lock:
                        self.jobs[job_id] = entry
                        self._save()

        t = threading.Thread(target=target, daemon=True)
        t.start()
        return entry

    def list_jobs(self):
        with self._lock:
            return list(self.jobs.values())

    def get_job(self, job_id: str):
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, patch: dict):
        with self._lock:
            if job_id not in self.jobs:
                return None
            self.jobs[job_id].update(patch)
            self._save()
            return self.jobs[job_id]

    def get_job_log(self, job_id: str) -> Optional[str]:
        job = self.get_job(job_id)
        if not job:
            return None
        p = Path(job['log'])
        if not p.exists():
            return ""
        return p.read_text()
