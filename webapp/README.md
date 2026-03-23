Local webapp (FastAPI backend + guided React SPA)

Quick start (assumes Python 3.10+ and a virtualenv):

1. Create and activate a virtualenv (if not already active):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install backend dependencies:

```bash
pip install -r webapp/backend/requirements.txt
```

3. Run the backend server (development):

```bash
uvicorn webapp.backend.app:app --reload --host 0.0.0.0 --port 8001
```

If your shell is not using `.venv`, run with explicit interpreter:

```bash
.venv/bin/python -m uvicorn webapp.backend.app:app --reload --host 0.0.0.0 --port 8001
```

4. Open http://localhost:8001 in your browser — the React SPA is served from the backend static files.

Guided workflow in the GUI:
- Select a material (metal) from the top dropdown.
- Run Step 1 (dataset generation) → Step 2 (training) → Step 3 (simulated evaluation).
- Use Step 5 and Step 6 for real X-ray image analysis and HKL prediction.
- Edit each step's config fields directly in the form before launching.
- Monitor status in the Jobs panel and open logs from the GUI.

API endpoints used by the GUI:
- `GET /api/workflow/meta` — materials, step definitions, and defaults.
- `GET /api/workflow/defaults/{step}` — defaults for one step/material.
- `POST /api/workflow/run` — launch a selected workflow step.
- `GET /api/jobs` and `GET /api/jobs/{job_id}/logs` — monitor jobs and logs.

Notes:
- Jobs are run as local subprocesses by the backend and stored under `webapp/backend/jobs/`.
- Generated config files are written to `webapp/backend/configs/`.
- If a step needs extra dependencies (e.g., TensorFlow stack), install them in the same `.venv`.
