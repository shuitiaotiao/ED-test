from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import ValidationRequest
from .service import artifact_path, list_runs, load_run, validate_upload


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="ED Web Platform",
    summary="Upload ED/QMC result bundles and compare energies plus Green functions in a browser.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs")
def get_runs(limit: int = Query(default=8, ge=1, le=30)) -> dict[str, object]:
    return {"runs": list_runs(limit=limit)}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, object]:
    try:
        return load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc


@app.get("/api/runs/{run_id}/artifacts/{artifact_key}")
def get_artifact(run_id: str, artifact_key: str) -> FileResponse:
    try:
        target = artifact_path(run_id, artifact_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown artifact: {artifact_key}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc
    return FileResponse(target, filename=target.name)


@app.post("/api/validate")
def post_validate(request: ValidationRequest) -> dict[str, object]:
    try:
        return validate_upload(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
