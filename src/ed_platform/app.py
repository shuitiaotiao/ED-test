from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import ValidationRequest
from .service import validate_upload


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


@app.post("/api/validate")
def post_validate(request: ValidationRequest) -> dict[str, object]:
    try:
        return validate_upload(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
