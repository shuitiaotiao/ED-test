from __future__ import annotations

import base64
import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from .benchmark import (
    compact_result_summary,
    compare_cases,
    discover_case_dirs,
    write_validation_report,
    write_validation_table_csv,
    write_validation_table_markdown,
)
from .schemas import UploadedEntry, ValidationRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_ROOT = PROJECT_ROOT / "storage" / "runs"
ARTIFACT_FILENAMES = {
    "report_json": "ed_validation_report.json",
    "comparison_csv": "ed_validation_comparison.csv",
    "comparison_md": "ed_validation_comparison.md",
}


def ensure_storage_root() -> Path:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    return STORAGE_ROOT


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify_label(label: str | None) -> str:
    if not label:
        return "manual-upload"
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in label.strip())
    collapsed = "-".join(part for part in cleaned.split("-") if part)
    return collapsed[:40] or "manual-upload"


def _safe_relative_path(raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/").strip()
    path = PurePosixPath(normalized)
    if not path.parts or path.is_absolute():
        raise ValueError(f"Unsupported upload path: {raw_path}")
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe upload path: {raw_path}")
    return Path(*path.parts)


def _decode_entry(entry: UploadedEntry) -> bytes:
    try:
        return base64.b64decode(entry.content_base64, validate=True)
    except Exception as exc:  # pragma: no cover - exercised through API
        raise ValueError(f"Failed to decode upload {entry.path}") from exc


def _write_regular_files(entries: list[UploadedEntry], input_root: Path) -> None:
    for entry in entries:
        target = input_root / _safe_relative_path(entry.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_decode_entry(entry))


def _extract_zip_entry(entry: UploadedEntry, input_root: Path) -> None:
    zip_bytes = io.BytesIO(_decode_entry(entry))
    with zipfile.ZipFile(zip_bytes) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = input_root / _safe_relative_path(member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)


def _materialize_uploads(request: ValidationRequest, input_root: Path) -> None:
    entries = request.files
    if len(entries) == 1 and entries[0].path.lower().endswith(".zip"):
        _extract_zip_entry(entries[0], input_root)
        return
    _write_regular_files(entries, input_root)


def _result_status(summary: dict[str, Any]) -> dict[str, str]:
    green_status = "missing"
    if summary.get("green_relative_frobenius_up") is not None:
        green_status = "compared"
    elif summary.get("green_comparison_error"):
        green_status = "shape-mismatch"
    elif summary.get("green_error"):
        green_status = "unavailable"

    ed_status = "skipped"
    if summary.get("ed_energy") is not None:
        ed_status = "available"
    elif summary.get("ed_error"):
        ed_status = "fallback"

    return {"green_status": green_status, "ed_status": ed_status}


def _build_run_payload(
    request: ValidationRequest,
    run_id: str,
    run_root: Path,
    case_dirs: list[Path],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for result in results:
        summary = compact_result_summary(result)
        summary.update(_result_status(summary))
        summaries.append(summary)

    return {
        "run_id": run_id,
        "label": request.label,
        "label_slug": _slugify_label(request.label),
        "created_at": _utc_timestamp(),
        "bc_y": request.bc_y,
        "max_basis_states": request.max_basis_states,
        "case_count": len(case_dirs),
        "case_names": [case_dir.name for case_dir in case_dirs],
        "cases": summaries,
        "downloads": {
            artifact_key: f"/api/runs/{run_id}/artifacts/{artifact_key}" for artifact_key in ARTIFACT_FILENAMES
        },
        "storage": {
            "run_root": str(run_root),
            "input_root": str(run_root / "input"),
            "report_root": str(run_root / "reports"),
        },
    }


def validate_upload(request: ValidationRequest) -> dict[str, Any]:
    ensure_storage_root()
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{_slugify_label(request.label)}-{uuid4().hex[:8]}"
    run_root = STORAGE_ROOT / run_id
    input_root = run_root / "input"
    report_root = run_root / "reports"
    input_root.mkdir(parents=True, exist_ok=False)
    report_root.mkdir(parents=True, exist_ok=False)

    _materialize_uploads(request, input_root)
    case_dirs = discover_case_dirs(input_root)
    if not case_dirs:
        raise ValueError("No ED case directories were found in the uploaded content.")

    results = compare_cases(case_dirs, bc_y=request.bc_y, max_basis_states=request.max_basis_states)

    report_json = report_root / ARTIFACT_FILENAMES["report_json"]
    report_csv = report_root / ARTIFACT_FILENAMES["comparison_csv"]
    report_md = report_root / ARTIFACT_FILENAMES["comparison_md"]
    write_validation_report(results, report_json)
    write_validation_table_csv(results, report_csv)
    write_validation_table_markdown(results, report_md)

    payload = _build_run_payload(request, run_id, run_root, case_dirs, results)
    (run_root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_run(run_id: str) -> dict[str, Any]:
    summary_path = STORAGE_ROOT / run_id / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(run_id)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def list_runs(limit: int = 8) -> list[dict[str, Any]]:
    ensure_storage_root()
    summaries: list[dict[str, Any]] = []
    for summary_path in STORAGE_ROOT.glob("*/summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        summaries.append(
            {
                "run_id": summary["run_id"],
                "label": summary.get("label"),
                "created_at": summary["created_at"],
                "case_count": summary["case_count"],
                "case_names": summary["case_names"],
            }
        )
    summaries.sort(key=lambda item: item["created_at"], reverse=True)
    return summaries[:limit]


def artifact_path(run_id: str, artifact_key: str) -> Path:
    if artifact_key not in ARTIFACT_FILENAMES:
        raise KeyError(artifact_key)
    target = STORAGE_ROOT / run_id / "reports" / ARTIFACT_FILENAMES[artifact_key]
    if not target.exists():
        raise FileNotFoundError(str(target))
    return target
