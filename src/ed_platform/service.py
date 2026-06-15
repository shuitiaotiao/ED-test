from __future__ import annotations

import base64
import io
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .benchmark import compact_result_summary, compare_cases, discover_case_dirs
from .schemas import UploadedEntry, ValidationRequest


ENERGY_ABS_TOLERANCE = 1.0e-3
GREEN_RELATIVE_TOLERANCE = 1.0e-1


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
                dest.write(source.read())


def _materialize_uploads(request: ValidationRequest, input_root: Path) -> None:
    entries = request.files
    if len(entries) == 1 and entries[0].path.lower().endswith(".zip"):
        _extract_zip_entry(entries[0], input_root)
        return
    _write_regular_files(entries, input_root)


def _make_check(
    *,
    name: str,
    status: str,
    summary: str,
    reference_source: str | None = None,
    metric: float | None = None,
    tolerance: float | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "summary": summary,
        "reference_source": reference_source,
        "metric": metric,
        "tolerance": tolerance,
        "details": details or {},
    }


def _build_energy_check(summary: dict[str, Any]) -> dict[str, Any]:
    qmc_energy = summary.get("qmc_energy")
    if qmc_energy is None:
        return _make_check(
            name="Energy",
            status="fail",
            summary="QMC energy could not be parsed from the uploaded outputs.",
        )

    ed_energy = summary.get("ed_energy")
    ed_minus_qmc = summary.get("ed_minus_qmc")
    if ed_energy is not None and ed_minus_qmc is not None:
        delta = abs(float(ed_minus_qmc))
        passed = delta <= ENERGY_ABS_TOLERANCE
        status = "pass" if passed else "fail"
        return _make_check(
            name="Energy",
            status=status,
            summary=(
                f"Exact ED energy comparison {'passed' if passed else 'failed'} with |ED - QMC| = {delta:.6g}."
            ),
            reference_source="exact_ed_energy",
            metric=delta,
            tolerance=ENERGY_ABS_TOLERANCE,
            details={
                "qmc_energy": qmc_energy,
                "reference_energy": ed_energy,
                "delta": ed_minus_qmc,
            },
        )

    if summary.get("ed_error"):
        note = "Exact ED is unavailable in the current environment, so the energy gate was skipped."
        if summary.get("one_body_energy") is not None:
            note += f" One-body reference energy = {summary['one_body_energy']:.6g}."
        return _make_check(
            name="Energy",
            status="skipped",
            summary=note,
            reference_source="exact_ed_unavailable",
            details={"qmc_energy": qmc_energy, "ed_error": summary.get("ed_error")},
        )

    return _make_check(
        name="Energy",
        status="skipped",
        summary="No enforceable energy comparison was produced for this case.",
        details={"qmc_energy": qmc_energy},
    )


def _build_green_check(summary: dict[str, Any]) -> dict[str, Any]:
    green_error = summary.get("green_error")
    green_shape_error = summary.get("green_comparison_error")
    if green_shape_error:
        return _make_check(
            name="Green function",
            status="fail",
            summary=f"Green comparison failed because the matrix shapes do not match: {green_shape_error}",
            reference_source=summary.get("green_reference_source"),
        )

    if green_error:
        return _make_check(
            name="Green function",
            status="fail",
            summary=f"Green comparison failed because the uploaded Green data is unavailable: {green_error}",
            reference_source=summary.get("green_reference_source"),
        )

    rel_up = summary.get("green_relative_frobenius_up")
    rel_dn = summary.get("green_relative_frobenius_dn")
    if rel_up is None or rel_dn is None:
        return _make_check(
            name="Green function",
            status="skipped",
            summary="No enforceable Green-function comparison was produced for this case.",
            reference_source=summary.get("green_reference_source"),
        )

    worst = max(abs(float(rel_up)), abs(float(rel_dn)))
    passed = worst <= GREEN_RELATIVE_TOLERANCE
    status = "pass" if passed else "fail"
    return _make_check(
        name="Green function",
        status=status,
        summary=(
            f"Relative Frobenius comparison {'passed' if passed else 'failed'} with max(relF_up, relF_dn) = {worst:.6g}."
        ),
        reference_source=summary.get("green_reference_source"),
        metric=worst,
        tolerance=GREEN_RELATIVE_TOLERANCE,
        details={
            "relative_frobenius_up": rel_up,
            "relative_frobenius_dn": rel_dn,
            "trace_qmc_up": summary.get("green_trace_qmc_up"),
            "trace_qmc_dn": summary.get("green_trace_qmc_dn"),
            "trace_ref_up": summary.get("green_trace_ref_up"),
            "trace_ref_dn": summary.get("green_trace_ref_dn"),
        },
    )


def _build_case_payload(summary: dict[str, Any]) -> dict[str, Any]:
    energy = _build_energy_check(summary)
    green = _build_green_check(summary)
    checks = [energy, green]

    failures = [check["summary"] for check in checks if check["status"] == "fail"]
    passes = [check for check in checks if check["status"] == "pass"]

    if failures:
        status = "fail"
        headline = failures[0]
    elif passes:
        status = "pass"
        headline = "All active checks passed for this case."
    else:
        status = "fail"
        headline = "No active validation checks were available for this case."
        failures.append(headline)

    return {
        "case": summary["case"],
        "lattice": f"{summary['lx']}x{summary['ly']}",
        "status": status,
        "headline": headline,
        "checks": checks,
        "failures": failures,
        "notes": summary.get("benchmark_notes", []),
        "metrics": {
            "qmc_energy": summary.get("qmc_energy"),
            "qmc_energy_source": summary.get("qmc_energy_source"),
            "ed_energy": summary.get("ed_energy"),
            "one_body_energy": summary.get("one_body_energy"),
            "ed_minus_qmc": summary.get("ed_minus_qmc"),
            "green_reference_source": summary.get("green_reference_source"),
        },
    }


def _count_check_statuses(cases: list[dict[str, Any]], check_name: str) -> dict[str, int]:
    totals = {"pass": 0, "fail": 0, "skipped": 0}
    for case in cases:
        for check in case["checks"]:
            if check["name"] == check_name:
                totals[check["status"]] += 1
    return totals


def _build_validation_payload(request: ValidationRequest, results: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [compact_result_summary(result) for result in results]
    cases = [_build_case_payload(summary) for summary in summaries]
    failed_cases = [case for case in cases if case["status"] == "fail"]
    passed_cases = [case for case in cases if case["status"] == "pass"]

    return {
        "label": request.label,
        "created_at": _utc_timestamp(),
        "case_count": len(cases),
        "overall_status": "fail" if failed_cases else "pass",
        "summary": {
            "passed_cases": len(passed_cases),
            "failed_cases": len(failed_cases),
            "energy": _count_check_statuses(cases, "Energy"),
            "green": _count_check_statuses(cases, "Green function"),
        },
        "rules": {
            "energy_abs_tolerance": ENERGY_ABS_TOLERANCE,
            "green_relative_tolerance": GREEN_RELATIVE_TOLERANCE,
            "energy_policy": "Use exact ED when available; otherwise mark the energy check as skipped.",
            "green_policy": "Use the benchmark Green reference and require matching shapes plus relative Frobenius error below tolerance.",
        },
        "cases": cases,
        "failed_cases": failed_cases,
    }


def validate_upload(request: ValidationRequest) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ed-platform-") as temp_dir:
        input_root = Path(temp_dir) / "input"
        input_root.mkdir(parents=True, exist_ok=True)

        _materialize_uploads(request, input_root)
        case_dirs = discover_case_dirs(input_root)
        if not case_dirs:
            raise ValueError("No ED case directories were found in the uploaded content.")

        results = compare_cases(case_dirs, bc_y=request.bc_y, max_basis_states=request.max_basis_states)
        return _build_validation_payload(request, results)
