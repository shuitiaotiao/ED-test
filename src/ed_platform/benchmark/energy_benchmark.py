from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .run_helper import parse_doping_token, resolve_hubbard_sector

try:
    from quspin.basis import spinful_fermion_basis_general, spinless_fermion_basis_general
    from quspin.operators import hamiltonian

    HAVE_QUSPIN = True
except ImportError:  # pragma: no cover - depends on user environment
    HAVE_QUSPIN = False
    spinful_fermion_basis_general = None
    spinless_fermion_basis_general = None
    hamiltonian = None


@dataclass
class CaseModel:
    case_dir: str
    folder_name: str
    lx: int
    ly: int
    nsites: int
    nup: int
    ndn: int
    total_electrons: int
    target_doping: float | None
    actual_doping: float | None
    u: float
    vpd: float
    t0: float
    t1: float
    t2: float
    tam: float
    h_pin: float
    pin_right_factor: float
    geom_mode: int | None = None
    theta_x_val: float | None = None
    theta_y_val: float | None = None
    tabc_mode: int | None = None
    pair_source_mode: int | None = None
    pair_frame_mode: int | None = None
    effective_onsite_u: float | None = None
    transformed_n_up_target: int | None = None
    transformed_n_dn_target: int | None = None
    transformed_total_particles: int | None = None
    benchmark_mode: str = "qmc_spinful_baseline"
    science_pph_active_source: str | None = None
    science_pph_input_controls_path: str | None = None
    science_pph_effective_controls_path: str | None = None
    science_pph_controls: dict[str, object] | None = None
    science_pph_input_controls: dict[str, object] | None = None
    science_pph_effective_controls: dict[str, object] | None = None

    @property
    def density(self) -> float:
        return self.total_electrons / float(self.nsites)


SCIENCE_PPH_REPORTED_KEYS = (
    "pair_source_mode",
    "pair_frame_mode",
    "complex_weight_mode",
    "nambu_stage_mode",
    "pph_gauge_mode",
    "pph_selfcons_mode",
    "lambda_d_val",
    "h_d_pair",
    "alpha_pph_trial",
    "mu_pph_trial",
    "science_pph_selfcons_autoload",
    "science_pph_selfcons_target_mode",
    "mu_nambu",
)

GEOM_CYLINDER = 0
GEOM_TORUS_REAL = 1
TABC_REAL_STAGE = 0
TABC_COMPLEX_STAGE = 1
PAIR_SOURCE_OFF = 0
PAIR_SOURCE_DWAVE = 1
PAIR_FRAME_NAMBU = 0
PAIR_FRAME_PPH_SCIENCE = 1


def _parse_float_token(text: str | None) -> float | None:
    if text is None:
        return None
    return float(text.replace("D", "E").replace("d", "e"))


def _extract_scalar(text: str, pattern: str, cast=float):
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    return cast(match.group(1))


def _read_parameter_file(path: Path) -> dict[str, float | int | None]:
    text = path.read_text(encoding="utf-8")
    return {
        "lx": _extract_scalar(text, r"integer\s*,\s*parameter\s*::\s*lx\s*=\s*(\d+)", int),
        "ly": _extract_scalar(text, r"ly\s*=\s*(\d+)\s*,\s*lxy", int),
        "nup": _extract_scalar(text, r"integer\s*,\s*parameter\s*::\s*NUP\s*=\s*(\d+)", int),
        "ndn": _extract_scalar(text, r"NDN\s*=\s*(\d+)\s*,\s*NELEC", int),
        "geom_mode": _extract_scalar(text, r"integer\s*::\s*geom_mode\s*=\s*([-\d]+)", int),
        "h_pin": _extract_scalar(text, r"real\(sp\)\s*::\s*h_pin\s*=\s*([^\s!\n]+)", _parse_float_token),
        "theta_x_val": _extract_scalar(text, r"real\(sp\)\s*::\s*theta_x_val\s*=\s*([^\s!\n]+)", _parse_float_token),
        "theta_y_val": _extract_scalar(text, r"real\(sp\)\s*::\s*theta_y_val\s*=\s*([^\s!\n]+)", _parse_float_token),
        "pin_right_factor": _extract_scalar(
            text,
            r"real\(sp\)\s*::\s*pin_right_factor\s*=\s*([^\s!\n]+)",
            _parse_float_token,
        ),
        "tabc_mode": _extract_scalar(text, r"integer\s*::\s*tabc_mode\s*=\s*([-\d]+)", int),
        "pair_source_mode": _extract_scalar(text, r"integer\s*::\s*pair_source_mode\s*=\s*([-\d]+)", int),
        "pair_frame_mode": _extract_scalar(text, r"integer\s*::\s*pair_frame_mode\s*=\s*([-\d]+)", int),
    }


def _read_indat(path: Path) -> dict[str, float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        raise ValueError(f"Unexpected in.dat format in {path}")

    hoppings = _parse_float_row(lines[4], expected=4)
    interactions = _parse_float_row(lines[5], expected=2)
    while len(hoppings) < 4:
        hoppings.append(0.0)
    while len(interactions) < 2:
        interactions.append(0.0)

    return {
        "t0": hoppings[0],
        "t1": hoppings[1],
        "t2": hoppings[2],
        "tam": hoppings[3],
        "u": interactions[0],
        "vpd": interactions[1],
    }


def _parse_float_row(line: str, expected: int) -> list[float]:
    values: list[float] = []
    for token in line.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    while len(values) < expected:
        values.append(0.0)
    return values[:expected]


def _coalesce(value, default):
    return default if value is None else value


def _parse_numeric_token(token: str) -> complex:
    text = token.strip()
    if not text:
        raise ValueError("Empty token")
    if text.startswith("(") and text.endswith(")") and "," in text:
        real_text, imag_text = text[1:-1].split(",", 1)
        return complex(_parse_float_token(real_text), _parse_float_token(imag_text))
    return complex(text.replace("D", "E").replace("d", "e").replace("i", "j").replace("I", "j"))


def _parse_namelist_scalar(token: str) -> object:
    text = token.strip().rstrip(",")
    if not text:
        raise ValueError("Empty namelist token")

    lowered = text.lower()
    if lowered in (".true.", "true"):
        return True
    if lowered in (".false.", "false"):
        return False
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text[1:-1]
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)

    try:
        return float(text.replace("D", "E").replace("d", "e"))
    except ValueError:
        return text


def _read_simple_namelist(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("!", 1)[0].strip()
        if not line or line.startswith("&") or line == "/":
            continue
        for match in re.finditer(r"([A-Za-z_]\w*)\s*=\s*([^,/\n]+)", line):
            key = match.group(1).lower()
            values[key] = _parse_namelist_scalar(match.group(2))
    return values


def _filter_science_pph_controls(values: dict[str, object]) -> dict[str, object]:
    return {key: values[key] for key in SCIENCE_PPH_REPORTED_KEYS if key in values}


def _read_science_pph_case_controls(case_path: Path) -> dict[str, object]:
    input_path = case_path / "science_pph_controls.nml"
    effective_path = case_path / "science_pph_effective_controls.nml"

    input_controls = _filter_science_pph_controls(_read_simple_namelist(input_path)) if input_path.exists() else {}
    effective_controls = _filter_science_pph_controls(_read_simple_namelist(effective_path)) if effective_path.exists() else {}

    if effective_controls:
        active_source = effective_path.name
        active_controls = dict(effective_controls)
    elif input_controls:
        active_source = input_path.name
        active_controls = dict(input_controls)
    else:
        active_source = None
        active_controls = {}

    return {
        "science_pph_active_source": active_source,
        "science_pph_input_controls_path": str(input_path) if input_path.exists() else None,
        "science_pph_effective_controls_path": str(effective_path) if effective_path.exists() else None,
        "science_pph_controls": active_controls or None,
        "science_pph_input_controls": input_controls or None,
        "science_pph_effective_controls": effective_controls or None,
    }


def _control_int(values: dict[str, object] | None, key: str, fallback: int) -> int:
    if values is None or key not in values or values[key] is None:
        return fallback
    return int(values[key])


def _control_float(values: dict[str, object] | None, key: str, fallback: float = 0.0) -> float:
    if values is None or key not in values or values[key] is None:
        return fallback
    return float(values[key])


def _case_requires_pairing_benchmark(
    pair_source_mode: int,
    pair_frame_mode: int,
    science_pph_controls: dict[str, object] | None,
) -> bool:
    if pair_source_mode == PAIR_SOURCE_OFF:
        return False

    h_d_pair = abs(_control_float(science_pph_controls, "h_d_pair", 0.0))
    lambda_d_val = abs(_control_float(science_pph_controls, "lambda_d_val", 0.0))
    nambu_stage_mode = _control_int(science_pph_controls, "nambu_stage_mode", 0)

    if pair_frame_mode == PAIR_FRAME_PPH_SCIENCE and h_d_pair > 1.0e-12:
        return True
    if pair_frame_mode == PAIR_FRAME_NAMBU and nambu_stage_mode != 0 and lambda_d_val > 1.0e-12:
        return True
    return False


def _load_matrix_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path))

    rows: list[list[complex]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries = [_parse_numeric_token(token) for token in stripped.replace(",", " ").split()]
        rows.append(entries)

    if not rows:
        raise ValueError(f"No numeric data found in {path}")

    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Inconsistent row width in {path}")

    matrix = np.array(rows, dtype=np.complex128)
    if np.max(np.abs(matrix.imag)) < 1.0e-12:
        return matrix.real
    return matrix


def _load_coordinate_matrix_file(path: Path) -> np.ndarray:
    rows: list[tuple[int, int, float]] = []
    max_i = 0
    max_j = 0
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.lower().startswith("r_x"):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        i = int(parts[0])
        j = int(parts[1])
        value = float(parts[2].replace("D", "E").replace("d", "e"))
        rows.append((i, j, value))
        max_i = max(max_i, i)
        max_j = max(max_j, j)

    if not rows:
        raise ValueError(f"No coordinate-matrix data found in {path}")

    matrix = np.zeros((max_i, max_j), dtype=np.float64)
    for i, j, value in rows:
        matrix[i - 1, j - 1] = value
    return matrix


def parse_case_folder_name(folder_name: str) -> dict[str, float | int | str | None]:
    lx_match = re.search(r"Lx(\d+)", folder_name)
    ly_match = re.search(r"Ly(\d+)", folder_name)
    total_electron_match = re.search(r"N(\d+)", folder_name)
    u_match = re.search(r"_u([-]?[\d.]+)", folder_name)
    tp_match = re.search(r"_tp([-]?[\d.]+)", folder_name)
    actual_doping_match = re.search(r"_act([-]?[\d.]+)", folder_name)
    requested_doping_match = re.search(r"_d(.+?)(?:_act|_u|_tp|_N|$)", folder_name)

    if not (lx_match and ly_match and total_electron_match and u_match):
        raise ValueError(f"Unrecognized case folder format: {folder_name}")

    target_doping = None
    requested_token = None
    if requested_doping_match:
        requested_token = requested_doping_match.group(1)
        try:
            target_doping, _ = parse_doping_token(requested_token.replace("_", "/"))
        except ValueError:
            target_doping = None

    return {
        "lx": int(lx_match.group(1)),
        "ly": int(ly_match.group(1)),
        "total_electrons": int(total_electron_match.group(1)),
        "u": float(u_match.group(1)),
        "t1": float(tp_match.group(1)) if tp_match else 0.0,
        "actual_doping": float(actual_doping_match.group(1)) if actual_doping_match else None,
        "target_doping": target_doping,
        "requested_token": requested_token,
    }


def _is_case_directory(path: Path) -> bool:
    return path.is_dir() and (path / "parameter.f90").exists() and (path / "in.dat").exists()


def discover_case_dirs(root: str | Path) -> list[Path]:
    root_path = Path(root).resolve()
    if not root_path.exists():
        return []

    if _is_case_directory(root_path):
        return [root_path]

    direct_case_dirs = sorted(
        path for path in root_path.iterdir() if path.name.startswith("Lx") and _is_case_directory(path)
    )
    if direct_case_dirs:
        return direct_case_dirs

    data_container_dirs = sorted(
        path for path in root_path.iterdir() if path.is_dir() and path.name.lower().startswith("data")
    )
    nested_case_dirs: list[Path] = []
    if data_container_dirs:
        nested_case_dirs = sorted(
            {
                path.resolve()
                for data_dir in data_container_dirs
                for path in data_dir.rglob("Lx*")
                if path.name.startswith("Lx") and _is_case_directory(path)
            },
            key=lambda path: str(path).lower(),
        )
        if nested_case_dirs:
            return nested_case_dirs

    nested_case_dirs = sorted(
        {
            path.resolve()
            for path in root_path.rglob("Lx*")
            if path.name.startswith("Lx") and _is_case_directory(path)
        },
        key=lambda path: str(path).lower(),
    )
    if nested_case_dirs:
        return nested_case_dirs

    source_dir = root_path / "source"
    if _is_case_directory(source_dir):
        return [source_dir]

    return []


def load_case_model(case_dir: str | Path) -> CaseModel:
    case_path = Path(case_dir).resolve()
    parameter_info = _read_parameter_file(case_path / "parameter.f90")
    indat_info = _read_indat(case_path / "in.dat")
    science_pph_info = _read_science_pph_case_controls(case_path)

    try:
        folder_info = parse_case_folder_name(case_path.name)
    except ValueError:
        folder_info = {}

    lx = int(parameter_info["lx"] or folder_info.get("lx"))
    ly = int(parameter_info["ly"] or folder_info.get("ly"))
    nup = int(parameter_info["nup"])
    ndn = int(parameter_info["ndn"])
    total_electrons = nup + ndn
    actual_doping_from_sector = 1.0 - total_electrons / float(lx * ly)
    pair_source_mode = _control_int(
        science_pph_info["science_pph_controls"],
        "pair_source_mode",
        int(_coalesce(parameter_info["pair_source_mode"], PAIR_SOURCE_OFF)),
    )
    pair_frame_mode = _control_int(
        science_pph_info["science_pph_controls"],
        "pair_frame_mode",
        int(_coalesce(parameter_info["pair_frame_mode"], PAIR_FRAME_NAMBU)),
    )
    effective_onsite_u = -abs(float(indat_info["u"])) if pair_frame_mode == PAIR_FRAME_PPH_SCIENCE else float(indat_info["u"])
    transformed_n_up_target = nup
    transformed_n_dn_target = lx * ly - ndn
    transformed_total_particles = transformed_n_up_target + transformed_n_dn_target
    science_pph_spinor_active = (
        pair_frame_mode == PAIR_FRAME_PPH_SCIENCE
        and _case_requires_pairing_benchmark(pair_source_mode, pair_frame_mode, science_pph_info["science_pph_controls"])
    )
    if science_pph_spinor_active:
        benchmark_mode = "science_pph_spinor"
    elif _case_requires_pairing_benchmark(pair_source_mode, pair_frame_mode, science_pph_info["science_pph_controls"]):
        benchmark_mode = "unsupported_pairing_state"
    else:
        benchmark_mode = "qmc_spinful_baseline"

    return CaseModel(
        case_dir=str(case_path),
        folder_name=case_path.name,
        lx=lx,
        ly=ly,
        nsites=lx * ly,
        nup=nup,
        ndn=ndn,
        total_electrons=total_electrons,
        target_doping=float(folder_info["target_doping"]) if folder_info.get("target_doping") is not None else actual_doping_from_sector,
        actual_doping=float(folder_info["actual_doping"]) if folder_info.get("actual_doping") is not None else actual_doping_from_sector,
        u=float(indat_info["u"]),
        vpd=float(indat_info["vpd"]),
        t0=float(indat_info["t0"]),
        t1=float(indat_info["t1"]),
        t2=float(indat_info["t2"]),
        tam=float(indat_info["tam"]),
        h_pin=float(_coalesce(parameter_info["h_pin"], 0.0)),
        pin_right_factor=float(_coalesce(parameter_info["pin_right_factor"], 1.0)),
        geom_mode=int(parameter_info["geom_mode"]) if parameter_info["geom_mode"] is not None else None,
        theta_x_val=float(_coalesce(parameter_info["theta_x_val"], 0.0)),
        theta_y_val=float(_coalesce(parameter_info["theta_y_val"], 0.0)),
        tabc_mode=int(_coalesce(parameter_info["tabc_mode"], TABC_REAL_STAGE)),
        pair_source_mode=pair_source_mode,
        pair_frame_mode=pair_frame_mode,
        effective_onsite_u=effective_onsite_u,
        transformed_n_up_target=transformed_n_up_target,
        transformed_n_dn_target=transformed_n_dn_target,
        transformed_total_particles=transformed_total_particles,
        benchmark_mode=benchmark_mode,
        science_pph_active_source=science_pph_info["science_pph_active_source"],
        science_pph_input_controls_path=science_pph_info["science_pph_input_controls_path"],
        science_pph_effective_controls_path=science_pph_info["science_pph_effective_controls_path"],
        science_pph_controls=science_pph_info["science_pph_controls"],
        science_pph_input_controls=science_pph_info["science_pph_input_controls"],
        science_pph_effective_controls=science_pph_info["science_pph_effective_controls"],
    )


def hilbert_size(model: CaseModel) -> int:
    if model.benchmark_mode == "science_pph_spinor":
        total_particles = int(_coalesce(model.transformed_total_particles, model.nsites))
        return math.comb(2 * model.nsites, total_particles)
    return math.comb(model.nsites, model.nup) * math.comb(model.nsites, model.ndn)


def _site_index(ix: int, iy: int, ly: int) -> int:
    return ix * ly + iy


def _site_coords(index: int, ly: int) -> tuple[int, int]:
    return index // ly, index % ly


def _uses_complex_stage(model: CaseModel) -> bool:
    return int(_coalesce(model.tabc_mode, TABC_REAL_STAGE)) == TABC_COMPLEX_STAGE or int(
        _coalesce(model.pair_source_mode, PAIR_SOURCE_OFF)
    ) != PAIR_SOURCE_OFF


def _wrap_target_real(
    model: CaseModel,
    ix0: int,
    iy0: int,
    dx: int,
    dy: int,
    phase_x: float,
    phase_y: float,
) -> tuple[bool, int | None, float]:
    nx = ix0 + dx
    ny = iy0 + dy
    hop_phase = 1.0
    geom_mode = int(_coalesce(model.geom_mode, GEOM_CYLINDER))

    if nx < 0 or nx >= model.lx:
        if geom_mode == GEOM_TORUS_REAL:
            nx = (nx + model.lx) % model.lx
            hop_phase *= phase_x
        else:
            return False, None, 0.0

    if ny < 0 or ny >= model.ly:
        ny = (ny + model.ly) % model.ly
        hop_phase *= phase_y

    return True, _site_index(nx, ny, model.ly), hop_phase


def _wrap_target_complex(
    model: CaseModel,
    ix0: int,
    iy0: int,
    dx: int,
    dy: int,
    phase_x: complex,
    phase_y: complex,
) -> tuple[bool, int | None, complex]:
    nx = ix0 + dx
    ny = iy0 + dy
    hop_phase = 1.0 + 0.0j
    geom_mode = int(_coalesce(model.geom_mode, GEOM_CYLINDER))

    if nx < 0 or nx >= model.lx:
        if geom_mode == GEOM_TORUS_REAL:
            hop_phase *= np.conj(phase_x) if nx < 0 else phase_x
            nx = (nx + model.lx) % model.lx
        else:
            return False, None, 0.0 + 0.0j

    if ny < 0 or ny >= model.ly:
        hop_phase *= np.conj(phase_y) if ny < 0 else phase_y
        ny = (ny + model.ly) % model.ly

    return True, _site_index(nx, ny, model.ly), hop_phase


def _add_hop_pair(
    matrix_up: np.ndarray,
    matrix_dn: np.ndarray,
    i: int,
    j: int,
    coeff_up: complex,
    coeff_dn: complex,
) -> None:
    matrix_up[i, j] += coeff_up
    matrix_up[j, i] += np.conj(coeff_up)
    matrix_dn[i, j] += coeff_dn
    matrix_dn[j, i] += np.conj(coeff_dn)


def build_qmc_one_body_matrices(model: CaseModel) -> tuple[np.ndarray, np.ndarray]:
    matrix_up = np.zeros((model.nsites, model.nsites), dtype=np.complex128)
    matrix_dn = np.zeros((model.nsites, model.nsites), dtype=np.complex128)

    ttp = model.t0 + model.tam
    ttn = model.t0 - model.tam
    pair_frame_mode = int(_coalesce(model.pair_frame_mode, PAIR_FRAME_NAMBU))
    geom_mode = int(_coalesce(model.geom_mode, GEOM_CYLINDER))

    for i in range(model.nsites):
        ix, iy = _site_coords(i, model.ly)

        if geom_mode == GEOM_CYLINDER and (ix == 0 or ix == model.lx - 1) and abs(model.h_pin) > 1.0e-12:
            stag_phase = 1.0 if (ix + iy) % 2 == 0 else -1.0
            pin_factor = model.pin_right_factor if ix == model.lx - 1 else 1.0
            pin_val = pin_factor * stag_phase * model.h_pin
            matrix_up[i, i] += pin_val
            matrix_dn[i, i] -= pin_val

        if _uses_complex_stage(model):
            phase_x = np.exp(1j * float(_coalesce(model.theta_x_val, 0.0)))
            phase_y = np.exp(1j * float(_coalesce(model.theta_y_val, 0.0)))

            valid, j, hop = _wrap_target_complex(model, ix, iy, 1, 0, phase_x, phase_y)
            if valid and j is not None:
                if pair_frame_mode == PAIR_FRAME_PPH_SCIENCE:
                    _add_hop_pair(matrix_up, matrix_dn, i, j, -model.t0 * hop, -model.t0 * hop)
                else:
                    _add_hop_pair(matrix_up, matrix_dn, i, j, -ttp * hop, -ttn * hop)

            valid, j, hop = _wrap_target_complex(model, ix, iy, 0, 1, phase_x, phase_y)
            if valid and j is not None:
                if pair_frame_mode == PAIR_FRAME_PPH_SCIENCE:
                    _add_hop_pair(matrix_up, matrix_dn, i, j, -model.t0 * hop, -model.t0 * hop)
                else:
                    _add_hop_pair(matrix_up, matrix_dn, i, j, -ttn * hop, -ttp * hop)

            valid, j, hop = _wrap_target_complex(model, ix, iy, 1, 1, phase_x, phase_y)
            if valid and j is not None:
                if pair_frame_mode == PAIR_FRAME_PPH_SCIENCE:
                    _add_hop_pair(matrix_up, matrix_dn, i, j, -model.t1 * hop, -model.t1 * hop)
                else:
                    _add_hop_pair(matrix_up, matrix_dn, i, j, -model.t1 * hop, +model.t1 * hop)

            valid, j, hop = _wrap_target_complex(model, ix, iy, 1, -1, phase_x, phase_y)
            if valid and j is not None:
                if pair_frame_mode == PAIR_FRAME_PPH_SCIENCE:
                    _add_hop_pair(matrix_up, matrix_dn, i, j, -model.t1 * hop, -model.t1 * hop)
                else:
                    _add_hop_pair(matrix_up, matrix_dn, i, j, +model.t1 * hop, -model.t1 * hop)
        else:
            phase_x = -1.0 if abs(float(_coalesce(model.theta_x_val, 0.0))) > 1.0e-3 else 1.0
            phase_y = -1.0 if abs(float(_coalesce(model.theta_y_val, 0.0))) > 1.0e-3 else 1.0

            valid, j, hop = _wrap_target_real(model, ix, iy, 1, 0, phase_x, phase_y)
            if valid and j is not None:
                _add_hop_pair(matrix_up, matrix_dn, i, j, -ttp * hop, -ttn * hop)

            valid, j, hop = _wrap_target_real(model, ix, iy, 0, 1, phase_x, phase_y)
            if valid and j is not None:
                _add_hop_pair(matrix_up, matrix_dn, i, j, -ttn * hop, -ttp * hop)

            valid, j, hop = _wrap_target_real(model, ix, iy, 1, 1, phase_x, phase_y)
            if valid and j is not None:
                _add_hop_pair(matrix_up, matrix_dn, i, j, -model.t1 * hop, +model.t1 * hop)

            valid, j, hop = _wrap_target_real(model, ix, iy, 1, -1, phase_x, phase_y)
            if valid and j is not None:
                _add_hop_pair(matrix_up, matrix_dn, i, j, +model.t1 * hop, -model.t1 * hop)

    return matrix_up, matrix_dn


def _science_site_eta(site_index: int, ly: int) -> int:
    ix, iy = _site_coords(site_index, ly)
    return 1 if (ix + iy) % 2 == 0 else -1


def build_pair_source_dwave_matrix(model: CaseModel, pair_amp: float) -> np.ndarray:
    delta = np.zeros((model.nsites, model.nsites), dtype=np.complex128)
    if abs(pair_amp) <= 1.0e-12:
        return delta

    phase_x = np.exp(1j * float(_coalesce(model.theta_x_val, 0.0)))
    phase_y = np.exp(1j * float(_coalesce(model.theta_y_val, 0.0)))

    for i in range(model.nsites):
        ix, iy = _site_coords(i, model.ly)

        valid, j, hop = _wrap_target_complex(model, ix, iy, 1, 0, phase_x, phase_y)
        if valid and j is not None:
            amp = complex(pair_amp) * hop
            delta[i, j] += amp
            delta[j, i] += amp

        valid, j, hop = _wrap_target_complex(model, ix, iy, 0, 1, phase_x, phase_y)
        if valid and j is not None:
            amp = -complex(pair_amp) * hop
            delta[i, j] += amp
            delta[j, i] += amp

    return delta


def build_science_pph_spinor_one_body_matrix(
    model: CaseModel,
    *,
    pair_amp: float,
    mu_trial: float,
    include_interaction_shift: bool,
) -> np.ndarray:
    tk_up, tk_dn = build_qmc_one_body_matrices(model)
    delta_local = build_pair_source_dwave_matrix(model, pair_amp)
    nsites = model.nsites
    spinor = np.zeros((2 * nsites, 2 * nsites), dtype=np.complex128)

    for i in range(nsites):
        eta_i = _science_site_eta(i, model.ly)
        up_shift_i = -float(_coalesce(model.effective_onsite_u, model.u)) - mu_trial if include_interaction_shift else -mu_trial
        dn_shift_i = mu_trial
        for j in range(nsites):
            eta_j = _science_site_eta(j, model.ly)
            spinor[i, j] = tk_up[i, j]
            spinor[nsites + i, nsites + j] = -float(eta_i * eta_j) * np.conj(tk_dn[i, j])
            spinor[nsites + i, j] = -float(eta_i) * delta_local[i, j] / math.sqrt(2.0)
        spinor[i, i] += up_shift_i
        spinor[nsites + i, nsites + i] += dn_shift_i

    spinor[:nsites, nsites:] = spinor[nsites:, :nsites].conj().T
    return spinor


def _science_pph_pair_amp(model: CaseModel) -> float:
    return _control_float(model.science_pph_controls, "h_d_pair", 0.0)


def map_science_pph_spinor_green_to_physical(model: CaseModel, g_spinor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nsites = model.nsites
    green_up = np.asarray(g_spinor[:nsites, :nsites], dtype=np.complex128)
    green_dn = np.zeros((nsites, nsites), dtype=np.complex128)

    for j in range(nsites):
        eta_j = _science_site_eta(j, model.ly)
        for i in range(nsites):
            eta_i = _science_site_eta(i, model.ly)
            green_dn[i, j] = -float(eta_i * eta_j) * g_spinor[nsites + j, nsites + i]
            if i == j:
                green_dn[i, j] += 1.0

    return green_up, green_dn


def measure_science_pph_anomalous_from_green(model: CaseModel, g_spinor: np.ndarray) -> dict[str, object]:
    nsites = model.nsites
    f_block = np.zeros((nsites, nsites), dtype=np.complex128)
    for j in range(nsites):
        eta_j = _science_site_eta(j, model.ly)
        for i in range(nsites):
            f_block[i, j] = float(eta_j) * g_spinor[i, nsites + j]

    s_pair = 0.0 + 0.0j
    d_pair = 0.0 + 0.0j
    norm_pair = float(np.sqrt(np.sum(np.abs(f_block) ** 2) / max(1.0, float(nsites * nsites))))

    for i in range(nsites):
        ix, iy = _site_coords(i, model.ly)
        valid_x, jx, _ = _wrap_target_real(model, ix, iy, 1, 0, 1.0, 1.0)
        valid_y, jy, _ = _wrap_target_real(model, ix, iy, 0, 1, 1.0, 1.0)
        if not valid_x or jx is None:
            jx = i
        if not valid_y or jy is None:
            jy = i
        fx = 0.5 * (f_block[i, jx] + f_block[jx, i])
        fy = 0.5 * (f_block[i, jy] + f_block[jy, i])
        s_pair += 0.5 * (fx + fy)
        d_pair += 0.5 * (fx - fy)

    s_pair /= float(nsites)
    d_pair /= float(nsites)
    return {
        "f_block": f_block,
        "s_pair": s_pair,
        "d_pair": d_pair,
        "norm_pair": norm_pair,
    }


def _matrix_to_quspin_terms(matrix: np.ndarray) -> tuple[list[list[complex | int]], list[list[complex | int]]]:
    hop_terms: list[list[complex | int]] = []
    diag_terms: list[list[complex | int]] = []
    for i in range(matrix.shape[0]):
        diagonal = matrix[i, i]
        if abs(diagonal) > 1.0e-12:
            diag_terms.append([diagonal, i])
        for j in range(matrix.shape[1]):
            if i == j:
                continue
            coeff = matrix[i, j]
            if abs(coeff) > 1.0e-12:
                hop_terms.append([coeff, i, j])
    return hop_terms, diag_terms


def build_standard_spinful_quspin_hamiltonian(model: CaseModel):
    basis = spinful_fermion_basis_general(model.nsites, Nf=(model.nup, model.ndn))
    matrix_up, matrix_dn = build_qmc_one_body_matrices(model)
    hop_up, diag_up = _matrix_to_quspin_terms(matrix_up)
    hop_dn, diag_dn = _matrix_to_quspin_terms(matrix_dn)
    interaction_strength = float(_coalesce(model.effective_onsite_u, model.u))
    interaction_list = [[interaction_strength, i, i] for i in range(model.nsites)]

    static = [
        ["+-|", hop_up],
        ["|+-", hop_dn],
        ["n|n", interaction_list],
        ["n|", diag_up],
        ["|n", diag_dn],
    ]

    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128, **no_checks)
    return H, basis


def build_science_pph_spinor_quspin_hamiltonian(model: CaseModel):
    if abs(model.vpd) > 1.0e-12:
        raise RuntimeError("Science PPH transformed exact benchmark currently supports only vpd = 0.")

    total_particles = int(_coalesce(model.transformed_total_particles, model.nsites))
    basis = spinless_fermion_basis_general(2 * model.nsites, Nf=total_particles)
    spinor_matrix = build_science_pph_spinor_one_body_matrix(
        model,
        pair_amp=_science_pph_pair_amp(model),
        mu_trial=0.0,
        include_interaction_shift=True,
    )
    hop_terms, diag_terms = _matrix_to_quspin_terms(spinor_matrix)
    interaction_strength = float(_coalesce(model.effective_onsite_u, model.u))
    interaction_list = [[interaction_strength, i, model.nsites + i] for i in range(model.nsites)]

    static = [
        ["+-", hop_terms],
        ["n", diag_terms],
        ["nn", interaction_list],
    ]

    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128, **no_checks)
    return H, basis


def build_quspin_hamiltonian(model: CaseModel, bc_y: str = "PBC"):
    if not HAVE_QUSPIN:
        raise RuntimeError("QuSpin is not available in the current Python environment.")
    if model.benchmark_mode == "science_pph_spinor":
        return build_science_pph_spinor_quspin_hamiltonian(model)
    if model.benchmark_mode == "qmc_spinful_baseline":
        return build_standard_spinful_quspin_hamiltonian(model)
    raise RuntimeError("Exact benchmark for this pairing mode is not implemented.")


def run_exact_diagonalization(
    model: CaseModel,
    bc_y: str = "PBC",
    max_basis_states: int = 250_000,
) -> dict[str, object]:
    hs = hilbert_size(model)
    result = {
        "method": "quspin_ed",
        "hilbert_size": hs,
        "feasible": hs <= max_basis_states and HAVE_QUSPIN,
        "energy": None,
        "error": None,
    }

    if model.benchmark_mode == "unsupported_pairing_state":
        result["feasible"] = False
        result["error"] = "This pairing mode is not implemented in the exact benchmark yet."
        return result

    if not HAVE_QUSPIN:
        result["error"] = "QuSpin is not available."
        return result

    if hs > max_basis_states:
        result["error"] = (
            f"Hilbert space too large for full ED: choose({model.nsites},{model.nup}) * "
            f"choose({model.nsites},{model.ndn}) = {hs}"
        )
        return result

    H, basis = build_quspin_hamiltonian(model, bc_y=bc_y)
    eigenvalues, vectors = H.eigsh(k=1, which="SA")
    ground_state = vectors[:, 0]
    result["energy"] = float(np.real_if_close(eigenvalues[0]))
    result["basis_states"] = int(basis.Ns)
    result["ground_state"] = ground_state
    result["basis"] = basis
    return result


def build_one_body_matrix(model: CaseModel, bc_y: str = "PBC") -> np.ndarray:
    if model.benchmark_mode == "science_pph_spinor":
        return build_science_pph_spinor_one_body_matrix(
            model,
            pair_amp=_science_pph_pair_amp(model),
            mu_trial=0.0,
            include_interaction_shift=True,
        )
    matrix_up, _ = build_qmc_one_body_matrices(model)
    return matrix_up


def run_noninteracting_reference(model: CaseModel, bc_y: str = "PBC") -> dict[str, object]:
    if model.benchmark_mode == "science_pph_spinor":
        spinor_matrix = build_science_pph_spinor_one_body_matrix(
            model,
            pair_amp=_science_pph_pair_amp(model),
            mu_trial=0.0,
            include_interaction_shift=True,
        )
        evals, evecs = np.linalg.eigh(spinor_matrix)
        order = np.argsort(evals.real)
        evals = evals[order]
        evecs = evecs[:, order]
        n_particles = int(_coalesce(model.transformed_total_particles, model.nsites))
        occupied = evecs[:, :n_particles]
        g_spinor = occupied @ occupied.conj().T
        green_up, green_dn = map_science_pph_spinor_green_to_physical(model, g_spinor)
        anomalous = measure_science_pph_anomalous_from_green(model, g_spinor)
        energy = float(np.sum(evals[:n_particles].real))
        return {
            "method": "one_body_reference",
            "energy": energy,
            "lowest_eigenvalues_spinor": evals[: min(12, len(evals))].tolist(),
            "green_up": green_up,
            "green_dn": green_dn,
            "spinor_green": g_spinor,
            "science_pph_anomalous": {
                "s_wave_re": float(anomalous["s_pair"].real),
                "s_wave_im": float(anomalous["s_pair"].imag),
                "d_wave_re": float(anomalous["d_pair"].real),
                "d_wave_im": float(anomalous["d_pair"].imag),
                "norm": float(anomalous["norm_pair"]),
            },
        }

    if model.benchmark_mode != "qmc_spinful_baseline":
        return {
            "method": "one_body_reference",
            "energy": None,
            "lowest_eigenvalues_up": None,
            "lowest_eigenvalues_dn": None,
            "green_up": None,
            "green_dn": None,
            "error": "Case activates paired/spinor propagation; spin-conserving one-body reference is not implemented.",
        }

    one_body_up, one_body_dn = build_qmc_one_body_matrices(model)
    evals_up, evecs_up = np.linalg.eigh(one_body_up)
    evals_dn, evecs_dn = np.linalg.eigh(one_body_dn)
    order_up = np.argsort(evals_up.real)
    order_dn = np.argsort(evals_dn.real)
    evals_up = evals_up[order_up]
    evals_dn = evals_dn[order_dn]
    evecs_up = evecs_up[:, order_up]
    evecs_dn = evecs_dn[:, order_dn]
    occ_up = evecs_up[:, : model.nup]
    occ_dn = evecs_dn[:, : model.ndn]
    green_up = occ_up @ occ_up.conj().T
    green_dn = occ_dn @ occ_dn.conj().T
    energy = float(np.sum(evals_up[: model.nup].real) + np.sum(evals_dn[: model.ndn].real))
    return {
        "method": "one_body_reference",
        "energy": energy,
        "lowest_eigenvalues_up": evals_up[: min(8, len(evals_up))].tolist(),
        "lowest_eigenvalues_dn": evals_dn[: min(8, len(evals_dn))].tolist(),
        "green_up": green_up,
        "green_dn": green_dn,
    }


def run_exact_green_reference(model: CaseModel, ed_result: dict[str, object]) -> dict[str, object]:
    if not HAVE_QUSPIN:
        return {"method": "quspin_ed_green", "available": False, "error": "QuSpin is not available."}
    if not ed_result.get("feasible", False):
        return {"method": "quspin_ed_green", "available": False, "error": ed_result.get("error")}

    basis = ed_result["basis"]
    ground_state = ed_result["ground_state"]
    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)

    if model.benchmark_mode == "science_pph_spinor":
        norb = 2 * model.nsites
        g_spinor = np.zeros((norb, norb), dtype=np.complex128)
        for i in range(norb):
            for j in range(norb):
                op = hamiltonian([["+-", [[1.0, i, j]]]], [], basis=basis, dtype=np.complex128, **no_checks)
                g_spinor[i, j] = np.vdot(ground_state, op.dot(ground_state))

        green_up, green_dn = map_science_pph_spinor_green_to_physical(model, g_spinor)
        anomalous = measure_science_pph_anomalous_from_green(model, g_spinor)
        return {
            "method": "quspin_ed_green",
            "available": True,
            "green_up": green_up,
            "green_dn": green_dn,
            "spinor_green": g_spinor,
            "science_pph_anomalous": {
                "s_wave_re": float(anomalous["s_pair"].real),
                "s_wave_im": float(anomalous["s_pair"].imag),
                "d_wave_re": float(anomalous["d_pair"].real),
                "d_wave_im": float(anomalous["d_pair"].imag),
                "norm": float(anomalous["norm_pair"]),
            },
        }

    green_up = np.zeros((model.nsites, model.nsites), dtype=np.complex128)
    green_dn = np.zeros((model.nsites, model.nsites), dtype=np.complex128)

    for i in range(model.nsites):
        for j in range(model.nsites):
            op_up = hamiltonian([["+-|", [[1.0, i, j]]]], [], basis=basis, dtype=np.complex128, **no_checks)
            op_dn = hamiltonian([["|+-", [[1.0, i, j]]]], [], basis=basis, dtype=np.complex128, **no_checks)
            green_up[i, j] = np.vdot(ground_state, op_up.dot(ground_state))
            green_dn[i, j] = np.vdot(ground_state, op_dn.dot(ground_state))

    return {
        "method": "quspin_ed_green",
        "available": True,
        "green_up": green_up,
        "green_dn": green_dn,
    }


def _infer_science_pph_sector_from_physical_greens(model: CaseModel, green_up: np.ndarray, green_dn: np.ndarray) -> dict[str, float]:
    n_up = float(np.trace(green_up).real)
    physical_n_dn = float(np.trace(green_dn).real)
    transformed_dn = float(model.nsites - physical_n_dn)
    return {
        "n_up_re": n_up,
        "n_dn_re": physical_n_dn,
        "tr_g22_re": transformed_dn,
        "transformed_total_re": n_up + transformed_dn,
    }


def _infer_science_pph_sector_from_spinor_green(model: CaseModel, g_spinor: np.ndarray) -> dict[str, float]:
    nsites = model.nsites
    n_up = float(np.trace(g_spinor[:nsites, :nsites]).real)
    tr_g22 = float(np.trace(g_spinor[nsites:, nsites:]).real)
    physical_n_dn = float(model.nsites - tr_g22)
    return {
        "n_up_re": n_up,
        "n_dn_re": physical_n_dn,
        "tr_g22_re": tr_g22,
        "transformed_total_re": n_up + tr_g22,
    }


def _normalize_science_pph_debug_breakdown(model: CaseModel, breakdown: dict[str, object]) -> dict[str, object]:
    if model.benchmark_mode != "science_pph_spinor" or not breakdown.get("available"):
        return breakdown

    normalized = dict(breakdown)
    transformed_dn_re = breakdown.get("tr_g22_re")
    transformed_dn_im = breakdown.get("tr_g22_im")
    if transformed_dn_re is not None:
        normalized["transformed_dn_re"] = transformed_dn_re
        normalized["n_dn_re"] = float(model.nsites - float(transformed_dn_re))
    if transformed_dn_im is not None:
        normalized["transformed_dn_im"] = transformed_dn_im
        normalized["n_dn_im"] = -float(transformed_dn_im)
    return normalized


def run_exact_science_pph_breakdown(
    model: CaseModel,
    ed_result: dict[str, object],
    ed_green: dict[str, object],
) -> dict[str, object]:
    if model.benchmark_mode != "science_pph_spinor":
        return {"available": False, "error": "Case is not in the Science PPH spinor benchmark mode."}
    if not HAVE_QUSPIN:
        return {"available": False, "error": "QuSpin is not available."}
    if not ed_result.get("feasible", False):
        return {"available": False, "error": ed_result.get("error")}
    if not ed_green.get("available", False) or ed_green.get("spinor_green") is None:
        return {"available": False, "error": ed_green.get("error", "Exact spinor Green matrix is unavailable.")}

    basis = ed_result["basis"]
    ground_state = ed_result["ground_state"]
    g_spinor = np.asarray(ed_green["spinor_green"], dtype=np.complex128)
    nsites = model.nsites
    spinor_matrix = build_science_pph_spinor_one_body_matrix(
        model,
        pair_amp=_science_pph_pair_amp(model),
        mu_trial=0.0,
        include_interaction_shift=True,
    )

    upper = g_spinor[:nsites, :nsites]
    lower = g_spinor[nsites:, nsites:]
    e_one_up = 0.0 + 0.0j
    e_one_dn = 0.0 + 0.0j
    e_pair = 0.0 + 0.0j

    for j in range(nsites):
        for i in range(nsites):
            e_one_up += spinor_matrix[j, i] * upper[i, j]
            e_one_dn += spinor_matrix[nsites + j, nsites + i] * lower[i, j]
            e_pair += spinor_matrix[j, nsites + i] * g_spinor[nsites + i, j]
            e_pair += spinor_matrix[nsites + j, i] * g_spinor[i, nsites + j]

    interaction_strength = float(_coalesce(model.effective_onsite_u, model.u))
    if abs(model.vpd) > 1.0e-12:
        return {
            "available": False,
            "error": "Science PPH exact energy breakdown currently supports only vpd = 0.",
        }

    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    interaction_terms = [[interaction_strength, i, nsites + i] for i in range(nsites)]
    interaction_op = hamiltonian([["nn", interaction_terms]], [], basis=basis, dtype=np.complex128, **no_checks)
    e_u = np.vdot(ground_state, interaction_op.dot(ground_state))
    e_v = 0.0 + 0.0j

    sector = _infer_science_pph_sector_from_spinor_green(model, g_spinor)
    return {
        "available": True,
        "source": "quspin_ed_state",
        "e_one_up_re": float(e_one_up.real),
        "e_one_up_im": float(e_one_up.imag),
        "e_one_dn_re": float(e_one_dn.real),
        "e_one_dn_im": float(e_one_dn.imag),
        "e_u_re": float(e_u.real),
        "e_u_im": float(e_u.imag),
        "e_u_proj_re": float(e_u.real),
        "e_u_proj_im": float(e_u.imag),
        "e_u_anom_re": 0.0,
        "e_u_anom_im": 0.0,
        "e_v_re": float(e_v.real),
        "e_v_im": float(e_v.imag),
        "e_pair_re": float(e_pair.real),
        "e_pair_im": float(e_pair.imag),
        "n_up_re": sector["n_up_re"],
        "n_up_im": 0.0,
        "n_dn_re": sector["n_dn_re"],
        "n_dn_im": 0.0,
        "tr_g22_re": sector["tr_g22_re"],
        "tr_g22_im": 0.0,
        "transformed_total_re": sector["transformed_total_re"],
        "e_total_re": float(ed_result["energy"]),
        "e_total_im": 0.0,
    }


def read_qmc_green_matrices(case_dir: str | Path) -> dict[str, object]:
    case_path = Path(case_dir)
    candidate_pairs = (
        ("qmc_green_up.npy", "qmc_green_dn.npy"),
        ("qmc_green_up.dat", "qmc_green_dn.dat"),
        ("green_up.npy", "green_dn.npy"),
        ("green_up.dat", "green_dn.dat"),
    )

    for up_name, dn_name in candidate_pairs:
        up_path = case_path / up_name
        dn_path = case_path / dn_name
        if up_path.exists() and dn_path.exists():
            return {
                "available": True,
                "source": f"{up_name}, {dn_name}",
                "green_up": _load_matrix_file(up_path),
                "green_dn": _load_matrix_file(dn_path),
            }

    dir_r_up = case_path / "dir-rVals" / "gx_up.dat"
    dir_r_dn = case_path / "dir-rVals" / "gx_dn.dat"
    if dir_r_up.exists() and dir_r_dn.exists():
        return {
            "available": True,
            "source": "dir-rVals/gx_up.dat, dir-rVals/gx_dn.dat",
            "green_up": _load_coordinate_matrix_file(dir_r_up),
            "green_dn": _load_coordinate_matrix_file(dir_r_dn),
        }

    return {
        "available": False,
        "source": None,
        "green_up": None,
        "green_dn": None,
        "error": "No QMC Green matrix files found. Expected one of qmc_green_up/down(.dat|.npy), green_up/down(.dat|.npy), or dir-rVals/gx_up.dat + gx_dn.dat.",
    }


def _read_science_pph_observables_file(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.lower().startswith("science") or stripped.lower().startswith("s_wave"):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        try:
            values = [float(token.replace("D", "E").replace("d", "e")) for token in parts[:5]]
        except ValueError:
            continue
        return {
            "s_wave_re": values[0],
            "s_wave_im": values[1],
            "d_wave_re": values[2],
            "d_wave_im": values[3],
            "norm": values[4],
        }
    return None


def read_qmc_science_pph_observables(case_dir: str | Path) -> dict[str, object]:
    case_path = Path(case_dir)
    mixed = _read_science_pph_observables_file(case_path / "science_pph_observables.dat")
    back_propagated = _read_science_pph_observables_file(case_path / "science_pph_bp_observables.dat")
    return {
        "mixed": mixed,
        "back_propagated": back_propagated,
        "available": mixed is not None or back_propagated is not None,
    }


def compare_science_pph_observables(reference: dict[str, float], measured: dict[str, float]) -> dict[str, float]:
    return {
        "delta_s_wave_re": measured["s_wave_re"] - reference["s_wave_re"],
        "delta_s_wave_im": measured["s_wave_im"] - reference["s_wave_im"],
        "delta_d_wave_re": measured["d_wave_re"] - reference["d_wave_re"],
        "delta_d_wave_im": measured["d_wave_im"] - reference["d_wave_im"],
        "delta_norm": measured["norm"] - reference["norm"],
    }


def _read_last_tagged_numeric_line(path: Path, preferred_tags: Sequence[str]) -> tuple[str, int, list[float]] | None:
    if not path.exists():
        return None

    preferred = set(preferred_tags)
    matches: list[tuple[str, int, list[float]]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        tag = parts[0]
        if tag not in preferred:
            continue
        try:
            sample_id = int(parts[1])
            values = [float(token.replace("D", "E").replace("d", "e")) for token in parts[2:]]
        except ValueError:
            continue
        matches.append((tag, sample_id, values))

    if not matches:
        return None
    matches.sort(key=lambda item: (preferred_tags.index(item[0]), item[1]))
    return matches[-1]


def read_qmc_science_pph_energy_breakdown(case_dir: str | Path) -> dict[str, object]:
    case_path = Path(case_dir)
    parsed = _read_last_tagged_numeric_line(
        case_path / "nambu_energy_breakdown_debug.dat",
        ("INIT_SPPH_E", "MEAS_SPPH_E"),
    )
    if parsed is None:
        return {"available": False, "error": "No INIT_SPPH_E or MEAS_SPPH_E entry found in nambu_energy_breakdown_debug.dat."}

    tag, sample_id, values = parsed
    if len(values) < 24:
        return {"available": False, "error": f"Unexpected Science PPH energy breakdown width ({len(values)}) in nambu_energy_breakdown_debug.dat."}

    result = {
        "available": True,
        "source": "nambu_energy_breakdown_debug.dat",
        "tag": tag,
        "sample_id": sample_id,
        "denom_re": values[0],
        "denom_im": values[1],
        "e_one_up_re": values[2],
        "e_one_up_im": values[3],
        "e_one_dn_re": values[4],
        "e_one_dn_im": values[5],
        "e_u_re": values[6],
        "e_u_im": values[7],
        "e_u_proj_re": values[8],
        "e_u_proj_im": values[9],
        "e_u_anom_re": values[10],
        "e_u_anom_im": values[11],
        "e_v_re": values[12],
        "e_v_im": values[13],
        "e_pair_re": values[14],
        "e_pair_im": values[15],
        "n_up_re": values[16],
        "n_up_im": values[17],
        "n_dn_re": values[18],
        "n_dn_im": values[19],
        "tr_g22_re": values[20],
        "tr_g22_im": values[21],
        "e_total_re": values[22],
        "e_total_im": values[23],
    }
    result["transformed_total_re"] = result["n_up_re"] + result["tr_g22_re"]
    return result


def read_qmc_science_pph_projector_debug(case_dir: str | Path) -> dict[str, object]:
    case_path = Path(case_dir)
    parsed = _read_last_tagged_numeric_line(
        case_path / "nambu_green_projector_debug.dat",
        ("INIT_SPPH_P", "MEAS_SPPH_P"),
    )
    if parsed is None:
        return {"available": False, "error": "No INIT_SPPH_P or MEAS_SPPH_P entry found in nambu_green_projector_debug.dat."}

    tag, sample_id, values = parsed
    if len(values) < 8:
        return {"available": False, "error": f"Unexpected Science PPH projector-debug width ({len(values)}) in nambu_green_projector_debug.dat."}

    return {
        "available": True,
        "source": "nambu_green_projector_debug.dat",
        "tag": tag,
        "sample_id": sample_id,
        "denom_re": values[0],
        "denom_im": values[1],
        "trace_re": values[2],
        "trace_im": values[3],
        "idem_resid_re": values[4],
        "idem_resid_im": values[5],
        "herm_resid_re": values[6],
        "herm_resid_im": values[7],
    }


def compare_science_pph_energy_breakdown(reference: dict[str, float], measured: dict[str, float]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in (
        "e_one_up_re",
        "e_one_dn_re",
        "e_u_re",
        "e_v_re",
        "e_pair_re",
        "e_total_re",
        "n_up_re",
        "n_dn_re",
        "tr_g22_re",
        "transformed_total_re",
    ):
        if key in reference and key in measured:
            deltas[f"delta_{key}"] = float(measured[key] - reference[key])
    return deltas


def compare_green_matrices(reference_up: np.ndarray, reference_dn: np.ndarray, qmc_up: np.ndarray, qmc_dn: np.ndarray) -> dict[str, float]:
    diff_up = np.asarray(qmc_up) - np.asarray(reference_up)
    diff_dn = np.asarray(qmc_dn) - np.asarray(reference_dn)
    ref_norm_up = float(np.linalg.norm(reference_up))
    ref_norm_dn = float(np.linalg.norm(reference_dn))
    trace_ref_up = float(np.trace(reference_up).real)
    trace_ref_dn = float(np.trace(reference_dn).real)
    trace_qmc_up = float(np.trace(qmc_up).real)
    trace_qmc_dn = float(np.trace(qmc_dn).real)

    return {
        "frobenius_up": float(np.linalg.norm(diff_up)),
        "frobenius_dn": float(np.linalg.norm(diff_dn)),
        "relative_frobenius_up": float(np.linalg.norm(diff_up) / ref_norm_up) if ref_norm_up > 0.0 else float(np.linalg.norm(diff_up)),
        "relative_frobenius_dn": float(np.linalg.norm(diff_dn) / ref_norm_dn) if ref_norm_dn > 0.0 else float(np.linalg.norm(diff_dn)),
        "max_abs_up": float(np.max(np.abs(diff_up))),
        "max_abs_dn": float(np.max(np.abs(diff_dn))),
        "trace_ref_up": trace_ref_up,
        "trace_ref_dn": trace_ref_dn,
        "trace_qmc_up": trace_qmc_up,
        "trace_qmc_dn": trace_qmc_dn,
        "trace_diff_up": trace_qmc_up - trace_ref_up,
        "trace_diff_dn": trace_qmc_dn - trace_ref_dn,
    }


def _validate_green_shapes(reference_up: np.ndarray, reference_dn: np.ndarray, qmc_up: np.ndarray, qmc_dn: np.ndarray) -> str | None:
    ref_up_shape = tuple(np.shape(reference_up))
    ref_dn_shape = tuple(np.shape(reference_dn))
    qmc_up_shape = tuple(np.shape(qmc_up))
    qmc_dn_shape = tuple(np.shape(qmc_dn))
    if ref_up_shape != qmc_up_shape or ref_dn_shape != qmc_dn_shape:
        return (
            "Green matrix shape mismatch: "
            f"qmc_up={qmc_up_shape}, ref_up={ref_up_shape}, "
            f"qmc_dn={qmc_dn_shape}, ref_dn={ref_dn_shape}"
        )
    return None


def _search_regex(text: str, patterns: Sequence[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return float(match.group(1))
    return None


def read_cpqmc_energy(case_dir: str | Path) -> dict[str, float | None]:
    case_path = Path(case_dir)
    result = {
        "mixed_energy": None,
        "mixed_energy_per_site": None,
        "total_energy": None,
        "total_energy_per_site": None,
        "initial_energy": None,
        "initial_energy_per_site": None,
        "comparison_energy": None,
        "comparison_energy_per_site": None,
        "comparison_energy_source": None,
    }

    corr_path = case_path / "corr.dat"
    if corr_path.exists():
        text = corr_path.read_text(encoding="utf-8", errors="ignore")
        result["mixed_energy"] = _search_regex(text, [r"Mixed Energy=\s*([-\d.Ee+]+)"])
        result["mixed_energy_per_site"] = _search_regex(text, [r"total energy per N:\s*([-\d.Ee+]+)"])
        result["total_energy"] = _search_regex(text, [r"total energy:\s*([-\d.Ee+]+)"])
        result["total_energy_per_site"] = _search_regex(text, [r"total energy per N:\s*([-\d.Ee+]+)"])

    for log_name in ("nohup.out", "job1.nohup.out", "job.nohup.out"):
        log_path = case_path / log_name
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        if result["initial_energy"] is None:
            result["initial_energy"] = _search_regex(text, [r"Initial Energy=\s*([-\d.Ee+]+)"])
        if result["initial_energy_per_site"] is None:
            result["initial_energy_per_site"] = _search_regex(text, [r"Initial Energy Per N=\s*([-\d.Ee+]+)"])
        if result["mixed_energy"] is None:
            result["mixed_energy"] = _search_regex(text, [r"Energy=\s*([-\d.Ee+]+)", r"Mixed Energy=\s*([-\d.Ee+]+)"])
        if result["total_energy"] is None:
            result["total_energy"] = _search_regex(text, [r"energy is the:\s*([-\d.Ee+]+)", r"total energy:\s*([-\d.Ee+]+)"])
        if result["total_energy_per_site"] is None:
            result["total_energy_per_site"] = _search_regex(text, [r"energy Per N is the:\s*([-\d.Ee+]+)", r"total energy per N:\s*([-\d.Ee+]+)"])

    out_path = case_path / "out.dat"
    if out_path.exists():
        text = out_path.read_text(encoding="utf-8", errors="ignore")
        if result["mixed_energy"] is None:
            result["mixed_energy"] = _search_regex(text, [r"Mixed Energy=\s*([-\d.Ee+]+)"])
        if result["total_energy"] is None:
            result["total_energy"] = _search_regex(text, [r"total energy:\s*([-\d.Ee+]+)"])
        if result["total_energy_per_site"] is None:
            result["total_energy_per_site"] = _search_regex(text, [r"total energy per N:\s*([-\d.Ee+]+)"])

    for energy_key, per_site_key, label in (
        ("total_energy", "total_energy_per_site", "total_energy"),
        ("mixed_energy", "mixed_energy_per_site", "mixed_energy"),
        ("initial_energy", "initial_energy_per_site", "initial_energy"),
    ):
        if result[energy_key] is not None:
            result["comparison_energy"] = result[energy_key]
            result["comparison_energy_per_site"] = result[per_site_key]
            result["comparison_energy_source"] = label
            break
    return result


def compare_case_to_ed(
    case_dir: str | Path,
    *,
    bc_y: str = "PBC",
    max_basis_states: int = 250_000,
) -> dict[str, object]:
    model = load_case_model(case_dir)
    qmc = read_cpqmc_energy(case_dir)
    ed = run_exact_diagonalization(model, bc_y=bc_y, max_basis_states=max_basis_states)
    one_body = run_noninteracting_reference(model, bc_y=bc_y)
    qmc_green = read_qmc_green_matrices(case_dir)
    qmc_science_pph = read_qmc_science_pph_observables(case_dir)
    ed_green = run_exact_green_reference(model, ed)
    qmc_science_pph_energy = read_qmc_science_pph_energy_breakdown(case_dir)
    qmc_science_pph_energy = _normalize_science_pph_debug_breakdown(model, qmc_science_pph_energy)
    qmc_science_pph_projector = read_qmc_science_pph_projector_debug(case_dir)
    ed_science_pph_energy = run_exact_science_pph_breakdown(model, ed, ed_green)

    result = {
        "case": model.folder_name,
        "model": asdict(model),
        "qmc": qmc,
        "ed": ed,
        "one_body": one_body,
        "qmc_green": qmc_green,
        "qmc_science_pph": qmc_science_pph,
        "ed_green": ed_green,
        "benchmark_notes": [],
        "green_comparison_error": None,
        "science_pph_reference_source": None,
        "science_pph_reference": None,
        "science_pph_qmc_source": None,
        "science_pph_qmc": None,
        "science_pph_comparison": None,
        "science_pph_energy_reference": ed_science_pph_energy,
        "science_pph_energy_qmc": qmc_science_pph_energy,
        "science_pph_energy_comparison": None,
        "science_pph_projector_qmc": qmc_science_pph_projector,
        "science_pph_sector_reference": None,
        "science_pph_sector_qmc": None,
        "science_pph_sector_qmc_from_debug": None,
        "science_pph_sector_comparison": None,
        "science_pph_sector_debug_comparison": None,
        "science_pph_sector_internal_vs_physical_delta": None,
    }

    science_pph_controls = model.science_pph_controls or {}
    if science_pph_controls:
        result["benchmark_notes"].append(
            f"Science PPH controls loaded from {model.science_pph_active_source}."
        )
        if model.benchmark_mode == "science_pph_spinor":
            result["benchmark_notes"].append(
                "This case is benchmarked in the transformed Science PPH spinor basis with fixed total transformed particle number."
            )
        elif model.benchmark_mode == "unsupported_pairing_state":
            result["benchmark_notes"].append(
                "This pairing mode is still unsupported by the exact benchmark."
            )
    if model.pair_frame_mode == PAIR_FRAME_PPH_SCIENCE and model.benchmark_mode == "qmc_spinful_baseline":
        result["benchmark_notes"].append(
            "This baseline uses the Science PPH frame without an active pairing source, so the QMC onsite interaction is the transformed attractive U = -|U|."
        )

    if ed["energy"] is not None and qmc["comparison_energy"] is not None:
        result["ed_minus_qmc"] = float(ed["energy"] - qmc["comparison_energy"])
    else:
        result["ed_minus_qmc"] = None

    if qmc["comparison_energy"] is not None:
        result["one_body_minus_qmc"] = None if one_body["energy"] is None else float(one_body["energy"] - qmc["comparison_energy"])
    else:
        result["one_body_minus_qmc"] = None

    if qmc_green["available"] and one_body.get("green_up") is not None and one_body.get("green_dn") is not None:
        if ed_green.get("available", False):
            result["green_reference_source"] = "quspin_ed_green"
            if ed_green.get("science_pph_anomalous") is not None:
                result["science_pph_reference_source"] = "quspin_ed_green"
                result["science_pph_reference"] = ed_green["science_pph_anomalous"]
            if model.benchmark_mode == "science_pph_spinor" and ed_green.get("spinor_green") is not None:
                result["science_pph_sector_reference"] = _infer_science_pph_sector_from_spinor_green(
                    model,
                    np.asarray(ed_green["spinor_green"]),
                )
            shape_error = _validate_green_shapes(
                ed_green["green_up"],
                ed_green["green_dn"],
                qmc_green["green_up"],
                qmc_green["green_dn"],
            )
            if shape_error is None:
                result["green_comparison"] = compare_green_matrices(
                    ed_green["green_up"],
                    ed_green["green_dn"],
                    qmc_green["green_up"],
                    qmc_green["green_dn"],
                )
            else:
                result["green_comparison"] = None
                result["green_comparison_error"] = shape_error
        else:
            result["green_reference_source"] = "one_body_reference_green"
            if one_body.get("science_pph_anomalous") is not None:
                result["science_pph_reference_source"] = "one_body_reference_green"
                result["science_pph_reference"] = one_body["science_pph_anomalous"]
            if model.benchmark_mode == "science_pph_spinor" and one_body.get("spinor_green") is not None:
                result["science_pph_sector_reference"] = _infer_science_pph_sector_from_spinor_green(
                    model,
                    np.asarray(one_body["spinor_green"]),
                )
            shape_error = _validate_green_shapes(
                one_body["green_up"],
                one_body["green_dn"],
                qmc_green["green_up"],
                qmc_green["green_dn"],
            )
            if shape_error is None:
                result["green_comparison"] = compare_green_matrices(
                    one_body["green_up"],
                    one_body["green_dn"],
                    qmc_green["green_up"],
                    qmc_green["green_dn"],
                )
            else:
                result["green_comparison"] = None
                result["green_comparison_error"] = shape_error
    else:
        result["green_reference_source"] = None
        result["green_comparison"] = None

    if model.benchmark_mode == "science_pph_spinor" and qmc_green["available"]:
        result["science_pph_sector_qmc"] = _infer_science_pph_sector_from_physical_greens(
            model,
            np.asarray(qmc_green["green_up"]),
            np.asarray(qmc_green["green_dn"]),
        )
        if result["science_pph_sector_reference"] is not None:
            result["science_pph_sector_comparison"] = compare_science_pph_energy_breakdown(
                result["science_pph_sector_reference"],
                result["science_pph_sector_qmc"],
            )
        if result["science_pph_sector_qmc"] is not None and model.transformed_total_particles is not None:
            total_mismatch = abs(
                result["science_pph_sector_qmc"]["transformed_total_re"] - float(model.transformed_total_particles)
            )
            if total_mismatch > 1.0e-3:
                result["benchmark_notes"].append(
                    "QMC physical Green implies a transformed total particle count that drifts away from the target sector."
                )

    if model.benchmark_mode == "science_pph_spinor" and qmc_science_pph_energy.get("available"):
        result["science_pph_sector_qmc_from_debug"] = {
            "source": qmc_science_pph_energy.get("source"),
            "tag": qmc_science_pph_energy.get("tag"),
            "sample_id": qmc_science_pph_energy.get("sample_id"),
            "n_up_re": qmc_science_pph_energy.get("n_up_re"),
            "n_dn_re": qmc_science_pph_energy.get("n_dn_re"),
            "tr_g22_re": qmc_science_pph_energy.get("tr_g22_re"),
            "transformed_total_re": qmc_science_pph_energy.get("transformed_total_re"),
        }
        if result["science_pph_sector_reference"] is not None:
            result["science_pph_sector_debug_comparison"] = compare_science_pph_energy_breakdown(
                result["science_pph_sector_reference"],
                result["science_pph_sector_qmc_from_debug"],
            )
        if result["science_pph_sector_qmc"] is not None:
            result["science_pph_sector_internal_vs_physical_delta"] = compare_science_pph_energy_breakdown(
                result["science_pph_sector_qmc_from_debug"],
                result["science_pph_sector_qmc"],
            )
            internal_total = result["science_pph_sector_qmc_from_debug"].get("transformed_total_re")
            physical_total = result["science_pph_sector_qmc"].get("transformed_total_re")
            if internal_total is not None and physical_total is not None:
                if abs(float(internal_total) - float(physical_total)) > 1.0e-3:
                    result["benchmark_notes"].append(
                        "QMC physical Green sector and QMC internal spinor-debug sector disagree, which points to an output/extraction inconsistency."
                    )

    if qmc_science_pph.get("back_propagated") is not None:
        result["science_pph_qmc_source"] = "science_pph_bp_observables.dat"
        result["science_pph_qmc"] = qmc_science_pph["back_propagated"]
    elif qmc_science_pph.get("mixed") is not None:
        result["science_pph_qmc_source"] = "science_pph_observables.dat"
        result["science_pph_qmc"] = qmc_science_pph["mixed"]

    if result["science_pph_reference"] is not None and result["science_pph_qmc"] is not None:
        result["science_pph_comparison"] = compare_science_pph_observables(
            result["science_pph_reference"],
            result["science_pph_qmc"],
        )

    if ed_science_pph_energy.get("available") and qmc_science_pph_energy.get("available"):
        result["science_pph_energy_comparison"] = compare_science_pph_energy_breakdown(
            ed_science_pph_energy,
            qmc_science_pph_energy,
        )

    return result


def compare_cases(
    case_dirs: Iterable[str | Path],
    *,
    bc_y: str = "PBC",
    max_basis_states: int = 250_000,
) -> list[dict[str, object]]:
    return [
        compare_case_to_ed(case_dir, bc_y=bc_y, max_basis_states=max_basis_states)
        for case_dir in case_dirs
    ]


def suggest_reduced_ed_benchmarks(
    case_dir: str | Path,
    *,
    small_lattices: Sequence[tuple[int, int]] = ((4, 2), (4, 4)),
) -> list[dict[str, object]]:
    model = load_case_model(case_dir)
    if model.target_doping is None and model.actual_doping is None:
        raise ValueError("The case folder name does not encode a doping token.")

    target_doping = model.target_doping if model.target_doping is not None else model.actual_doping
    assert target_doping is not None

    suggestions: list[dict[str, object]] = []
    for lx, ly in small_lattices:
        sector = resolve_hubbard_sector(lx=lx, ly=ly, target_doping=target_doping, t=1.0, t_prime=abs(model.t1))
        suggestions.append(
            {
                "benchmark_lx": lx,
                "benchmark_ly": ly,
                "benchmark_nup": sector["nup"],
                "benchmark_ndn": sector["ndn"],
                "benchmark_total_electrons": sector["total_electrons"],
                "benchmark_actual_doping": sector["actual_doping"],
                "benchmark_density": sector["actual_density"],
                "target_doping": target_doping,
                "u": model.u,
                "t0": model.t0,
                "t1": model.t1,
                "h_pin": model.h_pin,
                "pin_right_factor": model.pin_right_factor,
            }
        )
    return suggestions


def summarize_for_terminal(result: dict[str, object]) -> str:
    model = result["model"]
    qmc = result["qmc"]
    ed = result["ed"]
    one_body = result["one_body"]

    lines = [
        f"Case: {result['case']}",
        f"  Lx x Ly = {model['lx']} x {model['ly']}",
        f"  Nup, Ndn = {model['nup']}, {model['ndn']}",
        f"  total electrons = {model['total_electrons']}",
        f"  transformed particles (up, dn~, total) = {model.get('transformed_n_up_target')} , {model.get('transformed_n_dn_target')} , {model.get('transformed_total_particles')}",
        f"  target / actual doping = {model['target_doping']} / {model['actual_doping']}",
        f"  U = {model['u']}, V = {model.get('vpd')}, t0 = {model['t0']}, t1 = {model['t1']}, h_pin = {model['h_pin']}",
        f"  effective onsite U used by QMC baseline = {model['effective_onsite_u']}",
        f"  geom/twist controls = geom_mode {model['geom_mode']}, tabc_mode {model['tabc_mode']}, theta_x {model['theta_x_val']}, theta_y {model['theta_y_val']}",
        f"  pair controls = pair_source_mode {model['pair_source_mode']}, pair_frame_mode {model['pair_frame_mode']}, benchmark_mode {model['benchmark_mode']}",
        f"  Hilbert size = {ed['hilbert_size']}",
        f"  QMC mixed energy = {qmc['mixed_energy']}",
        f"  QMC total energy = {qmc['total_energy']}",
        f"  QMC initial energy = {qmc['initial_energy']}",
        f"  QMC comparison energy ({qmc['comparison_energy_source']}) = {qmc['comparison_energy']}",
        f"  one-body reference energy = {one_body['energy']}",
    ]

    science_pph_controls = model.get("science_pph_controls") or {}
    if science_pph_controls:
        lines.append(
            "  Science PPH controls "
            f"({model.get('science_pph_active_source')}): "
            f"pair_source_mode={science_pph_controls.get('pair_source_mode')}, "
            f"pair_frame_mode={science_pph_controls.get('pair_frame_mode')}, "
            f"pph_selfcons_mode={science_pph_controls.get('pph_selfcons_mode')}, "
            f"h_d_pair={science_pph_controls.get('h_d_pair')}, "
            f"alpha_pph_trial={science_pph_controls.get('alpha_pph_trial')}, "
            f"mu_pph_trial={science_pph_controls.get('mu_pph_trial')}, "
            f"target_mode={science_pph_controls.get('science_pph_selfcons_target_mode')}"
        )

    if ed["energy"] is not None:
        lines.append(f"  exact ED energy = {ed['energy']}")
        lines.append(f"  ED - QMC = {result['ed_minus_qmc']}")
    else:
        lines.append(f"  exact ED skipped: {ed['error']}")

    if result.get("science_pph_reference") is not None:
        ref = result["science_pph_reference"]
        lines.append(
            "  Science PPH reference "
            f"({result.get('science_pph_reference_source')}): "
            f"s=({ref['s_wave_re']},{ref['s_wave_im']}), "
            f"d=({ref['d_wave_re']},{ref['d_wave_im']}), norm={ref['norm']}"
        )
    if result.get("science_pph_qmc") is not None:
        measured = result["science_pph_qmc"]
        lines.append(
            "  Science PPH QMC "
            f"({result.get('science_pph_qmc_source')}): "
            f"s=({measured['s_wave_re']},{measured['s_wave_im']}), "
            f"d=({measured['d_wave_re']},{measured['d_wave_im']}), norm={measured['norm']}"
        )
    if result.get("science_pph_comparison") is not None:
        delta = result["science_pph_comparison"]
        lines.append(
            "  Science PPH delta: "
            f"ds=({delta['delta_s_wave_re']},{delta['delta_s_wave_im']}), "
            f"dd=({delta['delta_d_wave_re']},{delta['delta_d_wave_im']}), "
            f"dnorm={delta['delta_norm']}"
        )

    if result.get("science_pph_sector_reference") is not None:
        sector_ref = result["science_pph_sector_reference"]
        lines.append(
            "  Science PPH sector reference: "
            f"n_up={sector_ref['n_up_re']}, n_dn={sector_ref['n_dn_re']}, "
            f"tr_g22={sector_ref['tr_g22_re']}, total~={sector_ref['transformed_total_re']}"
        )
    if result.get("science_pph_sector_qmc") is not None:
        sector_qmc = result["science_pph_sector_qmc"]
        lines.append(
            "  Science PPH sector from QMC physical Greens: "
            f"n_up={sector_qmc['n_up_re']}, n_dn={sector_qmc['n_dn_re']}, "
            f"tr_g22={sector_qmc['tr_g22_re']}, total~={sector_qmc['transformed_total_re']}"
        )
    if result.get("science_pph_sector_qmc_from_debug") is not None:
        sector_dbg = result["science_pph_sector_qmc_from_debug"]
        lines.append(
            "  Science PPH sector from QMC spinor debug: "
            f"n_up={sector_dbg['n_up_re']}, n_dn={sector_dbg['n_dn_re']}, "
            f"tr_g22={sector_dbg['tr_g22_re']}, total~={sector_dbg['transformed_total_re']}"
        )
    if result.get("science_pph_sector_debug_comparison") is not None:
        sector_dbg_delta = result["science_pph_sector_debug_comparison"]
        lines.append(
            "  Science PPH sector delta (debug - exact): "
            f"dn_up={sector_dbg_delta.get('delta_n_up_re')}, "
            f"dtr_g22={sector_dbg_delta.get('delta_tr_g22_re')}, "
            f"dtotal={sector_dbg_delta.get('delta_transformed_total_re')}"
        )
    if result.get("science_pph_sector_internal_vs_physical_delta") is not None:
        sector_delta = result["science_pph_sector_internal_vs_physical_delta"]
        lines.append(
            "  Science PPH sector delta (physical - debug): "
            f"dn_up={sector_delta.get('delta_n_up_re')}, "
            f"dtr_g22={sector_delta.get('delta_tr_g22_re')}, "
            f"dtotal={sector_delta.get('delta_transformed_total_re')}"
        )
    if result.get("science_pph_energy_reference", {}).get("available"):
        e_ref = result["science_pph_energy_reference"]
        lines.append(
            "  Science PPH exact breakdown: "
            f"e1_up={e_ref['e_one_up_re']}, e1_dn={e_ref['e_one_dn_re']}, "
            f"e_u={e_ref['e_u_re']}, e_pair={e_ref['e_pair_re']}, e_total={e_ref['e_total_re']}"
        )
    if result.get("science_pph_energy_qmc", {}).get("available"):
        e_qmc = result["science_pph_energy_qmc"]
        lines.append(
            "  Science PPH QMC breakdown "
            f"({e_qmc['tag']}): "
            f"e1_up={e_qmc['e_one_up_re']}, e1_dn={e_qmc['e_one_dn_re']}, "
            f"e_u={e_qmc['e_u_re']}, e_pair={e_qmc['e_pair_re']}, e_total={e_qmc['e_total_re']}"
        )
    if result.get("science_pph_energy_comparison") is not None:
        e_delta = result["science_pph_energy_comparison"]
        lines.append(
            "  Science PPH breakdown delta: "
            f"de_total={e_delta.get('delta_e_total_re')}, "
            f"de_pair={e_delta.get('delta_e_pair_re')}, "
            f"dn_up={e_delta.get('delta_n_up_re')}, "
            f"dtr_g22={e_delta.get('delta_tr_g22_re')}"
        )
    if result.get("science_pph_projector_qmc", {}).get("available"):
        proj = result["science_pph_projector_qmc"]
        lines.append(
            "  Science PPH projector debug "
            f"({proj['tag']}): "
            f"trace={proj['trace_re']}, idem={proj['idem_resid_re']}, herm={proj['herm_resid_re']}"
        )

    if result["green_comparison"] is not None:
        green = result["green_comparison"]
        lines.append(
            "  Green comparison "
            f"({result['green_reference_source']}): "
            f"relF_up={green['relative_frobenius_up']}, relF_dn={green['relative_frobenius_dn']}, "
            f"max_up={green['max_abs_up']}, max_dn={green['max_abs_dn']}, "
            f"trace_qmc=({green['trace_qmc_up']},{green['trace_qmc_dn']}), "
            f"trace_ref=({green['trace_ref_up']},{green['trace_ref_dn']})"
        )
    elif result.get("green_comparison_error"):
        lines.append(f"  Green comparison skipped: {result['green_comparison_error']}")
    else:
        qmc_green = result["qmc_green"]
        lines.append(f"  Green comparison skipped: {qmc_green.get('error', 'QMC Green matrices unavailable.')}")

    for note in result.get("benchmark_notes", []):
        lines.append(f"  Note: {note}")

    return "\n".join(lines)


def compact_result_summary(result: dict[str, object]) -> dict[str, object]:
    model = result["model"]
    qmc = result["qmc"]
    ed = result["ed"]
    green = result["green_comparison"]
    science_pph_controls = model.get("science_pph_controls") or {}
    return {
        "case": result["case"],
        "case_dir": model["case_dir"],
        "lx": model["lx"],
        "ly": model["ly"],
        "nup": model["nup"],
        "ndn": model["ndn"],
        "target_doping": model["target_doping"],
        "actual_doping": model["actual_doping"],
        "u": model["u"],
        "vpd": model.get("vpd"),
        "t0": model["t0"],
        "t1": model["t1"],
        "h_pin": model["h_pin"],
        "effective_onsite_u": model.get("effective_onsite_u"),
        "transformed_n_up_target": model.get("transformed_n_up_target"),
        "transformed_n_dn_target": model.get("transformed_n_dn_target"),
        "transformed_total_particles": model.get("transformed_total_particles"),
        "geom_mode": model.get("geom_mode"),
        "theta_x_val": model.get("theta_x_val"),
        "theta_y_val": model.get("theta_y_val"),
        "tabc_mode": model.get("tabc_mode"),
        "pair_source_mode": model.get("pair_source_mode"),
        "pair_frame_mode": model.get("pair_frame_mode"),
        "benchmark_mode": model.get("benchmark_mode"),
        "science_pph_active_source": model.get("science_pph_active_source"),
        "science_pph_input_controls_path": model.get("science_pph_input_controls_path"),
        "science_pph_effective_controls_path": model.get("science_pph_effective_controls_path"),
        "science_pph_input_controls_present": model.get("science_pph_input_controls_path") is not None,
        "science_pph_effective_controls_present": model.get("science_pph_effective_controls_path") is not None,
        "science_pph_pair_source_mode": science_pph_controls.get("pair_source_mode"),
        "science_pph_pair_frame_mode": science_pph_controls.get("pair_frame_mode"),
        "science_pph_complex_weight_mode": science_pph_controls.get("complex_weight_mode"),
        "science_pph_nambu_stage_mode": science_pph_controls.get("nambu_stage_mode"),
        "science_pph_pph_gauge_mode": science_pph_controls.get("pph_gauge_mode"),
        "science_pph_pph_selfcons_mode": science_pph_controls.get("pph_selfcons_mode"),
        "science_pph_lambda_d_val": science_pph_controls.get("lambda_d_val"),
        "science_pph_h_d_pair": science_pph_controls.get("h_d_pair"),
        "science_pph_alpha_pph_trial": science_pph_controls.get("alpha_pph_trial"),
        "science_pph_mu_pph_trial": science_pph_controls.get("mu_pph_trial"),
        "science_pph_selfcons_autoload": science_pph_controls.get("science_pph_selfcons_autoload"),
        "science_pph_selfcons_target_mode": science_pph_controls.get("science_pph_selfcons_target_mode"),
        "science_pph_mu_nambu": science_pph_controls.get("mu_nambu"),
        "hilbert_size": ed["hilbert_size"],
        "qmc_energy_source": qmc["comparison_energy_source"],
        "qmc_energy": qmc["comparison_energy"],
        "qmc_mixed_energy": qmc["mixed_energy"],
        "qmc_total_energy": qmc["total_energy"],
        "qmc_initial_energy": qmc["initial_energy"],
        "ed_energy": ed["energy"],
        "ed_error": ed["error"],
        "one_body_energy": result["one_body"]["energy"],
        "ed_minus_qmc": result["ed_minus_qmc"],
        "one_body_minus_qmc": result["one_body_minus_qmc"],
        "benchmark_notes": result.get("benchmark_notes", []),
        "science_pph_reference_source": result.get("science_pph_reference_source"),
        "science_pph_reference": result.get("science_pph_reference"),
        "science_pph_qmc_source": result.get("science_pph_qmc_source"),
        "science_pph_qmc": result.get("science_pph_qmc"),
        "science_pph_comparison": result.get("science_pph_comparison"),
        "science_pph_sector_reference": result.get("science_pph_sector_reference"),
        "science_pph_sector_qmc": result.get("science_pph_sector_qmc"),
        "science_pph_sector_qmc_from_debug": result.get("science_pph_sector_qmc_from_debug"),
        "science_pph_sector_comparison": result.get("science_pph_sector_comparison"),
        "science_pph_sector_debug_comparison": result.get("science_pph_sector_debug_comparison"),
        "science_pph_sector_internal_vs_physical_delta": result.get("science_pph_sector_internal_vs_physical_delta"),
        "science_pph_energy_reference": result.get("science_pph_energy_reference"),
        "science_pph_energy_qmc": result.get("science_pph_energy_qmc"),
        "science_pph_energy_comparison": result.get("science_pph_energy_comparison"),
        "science_pph_projector_qmc": result.get("science_pph_projector_qmc"),
        "green_reference_source": result["green_reference_source"],
        "green_available": result["qmc_green"]["available"],
        "green_error": result["qmc_green"].get("error"),
        "green_comparison_error": result.get("green_comparison_error"),
        "green_relative_frobenius_up": None if green is None else green["relative_frobenius_up"],
        "green_relative_frobenius_dn": None if green is None else green["relative_frobenius_dn"],
        "green_max_abs_up": None if green is None else green["max_abs_up"],
        "green_max_abs_dn": None if green is None else green["max_abs_dn"],
        "green_trace_qmc_up": None if green is None else green["trace_qmc_up"],
        "green_trace_qmc_dn": None if green is None else green["trace_qmc_dn"],
        "green_trace_ref_up": None if green is None else green["trace_ref_up"],
        "green_trace_ref_dn": None if green is None else green["trace_ref_dn"],
    }


def write_validation_report(results: Sequence[dict[str, object]], output_path: str | Path) -> Path:
    output = Path(output_path)
    payload = [compact_result_summary(result) for result in results]
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def build_validation_table_rows(results: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        summary = compact_result_summary(result)
        science_pph_ref = summary.get("science_pph_reference") or {}
        science_pph_qmc = summary.get("science_pph_qmc") or {}
        science_pph_energy_ref = summary.get("science_pph_energy_reference") or {}
        science_pph_energy_qmc = summary.get("science_pph_energy_qmc") or {}
        science_pph_sector_ref = summary.get("science_pph_sector_reference") or {}
        science_pph_sector_qmc = summary.get("science_pph_sector_qmc") or {}
        science_pph_sector_debug = summary.get("science_pph_sector_qmc_from_debug") or {}
        science_pph_energy_cmp = summary.get("science_pph_energy_comparison") or {}
        science_pph_sector_cmp = summary.get("science_pph_sector_comparison") or {}

        rows.append(
            {
                "case": summary["case"],
                "case_dir": summary["case_dir"],
                "lx": summary["lx"],
                "ly": summary["ly"],
                "nup": summary["nup"],
                "ndn": summary["ndn"],
                "target_doping": summary["target_doping"],
                "actual_doping": summary["actual_doping"],
                "u": summary["u"],
                "vpd": summary["vpd"],
                "t0": summary["t0"],
                "t1": summary["t1"],
                "h_pin": summary["h_pin"],
                "benchmark_mode": summary["benchmark_mode"],
                "pair_source_mode": summary["pair_source_mode"],
                "pair_frame_mode": summary["pair_frame_mode"],
                "science_pph_active_source": summary["science_pph_active_source"],
                "science_pph_h_d_pair": summary["science_pph_h_d_pair"],
                "science_pph_alpha_pph_trial": summary["science_pph_alpha_pph_trial"],
                "science_pph_mu_pph_trial": summary["science_pph_mu_pph_trial"],
                "qmc_energy_source": summary["qmc_energy_source"],
                "qmc_energy": summary["qmc_energy"],
                "ed_energy": summary["ed_energy"],
                "ed_minus_qmc": summary["ed_minus_qmc"],
                "qmc_mixed_energy": summary["qmc_mixed_energy"],
                "qmc_total_energy": summary["qmc_total_energy"],
                "qmc_initial_energy": summary["qmc_initial_energy"],
                "one_body_energy": summary["one_body_energy"],
                "green_reference_source": summary["green_reference_source"],
                "green_trace_up_qmc": summary["green_trace_qmc_up"],
                "green_trace_up_ed": summary["green_trace_ref_up"],
                "green_trace_dn_qmc": summary["green_trace_qmc_dn"],
                "green_trace_dn_ed": summary["green_trace_ref_dn"],
                "green_relative_frobenius_up": summary["green_relative_frobenius_up"],
                "green_relative_frobenius_dn": summary["green_relative_frobenius_dn"],
                "green_max_abs_up": summary["green_max_abs_up"],
                "green_max_abs_dn": summary["green_max_abs_dn"],
                "science_pph_s_wave_re_qmc": science_pph_qmc.get("s_wave_re"),
                "science_pph_s_wave_re_ed": science_pph_ref.get("s_wave_re"),
                "science_pph_s_wave_im_qmc": science_pph_qmc.get("s_wave_im"),
                "science_pph_s_wave_im_ed": science_pph_ref.get("s_wave_im"),
                "science_pph_d_wave_re_qmc": science_pph_qmc.get("d_wave_re"),
                "science_pph_d_wave_re_ed": science_pph_ref.get("d_wave_re"),
                "science_pph_d_wave_im_qmc": science_pph_qmc.get("d_wave_im"),
                "science_pph_d_wave_im_ed": science_pph_ref.get("d_wave_im"),
                "science_pph_norm_qmc": science_pph_qmc.get("norm"),
                "science_pph_norm_ed": science_pph_ref.get("norm"),
                "science_pph_energy_total_qmc": science_pph_energy_qmc.get("e_total_re"),
                "science_pph_energy_total_ed": science_pph_energy_ref.get("e_total_re"),
                "science_pph_energy_total_delta": science_pph_energy_cmp.get("delta_e_total_re"),
                "science_pph_n_up_qmc": science_pph_energy_qmc.get("n_up_re"),
                "science_pph_n_up_ed": science_pph_energy_ref.get("n_up_re"),
                "science_pph_n_dn_qmc": science_pph_energy_qmc.get("n_dn_re"),
                "science_pph_n_dn_ed": science_pph_energy_ref.get("n_dn_re"),
                "science_pph_transformed_total_qmc": science_pph_energy_qmc.get("transformed_total_re"),
                "science_pph_transformed_total_ed": science_pph_energy_ref.get("transformed_total_re"),
                "science_pph_sector_total_from_green_qmc": science_pph_sector_qmc.get("transformed_total_re"),
                "science_pph_sector_total_from_green_ed": science_pph_sector_ref.get("transformed_total_re"),
                "science_pph_sector_total_from_debug_qmc": science_pph_sector_debug.get("transformed_total_re"),
                "science_pph_sector_total_delta_green_vs_ed": science_pph_sector_cmp.get("delta_transformed_total_re"),
                "ed_error": summary["ed_error"],
                "green_error": summary["green_error"],
                "green_comparison_error": summary["green_comparison_error"],
                "benchmark_notes": " | ".join(summary.get("benchmark_notes", [])),
            }
        )
    return rows


def write_validation_table_csv(results: Sequence[dict[str, object]], output_path: str | Path) -> Path:
    output = Path(output_path)
    rows = build_validation_table_rows(results)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _format_markdown_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.10g}"
    return str(value).replace("\n", "<br>")


def write_validation_table_markdown(results: Sequence[dict[str, object]], output_path: str | Path) -> Path:
    output = Path(output_path)
    rows = build_validation_table_rows(results)
    columns = [
        ("Case", "case"),
        ("Lattice", None),
        ("Target dop", "target_doping"),
        ("Actual dop", "actual_doping"),
        ("Mode", "benchmark_mode"),
        ("QMC energy", "qmc_energy"),
        ("ED energy", "ed_energy"),
        ("ED-QMC", "ed_minus_qmc"),
        ("QMC tr Gup", "green_trace_up_qmc"),
        ("ED tr Gup", "green_trace_up_ed"),
        ("QMC tr Gdn", "green_trace_dn_qmc"),
        ("ED tr Gdn", "green_trace_dn_ed"),
        ("relF up", "green_relative_frobenius_up"),
        ("relF dn", "green_relative_frobenius_dn"),
        ("SPPH E qmc", "science_pph_energy_total_qmc"),
        ("SPPH E ed", "science_pph_energy_total_ed"),
        ("SPPH N~ qmc", "science_pph_transformed_total_qmc"),
        ("SPPH N~ ed", "science_pph_transformed_total_ed"),
    ]

    header = "| " + " | ".join(title for title, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body: list[str] = []
    for row in rows:
        lattice = f"{row['lx']}x{row['ly']}"
        values = []
        for _, key in columns:
            if key is None:
                values.append(lattice)
            else:
                values.append(_format_markdown_value(row.get(key)))
        body.append("| " + " | ".join(values) + " |")

    output.write_text("\n".join([header, divider, *body]) + "\n", encoding="utf-8")
    return output


__all__ = [
    "CaseModel",
    "build_pair_source_dwave_matrix",
    "build_qmc_one_body_matrices",
    "build_science_pph_spinor_one_body_matrix",
    "HAVE_QUSPIN",
    "build_one_body_matrix",
    "build_quspin_hamiltonian",
    "compare_case_to_ed",
    "compare_cases",
    "compare_green_matrices",
    "compare_science_pph_energy_breakdown",
    "compare_science_pph_observables",
    "compact_result_summary",
    "discover_case_dirs",
    "hilbert_size",
    "load_case_model",
    "read_qmc_green_matrices",
    "read_qmc_science_pph_energy_breakdown",
    "read_qmc_science_pph_observables",
    "read_qmc_science_pph_projector_debug",
    "read_cpqmc_energy",
    "run_exact_diagonalization",
    "run_exact_science_pph_breakdown",
    "run_exact_green_reference",
    "run_noninteracting_reference",
    "suggest_reduced_ed_benchmarks",
    "summarize_for_terminal",
    "write_validation_report",
    "write_validation_table_csv",
    "write_validation_table_markdown",
]
