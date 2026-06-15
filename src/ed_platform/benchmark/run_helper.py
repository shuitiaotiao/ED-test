from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence


def _format_number(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def parse_doping_token(token: str | float | int) -> tuple[float, str]:
    if isinstance(token, (float, int)):
        value = float(token)
        label = _format_number(value).replace(".", "p")
        return value, label

    text = str(token).strip()
    if "/" in text:
        frac = Fraction(text)
        return float(frac), text.replace("/", "_")

    value = float(text)
    return value, text.replace(".", "p")


def hubbard_dispersion(kx: float, ky: float, t: float = 1.0, t_prime: float = 0.0) -> float:
    return -2.0 * t * (math.cos(kx) + math.cos(ky)) - 4.0 * t_prime * math.cos(kx) * math.cos(ky)


def _allowed_momenta(length: int) -> list[float]:
    return [2.0 * math.pi * index / length for index in range(length)]


def noninteracting_energies(lx: int, ly: int, t: float = 1.0, t_prime: float = 0.0) -> list[float]:
    energies: list[float] = []
    for kx in _allowed_momenta(lx):
        for ky in _allowed_momenta(ly):
            energies.append(hubbard_dispersion(kx, ky, t=t, t_prime=t_prime))
    energies.sort()
    return energies


def shell_gap(energies: Sequence[float], nfill_per_spin: int, tolerance: float = 1.0e-10) -> float:
    if nfill_per_spin <= 0 or nfill_per_spin >= len(energies):
        return float("inf")
    gap = energies[nfill_per_spin] - energies[nfill_per_spin - 1]
    if abs(gap) < tolerance:
        return 0.0
    return gap


def resolve_hubbard_sector(
    lx: int,
    ly: int,
    target_doping: float,
    t: float = 1.0,
    t_prime: float = 0.0,
    prefer_closed_shell: bool = False,
) -> dict[str, object]:
    nsites = lx * ly
    energies = noninteracting_energies(lx, ly, t=t, t_prime=t_prime)
    candidates: list[dict[str, object]] = []

    for holes in range(0, nsites + 1, 2):
        total_electrons = nsites - holes
        nup = total_electrons // 2
        ndn = total_electrons // 2
        actual_doping = holes / nsites
        gap = shell_gap(energies, nup)
        closed_shell = gap > 1.0e-10
        candidates.append(
            {
                "holes": holes,
                "total_electrons": total_electrons,
                "nup": nup,
                "ndn": ndn,
                "actual_doping": actual_doping,
                "doping_error": abs(actual_doping - target_doping),
                "shell_gap": gap,
                "closed_shell": closed_shell,
            }
        )

    eligible = [candidate for candidate in candidates if candidate["closed_shell"]]
    search_pool = eligible if prefer_closed_shell and eligible else candidates
    best = min(
        search_pool,
        key=lambda candidate: (
            candidate["doping_error"],
            0 if candidate["closed_shell"] else 1,
            -float(candidate["shell_gap"]),
            candidate["holes"],
        ),
    )

    result = dict(best)
    result.update(
        {
            "lx": lx,
            "ly": ly,
            "nsites": nsites,
            "target_doping": target_doping,
            "target_density": 1.0 - target_doping,
            "actual_density": result["total_electrons"] / nsites,
        }
    )
    return result


__all__ = [
    "hubbard_dispersion",
    "noninteracting_energies",
    "parse_doping_token",
    "resolve_hubbard_sector",
    "shell_gap",
]
