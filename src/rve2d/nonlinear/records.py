"""Load paths, step records and result files shared by the Python and Julia backends.

Nothing here needs PyTorch, so the Julia backend (``nonlinear.backend: julia``) runs
without the ``nonlinear`` extra installed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rve2d.config import NONLINEAR_ACTIVE_COMPONENTS, NonlinearLoadConfig

COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")
COMPONENT_INDEX = {name: index for index, name in enumerate(COMPONENTS)}

_RECORD_COLUMNS = (
    "iterations",
    "max_damage",
    "debonded_fraction",
    "mean_damage",
    "yielded_fraction",
    "mean_eqps_matrix",
    "mean_eqps_fibre",
    "work_density",
)


@dataclass(frozen=True)
class LoadPath:
    """Macro loading as a function of pseudo-time ``t`` in ``[0, end_time]``."""

    prescribed_strain: dict[int, float]  # component -> strain at load factor 1
    fixed_zero: list[int]  # components held at zero strain (inactive or strain-controlled at 0)
    stress_controlled: dict[int, float]  # component -> target macro stress at load factor 1
    unload: bool = False

    @property
    def end_time(self) -> float:
        return 2.0 if self.unload else 1.0

    def factor(self, t: float) -> float:
        return t if t <= 1.0 else 2.0 - t

    @property
    def free(self) -> list[int]:
        return sorted(self.stress_controlled)


@dataclass
class StepRecord:
    time: float
    load_factor: float
    macro_strain: list[float]
    macro_stress: list[float]
    iterations: int
    max_damage: float
    damaged_fraction: float
    mean_damage: float
    yielded_fraction: float
    mean_eqps_matrix: float
    mean_eqps_fibre: float
    work_density: float


@dataclass(frozen=True)
class NonlinearResult:
    summary_path: Path
    response_path: Path
    field_files: list[Path]
    completed: bool
    peak_stress: float
    strain_at_peak: float
    records: list[StepRecord]


def load_path_from_config(kinematics: str, load: NonlinearLoadConfig) -> LoadPath:
    active = [COMPONENT_INDEX[c] for c in NONLINEAR_ACTIVE_COMPONENTS[kinematics]]
    inactive = [i for i in range(6) if i not in active]
    loaded = COMPONENT_INDEX[load.component]
    others = [i for i in active if i != loaded]
    if load.type == "uniaxial_stress":
        stress_free = dict.fromkeys(others, 0.0)
        return LoadPath({loaded: load.max_strain}, inactive, stress_free, load.unload)
    return LoadPath({loaded: load.max_strain}, inactive + others, {}, load.unload)


def write_response_csv(path: Path, records: list[StepRecord]) -> Path:
    header = (
        ["step", "time", "load_factor"]
        + [f"e_{c}" for c in COMPONENTS]
        + [f"s_{c}" for c in COMPONENTS]
        + list(_RECORD_COLUMNS)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, r in enumerate(records):
            writer.writerow(
                [
                    index,
                    r.time,
                    r.load_factor,
                    *r.macro_strain,
                    *r.macro_stress,
                    r.iterations,
                    r.max_damage,
                    r.damaged_fraction,
                    r.mean_damage,
                    r.yielded_fraction,
                    r.mean_eqps_matrix,
                    r.mean_eqps_fibre,
                    r.work_density,
                ]
            )
    return path


def read_response_csv(path: Path) -> list[StepRecord]:
    """Step records from a response CSV written by either backend."""
    records = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                StepRecord(
                    time=float(row["time"]),
                    load_factor=float(row["load_factor"]),
                    macro_strain=[float(row[f"e_{c}"]) for c in COMPONENTS],
                    macro_stress=[float(row[f"s_{c}"]) for c in COMPONENTS],
                    iterations=int(row["iterations"]),
                    max_damage=float(row["max_damage"]),
                    damaged_fraction=float(row["debonded_fraction"]),
                    mean_damage=float(row["mean_damage"]),
                    yielded_fraction=float(row["yielded_fraction"]),
                    mean_eqps_matrix=float(row["mean_eqps_matrix"]),
                    mean_eqps_fibre=float(row["mean_eqps_fibre"]),
                    work_density=float(row["work_density"]),
                )
            )
    return records


def write_summary_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def response_summary(records: list[StepRecord], load: NonlinearLoadConfig) -> dict[str, Any]:
    """Initial modulus, peak stress and final state of the loaded component."""
    loaded = COMPONENT_INDEX[load.component]
    sign = 1.0 if load.max_strain > 0 else -1.0
    peak = max(records, key=lambda r: sign * r.macro_stress[loaded])
    first = records[1] if len(records) > 1 else records[0]
    initial_modulus = (
        first.macro_stress[loaded] / first.macro_strain[loaded]
        if first.macro_strain[loaded] != 0.0
        else None
    )
    return {
        "initial_modulus": initial_modulus,
        "peak_stress": peak.macro_stress[loaded],
        "strain_at_peak": peak.macro_strain[loaded],
        "final": asdict(records[-1]),
    }
