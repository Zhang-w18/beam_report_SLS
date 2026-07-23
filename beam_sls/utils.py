from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def occupied_bandwidth_hz(cfg: Mapping[str, Any]) -> float:
    """Return active PDSCH bandwidth derived from PRBs and subcarrier spacing."""
    num_prbs = int(cfg.get("pdsch", {}).get("num_prbs", 0))
    scs_khz = float(cfg.get("system", {}).get("subcarrier_spacing_khz", 0.0))
    if num_prbs <= 0:
        raise ValueError("pdsch.num_prbs must be > 0")
    if scs_khz <= 0.0:
        raise ValueError("system.subcarrier_spacing_khz must be > 0")
    return float(num_prbs * 12.0 * scs_khz * 1e3)


def db_to_lin(x_db: float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, np.asarray(x_db) / 10.0)


def lin_to_db(x: float | np.ndarray, floor: float = 1e-30) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x), floor))


def dbm_to_watt(x_dbm: float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, (np.asarray(x_dbm) - 30.0) / 10.0)


def watt_to_dbm(x_watt: float | np.ndarray, floor: float = 1e-30) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x_watt), floor)) + 30.0


def thermal_noise_watt(bandwidth_hz: float,
                       noise_density_dbm_per_hz: float = -174.0,
                       noise_figure_db: float = 7.0) -> float:
    n_dbm = noise_density_dbm_per_hz + 10.0 * math.log10(float(bandwidth_hz)) + noise_figure_db
    return float(dbm_to_watt(n_dbm))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def rng_from_seed(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def write_csv(path: str | os.PathLike[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Preserve insertion order from the first row and append unseen keys.
    fieldnames: List[str] = list(rows[0].keys())
    seen = set(fieldnames)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_value(r.get(k, "")) for k in fieldnames})


def _csv_value(v: Any) -> Any:
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, np.generic):
        return v.item()
    return v


def write_json(path: str | os.PathLike[str], obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if hasattr(obj, "to_dict"):
        return to_jsonable(obj.to_dict())
    return obj


def percentile(values: Sequence[float], pct: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def flatten(list_of_lists: Iterable[Iterable[Any]]) -> List[Any]:
    return [x for xs in list_of_lists for x in xs]
