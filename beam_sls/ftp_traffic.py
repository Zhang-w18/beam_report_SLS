from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FTPArrivalOnlyConfig:
    """3GPP FTP Model 3 arrival-only parameters.

    File size is retained as metadata.  Arrival-only mode does not model
    transmission completion or a queue, so it does not affect UE counts.
    """

    arrival_rate_per_ue_s: float
    file_size_mbytes: float
    observation_interval_s: float

    @classmethod
    def from_dict(cls, cfg: Mapping) -> "FTPArrivalOnlyConfig":
        rate = float(cfg.get("arrival_rate_per_ue_s", 0.0))
        size = float(cfg.get("file_size_mbytes", 0.5))
        interval = float(cfg.get("observation_interval_ms", 100.0)) * 1e-3
        if rate < 0.0:
            raise ValueError("service_beam_statistics.traffic.arrival_rate_per_ue_s must be >= 0")
        if size <= 0.0:
            raise ValueError("service_beam_statistics.traffic.file_size_mbytes must be > 0")
        if interval <= 0.0:
            raise ValueError("service_beam_statistics.observation_interval_ms must be > 0")
        return cls(rate, size, interval)

    @property
    def mean_arrivals_per_ue_per_window(self) -> float:
        return float(self.arrival_rate_per_ue_s * self.observation_interval_s)


FTPModel3ArrivalOnlyConfig = FTPArrivalOnlyConfig


def sample_ue_arrivals(
    rng: np.random.Generator,
    traffic: FTPArrivalOnlyConfig,
    num_ues: int,
) -> np.ndarray:
    """Sample one independent Poisson arrival count for every fixed UE."""
    if int(num_ues) < 0:
        raise ValueError("num_ues must be >= 0")
    return np.asarray(
        rng.poisson(traffic.mean_arrivals_per_ue_per_window, size=int(num_ues)),
        dtype=int,
    )


def count_arriving_ues_by_beam(
    ue_ids: Sequence[int],
    num_arrivals: Sequence[int],
    best_service_beam_by_ue: Mapping[int, int],
    num_beams: int,
) -> np.ndarray:
    """Count each UE once, using its cached service beam."""
    if len(ue_ids) != len(num_arrivals):
        raise ValueError("ue_ids and num_arrivals must have the same length")
    if int(num_beams) < 0:
        raise ValueError("num_beams must be >= 0")
    counts = np.zeros(int(num_beams), dtype=int)
    for ue_id, arrivals in zip(ue_ids, num_arrivals):
        if int(arrivals) <= 0:
            continue
        try:
            beam_index = int(best_service_beam_by_ue[int(ue_id)])
        except KeyError as exc:
            raise KeyError(f"No cached service beam for ue_id={ue_id}") from exc
        if not 0 <= beam_index < int(num_beams):
            raise ValueError(f"Cached beam index {beam_index} is outside [0, {num_beams})")
        counts[beam_index] += 1
    return counts


def count_ues_by_beam(
    ue_ids: Sequence[int],
    best_service_beam_by_ue: Mapping[int, int],
    num_beams: int,
) -> np.ndarray:
    """Count every candidate UE once by its cached best service beam."""
    if int(num_beams) < 0:
        raise ValueError("num_beams must be >= 0")
    counts = np.zeros(int(num_beams), dtype=int)
    for ue_id in ue_ids:
        try:
            beam_index = int(best_service_beam_by_ue[int(ue_id)])
        except KeyError as exc:
            raise KeyError(f"No cached service beam for ue_id={ue_id}") from exc
        if not 0 <= beam_index < int(num_beams):
            raise ValueError(f"Cached beam index {beam_index} is outside [0, {num_beams})")
        counts[beam_index] += 1
    return counts


# Explicit name for the window-level definition used by the collision metric.
count_active_ues_by_beam = count_arriving_ues_by_beam
