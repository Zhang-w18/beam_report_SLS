from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set

import numpy as np

from .codebook import BeamId
from .measurement import MeasurementResult
from .utils import lin_to_db


@dataclass
class ServiceCandidate:
    beam_index: int
    su_snr_db: float
    su_mcs: int
    conflict_beams: Set[int] = field(default_factory=set)

    def to_dict(self, beam_ids: Sequence[BeamId]) -> Dict:
        return {
            "beam_index": self.beam_index,
            "beam_id": beam_ids[self.beam_index].short(),
            "su_snr_db": self.su_snr_db,
            "su_mcs": self.su_mcs,
            "conflict_beams": [beam_ids[i].short() for i in sorted(self.conflict_beams)],
        }


@dataclass
class UEReport:
    ue_id: int
    scheme: str
    candidates: List[ServiceCandidate]
    # For oracle only; scheduler uses these if available.
    full_gamma: np.ndarray | None = None
    full_service_power_w: np.ndarray | None = None
    full_noise_power_w: float | None = None

    def candidate_indices(self) -> List[int]:
        return [c.beam_index for c in self.candidates]

    def candidate_by_beam(self, b: int) -> ServiceCandidate | None:
        for c in self.candidates:
            if c.beam_index == b:
                return c
        return None

    def to_dict(self, beam_ids: Sequence[BeamId]) -> Dict:
        return {
            "ue_id": self.ue_id,
            "scheme": self.scheme,
            "candidates": [c.to_dict(beam_ids) for c in self.candidates],
        }


def _top_service_indices(meas: MeasurementResult, ue_id: int, k: int) -> List[int]:
    scores = meas.su_snr_db[ue_id]
    idx = np.argsort(scores)[::-1][:int(k)]
    return [int(i) for i in idx]


def make_reports(meas: MeasurementResult,
                 beam_ids: Sequence[BeamId],
                 schemes: Sequence[str],
                 k1: int,
                 oracle_top_k: int,
                 k2: int,
                 threshold_db: float) -> Dict[str, List[UEReport]]:
    out: Dict[str, List[UEReport]] = {s: [] for s in schemes}
    num_u = meas.service_power_w.shape[0]
    threshold_lin = float(10.0 ** (float(threshold_db) / 10.0))

    for scheme in schemes:
        for u in range(num_u):
            if scheme == "full_gamma":
                top = _top_service_indices(meas, u, oracle_top_k)
            else:
                top = _top_service_indices(meas, u, k1)
            cands: List[ServiceCandidate] = []
            for m in top:
                conflicts: Set[int] = set()
                if scheme == "topk_conflict_id":
                    row = meas.gamma[u, m, :].copy()
                    row[m] = np.inf
                    conflicts = set(int(i) for i in np.argsort(row)[:int(k2)])
                elif scheme == "threshold_conflict_set":
                    row = meas.gamma[u, m, :]
                    conflicts = set(int(i) for i in np.where(row < threshold_lin)[0] if int(i) != m)
                elif scheme == "full_gamma":
                    # Oracle does not need ID-only conflict sets.
                    conflicts = set()
                elif scheme == "baseline":
                    conflicts = set()
                else:
                    raise ValueError(f"Unknown feedback scheme: {scheme}")
                cands.append(ServiceCandidate(
                    beam_index=int(m),
                    su_snr_db=float(meas.su_snr_db[u, m]),
                    su_mcs=int(meas.su_mcs[u, m]),
                    conflict_beams=conflicts,
                ))
            rep = UEReport(ue_id=u, scheme=scheme, candidates=cands)
            if scheme == "full_gamma":
                rep.full_gamma = meas.gamma[u].copy()
                rep.full_service_power_w = meas.service_power_w[u].copy()
                rep.full_noise_power_w = float(meas.noise_power_w)
            out[scheme].append(rep)
    return out
