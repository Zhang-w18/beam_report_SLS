from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Set

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
    su_outage: bool = False

    def to_dict(self, beam_ids: Sequence[BeamId]) -> Dict:
        return {
            "beam_index": self.beam_index,
            "beam_id": beam_ids[self.beam_index].short(),
            "su_snr_db": self.su_snr_db,
            "su_mcs": self.su_mcs,
            "su_outage": self.su_outage,
            "conflict_beams": [beam_ids[i].short() for i in sorted(self.conflict_beams)],
        }


@dataclass
class UEReport:
    ue_id: int
    scheme: str
    candidates: List[ServiceCandidate]
    site_id: int | None = None
    serving_cell: int | None = None
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
            "site_id": self.site_id,
            "serving_cell": self.serving_cell,
            "scheme": self.scheme,
            "candidates": [c.to_dict(beam_ids) for c in self.candidates],
        }


def _top_service_indices(meas: MeasurementResult,
                         ue_id: int,
                         k: int,
                         allowed_beam_indices: Sequence[int] | None = None) -> List[int]:
    scores = meas.su_snr_db[ue_id]
    if allowed_beam_indices is None:
        idx = np.argsort(scores)[::-1][:int(k)]
        return [int(i) for i in idx]
    allowed = np.asarray([int(i) for i in allowed_beam_indices], dtype=int)
    if allowed.size == 0:
        return []
    order = np.argsort(scores[allowed])[::-1][:int(k)]
    idx = allowed[order]
    return [int(i) for i in idx]


def make_reports(meas: MeasurementResult,
                 beam_ids: Sequence[BeamId],
                 schemes: Sequence[str],
                 k1: int,
                 oracle_top_k: int,
                 k2: int,
                 threshold_db: float,
                 ue_site_ids: Mapping[int, int] | None = None,
                 ue_serving_cells: Mapping[int, int] | None = None,
                 candidate_beam_indices_by_ue: Mapping[int, Sequence[int]] | None = None) -> Dict[str, List[UEReport]]:
    out: Dict[str, List[UEReport]] = {s: [] for s in schemes}
    num_u = meas.service_power_w.shape[0]
    threshold_lin = float(10.0 ** (float(threshold_db) / 10.0))

    for scheme in schemes:
        for u in range(num_u):
            allowed = None if candidate_beam_indices_by_ue is None else candidate_beam_indices_by_ue.get(u, [])
            if scheme == "full_gamma":
                top = _top_service_indices(meas, u, oracle_top_k, allowed)
            else:
                top = _top_service_indices(meas, u, k1, allowed)
            allowed_set = None if allowed is None else {int(i) for i in allowed}
            cands: List[ServiceCandidate] = []
            for m in top:
                conflicts: Set[int] = set()
                if scheme == "topk_conflict_id":
                    row = meas.gamma[u, m, :].copy()
                    row[m] = np.inf
                    if allowed_set is None:
                        candidates = [int(i) for i in np.argsort(row) if int(i) != int(m)]
                    else:
                        candidates = sorted((int(i) for i in allowed_set if int(i) != int(m)),
                                            key=lambda i: float(row[i]))
                    conflicts = set(candidates[:int(k2)])
                elif scheme == "threshold_conflict_set":
                    row = meas.gamma[u, m, :]
                    if allowed_set is None:
                        conflicts = set(int(i) for i in np.where(row < threshold_lin)[0] if int(i) != m)
                    else:
                        conflicts = set(int(i) for i in allowed_set if int(i) != m and float(row[int(i)]) < threshold_lin)
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
                    su_outage=False if meas.su_outage is None else bool(meas.su_outage[u, m]),
                ))
            rep = UEReport(ue_id=u,
                           scheme=scheme,
                           candidates=cands,
                           site_id=None if ue_site_ids is None else int(ue_site_ids.get(u, 0)),
                           serving_cell=None if ue_serving_cells is None else int(ue_serving_cells.get(u, 0)))
            if scheme == "full_gamma":
                rep.full_gamma = meas.gamma[u].copy()
                rep.full_service_power_w = meas.service_power_w[u].copy()
                rep.full_noise_power_w = float(meas.noise_power_w)
            out[scheme].append(rep)
    return out
