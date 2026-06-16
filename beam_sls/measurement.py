from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

import numpy as np

from .codebook import BeamId
from .utils import lin_to_db


class SparseUEGammaView:
    def __init__(self, parent: "SparseGamma", ue_id: int):
        self._parent = parent
        self._ue_id = int(ue_id)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            return self._parent[self._ue_id, key[0], key[1]]
        raise IndexError("SparseUEGammaView expects [service_beam, interferer_beam]")

    def copy(self):
        return self


class SparseGamma:
    """Domain-local Gamma storage with global beam-index access.

    For each UE, only the scheduling-domain block is stored. Accessing a pair
    outside the UE's domain returns zero, matching the dense default fill value.
    """

    def __init__(self, num_ues: int, num_beams: int, blocks: Mapping[int, tuple[np.ndarray, np.ndarray]]):
        self.shape = (int(num_ues), int(num_beams), int(num_beams))
        self._blocks = blocks
        self._local_index: dict[int, dict[int, int]] = {
            int(u): {int(b): i for i, b in enumerate(indices)}
            for u, (indices, _block) in blocks.items()
        }

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 3:
                raise IndexError("SparseGamma expects [ue, service_beam, interferer_beam]")
            u, m, n = key
            return self._get(int(u), m, n)
        return SparseUEGammaView(self, int(key))

    def _get(self, u: int, m, n):
        indices, block = self._blocks.get(int(u), (np.asarray([], dtype=int), np.zeros((0, 0), dtype=float)))
        local = self._local_index.get(int(u), {})
        if isinstance(n, slice):
            out = np.zeros((self.shape[2],), dtype=float)
            lm = local.get(int(m))
            if lm is None:
                return out[n]
            out[indices] = block[lm, :]
            return out[n]
        lm = local.get(int(m))
        ln = local.get(int(n))
        if lm is None or ln is None:
            return 0.0
        return float(block[lm, ln])


@dataclass
class MeasurementResult:
    # S[ue, beam], I[ue, service_beam, interferer_beam], Gamma same shape.
    service_power_w: np.ndarray
    interference_power_w: np.ndarray
    gamma: np.ndarray | SparseGamma
    noise_power_w: float
    selected_rx_beam: np.ndarray  # [ue, service_beam] rx beam index
    su_mcs: np.ndarray            # [ue, beam]
    su_snr_db: np.ndarray         # [ue, beam]


def compute_gamma_measurement(h_freq: np.ndarray,
                              tx_beams: np.ndarray,
                              rx_beams: np.ndarray,
                              beam_ids: List[BeamId],
                              tx_power_w_per_panel: float,
                              noise_power_w: float,
                              link_adapter=None,
                              candidate_beam_indices_by_ue: Mapping[int, Sequence[int]] | None = None) -> MeasurementResult:
    """Compute S, I, Gamma from channel and DFT beams.

    h_freq shape: [num_ue, num_tx_unit, num_freq, num_rx_ant, num_tx_ant]
    tx_beams shape: [num_beams_total, num_tx_ant], each beam's tx_unit is in beam_ids
    rx_beams shape: [num_rx_beams, num_rx_ant]

    v2.4 uses a vectorized implementation. For every UE, it first computes the
    TX-beam effective receive vector h(f)w, then projects the result onto all RX
    beams. If candidate_beam_indices_by_ue is provided, only those beams are
    measured for that UE; service/interferer Gamma entries outside the UE's
    scheduling domain remain zero and are not used by report generation.
    """
    num_u = int(h_freq.shape[0])
    num_b = len(beam_ids)
    num_r = int(rx_beams.shape[0])
    num_f = int(h_freq.shape[2])
    num_rx = int(rx_beams.shape[1])

    allowed_by_ue: List[List[int]] = []
    for u in range(num_u):
        if candidate_beam_indices_by_ue is None:
            allowed = list(range(num_b))
        else:
            raw = candidate_beam_indices_by_ue.get(u, [])
            seen = set()
            allowed = []
            for b in raw:
                bi = int(b)
                if 0 <= bi < num_b and bi not in seen:
                    allowed.append(bi)
                    seen.add(bi)
        allowed_by_ue.append(allowed)

    use_sparse_gamma = candidate_beam_indices_by_ue is not None

    s = np.zeros((num_u, num_b), dtype=float)
    selected = np.zeros((num_u, num_b), dtype=int)
    i_pow = np.zeros((0, 0, 0), dtype=float) if use_sparse_gamma else np.zeros((num_u, num_b, num_b), dtype=float)
    gamma_dense = None if use_sparse_gamma else np.zeros((num_u, num_b, num_b), dtype=float)
    sparse_blocks: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # Group global beams by TX-unit to avoid repeatedly reading the same channel.
    unit_to_beams = {}
    for bi, bid in enumerate(beam_ids):
        unit_to_beams.setdefault(int(bid.tx_unit), []).append(bi)

    rx_conj = np.conjugate(rx_beams).astype(np.complex128)

    for u in range(num_u):
        allowed = allowed_by_ue[u]
        if not allowed:
            continue
        allowed_set = set(allowed)
        local_unit_to_beams = {
            unit: [bi for bi in inds if bi in allowed_set]
            for unit, inds in unit_to_beams.items()
        }

        # hf_allowed[k, f, nr] = H_u,txunit(allowed[k])[f,:,:] @ w_allowed[k].
        hf_allowed = np.zeros((len(allowed), num_f, num_rx), dtype=np.complex128)
        local_pos = {int(b): i for i, b in enumerate(allowed)}
        for unit, inds in local_unit_to_beams.items():
            if not inds:
                continue
            h = h_freq[u, unit]  # [F, Nr, Nt]
            fmat = tx_beams[inds]  # [K, Nt]
            locs = [local_pos[int(b)] for b in inds]
            hf_allowed[np.asarray(locs, dtype=int), :, :] = np.einsum("frt,kt->kfr", h, fmat, optimize=True)

        # z_all[r, b, f] = q_r^H H w_b
        z_all = np.einsum("rn,bfn->rbf", rx_conj, hf_allowed, optimize=True)
        pwr_rb_allowed = np.mean(np.abs(z_all) ** 2, axis=2)  # [R, B_allowed]
        selected_allowed = np.argmax(pwr_rb_allowed, axis=0)
        allowed_arr = np.asarray(allowed, dtype=int)
        selected[u, allowed_arr] = selected_allowed
        s[u, allowed_arr] = float(tx_power_w_per_panel) * pwr_rb_allowed[selected_allowed, np.arange(len(allowed))]

        gamma_local = np.zeros((len(allowed), len(allowed)), dtype=float)
        for local_m, m in enumerate(allowed):
            row = float(tx_power_w_per_panel) * pwr_rb_allowed[int(selected[u, m]), :]
            gamma_row = s[u, m] / np.maximum(row + float(noise_power_w), 1e-30)
            gamma_row[local_m] = s[u, m] / max(float(noise_power_w), 1e-30)
            gamma_local[local_m, :] = gamma_row
            if gamma_dense is not None:
                i_pow[u, m, allowed_arr] = row
                i_pow[u, m, m] = 0.0
                gamma_dense[u, m, allowed_arr] = gamma_row
                gamma_dense[u, m, m] = gamma_row[local_m]
        if use_sparse_gamma:
            sparse_blocks[int(u)] = (allowed_arr, gamma_local)

    gamma = SparseGamma(num_u, num_b, sparse_blocks) if use_sparse_gamma else gamma_dense
    if use_sparse_gamma:
        su_snr_db = np.full((num_u, num_b), float(lin_to_db(0.0)), dtype=float)
        for u in range(num_u):
            for b in allowed_by_ue[u]:
                su_snr_db[u, b] = float(lin_to_db(float(gamma[u, b, b])))
    else:
        su_snr_db = lin_to_db(np.diagonal(gamma, axis1=1, axis2=2))
    su_mcs = np.zeros_like(su_snr_db, dtype=int)
    for u in range(num_u):
        for b in allowed_by_ue[u]:
            if link_adapter is not None:
                su_mcs[u, b] = int(link_adapter.select_mcs_from_sinr_lin(float(gamma[u, b, b])))
            else:
                from .mcs import select_mcs_from_sinr_lin
                su_mcs[u, b] = select_mcs_from_sinr_lin(gamma[u, b, b]).index

    return MeasurementResult(service_power_w=s,
                             interference_power_w=i_pow,
                             gamma=gamma,
                             noise_power_w=float(noise_power_w),
                             selected_rx_beam=selected,
                             su_mcs=su_mcs,
                             su_snr_db=su_snr_db)
