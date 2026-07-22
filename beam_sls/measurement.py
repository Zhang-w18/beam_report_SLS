from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, List, Mapping, Sequence

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
    # S[ue, beam], Gamma[ue, service_beam, interferer_beam].
    service_power_w: np.ndarray
    # Deprecated compatibility field. Pairwise interference is already encoded
    # in Gamma and no in-repository consumer reads this duplicate B x B tensor.
    interference_power_w: np.ndarray
    gamma: np.ndarray | SparseGamma
    noise_power_w: float
    selected_rx_beam: np.ndarray  # [ue, service_beam] rx beam index
    # Link adaptation is intentionally deferred until feedback generation has
    # selected the reported service beams. Unselected entries remain -1.
    su_mcs: np.ndarray            # [ue, beam]
    su_snr_db: np.ndarray         # [ue, beam]
    su_outage: np.ndarray | None = None  # [ue, beam], valid for selected/reportable beams
    compute_backend: str = "numpy"
    backend_status: str = "OK"
    elapsed_s: float = 0.0


def _resolve_compute_backend(backend: str):
    """Return (array module, effective name, status) without requiring CuPy on CPU."""
    requested = str(backend or "numpy").strip().lower()
    aliases = {
        "cpu": "numpy",
        "np": "numpy",
        "gpu": "cupy",
        "cuda": "cupy",
        "cp": "cupy",
    }
    requested = aliases.get(requested, requested)
    if requested not in ("numpy", "cupy", "auto"):
        raise ValueError("measurement.gamma_backend must be one of: numpy, cupy, auto")
    if requested == "numpy":
        return np, "numpy", "OK"
    try:
        import cupy as cp

        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count < 1:
            raise RuntimeError("no CUDA device is visible")
        # Fail here, with an actionable message, rather than at the first large
        # allocation after channel generation.
        cp.cuda.Device().use()
        return cp, "cupy", f"OK: CUDA devices={device_count}"
    except Exception as exc:
        if requested == "auto":
            return np, "numpy", f"FALLBACK: CuPy unavailable: {type(exc).__name__}: {exc}"
        raise RuntimeError(
            "measurement.gamma_backend=cupy requires a working CuPy package "
            "matching the server CUDA version and a visible NVIDIA GPU"
        ) from exc


def _to_numpy(xp: Any, value) -> np.ndarray:
    if xp is np:
        return np.asarray(value)
    return xp.asnumpy(value)


def compute_gamma_measurement(h_freq: np.ndarray,
                              tx_beams: np.ndarray,
                              rx_beams: np.ndarray,
                              beam_ids: List[BeamId],
                              tx_power_w_per_panel: float,
                              noise_power_w: float,
                              candidate_beam_indices_by_ue: Mapping[int, Sequence[int]] | None = None,
                              compute_backend: str = "numpy",
                              ue_batch_size: int | None = None) -> MeasurementResult:
    """Compute service power, selected RX beams, and Gamma.

    h_freq shape: [num_ue, num_tx_unit, num_freq, num_rx_ant, num_tx_ant]
    tx_beams shape: [num_beams_total, num_tx_ant], each beam's tx_unit is in beam_ids
    rx_beams shape: [num_rx_beams, num_rx_ant]

    The implementation groups UEs with the same scheduling-domain beam list and
    processes each group in batches. ``compute_backend=numpy`` uses the CPU;
    ``cupy`` performs the same batched einsum/gather operations on CUDA; and
    ``auto`` uses CuPy when available, otherwise NumPy. Only batch-sized channel
    slices are copied to the GPU so the full H tensor need not fit in VRAM.

    If candidate_beam_indices_by_ue is provided, only those beams are measured
    for that UE; service/interferer Gamma entries outside the UE's scheduling
    domain remain zero and are not used by report generation.
    """
    started_at = perf_counter()
    num_u = int(h_freq.shape[0])
    num_b = len(beam_ids)
    num_f = int(h_freq.shape[2])
    num_rx = int(rx_beams.shape[1])
    xp, effective_backend, backend_status = _resolve_compute_backend(compute_backend)
    if ue_batch_size is None or int(ue_batch_size) <= 0:
        batch_size = 8 if effective_backend == "cupy" else 1
    else:
        batch_size = int(ue_batch_size)

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
    # ``interference_power_w`` duplicated the B x B data represented by Gamma,
    # was never consumed, and could be one of the largest allocations. Keep an
    # empty compatibility value instead of materializing it.
    i_pow = np.zeros((0, 0, 0), dtype=float)
    gamma_dense = None if use_sparse_gamma else np.zeros((num_u, num_b, num_b), dtype=float)
    sparse_blocks: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # Group global beams by TX-unit to avoid repeatedly reading the same channel.
    unit_to_beams = {}
    for bi, bid in enumerate(beam_ids):
        unit_to_beams.setdefault(int(bid.tx_unit), []).append(bi)

    rx_conj = xp.asarray(np.conjugate(rx_beams), dtype=xp.complex128)
    tx_beams_backend = xp.asarray(tx_beams, dtype=xp.complex128)

    # Per-site/per-sector domains naturally create repeated beam lists. Grouping
    # them lets one CUDA launch process several UEs while preserving global beam
    # indexing in the returned MeasurementResult.
    ue_groups: dict[tuple[int, ...], list[int]] = {}
    for u, allowed in enumerate(allowed_by_ue):
        if allowed:
            ue_groups.setdefault(tuple(allowed), []).append(u)

    for allowed_tuple, group_ues in ue_groups.items():
        allowed = list(allowed_tuple)
        allowed_arr = np.asarray(allowed, dtype=int)
        allowed_set = set(allowed)
        local_pos = {int(b): i for i, b in enumerate(allowed)}
        local_unit_to_beams = {
            unit: [bi for bi in inds if bi in allowed_set]
            for unit, inds in unit_to_beams.items()
        }
        unit_compute_specs = [
            (
                unit,
                inds,
                xp.asarray([local_pos[int(b)] for b in inds], dtype=xp.int64),
            )
            for unit, inds in local_unit_to_beams.items()
            if inds
        ]
        for start in range(0, len(group_ues), batch_size):
            batch_ues = np.asarray(group_ues[start:start + batch_size], dtype=int)
            batch_u = len(batch_ues)
            num_allowed = len(allowed)
            # hf_allowed[u, b, f, nr] = H[u,txunit(b),f,:,:] @ w[b].
            hf_allowed = xp.zeros(
                (batch_u, num_allowed, num_f, num_rx), dtype=xp.complex128
            )
            for unit, inds, locs in unit_compute_specs:
                h = xp.asarray(h_freq[batch_ues, unit], dtype=xp.complex128)
                fmat = tx_beams_backend[inds]
                hf_allowed[:, locs, :, :] = xp.einsum(
                    "ufrt,kt->ukfr", h, fmat, optimize=True
                )

            # z_all[u, r, b, f] = q_r^H H_u w_b
            z_all = xp.einsum("rn,ubfn->urbf", rx_conj, hf_allowed, optimize=True)
            pwr_rb = xp.mean(xp.abs(z_all) ** 2, axis=3)  # [U_batch, R, B_allowed]
            selected_batch = xp.argmax(pwr_rb, axis=1)   # [U_batch, B_allowed]
            u_index = xp.arange(batch_u)[:, None]
            b_index = xp.arange(num_allowed)[None, :]
            service_batch = float(tx_power_w_per_panel) * pwr_rb[
                u_index, selected_batch, b_index
            ]

            # Each service beam m uses its selected RX state against every
            # potential interferer n. Advanced indexing creates [U, m, n].
            interference_batch = float(tx_power_w_per_panel) * pwr_rb[
                u_index, selected_batch, :
            ]
            gamma_batch = service_batch[:, :, None] / xp.maximum(
                interference_batch + float(noise_power_w), 1e-30
            )
            diag = xp.arange(num_allowed)
            gamma_batch[:, diag, diag] = service_batch / max(float(noise_power_w), 1e-30)

            selected_np = _to_numpy(xp, selected_batch).astype(int, copy=False)
            service_np = _to_numpy(xp, service_batch).astype(float, copy=False)
            gamma_np = _to_numpy(xp, gamma_batch).astype(float, copy=False)
            selected[np.ix_(batch_ues, allowed_arr)] = selected_np
            s[np.ix_(batch_ues, allowed_arr)] = service_np
            for local_u, global_u in enumerate(batch_ues):
                if use_sparse_gamma:
                    sparse_blocks[int(global_u)] = (allowed_arr, gamma_np[local_u])
                else:
                    gamma_dense[int(global_u)][np.ix_(allowed_arr, allowed_arr)] = gamma_np[local_u]

    gamma = SparseGamma(num_u, num_b, sparse_blocks) if use_sparse_gamma else gamma_dense
    if use_sparse_gamma:
        su_snr_db = np.full((num_u, num_b), float(lin_to_db(0.0)), dtype=float)
        for u in range(num_u):
            for b in allowed_by_ue[u]:
                su_snr_db[u, b] = float(lin_to_db(float(gamma[u, b, b])))
    else:
        su_snr_db = lin_to_db(np.diagonal(gamma, axis1=1, axis2=2))
    # Do not invoke link abstraction for every measured beam. Feedback first
    # selects its top service beams from SU-SNR; make_reports() then fills only
    # those entries. -1 makes accidental use of an unadapted beam visible.
    su_mcs = np.full_like(su_snr_db, -1, dtype=int)
    su_outage = np.zeros_like(su_snr_db, dtype=bool)

    return MeasurementResult(service_power_w=s,
                             interference_power_w=i_pow,
                             gamma=gamma,
                             noise_power_w=float(noise_power_w),
                             selected_rx_beam=selected,
                             su_mcs=su_mcs,
                             su_snr_db=su_snr_db,
                             su_outage=su_outage,
                             compute_backend=effective_backend,
                             backend_status=backend_status,
                             elapsed_s=float(perf_counter() - started_at))
