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
    """Sparse Gamma storage with global service/interferer beam access."""

    def __init__(self, num_ues: int, num_beams: int, blocks: Mapping[int, tuple]):
        self.shape = (int(num_ues), int(num_beams), int(num_beams))
        self._blocks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for u, raw in blocks.items():
            if len(raw) == 2:
                indices, block = raw
                self._blocks[int(u)] = (indices, indices, block)
            else:
                service_indices, interferer_indices, block = raw
                self._blocks[int(u)] = (
                    service_indices, interferer_indices, block
                )
        self._service_local_index = {
            u: {int(b): i for i, b in enumerate(service_indices)}
            for u, (service_indices, _interferer_indices, _block)
            in self._blocks.items()
        }
        self._interferer_local_index = {
            u: {int(b): i for i, b in enumerate(interferer_indices)}
            for u, (_service_indices, interferer_indices, _block)
            in self._blocks.items()
        }

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 3:
                raise IndexError("SparseGamma expects [ue, service_beam, interferer_beam]")
            u, m, n = key
            return self._get(int(u), m, n)
        return SparseUEGammaView(self, int(key))

    def _get(self, u: int, m, n):
        service_indices, interferer_indices, block = self._blocks.get(
            int(u),
            (
                np.asarray([], dtype=int),
                np.asarray([], dtype=int),
                np.zeros((0, 0), dtype=float),
            ),
        )
        service_local = self._service_local_index.get(int(u), {})
        interferer_local = self._interferer_local_index.get(int(u), {})
        if isinstance(n, slice):
            out = np.zeros((self.shape[2],), dtype=float)
            lm = service_local.get(int(m))
            if lm is None:
                return out[n]
            out[interferer_indices] = block[lm, :]
            return out[n]
        lm = service_local.get(int(m))
        ln = interferer_local.get(int(n))
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


def associate_ues_by_average_rsrp(h_freq: np.ndarray,
                                  tx_beams: np.ndarray,
                                  rx_beams: np.ndarray,
                                  beam_ids: Sequence[BeamId],
                                  topology,
                                  tx_power_w_per_panel: float) -> Dict[int, Dict[int, float]]:
    """Associate each UE to the cell with maximum wideband-average beam RSRP.

    For cell c, RSRP_bar[u,c] is the maximum over its TX beams and UE RX
    beams of P_tx * mean_f(|q^H H[u,tx(b),f] w_b|^2). Cell-ID breaks ties.
    """
    if int(h_freq.shape[-1]) != int(tx_beams.shape[1]):
        raise ValueError("TX channel/beam dimension mismatch during cell association")
    rx_conj = np.conjugate(np.asarray(rx_beams, dtype=np.complex128))
    tx_matrix = np.asarray(tx_beams, dtype=np.complex128)
    beams_by_unit: Dict[int, List[int]] = {}
    for index, beam in enumerate(beam_ids):
        beams_by_unit.setdefault(int(beam.tx_unit), []).append(int(index))
    scores: Dict[int, Dict[int, float]] = {}
    for ue_index, ue in enumerate(topology.ues):
        cell_scores: Dict[int, float] = {}
        for tx_unit, beam_indices in beams_by_unit.items():
            h = np.asarray(h_freq[ue_index, tx_unit], dtype=np.complex128)
            z = np.einsum(
                "rn,fnt,kt->rkf",
                rx_conj,
                h,
                tx_matrix[beam_indices],
                optimize=True,
            )
            powers = float(tx_power_w_per_panel) * np.mean(
                np.abs(z) ** 2, axis=2
            )
            for local_index, beam_index in enumerate(beam_indices):
                cell = int(beam_ids[beam_index].cell)
                value = float(np.max(powers[:, local_index]))
                cell_scores[cell] = max(cell_scores.get(cell, 0.0), value)
        if not cell_scores:
            raise ValueError(f"No cell RSRP candidates for UE {ue.ue_id}")
        serving_cell = min(
            cell_scores,
            key=lambda cell: (-cell_scores[cell], int(cell)),
        )
        ue.serving_cell = int(serving_cell)
        ue.site_id = int(topology.sector_by_cell(serving_cell).site_id)
        scores[int(ue.ue_id)] = cell_scores
    return scores


def compute_gamma_measurement(h_freq: np.ndarray,
                              tx_beams: np.ndarray,
                              rx_beams: np.ndarray,
                              beam_ids: List[BeamId],
                              tx_power_w_per_panel: float,
                              noise_power_w: float,
                              candidate_beam_indices_by_ue: Mapping[int, Sequence[int]] | None = None,
                              compute_backend: str = "numpy",
                              ue_batch_size: int | None = None,
                              service_beam_indices_by_ue: Mapping[int, Sequence[int]] | None = None,
                              interference_beam_indices_by_ue: Mapping[int, Sequence[int]] | None = None) -> MeasurementResult:
    """Compute service power, selected RX beams, and Gamma.

    h_freq shape: [num_ue, num_tx_unit, num_freq, num_rx_ant, num_tx_ant]
    tx_beams shape: [num_beams_total, num_tx_ant], each beam's tx_unit is in beam_ids
    rx_beams shape: [num_rx_beams, num_rx_ant]

    The implementation groups UEs with the same service/interference beam lists and
    processes each group in batches. ``compute_backend=numpy`` uses the CPU;
    ``cupy`` performs the same batched einsum/gather operations on CUDA; and
    ``auto`` uses CuPy when available, otherwise NumPy. Only batch-sized channel
    slices are copied to the GPU so the full H tensor need not fit in VRAM.

    ``service_beam_indices_by_ue`` selects beams eligible to serve each UE.
    ``interference_beam_indices_by_ue`` selects beams measured as possible
    interferers. The legacy ``candidate_beam_indices_by_ue`` applies the same
    list to both dimensions.
    """
    started_at = perf_counter()
    if int(h_freq.shape[-1]) != int(tx_beams.shape[1]):
        raise ValueError(
            "TX channel/beam dimension mismatch: "
            f"h_freq ntx={h_freq.shape[-1]}, tx_beams={tx_beams.shape[1]}"
        )
    num_u = int(h_freq.shape[0])
    num_b = len(beam_ids)
    num_f = int(h_freq.shape[2])
    num_rx = int(rx_beams.shape[1])
    xp, effective_backend, backend_status = _resolve_compute_backend(compute_backend)
    if ue_batch_size is None or int(ue_batch_size) <= 0:
        batch_size = 8 if effective_backend == "cupy" else 1
    else:
        batch_size = int(ue_batch_size)

    if candidate_beam_indices_by_ue is not None:
        if service_beam_indices_by_ue is not None or interference_beam_indices_by_ue is not None:
            raise ValueError(
                "Use candidate_beam_indices_by_ue or the separate service/"
                "interference mappings, not both"
            )
        service_beam_indices_by_ue = candidate_beam_indices_by_ue
        interference_beam_indices_by_ue = candidate_beam_indices_by_ue

    def normalize_indices(mapping, u: int) -> List[int]:
        if mapping is None:
            return list(range(num_b))
        raw = mapping.get(u, [])
        seen = set()
        result = []
        for b in raw:
            bi = int(b)
            if 0 <= bi < num_b and bi not in seen:
                result.append(bi)
                seen.add(bi)
        return result

    service_by_ue: List[List[int]] = []
    interferer_by_ue: List[List[int]] = []
    for u in range(num_u):
        service = normalize_indices(service_beam_indices_by_ue, u)
        measured_interferers = normalize_indices(
            interference_beam_indices_by_ue, u
        )
        # The desired beam must also be available on the interferer dimension
        # so Gamma[m,m] can represent standalone SNR.
        interferers = list(dict.fromkeys(measured_interferers + service))
        service_by_ue.append(service)
        interferer_by_ue.append(interferers)

    use_sparse_gamma = (
        service_beam_indices_by_ue is not None
        or interference_beam_indices_by_ue is not None
    )

    s = np.zeros((num_u, num_b), dtype=float)
    selected = np.zeros((num_u, num_b), dtype=int)
    # ``interference_power_w`` duplicated the B x B data represented by Gamma,
    # was never consumed, and could be one of the largest allocations. Keep an
    # empty compatibility value instead of materializing it.
    i_pow = np.zeros((0, 0, 0), dtype=float)
    gamma_dense = None if use_sparse_gamma else np.zeros((num_u, num_b, num_b), dtype=float)
    sparse_blocks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # Group global beams by TX-unit to avoid repeatedly reading the same channel.
    unit_to_beams = {}
    for bi, bid in enumerate(beam_ids):
        unit_to_beams.setdefault(int(bid.tx_unit), []).append(bi)

    rx_conj = xp.asarray(np.conjugate(rx_beams), dtype=xp.complex128)
    tx_beams_backend = xp.asarray(tx_beams, dtype=xp.complex128)

    # Repeated service/measurement-domain pairs can share one batched kernel
    # while preserving global beam indexing in the returned MeasurementResult.
    ue_groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for u, (service, interferers) in enumerate(zip(service_by_ue, interferer_by_ue)):
        if service:
            ue_groups.setdefault(
                (tuple(service), tuple(interferers)), []
            ).append(u)

    for (service_tuple, interferer_tuple), group_ues in ue_groups.items():
        service_indices = list(service_tuple)
        interferer_indices = list(interferer_tuple)
        compute_indices = list(dict.fromkeys(service_indices + interferer_indices))
        service_arr = np.asarray(service_indices, dtype=int)
        interferer_arr = np.asarray(interferer_indices, dtype=int)
        compute_set = set(compute_indices)
        local_pos = {int(b): i for i, b in enumerate(compute_indices)}
        service_locs = xp.asarray(
            [local_pos[b] for b in service_indices], dtype=xp.int64
        )
        interferer_locs = xp.asarray(
            [local_pos[b] for b in interferer_indices], dtype=xp.int64
        )
        local_unit_to_beams = {
            unit: [bi for bi in inds if bi in compute_set]
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
            num_compute = len(compute_indices)
            # hf_allowed[u, b, f, nr] = H[u,txunit(b),f,:,:] @ w[b].
            hf_allowed = xp.zeros(
                (batch_u, num_compute, num_f, num_rx), dtype=xp.complex128
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
            service_pwr_rb = pwr_rb[:, :, service_locs]
            selected_batch = xp.argmax(service_pwr_rb, axis=1)
            u_index = xp.arange(batch_u)[:, None]
            service_index = xp.arange(len(service_indices))[None, :]
            service_batch = float(tx_power_w_per_panel) * service_pwr_rb[
                u_index, selected_batch, service_index
            ]

            # Each service beam m uses its selected RX state against every
            # potential interferer n. Advanced indexing creates [U, m, n].
            interference_batch = float(tx_power_w_per_panel) * pwr_rb[
                u_index, selected_batch, :
            ][:, :, interferer_locs]
            gamma_batch = service_batch[:, :, None] / xp.maximum(
                interference_batch + float(noise_power_w), 1e-30
            )
            interferer_pos = {
                int(b): i for i, b in enumerate(interferer_indices)
            }
            for service_pos, beam_index in enumerate(service_indices):
                gamma_batch[:, service_pos, interferer_pos[beam_index]] = (
                    service_batch[:, service_pos]
                    / max(float(noise_power_w), 1e-30)
                )

            selected_np = _to_numpy(xp, selected_batch).astype(int, copy=False)
            service_np = _to_numpy(xp, service_batch).astype(float, copy=False)
            gamma_np = _to_numpy(xp, gamma_batch).astype(float, copy=False)
            selected[np.ix_(batch_ues, service_arr)] = selected_np
            s[np.ix_(batch_ues, service_arr)] = service_np
            for local_u, global_u in enumerate(batch_ues):
                if use_sparse_gamma:
                    sparse_blocks[int(global_u)] = (
                        service_arr, interferer_arr, gamma_np[local_u]
                    )
                else:
                    gamma_dense[int(global_u)][
                        np.ix_(service_arr, interferer_arr)
                    ] = gamma_np[local_u]

    gamma = SparseGamma(num_u, num_b, sparse_blocks) if use_sparse_gamma else gamma_dense
    if use_sparse_gamma:
        su_snr_db = np.full((num_u, num_b), float(lin_to_db(0.0)), dtype=float)
        for u in range(num_u):
            for b in service_by_ue[u]:
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
