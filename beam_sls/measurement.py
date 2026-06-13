from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .codebook import BeamId
from .utils import lin_to_db


@dataclass
class MeasurementResult:
    # S[ue, beam], I[ue, service_beam, interferer_beam], Gamma same shape.
    service_power_w: np.ndarray
    interference_power_w: np.ndarray
    gamma: np.ndarray
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
                              link_adapter=None) -> MeasurementResult:
    """Compute S, I, Gamma from channel and DFT beams.

    h_freq shape: [num_ue, num_tx_unit, num_freq, num_rx_ant, num_tx_ant]
    tx_beams shape: [num_beams_total, num_tx_ant], each beam's tx_unit is in beam_ids
    rx_beams shape: [num_rx_beams, num_rx_ant]

    v2.4 uses a vectorized implementation. For every UE, it first computes the
    TX-beam effective receive vector h(f)w for all TX beams, then projects the
    result onto all RX beams. The selected RX beam for each service beam is then
    reused to compute all service/interferer Gamma entries without an explicit
    Python B^2 loop.
    """
    num_u = int(h_freq.shape[0])
    num_b = len(beam_ids)
    num_r = int(rx_beams.shape[0])
    num_f = int(h_freq.shape[2])
    num_rx = int(rx_beams.shape[1])

    s = np.zeros((num_u, num_b), dtype=float)
    selected = np.zeros((num_u, num_b), dtype=int)
    i_pow = np.zeros((num_u, num_b, num_b), dtype=float)
    gamma = np.zeros((num_u, num_b, num_b), dtype=float)

    # Group global beams by TX-unit to avoid repeatedly reading the same channel.
    unit_to_beams = {}
    for bi, bid in enumerate(beam_ids):
        unit_to_beams.setdefault(int(bid.tx_unit), []).append(bi)

    rx_conj = np.conjugate(rx_beams).astype(np.complex128)

    for u in range(num_u):
        # hf_all[b, f, nr] = H_u,txunit(b)[f,:,:] @ w_b
        hf_all = np.zeros((num_b, num_f, num_rx), dtype=np.complex128)
        for unit, inds in unit_to_beams.items():
            h = h_freq[u, unit]  # [F, Nr, Nt]
            fmat = tx_beams[inds]  # [K, Nt]
            hf_all[np.asarray(inds, dtype=int), :, :] = np.einsum("frt,kt->kfr", h, fmat, optimize=True)

        # z_all[r, b, f] = q_r^H H w_b
        z_all = np.einsum("rn,bfn->rbf", rx_conj, hf_all, optimize=True)
        pwr_rb = np.mean(np.abs(z_all) ** 2, axis=2)  # [R, B]
        selected[u, :] = np.argmax(pwr_rb, axis=0)
        s[u, :] = float(tx_power_w_per_panel) * pwr_rb[selected[u, :], np.arange(num_b)]

        for m in range(num_b):
            row = float(tx_power_w_per_panel) * pwr_rb[selected[u, m], :]
            i_pow[u, m, :] = row
            i_pow[u, m, m] = 0.0
            gamma[u, m, :] = s[u, m] / np.maximum(row + float(noise_power_w), 1e-30)
            gamma[u, m, m] = s[u, m] / max(float(noise_power_w), 1e-30)

    su_snr_db = lin_to_db(np.diagonal(gamma, axis1=1, axis2=2))
    su_mcs = np.zeros_like(su_snr_db, dtype=int)
    for u in range(num_u):
        for b in range(num_b):
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
