from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .utils import lin_to_db


@dataclass(frozen=True)
class McsEntry:
    index: int
    q_m: int
    code_rate: float
    spectral_efficiency: float
    sinr_threshold_db: float


def default_mcs_table() -> List[McsEntry]:
    """A compact NR-like MCS surrogate table.

    The spectral-efficiency values follow the familiar monotonic shape of NR MCS
    tables, but the SINR thresholds are deliberately surrogate values for SLS
    ranking, not a standards-accurate BLER table. The first project version uses
    them for CQI/MCS surrogate reporting and for a smooth BLER model.
    """
    ses = [
        0.2344, 0.3770, 0.6016, 0.8770, 1.1758,
        1.4766, 1.6953, 1.9141, 2.1602, 2.4063,
        2.7305, 3.0293, 3.3223, 3.6094, 3.9023,
        4.2129, 4.5234, 4.8164, 5.1152, 5.3320,
        5.5547, 5.8906, 6.2266, 6.5703, 6.9141,
        7.1602, 7.4063, 7.9141, 8.4375,
    ]
    # Calibrated to pick increasingly aggressive MCS values. This table is easy
    # to replace once real Sionna SYS PHYAbstraction tables are wired in.
    thresholds = [
        -6.5, -5.2, -3.9, -2.4, -1.0,
        0.4, 1.5, 2.6, 3.6, 4.7,
        5.9, 7.0, 8.2, 9.4, 10.6,
        11.8, 13.0, 14.1, 15.3, 16.4,
        17.5, 18.7, 19.9, 21.1, 22.4,
        23.6, 24.8, 26.4, 28.0,
    ]
    qms = []
    for se in ses:
        if se <= 2.4:
            qms.append(2)
        elif se <= 5.3:
            qms.append(4)
        else:
            qms.append(6)
    entries = []
    for i, (se, th, qm) in enumerate(zip(ses, thresholds, qms)):
        entries.append(McsEntry(i, qm, min(se / qm, 0.95), se, th))
    return entries


MCS_TABLE = default_mcs_table()


def select_mcs_from_sinr_db(sinr_db: float, margin_db: float = 0.0) -> McsEntry:
    x = float(sinr_db) + float(margin_db)
    chosen = MCS_TABLE[0]
    for e in MCS_TABLE:
        if x >= e.sinr_threshold_db:
            chosen = e
        else:
            break
    return chosen


def select_mcs_from_sinr_lin(sinr_lin: float, margin_db: float = 0.0) -> McsEntry:
    return select_mcs_from_sinr_db(float(lin_to_db(sinr_lin)), margin_db=margin_db)


def bler_from_sinr_db(sinr_db: float, mcs_index: int, slope: float = 1.1) -> float:
    e = MCS_TABLE[int(np.clip(mcs_index, 0, len(MCS_TABLE) - 1))]
    x = slope * (float(sinr_db) - e.sinr_threshold_db)
    # Logistic with threshold close to 50% BLER; link adaptation backs off via
    # table thresholds and optional OLLA. Good enough for first SLS ranking.
    if x > 60:
        return 0.0
    if x < -60:
        return 1.0
    return float(1.0 / (1.0 + np.exp(x)))


def tbs_bits_from_mcs(mcs_index: int,
                      num_prbs: int,
                      num_symbols: int,
                      dmrs_overhead_re_per_prb: int,
                      num_layers: int = 1) -> int:
    e = MCS_TABLE[int(np.clip(mcs_index, 0, len(MCS_TABLE) - 1))]
    re_per_prb = max(0, 12 * int(num_symbols) - int(dmrs_overhead_re_per_prb))
    bits = int(np.floor(int(num_prbs) * re_per_prb * int(num_layers) * e.spectral_efficiency))
    # NR TBS has a granularity; use a lightweight 8-bit rounding for reports.
    return max(0, (bits // 8) * 8)


def rate_mbps_from_mcs(mcs_index: int,
                       num_prbs: int,
                       num_symbols: int,
                       dmrs_overhead_re_per_prb: int,
                       slot_duration_ms: float,
                       num_layers: int = 1) -> float:
    bits = tbs_bits_from_mcs(mcs_index, num_prbs, num_symbols,
                             dmrs_overhead_re_per_prb, num_layers)
    return bits / (float(slot_duration_ms) * 1e-3) / 1e6
