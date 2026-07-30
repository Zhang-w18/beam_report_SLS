from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, log2
from typing import List


@dataclass(frozen=True)
class McsEntry:
    """One row of TS 38.214 Table 5.1.3.1-1 (PDSCH MCS table 1)."""

    index: int
    q_m: int
    code_rate_x1024: int
    code_rate: float
    spectral_efficiency: float


# TS 38.214 Table 5.1.3.1-1. Keep Qm and R explicit: inferring them from a
# spectral-efficiency array was the source of the old, shifted MCS semantics.
_PDSCH_TABLE_1_QM = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
)
_PDSCH_TABLE_1_R_X1024 = (
    120, 157, 193, 251, 308, 379, 449, 526, 602, 679,
    340, 378, 434, 490, 553, 616, 658,
    438, 466, 517, 567, 616, 666, 719, 772, 822, 873, 910, 948,
)


def pdsch_mcs_table_1() -> List[McsEntry]:
    return [
        McsEntry(
            index=index,
            q_m=q_m,
            code_rate_x1024=rate_x1024,
            code_rate=rate_x1024 / 1024.0,
            spectral_efficiency=q_m * rate_x1024 / 1024.0,
        )
        for index, (q_m, rate_x1024) in enumerate(
            zip(_PDSCH_TABLE_1_QM, _PDSCH_TABLE_1_R_X1024)
        )
    ]


MCS_TABLE = pdsch_mcs_table_1()

# TS 38.214 Table 5.1.3.2-1.
_SMALL_TBS_BITS = (
    24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
    136, 144, 152, 160, 168, 176, 184, 192, 208, 224, 240, 256,
    272, 288, 304, 320, 336, 352, 368, 384, 408, 432, 456, 480,
    504, 528, 552, 576, 608, 640, 672, 704, 736, 768, 808, 848,
    888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256,
    1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800,
    1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536,
    2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496,
    3624, 3752, 3824,
)


def _mcs_entry(mcs_index: int) -> McsEntry:
    index = int(mcs_index)
    if not 0 <= index < len(MCS_TABLE):
        raise ValueError(f"PDSCH MCS table 1 index must be in [0, 28], got {index}")
    return MCS_TABLE[index]


def tbs_bits_from_mcs(mcs_index: int,
                      num_prbs: int,
                      num_symbols: int,
                      dmrs_overhead_re_per_prb: int,
                      num_layers: int = 1) -> int:
    """Calculate TBS using TS 38.214 Sec. 5.1.3.2."""

    entry = _mcs_entry(mcs_index)
    num_prbs = int(num_prbs)
    num_symbols = int(num_symbols)
    dmrs_re = int(dmrs_overhead_re_per_prb)
    num_layers = int(num_layers)
    if num_prbs <= 0 or num_symbols <= 0 or num_layers <= 0 or dmrs_re < 0:
        raise ValueError("PDSCH PRBs, symbols and layers must be positive; DMRS RE must be >= 0")

    # N'_RE is capped at 156 RE/PRB by Sec. 5.1.3.2.
    n_re_per_prb = min(156, max(0, 12 * num_symbols - dmrs_re))
    n_info = (
        n_re_per_prb
        * num_prbs
        * entry.q_m
        * entry.code_rate
        * num_layers
    )
    if n_info <= 0:
        return 0

    if n_info <= 3824:
        n = max(3, floor(log2(n_info)) - 6)
        n_info_quantized = max(24, (2 ** n) * floor(n_info / (2 ** n)))
        return next(tbs for tbs in _SMALL_TBS_BITS if tbs >= n_info_quantized)

    n = floor(log2(n_info - 24)) - 5
    quantum = 2 ** n
    # 38.214 round(); all operands are positive, so floor(x+0.5) avoids
    # Python's ties-to-even behavior.
    n_info_quantized = max(
        3840,
        quantum * floor((n_info - 24) / quantum + 0.5),
    )
    if entry.code_rate <= 0.25:
        num_code_blocks = ceil((n_info_quantized + 24) / 3816)
    elif n_info_quantized > 8424:
        num_code_blocks = ceil((n_info_quantized + 24) / 8424)
    else:
        num_code_blocks = 1
    return int(
        8
        * num_code_blocks
        * ceil((n_info_quantized + 24) / (8 * num_code_blocks))
        - 24
    )


def rate_mbps_from_mcs(mcs_index: int,
                       num_prbs: int,
                       num_symbols: int,
                       dmrs_overhead_re_per_prb: int,
                       slot_duration_ms: float,
                       num_layers: int = 1) -> float:
    bits = tbs_bits_from_mcs(
        mcs_index,
        num_prbs,
        num_symbols,
        dmrs_overhead_re_per_prb,
        num_layers,
    )
    return bits / (float(slot_duration_ms) * 1e-3) / 1e6
