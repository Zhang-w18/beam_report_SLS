from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .mcs import (MCS_TABLE, bler_from_sinr_db, rate_mbps_from_mcs,
                  select_mcs_from_sinr_db, select_mcs_from_sinr_lin,
                  tbs_bits_from_mcs)
from .utils import lin_to_db


class LinkAdaptationBackendError(RuntimeError):
    pass


@dataclass
class LinkAdaptationStatus:
    backend: str
    status: str
    target_bler: float
    mcs_table_index: int = 1
    mcs_category: int = 1


class BaseLinkAdapter:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        la = cfg.get("link_abstraction", {})
        self.target_bler = float(cfg.get("system", {}).get("target_bler", la.get("target_bler", 0.1)))
        self.mcs_table_index = int(la.get("mcs_table_index", 1))
        self.mcs_category = int(la.get("mcs_category", 1))
        self.status = LinkAdaptationStatus("unknown", "not initialized", self.target_bler,
                                           self.mcs_table_index, self.mcs_category)

    def select_mcs_from_sinr_db(self, sinr_db: float, num_allocated_re: int | None = None) -> int:
        return int(select_mcs_from_sinr_db(sinr_db).index)

    def select_mcs_from_sinr_lin(self, sinr_lin: float, num_allocated_re: int | None = None) -> int:
        return int(select_mcs_from_sinr_lin(sinr_lin).index)

    def tbler_from_sinr_db(self, sinr_db: float, mcs_index: int, num_allocated_re: int | None = None) -> float:
        slope = float(self.cfg.get("link_abstraction", {}).get("bler_curve_slope", 1.1))
        return float(bler_from_sinr_db(sinr_db, mcs_index, slope=slope))

    def tbler_from_sinr_lin(self, sinr_lin: float, mcs_index: int, num_allocated_re: int | None = None) -> float:
        return self.tbler_from_sinr_db(float(lin_to_db(sinr_lin)), mcs_index, num_allocated_re)

    def tbs_bits(self, mcs_index: int) -> int:
        pdsch = self.cfg.get("pdsch", {})
        return tbs_bits_from_mcs(mcs_index,
                                 num_prbs=int(pdsch.get("num_prbs", 132)),
                                 num_symbols=int(pdsch.get("num_symbols", 12)),
                                 dmrs_overhead_re_per_prb=int(pdsch.get("dmrs_overhead_re_per_prb", 18)),
                                 num_layers=int(pdsch.get("num_layers_per_ue", 1)))

    def rate_mbps(self, mcs_index: int) -> float:
        pdsch = self.cfg.get("pdsch", {})
        return rate_mbps_from_mcs(mcs_index,
                                  num_prbs=int(pdsch.get("num_prbs", 132)),
                                  num_symbols=int(pdsch.get("num_symbols", 12)),
                                  dmrs_overhead_re_per_prb=int(pdsch.get("dmrs_overhead_re_per_prb", 18)),
                                  slot_duration_ms=float(pdsch.get("slot_duration_ms", 0.125)),
                                  num_layers=int(pdsch.get("num_layers_per_ue", 1)))

    def allocated_re(self) -> int:
        pdsch = self.cfg.get("pdsch", {})
        return max(0, int(pdsch.get("num_prbs", 132)) * (12 * int(pdsch.get("num_symbols", 12)) - int(pdsch.get("dmrs_overhead_re_per_prb", 18))) * int(pdsch.get("num_layers_per_ue", 1)))


class PrecomputedTableFallbackAdapter(BaseLinkAdapter):
    """Fast local fallback: precompute BLER curves on a grid and use interpolation.

    This preserves the v2 call path (ILLA style: choose highest MCS whose table
    TBLER <= target) even when Sionna SYS is not importable on the current host.
    """

    def __init__(self, cfg: Dict, reason: str = ""):
        super().__init__(cfg)
        la = cfg.get("link_abstraction", {})
        self.snr_grid_db = np.arange(float(la.get("fallback_snr_min_db", -8.0)),
                                     float(la.get("fallback_snr_max_db", 35.01)),
                                     float(la.get("fallback_snr_step_db", 0.05)))
        slope = float(la.get("bler_curve_slope", 1.1))
        table = []
        for m in range(len(MCS_TABLE)):
            table.append([bler_from_sinr_db(x, m, slope=slope) for x in self.snr_grid_db])
        self.tbler_table = np.asarray(table, dtype=float)
        status = "OK" if not reason else f"FALLBACK: {reason}"
        self.status = LinkAdaptationStatus("fallback_precomputed_table", status, self.target_bler,
                                           self.mcs_table_index, self.mcs_category)

    def _interp_tbler(self, sinr_db: float, mcs_index: int) -> float:
        m = int(np.clip(mcs_index, 0, len(MCS_TABLE) - 1))
        return float(np.interp(float(sinr_db), self.snr_grid_db, self.tbler_table[m], left=1.0, right=0.0))

    def select_mcs_from_sinr_db(self, sinr_db: float, num_allocated_re: int | None = None) -> int:
        # Sionna ILLA principle: highest MCS whose table TBLER <= target.
        chosen = 0
        found = False
        for m in range(len(MCS_TABLE)):
            if self._interp_tbler(sinr_db, m) <= self.target_bler:
                chosen = m
                found = True
        if not found:
            return 0
        return int(chosen)

    def tbler_from_sinr_db(self, sinr_db: float, mcs_index: int, num_allocated_re: int | None = None) -> float:
        return self._interp_tbler(sinr_db, mcs_index)


class SionnaSYSAdapter(BaseLinkAdapter):
    """Adapter for Sionna SYS PHYAbstraction and InnerLoopLinkAdaptation.

    Important for Sionna 1.0.2 in the user's environment:
    -------------------------------------------------------
    Although PyTorch must be importable by parts of Sionna 1.x, the SYS
    link-adaptation call path used here relies on TensorFlow tensors internally
    (e.g., ``sionna.sys.utils.is_scheduled_in_slot`` calls ``tf.cast``).  Passing
    torch tensors therefore leads to errors such as

        TypeError: Cannot convert the argument `type_value`: torch.int32 to a TensorFlow DType

    This adapter uses TensorFlow tensors for all Sionna SYS calls and keeps
    PyTorch only as an environment dependency/probe item.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        try:
            import tensorflow as tf  # type: ignore
            from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation  # type: ignore
        except Exception as e:
            raise LinkAdaptationBackendError(f"Sionna SYS unavailable: {type(e).__name__}: {e}") from e
        self.tf = tf
        la = cfg.get("link_abstraction", {})
        device = cfg.get("sionna", {}).get("device", None)
        precision = cfg.get("sionna", {}).get("precision", None)
        phy_kwargs = {}
        if la.get("bler_table_files", "default") is not None:
            phy_kwargs["load_bler_tables_from"] = la.get("bler_table_files", "default")
        if precision is not None:
            phy_kwargs["precision"] = precision
        # Device is intentionally not forwarded here for TensorFlow-backed Sionna
        # 1.0.2 because TF device placement is controlled globally by
        # CUDA_VISIBLE_DEVICES / tf.config, while several Sionna SYS constructors
        # do not accept a PyTorch-style device argument.
        self.phy_abs = PHYAbstraction(**phy_kwargs)
        illa_kwargs = {"phy_abstraction": self.phy_abs, "bler_target": self.target_bler}
        if precision is not None:
            illa_kwargs["precision"] = precision
        self.illa = InnerLoopLinkAdaptation(**illa_kwargs)
        self.device = device
        self.status = LinkAdaptationStatus("sionna_sys_phy_abstraction_tf", "OK",
                                           self.target_bler, self.mcs_table_index, self.mcs_category)

    def _tensor(self, x, dtype=None):
        if dtype is None:
            return self.tf.convert_to_tensor(x)
        return self.tf.convert_to_tensor(x, dtype=dtype)

    def _to_float(self, x) -> float:
        if hasattr(x, "detach"):
            arr = x.detach().cpu().numpy()
        elif hasattr(x, "numpy"):
            arr = x.numpy()
        else:
            arr = np.asarray(x)
        return float(np.asarray(arr).reshape(-1)[0])

    def _to_int(self, x) -> int:
        return int(round(self._to_float(x)))

    def select_mcs_from_sinr_lin(self, sinr_lin: float, num_allocated_re: int | None = None) -> int:
        nre = int(num_allocated_re or self.allocated_re())
        try:
            out = self.illa(sinr_eff=self._tensor([float(sinr_lin)], self.tf.float32),
                            num_allocated_re=self._tensor([nre], self.tf.int32),
                            mcs_table_index=int(self.mcs_table_index),
                            mcs_category=int(self.mcs_category),
                            return_lowest_available_mcs=True)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                selected = self._to_int(out[0])
                lowest_available = self._to_int(out[1])
                # Sionna SYS returns +/-inf for unavailable BLER table entries.
                # Its ILLA comparison can then treat -inf TBLER as valid and
                # select an MCS below the first finite table row. Keep the SYS
                # choice, but never below the lowest valid table-backed MCS.
                return max(selected, lowest_available)
            return self._to_int(out)
        except Exception as e:
            raise LinkAdaptationBackendError(f"Sionna ILLA call failed: {type(e).__name__}: {e}") from e

    def select_mcs_from_sinr_db(self, sinr_db: float, num_allocated_re: int | None = None) -> int:
        return self.select_mcs_from_sinr_lin(float(10.0 ** (float(sinr_db) / 10.0)), num_allocated_re)

    def tbler_from_sinr_lin(self, sinr_lin: float, mcs_index: int, num_allocated_re: int | None = None) -> float:
        nre = int(num_allocated_re or self.allocated_re())
        try:
            res = self.phy_abs(mcs_index=self._tensor([int(mcs_index)], self.tf.int32),
                               sinr_eff=self._tensor([float(sinr_lin)], self.tf.float32),
                               num_allocated_re=self._tensor([nre], self.tf.int32),
                               mcs_table_index=int(self.mcs_table_index),
                               mcs_category=int(self.mcs_category),
                               check_mcs_index_validity=False)
            # Documented output tuple: decoded bits, HARQ, sinr_eff, tbler, bler.
            if isinstance(res, (tuple, list)) and len(res) >= 4:
                tbler = self._to_float(res[3])
                if (not np.isfinite(tbler)) or tbler < 0.0:
                    return 1.0
                return float(np.clip(tbler, 0.0, 1.0))
            if isinstance(res, dict) and "tbler" in res:
                tbler = self._to_float(res["tbler"])
                if (not np.isfinite(tbler)) or tbler < 0.0:
                    return 1.0
                return float(np.clip(tbler, 0.0, 1.0))
            raise LinkAdaptationBackendError(f"Unexpected PHYAbstraction output type={type(res)}")
        except Exception as e:
            if isinstance(e, LinkAdaptationBackendError):
                raise
            raise LinkAdaptationBackendError(f"Sionna PHYAbstraction call failed: {type(e).__name__}: {e}") from e

    def tbler_from_sinr_db(self, sinr_db: float, mcs_index: int, num_allocated_re: int | None = None) -> float:
        return self.tbler_from_sinr_lin(float(10.0 ** (float(sinr_db) / 10.0)), mcs_index, num_allocated_re)

def make_link_adapter(cfg: Dict) -> BaseLinkAdapter:
    mode = str(cfg.get("link_abstraction", {}).get("mode", "sionna_sys_precomputed_bler")).lower()
    if mode in ("sionna_sys_precomputed_bler", "sionna_sys", "sionna_illa"):
        try:
            return SionnaSYSAdapter(cfg)
        except Exception as e:
            if bool(cfg.get("sionna", {}).get("fallback_to_numpy_if_unavailable", True)):
                return PrecomputedTableFallbackAdapter(cfg, reason=f"{type(e).__name__}: {e}")
            raise
    if mode in ("fallback_precomputed_table", "precomputed_table"):
        return PrecomputedTableFallbackAdapter(cfg)
    if mode in ("mcs_surrogate", "logistic_surrogate"):
        adapter = BaseLinkAdapter(cfg)
        adapter.status = LinkAdaptationStatus("v1_logistic_surrogate", "OK", adapter.target_bler)
        return adapter
    raise ValueError(f"Unsupported link_abstraction.mode={mode}")
