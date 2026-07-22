from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
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
        self._call_counters = {
            "illa_invocations": 0,
            "illa_items": 0,
            "phy_abstraction_invocations": 0,
            "phy_abstraction_items": 0,
        }

    def snapshot_counters(self) -> Dict[str, int]:
        return {k: int(v) for k, v in self._call_counters.items()}

    def schedule_map_from_sinr_db(self, sinr_db) -> Tuple[np.ndarray, np.ndarray]:
        """Return ILLA-selected MCS and outage for an arbitrary SINR array.

        Backends may override this with a true batched implementation.  The
        default loop keeps custom/test adapters compatible.
        """
        x = np.asarray(sinr_db, dtype=float)
        flat = x.reshape(-1)
        mcs = np.empty(flat.size, dtype=np.int32)
        outage = np.empty(flat.size, dtype=bool)
        for i, value in enumerate(flat):
            selected = int(self.select_mcs_from_sinr_db(float(value)))
            mcs[i] = selected
            outage[i] = bool(self.is_outage_from_sinr_db(float(value), selected))
        return mcs.reshape(x.shape), outage.reshape(x.shape)

    def select_mcs_from_sinr_db(self, sinr_db: float, num_allocated_re: int | None = None) -> int:
        return int(select_mcs_from_sinr_db(sinr_db).index)

    def select_mcs_from_sinr_lin(self, sinr_lin: float, num_allocated_re: int | None = None) -> int:
        return int(select_mcs_from_sinr_lin(sinr_lin).index)

    def tbler_from_sinr_db(self, sinr_db: float, mcs_index: int, num_allocated_re: int | None = None) -> float:
        slope = float(self.cfg.get("link_abstraction", {}).get("bler_curve_slope", 1.1))
        return float(bler_from_sinr_db(sinr_db, mcs_index, slope=slope))

    def tbler_from_sinr_lin(self, sinr_lin: float, mcs_index: int, num_allocated_re: int | None = None) -> float:
        return self.tbler_from_sinr_db(float(lin_to_db(sinr_lin)), mcs_index, num_allocated_re)

    def is_outage_from_sinr_db(self,
                               sinr_db: float,
                               mcs_index: int | None = None,
                               num_allocated_re: int | None = None) -> bool:
        """Return whether even the ILLA-selected MCS misses the BLER target."""
        mcs = self.select_mcs_from_sinr_db(sinr_db, num_allocated_re) if mcs_index is None else int(mcs_index)
        tbler = float(self.tbler_from_sinr_db(sinr_db, mcs, num_allocated_re))
        return (not np.isfinite(tbler)) or tbler > self.target_bler

    def is_outage_from_sinr_lin(self,
                                sinr_lin: float,
                                mcs_index: int | None = None,
                                num_allocated_re: int | None = None) -> bool:
        """Return whether the selected MCS misses the target for linear SINR."""
        mcs = self.select_mcs_from_sinr_lin(sinr_lin, num_allocated_re) if mcs_index is None else int(mcs_index)
        tbler = float(self.tbler_from_sinr_lin(sinr_lin, mcs, num_allocated_re))
        return (not np.isfinite(tbler)) or tbler > self.target_bler

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

    def schedule_map_from_sinr_db(self, sinr_db) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(sinr_db, dtype=float)
        chosen = np.zeros(x.shape, dtype=np.int32)
        found = np.zeros(x.shape, dtype=bool)
        for m in range(len(MCS_TABLE)):
            values = np.interp(x, self.snr_grid_db, self.tbler_table[m], left=1.0, right=0.0)
            valid = values <= self.target_bler
            chosen[valid] = int(m)
            found |= valid
        selected_tbler = np.empty(x.shape, dtype=float)
        for m in range(len(MCS_TABLE)):
            mask = chosen == m
            if np.any(mask):
                selected_tbler[mask] = np.interp(
                    x[mask], self.snr_grid_db, self.tbler_table[m], left=1.0, right=0.0,
                )
        return chosen, (~found) | (~np.isfinite(selected_tbler)) | (selected_tbler > self.target_bler)


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

    def _to_numpy(self, x) -> np.ndarray:
        if hasattr(x, "detach"):
            return np.asarray(x.detach().cpu().numpy())
        if hasattr(x, "numpy"):
            return np.asarray(x.numpy())
        return np.asarray(x)

    def _to_int(self, x) -> int:
        return int(round(self._to_float(x)))

    def select_mcs_from_sinr_lin(self, sinr_lin: float, num_allocated_re: int | None = None) -> int:
        nre = int(num_allocated_re or self.allocated_re())
        try:
            self._call_counters["illa_invocations"] += 1
            self._call_counters["illa_items"] += 1
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
            self._call_counters["phy_abstraction_invocations"] += 1
            self._call_counters["phy_abstraction_items"] += 1
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

    def schedule_map_from_sinr_db(self, sinr_db) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(sinr_db, dtype=float)
        flat = x.reshape(-1)
        nre = int(self.allocated_re())
        sinr_lin = np.power(10.0, flat / 10.0).astype(np.float32)
        nre_values = np.full(flat.size, nre, dtype=np.int32)
        try:
            self._call_counters["illa_invocations"] += 1
            self._call_counters["illa_items"] += int(flat.size)
            out = self.illa(
                sinr_eff=self._tensor(sinr_lin, self.tf.float32),
                num_allocated_re=self._tensor(nre_values, self.tf.int32),
                mcs_table_index=int(self.mcs_table_index),
                mcs_category=int(self.mcs_category),
                return_lowest_available_mcs=True,
            )
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                selected = self._to_numpy(out[0]).reshape(-1).astype(np.int32)
                lowest = self._to_numpy(out[1]).reshape(-1).astype(np.int32)
                if lowest.size == 1:
                    lowest = np.full(selected.size, int(lowest[0]), dtype=np.int32)
                mcs = np.maximum(selected, lowest)
            else:
                mcs = self._to_numpy(out).reshape(-1).astype(np.int32)

            self._call_counters["phy_abstraction_invocations"] += 1
            self._call_counters["phy_abstraction_items"] += int(flat.size)
            res = self.phy_abs(
                mcs_index=self._tensor(mcs, self.tf.int32),
                sinr_eff=self._tensor(sinr_lin, self.tf.float32),
                num_allocated_re=self._tensor(nre_values, self.tf.int32),
                mcs_table_index=int(self.mcs_table_index),
                mcs_category=int(self.mcs_category),
                check_mcs_index_validity=False,
            )
            if isinstance(res, (tuple, list)) and len(res) >= 4:
                tbler = self._to_numpy(res[3]).reshape(-1).astype(float)
            elif isinstance(res, dict) and "tbler" in res:
                tbler = self._to_numpy(res["tbler"]).reshape(-1).astype(float)
            else:
                raise LinkAdaptationBackendError(f"Unexpected PHYAbstraction output type={type(res)}")
            outage = (~np.isfinite(tbler)) | (tbler < 0.0) | (tbler > self.target_bler)
            return mcs.reshape(x.shape), outage.reshape(x.shape)
        except Exception as e:
            if isinstance(e, LinkAdaptationBackendError):
                raise
            raise LinkAdaptationBackendError(f"Batched Sionna scheduler mapping failed: {type(e).__name__}: {e}") from e


class SchedulerLinkLookup:
    """Fast scheduler-only SINR -> (MCS, outage, rate) decision map.

    A dense scan discovers all decision transitions. Each transition is refined
    against the original backend, then normal scheduling uses only NumPy
    ``searchsorted``. Values very near a transition and values outside the
    configured scan range fall back to the original backend.
    """

    def __init__(self, backing: BaseLinkAdapter, cfg: Dict):
        self.backing = backing
        self.cfg = cfg
        lookup_cfg = cfg.get("scheduler", {}).get("link_lookup", {}) or {}
        self.min_db = float(lookup_cfg.get("sinr_min_db", -40.0))
        self.max_db = float(lookup_cfg.get("sinr_max_db", 80.0))
        self.scan_step_db = float(lookup_cfg.get("scan_step_db", 0.01))
        self.refine_tolerance_db = float(lookup_cfg.get("refine_tolerance_db", 1e-6))
        self.boundary_guard_db = float(lookup_cfg.get("boundary_guard_db", 1e-5))
        self.validation_points = int(lookup_cfg.get("validation_points", 257))
        if self.max_db <= self.min_db or self.scan_step_db <= 0.0:
            raise ValueError("scheduler.link_lookup requires max>min and scan_step_db>0")
        self._lookup_counters = {
            "lookup_queries": 0,
            "lookup_boundary_fallback_items": 0,
            "lookup_out_of_range_items": 0,
        }
        started = perf_counter()
        self._build()
        self.build_elapsed_s = float(perf_counter() - started)
        self.status = {
            "enabled": True,
            "sinr_min_db": self.min_db,
            "sinr_max_db": self.max_db,
            "scan_step_db": self.scan_step_db,
            "refine_tolerance_db": self.refine_tolerance_db,
            "boundary_guard_db": self.boundary_guard_db,
            "num_decision_regions": int(len(self.region_mcs)),
            "num_boundaries": int(len(self.boundaries_db)),
            "validation_points": int(self.validated_point_count),
            "build_elapsed_s": self.build_elapsed_s,
        }

    @property
    def target_bler(self) -> float:
        return float(self.backing.target_bler)

    def rate_mbps(self, mcs_index: int) -> float:
        return float(self.backing.rate_mbps(int(mcs_index)))

    def tbs_bits(self, mcs_index: int) -> int:
        return int(self.backing.tbs_bits(int(mcs_index)))

    def allocated_re(self) -> int:
        return int(self.backing.allocated_re())

    def _build(self) -> None:
        count = int(np.ceil((self.max_db - self.min_db) / self.scan_step_db))
        grid = np.linspace(self.min_db, self.max_db, count + 1, dtype=float)
        mcs, outage = self.backing.schedule_map_from_sinr_db(grid)
        mcs = np.asarray(mcs, dtype=np.int32).reshape(-1)
        outage = np.asarray(outage, dtype=bool).reshape(-1)
        change = np.flatnonzero((mcs[1:] != mcs[:-1]) | (outage[1:] != outage[:-1])) + 1

        region_indices = [0, *[int(i) for i in change]]
        lo = grid[change - 1].astype(float, copy=True)
        hi = grid[change].astype(float, copy=True)
        left_mcs = mcs[change - 1]
        left_outage = outage[change - 1]
        while lo.size and np.any((hi - lo) > self.refine_tolerance_db):
            active = (hi - lo) > self.refine_tolerance_db
            mid = 0.5 * (lo[active] + hi[active])
            mid_mcs, mid_outage = self.backing.schedule_map_from_sinr_db(mid)
            same_as_left = (
                (np.asarray(mid_mcs).reshape(-1) == left_mcs[active])
                & (np.asarray(mid_outage).reshape(-1) == left_outage[active])
            )
            active_indices = np.flatnonzero(active)
            lo[active_indices[same_as_left]] = mid[same_as_left]
            hi[active_indices[~same_as_left]] = mid[~same_as_left]

        self.boundaries_db = 0.5 * (lo + hi)
        self.region_mcs = np.asarray([mcs[i] for i in region_indices], dtype=np.int32)
        self.region_outage = np.asarray([outage[i] for i in region_indices], dtype=bool)
        self.rate_by_mcs = np.asarray(
            [self.backing.rate_mbps(m) for m in range(len(MCS_TABLE))], dtype=float,
        )
        validation = np.linspace(
            self.min_db, self.max_db, max(2, self.validation_points), dtype=float,
        )
        if self.boundaries_db.size:
            distance = np.min(np.abs(validation[:, None] - self.boundaries_db[None, :]), axis=1)
            validation = validation[distance > max(self.boundary_guard_db, self.refine_tolerance_db) * 2.0]
        expected_mcs, expected_outage = self.backing.schedule_map_from_sinr_db(validation)
        region = np.searchsorted(self.boundaries_db, validation, side="right")
        actual_mcs = self.region_mcs[region]
        actual_outage = self.region_outage[region]
        if (not np.array_equal(actual_mcs, np.asarray(expected_mcs).reshape(-1))
                or not np.array_equal(actual_outage, np.asarray(expected_outage).reshape(-1))):
            raise LinkAdaptationBackendError(
                "scheduler.link_lookup validation failed; reduce scan_step_db or disable the lookup"
            )
        self.validated_point_count = int(validation.size)

    def snapshot_counters(self) -> Dict[str, int]:
        out = dict(self._lookup_counters)
        if hasattr(self.backing, "snapshot_counters"):
            out.update(self.backing.snapshot_counters())
        return {k: int(v) for k, v in out.items()}

    def map_sinr_db(self, sinr_db) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(sinr_db, dtype=float)
        flat = x.reshape(-1)
        self._lookup_counters["lookup_queries"] += int(flat.size)
        region = np.searchsorted(self.boundaries_db, flat, side="right")
        mcs = self.region_mcs[region].copy()
        outage = self.region_outage[region].copy()

        fallback = (flat < self.min_db) | (flat > self.max_db) | (~np.isfinite(flat))
        outside = fallback.copy()
        if self.boundaries_db.size and self.boundary_guard_db > 0.0:
            pos = np.searchsorted(self.boundaries_db, flat)
            distance = np.full(flat.size, np.inf, dtype=float)
            left = pos > 0
            right = pos < self.boundaries_db.size
            distance[left] = np.minimum(distance[left], np.abs(flat[left] - self.boundaries_db[pos[left] - 1]))
            distance[right] = np.minimum(distance[right], np.abs(flat[right] - self.boundaries_db[pos[right]]))
            near = distance <= self.boundary_guard_db
            fallback |= near
            self._lookup_counters["lookup_boundary_fallback_items"] += int(np.count_nonzero(near))
        self._lookup_counters["lookup_out_of_range_items"] += int(np.count_nonzero(outside))
        if np.any(fallback):
            fb_mcs, fb_outage = self.backing.schedule_map_from_sinr_db(flat[fallback])
            mcs[fallback] = np.asarray(fb_mcs, dtype=np.int32).reshape(-1)
            outage[fallback] = np.asarray(fb_outage, dtype=bool).reshape(-1)
        rates = self.rate_by_mcs[np.clip(mcs, 0, len(self.rate_by_mcs) - 1)]
        rates = np.where(outage, 0.0, rates)
        return mcs.reshape(x.shape), outage.reshape(x.shape), rates.reshape(x.shape)

    def map_sinr_lin(self, sinr_lin) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(sinr_lin, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10.0 * np.log10(values)
        return self.map_sinr_db(db)

    def select_mcs_from_sinr_db(self, sinr_db: float, num_allocated_re: int | None = None) -> int:
        mcs, _, _ = self.map_sinr_db(np.asarray([sinr_db], dtype=float))
        return int(mcs[0])

    def select_mcs_from_sinr_lin(self, sinr_lin: float, num_allocated_re: int | None = None) -> int:
        mcs, _, _ = self.map_sinr_lin(np.asarray([sinr_lin], dtype=float))
        return int(mcs[0])

    def is_outage_from_sinr_db(self, sinr_db: float, mcs_index: int | None = None,
                               num_allocated_re: int | None = None) -> bool:
        mcs, outage, _ = self.map_sinr_db(np.asarray([sinr_db], dtype=float))
        if mcs_index is not None and int(mcs_index) != int(mcs[0]):
            return bool(self.backing.is_outage_from_sinr_db(sinr_db, mcs_index, num_allocated_re))
        return bool(outage[0])

    def is_outage_from_sinr_lin(self, sinr_lin: float, mcs_index: int | None = None,
                                num_allocated_re: int | None = None) -> bool:
        mcs, outage, _ = self.map_sinr_lin(np.asarray([sinr_lin], dtype=float))
        if mcs_index is not None and int(mcs_index) != int(mcs[0]):
            return bool(self.backing.is_outage_from_sinr_lin(sinr_lin, mcs_index, num_allocated_re))
        return bool(outage[0])


def make_scheduler_link_adapter(link_adapter: BaseLinkAdapter, cfg: Dict):
    lookup_cfg = cfg.get("scheduler", {}).get("link_lookup", {}) or {}
    if not bool(lookup_cfg.get("enabled", True)):
        return link_adapter
    return SchedulerLinkLookup(link_adapter, cfg)

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
