from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .codebook import ArrayConfig


SUB_CONNECTED_ALIASES = {
    "sub_connected",
    "sub-connected",
    "subconnected",
    "panel_polarization_connected",
    "panel_polarization_subarray",
    "panel-pol-subarray",
    "panel_pol_subarray",
    "case1",
    "case_1",
}

FULLY_CONNECTED_ALIASES = {
    "fully_connected",
    "fully-connected",
    "fullyconnected",
    "fully_connected_hybrid",
    "hybrid_fully_connected",
    "case2",
    "case_2",
}


def normalize_connectivity(value: str | None) -> str:
    s = str(value or "panel_polarization_subarray").lower()
    if s in SUB_CONNECTED_ALIASES:
        return "panel_polarization_subarray"
    if s in FULLY_CONNECTED_ALIASES:
        return "fully_connected"
    raise ValueError(
        f"Unsupported rf_architecture.txru_connectivity={value!r}; "
        "use 'panel_polarization_subarray' or 'fully_connected'."
    )


@dataclass(frozen=True)
class TxUnitDescriptor:
    """One independently scheduled analog-beam transmitting unit within one TRP.

    The descriptor is local to a sector/TRP. A network-wide beam ID maps this
    local unit to a global channel tensor index.
    """

    local_tx_unit: int
    beam_scope: str
    array_panel_index: Optional[int] = None
    polarization_index: Optional[int] = None
    txru_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local_tx_unit": int(self.local_tx_unit),
            "beam_scope": self.beam_scope,
            "array_panel_index": None if self.array_panel_index is None else int(self.array_panel_index),
            "polarization_index": None if self.polarization_index is None else int(self.polarization_index),
            "txru_index": None if self.txru_index is None else int(self.txru_index),
        }


@dataclass(frozen=True)
class RFArchitecture:
    connectivity: str
    allow_independent_polarization_beams: bool
    num_txru: int
    num_array_panels: int
    polarization_count: int
    tx_units_per_trp: int
    max_parallel_beams_per_trp: int
    dynamic_beam_assignment: bool
    measurement_panel_index: Optional[int]
    compact_panel_channel: bool
    effective_beam_scope: str
    tx_units: List[TxUnitDescriptor]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "txru_connectivity": self.connectivity,
            "allow_independent_polarization_beams": bool(self.allow_independent_polarization_beams),
            "num_txru": int(self.num_txru),
            "num_array_panels": int(self.num_array_panels),
            "polarization_count": int(self.polarization_count),
            "tx_units_per_trp": int(self.tx_units_per_trp),
            "max_parallel_beams_per_trp": int(self.max_parallel_beams_per_trp),
            "dynamic_beam_assignment": bool(self.dynamic_beam_assignment),
            "measurement_panel_index": (
                None if self.measurement_panel_index is None
                else int(self.measurement_panel_index)
            ),
            "compact_panel_channel": bool(self.compact_panel_channel),
            "effective_beam_scope": self.effective_beam_scope,
            "tx_units": [u.to_dict() for u in self.tx_units],
            "explanation": self.explanation,
        }


def resolve_rf_architecture(cfg: Dict[str, Any], tx_array: ArrayConfig) -> RFArchitecture:
    """Resolve RF architecture, TX units, and MU-order physical limit.

    Two modes are supported:

    1. panel_polarization_subarray: when independent polarization beams are
       enabled, each TXRU controls one panel-polarization subarray. When
       disabled (the default), the TRP exposes a shared full-array codebook;
       both polarizations share weights and selected codewords are dynamically
       assigned to panel resources. The parallel-beam limit is then the number
       of physical panels.
    2. fully_connected: each TXRU can apply independent analog weights over the
       whole TRP array. Every TXRU can form one full-array beam.
    """

    rf_cfg = cfg.get("rf_architecture", {}) or {}
    conn = normalize_connectivity(rf_cfg.get("txru_connectivity", "panel_polarization_subarray"))
    allow_pol = bool(rf_cfg.get("allow_independent_polarization_beams", False))
    p = max(1, int(tx_array.polarization_count))
    panels = max(1, int(tx_array.num_array_panels))
    requested_txru = tx_array.num_txru
    default_txru = panels * p if conn == "panel_polarization_subarray" else max(1, panels * p)
    num_txru = int(rf_cfg.get("num_txru", requested_txru if requested_txru is not None else default_txru))
    if num_txru <= 0:
        raise ValueError("rf_architecture.num_txru must be positive")

    tx_units: List[TxUnitDescriptor] = []
    dynamic_beam_assignment = False
    measurement_panel_index: Optional[int] = None
    compact_panel_channel = False
    if conn == "panel_polarization_subarray":
        if allow_pol:
            # Each physical panel-polarization pair is an independent subarray.
            candidates = []
            for panel_idx in range(panels):
                for pol_idx in range(p):
                    candidates.append((panel_idx, pol_idx))
            if num_txru < len(candidates):
                candidates = candidates[:num_txru]
            # If the user configured more TXRUs than panel*polarization, repeat
            # deterministically. This is a modeling fallback; status JSON shows it.
            while len(candidates) < num_txru:
                candidates.append(candidates[len(candidates) % max(1, panels * p)])
            for i, (panel_idx, pol_idx) in enumerate(candidates):
                tx_units.append(TxUnitDescriptor(
                    local_tx_unit=i,
                    beam_scope="panel_polarization_subarray",
                    array_panel_index=int(panel_idx),
                    polarization_index=int(pol_idx),
                    txru_index=i,
                ))
            effective_scope = "panel_polarization_subarray"
            explanation = (
                "sub-connected panel-polarization architecture: each TXRU controls "
                "one panel-polarization subarray; independent polarization beams are enabled; "
                "therefore max_parallel_beams_per_trp equals num_txru."
            )
        else:
            # Build one shared single-panel codebook on a configurable reference
            # panel for SLS measurement. The panel index is a measurement
            # reference only; selected codewords are assigned to free panel/TXRU
            # resources after scheduling.
            measurement_panel_index = int(
                cfg.get("measurement", {}).get("tx_panel_index", 0)
            )
            if not 0 <= measurement_panel_index < panels:
                raise ValueError(
                    "measurement.tx_panel_index must be in "
                    f"[0, {panels - 1}], got {measurement_panel_index}"
                )
            measurement_cfg = cfg.get("measurement", {})
            compact_panel_channel = bool(
                measurement_cfg.get(
                    "use_panel_channel_views",
                    measurement_cfg.get("compact_tx_panel_channel", True),
                )
            )
            tx_units.append(TxUnitDescriptor(
                local_tx_unit=0,
                beam_scope="per_panel",
                array_panel_index=measurement_panel_index,
                polarization_index=None,
                txru_index=None,
            ))
            effective_scope = "per_panel"
            dynamic_beam_assignment = True
            explanation = (
                "shared single-panel codebook with dynamic beam-to-TXRU assignment: "
                f"panel {measurement_panel_index} is used only as the SLS measurement "
                "reference, both polarizations use the same spatial weights, codewords "
                "are not bound to a TXRU, and simultaneous beams equal physical panels. "
                "The full TRP channel is retained; "
                f"panel-only compute views are {compact_panel_channel}."
            )
    else:
        # Fully-connected hybrid: every TXRU can form a full-array beam.
        for i in range(num_txru):
            tx_units.append(TxUnitDescriptor(
                local_tx_unit=i,
                beam_scope="joint",
                array_panel_index=None,
                polarization_index=None,
                txru_index=i,
            ))
        effective_scope = "joint"
        explanation = (
            "fully-connected hybrid architecture: each TXRU is connected to the full TRP array "
            "and can form one full-array DFT beam; therefore max_parallel_beams_per_trp equals num_txru."
        )

    max_parallel = panels if dynamic_beam_assignment else len(tx_units)
    return RFArchitecture(
        connectivity=conn,
        allow_independent_polarization_beams=allow_pol,
        num_txru=num_txru,
        num_array_panels=panels,
        polarization_count=p,
        tx_units_per_trp=max_parallel,
        max_parallel_beams_per_trp=max_parallel,
        dynamic_beam_assignment=dynamic_beam_assignment,
        measurement_panel_index=measurement_panel_index,
        compact_panel_channel=compact_panel_channel,
        effective_beam_scope=effective_scope,
        tx_units=tx_units,
        explanation=explanation,
    )


def trps_per_sector(cfg: Dict[str, Any]) -> int:
    trp_cfg = cfg.get("trp", {}) or {}
    # Backward compatible: previous versions used num_panels_per_sector as the
    # schedulable unit count. In v2.4, num_trps_per_sector is preferred.
    return int(trp_cfg.get("num_trps_per_sector", trp_cfg.get("num_panels_per_sector", 1)))


def tx_units_per_sector(cfg: Dict[str, Any], rf: RFArchitecture) -> int:
    return max(1, trps_per_sector(cfg)) * int(rf.tx_units_per_trp)


def resolved_max_mu_order(cfg: Dict[str, Any], rf: RFArchitecture) -> int:
    raw = cfg.get("scheduler", {}).get("max_mu_order", "auto")
    topology = cfg.get("topology", {}) or {}
    scheduler = cfg.get("scheduler", {}) or {}
    cluster_mode = str(scheduler.get("cluster_mode") or "").lower()
    domain_mode = str(scheduler.get("domain_mode", "global")).lower()
    sectors_per_site = max(1, int(topology.get("sectors_per_site", 3)))
    num_sites = max(1, int(topology.get("num_sites", 1)))
    if cluster_mode in ("per_cell", "cell", "sector") or (
        not cluster_mode
        and domain_mode in ("per_sector_independent", "per_sector", "sector")
    ):
        trps_in_domain = 1
    elif cluster_mode in ("per_site", "site") or (
        not cluster_mode
        and domain_mode in ("per_site_joint", "per_site", "site")
    ):
        trps_in_domain = sectors_per_site
    elif cluster_mode in ("custom", "manual"):
        groups = scheduler.get("static_clusters", []) or []
        sizes = [
            len(group.get("cell_ids", [])) if isinstance(group, dict)
            else len(group)
            for group in groups
        ]
        trps_in_domain = max(sizes, default=1)
    else:
        trps_in_domain = num_sites * sectors_per_site
    physical_cap = (
        int(rf.max_parallel_beams_per_trp)
        * max(1, trps_per_sector(cfg))
        * trps_in_domain
    )
    if raw is None or str(raw).lower() == "auto":
        return max(1, physical_cap)
    requested = int(raw)
    if bool(cfg.get("scheduler", {}).get("cap_mu_order_by_rf", True)):
        return max(1, min(requested, physical_cap))
    return max(1, requested)
