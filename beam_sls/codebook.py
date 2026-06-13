from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


PANEL_INDEPENDENT_SCOPES = {"per_panel", "panel", "panel_independent", "independent_panel", "independent_panels"}
PANEL_POLARIZATION_SCOPES = {"panel_polarization_subarray", "panel_polarization", "per_panel_polarization", "per_txru_subarray"}
JOINT_SCOPES = {"full_array", "joint", "joint_array", "all_panels_joint", "joint_panels"}


def normalize_beam_scope(scope: str | None) -> str:
    s = str(scope or "joint").lower()
    if s in PANEL_INDEPENDENT_SCOPES:
        return "per_panel"
    if s in PANEL_POLARIZATION_SCOPES:
        return "panel_polarization_subarray"
    if s in JOINT_SCOPES:
        return "joint"
    raise ValueError(f"Unsupported beam_scope={scope!r}; use 'joint'/'full_array', 'per_panel', or 'panel_polarization_subarray'.")


@dataclass(frozen=True)
class ArrayConfig:
    """Configurable planar array description.

    The project uses the 3GPP-style notation

        (M, N, P, Mg, Ng; Mp, Np)

    with horizontal DFT dimension

        H = N * Ng * Np

    and vertical DFT dimension

        V = M * Mg * Mp.

    Therefore the full spatial DFT codebook size is H*V, i.e.

        N * Ng * Np * M * Mg * Mp.

    The polarization count P expands the vector length but does not multiply the
    number of DFT beam directions. For panel-independent transmission, each
    physical panel uses its own local DFT codebook with dimensions N by M and the
    resulting local vector is zero-padded into the full-array AE vector.
    """

    num_h: int
    num_v: int
    d_h_lambda: float = 0.5
    d_v_lambda: float = 0.5
    polarization_count: int = 1
    num_txru: int | None = None
    num_beams_h: int | None = None
    num_beams_v: int | None = None
    beam_scope: str = "joint"
    sampling_mode: str = "uniform"
    vertical_beam_mode: str = "scan"
    fixed_v_index: int | None = None
    model: str = "legacy_upa"
    M: int | None = None
    N: int | None = None
    P: int | None = None
    Mg: int | None = None
    Ng: int | None = None
    Mp: int | None = None
    Np: int | None = None

    @property
    def normalized_beam_scope(self) -> str:
        return normalize_beam_scope(self.beam_scope)

    @property
    def num_spatial(self) -> int:
        return int(self.num_h) * int(self.num_v)

    @property
    def num_ant(self) -> int:
        return self.num_spatial * int(self.polarization_count)

    @property
    def expected_ae(self) -> int:
        if all(v is not None for v in (self.M, self.N, self.P, self.Mg, self.Ng, self.Mp, self.Np)):
            return int(self.M) * int(self.N) * int(self.P) * int(self.Mg) * int(self.Ng) * int(self.Mp) * int(self.Np)
        return self.num_ant

    @property
    def panel_num_h(self) -> int:
        return int(self.N) if self.N is not None else int(self.num_h)

    @property
    def panel_num_v(self) -> int:
        return int(self.M) if self.M is not None else int(self.num_v)

    @property
    def panel_grid_h(self) -> int:
        return int((self.Ng or 1) * (self.Np or 1)) if self.model == "tr38901_panel" else 1

    @property
    def panel_grid_v(self) -> int:
        return int((self.Mg or 1) * (self.Mp or 1)) if self.model == "tr38901_panel" else 1

    @property
    def num_array_panels(self) -> int:
        return int(self.panel_grid_h * self.panel_grid_v)

    @property
    def codebook_num_h(self) -> int:
        return self.panel_num_h if self.normalized_beam_scope in ("per_panel", "panel_polarization_subarray") else int(self.num_h)

    @property
    def codebook_num_v(self) -> int:
        return self.panel_num_v if self.normalized_beam_scope in ("per_panel", "panel_polarization_subarray") else int(self.num_v)

    @property
    def full_codebook_size(self) -> int:
        return int(self.num_h) * int(self.num_v)

    @property
    def per_panel_codebook_size(self) -> int:
        return int(self.panel_num_h) * int(self.panel_num_v)

    @property
    def configured_beams_per_codebook(self) -> int | None:
        if self.num_beams_h is None:
            return None
        if str(self.vertical_beam_mode).lower() == "fixed" or self.fixed_v_index is not None:
            return int(self.num_beams_h)
        if self.num_beams_v is None:
            return None
        return int(self.num_beams_h) * int(self.num_beams_v)

    def to_dict(self) -> Dict[str, int | float | str | None]:
        return {
            "model": self.model,
            "num_h": int(self.num_h),
            "num_v": int(self.num_v),
            "polarization_count": int(self.polarization_count),
            "num_spatial": int(self.num_spatial),
            "num_ant": int(self.num_ant),
            "expected_ae": int(self.expected_ae),
            "num_txru_or_rxru": None if self.num_txru is None else int(self.num_txru),
            "num_txru": None if self.num_txru is None else int(self.num_txru),
            "num_beams_h": None if self.num_beams_h is None else int(self.num_beams_h),
            "num_beams_v": None if self.num_beams_v is None else int(self.num_beams_v),
            "num_beams_configured_per_codebook": self.configured_beams_per_codebook,
            "beam_scope": self.normalized_beam_scope,
            "sampling_mode": self.sampling_mode,
            "vertical_beam_mode": self.vertical_beam_mode,
            "fixed_v_index": self.fixed_v_index,
            "d_h_lambda": float(self.d_h_lambda),
            "d_v_lambda": float(self.d_v_lambda),
            "M": self.M,
            "N": self.N,
            "P": self.P,
            "Mg": self.Mg,
            "Ng": self.Ng,
            "Mp": self.Mp,
            "Np": self.Np,
            "panel_num_h": int(self.panel_num_h),
            "panel_num_v": int(self.panel_num_v),
            "panel_grid_h": int(self.panel_grid_h),
            "panel_grid_v": int(self.panel_grid_v),
            "num_array_panels": int(self.num_array_panels),
            "full_codebook_size": int(self.full_codebook_size),
            "per_panel_codebook_size": int(self.per_panel_codebook_size),
        }

    @classmethod
    def from_dict(cls, cfg: Dict) -> "ArrayConfig":
        model = str(cfg.get("model", cfg.get("antenna_model", "legacy_upa"))).lower()
        vertical_cfg = cfg.get("vertical_beam", {}) or {}
        fixed_v = cfg.get("fixed_v_index", vertical_cfg.get("fixed_v_index", None))
        fixed_v_index = None if fixed_v is None else int(fixed_v)
        vertical_mode = str(cfg.get("vertical_beam_mode", vertical_cfg.get("mode", "scan"))).lower()
        sampling_mode = str(cfg.get("sampling_mode", cfg.get("codebook_sampling", "uniform"))).lower()
        beam_scope = normalize_beam_scope(cfg.get("beam_scope", "joint"))
        if model in ("tr38901_panel", "3gpp_panel", "trp_3gpp") or any(k in cfg for k in ("M", "N", "P", "Mg", "Ng", "Mp", "Np")):
            M = int(cfg.get("M", 1))
            N = int(cfg.get("N", 1))
            P = int(cfg.get("P", cfg.get("polarization_count", 1)))
            Mg = int(cfg.get("Mg", 1))
            Ng = int(cfg.get("Ng", 1))
            Mp = int(cfg.get("Mp", 1))
            Np = int(cfg.get("Np", 1))
            num_h = int(cfg.get("num_h", N * Ng * Np))
            num_v = int(cfg.get("num_v", M * Mg * Mp))
            out = cls(
                num_h=num_h,
                num_v=num_v,
                d_h_lambda=float(cfg.get("d_h_lambda", cfg.get("dH", 0.5))),
                d_v_lambda=float(cfg.get("d_v_lambda", cfg.get("dV", 0.5))),
                polarization_count=P,
                num_txru=None if cfg.get("num_txru", cfg.get("num_rxru")) is None else int(cfg.get("num_txru", cfg.get("num_rxru"))),
                num_beams_h=None if cfg.get("num_beams_h") is None else int(cfg.get("num_beams_h")),
                num_beams_v=None if cfg.get("num_beams_v") is None else int(cfg.get("num_beams_v")),
                beam_scope=beam_scope,
                sampling_mode=sampling_mode,
                vertical_beam_mode=vertical_mode,
                fixed_v_index=fixed_v_index,
                model="tr38901_panel",
                M=M,
                N=N,
                P=P,
                Mg=Mg,
                Ng=Ng,
                Mp=Mp,
                Np=Np,
            )
            if out.num_ant != out.expected_ae:
                raise ValueError(
                    f"Invalid TRP array config: derived num_ant={out.num_ant}, "
                    f"but M*N*P*Mg*Ng*Mp*Np={out.expected_ae}"
                )
            return out
        return cls(
            num_h=int(cfg["num_h"]),
            num_v=int(cfg["num_v"]),
            d_h_lambda=float(cfg.get("d_h_lambda", cfg.get("dH", 0.5))),
            d_v_lambda=float(cfg.get("d_v_lambda", cfg.get("dV", 0.5))),
            polarization_count=int(cfg.get("polarization_count", 1)),
            num_txru=None if cfg.get("num_txru", cfg.get("num_rxru")) is None else int(cfg.get("num_txru", cfg.get("num_rxru"))),
            num_beams_h=None if cfg.get("num_beams_h") is None else int(cfg.get("num_beams_h")),
            num_beams_v=None if cfg.get("num_beams_v") is None else int(cfg.get("num_beams_v")),
            beam_scope=beam_scope,
            sampling_mode=sampling_mode,
            vertical_beam_mode=vertical_mode,
            fixed_v_index=fixed_v_index,
            model="legacy_upa",
        )


def _dft_vectors(n: int) -> np.ndarray:
    idx = np.arange(n)
    k = np.arange(n)
    mat = np.exp(1j * 2.0 * np.pi * np.outer(idx, k) / float(n)) / np.sqrt(float(n))
    return mat.astype(np.complex128)


def sampled_dft_indices(n: int, num_samples: int | None, mode: str = "uniform") -> List[int]:
    """Select DFT beam indices for SLS scanning.

    The full codebook is always the DFT grid. num_samples only controls how many
    directions are scanned. The default is uniform sampling over the DFT index
    range, which matches the requested interpretation of horizontal*vertical
    sampling counts.
    """
    n = int(n)
    if n <= 0:
        return []
    if num_samples is None or int(num_samples) >= n:
        return list(range(n))
    m = max(1, int(num_samples))
    mode = str(mode or "uniform").lower()
    if mode == "centered":
        order = [0]
        for k in range(1, n):
            if k <= n // 2:
                order.append(k)
            else:
                break
        for k in range(n - 1, n // 2, -1):
            order.append(k)
        out: List[int] = []
        for x in order + list(range(n)):
            if x not in out:
                out.append(x)
            if len(out) >= m:
                return out
        return out
    raw = np.linspace(0, n - 1, m)
    out = []
    for x in raw:
        ix = int(round(float(x)))
        if ix not in out:
            out.append(ix)
    # Rounding can create duplicates for small n. Fill deterministically.
    if len(out) < m:
        for ix in range(n):
            if ix not in out:
                out.append(ix)
            if len(out) >= m:
                break
    return out[:m]


def _expand_polarization(vec: np.ndarray, polarization_count: int) -> np.ndarray:
    p = int(polarization_count)
    if p <= 1:
        return vec.astype(np.complex128)
    out = np.kron(np.ones(p, dtype=np.complex128) / np.sqrt(float(p)), vec.astype(np.complex128))
    return out.astype(np.complex128)


def _expand_single_polarization(vec: np.ndarray, polarization_count: int, polarization_index: int | None) -> np.ndarray:
    """Place a spatial vector on one polarization block of the AE vector.

    The AE ordering used by this prototype is [pol0 spatial-grid, pol1 spatial-grid, ...].
    If polarization_index is None, the vector is distributed over all polarizations.
    """
    p = int(polarization_count)
    if polarization_index is None or p <= 1:
        return _expand_polarization(vec, p)
    out = np.zeros(p * vec.size, dtype=np.complex128)
    pol = int(polarization_index) % p
    out[pol * vec.size:(pol + 1) * vec.size] = vec.astype(np.complex128)
    return out


def _spatial_dft_vector(num_h: int, num_v: int, h_index: int, v_index: int) -> np.ndarray:
    vh = _dft_vectors(int(num_h))
    vv = _dft_vectors(int(num_v))
    return np.kron(vv[:, int(v_index)], vh[:, int(h_index)]).astype(np.complex128)


def _panel_padded_spatial_vector(array_cfg: ArrayConfig, panel_index: int, h_index: int, v_index: int) -> np.ndarray:
    local = _spatial_dft_vector(array_cfg.panel_num_h, array_cfg.panel_num_v, h_index, v_index)
    full_grid = np.zeros((array_cfg.num_v, array_cfg.num_h), dtype=np.complex128)
    p = int(panel_index)
    panel_cols = int(array_cfg.panel_grid_h)
    row = p // panel_cols
    col = p % panel_cols
    r0 = row * array_cfg.panel_num_v
    c0 = col * array_cfg.panel_num_h
    local_grid = local.reshape((array_cfg.panel_num_v, array_cfg.panel_num_h))
    full_grid[r0:r0 + array_cfg.panel_num_v, c0:c0 + array_cfg.panel_num_h] = local_grid
    return full_grid.reshape(-1)


def dft_2d_codebook(num_h: int,
                    num_v: int,
                    max_beams: int | None = None,
                    polarization_count: int = 1,
                    num_beams_h: int | None = None,
                    num_beams_v: int | None = None,
                    fixed_v_index: int | None = None,
                    sampling_mode: str = "uniform") -> np.ndarray:
    """Return DFT beams shaped [num_beams, num_ant]."""
    h_order = sampled_dft_indices(int(num_h), num_beams_h, sampling_mode)
    if fixed_v_index is not None:
        v_order = [int(fixed_v_index) % int(num_v)]
    else:
        v_order = sampled_dft_indices(int(num_v), num_beams_v, sampling_mode)
    beams: List[np.ndarray] = []
    for ih in h_order:
        for iv in v_order:
            spatial = _spatial_dft_vector(num_h, num_v, ih, iv)
            beams.append(_expand_polarization(spatial, polarization_count))
    cb = np.asarray(beams, dtype=np.complex128)
    if max_beams is not None:
        cb = cb[:int(max_beams)]
    cb = cb / np.maximum(np.linalg.norm(cb, axis=1, keepdims=True), 1e-12)
    return cb


def _fixed_v_for_array(array_cfg: ArrayConfig, fixed_v_index: int | None = None) -> int | None:
    if fixed_v_index is not None:
        return int(fixed_v_index)
    if str(array_cfg.vertical_beam_mode).lower() == "fixed" or array_cfg.fixed_v_index is not None:
        return 0 if array_cfg.fixed_v_index is None else int(array_cfg.fixed_v_index)
    return None


def dft_codebook_from_array(array_cfg: ArrayConfig, max_beams: int | None = None,
                            fixed_v_index: int | None = None) -> np.ndarray:
    scope = array_cfg.normalized_beam_scope
    if scope == "per_panel":
        # Backward-compatible API returns the local per-panel codebook. Network
        # TX beam construction uses zero-padded vectors via build_network_tx_beams().
        return dft_2d_codebook(array_cfg.panel_num_h, array_cfg.panel_num_v, max_beams=max_beams,
                               polarization_count=array_cfg.polarization_count,
                               num_beams_h=array_cfg.num_beams_h,
                               num_beams_v=array_cfg.num_beams_v,
                               fixed_v_index=_fixed_v_for_array(array_cfg, fixed_v_index),
                               sampling_mode=array_cfg.sampling_mode)
    return dft_2d_codebook(array_cfg.num_h, array_cfg.num_v, max_beams=max_beams,
                           polarization_count=array_cfg.polarization_count,
                           num_beams_h=array_cfg.num_beams_h,
                           num_beams_v=array_cfg.num_beams_v,
                           fixed_v_index=_fixed_v_for_array(array_cfg, fixed_v_index),
                           sampling_mode=array_cfg.sampling_mode)


def codebook_entries_from_array(array_cfg: ArrayConfig,
                                max_beams: int | None = None,
                                fixed_v_index: int | None = None,
                                panel_index: int | None = None,
                                polarization_index: int | None = None,
                                force_scope: str | None = None) -> List[Dict]:
    """Build DFT beam entries with metadata.

    In joint mode, entries are full-array DFT beams. In per-panel mode, entries
    are zero-padded local-panel DFT beams, one set per panel_index. In
    panel_polarization_subarray mode, the spatial vector is also restricted to a
    single polarization block so that different polarizations can use different beams.
    """
    fixed_v = _fixed_v_for_array(array_cfg, fixed_v_index)
    entries: List[Dict] = []
    scope = normalize_beam_scope(force_scope) if force_scope is not None else array_cfg.normalized_beam_scope
    if scope in ("per_panel", "panel_polarization_subarray"):
        panels = [int(panel_index)] if panel_index is not None else list(range(array_cfg.num_array_panels))
        for p in panels:
            h_order = sampled_dft_indices(array_cfg.panel_num_h, array_cfg.num_beams_h, array_cfg.sampling_mode)
            if fixed_v is not None:
                v_order = [int(fixed_v) % array_cfg.panel_num_v]
            else:
                v_order = sampled_dft_indices(array_cfg.panel_num_v, array_cfg.num_beams_v, array_cfg.sampling_mode)
            local_count = 0
            for ih in h_order:
                for iv in v_order:
                    spatial = _panel_padded_spatial_vector(array_cfg, p, ih, iv)
                    if scope == "panel_polarization_subarray":
                        vec = _expand_single_polarization(spatial, array_cfg.polarization_count, polarization_index)
                    else:
                        vec = _expand_polarization(spatial, array_cfg.polarization_count)
                    vec = vec / max(np.linalg.norm(vec), 1e-12)
                    entries.append({
                        "vector": vec,
                        "panel_index": p,
                        "h_index": int(ih),
                        "v_index": int(iv),
                        "beam_scope": scope,
                        "polarization_index": None if polarization_index is None else int(polarization_index),
                        "codebook_size": int(array_cfg.per_panel_codebook_size),
                    })
                    local_count += 1
                    if max_beams is not None and local_count >= int(max_beams):
                        break
                if max_beams is not None and local_count >= int(max_beams):
                    break
        return entries

    h_order = sampled_dft_indices(array_cfg.num_h, array_cfg.num_beams_h, array_cfg.sampling_mode)
    if fixed_v is not None:
        v_order = [int(fixed_v) % array_cfg.num_v]
    else:
        v_order = sampled_dft_indices(array_cfg.num_v, array_cfg.num_beams_v, array_cfg.sampling_mode)
    count = 0
    for ih in h_order:
        for iv in v_order:
            spatial = _spatial_dft_vector(array_cfg.num_h, array_cfg.num_v, ih, iv)
            vec = _expand_polarization(spatial, array_cfg.polarization_count)
            vec = vec / max(np.linalg.norm(vec), 1e-12)
            entries.append({
                "vector": vec,
                "panel_index": 0,
                "h_index": int(ih),
                "v_index": int(iv),
                "beam_scope": "joint",
                "polarization_index": None,
                "codebook_size": int(array_cfg.full_codebook_size),
            })
            count += 1
            if max_beams is not None and count >= int(max_beams):
                return entries
    return entries


def upa_steering_vector(num_h: int,
                        num_v: int,
                        azimuth_rad: float,
                        elevation_rad: float = 0.0,
                        d_h_lambda: float = 0.5,
                        d_v_lambda: float = 0.5,
                        polarization_count: int = 1) -> np.ndarray:
    """Simple UPA steering vector with optional duplicated polarization dimension."""
    h = np.arange(int(num_h))
    v = np.arange(int(num_v))
    u_h = np.sin(float(azimuth_rad)) * np.cos(float(elevation_rad))
    u_v = np.sin(float(elevation_rad))
    ah = np.exp(1j * 2.0 * np.pi * float(d_h_lambda) * h * u_h)
    av = np.exp(1j * 2.0 * np.pi * float(d_v_lambda) * v * u_v)
    spatial = np.kron(av, ah).astype(np.complex128)
    spatial = spatial / np.sqrt(float(num_h * num_v))
    return _expand_polarization(spatial, polarization_count)


def steering_vector_from_array(array_cfg: ArrayConfig,
                               azimuth_rad: float,
                               elevation_rad: float = 0.0) -> np.ndarray:
    return upa_steering_vector(array_cfg.num_h, array_cfg.num_v,
                               azimuth_rad, elevation_rad,
                               array_cfg.d_h_lambda, array_cfg.d_v_lambda,
                               array_cfg.polarization_count)


@dataclass(frozen=True)
class BeamId:
    cell: int
    trp: int
    panel: int
    beam: int
    global_index: int
    tx_unit: int = 0
    h_index: int | None = None
    v_index: int | None = None
    beam_scope: str = "joint"
    codebook_size: int | None = None
    array_panel_index: int | None = None
    polarization_index: int | None = None
    txru_index: int | None = None
    rf_connectivity: str | None = None

    def panel_key(self) -> Tuple[int, int, int]:
        return (self.cell, self.trp, self.panel)

    def short(self) -> str:
        return f"c{self.cell}t{self.trp}p{self.panel}b{self.beam}"

    def to_dict(self):
        return {
            "cell": int(self.cell),
            "trp": int(self.trp),
            "panel": int(self.panel),
            "beam": int(self.beam),
            "global_index": int(self.global_index),
            "tx_unit": int(self.tx_unit),
            "h_index": None if self.h_index is None else int(self.h_index),
            "v_index": None if self.v_index is None else int(self.v_index),
            "beam_scope": self.beam_scope,
            "codebook_size": None if self.codebook_size is None else int(self.codebook_size),
            "array_panel_index": None if self.array_panel_index is None else int(self.array_panel_index),
            "polarization_index": None if self.polarization_index is None else int(self.polarization_index),
            "txru_index": None if self.txru_index is None else int(self.txru_index),
            "rf_connectivity": self.rf_connectivity,
        }


def build_tx_beams(num_panels: int,
                   tx_cfg: ArrayConfig,
                   max_beams_per_panel: int | None,
                   fixed_v_index: int | None = None) -> Tuple[List[BeamId], np.ndarray]:
    beam_ids: List[BeamId] = []
    beam_vecs: List[np.ndarray] = []
    gi = 0
    phys = max(1, tx_cfg.num_array_panels if tx_cfg.normalized_beam_scope == "per_panel" else 1)
    for p in range(int(num_panels)):
        panel_idx = p % phys
        entries = codebook_entries_from_array(tx_cfg, max_beams=max_beams_per_panel,
                                              fixed_v_index=fixed_v_index,
                                              panel_index=panel_idx if tx_cfg.normalized_beam_scope == "per_panel" else None)
        for k, ent in enumerate(entries):
            beam_ids.append(BeamId(cell=0, trp=0, panel=p, beam=k, global_index=gi,
                                   tx_unit=p, h_index=ent["h_index"], v_index=ent["v_index"],
                                   beam_scope=ent["beam_scope"], codebook_size=ent["codebook_size"],
                                   array_panel_index=ent["panel_index"]))
            beam_vecs.append(ent["vector"])
            gi += 1
    return beam_ids, np.asarray(beam_vecs, dtype=np.complex128)


def build_network_tx_beams(num_cells: int,
                           panels_per_cell: int,
                           tx_cfg: ArrayConfig,
                           max_beams_per_panel: int | None,
                           site_id_by_cell: list[int] | None = None,
                           fixed_v_index: int | None = None,
                           rf_architecture=None) -> Tuple[List[BeamId], np.ndarray]:
    """Build network TX DFT beams under the resolved RF architecture.

    BeamId.tx_unit indexes the channel tensor axis [ue, tx_unit, freq, nrx, ntx].

    If rf_architecture is provided, panels_per_cell is interpreted as the number
    of local TX units per sector/cell. In the default sub-connected mode, these
    TX units are panel-polarization subarrays. In fully-connected mode, these TX
    units are full-array TXRUs. If rf_architecture is omitted, the legacy v2.3
    behavior is retained.
    """
    beam_ids: List[BeamId] = []
    beam_vecs: List[np.ndarray] = []
    gi = 0
    tx_unit = 0
    if site_id_by_cell is None:
        site_id_by_cell = [0 for _ in range(int(num_cells))]

    if rf_architecture is not None:
        local_units = list(rf_architecture.tx_units)
        if not local_units:
            raise ValueError("Resolved RF architecture has no TX units")
        for c in range(int(num_cells)):
            site_id = int(site_id_by_cell[c]) if c < len(site_id_by_cell) else 0
            for local_idx, unit in enumerate(local_units):
                entries = codebook_entries_from_array(
                    tx_cfg,
                    max_beams=max_beams_per_panel,
                    fixed_v_index=fixed_v_index,
                    panel_index=unit.array_panel_index,
                    polarization_index=unit.polarization_index,
                    force_scope=unit.beam_scope,
                )
                for k, ent in enumerate(entries):
                    beam_ids.append(BeamId(
                        cell=c,
                        trp=site_id,
                        panel=local_idx,
                        beam=k,
                        global_index=gi,
                        tx_unit=tx_unit,
                        h_index=ent["h_index"],
                        v_index=ent["v_index"],
                        beam_scope=ent["beam_scope"],
                        codebook_size=ent["codebook_size"],
                        array_panel_index=ent["panel_index"],
                        polarization_index=ent.get("polarization_index"),
                        txru_index=unit.txru_index,
                        rf_connectivity=rf_architecture.connectivity,
                    ))
                    beam_vecs.append(ent["vector"])
                    gi += 1
                tx_unit += 1
        return beam_ids, np.asarray(beam_vecs, dtype=np.complex128)

    # Legacy fallback.
    phys = max(1, tx_cfg.num_array_panels if tx_cfg.normalized_beam_scope == "per_panel" else 1)
    for c in range(int(num_cells)):
        site_id = int(site_id_by_cell[c]) if c < len(site_id_by_cell) else 0
        for p in range(int(panels_per_cell)):
            panel_idx = p % phys
            entries = codebook_entries_from_array(tx_cfg, max_beams=max_beams_per_panel,
                                                  fixed_v_index=fixed_v_index,
                                                  panel_index=panel_idx if tx_cfg.normalized_beam_scope == "per_panel" else None)
            for k, ent in enumerate(entries):
                beam_ids.append(BeamId(cell=c, trp=site_id, panel=p, beam=k, global_index=gi,
                                       tx_unit=tx_unit, h_index=ent["h_index"], v_index=ent["v_index"],
                                       beam_scope=ent["beam_scope"], codebook_size=ent["codebook_size"],
                                       array_panel_index=ent["panel_index"],
                                       polarization_index=ent.get("polarization_index")))
                beam_vecs.append(ent["vector"])
                gi += 1
            tx_unit += 1
    return beam_ids, np.asarray(beam_vecs, dtype=np.complex128)
