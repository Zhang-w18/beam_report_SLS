from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .codebook import (
    ArrayConfig,
    sionna_panelarray_source_indices,
    steering_vector_from_array,
)
from .rf import resolve_rf_architecture, trps_per_sector
from .topology import Sector, Site, Topology, UE
from .utils import db_to_lin, occupied_bandwidth_hz


class ChannelBackendError(RuntimeError):
    pass


@dataclass
class ChannelRealization:
    # H[ue, tx_unit, freq, nrx, ntx]
    h_freq: np.ndarray
    freqs_hz: np.ndarray
    pathloss_db: np.ndarray  # [ue]
    shadow_db: np.ndarray    # [ue]
    backend: str = "numpy_geometric_uma"
    backend_status: str = "OK"
    pathloss_db_by_site: np.ndarray | None = None
    shadow_db_by_site: np.ndarray | None = None


def _bessel_j0(x: float) -> float:
    """Bessel J0 without adding a SciPy dependency."""
    xx = 0.25 * float(x) * float(x)
    term = 1.0
    total = 1.0
    for k in range(1, 64):
        term *= -xx / float(k * k)
        total += term
        if abs(term) <= 1e-15 * max(1.0, abs(total)):
            break
    return float(total)


@dataclass
class DopplerChannelEvolver:
    """Advance only small-scale fading while retaining one drop's large scale.

    A link-wise first-order complex Gauss-Markov process uses the Clarke/Jakes
    correlation rho=J0(2*pi*f_D*delta_t). One scalar fading state per
    (UE, TX unit) keeps the update small even when H has many antennas, while
    path loss, shadow fading, delay profile, and spatial signature stay fixed.
    """

    h_freq: np.ndarray
    rho: float
    fading_by_link: np.ndarray
    rng: np.random.Generator
    tti_index: int = 0

    @classmethod
    def from_drop(cls,
                  h_freq: np.ndarray,
                  cfg: Dict,
                  rng: np.random.Generator) -> "DopplerChannelEvolver":
        fc_hz = float(cfg["scenario"]["carrier_frequency_ghz"]) * 1e9
        speed_mps = float(cfg.get("ue_drop", {}).get("speed_kmh", 0.0)) / 3.6
        tti_s = float(cfg["pdsch"].get("slot_duration_ms", 0.125)) * 1e-3
        max_doppler_hz = speed_mps * fc_hz / 299_792_458.0
        rho = _bessel_j0(2.0 * pi * max_doppler_hz * tti_s)
        # A negative J0 is valid, but numerical roundoff must not make |rho|>1.
        # Preserve rho=1 exactly so zero-speed UEs remain static.
        rho = float(np.clip(rho, -1.0, 1.0))
        base = np.asarray(h_freq, dtype=np.complex128)
        fading = np.ones(base.shape[:2] + (1, 1, 1), dtype=np.complex128)
        return cls(h_freq=base.copy(), rho=rho, fading_by_link=fading, rng=rng)

    def current(self) -> np.ndarray:
        return self.h_freq

    def advance(self) -> np.ndarray:
        if abs(self.rho) >= 1.0 - 1e-12:
            self.tti_index += 1
            return self.h_freq
        innovation = (
            self.rng.standard_normal(self.fading_by_link.shape)
            + 1j * self.rng.standard_normal(self.fading_by_link.shape)
        ) / np.sqrt(2.0)
        next_fading = (
            self.rho * self.fading_by_link
            + np.sqrt(max(0.0, 1.0 - self.rho * self.rho)) * innovation
        )
        # Maintain H_t = H_0*g_t without retaining another full H tensor.
        # Exact zeros have probability zero; the guard only avoids a numerical
        # division fault in pathological injected test data.
        denominator = np.where(
            np.abs(self.fading_by_link) > 1e-15,
            self.fading_by_link,
            1e-15 + 0.0j,
        )
        self.h_freq *= next_fading / denominator
        self.fading_by_link = next_fading
        self.tti_index += 1
        return self.h_freq


def uma_like_pathloss_db(distance_2d_m: float, fc_ghz: float, exponent: float = 3.0) -> float:
    """Lightweight UMa-like pathloss surrogate retained as fallback."""
    d = max(float(distance_2d_m), 1.0)
    return float(32.4 + 20.0 * np.log10(float(fc_ghz)) + 10.0 * float(exponent) * np.log10(d))


def _tx_units_from_topology(topology: Topology, cfg: Dict, tx_array: ArrayConfig | None = None) -> List[Tuple[int, Site, Sector, int, float]]:
    """Return list of (tx_unit, site, sector, local_unit, unit_boresight_deg).

    The resolved RF architecture keeps the channel tensor axis aligned with the
    beam generator. In shared-codebook mode there is one channel axis per TRP
    and selected codewords are not bound to TXRUs. The legacy independent-
    polarization mode retains one axis per panel-polarization subarray. In
    fully-connected mode, each TXRU is a full-array TX unit.
    """
    trp = cfg.get("trp", {})
    if tx_array is not None:
        rf = resolve_rf_architecture(cfg, tx_array)
        local_units = list(rf.tx_units)
        num_trps = max(1, trps_per_sector(cfg))
    else:
        rf = None
        panels_per_cell = int(trp.get("num_panels_per_sector", trp.get("num_panels", 1)))
        local_units = [None for _ in range(panels_per_cell)]
        num_trps = 1
    offsets = list(trp.get("panel_azimuth_offsets_deg", [0.0]))
    units = []
    idx = 0
    for sec in topology.sectors:
        site = topology.site_by_id(sec.site_id)
        for trp_idx in range(num_trps):
            for local_idx, unit in enumerate(local_units):
                if unit is None:
                    array_panel_index = local_idx
                else:
                    array_panel_index = 0 if unit.array_panel_index is None else int(unit.array_panel_index)
                off = float(offsets[array_panel_index]) if array_panel_index < len(offsets) else 0.0
                units.append((idx, site, sec, local_idx, float(sec.azimuth_deg) + off))
                idx += 1
    return units

def _relative_angle_from_site(ue: UE, site: Site, boresight_deg: float) -> float:
    panel_boresight = np.deg2rad(float(boresight_deg))
    az = ue.azimuth_from_site_rad(site)
    rel = az - panel_boresight
    return float(np.arctan2(np.sin(rel), np.cos(rel)))


def generate_numpy_geometric_channel(topology: Topology,
                                      cfg: Dict,
                                      tx_array: ArrayConfig,
                                      rx_array: ArrayConfig,
                                      rng: np.random.Generator,
                                      backend_name: str = "numpy_geometric_uma") -> ChannelRealization:
    sc = cfg["scenario"]
    meas = cfg["measurement"]
    tx_units = _tx_units_from_topology(topology, cfg, tx_array)

    num_f = int(meas["num_freq_points"])
    bw_hz = occupied_bandwidth_hz(cfg)
    freqs = np.linspace(-bw_hz / 2.0, bw_hz / 2.0, num_f, endpoint=False)
    num_l = int(sc.get("num_clusters", 8))
    delay_spread_s = float(sc.get("delay_spread_ns", 100.0)) * 1e-9
    fc_ghz = float(sc["carrier_frequency_ghz"])
    shadow_std = float(sc.get("shadow_fading_std_db", 4.0))
    exponent = float(sc.get("pathloss_exponent", 3.0))
    num_u = len(topology.ues)
    num_tx = len(tx_units)
    h = np.zeros(
        (num_u, num_tx, num_f, rx_array.num_ant, tx_array.num_ant),
        dtype=np.complex128,
    )
    pl_db = np.zeros(num_u, dtype=float)
    shadow_db = np.zeros(num_u, dtype=float)
    num_sites = len(topology.sites)
    pl_db_by_site = np.zeros((num_u, num_sites), dtype=float)
    shadow_db_by_site = np.zeros((num_u, num_sites), dtype=float)

    for ui, ue in enumerate(topology.ues):
        # One large-scale state per UE-site link. Reusing it across sectors at
        # the same site avoids artificial sector-specific shadowing while
        # allowing RSRP-based association across sites.
        site_large_scale = {}
        for site in topology.sites:
            pathloss = uma_like_pathloss_db(
                ue.distance_to_site_2d_m(site), fc_ghz, exponent
            )
            shadow = float(rng.normal(0.0, shadow_std))
            site_large_scale[int(site.site_id)] = (pathloss, shadow)
            pl_db_by_site[ui, int(site.site_id)] = pathloss
            shadow_db_by_site[ui, int(site.site_id)] = shadow
        pl_db[ui], shadow_db[ui] = site_large_scale[int(ue.site_id)]
        for tx_unit, site, sec, local_panel, boresight in tx_units:
            link_pathloss_db, link_shadow_db = site_large_scale[int(site.site_id)]
            gain_lin = float(db_to_lin(-(link_pathloss_db + link_shadow_db)))
            rel_az = _relative_angle_from_site(ue, site, boresight)
            base_tx_az = rel_az
            base_rx_az = np.pi + rel_az
            horizontal_m = max(ue.distance_to_site_2d_m(site), 1e-9)
            base_tx_el = float(np.arctan2(ue.z_m - site.z_m, horizontal_m))
            base_rx_el = -base_tx_el
            delays = rng.exponential(scale=max(delay_spread_s, 1e-12), size=num_l)
            delays = delays - np.min(delays)
            powers = np.exp(-delays / max(delay_spread_s, 1e-12))
            powers = powers / np.sum(powers)
            coeffs = (rng.normal(size=num_l) + 1j * rng.normal(size=num_l)) / np.sqrt(2.0)
            for l in range(num_l):
                tx_az = base_tx_az + rng.normal(0.0, np.deg2rad(8.0))
                rx_az = base_rx_az + rng.normal(0.0, np.deg2rad(20.0))
                tx_el = base_tx_el + rng.normal(0.0, np.deg2rad(3.0))
                rx_el = base_rx_el + rng.normal(0.0, np.deg2rad(10.0))
                atx = steering_vector_from_array(tx_array, tx_az, tx_el)
                arx = steering_vector_from_array(rx_array, rx_az, rx_el)
                outer = np.outer(arx, np.conjugate(atx))
                phase = np.exp(-1j * 2.0 * np.pi * freqs * delays[l])
                h[ui, tx_unit, :, :, :] += (np.sqrt(gain_lin * powers[l]) * coeffs[l] * phase)[:, None, None] * outer[None, :, :]
    return ChannelRealization(
        h_freq=h,
        freqs_hz=freqs,
        pathloss_db=pl_db,
        shadow_db=shadow_db,
        backend=backend_name,
        backend_status="OK",
        pathloss_db_by_site=pl_db_by_site,
        shadow_db_by_site=shadow_db_by_site,
    )


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def sionna_cir_to_internal_frequency_response(
    a: np.ndarray,
    tau: np.ndarray,
    freqs_hz: np.ndarray,
    tx_array: ArrayConfig,
    rx_array: ArrayConfig,
    time_index: int = 0,
) -> np.ndarray:
    """Convert official Sionna CIR axes to ``H[U,TX,F,Nr,Nt]``.

    Expected Sionna axes are
    ``a[B,RX,RX_ANT,TX,TX_ANT,PATH,TIME]`` and
    ``tau[B,RX,TX,PATH]``. The antenna axes are then explicitly permuted from
    Sionna ``PanelArray`` ordering to this project's codebook ordering.
    """
    a_np = np.asarray(a)
    tau_np = np.asarray(tau)
    if a_np.ndim != 7 or tau_np.ndim != 4:
        raise ChannelBackendError(
            f"Unexpected Sionna CIR shapes a={a_np.shape}, tau={tau_np.shape}"
        )
    if a_np.shape[0] != 1 or tau_np.shape[0] != 1:
        raise ChannelBackendError("The simulator expects one Sionna batch per drop")
    if not 0 <= int(time_index) < int(a_np.shape[-1]):
        raise ChannelBackendError(
            f"Sionna time_index={time_index} outside TIME axis {a_np.shape[-1]}"
        )
    a0 = a_np[0, :, :, :, :, :, int(time_index)]  # [U,Nr,TX,Nt,L]
    t0 = tau_np[0]                                  # [U,TX,L]
    if (
        a0.shape[0] != t0.shape[0]
        or a0.shape[2] != t0.shape[1]
        or a0.shape[4] != t0.shape[2]
    ):
        raise ChannelBackendError(
            f"Incompatible Sionna CIR axes a={a_np.shape}, tau={tau_np.shape}"
        )
    if int(a0.shape[1]) != int(rx_array.num_ant):
        raise ChannelBackendError(
            f"Sionna RX antenna dimension {a0.shape[1]} != configured {rx_array.num_ant}"
        )
    if int(a0.shape[3]) != int(tx_array.num_ant):
        raise ChannelBackendError(
            f"Sionna TX antenna dimension {a0.shape[3]} != configured {tx_array.num_ant}"
        )

    phase = np.exp(
        -1j
        * 2.0
        * np.pi
        * np.asarray(freqs_hz, dtype=float)[None, None, :, None]
        * t0[:, :, None, :]
    )  # [U,TX,F,L]
    h_sionna = np.einsum("urtnl,utfl->utfrn", a0, phase, optimize=True)
    rx_source = sionna_panelarray_source_indices(rx_array)
    tx_source = sionna_panelarray_source_indices(tx_array)
    return np.take(
        np.take(h_sionna, rx_source, axis=3),
        tx_source,
        axis=4,
    )


class SionnaTR38901Adapter:
    """Best-effort adapter for Sionna PHY TR 38.901 UMa/UMi/RMa.

    Sionna's public API changed across versions. This adapter targets the
    documented sionna.phy.channel.tr38901 API where UMa/UMi/RMa generate CIR
    coefficients and delays after set_topology(). If construction fails, callers
    can either raise or fall back to numpy_geometric_uma depending on config.
    """

    def __init__(self, model_name: str, cfg: Dict, tx_array: ArrayConfig, rx_array: ArrayConfig):
        self.model_name = model_name
        self.cfg = cfg
        self.tx_array = tx_array
        self.rx_array = rx_array
        self.status = "not_initialized"

    @property
    def scenario(self) -> str:
        return self.model_name.replace("sionna_tr38901_", "").lower()

    def generate(self, topology: Topology, rng: np.random.Generator) -> ChannelRealization:
        try:
            import tensorflow as tf  # type: ignore
            from sionna.phy.channel.tr38901 import PanelArray, UMa, UMi, RMa  # type: ignore
        except Exception as e:
            raise ChannelBackendError(f"Sionna TR38901 backend unavailable: {type(e).__name__}: {e}") from e

        sc = self.cfg["scenario"]
        meas = self.cfg["measurement"]
        sionna_cfg = self.cfg.get("sionna", {})
        device = sionna_cfg.get("device", None)
        precision = sionna_cfg.get("precision", None)
        fc_hz = float(sc["carrier_frequency_ghz"]) * 1e9

        def _panel_kwargs(array_cfg: ArrayConfig, role: str) -> List[Dict]:
            # For 3GPP-style config, use per-panel dimensions M/N and panel
            # grid Mg/Mp by Ng/Np. For legacy config, fall back to one panel
            # with num_v by num_h elements.
            if array_cfg.model == "tr38901_panel":
                rows_per_panel = int(array_cfg.M or array_cfg.num_v)
                cols_per_panel = int(array_cfg.N or array_cfg.num_h)
                rows_panels = int(array_cfg.Mg or 1) * int(array_cfg.Mp or 1)
                cols_panels = int(array_cfg.Ng or 1) * int(array_cfg.Np or 1)
            else:
                rows_per_panel = int(array_cfg.num_v)
                cols_per_panel = int(array_cfg.num_h)
                rows_panels = 1
                cols_panels = 1
            pol_default = "dual" if int(array_cfg.polarization_count) == 2 else "single"
            pol_type_default = "cross" if pol_default == "dual" else "V"
            polarization = sionna_cfg.get(f"{role}_polarization", pol_default)
            polarization_type = sionna_cfg.get(f"{role}_polarization_type", pol_type_default)
            antenna_pattern = sionna_cfg.get(f"{role}_antenna_pattern", "38.901" if role == "bs" else "omni")
            base = dict(
                polarization=polarization,
                polarization_type=polarization_type,
                antenna_pattern=antenna_pattern,
                carrier_frequency=fc_hz,
                element_vertical_spacing=float(array_cfg.d_v_lambda),
                element_horizontal_spacing=float(array_cfg.d_h_lambda),
                panel_vertical_spacing=float(rows_per_panel)
                * float(array_cfg.d_v_lambda),
                panel_horizontal_spacing=float(cols_per_panel)
                * float(array_cfg.d_h_lambda),
            )
            return [
                # Official Sionna 1.2+ names: num_rows/num_cols are panel-grid
                # dimensions, not total antenna-element dimensions.
                dict(base, num_rows_per_panel=rows_per_panel,
                     num_cols_per_panel=cols_per_panel,
                     num_rows=rows_panels, num_cols=cols_panels),
                # Compatibility with releases that used explicit panel suffixes.
                dict(base, num_rows_per_panel=rows_per_panel, num_cols_per_panel=cols_per_panel,
                     num_rows_panels=rows_panels, num_cols_panels=cols_panels),
            ]

        def _make_panel_array(array_cfg: ArrayConfig, role: str):
            last_error = None
            for kwargs_pa in _panel_kwargs(array_cfg, role):
                try:
                    return PanelArray(**kwargs_pa)
                except TypeError as e:
                    last_error = e
            raise last_error if last_error is not None else ChannelBackendError("PanelArray construction failed")

        bs_array = _make_panel_array(self.tx_array, "bs")
        ut_array = _make_panel_array(self.rx_array, "ut")

        cls = {"uma": UMa, "umi": UMi, "rma": RMa}.get(self.scenario)
        if cls is None:
            raise ChannelBackendError(f"Unsupported Sionna TR38901 scenario: {self.scenario}")
        kwargs = dict(carrier_frequency=fc_hz,
                      ut_array=ut_array,
                      bs_array=bs_array,
                      direction="downlink",
                      enable_pathloss=bool(sc.get("enable_pathloss", True)),
                      enable_shadow_fading=bool(sc.get("enable_shadow_fading", True)))
        if self.scenario in ("uma", "umi"):
            kwargs["o2i_model"] = sc.get("o2i_model", "low")
        if self.scenario == "rma":
            kwargs["average_street_width"] = float(sc.get("average_street_width", 20.0))
            kwargs["average_building_height"] = float(sc.get("average_building_height", 5.0))
        if precision is not None:
            kwargs["precision"] = precision
        if device is not None:
            kwargs["device"] = device
        channel_model = cls(**kwargs)

        tx_units = _tx_units_from_topology(topology, self.cfg, self.tx_array)
        bs_locs = np.asarray([[site.x_m, site.y_m, site.z_m] for _, site, _, _, _ in tx_units], dtype=np.float32)
        bs_orient = np.asarray([[np.deg2rad(boresight), 0.0, 0.0] for _, _, _, _, boresight in tx_units], dtype=np.float32)
        ut_locs = np.asarray([[u.x_m, u.y_m, u.z_m] for u in topology.ues], dtype=np.float32)
        ut_orient = np.zeros((len(topology.ues), 3), dtype=np.float32)
        speed_mps = float(self.cfg.get("ue_drop", {}).get("speed_kmh", 3.0)) / 3.6
        ut_vel = np.zeros((len(topology.ues), 3), dtype=np.float32)
        ut_vel[:, 0] = speed_mps
        in_state = np.zeros((len(topology.ues),), dtype=bool)

        def T(x, dtype=None):
            if dtype is None:
                return tf.convert_to_tensor(x)
            return tf.convert_to_tensor(x, dtype=dtype)

        # Batch dimension = 1. Sionna 1.0.2 TR38901 in this environment is
        # TensorFlow-backed; use TF tensors rather than torch tensors.
        channel_model.set_topology(T(ut_locs[None, ...], tf.float32),
                                   T(bs_locs[None, ...], tf.float32),
                                   T(ut_orient[None, ...], tf.float32),
                                   T(bs_orient[None, ...], tf.float32),
                                   T(ut_vel[None, ...], tf.float32),
                                   T(in_state[None, ...], tf.bool))

        bw_hz = occupied_bandwidth_hz(self.cfg)
        num_f = int(meas["num_freq_points"])
        freqs = np.linspace(-bw_hz / 2.0, bw_hz / 2.0, num_f, endpoint=False)
        a, tau = channel_model(num_time_samples=1, sampling_frequency=bw_hz)
        a_np = _to_numpy(a)
        tau_np = _to_numpy(tau)
        h = sionna_cir_to_internal_frequency_response(
            a_np, tau_np, freqs, self.tx_array, self.rx_array, time_index=0
        )
        num_u = int(h.shape[0])
        # Pathloss is already included by Sionna. Fill diagnostic arrays with NaN.
        status = (
            f"OK: Sionna CIR a={a_np.shape}, tau={tau_np.shape}, "
            f"internal_h={h.shape}, time_index=0, antenna_order=explicitly_mapped"
        )
        return ChannelRealization(h_freq=h, freqs_hz=freqs,
                                  pathloss_db=np.full(num_u, np.nan),
                                  shadow_db=np.full(num_u, np.nan),
                                  backend=self.model_name,
                                  backend_status=status)


def generate_channel(topology: Topology,
                     cfg: Dict,
                     tx_array: ArrayConfig,
                     rx_array: ArrayConfig,
                     rng: np.random.Generator) -> ChannelRealization:
    model = str(cfg.get("scenario", {}).get("channel_model", "numpy_geometric_uma")).lower()
    if model.startswith("sionna_tr38901_"):
        try:
            channel = SionnaTR38901Adapter(model, cfg, tx_array, rx_array).generate(topology, rng)
        except Exception as e:
            if bool(cfg.get("sionna", {}).get("fallback_to_numpy_if_unavailable", True)):
                channel = generate_numpy_geometric_channel(
                    topology, cfg, tx_array, rx_array, rng,
                    backend_name=f"fallback_numpy_for_{model}",
                )
                channel.backend_status = f"FALLBACK: {type(e).__name__}: {e}"
            else:
                raise
    elif model in ("numpy_geometric_uma", "numpy_geometric"):
        channel = generate_numpy_geometric_channel(topology, cfg, tx_array, rx_array, rng)
    else:
        raise ValueError(f"Unsupported scenario.channel_model={model}")
    return channel


class SionnaImportProbe:
    """Probe Sionna 1.x/2.x modules without making the default run depend on them."""

    def run(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        modules = [
            "sionna",
            "sionna.phy",
            "sionna.phy.channel",
            "sionna.phy.channel.tr38901",
            "sionna.phy.ofdm",
            "sionna.sys",
            "torch",
            "tensorflow",
        ]
        for m in modules:
            try:
                mod = __import__(m, fromlist=["*"])
                ver = getattr(mod, "__version__", None)
                out[m] = "OK" + (f" version={ver}" if ver else "")
            except Exception as e:  # pragma: no cover - depends on local env
                out[m] = f"FAILED: {type(e).__name__}: {e}"
        return out
