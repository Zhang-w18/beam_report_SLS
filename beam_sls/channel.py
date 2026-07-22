from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .codebook import ArrayConfig, steering_vector_from_array
from .rf import resolve_rf_architecture, trps_per_sector
from .topology import Sector, Site, Topology, UE
from .utils import db_to_lin


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


def uma_like_pathloss_db(distance_2d_m: float, fc_ghz: float, exponent: float = 3.0) -> float:
    """Lightweight UMa-like pathloss surrogate retained as fallback."""
    d = max(float(distance_2d_m), 1.0)
    return float(32.4 + 20.0 * np.log10(float(fc_ghz)) + 10.0 * float(exponent) * np.log10(d))


def _tx_units_from_topology(topology: Topology, cfg: Dict, tx_array: ArrayConfig | None = None) -> List[Tuple[int, Site, Sector, int, float]]:
    """Return list of (tx_unit, site, sector, local_unit, unit_boresight_deg).

    v2.4 uses the resolved RF architecture so that the channel tensor's TX-unit
    axis matches the beam generator exactly. In the default sub-connected mode,
    each TX unit is one panel-polarization subarray; TX units sharing the same
    physical panel also share the same boresight. In fully-connected mode, each
    TXRU is a full-array TX unit and uses the sector boresight unless offsets are
    explicitly configured.
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
    bw_hz = float(cfg["system"]["bandwidth_mhz"]) * 1e6
    freqs = np.linspace(-bw_hz / 2.0, bw_hz / 2.0, num_f, endpoint=False)
    num_l = int(sc.get("num_clusters", 8))
    delay_spread_s = float(sc.get("delay_spread_ns", 100.0)) * 1e-9
    fc_ghz = float(sc["carrier_frequency_ghz"])
    shadow_std = float(sc.get("shadow_fading_std_db", 4.0))
    exponent = float(sc.get("pathloss_exponent", 3.0))

    num_u = len(topology.ues)
    num_tx = len(tx_units)
    h = np.zeros((num_u, num_tx, num_f, rx_array.num_ant, tx_array.num_ant), dtype=np.complex128)
    pl_db = np.zeros(num_u, dtype=float)
    shadow_db = np.zeros(num_u, dtype=float)

    for ui, ue in enumerate(topology.ues):
        serving_site = topology.site_by_id(ue.site_id)
        pl_db[ui] = uma_like_pathloss_db(ue.distance_to_site_2d_m(serving_site), fc_ghz, exponent)
        shadow_db[ui] = rng.normal(0.0, shadow_std)
        gain_lin = float(db_to_lin(-(pl_db[ui] + shadow_db[ui])))
        for tx_unit, site, sec, local_panel, boresight in tx_units:
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
    return ChannelRealization(h_freq=h, freqs_hz=freqs, pathloss_db=pl_db, shadow_db=shadow_db,
                              backend=backend_name, backend_status="OK")


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


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
            base = dict(polarization=polarization,
                        polarization_type=polarization_type,
                        antenna_pattern=antenna_pattern,
                        carrier_frequency=fc_hz)
            return [
                dict(base, num_rows_per_panel=rows_per_panel, num_cols_per_panel=cols_per_panel,
                     num_rows_panels=rows_panels, num_cols_panels=cols_panels),
                dict(base, num_rows_per_panel=rows_per_panel, num_cols_per_panel=cols_per_panel),
                dict(base, num_rows=rows_per_panel * rows_panels, num_cols=cols_per_panel * cols_panels),
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

        bw_hz = float(self.cfg["system"]["bandwidth_mhz"]) * 1e6
        num_f = int(meas["num_freq_points"])
        freqs = np.linspace(-bw_hz / 2.0, bw_hz / 2.0, num_f, endpoint=False)
        a, tau = channel_model(num_time_samples=1, sampling_frequency=bw_hz)
        a_np = _to_numpy(a)
        tau_np = _to_numpy(tau)
        # Documented shape for Sionna 1.0.2 in this environment:
        #   a   [B, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, T]
        #   tau [B, num_rx, num_tx, num_paths]
        # Convert to the simulator-internal shape H[U, TX, F, Nr, Nt].
        if a_np.ndim != 7 or tau_np.ndim != 4:
            raise ChannelBackendError(f"Unexpected Sionna CIR shapes a={a_np.shape}, tau={tau_np.shape}")
        a0 = a_np[0, :, :, :, :, :, 0]  # [U, Nr, TX, Nt, L]
        t0 = tau_np[0]                  # [U, TX, L]
        num_u, nrx_sionna, num_tx, ntx_sionna, num_path = a0.shape
        h = np.zeros((num_u, num_tx, num_f, nrx_sionna, ntx_sionna), dtype=np.complex128)
        for fi, f in enumerate(freqs):
            ph = np.exp(-1j * 2.0 * np.pi * f * t0)  # [U, TX, L]
            # a0 indices are u,r,t,n,l. The previous v2.4.1 hotfix used
            # u,r,n,t,l, which swaps TX-unit and TX-antenna axes and fails
            # for multi-sector/multi-TXRU topologies.
            h[:, :, fi, :, :] = np.einsum("urtnl,utl->utrn", a0, ph, optimize=True)

        def _match_antenna_axis(x: np.ndarray, axis: int, target: int, role: str) -> np.ndarray:
            """Match Sionna PanelArray antenna count to the simulator AE vector.

            Sionna PanelArray reports one antenna dimension per spatial element for
            the dual-polarized panel configuration used here, while the simulator
            explicitly represents the 3GPP P dimension in the beam vector. If the
            simulator target dimension is an integer multiple of the Sionna
            dimension, duplicate the channel over the polarization blocks and
            scale by sqrt(factor) so unit-norm single-polarization DFT beams keep
            the expected per-port power normalization.
            """
            cur = int(x.shape[axis])
            tgt = int(target)
            if cur == tgt:
                return x
            if tgt > cur and tgt % cur == 0:
                rep = tgt // cur
                return np.repeat(x, rep, axis=axis) / np.sqrt(float(rep))
            if cur > tgt and cur % tgt == 0:
                # Conservative fallback for APIs that already expand polarization
                # more than the simulator vector. Average groups rather than silently
                # truncating.
                rep = cur // tgt
                new_shape = list(x.shape)
                new_shape[axis] = tgt
                new_shape.insert(axis + 1, rep)
                return x.reshape(new_shape).mean(axis=axis + 1) * np.sqrt(float(rep))
            raise ChannelBackendError(
                f"Sionna {role} antenna dimension {cur} is incompatible with simulator target {tgt}. "
                f"Check tx_array/ue_array P/M/N/Mg/Ng/Mp/Np and Sionna PanelArray polarization settings."
            )

        h = _match_antenna_axis(h, axis=3, target=self.rx_array.num_ant, role="RX")
        h = _match_antenna_axis(h, axis=4, target=self.tx_array.num_ant, role="TX")
        # Pathloss is already included by Sionna. Fill diagnostic arrays with NaN.
        status = (
            f"OK: Sionna CIR a={a_np.shape}, tau={tau_np.shape}, "
            f"internal_h={h.shape}, sionna_rx_ant={nrx_sionna}, sionna_tx_ant={ntx_sionna}"
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
            return SionnaTR38901Adapter(model, cfg, tx_array, rx_array).generate(topology, rng)
        except Exception as e:
            if bool(cfg.get("sionna", {}).get("fallback_to_numpy_if_unavailable", True)):
                ch = generate_numpy_geometric_channel(topology, cfg, tx_array, rx_array, rng,
                                                      backend_name=f"fallback_numpy_for_{model}")
                ch.backend_status = f"FALLBACK: {type(e).__name__}: {e}"
                return ch
            raise
    if model in ("numpy_geometric_uma", "numpy_geometric"):
        return generate_numpy_geometric_channel(topology, cfg, tx_array, rx_array, rng)
    raise ValueError(f"Unsupported scenario.channel_model={model}")


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
