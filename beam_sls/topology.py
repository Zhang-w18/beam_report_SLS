from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class Site:
    site_id: int
    x_m: float
    y_m: float
    z_m: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "site_id": self.site_id,
            "x_m": float(self.x_m),
            "y_m": float(self.y_m),
            "z_m": float(self.z_m),
        }


@dataclass
class Sector:
    cell_id: int
    site_id: int
    sector_id: int
    azimuth_deg: float
    width_deg: float = 120.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "cell_id": int(self.cell_id),
            "site_id": int(self.site_id),
            "sector_id": int(self.sector_id),
            "azimuth_deg": float(self.azimuth_deg),
            "width_deg": float(self.width_deg),
        }


@dataclass
class UE:
    ue_id: int
    x_m: float
    y_m: float
    z_m: float = 1.5
    serving_cell: int = 0
    site_id: int = 0

    @property
    def distance_2d_m(self) -> float:
        return float(np.hypot(self.x_m, self.y_m))

    @property
    def azimuth_rad(self) -> float:
        return float(np.arctan2(self.y_m, self.x_m))

    def distance_to_site_2d_m(self, site: Site) -> float:
        return float(np.hypot(self.x_m - site.x_m, self.y_m - site.y_m))

    def azimuth_from_site_rad(self, site: Site) -> float:
        return float(np.arctan2(self.y_m - site.y_m, self.x_m - site.x_m))

    def to_dict(self) -> Dict[str, float]:
        return {
            "ue_id": int(self.ue_id),
            "x_m": float(self.x_m),
            "y_m": float(self.y_m),
            "z_m": float(self.z_m),
            "serving_cell": int(self.serving_cell),
            "site_id": int(self.site_id),
            "distance_2d_m": self.distance_2d_m,
            "azimuth_deg": float(np.rad2deg(self.azimuth_rad)),
        }


@dataclass
class Topology:
    ues: List[UE]
    sites: List[Site]
    sectors: List[Sector]
    carrier_frequency_ghz: float
    isd_m: float
    layout: str = "one_site_three_sector"
    # Backward-compatible fields used by some v1 helpers.
    sector_azimuth_deg: float = 0.0
    sector_width_deg: float = 120.0
    metadata: Dict = field(default_factory=dict)

    def site_by_id(self, site_id: int) -> Site:
        for s in self.sites:
            if s.site_id == site_id:
                return s
        raise KeyError(f"Unknown site_id={site_id}")

    def sector_by_cell(self, cell_id: int) -> Sector:
        for s in self.sectors:
            if s.cell_id == cell_id:
                return s
        raise KeyError(f"Unknown cell_id={cell_id}")

    @property
    def num_cells(self) -> int:
        return len(self.sectors)


def _angle_wrap(rad: float) -> float:
    return float(np.arctan2(np.sin(rad), np.cos(rad)))


def default_sector_azimuths(num_sectors: int) -> List[float]:
    if int(num_sectors) == 3:
        # Common three-sector convention: boresights separated by 120 deg.
        return [30.0, 150.0, 270.0]
    return [float(k) * 360.0 / float(num_sectors) for k in range(int(num_sectors))]


def build_sites(layout: str, isd_m: float, bs_height_m: float) -> List[Site]:
    if layout in ("single_cell", "one_site", "one_site_three_sector"):
        return [Site(site_id=0, x_m=0.0, y_m=0.0, z_m=float(bs_height_m))]
    if layout == "two_site_line":
        return [Site(0, 0.0, 0.0, float(bs_height_m)),
                Site(1, float(isd_m), 0.0, float(bs_height_m))]
    raise ValueError(f"Unsupported topology.layout={layout}")


def build_sectors(sites: Sequence[Site],
                  sectors_per_site: int,
                  sector_azimuths_deg: Sequence[float] | None,
                  sector_width_deg: float) -> List[Sector]:
    az = list(sector_azimuths_deg) if sector_azimuths_deg is not None else default_sector_azimuths(sectors_per_site)
    if len(az) < int(sectors_per_site):
        az = az + default_sector_azimuths(sectors_per_site)[len(az):]
    sectors: List[Sector] = []
    cell_id = 0
    for site in sites:
        for sid in range(int(sectors_per_site)):
            sectors.append(Sector(cell_id=cell_id,
                                  site_id=site.site_id,
                                  sector_id=sid,
                                  azimuth_deg=float(az[sid]),
                                  width_deg=float(sector_width_deg)))
            cell_id += 1
    return sectors


def drop_uniform_sector(num_ues: int,
                        min_radius_m: float,
                        max_radius_m: float,
                        sector: Sector,
                        site: Site,
                        rng: np.random.Generator,
                        start_ue_id: int = 0) -> List[UE]:
    center = np.deg2rad(float(sector.azimuth_deg))
    half = np.deg2rad(float(sector.width_deg) / 2.0)
    ues: List[UE] = []
    for k in range(int(num_ues)):
        r2 = rng.uniform(float(min_radius_m) ** 2, float(max_radius_m) ** 2)
        r = float(np.sqrt(r2))
        a = float(rng.uniform(center - half, center + half))
        ues.append(UE(start_ue_id + k,
                      float(site.x_m + r * np.cos(a)),
                      float(site.y_m + r * np.sin(a)),
                      z_m=1.5,
                      serving_cell=sector.cell_id,
                      site_id=site.site_id))
    return ues


def make_topology(cfg: Dict, rng: np.random.Generator) -> Topology:
    sc = cfg["scenario"]
    topo_cfg = cfg.get("topology", {})
    ud = cfg["ue_drop"]
    layout = str(topo_cfg.get("layout", "one_site_three_sector"))
    sectors_per_site = int(topo_cfg.get("sectors_per_site", 3 if layout == "one_site_three_sector" else 1))
    isd_m = float(topo_cfg.get("isd_m", sc.get("isd_m", 500.0)))
    bs_height_m = float(topo_cfg.get("bs_height_m", 25.0))
    sector_width_deg = float(topo_cfg.get("sector_width_deg", sc.get("sector_width_deg", 120.0)))
    sector_az = topo_cfg.get("sector_azimuths_deg", None)

    sites = build_sites(layout, isd_m, bs_height_m)
    sectors = build_sectors(sites, sectors_per_site, sector_az, sector_width_deg)

    ues: List[UE] = []
    uid = 0
    if ud.get("distribution", "uniform_in_sector") != "uniform_in_sector":
        raise ValueError("v2 currently supports ue_drop.distribution=uniform_in_sector")
    for sec in sectors:
        site = next(s for s in sites if s.site_id == sec.site_id)
        new_ues = drop_uniform_sector(
            num_ues=int(ud["num_ut_per_sector"]),
            min_radius_m=float(sc["min_ue_distance_m"]),
            max_radius_m=float(sc["max_ue_distance_m"]),
            sector=sec,
            site=site,
            rng=rng,
            start_ue_id=uid,
        )
        ues.extend(new_ues)
        uid += len(new_ues)

    first_sector = sectors[0] if sectors else Sector(0, 0, 0, 0.0, sector_width_deg)
    return Topology(ues=ues,
                    sites=sites,
                    sectors=sectors,
                    carrier_frequency_ghz=float(sc["carrier_frequency_ghz"]),
                    isd_m=isd_m,
                    layout=layout,
                    sector_azimuth_deg=float(first_sector.azimuth_deg),
                    sector_width_deg=float(sector_width_deg),
                    metadata={"topology_config": topo_cfg})


def make_phase1_topology(cfg: Dict, rng: np.random.Generator) -> Topology:
    """Backward-compatible alias used by v1 scripts/tests."""
    return make_topology(cfg, rng)


def topology_to_rows(topo: Topology) -> Tuple[List[Dict], List[Dict]]:
    site_rows = [s.to_dict() for s in topo.sites]
    sector_rows = [s.to_dict() for s in topo.sectors]
    return site_rows, sector_rows
