import numpy as np
import pytest

from beam_sls.config import load_config
from beam_sls.codebook import BeamId
from beam_sls.measurement import associate_ues_by_average_rsrp
from beam_sls.topology import (
    Sector,
    Site,
    Topology,
    UE,
    assign_ues_to_scheduling_clusters,
    cluster_cell_ids_by_ue,
    make_topology,
    measurement_cell_ids_by_ue,
    neighbor_area_vertex_sites,
    normalize_measurement_domain,
    resolve_static_scheduling_clusters,
    serving_cell_from_position,
)


def _seven_site_topology(num_ues_per_cell=1):
    cfg = load_config(None)
    cfg["topology"]["layout"] = "seven_site_hex"
    cfg["topology"]["num_sites"] = 7
    cfg["ue_drop"]["num_ut_per_sector"] = num_ues_per_cell
    return make_topology(cfg, np.random.default_rng(20260724))


def test_drop_keeps_exact_count_per_geometric_cell_without_rsrp():
    topo = _seven_site_topology(num_ues_per_cell=2)
    counts = {sector.cell_id: 0 for sector in topo.sectors}

    for ue in topo.ues:
        cell_id, site_id = serving_cell_from_position(
            ue.x_m, ue.y_m, topo.sites, topo.sectors
        )
        assert (ue.serving_cell, ue.site_id) == (cell_id, site_id)
        counts[cell_id] += 1

    assert set(counts.values()) == {2}


def test_trp_and_site_measurement_domains_are_independent():
    topo = _seven_site_topology()
    trp_cells = measurement_cell_ids_by_ue(topo, "trp")
    site_cells = measurement_cell_ids_by_ue(topo, "site")

    for ue in topo.ues:
        assert trp_cells[ue.ue_id] == [ue.serving_cell]
        assert len(site_cells[ue.ue_id]) == 3
        assert {
            topo.sector_by_cell(cell_id).site_id
            for cell_id in site_cells[ue.ue_id]
        } == {ue.site_id}


def test_neighbor_area_uses_one_cell_per_hex_vertex_not_whole_sites():
    topo = _seven_site_topology()
    # Put a deterministic UE inside the center site's cell. The six ring sites
    # are the available vertices of its first-tier hexagonal neighborhood.
    topo.ues = [UE(0, 10.0, 0.0, serving_cell=0, site_id=0)]

    vertices = neighbor_area_vertex_sites(topo, topo.ues[0])
    cells = measurement_cell_ids_by_ue(topo, "neighbor_area")[0]

    assert len(vertices) == 6
    assert len(cells) == 6
    assert 0 not in cells
    vertex_site_ids = {site.site_id for site in vertices}
    selected_vertex_sites = {
        topo.sector_by_cell(cell_id).site_id
        for cell_id in cells
    }
    assert selected_vertex_sites == vertex_site_ids
    assert all(
        sum(
            topo.sector_by_cell(cell_id).site_id == site_id
            for cell_id in cells
        ) == 1
        for site_id in vertex_site_ids
    )
    ue = topo.ues[0]
    for cell_id in cells:
        selected = topo.sector_by_cell(cell_id)
        vertex = topo.site_by_id(selected.site_id)
        ue_azimuth_deg = np.rad2deg(
            np.arctan2(ue.y_m - vertex.y_m, ue.x_m - vertex.x_m)
        )

        def angular_error(sector):
            delta = np.deg2rad(ue_azimuth_deg - sector.azimuth_deg)
            return abs(np.arctan2(np.sin(delta), np.cos(delta)))

        assert angular_error(selected) == min(
            angular_error(sector)
            for sector in topo.sectors
            if sector.site_id == selected.site_id
        )


def test_measurement_domain_rejects_unknown_mode():
    assert normalize_measurement_domain("cell") == "trp"
    assert normalize_measurement_domain("neighbour_area") == "neighbor_area"
    with pytest.raises(ValueError, match="measurement.domain_mode"):
        normalize_measurement_domain("rsrp")


def _two_cell_topology():
    return Topology(
        ues=[
            UE(0, 0.0, 0.0, serving_cell=0, site_id=0),
            UE(1, 1.0, 0.0, serving_cell=1, site_id=1),
        ],
        sites=[Site(0, 0.0, 0.0, 25.0), Site(1, 100.0, 0.0, 25.0)],
        sectors=[
            Sector(0, 0, 0, 0.0),
            Sector(1, 1, 0, 180.0),
        ],
        carrier_frequency_ghz=3.5,
        isd_m=100.0,
    )


def test_average_rsrp_association_uses_frequency_average_and_cell_id_tie_break():
    topo = _two_cell_topology()
    beam_ids = [
        BeamId(0, 0, 0, 0, 0, tx_unit=0),
        BeamId(1, 1, 0, 0, 1, tx_unit=1),
    ]
    # UE0: cell 1 has larger mean power. UE1: equal mean power, so cell 0 wins.
    h = np.zeros((2, 2, 2, 1, 1), dtype=np.complex128)
    h[0, 0, :, 0, 0] = [1.0, 1.0]
    h[0, 1, :, 0, 0] = [1.0, 3.0]
    # Both cells have mean power 4 for UE1: mean([0, 8]) == mean([4, 4]).
    h[1, 0, :, 0, 0] = [0.0, np.sqrt(8.0)]
    h[1, 1, :, 0, 0] = [2.0, 2.0]
    scores = associate_ues_by_average_rsrp(
        h,
        np.ones((2, 1), dtype=np.complex128),
        np.ones((1, 1), dtype=np.complex128),
        beam_ids,
        topo,
        tx_power_w_per_panel=1.0,
    )
    assert topo.ues[0].serving_cell == 1
    assert topo.ues[0].site_id == 1
    assert scores[0][1] > scores[0][0]
    assert topo.ues[1].serving_cell == 0


def test_custom_static_clusters_are_disjoint_complete_and_define_measurement():
    topo = _two_cell_topology()
    clusters, owner = resolve_static_scheduling_clusters(
        topo,
        {
            "cluster_mode": "custom",
            "static_clusters": [
                {"cluster_id": 7, "cell_ids": [0]},
                {"cluster_id": 8, "cell_ids": [1]},
            ],
        },
    )
    ue_clusters = assign_ues_to_scheduling_clusters(topo, owner)
    measured = cluster_cell_ids_by_ue(topo, clusters)
    assert ue_clusters == {0: 7, 1: 8}
    assert measured == {0: [0], 1: [1]}

    with pytest.raises(ValueError, match="belongs to scheduling clusters"):
        resolve_static_scheduling_clusters(
            topo,
            {
                "cluster_mode": "custom",
                "static_clusters": [[0, 1], [1]],
            },
        )
    with pytest.raises(ValueError, match="missing cells"):
        resolve_static_scheduling_clusters(
            topo,
            {"cluster_mode": "custom", "static_clusters": [[0]]},
        )
