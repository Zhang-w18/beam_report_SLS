import numpy as np

from beam_sls.config import load_config
from beam_sls.link_adaptation import (
    BaseLinkAdapter,
    LinkAdaptationBackendError,
    PHYTblerLookup,
    SionnaSYSAdapter,
)


def test_sionna_sys_invalid_low_sinr_treated_as_error_probability_one():
    cfg = load_config(None)
    try:
        adapter = SionnaSYSAdapter(cfg)
    except LinkAdaptationBackendError:
        return

    tbler = adapter.tbler_from_sinr_db(-20.0, 2)
    assert tbler == 1.0
    assert adapter.select_mcs_from_sinr_db(-20.0) >= 3
    assert adapter.is_outage_from_sinr_db(-20.0) is True


def test_phy_tbler_lookup_builds_in_batches_and_interpolates_without_backend_calls():
    class LinearTblerAdapter(BaseLinkAdapter):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.batch_calls = 0

        def tbler_from_sinr_db_batch(self, sinr_db, mcs_index,
                                     num_allocated_re=None):
            self.batch_calls += 1
            sinr, mcs = np.broadcast_arrays(sinr_db, mcs_index)
            return np.clip(0.5 + 0.01 * mcs - 0.1 * sinr, 0.0, 1.0)

        def tbs_bits(self, mcs_index):
            return int(mcs_index) + 1

    cfg = load_config(None)
    cfg["link_abstraction"]["tbler_lookup"].update({
        "sinr_min_db": -2.0,
        "sinr_max_db": 2.0,
        "sinr_step_db": 1.0,
        "build_batch_size": 7,
    })
    backing = LinearTblerAdapter(cfg)
    lookup = PHYTblerLookup(backing, cfg)
    calls_after_build = backing.batch_calls

    actual = lookup.tbler_from_sinr_db_batch(
        np.asarray([-1.5, 0.25, 1.75]),
        np.asarray([0, 7, 28]),
    )
    expected = np.clip(
        0.5 + 0.01 * np.asarray([0, 7, 28])
        - 0.1 * np.asarray([-1.5, 0.25, 1.75]),
        0.0,
        1.0,
    )

    assert np.allclose(actual, expected)
    assert backing.batch_calls == calls_after_build
    assert lookup.lookup_status["num_table_points"] == 29 * 5
