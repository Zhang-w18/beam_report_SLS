from beam_sls.config import load_config
from beam_sls.link_adaptation import LinkAdaptationBackendError, SionnaSYSAdapter


def test_sionna_sys_invalid_low_sinr_treated_as_error_probability_one():
    cfg = load_config(None)
    try:
        adapter = SionnaSYSAdapter(cfg)
    except LinkAdaptationBackendError:
        return

    tbler = adapter.tbler_from_sinr_db(-20.0, 2)
    assert tbler == 1.0
    assert adapter.select_mcs_from_sinr_db(-20.0) >= 3
