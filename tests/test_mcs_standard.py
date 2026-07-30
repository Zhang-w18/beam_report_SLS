import numpy as np
import pytest

from beam_sls.config import load_config
from beam_sls.link_adaptation import BaseLinkAdapter, SionnaSYSAdapter, make_link_adapter
from beam_sls.mcs import MCS_TABLE, rate_mbps_from_mcs, tbs_bits_from_mcs


def test_pdsch_table_1_mcs_10_has_3gpp_semantics():
    entry = MCS_TABLE[10]

    assert entry.q_m == 4
    assert entry.code_rate_x1024 == 340
    assert np.isclose(entry.code_rate, 340 / 1024)
    assert np.isclose(entry.spectral_efficiency, 1.328125)


def test_tbs_and_rate_use_ts_38214_quantization():
    kwargs = {
        "mcs_index": 10,
        "num_prbs": 132,
        "num_symbols": 12,
        "dmrs_overhead_re_per_prb": 18,
        "num_layers": 1,
    }

    assert tbs_bits_from_mcs(**kwargs) == 22032
    assert np.isclose(
        rate_mbps_from_mcs(**kwargs, slot_duration_ms=0.125),
        176.256,
    )


def test_link_adaptation_fallback_mode_is_rejected():
    cfg = load_config(None)
    cfg["link_abstraction"]["mode"] = "fallback_precomputed_table"

    with pytest.raises(ValueError, match="Unsupported link_abstraction.mode"):
        make_link_adapter(cfg)


def test_sionna_adapter_rate_and_tbs_use_sionna_nr_utilities():
    cfg = load_config(None)
    adapter = object.__new__(SionnaSYSAdapter)
    BaseLinkAdapter.__init__(adapter, cfg)
    calls = {}

    def decode_mcs_index(mcs_index, **kwargs):
        calls["decode"] = (mcs_index, kwargs)
        return 4, 340 / 1024

    def calculate_tb_size(**kwargs):
        calls["tbs"] = kwargs
        return 22032, 0, 0, 0, 0, 0

    adapter.decode_mcs_index = decode_mcs_index
    adapter.calculate_tb_size = calculate_tb_size

    assert adapter.tbs_bits(10) == 22032
    assert np.isclose(adapter.rate_mbps(10), 176.256)
    assert calls["decode"][0] == 10
    assert calls["decode"][1] == {"table_index": 1, "is_pusch": False}
    assert calls["tbs"]["modulation_order"] == 4
    assert np.isclose(calls["tbs"]["target_coderate"], 340 / 1024)
