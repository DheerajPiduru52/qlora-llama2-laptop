import pytest

from qlora_pipeline.benchmark import METHODS, check_identical, pct_reduction, seven_b_estimates
from qlora_pipeline.config import load_config


def test_proxy_configs_identical_where_required():
    cfgs = {m: load_config(c) for m, c in METHODS}
    same = check_identical(cfgs)
    assert same["train.effective_batch_size"] == 32 and same["data.max_seq_length"] == 1024
    assert {c.method for c in cfgs.values()} == {"full", "lora", "qlora"}
    assert cfgs["full"].train.learning_rate == 2e-5 and cfgs["qlora"].train.learning_rate == 2e-4


def test_check_identical_rejects_mismatch():
    cfgs = {m: load_config(c) for m, c in METHODS}
    cfgs["lora"] = load_config("proxy_lora.yaml", ["seed=7"])
    with pytest.raises(ValueError, match="seed"):
        check_identical(cfgs)


def test_seven_b_estimates_formula():
    est = seven_b_estimates({"peak_mem_allocated_gb": 8.0, "mem_after_load_gb": 4.35})
    p_gb = est["params_b"] * 1e9 / 1024**3
    assert est["anchor"]["activations_plus_lora_state_gb"] == pytest.approx(3.65)
    assert est["bf16_lora"]["total_gb"] == pytest.approx(2 * p_gb + 3.65, abs=0.02)
    assert est["full_ft_8bit_adam"]["total_gb"] == pytest.approx(6 * p_gb + 3.65, abs=0.02)
    assert est["full_ft_fp32_adam"]["total_gb"] == pytest.approx(16 * p_gb + 3.65, abs=0.02)


def test_pct_reduction():
    assert pct_reduction(25, 100) == 75.0 and pct_reduction(110, 100) == -10.0
