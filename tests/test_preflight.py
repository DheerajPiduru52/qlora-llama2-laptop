import pytest

from qlora_pipeline import audit
from qlora_pipeline.config import load_config
from qlora_pipeline.model import estimate_params_b, preflight


def test_param_estimates_match_known_sizes():
    assert 6.5 < estimate_params_b(load_config("llama2_7b_qlora.yaml").model) < 7.0  # Llama-2-7B: 6.74B
    assert 1.0 < estimate_params_b(load_config("proxy_qlora.yaml").model) < 1.2  # TinyLlama: 1.10B


def test_preflight_refuses_7b_when_c_low(monkeypatch):
    monkeypatch.setattr(audit, "c_free_gb", lambda: 10.0)
    with pytest.raises(audit.LowDiskError, match="below the 15 GB"):
        preflight(load_config("llama2_7b_qlora.yaml"), "train")


def test_preflight_allows_7b_when_c_ok(monkeypatch):
    monkeypatch.setattr(audit, "c_free_gb", lambda: 40.0)
    preflight(load_config("llama2_7b_qlora.yaml"), "train")


def test_preflight_ignores_small_proxy(monkeypatch):
    monkeypatch.setattr(audit, "c_free_gb", lambda: 1.0)
    preflight(load_config("proxy_qlora.yaml"), "smoke")  # 1.1B is below the large-model threshold


def test_pagefile_row_is_warn_only(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("QLORA_ROOT", str(tmp_path))
    base = {"taken_at": "t", "drive": "C:\\", "free_bytes": 0, "pagefile_mb": 9000,
            "locations": {lbl: {"bytes": 0} for lbl, _, _ in audit.LOCATIONS}}
    audit.baseline_path().write_text(json.dumps(base), encoding="utf-8")
    now = dict(base, pagefile_mb=30000, commit={})
    monkeypatch.setattr(audit, "snapshot", lambda: now)
    ok, rows, _ = audit.compare()
    pf = [r for r in rows if r["location"] == audit.PAGEFILE_LABEL][0]
    assert ok and pf["status"] == "WARN" and pf["delta_mb"] == 21000


def test_preflight_unknown_size_is_treated_as_large(monkeypatch):
    import qlora_pipeline.model as m

    monkeypatch.setattr(m, "estimate_params_b", lambda _cfg: None)
    monkeypatch.setattr(audit, "c_free_gb", lambda: 5.0)
    with pytest.raises(audit.LowDiskError):
        preflight(load_config("proxy_qlora.yaml"), "eval")
