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
        preflight(load_config("llama2_7b_qlora.yaml"), "train", job="train-qlora")


def test_preflight_allows_7b_when_c_and_commit_ok(monkeypatch):
    monkeypatch.setattr(audit, "c_free_gb", lambda: 40.0)
    monkeypatch.setattr(audit, "commit_stats", lambda: {"commit_total_gb": 10.0, "commit_limit_gb": 47.0})
    monkeypatch.setattr(audit, "job_peak_gb", lambda _k: None)
    preflight(load_config("llama2_7b_qlora.yaml"), "train", job="train-qlora")


def test_preflight_ignores_small_proxy(monkeypatch):
    monkeypatch.setattr(audit, "c_free_gb", lambda: 1.0)
    preflight(load_config("proxy_qlora.yaml"), "smoke", job="train-qlora")  # 1.1B is below the large-model threshold


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
        preflight(load_config("proxy_qlora.yaml"), "eval", job="eval")


def test_commit_headroom_refuses_and_lists_processes(monkeypatch, tmp_path):
    monkeypatch.setenv("QLORA_ROOT", str(tmp_path))
    monkeypatch.setattr(audit, "commit_stats", lambda: {"commit_total_gb": 40.0, "commit_limit_gb": 47.0})
    monkeypatch.setattr(audit, "top_commit_processes",
                        lambda n=5: [{"name": "brave", "pid": 1, "commit_gb": 3.2}])  # fmt: skip
    with pytest.raises(audit.CommitHeadroomError, match=r"(?s)needed 21\.0 GB .*default 18.*brave"):
        audit.require_commit_headroom("eval:x")  # 7 GB headroom < 18 + 3


def test_commit_headroom_uses_measured_peak(monkeypatch, tmp_path):
    monkeypatch.setenv("QLORA_ROOT", str(tmp_path))
    audit.record_job_peak("eval:x", 9.5, "test")
    audit.record_job_peak("eval:x", 7.0, "smaller later run")  # max is kept
    assert audit.job_peak_gb("eval:x")["peak_gb"] == 9.5
    monkeypatch.setattr(audit, "commit_stats", lambda: {"commit_total_gb": 30.0, "commit_limit_gb": 43.0})
    info = audit.require_commit_headroom("eval:x")  # 13 GB headroom >= 9.5 + 3
    assert info["need_gb"] == 12.5 and "measured peak" in info["basis"]
