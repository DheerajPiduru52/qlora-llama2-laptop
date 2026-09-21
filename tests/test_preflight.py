import pytest

from qlora_pipeline import audit
from qlora_pipeline.config import load_config
from qlora_pipeline.model import estimate_params_b, preflight

# Fixed C: pagefile 4-16 GB, 31.4 GB RAM (this machine's real configuration).
CAPPED = {"allocated_mb": 4096, "max_mb": 16384}


def _cs(total, limit=35.4, ram=31.4):
    return {"commit_total_gb": total, "commit_limit_gb": limit, "physical_total_gb": ram}


@pytest.fixture()
def capped(monkeypatch):
    monkeypatch.setattr(audit, "pagefile_settings", lambda: dict(CAPPED))


def test_param_estimates_match_known_sizes():
    assert 6.5 < estimate_params_b(load_config("llama2_7b_qlora.yaml").model) < 7.0  # Llama-2-7B: 6.74B
    assert 1.0 < estimate_params_b(load_config("proxy_qlora.yaml").model) < 1.2  # TinyLlama: 1.10B


def test_preflight_refuses_7b_when_c_low(monkeypatch):
    monkeypatch.setattr(audit, "c_free_gb", lambda: 10.0)
    with pytest.raises(audit.LowDiskError, match="below the 15 GB"):
        preflight(load_config("llama2_7b_qlora.yaml"), "train", job="train-qlora")


def test_preflight_allows_7b_when_c_and_commit_ok(monkeypatch, capped):
    monkeypatch.setattr(audit, "c_free_gb", lambda: 40.0)
    monkeypatch.setattr(audit, "commit_stats", lambda: _cs(10.0))
    monkeypatch.setattr(audit, "job_peak_gb", lambda _k: None)
    preflight(load_config("llama2_7b_qlora.yaml"), "train", job="train-qlora")


def test_preflight_ignores_small_proxy(monkeypatch):
    monkeypatch.setattr(audit, "c_free_gb", lambda: 1.0)
    preflight(load_config("proxy_qlora.yaml"), "smoke", job="train-qlora")  # 1.1B is below the large-model threshold


def test_pagefile_row_is_warn_only(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("QLORA_ROOT", str(tmp_path))
    base = {"taken_at": "t", "drive": "C:\\", "free_bytes": 0, "pagefile_mb": 9000,
            "locations": {lbl: {"bytes": 0} for lbl, _, _ in audit.LOCATIONS}}  # fmt: skip
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


def test_headroom_counts_pagefile_cap(monkeypatch, tmp_path, capped):
    """Current limit 35.4 - 14.9 = 20.5 would fail 23.4; the capped limit 31.4 + 16 = 47.4 gives 32.5 and passes."""
    monkeypatch.setenv("QLORA_ROOT", str(tmp_path))
    audit.record_job_peak("eval:x", 20.4, "partial", lower_bound=True)
    monkeypatch.setattr(audit, "commit_stats", lambda: _cs(14.9))
    monkeypatch.setattr(audit, "c_free_gb", lambda: 38.0)
    info = audit.require_commit_headroom("eval:x")
    assert info["limit_gb"] == pytest.approx(47.4) and info["headroom_gb"] == pytest.approx(32.5)
    assert info["pagefile_growth_left_gb"] == 12.0 and "lower-bound" in info["basis"]


def test_headroom_refuses_and_lists_processes(monkeypatch, tmp_path, capped):
    monkeypatch.setenv("QLORA_ROOT", str(tmp_path))
    monkeypatch.setattr(audit, "commit_stats", lambda: _cs(40.0))  # 47.4 - 40 = 7.4 < 18 + 3
    monkeypatch.setattr(audit, "c_free_gb", lambda: 38.0)
    monkeypatch.setattr(audit, "top_commit_processes", lambda n=5: [{"name": "brave", "pid": 1, "commit_gb": 3.2}])
    with pytest.raises(audit.CommitHeadroomError, match=r"(?s)needed 21\.0 GB .*default 18.*brave"):
        audit.require_commit_headroom("eval:x")


def test_c_must_cover_remaining_pagefile_growth(monkeypatch, tmp_path, capped):
    monkeypatch.setenv("QLORA_ROOT", str(tmp_path))
    monkeypatch.setattr(audit, "commit_stats", lambda: _cs(10.0))
    monkeypatch.setattr(audit, "c_free_gb", lambda: 21.0)  # growth left 12 GB + 10 GB margin = 22 > 21
    with pytest.raises(audit.LowDiskError, match=r"need 22\.0 GB"):
        audit.require_commit_headroom("eval:x")
    monkeypatch.setattr(audit, "c_free_gb", lambda: 22.5)
    audit.require_commit_headroom("eval:x")


def test_system_managed_pagefile_falls_back_to_current_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("QLORA_ROOT", str(tmp_path))
    monkeypatch.setattr(audit, "pagefile_settings", lambda: {"allocated_mb": 9000, "max_mb": None})
    monkeypatch.setattr(audit, "commit_stats", lambda: _cs(14.0, limit=40.0))
    info = audit.require_commit_headroom("eval:x")
    assert info["limit_gb"] == 40.0 and info["pagefile_growth_left_gb"] is None


def test_lower_bound_is_replaced_by_first_full_measurement(tmp_path, monkeypatch):
    monkeypatch.setenv("QLORA_ROOT", str(tmp_path))
    audit.record_job_peak("eval:x", 20.4, "partial run", lower_bound=True)
    e = audit.record_job_peak("eval:x", 13.1, "full run")
    assert e["peak_gb"] == 13.1 and not e["lower_bound"] and "replaced lower bound 20.4" in e["latest_note"]
    e = audit.record_job_peak("eval:x", 12.0, "smaller later full run")
    assert e["peak_gb"] == 13.1  # after that, the max of full runs is kept


def test_measured_peak_drives_need(monkeypatch, tmp_path, capped):
    monkeypatch.setenv("QLORA_ROOT", str(tmp_path))
    audit.record_job_peak("eval:x", 9.5, "full")
    monkeypatch.setattr(audit, "commit_stats", lambda: _cs(30.0))
    monkeypatch.setattr(audit, "c_free_gb", lambda: 38.0)
    info = audit.require_commit_headroom("eval:x")
    assert info["need_gb"] == 12.5 and "measured peak" in info["basis"]


def test_stale_peak_is_ignored_then_replaced(monkeypatch, tmp_path, capped):
    import json

    monkeypatch.setenv("QLORA_ROOT", str(tmp_path))
    audit.record_job_peak("eval:x", 30.16, "fallback on")
    data = json.loads(audit.peaks_path().read_text())
    data["eval:x"]["stale"] = True
    audit.peaks_path().write_text(json.dumps(data))
    monkeypatch.setattr(audit, "commit_stats", lambda: _cs(20.0))  # 47.4 - 20 = 27.4 >= 18 + 3 (stale ignored)
    monkeypatch.setattr(audit, "c_free_gb", lambda: 38.0)
    assert "default 18" in audit.require_commit_headroom("eval:x")["basis"]
    e = audit.record_job_peak("eval:x", 17.2, "probe, fallback off")
    assert e["peak_gb"] == 17.2 and "replaced stale 30.16" in e["latest_note"] and not e.get("stale")


def test_explicit_replace_overrides_max(tmp_path, monkeypatch):
    monkeypatch.setenv("QLORA_ROOT", str(tmp_path))
    audit.record_job_peak("eval:x", 20.0, "a")
    assert audit.record_job_peak("eval:x", 15.0, "b", replace=True)["peak_gb"] == 15.0


def test_model_path_accepts_local_model_dir(tmp_path):
    from qlora_pipeline.config import ModelCfg
    from qlora_pipeline.model import model_path

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert model_path(ModelCfg(name=str(tmp_path))) == tmp_path
