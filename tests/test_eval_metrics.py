import math

from qlora_pipeline.evaluate import rouge_l, slot_stats


def test_rouge_l_identical_and_disjoint():
    assert rouge_l(["the order is shipped"], ["the order is shipped"])[0] == 100.0
    assert rouge_l(["banana"], ["the order is shipped"])[0] == 0.0


def test_slot_stats_recall_and_invented():
    refs = ["Use {{Order Number}} and {{Website URL}}", "no slots here", "call {{Customer Support Phone Number}}"]
    preds = ["Use {{Order Number}} at {{Invented Slot}}", "fine {{Extra}}", "call {{Customer Support Phone Number}}"]
    s = slot_stats(preds, refs)
    assert math.isclose(s["slot_recall_pct"], 100 * 2 / 3)
    assert math.isclose(s["rows_all_slots_kept_pct"], 50.0)
    assert s["rows_with_slots"] == 2 and s["invented_slots"] == 2


def test_slot_stats_counts_duplicates():
    s = slot_stats(["{{A}}"], ["{{A}} then {{A}}"])
    assert math.isclose(s["slot_recall_pct"], 50.0)


def _v(tuned, base):
    return {"tuned": {"lm_eval": tuned}, "base": {"lm_eval": base}}


def test_forgetting_flags_only_real_drops():
    from qlora_pipeline.config import load_config
    from qlora_pipeline.evaluate import forgetting_verdict

    cfg = load_config("base.yaml")  # tolerance 2.0
    noisy = forgetting_verdict(cfg, _v({"t": {"acc_norm": 60.0, "acc_norm_stderr": 3.0}},
                                       {"t": {"acc_norm": 63.0, "acc_norm_stderr": 3.0}}))  # -3 < 1.5*4.24
    assert noisy["flags"] == []
    real = forgetting_verdict(cfg, _v({"t": {"acc_norm": 50.0, "acc_norm_stderr": 3.0}},
                                      {"t": {"acc_norm": 63.0, "acc_norm_stderr": 3.0}}))
    assert len(real["flags"]) == 1 and real["deltas"]["t"]["delta"] == -13.0
    small_se = forgetting_verdict(cfg, _v({"t": {"acc": 60.0}}, {"t": {"acc": 61.0}}))
    assert small_se["flags"] == []  # within tolerance


def test_pct_handles_na():
    from qlora_pipeline.evaluate import _pct

    assert _pct(0.5) == 50.0 and math.isnan(_pct("N/A"))
