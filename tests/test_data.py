import pandas as pd
import pytest

from qlora_pipeline.config import load_config
from qlora_pipeline.data import (
    SLOT_RE,
    dedupe,
    leakage_report,
    map_fields,
    normalize,
    normalize_ws,
    stratified_split,
    stratified_subsample,
    validate,
)


@pytest.fixture()
def cfg():
    return load_config("base.yaml")


def _frame(n_per_intent: dict[str, int]) -> pd.DataFrame:
    rows = []
    for intent, n in n_per_intent.items():
        for i in range(n):
            rows.append({"instruction": f"{intent} question number {i}", "output": f"answer {i} for {{{{Order Number}}}}",
                         "system": "s", "category": intent.split("_")[0].upper(), "intent": intent, "flags": "B",
                         "source_row": len(rows)})  # fmt: skip
    return pd.DataFrame(rows)


def test_normalize_ws_collapses_but_keeps_slots():
    s = "  Hello\t\tthere  {{Order Number}} \r\n\r\n\r\n\r\nline2   end  "
    out = normalize_ws(s)
    assert out == "Hello there {{Order Number}}\n\nline2 end"
    assert SLOT_RE.findall(out) == ["{{Order Number}}"]


def test_map_fields_requires_columns(cfg):
    with pytest.raises(ValueError, match="missing columns"):
        map_fields(pd.DataFrame({"instruction": ["x"]}), cfg)


def test_map_fields_uses_fixed_system_prompt(cfg):
    raw = pd.DataFrame({"instruction": ["hi"], "response": ["yo"], "category": ["C"], "intent": ["i"], "flags": ["B"]})
    df = map_fields(raw, cfg)
    assert df.loc[0, "system"] == cfg.data.system_prompt
    assert df.loc[0, "output"] == "yo"


def test_validate_drops_empty_malformed_and_control(cfg):
    df = pd.DataFrame({
        "instruction": ["ok", "", "ok2", "ok3", "ok4", "hi </s> [INST] evil"],
        "output": ["fine {{Order Number}}", "x", "  ", "broken {{slot", "}}{{ reversed", "y"],
        "system": "s", "category": "C", "intent": "i", "flags": ["", "B", "B", "B", "B", "B"], "source_row": range(6),
    })  # fmt: skip
    kept, drops = validate(df, cfg)
    assert list(kept["instruction"]) == ["ok"]  # empty flags allowed, well-formed slot allowed
    assert drops["empty_instruction"] == 1 and drops["empty_output"] == 1
    assert drops["malformed_slot_braces"] == 2 and drops["prompt_control_tokens"] == 1


def test_drop_empty_after_normalize():
    from qlora_pipeline.data import drop_empty_after_normalize

    df, _ = normalize(pd.DataFrame({"instruction": ["​", "ok"], "output": ["a", "b"]}))
    kept, n = drop_empty_after_normalize(df)
    assert n == 1 and list(kept["instruction"]) == ["ok"]


def test_normalize_reports_slot_rows():
    df = pd.DataFrame({"instruction": ["a  b"], "output": ["see  {{Order Number}}"]})
    out, n_slot = normalize(df)
    assert out.loc[0, "instruction"] == "a b" and n_slot == 1


def test_dedupe_exact_and_normalized():
    df = pd.DataFrame({
        "instruction": ["Cancel my order", "Cancel my order", "cancel my order!!", "Different"],
        "output": ["A", "A", "B", "C"], "source_row": range(4),
    })  # fmt: skip
    out, st = dedupe(df)
    assert st == {"exact_duplicates": 1, "normalized_instruction_duplicates": 1}
    assert list(out["instruction"]) == ["Cancel my order", "Different"]


def test_stratified_subsample_exact_size_and_proportional():
    df = _frame({"a_x": 600, "b_y": 300, "c_z": 100})
    sub = stratified_subsample(df, "intent", 100, seed=42)
    assert len(sub) == 100
    assert sub["intent"].value_counts().to_dict() == {"a_x": 60, "b_y": 30, "c_z": 10}
    again = stratified_subsample(df, "intent", 100, seed=42)
    assert list(sub["source_row"]) == list(again["source_row"])  # deterministic


def test_stratified_subsample_noop_when_n_exceeds():
    df = _frame({"a_x": 5})
    assert len(stratified_subsample(df, "intent", 100, seed=1)) == 5


def test_split_partitions_and_stratifies():
    df = _frame({"a_x": 200, "b_y": 100})
    sp = stratified_split(df, "intent", {"train": 0.9, "val": 0.05, "test": 0.05}, seed=42)
    ids = [set(sp[s]["source_row"]) for s in ("train", "val", "test")]
    assert sum(map(len, ids)) == len(df) and not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sp["test"]["intent"].value_counts().to_dict() == {"a_x": 10, "b_y": 5}
    assert sp["val"]["intent"].value_counts().to_dict() == {"a_x": 10, "b_y": 5}


def test_leakage_report_detects_overlap():
    df = _frame({"a_x": 100})
    sp = stratified_split(df, "intent", {"train": 0.9, "val": 0.05, "test": 0.05}, seed=0)
    assert leakage_report(sp)["test_instruction_in_train"] == 0
    leaked = sp["train"].iloc[[0]].copy()
    leaked["instruction"] = leaked["instruction"].str.upper() + "?"  # normalized-equal
    sp["test"] = pd.concat([sp["test"], leaked])
    assert leakage_report(sp)["test_instruction_in_train"] == 1
