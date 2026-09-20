import pytest
import yaml
from pydantic import ValidationError

from qlora_pipeline.config import apply_overrides, load_config


def test_base_loads_and_matches_spec():
    cfg = load_config("base.yaml")
    assert cfg.experiment == "qlora-cs-practice"
    assert cfg.seed == 42
    assert cfg.model.name == "meta-llama/Llama-2-7b-chat-hf"
    assert cfg.data.subsample_size == 6000
    assert cfg.data.stratify_field == "intent"
    assert cfg.model.pad_token != "</s>"


def test_overrides_are_typed():
    cfg = load_config("base.yaml", ["data.subsample_size=3000", "seed=7", "model.name=foo/bar"])
    assert cfg.data.subsample_size == 3000 and isinstance(cfg.data.subsample_size, int)
    assert cfg.seed == 7 and cfg.model.name == "foo/bar"


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        load_config("base.yaml", ["data.subsampel_size=10"])


def test_split_must_sum_to_one():
    with pytest.raises(ValidationError):
        load_config("base.yaml", ["data.split={train: 0.9, val: 0.2, test: 0.05}"])


def test_extends_deep_merges(tmp_path):
    base = load_config("base.yaml").to_dict()
    (tmp_path / "parent.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    (tmp_path / "child.yaml").write_text("extends: parent.yaml\ndata:\n  max_seq_length: 2048\n", encoding="utf-8")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.data.max_seq_length == 2048
    assert cfg.data.subsample_size == 6000  # sibling keys survive the merge


def test_apply_overrides_does_not_mutate_input():
    d = {"a": {"b": 1}}
    apply_overrides(d, ["a.b=2"])
    assert d == {"a": {"b": 1}}


def test_fingerprint_changes_with_config():
    assert load_config("base.yaml").fingerprint() != load_config("base.yaml", ["seed=1"]).fingerprint()


def test_yaml12_booleans_no_is_a_string():
    cfg = load_config("base.yaml", ["train.eval_strategy=no", "train.bf16=false"])
    assert cfg.train.eval_strategy == "no" and cfg.train.bf16 is False


def test_grad_accum_derived():
    cfg = load_config("llama2_7b_qlora.yaml")
    assert cfg.train.grad_accum * cfg.train.per_device_train_batch_size == 32


def test_eval_quantized_accepts_bool_and_auto():
    assert load_config("base.yaml", ["eval.quantized=true"]).eval.quantized is True
    assert load_config("base.yaml", ["eval.quantized=false"]).eval.quantized is False
    assert load_config("base.yaml").eval.quantized == "auto"
