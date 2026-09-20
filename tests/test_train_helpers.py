import pytest

from qlora_pipeline.config import load_config
from qlora_pipeline.train import resume_fingerprint, smoke_verdict


def test_smoke_verdict_pass_and_fail():
    ok, first, last = smoke_verdict([2.0] * 5 + [1.5] * 5 + [1.0] * 5)
    assert ok and first == 2.0 and last == 1.0
    assert not smoke_verdict([1.0] * 5 + [2.0] * 5)[0]


def test_smoke_verdict_needs_enough_points():
    with pytest.raises(ValueError):
        smoke_verdict([1.0, 0.9])


def test_resume_fingerprint_ignores_resume_but_not_hparams():
    base = resume_fingerprint(load_config("llama2_7b_qlora.yaml"))
    assert resume_fingerprint(load_config("llama2_7b_qlora.yaml", ["train.resume=none"])) == base
    assert resume_fingerprint(load_config("llama2_7b_qlora.yaml", ["train.learning_rate=1.0e-4"])) != base
