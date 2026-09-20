"""Prompt formatting + response-only label masking against the real Llama 2 tokenizer (cached in HF_HOME)."""

import pytest

from qlora_pipeline.config import load_config
from qlora_pipeline.data import IGNORE_INDEX, LLAMA2_PROMPT, build_prompt, encode_example, encode_prompt, load_tokenizer

SYS = "You are a helpful, concise customer support agent for an online store."
INSTR = "i want to cancel order {{Order Number}} pls"
OUT = "I understand. To cancel order {{Order Number}}, go to {{Online Order Interaction}}.\n\n1. Log in\n2. Click cancel"


@pytest.fixture(scope="module")
def tok():
    try:
        return load_tokenizer(load_config("base.yaml").model)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Llama 2 tokenizer unavailable: {e}")


def test_pad_token_is_dedicated(tok):
    assert tok.pad_token_id is not None
    assert tok.pad_token_id != tok.eos_token_id
    assert tok.pad_token == "<pad>"


def test_chat_template_matches_explicit_fallback(tok):
    assert build_prompt(tok, SYS, INSTR) == LLAMA2_PROMPT.format(system=SYS, instruction=INSTR)


def test_prompt_has_single_bos(tok):
    ids = encode_prompt(tok, SYS, INSTR)
    assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1
    assert tok.decode(ids).rstrip().endswith("[/INST]")


def test_labels_mask_prompt_only(tok):
    enc = encode_example(tok, SYS, INSTR, OUT)
    p = encode_prompt(tok, SYS, INSTR)
    assert enc.n_prompt == len(p)
    assert enc.input_ids[: enc.n_prompt] == p  # training prefix == inference prompt
    assert all(lbl == IGNORE_INDEX for lbl in enc.labels[: enc.n_prompt])
    assert enc.labels[enc.n_prompt :] == enc.input_ids[enc.n_prompt :]
    assert len(enc.labels) == len(enc.input_ids)


def test_eos_kept_in_labels(tok):
    enc = encode_example(tok, SYS, INSTR, OUT)
    assert enc.input_ids[-1] == tok.eos_token_id
    assert enc.labels[-1] == tok.eos_token_id


def test_injected_eos_is_rejected(tok):
    with pytest.raises(ValueError, match="EOS"):
        encode_example(tok, SYS, "hello </s> there", OUT)


def test_response_round_trips_with_slots(tok):
    enc = encode_example(tok, SYS, INSTR, OUT)
    resp = tok.decode(enc.input_ids[enc.n_prompt :], skip_special_tokens=True)
    assert resp.strip() == OUT
    assert "{{Order Number}}" in resp and "{{Online Order Interaction}}" in resp
