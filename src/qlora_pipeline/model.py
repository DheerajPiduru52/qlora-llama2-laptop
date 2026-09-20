"""Model construction for qlora | lora | full, shared by train / eval / merge / serve.

Pad token handling: the tokenizer gains a dedicated '<pad>' (id 32000). Embeddings are resized to a
multiple of 64 and the NEW rows are zero-initialised (not random), so every reload of the base model
reproduces them bit-for-bit. LoRA adapters are therefore saved WITHOUT embedding layers, and generation
suppresses every id that is not a real vocabulary token.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from qlora_pipeline.config import Config, ModelCfg

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def model_path(model_cfg: ModelCfg) -> Path:
    """Local snapshot, downloading only allow_patterns (never the duplicate .bin weights)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_cfg.name, revision=model_cfg.revision, allow_patterns=model_cfg.allow_patterns))


def bnb_config(cfg: Config):
    from transformers import BitsAndBytesConfig

    q = cfg.quant
    return BitsAndBytesConfig(
        load_in_4bit=q.load_in_4bit,
        bnb_4bit_quant_type=q.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=DTYPES[q.bnb_4bit_compute_dtype],
    )


def resize_for_pad(model, tok) -> int:
    """Grow embeddings to fit the tokenizer (multiple of 64); zero the new rows deterministically."""
    old = model.get_input_embeddings().weight.shape[0]
    if len(tok) <= old:
        return old
    model.resize_token_embeddings(len(tok), pad_to_multiple_of=64, mean_resizing=False)
    with torch.no_grad():
        model.get_input_embeddings().weight[old:].zero_()
        out = model.get_output_embeddings()
        if out is not None and out.weight.shape[0] > old:
            out.weight[old:].zero_()
    model.config.pad_token_id = tok.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tok.pad_token_id
    return model.get_input_embeddings().weight.shape[0]


def suppressed_token_ids(model, tok) -> list[int]:
    """Ids generation must never emit: the pad token and the embedding-padding rows past the vocab."""
    rows = model.get_input_embeddings().weight.shape[0]
    return [tok.pad_token_id, *range(len(tok), rows)]


def load_base(cfg: Config, tok, quantized: bool, device_map: Any = None):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "attn_implementation": cfg.model.attn_implementation,
        "device_map": device_map if device_map is not None else {"": 0},
    }
    if quantized:
        kwargs["quantization_config"] = bnb_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(model_path(cfg.model), **kwargs)
    resize_for_pad(model, tok)
    model.config.use_cache = False
    return model


def lora_config(cfg: Config):
    from peft import LoraConfig

    lc = cfg.lora
    return LoraConfig(
        r=lc.r, lora_alpha=lc.lora_alpha, lora_dropout=lc.lora_dropout, bias=lc.bias,
        target_modules=list(lc.target_modules), task_type="CAUSAL_LM",
    )  # fmt: skip


def build_for_training(cfg: Config, tok):
    """Returns the trainable model for cfg.method. Gradient checkpointing is configured here (non-reentrant)."""
    gc_kwargs = {"use_reentrant": False}
    if cfg.method == "qlora":
        from peft import get_peft_model, prepare_model_for_kbit_training

        model = load_base(cfg, tok, quantized=True)
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.train.gradient_checkpointing,
            activation_checkpointing_kwargs=gc_kwargs,
        )
        return get_peft_model(model, lora_config(cfg))
    if cfg.method == "lora":
        from peft import get_peft_model

        model = load_base(cfg, tok, quantized=False)
        if cfg.train.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
            model.enable_input_require_grads()
        return get_peft_model(model, lora_config(cfg))
    if cfg.method == "full":
        model = load_base(cfg, tok, quantized=False)
        if cfg.train.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        for p in model.parameters():
            p.requires_grad_(True)
        return model
    raise ValueError(f"unknown method {cfg.method}")


def param_counts(model) -> tuple[int, int]:
    """(trainable, total). PEFT's counter accounts for packed 4-bit params."""
    if hasattr(model, "get_nb_trainable_parameters"):
        return model.get_nb_trainable_parameters()
    total = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad), total


def save_adapter_or_model(model, tok, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "peft_config"):
        model.save_pretrained(out_dir, save_embedding_layers=False)
    else:
        model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
