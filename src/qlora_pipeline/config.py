"""YAML config: `extends:` inheritance, deep merge, `--set a.b=v` overrides, pydantic validation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from qlora_pipeline.paths import configs_dir


class _Yaml12Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 booleans: only true/false. YAML 1.1 would turn `no`/`yes`/`on`/`off` into bools,
    which breaks values like `eval_strategy: no`."""


_Yaml12Loader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.copy().items()
}
_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def _yaml_load(text: str) -> Any:
    return yaml.load(text, Loader=_Yaml12Loader)  # noqa: S506 - SafeLoader subclass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelCfg(_Strict):
    name: str
    tokenizer: str | None = None
    revision: str | None = None
    pad_token: str = "<pad>"
    attn_implementation: str = "sdpa"
    allow_patterns: list[str] = Field(default_factory=lambda: ["*.safetensors", "*.json", "tokenizer.model"])

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer or self.name


class SplitCfg(_Strict):
    train: float = 0.90
    val: float = 0.05
    test: float = 0.05

    @model_validator(mode="after")
    def _sums_to_one(self) -> SplitCfg:
        if abs(self.train + self.val + self.test - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to 1")
        if min(self.train, self.val, self.test) <= 0:
            raise ValueError("split fractions must be > 0")
        return self


class DataCfg(_Strict):
    hf_dataset: str
    hf_revision: str | None = None
    hf_filename: str
    instruction_field: str = "instruction"
    output_field: str = "response"
    metadata_fields: list[str] = Field(default_factory=lambda: ["category", "intent", "flags"])
    stratify_field: str = "intent"
    system_prompt: str
    subsample_size: int = Field(6000, gt=0)
    split: SplitCfg = Field(default_factory=SplitCfg)
    max_seq_length: int = Field(1024, gt=0, le=4096)
    processed_name: str = "bitext_cs"


class QuantCfg(_Strict):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"


class LoraCfg(_Strict):
    r: int = Field(16, gt=0)
    lora_alpha: int = 32
    lora_dropout: float = Field(0.05, ge=0, lt=1)
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


class TrainCfg(_Strict):
    output_name: str = "llama2-7b-qlora"   # runs/<name>/ checkpoints, artifacts/adapters/<name>/ best adapter
    num_train_epochs: float = 3
    max_steps: int = -1                    # > 0 overrides epochs (smoke test)
    per_device_train_batch_size: int = Field(4, gt=0)
    per_device_eval_batch_size: int = Field(4, gt=0)
    effective_batch_size: int = Field(32, gt=0)  # grad accumulation = effective / per_device
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(0.03, ge=0, lt=1)
    max_grad_norm: float = 0.3
    weight_decay: float = 0.0
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    gradient_checkpointing: bool = True
    eval_strategy: Literal["epoch", "steps", "no"] = "epoch"
    eval_steps: int | None = None
    save_steps: int = Field(50, gt=0)      # periodic checkpoints for resume (laptop throttling)
    save_total_limit: int = Field(2, gt=0)
    logging_steps: int = Field(5, gt=0)
    dataloader_num_workers: int = Field(0, ge=0)
    pad_to_multiple_of: int | None = 8
    resume: str = "auto"                   # auto | none | <checkpoint path>
    telemetry_interval_s: float = 2.0      # NVML sampling period

    @model_validator(mode="after")
    def _accum_divides(self) -> TrainCfg:
        if self.effective_batch_size % self.per_device_train_batch_size:
            raise ValueError("effective_batch_size must be a multiple of per_device_train_batch_size")
        return self

    @property
    def grad_accum(self) -> int:
        return self.effective_batch_size // self.per_device_train_batch_size


class EvalCfg(_Strict):
    weights: str | None = None            # adapter or full-model dir; null -> artifacts/adapters/<train.output_name>
    quantized: bool | Literal["auto"] = "auto"  # auto: 4-bit iff method == qlora (as trained)
    max_new_tokens: int = Field(512, gt=0)
    gen_batch_size: int = Field(8, gt=0)   # 16 fragmented the KV-cache allocator past 16 GB VRAM (probe 2026-11-19)
    ppl_batch_size: int = Field(8, gt=0)
    test_limit: int | None = None         # null = full test split
    test_select: Literal["first", "longest"] = "first"  # which rows test_limit keeps (longest = memory worst case)
    max_reserved_gb: float | None = 14.0  # hard cap on PyTorch reserved VRAM (set_per_process_memory_fraction)
    continue_on_oom: bool = False         # probe mode: record a stage OOM and continue instead of failing
    compare_base: bool = True             # also score the untuned base (adapter disabled)
    rouge_tolerance: float = 1.0          # parity tolerance (ROUGE-L points) used by Phase 4
    lm_eval_tasks: list[str] = Field(default_factory=lambda: ["arc_easy", "hellaswag"])
    lm_eval_limit: int | None = 200
    lm_eval_batch_size: int = Field(8, gt=0)
    forgetting_tolerance: float = 2.0     # flag if tuned acc drops more than this (points) vs base


class MergeCfg(_Strict):
    dtype: Literal["float16", "bfloat16"] = "float16"   # spec: fp16 merge on CPU (Llama 2 ships fp16 weights)
    validate_n: int = Field(50, gt=0)                   # first N test rows, greedy, merged vs adapter
    rouge_tolerance: float = 0.5                         # |ROUGE-L merged - adapter| must be <= this
    registry_name: str = "qlora-cs-llama2-7b-chat"
    alias: str = "champion"


class SafetyCfg(_Strict):
    min_c_free_gb: float = 15.0             # pre-flight: refuse large-model GPU jobs below this C: free space
    large_model_min_params_b: float = 3.0   # "large" = estimated params >= this (billions); 7B is large, 1.1B is not
    commit_default_peak_gb: float = 18.0    # assumed job peak commit when not yet measured
    commit_margin_gb: float = 3.0           # headroom required on top of the job's peak


class Config(_Strict):
    experiment: str
    seed: int = 42
    method: Literal["qlora", "lora", "full"] = "qlora"
    model: ModelCfg
    data: DataCfg
    quant: QuantCfg = Field(default_factory=QuantCfg)
    lora: LoraCfg = Field(default_factory=LoraCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    safety: SafetyCfg = Field(default_factory=SafetyCfg)
    merge: MergeCfg = Field(default_factory=MergeCfg)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_chain(path: Path, seen: set[Path] | None = None) -> dict:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular `extends` at {path}")
    seen.add(path)
    raw = _yaml_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent:
        return _deep_merge(_load_yaml_chain(path.parent / parent, seen), raw)
    return raw


def parse_override(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ValueError(f"override must be key.path=value, got {item!r}")
    key, val = item.split("=", 1)
    return key.strip().split("."), _yaml_load(val)  # YAML 1.2 typing: 6000 -> int, true -> bool, [a,b] -> list


def apply_overrides(d: dict, overrides: list[str]) -> dict:
    d = json.loads(json.dumps(d))  # deep copy
    for item in overrides or []:
        keys, val = parse_override(item)
        cur = d
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
            if not isinstance(cur, dict):
                raise ValueError(f"cannot set {item!r}: {k} is not a mapping")
        cur[keys[-1]] = val
    return d


def resolve_config_path(name: str | Path) -> Path:
    p = Path(name)
    if p.is_file():
        return p
    cand = configs_dir() / p
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"config not found: {name} (also looked in {configs_dir()})")


def load_config(name: str | Path = "base.yaml", overrides: list[str] | None = None) -> Config:
    raw = _load_yaml_chain(resolve_config_path(name))
    return Config.model_validate(apply_overrides(raw, overrides or []))
