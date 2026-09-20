"""YAML config: `extends:` inheritance, deep merge, `--set a.b=v` overrides, pydantic validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from qlora_pipeline.paths import configs_dir


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


class Config(_Strict):
    experiment: str
    seed: int = 42
    model: ModelCfg
    data: DataCfg

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
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent:
        return _deep_merge(_load_yaml_chain(path.parent / parent, seen), raw)
    return raw


def parse_override(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ValueError(f"override must be key.path=value, got {item!r}")
    key, val = item.split("=", 1)
    return key.strip().split("."), yaml.safe_load(val)  # YAML typing: 6000 -> int, true -> bool, [a,b] -> list


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
