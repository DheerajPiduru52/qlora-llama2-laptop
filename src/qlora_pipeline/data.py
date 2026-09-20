"""Phase 1 data pipeline + the single prompt format shared by training, eval and serving.

download -> map fields -> validate -> normalize whitespace -> dedupe (exact + normalized) ->
stratified subsample -> token-length filter -> 90/5/5 stratified split -> leakage check ->
jsonl + manifest + data card -> MLflow.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qlora_pipeline.config import Config, ModelCfg
from qlora_pipeline.paths import data_processed_dir, data_raw_dir, ensure_dir, reports_dir

IGNORE_INDEX = -100
SPLITS = ("train", "val", "test")
SLOT_RE = re.compile(r"\{\{[^{}]+\}\}")

# Used only when the tokenizer ships no chat template. Matches the Llama 2 chat format exactly.
LLAMA2_PROMPT = "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"


# --------------------------------------------------------------------------- prompt format


def load_tokenizer(model_cfg: ModelCfg, padding_side: str = "right"):
    """Tokenizer with a dedicated pad token (never EOS). Caller resizes model embeddings if it grew."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name, revision=model_cfg.revision)
    if tok.pad_token is None or tok.pad_token_id == tok.eos_token_id:
        tok.add_special_tokens({"pad_token": model_cfg.pad_token})
    if tok.pad_token_id == tok.eos_token_id:
        raise ValueError("pad token must differ from EOS")
    tok.padding_side = padding_side
    return tok


def build_prompt(tok, system: str, instruction: str) -> str:
    """Prompt text up to and including [/INST]. Includes the literal BOS string '<s>'."""
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": instruction}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return LLAMA2_PROMPT.format(system=system, instruction=instruction)


def encode_prompt(tok, system: str, instruction: str) -> list[int]:
    """Token ids fed to the model at inference time."""
    return tok(build_prompt(tok, system, instruction), add_special_tokens=False)["input_ids"]


@dataclass
class Encoded:
    input_ids: list[int]
    labels: list[int]
    n_prompt: int

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)


def encode_example(tok, system: str, instruction: str, output: str) -> Encoded:
    """Prompt + ' ' + response + EOS, tokenized jointly; prompt tokens masked with -100.

    Joint tokenization, then a prefix check, guarantees the trained boundary is exactly the token
    sequence the model sees at inference (encode_prompt), so serving and training never disagree.
    """
    prompt = build_prompt(tok, system, instruction)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full = tok(prompt + " " + output, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if full[: len(p_ids)] != p_ids:
        raise ValueError("prompt tokens are not a prefix of prompt+response tokens")
    if full.count(tok.bos_token_id) != 1 or full[0] != tok.bos_token_id:
        raise ValueError("expected exactly one BOS at position 0")
    if full.count(tok.eos_token_id) != 1:
        raise ValueError("expected exactly one EOS, at the end (special-token text in the example?)")
    labels = [IGNORE_INDEX] * len(p_ids) + full[len(p_ids) :]
    return Encoded(input_ids=full, labels=labels, n_prompt=len(p_ids))


# --------------------------------------------------------------------------- text cleaning

_WS = re.compile(r"[ \t  -​　]+")


def normalize_ws(text: str) -> str:
    """Unify newlines, collapse runs of spaces/tabs, strip line ends, cap blank lines at one."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def norm_key(text: str) -> str:
    """Aggressive key for near-duplicate detection: lowercase, punctuation-insensitive, slots kept."""
    return re.sub(r"[^\w{}]+", " ", text.lower()).strip()


# --------------------------------------------------------------------------- stages


def download_raw(cfg: Config) -> Path:
    from huggingface_hub import hf_hub_download

    dest = ensure_dir(data_raw_dir() / cfg.data.hf_dataset.replace("/", "__"))
    return Path(
        hf_hub_download(
            cfg.data.hf_dataset, cfg.data.hf_filename, repo_type="dataset",
            revision=cfg.data.hf_revision, local_dir=dest,
        )
    )  # fmt: skip


def map_fields(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = cfg.data
    needed = [d.instruction_field, d.output_field, *d.metadata_fields]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"source is missing columns {missing}; has {list(raw.columns)}")
    df = pd.DataFrame(
        {"instruction": raw[d.instruction_field], "output": raw[d.output_field], "system": d.system_prompt}
    )
    for m in d.metadata_fields:
        df[m] = raw[m]
    df["source_row"] = raw.index
    return df


def validate(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, int]]:
    drops: dict[str, int] = {}
    keep = pd.Series(True, index=df.index)
    for col in ["instruction", "output", cfg.data.stratify_field, *cfg.data.metadata_fields]:
        if col == "flags":  # flags may legitimately be empty
            continue
        bad = df[col].isna() | (df[col].astype(str).str.strip() == "")
        drops[f"empty_{col}"] = int((bad & keep).sum())
        keep &= ~bad
    # Every '{{' / '}}' must belong to a well-formed {{slot}}; anything else would corrupt slot checks.
    def _bad_braces(s: str) -> bool:
        rest = SLOT_RE.sub("", s)
        return "{{" in rest or "}}" in rest

    bad_braces = df["output"].astype(str).map(_bad_braces)
    drops["malformed_slot_braces"] = int((bad_braces & keep).sum())
    keep &= ~bad_braces
    # Literal prompt-control text in user/agent text would alter the prompt structure or inject BOS/EOS.
    has_ctrl = df["instruction"].astype(str).map(_has_control) | df["output"].astype(str).map(_has_control)
    drops["prompt_control_tokens"] = int((has_ctrl & keep).sum())
    keep &= ~has_ctrl
    return df[keep].copy(), drops


CONTROL_STRINGS = ("<s>", "</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "<pad>", "<unk>")


def _has_control(s: str) -> bool:
    return any(c in s for c in CONTROL_STRINGS)


def drop_empty_after_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    bad = (df["instruction"] == "") | (df["output"] == "")
    return df[~bad].copy(), int(bad.sum())


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    before = df["output"].map(lambda s: sorted(SLOT_RE.findall(s)))
    for col in ("instruction", "output"):
        df[col] = df[col].astype(str).map(normalize_ws)
    after = df["output"].map(lambda s: sorted(SLOT_RE.findall(s)))
    changed = int((before != after).sum())
    if changed:
        raise ValueError(f"whitespace normalization altered template slots in {changed} rows")
    return df, int(df["output"].map(lambda s: bool(SLOT_RE.search(s))).sum())


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    n0 = len(df)
    df = df.drop_duplicates(subset=["instruction", "output"], keep="first")
    n1 = len(df)
    # Same customer message (modulo case/punctuation/whitespace) -> keep one. This is also what makes
    # the train/test split leakage-free on the input side.
    df = df.assign(_k=df["instruction"].map(norm_key)).drop_duplicates(subset="_k", keep="first")
    df = df.drop(columns="_k")
    return df, {"exact_duplicates": n0 - n1, "normalized_instruction_duplicates": n1 - len(df)}


def _largest_remainder(sizes: pd.Series, total: int) -> pd.Series:
    """Proportional integer allocation that sums exactly to `total`, never exceeding a group's size."""
    total = min(total, int(sizes.sum()))
    quota = sizes / sizes.sum() * total
    alloc = np.floor(quota).astype(int).clip(upper=sizes)
    rem = (quota - alloc).sort_values(ascending=False, kind="mergesort")
    for k in rem.index:
        if alloc.sum() >= total:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
    return alloc


def stratified_subsample(df: pd.DataFrame, by: str, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.copy()
    alloc = _largest_remainder(df.groupby(by).size(), n)
    parts = [g.sample(n=int(alloc[k]), random_state=seed) for k, g in df.groupby(by) if alloc[k] > 0]
    return pd.concat(parts).sort_values("source_row")


def stratified_split(df: pd.DataFrame, by: str, fracs: dict[str, float], seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[pd.DataFrame]] = {s: [] for s in SPLITS}
    for _, g in df.groupby(by, sort=True):
        idx = rng.permutation(len(g))
        n = len(g)
        n_test = max(1, round(n * fracs["test"])) if n >= 3 else 0
        n_val = max(1, round(n * fracs["val"])) if n >= 3 else 0
        g = g.iloc[idx]
        out["test"].append(g.iloc[:n_test])
        out["val"].append(g.iloc[n_test : n_test + n_val])
        out["train"].append(g.iloc[n_test + n_val :])
    return {s: pd.concat(v).sort_values("source_row").reset_index(drop=True) for s, v in out.items()}


def leakage_report(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    tr_in = set(splits["train"]["instruction"].map(norm_key))
    tr_pair = set(
        zip(splits["train"]["instruction"].map(norm_key), splits["train"]["output"].map(norm_key), strict=True)
    )
    rep = {}
    for s in ("val", "test"):
        keys = splits[s]["instruction"].map(norm_key)
        rep[f"{s}_instruction_in_train"] = int(keys.isin(tr_in).sum())
        pairs = zip(keys, splits[s]["output"].map(norm_key), strict=True)
        rep[f"{s}_pair_in_train"] = sum(p in tr_pair for p in pairs)
    vt = set(splits["val"]["instruction"].map(norm_key)) & set(splits["test"]["instruction"].map(norm_key))
    rep["val_test_instruction_overlap"] = len(vt)
    return rep


def response_overlap(splits: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Informational, not a failure: Bitext responses are templated, so eval responses may be near-copies
    of train responses. This bounds how much of an eval score could come from memorized templates."""
    tr_exact = set(splits["train"]["output"])
    tr_norm = set(splits["train"]["output"].map(norm_key))
    out = {}
    for s in ("val", "test"):
        n = max(len(splits[s]), 1)
        out[f"{s}_response_exact_in_train_pct"] = round(100 * splits[s]["output"].isin(tr_exact).sum() / n, 2)
        out[f"{s}_response_normalized_in_train_pct"] = round(
            100 * splits[s]["output"].map(norm_key).isin(tr_norm).sum() / n, 2
        )
    return out


# --------------------------------------------------------------------------- outputs


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def processed_dir(cfg: Config) -> Path:
    return data_processed_dir() / cfg.data.processed_name


def load_split(cfg: Config, split: str) -> list[dict[str, Any]]:
    return read_jsonl(processed_dir(cfg) / f"{split}.jsonl")


def dataset_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _pct(arr: np.ndarray) -> dict[str, int]:
    qs = {"min": 0, "p50": 50, "p90": 90, "p95": 95, "p99": 99, "max": 100}
    return {k: int(np.percentile(arr, q)) for k, q in qs.items()}


def _mask_sample_md(tok, row: dict, enc: Encoded) -> str:
    prompt_txt = tok.decode(enc.input_ids[: enc.n_prompt])
    resp_txt = tok.decode(enc.input_ids[enc.n_prompt :])
    lo, hi = max(0, enc.n_prompt - 4), min(enc.n_tokens, enc.n_prompt + 5)
    rows = "\n".join(
        f"| {i} | `{tok.convert_ids_to_tokens(enc.input_ids[i])}` | {enc.input_ids[i]} | "
        f"{'-100 (masked)' if enc.labels[i] == IGNORE_INDEX else enc.labels[i]} |"
        for i in range(lo, hi)
    )
    last = enc.n_tokens - 1
    rows += (
        f"\n| ... | | | |\n| {last} | `{tok.convert_ids_to_tokens(enc.input_ids[last])}` | "
        f"{enc.input_ids[last]} | {enc.labels[last]} (EOS kept) |"
    )
    return (
        f"**category** `{row['category']}` / **intent** `{row['intent']}` / {enc.n_tokens} tokens "
        f"({enc.n_prompt} masked prompt, {enc.n_tokens - enc.n_prompt} trained)\n\n"
        f"Masked prompt (labels = -100):\n```text\n{prompt_txt}\n```\n"
        f"Trained response (labels = input ids, incl. EOS):\n```text\n{resp_txt}\n```\n"
        f"Token boundary:\n\n| pos | token | id | label |\n|---|---|---|---|\n{rows}\n"
    )


def write_data_card(cfg: Config, tok, splits, stats: dict, samples: list[tuple[dict, Encoded]]) -> Path:
    cat_counts = pd.DataFrame({s: splits[s]["category"].value_counts() for s in SPLITS}).fillna(0).astype(int)
    cat_counts["total"] = cat_counts.sum(axis=1)
    cat_counts = cat_counts.sort_values("total", ascending=False)
    cat_md = "| category | train | val | test | total |\n|---|---:|---:|---:|---:|\n" + "\n".join(
        f"| {c} | {r.train} | {r.val} | {r.test} | {r.total} |" for c, r in cat_counts.iterrows()
    )
    tl = stats["token_lengths"]
    tl_md = "| | " + " | ".join(tl["corpus_before_filter"]) + " |\n|---|" + "---:|" * len(tl["corpus_before_filter"]) + "\n"
    for name, d in tl.items():
        tl_md += f"| {name} | " + " | ".join(str(v) for v in d.values()) + " |\n"
    funnel = stats["funnel"]
    funnel_md = "\n".join(f"| {k} | {v} |" for k, v in funnel.items())
    leak_md = "\n".join(f"| {k} | {v} |" for k, v in stats["leakage"].items())
    samples_md = "\n\n".join(f"### Sample {i + 1}\n\n" + _mask_sample_md(tok, r, e) for i, (r, e) in enumerate(samples))
    n_total = sum(len(splits[s]) for s in SPLITS)
    p99 = tl["corpus_before_filter"]["p99"]
    seq_note = (
        f"p99 = {p99} tokens <= max_seq_length = {cfg.data.max_seq_length}: no change needed."
        if p99 <= cfg.data.max_seq_length
        else f"**p99 = {p99} > max_seq_length = {cfg.data.max_seq_length}: raise max_seq_length.**"
    )
    md = f"""# Data card: {cfg.data.processed_name}

- Source: `{cfg.data.hf_dataset}` @ `{cfg.data.hf_revision}` / `{cfg.data.hf_filename}`
- Source file sha256: `{stats['source_sha256']}`
- **Dataset hash (train+val+test jsonl): `{stats['dataset_hash']}`**
- Tokenizer: `{cfg.model.tokenizer_name}` (pad token `{tok.pad_token}` id {tok.pad_token_id}, eos id {tok.eos_token_id})
- Seed {cfg.seed}; subsample {cfg.data.subsample_size} stratified by `{cfg.data.stratify_field}`;
  split {cfg.data.split.train}/{cfg.data.split.val}/{cfg.data.split.test} stratified by `{cfg.data.stratify_field}`
- System prompt: "{cfg.data.system_prompt}"
- Responses with `{{{{template slots}}}}`: {stats['slot_rows_final']} of {n_total} (kept verbatim)

## Funnel

| stage | rows |
|---|---:|
{funnel_md}

Dropped at validation: {json.dumps(stats['validation_drops'])}.
Dedupe: {json.dumps(stats['dedupe'])}.

## Split sizes

train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])};
{splits['train'][cfg.data.stratify_field].nunique()} intents, {splits['train']['category'].nunique()} categories.

## Counts per category

{cat_md}

## Token lengths (prompt + response + EOS)

{tl_md}
{seq_note}

## Leakage check (normalized text)

| check | count |
|---|---:|
{leak_md}

Response overlap with train (informational): {json.dumps(stats['response_overlap'])}.
Bitext responses are templated per intent, so eval responses can be near-copies of train responses.
This inflates absolute ROUGE-L for any model trained on this data; compare base vs tuned and
QLoRA vs full FT on the same split rather than reading absolute scores.

## Label-mask samples

Loss is computed on response tokens only. System and user tokens get label -100; EOS is kept so the
model learns to stop.

{samples_md}
"""
    path = ensure_dir(reports_dir()) / "data_card.md"
    path.write_text(md, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- driver


def run_data(cfg: Config, log_mlflow: bool = True) -> dict[str, Any]:
    from qlora_pipeline import telemetry

    if log_mlflow:
        telemetry.require_mlflow()  # fail fast, before doing work
    tok = load_tokenizer(cfg.model)

    src = download_raw(cfg)
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    raw = pd.read_csv(src, dtype=str, keep_default_na=False, encoding="utf-8")
    funnel = {"raw": len(raw)}

    df = map_fields(raw, cfg)
    df, vdrops = validate(df, cfg)
    funnel["after_validation"] = len(df)
    df, _ = normalize(df)
    df, vdrops["empty_after_normalize"] = drop_empty_after_normalize(df)
    funnel["after_normalize"] = len(df)
    df, dd = dedupe(df)
    funnel["after_dedupe"] = len(df)

    # Length filter BEFORE the subsample, so the subsample keeps its exact size and intent proportions.
    encs = [encode_example(tok, r.system, r.instruction, r.output) for r in df.itertuples()]
    df["n_tokens"] = [e.n_tokens for e in encs]
    df["n_prompt_tokens"] = [e.n_prompt for e in encs]
    lengths_before = df["n_tokens"].to_numpy()  # whole deduped corpus, decides max_seq_length
    df = df[df["n_tokens"] <= cfg.data.max_seq_length]
    funnel["after_length_filter"] = len(df)

    df = stratified_subsample(df, cfg.data.stratify_field, cfg.data.subsample_size, cfg.seed)
    funnel["after_subsample"] = len(df)

    fr = cfg.data.split.model_dump()
    splits = stratified_split(df, cfg.data.stratify_field, fr, cfg.seed)
    for s in SPLITS:
        funnel[f"split_{s}"] = len(splits[s])
    leak = leakage_report(splits)
    if leak["val_instruction_in_train"] or leak["test_instruction_in_train"] or leak["val_test_instruction_overlap"]:
        raise ValueError(f"leakage detected: {leak}")

    out = ensure_dir(processed_dir(cfg))
    files = []
    for s in SPLITS:
        p = out / f"{s}.jsonl"
        write_jsonl(splits[s], p)
        files.append(p)
    dhash = dataset_hash(files)

    stats = {
        "response_overlap": response_overlap(splits),
        "source_sha256": src_sha,
        "dataset_hash": dhash,
        "funnel": funnel,
        "validation_drops": vdrops,
        "dedupe": dd,
        "leakage": leak,
        "slot_rows_final": int(sum(splits[s]["output"].map(lambda t: bool(SLOT_RE.search(t))).sum() for s in SPLITS)),
        "token_lengths": {
            "corpus_before_filter": _pct(lengths_before),
            **{s: _pct(splits[s]["n_tokens"].to_numpy()) for s in SPLITS},
            "prompt_only": _pct(df["n_prompt_tokens"].to_numpy()),
        },
    }
    manifest = {"config": cfg.to_dict(), "config_fingerprint": cfg.fingerprint(), **stats,
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 3 samples from different categories, deterministic.
    tr = splits["train"]
    picks = tr.groupby("category").head(1).sample(n=min(3, tr["category"].nunique()), random_state=cfg.seed)
    samples = [
        (r, encode_example(tok, r["system"], r["instruction"], r["output"])) for r in picks.to_dict(orient="records")
    ]
    card = write_data_card(cfg, tok, splits, stats, samples)

    if log_mlflow:
        telemetry.log_data_run(cfg, stats, splits, card, out / "manifest.json")
    return {"dataset_hash": dhash, "out_dir": str(out), "data_card": str(card), **stats}
