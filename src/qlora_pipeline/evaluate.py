"""Phase 3 evaluation: base vs tuned on the test split with one fixed greedy generation config.

Per variant: ROUGE-L (overall + per category), val perplexity (response tokens only, token-weighted),
{{template slot}} preservation, lm-evaluation-harness regression (forgetting check).
Writes artifacts/reports/eval_<name>.md + predictions CSV and logs everything to MLflow.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from qlora_pipeline import telemetry
from qlora_pipeline.config import Config
from qlora_pipeline.data import SLOT_RE, encode_example, encode_prompt, load_split, load_tokenizer
from qlora_pipeline.paths import adapters_dir, ensure_dir, ensure_under_root, reports_dir, runs_dir

GB = 1024**3


# --------------------------------------------------------------------------- pure metrics


def rouge_l(preds: list[str], refs: list[str]) -> list[float]:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [100 * sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs, strict=True)]


def slot_stats(preds: list[str], refs: list[str]) -> dict[str, float]:
    """Slot recall = reference slots reproduced verbatim; invented = slots in the prediction not in the reference."""
    ref_total = kept = rows_with = rows_all_kept = invented = 0
    for p, r in zip(preds, refs, strict=True):
        rs, ps = Counter(SLOT_RE.findall(r)), Counter(SLOT_RE.findall(p))
        if rs:
            rows_with += 1
            k = sum(min(c, ps[s]) for s, c in rs.items())
            ref_total += sum(rs.values())
            kept += k
            rows_all_kept += k == sum(rs.values())
        invented += sum((ps - rs).values())
    return {
        "slot_recall_pct": 100 * kept / ref_total if ref_total else float("nan"),
        "rows_all_slots_kept_pct": 100 * rows_all_kept / rows_with if rows_with else float("nan"),
        "rows_with_slots": rows_with,
        "invented_slots": invented,
        "invented_slots_per_row": invented / max(len(preds), 1),
    }


def per_category(df: pd.DataFrame, col: str) -> dict[str, float]:
    return df.groupby("category")[col].mean().round(2).to_dict()


# --------------------------------------------------------------------------- model-based


def generation_config(cfg: Config, tok, model):
    from transformers import GenerationConfig

    from qlora_pipeline.model import suppressed_token_ids

    return GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=cfg.eval.max_new_tokens,
        temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        suppress_tokens=suppressed_token_ids(model, tok),
    )  # fmt: skip


def generate(model, tok, cfg: Config, rows: list[dict], gen_cfg) -> list[dict]:
    import torch

    enc = [encode_prompt(tok, r["system"], r["instruction"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -len(enc[i]))  # longest first: OOM shows up immediately
    out: list[dict | None] = [None] * len(rows)
    bs = cfg.eval.gen_batch_size
    for b in range(0, len(order), bs):
        idx = order[b : b + bs]
        width = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, i in enumerate(idx):  # left padding for decoder-only generation
            ids[j, width - len(enc[i]) :] = torch.tensor(enc[i])
            mask[j, width - len(enc[i]) :] = 1
        with torch.inference_mode():
            gen = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), generation_config=gen_cfg)
        for j, i in enumerate(idx):
            new = gen[j, width:].tolist()
            n = new.index(tok.eos_token_id) + 1 if tok.eos_token_id in new else len(new)
            n_real = sum(t != tok.pad_token_id for t in new[:n])
            out[i] = {
                "pred": tok.decode(new[:n], skip_special_tokens=True).strip(),
                "gen_tokens": n_real,
                "hit_max_new_tokens": tok.eos_token_id not in new,
            }
        del gen
        torch.cuda.empty_cache()  # the growing KV cache fragments the allocator; drop fragments between batches
    return out  # type: ignore[return-value]


def perplexity(model, tok, cfg: Config, rows: list[dict]) -> float:
    """exp(mean NLL) over response tokens (prompt masked), token-weighted across the split."""
    import torch

    encs = [encode_example(tok, r["system"], r["instruction"], r["output"]) for r in rows]
    nll_sum, n_tok = 0.0, 0
    bs = cfg.eval.ppl_batch_size
    for b in range(0, len(encs), bs):
        batch = encs[b : b + bs]
        width = max(e.n_tokens for e in batch)
        ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, e in enumerate(batch):
            ids[j, : e.n_tokens] = torch.tensor(e.input_ids)
            labels[j, : e.n_tokens] = torch.tensor(e.labels)
            mask[j, : e.n_tokens] = 1
        with torch.inference_mode():
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False).logits.float()
        shift_logits, shift_labels = logits[:, :-1], labels[:, 1:].cuda()
        nll = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100, reduction="sum"
        )
        nll_sum += nll.item()
        n_tok += int((shift_labels != -100).sum())
        del logits, shift_logits
    return math.exp(nll_sum / n_tok)


def _pct(v) -> float:
    """lm_eval may report 'N/A' for a stderr (tiny limits); keep NaN instead of crashing after a long run."""
    try:
        return round(100 * float(v), 2)
    except (TypeError, ValueError):
        return float("nan")


def lm_eval_regression(model, tok, cfg: Config, stage_mem: dict | None = None) -> dict[str, dict[str, float]]:
    """One simple_evaluate per task so each task gets its own peak-VRAM reading."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=cfg.eval.lm_eval_batch_size)
    out = {}
    for task in cfg.eval.lm_eval_tasks:

        def _run(task=task):
            return simple_evaluate(
                model=lm, tasks=[task], limit=cfg.eval.lm_eval_limit, log_samples=False, random_seed=cfg.seed,
                numpy_random_seed=cfg.seed, torch_random_seed=cfg.seed, fewshot_random_seed=cfg.seed,
            )  # fmt: skip

        res = _stage(cfg, stage_mem, f"lm_eval.{task}", _run)
        if res is None:  # OOM in probe mode
            out[task] = {}
            continue
        keep = ("acc", "acc_norm", "acc_stderr", "acc_norm_stderr")
        for t, m in res["results"].items():
            out[t] = {k.split(",")[0]: _pct(v) for k, v in m.items() if k.split(",")[0] in keep}
    return out


def _stage(cfg: Config, stage_mem: dict | None, name: str, fn):
    """Run one eval stage with its own peak-VRAM window; release cached blocks afterwards (WDDM commit)."""
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0, oom, result = time.time(), False, None
    try:
        result = fn()
    except RuntimeError as e:  # torch.cuda.OutOfMemoryError, or AcceleratorError "CUDA error: out of memory"
        if not cfg.eval.continue_on_oom or "out of memory" not in str(e).lower():
            raise
        oom = True
    rec = {"peak_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 3),
           "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
           "secs": round(time.time() - t0, 1), "oom": oom}  # fmt: skip
    torch.cuda.empty_cache()
    if stage_mem is not None:
        stage_mem[name] = rec
    print(f"  stage {name:22} peak reserved {rec['peak_reserved_gb']:6.2f} GB  allocated {rec['peak_allocated_gb']:6.2f} GB"
          f"  {rec['secs']:7.1f}s{'  OOM' if oom else ''}", flush=True)
    return result


def score_variant(model, tok, cfg: Config, test_rows, val_rows, gen_cfg, do_lm_eval: bool) -> dict[str, Any]:
    stage_mem: dict[str, dict] = {}
    gens = _stage(cfg, stage_mem, "generation", lambda: generate(model, tok, cfg, test_rows, gen_cfg))
    if gens is None:  # OOM in probe mode
        gens = [{"pred": "", "gen_tokens": 0, "hit_max_new_tokens": False} for _ in test_rows]
    t_gen = stage_mem["generation"]["secs"]
    preds = [g["pred"] for g in gens]
    refs = [r["output"] for r in test_rows]
    res: dict[str, Any] = {
        "preds": preds,
        "gens": gens,
        "rougeL": rouge_l(preds, refs),
        "slots": slot_stats(preds, refs),
        "gen_time_s": round(t_gen, 1),
        "gen_tokens_per_s": round(sum(g["gen_tokens"] for g in gens) / max(t_gen, 1e-9), 1),
        "hit_max_new_tokens_pct": round(100 * sum(g["hit_max_new_tokens"] for g in gens) / len(gens), 2),
        "mean_gen_tokens": round(sum(g["gen_tokens"] for g in gens) / len(gens), 1),
    }
    ppl = _stage(cfg, stage_mem, "perplexity", lambda: perplexity(model, tok, cfg, val_rows))
    res["val_perplexity"] = ppl if ppl is not None else float("nan")
    res["lm_eval"] = lm_eval_regression(model, tok, cfg, stage_mem) if do_lm_eval else {}
    res["stage_mem"] = stage_mem
    res["peak_mem_reserved_gb"] = max(s["peak_reserved_gb"] for s in stage_mem.values())
    return res


# --------------------------------------------------------------------------- driver


def resolve_weights(cfg: Config) -> Path:
    p = Path(cfg.eval.weights) if cfg.eval.weights else adapters_dir() / cfg.train.output_name
    p = ensure_under_root(p)
    if not p.is_dir():
        raise FileNotFoundError(f"weights not found: {p}")
    return p


def _report(cfg: Config, name: str, weights: Path, v: dict[str, dict], df: pd.DataFrame, gen_cfg) -> str:
    variants = list(v)
    overall = "| metric | " + " | ".join(variants) + " |\n|---|" + "---:|" * len(variants) + "\n"
    rows = [
        ("ROUGE-L (mean, test)", lambda r: f"{sum(r['rougeL']) / len(r['rougeL']):.2f}"),
        ("val perplexity (response tokens)", lambda r: f"{r['val_perplexity']:.3f}"),
        ("slot recall %", lambda r: f"{r['slots']['slot_recall_pct']:.1f}"),
        ("rows with all slots kept %", lambda r: f"{r['slots']['rows_all_slots_kept_pct']:.1f}"),
        ("invented slots / row", lambda r: f"{r['slots']['invented_slots_per_row']:.3f}"),
        ("mean generated tokens", lambda r: f"{r['mean_gen_tokens']}"),
        ("hit max_new_tokens %", lambda r: f"{r['hit_max_new_tokens_pct']}"),
        ("generation tokens/s (batched)", lambda r: f"{r['gen_tokens_per_s']}"),
    ]
    for label, f in rows:
        overall += f"| {label} | " + " | ".join(f(v[k]) for k in variants) + " |\n"

    cats = sorted(df["category"].unique())
    cat_md = "| category | n | " + " | ".join(variants) + " |\n|---|---:|" + "---:|" * len(variants) + "\n"
    for c in cats:
        sub = df[df["category"] == c]
        cat_md += f"| {c} | {len(sub)} | " + " | ".join(f"{sub[f'rougeL_{k}'].mean():.2f}" for k in variants) + " |\n"

    lm_md = ""
    tasks = list(cfg.eval.lm_eval_tasks)
    if all(v[k]["lm_eval"] for k in variants):
        lm_md = "| task | metric | " + " | ".join(variants) + " |\n|---|---|" + "---:|" * len(variants) + "\n"
        for t in tasks:
            for m in ("acc", "acc_norm"):
                if m in v[variants[0]]["lm_eval"].get(t, {}):
                    lm_md += f"| {t} | {m} | " + " | ".join(
                        f"{v[k]['lm_eval'][t][m]:.1f} ± {v[k]['lm_eval'][t].get(m + '_stderr', float('nan')):.1f}"
                        for k in variants
                    ) + " |\n"
    e = cfg.eval
    batches = f"batch sizes: gen {e.gen_batch_size}, ppl {e.ppl_batch_size}, lm-eval {e.lm_eval_batch_size}"
    stages = list(next(iter(v.values()))["stage_mem"])
    mem_md = "| stage | " + " | ".join(f"{k} reserved / allocated GB" for k in variants) + " |\n|---|" + "---:|" * len(
        variants) + "\n"
    for st in stages:
        cells = []
        for k in variants:
            m = v[k]["stage_mem"].get(st, {})
            cells.append(f"{m.get('peak_reserved_gb', float('nan')):.2f} / {m.get('peak_allocated_gb', float('nan')):.2f}"
                         + (" OOM" if m.get("oom") else ""))
        mem_md += f"| {st} | " + " | ".join(cells) + " |\n"
    verdict = forgetting_verdict(cfg, v)
    examples = ""
    for _, r in df.head(3).iterrows():
        examples += f"\n**[{r['category']} / {r['intent']}]** {r['instruction']}\n\n"
        for k in variants:
            examples += f"- **{k}** (ROUGE-L {r[f'rougeL_{k}']:.1f}): {r[f'pred_{k}'][:500]}\n"
        examples += f"- **reference**: {r['reference'][:500]}\n"
    return f"""# Evaluation: {name}

- Weights: `{weights}`; base model `{cfg.model.name}` @ `{cfg.model.revision}`; 4-bit: {v[variants[0]]['quantized']}
- Test split: {len(df)} examples; val split for perplexity. Same prompt template as training.
- Fixed generation config: greedy (`do_sample=False`, `num_beams=1`), `max_new_tokens={gen_cfg.max_new_tokens}`,
  `repetition_penalty=1.0`, pad and embedding-padding ids suppressed.
- ROUGE-L = rouge_score F-measure x100 with stemming. Caveat: Bitext responses are templated per intent,
  so absolute ROUGE-L is high for any model that learns the house style; use it comparatively.

## Overall

{overall}
## ROUGE-L per category

{cat_md}
## Peak GPU memory per stage ({batches})

{mem_md}
## Forgetting check (lm-evaluation-harness, limit={cfg.eval.lm_eval_limit} per task, ± = stderr)

{lm_md or '_skipped_'}
{verdict['text']}

## First 3 test examples (full table: predictions CSV / MLflow `predictions.json`)
{examples}
"""


def forgetting_verdict(cfg: Config, v: dict[str, dict]) -> dict[str, Any]:
    """Flag a task only if tuned drops below base by more than the tolerance AND by more than 1.5x the
    combined stderr (with --limit, stderr is several points, so a bare threshold would flag noise)."""
    if "base" not in v or not v["base"]["lm_eval"] or not v["tuned"]["lm_eval"]:
        return {"text": "", "flags": [], "deltas": {}}
    flags, deltas = [], {}
    for t, mt in v["tuned"]["lm_eval"].items():
        mb = v["base"]["lm_eval"].get(t, {})
        key = "acc_norm" if "acc_norm" in mt and "acc_norm" in mb else "acc"
        d = mt[key] - mb[key]
        se = math.sqrt(mt.get(f"{key}_stderr", 0) ** 2 + mb.get(f"{key}_stderr", 0) ** 2)
        se = se if math.isfinite(se) else 0.0
        deltas[t] = {"metric": key, "delta": round(d, 2), "combined_stderr": round(se, 2)}
        if d < -cfg.eval.forgetting_tolerance and d < -1.5 * se:
            flags.append(f"{t} {key} {d:+.1f} (stderr {se:.1f})")
    desc = ", ".join(f"{t} {x['metric']} {x['delta']:+.1f} (±{x['combined_stderr']:.1f})" for t, x in deltas.items())
    status = "FLAG: possible forgetting" if flags else "no forgetting beyond tolerance"
    text = (f"Tuned minus base: {desc} points. Rule: flag if drop > {cfg.eval.forgetting_tolerance} pts and "
            f"> 1.5x combined stderr: **{status}**" + (f" ({'; '.join(flags)})." if flags else "."))  # fmt: skip
    return {"text": text, "flags": flags, "deltas": deltas}


def run_eval(cfg: Config, do_lm_eval: bool = True, tag: str | None = None, record_peak_commit: bool = True,
             replace_peak: bool = False) -> dict[str, Any]:  # fmt: skip
    import mlflow

    from qlora_pipeline.model import is_adapter_dir, load_for_inference, preflight

    preflight(cfg, f"eval {tag or cfg.train.output_name}", job="eval")
    telemetry.require_mlflow()
    weights = resolve_weights(cfg)
    name = tag or cfg.train.output_name
    quantized = cfg.method == "qlora" if cfg.eval.quantized == "auto" else bool(cfg.eval.quantized)
    tok = load_tokenizer(cfg.model, padding_side="left")
    test_rows = load_split(cfg, "test")
    if cfg.eval.test_limit:
        if cfg.eval.test_select == "longest":  # worst case for prompt length / KV cache
            test_rows = sorted(test_rows, key=lambda r: -len(encode_prompt(tok, r["system"], r["instruction"])))
        test_rows = test_rows[: cfg.eval.test_limit]
    val_rows = load_split(cfg, "val")

    cap = telemetry.cap_vram(cfg.eval.max_reserved_gb)
    print(f"VRAM cap: reserved limited to {cap:.2f} GB of {telemetry.physical_vram_gb():.2f} GB physical")
    model = load_for_inference(cfg, tok, weights, quantized=quantized)
    gen_cfg = generation_config(cfg, tok, model)
    variants: dict[str, dict] = {}
    variants["tuned"] = score_variant(model, tok, cfg, test_rows, val_rows, gen_cfg, do_lm_eval)
    if cfg.eval.compare_base and is_adapter_dir(weights):
        with model.disable_adapter():  # the identical base weights, same quantization
            variants["base"] = score_variant(model, tok, cfg, test_rows, val_rows, gen_cfg, do_lm_eval)
        variants = {"base": variants["base"], "tuned": variants["tuned"]}
    else:
        print("NOTE: base comparison skipped (compare_base=false or weights are a full model, not an adapter)")
    for r in variants.values():
        r["quantized"] = quantized

    df = pd.DataFrame({
        "id": [r["source_row"] for r in test_rows], "category": [r["category"] for r in test_rows],
        "intent": [r["intent"] for r in test_rows], "instruction": [r["instruction"] for r in test_rows],
        "reference": [r["output"] for r in test_rows],
    })  # fmt: skip
    for k, r in variants.items():
        df[f"pred_{k}"] = r["preds"]
        df[f"rougeL_{k}"] = [round(x, 2) for x in r["rougeL"]]
        df[f"hit_max_{k}"] = [g["hit_max_new_tokens"] for g in r["gens"]]

    rep_dir = ensure_dir(reports_dir())
    csv_path = rep_dir / f"eval_{name}_predictions.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    md_path = rep_dir / f"eval_{name}.md"
    md_path.write_text(_report(cfg, name, weights, variants, df, gen_cfg), encoding="utf-8")

    summary: dict[str, Any] = {"name": name, "weights": str(weights), "quantized": quantized, "n_test": len(df),
                               "generation": gen_cfg.to_dict()}  # fmt: skip
    for k, r in variants.items():
        summary[k] = {
            "rougeL": round(sum(r["rougeL"]) / len(r["rougeL"]), 3),
            "rougeL_per_category": per_category(df, f"rougeL_{k}"),
            "val_perplexity": round(r["val_perplexity"], 4),
            **{f"slots.{a}": b for a, b in r["slots"].items()},
            "lm_eval": r["lm_eval"], "gen_time_s": r["gen_time_s"], "gen_tokens_per_s": r["gen_tokens_per_s"],
            "hit_max_new_tokens_pct": r["hit_max_new_tokens_pct"], "mean_gen_tokens": r["mean_gen_tokens"],
            "peak_mem_reserved_gb": r["peak_mem_reserved_gb"], "stage_mem": r["stage_mem"],
        }  # fmt: skip
    from qlora_pipeline.audit import process_commit_gb
    from qlora_pipeline.model import record_peak

    summary["process_peak_commit_gb"] = (
        record_peak(cfg, "eval", f"eval {name}", replace=replace_peak) if record_peak_commit
        else round(process_commit_gb().get("peak_private_gb", float("nan")), 2)
    )
    worst = max(summary[k]["peak_mem_reserved_gb"] for k in variants)
    summary["physical_vram_gb"] = round(telemetry.physical_vram_gb(), 2)
    summary["sysmem_fallback_detected"] = worst > summary["physical_vram_gb"]
    fv = forgetting_verdict(cfg, variants)
    summary["forgetting"], summary["forgetting_deltas"] = fv["flags"], fv["deltas"]

    train_run = runs_dir() / cfg.train.output_name / "mlflow_run_id.txt"
    tags = {"stage": "eval", "method": cfg.method, "weights": str(weights), "eval_name": name,
            "train_run_id": train_run.read_text(encoding="utf-8").strip() if train_run.exists() else ""}  # fmt: skip
    with telemetry.start_run(cfg, f"eval-{name}", tags) as run:
        mlflow.log_params(telemetry.flatten({"eval": cfg.eval.model_dump(mode="json")}))
        mlflow.log_dict(gen_cfg.to_dict(), "generation_config.json")
        metrics = {}
        for k in variants:
            s = summary[k]
            metrics.update({f"{k}.rougeL": s["rougeL"], f"{k}.val_perplexity": s["val_perplexity"],
                            f"{k}.hit_max_new_tokens_pct": s["hit_max_new_tokens_pct"],
                            f"{k}.peak_mem_reserved_gb": s["peak_mem_reserved_gb"]})  # fmt: skip
            metrics.update({f"{k}.{a}": b for a, b in s.items() if a.startswith("slots.") and isinstance(b, int | float)
                            and math.isfinite(b)})  # fmt: skip
            metrics.update({f"{k}.rougeL.{re.sub(r'[^A-Za-z0-9_]', '_', c)}": x
                            for c, x in s["rougeL_per_category"].items()})  # fmt: skip
            for t, m in s["lm_eval"].items():
                metrics.update({f"{k}.lm_eval.{t}.{a}": b for a, b in m.items() if math.isfinite(b)})
            for st, mem in s["stage_mem"].items():
                metrics.update({f"{k}.mem.{st}.peak_reserved_gb": mem["peak_reserved_gb"],
                                f"{k}.mem.{st}.peak_allocated_gb": mem["peak_allocated_gb"]})  # fmt: skip
        if "base" in variants:
            metrics["delta.rougeL"] = summary["tuned"]["rougeL"] - summary["base"]["rougeL"]
        mlflow.log_metrics(metrics)
        mlflow.log_table(df, "predictions.json")
        mlflow.log_artifact(str(md_path))
        mlflow.log_artifact(str(csv_path))
        summary["mlflow_run_id"] = run.info.run_id
    (rep_dir / f"eval_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
