"""Phase 4 benchmark: TinyLlama-1.1B proxy trained as full FT, bf16 LoRA and QLoRA with identical data, prompt
template, seq length, effective batch, steps and seed; then 7B estimates (labelled ESTIMATED) next to the
7B QLoRA measurement from Phase 2.

Every training and eval runs in its own subprocess so peak-memory counters cannot leak between methods.
Before each run, a separate process confirms the NVIDIA sysmem fallback is OFF (allocating more than physical
VRAM must fail); after each run, peak reserved VRAM is checked against physical VRAM.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from qlora_pipeline import telemetry
from qlora_pipeline.config import Config, load_config
from qlora_pipeline.paths import ensure_dir, reports_dir, runs_dir

GB = 1024**3
METHODS = [("full", "proxy_full.yaml"), ("lora", "proxy_lora.yaml"), ("qlora", "proxy_qlora.yaml")]
SEVEN_B_CONFIG = "llama2_7b_qlora.yaml"

# Must be identical across the three proxy runs (dotted config paths).
SAME = ["model.name", "model.revision", "data.processed_name", "data.max_seq_length", "data.system_prompt",
        "train.effective_batch_size", "train.per_device_train_batch_size", "train.num_train_epochs",
        "train.max_steps", "seed", "train.gradient_checkpointing", "train.optim", "train.bf16",
        "train.lr_scheduler_type", "train.warmup_ratio", "train.max_grad_norm"]  # fmt: skip

_FALLBACK_PROBE = r"""
import torch
phys = torch.cuda.get_device_properties(0).total_memory
try:
    x = torch.empty(phys + (1 << 30), dtype=torch.uint8, device="cuda")
    del x
    print("FALLBACK_ON")
except RuntimeError as e:
    print("FALLBACK_OFF" if "out of memory" in str(e).lower() else "PROBE_ERROR " + str(e)[:200])
"""


def assert_sysmem_fallback_off() -> None:
    """In a fresh process, allocate physical VRAM + 1 GB: with fallback off this must raise OOM."""
    r = subprocess.run([sys.executable, "-c", _FALLBACK_PROBE], capture_output=True, text=True, timeout=300)
    verdict = (r.stdout.strip().splitlines() or ["PROBE_ERROR no output"])[-1]
    if verdict != "FALLBACK_OFF":
        raise RuntimeError(f"sysmem fallback check failed: {verdict}. Set NVIDIA 'CUDA - Sysmem Fallback Policy' "
                           "to 'Prefer No Sysmem Fallback' before benchmarking.")  # fmt: skip
    print("pre-run check: CUDA sysmem fallback is OFF (allocating physical VRAM + 1 GB raised OOM)", flush=True)


def _get(cfg: Config, dotted: str) -> Any:
    cur: Any = cfg.to_dict()
    for k in dotted.split("."):
        cur = cur[k]
    return cur


def check_identical(cfgs: dict[str, Config]) -> dict[str, Any]:
    ref = next(iter(cfgs.values()))
    diffs = {k: {m: _get(c, k) for m, c in cfgs.items()} for k in SAME
             if len({json.dumps(_get(c, k)) for c in cfgs.values()}) > 1}  # fmt: skip
    if diffs:
        raise ValueError(f"proxy configs differ in settings that must match: {diffs}")
    return {k: _get(ref, k) for k in SAME}


def _cli(*args: str, log: Path) -> None:
    cmd = [sys.executable, "-m", "qlora_pipeline.cli", *args]
    print("running:", " ".join(args), f"(log: {log})", flush=True)
    with log.open("w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"`qlora {' '.join(args)}` failed (exit {rc}); see {log}")


def _expected_steps(cfg: Config, n_train: int) -> int:
    per_epoch = math.ceil(math.ceil(n_train / cfg.train.per_device_train_batch_size) / cfg.train.grad_accum)
    return cfg.train.max_steps if cfg.train.max_steps > 0 else int(per_epoch * cfg.train.num_train_epochs)


def reusable_training(cfg: Config, n_train: int) -> tuple[bool, str]:
    """A finished run is reused only if it is complete, was not resumed, and was made with the same config + data."""
    from qlora_pipeline.train import resume_fingerprint, verify_dataset

    out = runs_dir() / cfg.train.output_name
    summ, meta = out / "summary.json", out / "run_meta.json"
    if not summ.exists() or not meta.exists():
        return False, "no completed run"
    s, m = json.loads(summ.read_text(encoding="utf-8")), json.loads(meta.read_text(encoding="utf-8"))
    checks = {
        "steps": s.get("global_steps") == _expected_steps(cfg, n_train),
        "not resumed": s.get("resumed_from") is None,
        "config": m.get("config_fingerprint") == resume_fingerprint(cfg),
        "dataset": m.get("dataset_hash") == s.get("dataset_hash") == verify_dataset(cfg),
    }
    bad = [k for k, ok in checks.items() if not ok]
    return (not bad), ("reusable" if not bad else f"stale ({', '.join(bad)})")


def _record_check(path: Path, what: str) -> None:
    from datetime import datetime

    path.write_text(json.dumps({"sysmem_fallback": "OFF", "checked_before": what,
                                "at": datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")  # fmt: skip


def run_one(method: str, cfg_name: str, cfg: Config, n_train: int, logs: Path, force: bool) -> dict[str, Any]:
    out = runs_dir() / cfg.train.output_name
    summ_path = out / "summary.json"
    reuse, why = (False, "--force") if force else reusable_training(cfg, n_train)
    if not reuse:
        print(f"{method}: training ({why})", flush=True)
        assert_sysmem_fallback_off()
        _record_check(out.parent / f"{out.name}.fallback_check_train.json", "training")
        _cli("train", "-c", cfg_name, "--set", "train.resume=none", log=logs / f"bench_train_{method}.log")
    else:
        print(f"{method}: REUSING completed training at {out} ({why})", flush=True)
    train = json.loads(summ_path.read_text(encoding="utf-8"))
    telemetry.check_no_sysmem_fallback(train["peak_mem_reserved_gb"], f"{method} training")

    tag = f"bench-{method}"
    ev_path = reports_dir() / f"eval_{tag}.json"
    ev_ok = (reuse and ev_path.exists() and ev_path.stat().st_mtime > summ_path.stat().st_mtime
             and json.loads(ev_path.read_text(encoding="utf-8")).get("weights") == train.get("adapter_dir"))  # fmt: skip
    if not ev_ok:
        assert_sysmem_fallback_off()
        _record_check(out.parent / f"{out.name}.fallback_check_eval.json", "eval")
        _cli("eval", "-c", cfg_name, "--tag", tag, "--skip-lm-eval", "--no-record-peak",
             "--set", "eval.compare_base=false", log=logs / f"bench_eval_{method}.log")  # fmt: skip
    else:
        print(f"{method}: REUSING eval {ev_path.name} (made from {train.get('adapter_dir')} after training)", flush=True)
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    telemetry.check_no_sysmem_fallback(ev["tuned"]["peak_mem_reserved_gb"], f"{method} eval")
    checks = {k: (out.parent / f"{out.name}.fallback_check_{k}.json") for k in ("train", "eval")}
    fb = {k: json.loads(v.read_text(encoding="utf-8"))["at"] if v.exists() else None for k, v in checks.items()}
    return {"train": train, "eval": ev, "reused_training": reuse, "reused_eval": ev_ok, "fallback_checked_at": fb}


def paired_bootstrap_ci(a: list[float], b: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over paired rows and its 95% percentile bootstrap CI."""
    import numpy as np

    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_rouge(method_a: str, method_b: str) -> tuple[float, float, float, int]:
    import pandas as pd

    fa = pd.read_csv(reports_dir() / f"eval_bench-{method_a}_predictions.csv", encoding="utf-8")
    fb = pd.read_csv(reports_dir() / f"eval_bench-{method_b}_predictions.csv", encoding="utf-8")
    m = fa[["id", "rougeL_tuned"]].merge(fb[["id", "rougeL_tuned"]], on="id", suffixes=("_a", "_b"))
    mean, lo, hi = paired_bootstrap_ci(m["rougeL_tuned_a"].tolist(), m["rougeL_tuned_b"].tolist())
    return round(mean, 2), round(lo, 2), round(hi, 2), len(m)


def seven_b_estimates(qlora_7b: dict[str, Any]) -> dict[str, Any]:
    """Analytical peak-memory estimates for 7B bf16 LoRA and full FT (ESTIMATED), anchored on the measured 7B
    QLoRA run: activations + LoRA optimizer overhead = measured QLoRA peak - its 4-bit weight footprint."""
    from qlora_pipeline.model import estimate_params_b

    p = estimate_params_b(load_config(SEVEN_B_CONFIG).model) * 1e9
    measured_peak = qlora_7b["peak_mem_allocated_gb"]
    weights_4bit = qlora_7b["mem_after_load_gb"]  # measured: 4-bit base + fp32-upcast norms/embeddings + LoRA
    act_and_lora = max(measured_peak - weights_4bit, 0.0)  # activations (GC on) + LoRA grads/optimizer, measured
    w_bf16 = 2 * p / GB
    est = {
        "params_b": round(p / 1e9, 3),
        "anchor": {"qlora_7b_peak_allocated_gb": measured_peak, "qlora_7b_after_load_gb": weights_4bit,
                   "activations_plus_lora_state_gb": round(act_and_lora, 2)},  # fmt: skip
        "bf16_lora": {
            "formula": "bf16 weights 2P + (activations + LoRA grads/optimizer, from the measured 7B QLoRA run)",
            "weights_gb": round(w_bf16, 2), "activations_plus_lora_gb": round(act_and_lora, 2),
            "total_gb": round(w_bf16 + act_and_lora, 2),
        },  # fmt: skip
        "full_ft_8bit_adam": {
            "formula": "bf16 weights 2P + bf16 grads 2P + paged 8-bit AdamW states 2P (2 x 1 byte) + activations",
            "weights_gb": round(w_bf16, 2), "grads_gb": round(w_bf16, 2), "optimizer_gb": round(2 * p / GB, 2),
            "activations_gb": round(act_and_lora, 2), "total_gb": round(6 * p / GB + act_and_lora, 2),
        },  # fmt: skip
        "full_ft_fp32_adam": {
            "formula": "bf16 weights 2P + bf16 grads 2P + fp32 master weights 4P + fp32 Adam m,v 8P + activations",
            "total_gb": round(16 * p / GB + act_and_lora, 2),
        },
    }
    return est


def pct_reduction(new: float, old: float) -> float:
    return round(100 * (old - new) / old, 1)


def write_report(cfgs, same, res, seven, tol, parity) -> tuple[Path, dict[str, float], bool, bool]:
    phys = telemetry.physical_vram_gb()
    rows, gpu_rows = "", ""
    for m in ("full", "lora", "qlora"):
        t, e = res[m]["train"], res[m]["eval"]["tuned"]
        gpu_rows += (f"| {m} | {t.get('gpu.throttle_power_pct', float('nan')):.0f}% | {t.get('gpu.sm_clock_min_mhz', 0):.0f} | "
                     f"{t.get('gpu.temp_max_c', 0):.0f} | {t.get('gpu.power_mean_w', 0):.1f} | "
                     f"{'reused' if res[m]['reused_training'] else 'this invocation'} | "
                     f"{res[m]['fallback_checked_at']['train'] or 'see logs/bench.log'} |\n")  # fmt: skip
        rows += (f"| {m} | {cfgs[m].train.learning_rate:g} | {t['trainable_pct']:.2f}% | {t['peak_mem_allocated_gb']:.2f} | "
                 f"{t['peak_mem_reserved_gb']:.2f} | {t['tokens_per_sec']:.0f} | {t['wall_time_s'] / 60:.1f} | "
                 f"{t['best_eval_loss']:.4f} | {e['val_perplexity']:.3f} | {e['rougeL']:.2f} |\n")  # fmt: skip
    q, lo, fu = (res[m]["train"] for m in ("qlora", "lora", "full"))
    red = {
        "alloc_vs_lora": pct_reduction(q["peak_mem_allocated_gb"], lo["peak_mem_allocated_gb"]),
        "alloc_vs_full": pct_reduction(q["peak_mem_allocated_gb"], fu["peak_mem_allocated_gb"]),
        "reserved_vs_lora": pct_reduction(q["peak_mem_reserved_gb"], lo["peak_mem_reserved_gb"]),
        "reserved_vs_full": pct_reduction(q["peak_mem_reserved_gb"], fu["peak_mem_reserved_gb"]),
    }
    q7 = seven["anchor"]["qlora_7b_peak_allocated_gb"]
    s_l, s_f8, s_f32 = (seven[k]["total_gb"] for k in ("bf16_lora", "full_ft_8bit_adam", "full_ft_fp32_adam"))
    mem_claim = red["alloc_vs_lora"] > 0 and red["alloc_vs_full"] > 0
    res_claim = red["reserved_vs_lora"] > 0 and red["reserved_vs_full"] > 0
    # (1) paged 8-bit AdamW state lives in CUDA managed memory, invisible to torch's allocator counters
    from qlora_pipeline.model import estimate_params_b

    p_proxy = estimate_params_b(cfgs["full"].model) * 1e9
    full_opt_gb = 2 * p_proxy / GB
    full_corr = fu["peak_mem_allocated_gb"] + full_opt_gb
    red["alloc_vs_full_corrected_est"] = pct_reduction(q["peak_mem_allocated_gb"], full_corr)
    # (2) validate the 7B anchor formula on the proxy, where bf16 LoRA was actually measured
    proxy_pred_lora = 2 * p_proxy / GB + max(q["peak_mem_allocated_gb"] - q["mem_after_load_gb"], 0.0)
    anchor_err = 100 * (proxy_pred_lora - lo["peak_mem_allocated_gb"]) / lo["peak_mem_allocated_gb"]
    # (5) paired bootstrap CI for QLoRA - full FT ROUGE-L
    d_mean, d_lo, d_hi, n_pair = paired_rouge("qlora", "full")
    parity["ci95"] = [d_lo, d_hi]
    parity["n_paired"] = n_pair
    same_md = "\n".join(f"| `{k}` | `{v}` |" for k, v in same.items())
    md = f"""# Benchmark: QLoRA vs bf16 LoRA vs full fine-tuning

Proxy: `{cfgs['qlora'].model.name}` @ `{cfgs['qlora'].model.revision}` (Llama 2 architecture and tokenizer).
Dataset hash `{q['dataset_hash']}`. GPU: RTX 5080 Laptop, {phys:.2f} GB physical VRAM. NVIDIA sysmem fallback was
verified OFF in a separate process (allocating physical VRAM + 1 GB must raise OOM) before each training and eval
subprocess when it ran (timestamps below; logs/bench.log). Every peak below is <= physical VRAM.
`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6` for all three runs.

## Proxy results (MEASURED, 1 epoch = {q['global_steps']} optimizer steps each)

| method | lr | trainable | peak allocated GB* | peak reserved GB* | tokens/s | wall min | best val loss (trainer) | val ppl (eval) | test ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}
(*) Peak memory = `torch.cuda.max_memory_allocated` / `max_memory_reserved`, reset before model load: weights,
gradients, activations, eval and any optimizer state held by PyTorch's allocator. **Not included: the paged
8-bit AdamW state**, which bitsandbytes keeps in CUDA managed memory outside PyTorch's counters. That matters only
for full FT: 2 bytes x {p_proxy / 1e9:.2f}B params = **{full_opt_gb:.2f} GB ESTIMATED**, so full FT's true peak is
about {full_corr:.2f} GB (ESTIMATED). LoRA/QLoRA optimizer state is ~{2 * q['trainable_params'] / GB:.3f} GB.

Tokens/s = non-padding tokens / training wall time excluding eval. "best val loss (trainer)" is the Trainer's
eval_loss (mean of per-batch losses); "val ppl (eval)" is exp of the token-weighted response NLL from `qlora eval`,
so exp(loss) and ppl differ slightly by construction. ROUGE-L on the {res['qlora']['eval']['n_test']}-example test
split, greedy, same generation config.

GPU conditions (one run per method, so throughput differences are confounded by throttling):

| method | power-throttled samples | min SM clock MHz | max temp C | mean power W | training | fallback check before training |
|---|---:|---:|---:|---:|---|---|
{gpu_rows}

### Claim 1: memory (proxy, measured)

| QLoRA reduction vs | peak allocated | peak reserved |
|---|---:|---:|
| bf16 LoRA | **{red['alloc_vs_lora']}%** | {red['reserved_vs_lora']}% |
| full FT | **{red['alloc_vs_full']}%** | {red['reserved_vs_full']}% |
| full FT incl. paged optimizer state (ESTIMATED) | ~{red['alloc_vs_full_corrected_est']}% | n/a |

**Claim 1 on peak allocated (memory the run actually needs): {'HELD' if mem_claim else 'DID NOT HOLD'}**:
QLoRA is {'lower than both' if mem_claim else 'not lower than both'} alternatives.
**On peak reserved (allocator cache included): {'HELD' if res_claim else 'DID NOT HOLD'}**:
QLoRA {'is lower than both' if res_claim else 'is not lower than both'} alternatives
({red['reserved_vs_lora']}% vs bf16 LoRA, {red['reserved_vs_full']}% vs full FT; negative = QLoRA reserved more).

Likely reason the proxy gap is small (reasoning, not separately measured): on a 1.1B model, weights are only ~2.1 GB in bf16, so 4-bit quantization saves
at most ~1.5 GB; activations and the fp32 logits over the 32,064-token vocabulary dominate the peak and are the
same for every method. Reserved memory adds PyTorch's cached blocks, which depend on allocation order and
fragmentation rather than on what each method needs, so it is a noisy basis for comparing methods. The savings
should grow with model size, since weights are the part quantization shrinks; the 7B section supports this with one
measurement against estimates, not a sweep.

### Claim 2: quality parity (proxy, measured)

QLoRA ROUGE-L {res['qlora']['eval']['tuned']['rougeL']:.2f} vs full FT {res['full']['eval']['tuned']['rougeL']:.2f}:
difference **{parity['delta']:+.2f}** points (paired over {n_pair} test rows, 95% bootstrap CI
[{d_lo:+.2f}, {d_hi:+.2f}]); tolerance ±{tol} → **{'PASS' if parity['pass'] else 'FAIL'}** on the point estimate.
{'The whole CI lies inside the tolerance.' if -tol <= d_lo and d_hi <= tol else 'The CI extends beyond the tolerance, so this sample cannot rule out a gap larger than ' + str(tol) + ' point(s).'}
Caveats: one seed per method; full FT used lr 2e-5 vs 2e-4. Read this as "within tolerance, not distinguishable
from noise at this sample size", not as proof of equality. (bf16 LoRA: {res['lora']['eval']['tuned']['rougeL']:.2f}.)
{parity['note']}

## Llama-2-7B

| method | peak GB | source |
|---|---:|---|
| QLoRA | {q7:.2f} allocated / {seven['anchor'].get('qlora_7b_peak_reserved_gb', float('nan')):.2f} reserved | **MEASURED** (Phase 2, `runs/llama2-7b-qlora`) |
| bf16 LoRA | {s_l:.1f} | **ESTIMATED** |
| full FT, paged 8-bit AdamW | {s_f8:.1f} | **ESTIMATED** |
| full FT, standard fp32 AdamW | {s_f32:.1f} | **ESTIMATED** |

P = {seven['params_b']}B parameters (from config.json). Activations + LoRA optimizer state =
{seven['anchor']['activations_plus_lora_state_gb']} GB, taken from the measured 7B QLoRA run (peak allocated
{q7:.2f} GB minus {seven['anchor']['qlora_7b_after_load_gb']:.2f} GB after load) and assumed equal for the other
methods (same architecture, batch 4, seq <= 1024, gradient checkpointing). "After load" is
`max_memory_allocated` right after loading, so it includes load transients and fp32-upcast norms/embeddings.
Anchor check on the proxy, where bf16 LoRA was measured: the same formula predicts
{proxy_pred_lora:.2f} GB vs {lo['peak_mem_allocated_gb']:.2f} GB measured ({anchor_err:+.0f}%). Treat the 7B
estimates as approximate, about ±1-2 GB.
- bf16 LoRA = {seven['bf16_lora']['formula']} = {seven['bf16_lora']['weights_gb']} + {seven['bf16_lora']['activations_plus_lora_gb']} = {s_l} GB
- full FT (8-bit) = {seven['full_ft_8bit_adam']['formula']} = {s_f8} GB
- full FT (fp32 Adam) = {seven['full_ft_fp32_adam']['formula']} = {s_f32} GB

7B QLoRA vs ESTIMATED bf16 LoRA: {pct_reduction(q7, s_l)}% lower; vs ESTIMATED full FT (8-bit Adam):
{pct_reduction(q7, s_f8)}% lower (fp32 Adam: {pct_reduction(q7, s_f32)}%). These compare QLoRA's peak *allocated*;
on *reserved*, measured 7B QLoRA used {seven['anchor'].get('qlora_7b_peak_reserved_gb', float('nan')):.2f} GB, only
{pct_reduction(seven['anchor'].get('qlora_7b_peak_reserved_gb', float('nan')), s_l)}% below the bf16 LoRA estimate
(the proxy's reserved-memory caveat applies). bf16 LoRA at ~{s_l:.1f} GB ESTIMATED is at or just above the
{phys:.1f} GB card (within estimate error); full FT is far beyond it. That is why both are estimated, not measured.

## Identical across the three proxy runs (asserted)

| setting | value |
|---|---|
{same_md}

Learning rate is the one deliberate difference: 2e-4 for LoRA/QLoRA (the QLoRA default), 2e-5 for full FT
(2e-4 is far too high for updating every weight).
"""
    p = ensure_dir(reports_dir()) / "benchmark.md"
    p.write_text(md, encoding="utf-8")
    return p, red, mem_claim, res_claim


def run_bench(force: bool = False) -> dict[str, Any]:
    import mlflow

    from qlora_pipeline.data import load_split
    from qlora_pipeline.paths import logs_dir

    telemetry.require_mlflow()
    cfgs = {m: load_config(c) for m, c in METHODS}
    same = check_identical(cfgs)
    n_train = len(load_split(cfgs["qlora"], "train"))
    logs = ensure_dir(logs_dir())
    res = {m: run_one(m, c, cfgs[m], n_train, logs, force) for m, c in METHODS}

    seven_cfg = load_config(SEVEN_B_CONFIG)
    q7 = json.loads((runs_dir() / seven_cfg.train.output_name / "summary.json").read_text(encoding="utf-8"))
    seven = seven_b_estimates(q7)
    seven["anchor"]["qlora_7b_peak_reserved_gb"] = q7["peak_mem_reserved_gb"]

    tol = cfgs["qlora"].eval.rouge_tolerance
    delta = res["qlora"]["eval"]["tuned"]["rougeL"] - res["full"]["eval"]["tuned"]["rougeL"]
    ok = abs(delta) <= tol
    note = "" if ok else (
        "Parity FAILED. Next experiment: match the update budget by sweeping the QLoRA lr (1e-4, 3e-4) and r (32, 64), "
        "and train full FT for the same number of steps with lr 1e-5/5e-5, then re-compare on the same test split."
    )
    parity = {"delta": round(delta, 2), "pass": ok, "tolerance": tol, "note": note}
    report, red, mem_claim, res_claim = write_report(cfgs, same, res, seven, tol, parity)

    with telemetry.start_run(cfgs["qlora"], "bench-comparison", {"stage": "bench"}) as run:
        for m in ("full", "lora", "qlora"):
            t, e = res[m]["train"], res[m]["eval"]["tuned"]
            mlflow.log_metrics({f"{m}.peak_mem_allocated_gb": t["peak_mem_allocated_gb"],
                                f"{m}.peak_mem_reserved_gb": t["peak_mem_reserved_gb"],
                                f"{m}.tokens_per_sec": t["tokens_per_sec"], f"{m}.wall_time_s": t["wall_time_s"],
                                f"{m}.best_eval_loss": t["best_eval_loss"], f"{m}.val_perplexity": e["val_perplexity"],
                                f"{m}.rougeL": e["rougeL"]})  # fmt: skip
            mlflow.set_tag(f"{m}.train_run_id", t["mlflow_run_id"])
        mlflow.log_metrics({f"reduction.{k}_pct": v for k, v in red.items()})
        mlflow.log_metrics({"parity.delta_rougeL": parity["delta"], "parity.pass": float(ok),
                            "claim1.memory_held": float(mem_claim), "claim1.reserved_held": float(res_claim),
                            "est7b.bf16_lora_gb": seven["bf16_lora"]["total_gb"],
                            "est7b.full_ft_8bit_gb": seven["full_ft_8bit_adam"]["total_gb"],
                            "est7b.full_ft_fp32_gb": seven["full_ft_fp32_adam"]["total_gb"]})  # fmt: skip
        mlflow.log_dict({"same": same, "seven_b": seven, "parity": parity, "reductions": red}, "benchmark.json")
        mlflow.log_artifact(str(report))
        run_id = run.info.run_id
    out = {"report": str(report), "reductions": red, "parity": parity, "memory_claim_held": mem_claim,
           "reserved_claim_held": res_claim,
           "seven_b": seven, "mlflow_run_id": run_id, "results": res}  # fmt: skip
    (reports_dir() / "benchmark.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
