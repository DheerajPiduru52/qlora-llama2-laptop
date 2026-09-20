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


def run_one(method: str, cfg_name: str, cfg: Config, n_train: int, logs: Path, force: bool) -> dict[str, Any]:
    out = runs_dir() / cfg.train.output_name
    summ_path = out / "summary.json"
    exp = _expected_steps(cfg, n_train)
    done = summ_path.exists() and json.loads(summ_path.read_text(encoding="utf-8")).get("global_steps") == exp
    if force or not done:
        assert_sysmem_fallback_off()
        _cli("train", "-c", cfg_name, "--set", "train.resume=none", log=logs / f"bench_train_{method}.log")
    else:
        print(f"{method}: reusing completed training ({exp} steps) at {out}", flush=True)
    train = json.loads(summ_path.read_text(encoding="utf-8"))
    telemetry.check_no_sysmem_fallback(train["peak_mem_reserved_gb"], f"{method} training")

    tag = f"bench-{method}"
    ev_path = reports_dir() / f"eval_{tag}.json"
    if force or not done or not ev_path.exists():
        assert_sysmem_fallback_off()
        _cli("eval", "-c", cfg_name, "--tag", tag, "--skip-lm-eval", "--no-record-peak",
             "--set", "eval.compare_base=false", log=logs / f"bench_eval_{method}.log")  # fmt: skip
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    telemetry.check_no_sysmem_fallback(ev["tuned"]["peak_mem_reserved_gb"], f"{method} eval")
    return {"train": train, "eval": ev}


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


def write_report(cfgs, same, res, seven, tol, parity) -> tuple[Path, dict[str, float], bool]:
    phys = telemetry.physical_vram_gb()
    rows = ""
    for m in ("full", "lora", "qlora"):
        t, e = res[m]["train"], res[m]["eval"]["tuned"]
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
    same_md = "\n".join(f"| `{k}` | `{v}` |" for k, v in same.items())
    md = f"""# Benchmark: QLoRA vs bf16 LoRA vs full fine-tuning

Proxy: `{cfgs['qlora'].model.name}` @ `{cfgs['qlora'].model.revision}` (Llama 2 architecture and tokenizer).
Dataset hash `{q['dataset_hash']}`. GPU: RTX 5080 Laptop, {phys:.2f} GB physical VRAM; NVIDIA sysmem fallback
verified OFF before every run, and every peak below is <= physical VRAM (no spill to system memory).
`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6` for all three runs.

## Proxy results (MEASURED, 1 epoch = {q['global_steps']} optimizer steps each)

| method | lr | trainable | peak allocated GB | peak reserved GB | tokens/s | wall min | best val loss | val ppl | test ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}
Peak memory = `torch.cuda.max_memory_allocated` / `max_memory_reserved`, reset before model load, so it covers
weights + gradients + optimizer + activations + eval. Tokens/s = non-padding tokens / training wall time
excluding eval. ROUGE-L on the {res['qlora']['eval']['n_test']}-example test split, greedy, same generation config.

### Claim 1: memory (proxy, measured)

| QLoRA reduction vs | peak allocated | peak reserved |
|---|---:|---:|
| bf16 LoRA | **{red['alloc_vs_lora']}%** | {red['reserved_vs_lora']}% |
| full FT | **{red['alloc_vs_full']}%** | {red['reserved_vs_full']}% |

**Claim 1 {'HELD' if mem_claim else 'DID NOT HOLD'}** on the proxy: QLoRA's peak allocated memory is
{'lower than both' if mem_claim else 'not lower than both'} alternatives.

### Claim 2: quality parity (proxy, measured)

QLoRA ROUGE-L {res['qlora']['eval']['tuned']['rougeL']:.2f} vs full FT {res['full']['eval']['tuned']['rougeL']:.2f}:
difference **{parity['delta']:+.2f}** points; tolerance ±{tol} → **{'PASS' if parity['pass'] else 'FAIL'}**.
(bf16 LoRA: {res['lora']['eval']['tuned']['rougeL']:.2f}.)
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
methods (same architecture, batch 4, seq <= 1024, gradient checkpointing).
- bf16 LoRA = {seven['bf16_lora']['formula']} = {seven['bf16_lora']['weights_gb']} + {seven['bf16_lora']['activations_plus_lora_gb']} = {s_l} GB
- full FT (8-bit) = {seven['full_ft_8bit_adam']['formula']} = {s_f8} GB
- full FT (fp32 Adam) = {seven['full_ft_fp32_adam']['formula']} = {s_f32} GB

7B QLoRA vs ESTIMATED bf16 LoRA: {pct_reduction(q7, s_l)}% lower; vs ESTIMATED full FT (8-bit Adam):
{pct_reduction(q7, s_f8)}% lower (fp32 Adam: {pct_reduction(q7, s_f32)}%). Neither 7B alternative fits in
{phys:.0f} GB, which is why they are estimated rather than measured.

## Identical across the three proxy runs (asserted)

| setting | value |
|---|---|
{same_md}

Learning rate is the one deliberate difference: 2e-4 for LoRA/QLoRA (the QLoRA default), 2e-5 for full FT
(2e-4 is far too high for updating every weight).
"""
    p = ensure_dir(reports_dir()) / "benchmark.md"
    p.write_text(md, encoding="utf-8")
    return p, red, mem_claim


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
    report, red, mem_claim = write_report(cfgs, same, res, seven, tol, parity)

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
                            "claim1.memory_held": float(mem_claim),
                            "est7b.bf16_lora_gb": seven["bf16_lora"]["total_gb"],
                            "est7b.full_ft_8bit_gb": seven["full_ft_8bit_adam"]["total_gb"],
                            "est7b.full_ft_fp32_gb": seven["full_ft_fp32_adam"]["total_gb"]})  # fmt: skip
        mlflow.log_dict({"same": same, "seven_b": seven, "parity": parity, "reductions": red}, "benchmark.json")
        mlflow.log_artifact(str(report))
        run_id = run.info.run_id
    out = {"report": str(report), "reductions": red, "parity": parity, "memory_claim_held": mem_claim,
           "seven_b": seven, "mlflow_run_id": run_id, "results": res}  # fmt: skip
    (reports_dir() / "benchmark.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
