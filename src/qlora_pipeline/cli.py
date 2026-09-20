"""`qlora` CLI. The env bootstrap runs at import time, before any heavy library is imported.
Heavy imports (torch, transformers, ...) live inside the command functions.
"""

from __future__ import annotations

from qlora_pipeline.bootstrap import load_env

load_env()  # must stay above every other project / third-party import

import subprocess  # noqa: E402
import sys  # noqa: E402

import typer  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Local QLoRA fine-tuning pipeline")


@app.command("env-check")
def env_check(
    skip_hf: bool = typer.Option(False, help="Skip the gated Llama 2 access check"),
) -> None:
    """GPU / torch / bitsandbytes diagnostics, a real 4-bit fwd+bwd, HF access, disk, cache paths."""
    from qlora_pipeline.envcheck import run_env_check

    raise typer.Exit(0 if run_env_check(skip_hf=skip_hf) else 1)


@app.command("audit-c")
def audit_c(
    baseline: bool = typer.Option(False, "--baseline", help="Record the baseline instead of comparing"),
    force: bool = typer.Option(False, help="Overwrite an existing baseline"),
    threshold_mb: float = typer.Option(50.0, help="Max allowed growth per location"),
) -> None:
    """C: drive footprint audit against logs/c_baseline.json."""
    from qlora_pipeline import audit

    args = ["--threshold-mb", str(threshold_mb)]
    if baseline:
        args.append("--baseline")
    if force:
        args.append("--force")
    raise typer.Exit(audit.main(args))


@app.command("mlflow")
def mlflow_server(
    host: str = "127.0.0.1",
    port: int = 5000,
) -> None:
    """Start the local MLflow tracking server (sqlite backend + artifacts under QLORA_ROOT/mlflow)."""
    from qlora_pipeline.paths import ensure_dir, mlflow_dir

    mdir = mlflow_dir()
    art = ensure_dir(mdir / "artifacts")
    db = (mdir / "mlflow.db").as_posix()
    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", f"sqlite:///{db}",
        "--default-artifact-root", art.as_uri(),
        "--host", host, "--port", str(port),
    ]  # fmt: skip
    typer.echo("Running: " + " ".join(cmd))
    raise typer.Exit(subprocess.call(cmd))


ConfigOpt = typer.Option("base.yaml", "--config", "-c", help="YAML in configs/ or a path")
SetOpt = typer.Option(None, "--set", help="Override, e.g. --set data.subsample_size=3000 (repeatable)")


@app.command("data")
def data_cmd(
    config: str = ConfigOpt,
    set_: list[str] = SetOpt,
    no_mlflow: bool = typer.Option(False, "--no-mlflow", help="Skip MLflow logging"),
) -> None:
    """Phase 1: download, clean, subsample, split, data card, log dataset hash to MLflow."""
    from qlora_pipeline.config import load_config
    from qlora_pipeline.data import run_data

    cfg = load_config(config, set_)
    res = run_data(cfg, log_mlflow=not no_mlflow)
    typer.echo(f"funnel:       {res['funnel']}")
    typer.echo(f"leakage:      {res['leakage']}")
    typer.echo(f"token p50/p95/p99/max (before filter): "
               f"{[res['token_lengths']['corpus_before_filter'][k] for k in ('p50', 'p95', 'p99', 'max')]}")  # fmt: skip
    typer.echo(f"dataset hash: {res['dataset_hash']}")
    typer.echo(f"output:       {res['out_dir']}")
    typer.echo(f"data card:    {res['data_card']}")


def _print_summary(s: dict) -> None:
    keys = ["output_name", "method", "global_steps", "trainable_pct", "peak_mem_allocated_gb", "peak_mem_reserved_gb",
            "tokens_per_sec", "wall_time_s", "train_loss_first", "train_loss_last", "best_eval_loss",
            "best_val_perplexity", "gpu.temp_max_c", "gpu.power_mean_w", "gpu.sm_clock_min_mhz",
            "gpu.throttle_thermal_pct", "adapter_dir", "mlflow_run_id"]  # fmt: skip
    for k in keys:
        typer.echo(f"  {k:24} {s.get(k)}")


@app.command("train")
def train_cmd(config: str = typer.Option("llama2_7b_qlora.yaml", "--config", "-c"), set_: list[str] = SetOpt) -> None:
    """Phase 2: train (qlora | lora | full per config). Resumes from the last checkpoint automatically."""
    from qlora_pipeline.config import load_config
    from qlora_pipeline.train import run_train

    _print_summary(run_train(load_config(config, set_), stage="train"))


@app.command("smoke")
def smoke_cmd(
    config: str = typer.Option("proxy_qlora.yaml", "--config", "-c"),
    steps: int = typer.Option(50, help="optimizer steps"),
    set_: list[str] = SetOpt,
) -> None:
    """Phase 2 gate: short proxy QLoRA run; PASS only if the loss goes down."""
    from qlora_pipeline.config import load_config
    from qlora_pipeline.train import run_train, smoke_verdict

    over = [f"train.max_steps={steps}", "train.eval_strategy=no", "train.logging_steps=1", "train.resume=none",
            f"train.save_steps={steps * 10}", "train.output_name=smoke-proxy-qlora", *(set_ or [])]  # fmt: skip
    s = run_train(load_config(config, over), stage="smoke")
    _print_summary(s)
    ok, first, last = smoke_verdict(s["loss_history"])
    typer.echo(f"smoke: mean loss first 5 steps {first:.4f} -> last 5 steps {last:.4f}: {'PASS' if ok else 'FAIL'}")
    raise typer.Exit(0 if ok else 1)


@app.command("eval")
def eval_cmd(
    config: str = typer.Option("llama2_7b_qlora.yaml", "--config", "-c"),
    set_: list[str] = SetOpt,
    skip_lm_eval: bool = typer.Option(False, "--skip-lm-eval", help="Skip the lm-evaluation-harness regression"),
    tag: str = typer.Option(None, help="Report name (default: train.output_name)"),
    no_record_peak: bool = typer.Option(False, "--no-record-peak", help="Do not store this run's commit peak"),
    replace_peak: bool = typer.Option(False, "--replace-peak", help="Store this run's peak, replacing the old one"),
) -> None:
    """Phase 3: base vs tuned on the test split (ROUGE-L, perplexity, slots, lm-eval forgetting check)."""
    from qlora_pipeline.config import load_config
    from qlora_pipeline.evaluate import run_eval

    s = run_eval(load_config(config, set_), do_lm_eval=not skip_lm_eval, tag=tag,
                 record_peak_commit=not no_record_peak, replace_peak=replace_peak)
    for k in ("base", "tuned"):
        if k in s:
            v = s[k]
            typer.echo(f"{k:6} ROUGE-L {v['rougeL']:.2f}  val ppl {v['val_perplexity']:.3f}  "
                       f"slot recall {v['slots.slot_recall_pct']:.1f}%  "
                       f"invented/row {v['slots.invented_slots_per_row']:.3f}  "
                       f"hit-max {v['hit_max_new_tokens_pct']}%  lm_eval {v['lm_eval']}")  # fmt: skip
    typer.echo(f"forgetting flags: {s['forgetting'] or 'none'}   mlflow run {s['mlflow_run_id']}")
    typer.echo(f"process peak commit {s['process_peak_commit_gb']} GB; physical VRAM {s['physical_vram_gb']} GB; "
               f"sysmem fallback detected: {s['sysmem_fallback_detected']}")
    for k in ("base", "tuned"):
        for st, m in s.get(k, {}).get("stage_mem", {}).items():
            typer.echo(f"  {k:5} {st:22} peak reserved {m['peak_reserved_gb']:6.2f} GB  allocated "
                       f"{m['peak_allocated_gb']:6.2f} GB{'  OOM' if m['oom'] else ''}")


@app.command("bench")
def bench_cmd(force: bool = typer.Option(False, help="Retrain/re-evaluate even if completed runs exist")) -> None:
    """Phase 4: proxy full FT vs bf16 LoRA vs QLoRA (memory, speed, ROUGE-L parity) + 7B estimates."""
    from qlora_pipeline.benchmark import run_bench

    out = run_bench(force=force)
    typer.echo(f"QLoRA peak-allocated reduction: vs LoRA {out['reductions']['alloc_vs_lora']}%, "
               f"vs full FT {out['reductions']['alloc_vs_full']}% -> memory claim "
               f"{'HELD' if out['memory_claim_held'] else 'DID NOT HOLD'} (allocated); reserved: "
               f"{'HELD' if out['reserved_claim_held'] else 'DID NOT HOLD'}")
    p = out["parity"]
    typer.echo(f"ROUGE-L QLoRA - full FT = {p['delta']:+.2f} (tolerance ±{p['tolerance']}): "
               f"{'PASS' if p['pass'] else 'FAIL'}")
    typer.echo(f"report: {out['report']}   mlflow run {out['mlflow_run_id']}")


@app.command("merge")
def merge_cmd(
    config: str = typer.Option("llama2_7b_qlora.yaml", "--config", "-c"),
    set_: list[str] = SetOpt,
    force: bool = typer.Option(False, help="Re-merge even if a merge of the same adapter exists"),
) -> None:
    """Phase 5: fp16 CPU merge -> validate (ROUGE-L merged vs adapter) -> MLflow registry alias."""
    from qlora_pipeline.config import load_config
    from qlora_pipeline.merge import run_merge

    s = run_merge(load_config(config, set_), config, skip_merge_if_present=not force, overrides=set_)
    v = s["validation"]
    typer.echo(f"merged: {s['merged_dir']}  (peak commit {s['merge_info']['process_peak_commit_gb']} GB)")
    typer.echo(f"validation n={v['n']}: merged {v['rougeL_merged']:.2f} vs adapter {v['rougeL_adapter']:.2f} "
               f"delta {v['delta']:+.2f} -> {'PASS' if v['pass'] else 'FAIL'}; identical outputs "
               f"{v['identical_outputs_pct']}%")
    typer.echo(f"registered: {s['registered'] or 'NOT registered (validation failed)'}   mlflow run {s['mlflow_run_id']}")
    raise typer.Exit(0 if v["pass"] else 1)


def _later(phase: int):
    def _stub() -> None:
        typer.echo(f"Not implemented yet: arrives in Phase {phase}.")
        raise typer.Exit(2)

    return _stub


for _name, _phase in [

    ("serve", 6), ("chat", 6), ("load-test", 6),
]:  # fmt: skip
    app.command(_name, help=f"(Phase {_phase})")(_later(_phase))


if __name__ == "__main__":
    app()
