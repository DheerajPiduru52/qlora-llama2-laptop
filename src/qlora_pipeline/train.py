"""Phase 2 training driver (TRL SFTTrainer) for qlora | lora | full, plus the 50-step smoke test.

- Data: pre-tokenized with data.encode_example (response-only labels); TRL's preparation is skipped.
- Checkpoints every `save_steps` to runs/<name>/ (resume after laptop throttling / interruptions).
- Eval each epoch; the best-val-loss weights are written to artifacts/adapters/<name>/ by a callback.
- MLflow: one run per output_name (re-attached on resume): config, git SHA, dataset hash, losses, lr,
  trainable %, peak VRAM (allocated + reserved), tokens/sec, GPU temp/power/clock, adapter artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import yaml

from qlora_pipeline import telemetry
from qlora_pipeline.config import Config
from qlora_pipeline.data import encode_example, load_split, load_tokenizer, processed_dir
from qlora_pipeline.paths import adapters_dir, ensure_dir, runs_dir

GB = 1024**3


def verify_dataset(cfg: Config) -> str:
    """Check the processed files still match their manifest; return the dataset hash."""
    man = json.loads((processed_dir(cfg) / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in man["files"].items():
        actual = hashlib.sha256((processed_dir(cfg) / name).read_bytes()).hexdigest()
        if actual != sha:
            raise RuntimeError(f"{name} changed since `qlora data` (sha mismatch). Re-run `qlora data`.")
    return man["dataset_hash"]


def resume_fingerprint(cfg: Config) -> str:
    """Config fingerprint that ignores the resume setting itself."""
    return cfg.model_copy(update={"train": cfg.train.model_copy(update={"resume": "auto"})}).fingerprint()


def build_dataset(cfg: Config, tok, split: str):
    from datasets import Dataset

    ids, labels = [], []
    for r in load_split(cfg, split):
        enc = encode_example(tok, r["system"], r["instruction"], r["output"])
        if enc.n_tokens > cfg.data.max_seq_length:
            raise ValueError(f"{split} example longer than max_seq_length ({enc.n_tokens})")
        ids.append(enc.input_ids)
        labels.append(enc.labels)
    return Dataset.from_dict({"input_ids": ids, "labels": labels})


def _callbacks(cfg: Config, tok, monitor: telemetry.GpuMonitor, best_dir: Path, state: dict[str, Any]):
    import mlflow
    import torch
    from transformers import TrainerCallback

    from qlora_pipeline.model import save_adapter_or_model

    class MlflowLog(TrainerCallback):
        def on_train_begin(self, args, st, control, **kw):
            state["tokens_at_start"] = st.num_input_tokens_seen
            state["step_at_start"] = st.global_step

        def on_log(self, args, st, control, logs=None, **kw):
            m = {k: float(v) for k, v in (logs or {}).items() if isinstance(v, int | float) and math.isfinite(v)}
            m.update(monitor.current())
            m["mem.allocated_gb"] = torch.cuda.memory_allocated() / GB
            m["mem.max_reserved_gb"] = torch.cuda.max_memory_reserved() / GB
            mlflow.log_metrics(m, step=st.global_step)

    class KeepBest(TrainerCallback):
        def on_evaluate(self, args, st, control, metrics=None, model=None, **kw):
            loss = (metrics or {}).get("eval_loss")
            if loss is None:
                return
            mlflow.log_metric("eval_perplexity", math.exp(loss), step=st.global_step)
            if loss < state.get("best_eval_loss", float("inf")):
                state.update(best_eval_loss=loss, best_step=st.global_step, best_epoch=st.epoch)
                save_adapter_or_model(model, tok, best_dir)
                (best_dir / "best.json").write_text(
                    json.dumps({"eval_loss": loss, "step": st.global_step, "epoch": st.epoch}), encoding="utf-8"
                )

    return [MlflowLog(), KeepBest()]


def _mlflow_run(cfg: Config, out_dir: Path, stage: str, dataset_hash: str, resuming: bool):
    import mlflow

    id_file = out_dir / "mlflow_run_id.txt"
    tags = {"stage": stage, "method": cfg.method, "model": cfg.model.name, "dataset_hash": dataset_hash,
            "output_name": cfg.train.output_name}  # fmt: skip
    if resuming and id_file.exists():
        mlflow.set_tracking_uri(telemetry.tracking_uri())
        run = mlflow.start_run(run_id=id_file.read_text(encoding="utf-8").strip())
        mlflow.set_tags({"resumed": "true", "git_sha_resume": telemetry.git_sha()})
        return run, False
    run = telemetry.start_run(cfg, cfg.train.output_name, tags)
    id_file.write_text(run.info.run_id, encoding="utf-8")
    return run, True


def run_train(cfg: Config, stage: str = "train") -> dict[str, Any]:
    import mlflow
    import torch
    from transformers import set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    from qlora_pipeline.model import build_for_training, param_counts, save_adapter_or_model

    telemetry.require_mlflow()
    dataset_hash = verify_dataset(cfg)
    set_seed(cfg.seed)
    tc = cfg.train
    out_dir = ensure_dir(runs_dir() / tc.output_name)
    best_dir = adapters_dir() / tc.output_name

    resume_from: str | None = None
    if tc.resume == "auto":
        resume_from = get_last_checkpoint(str(out_dir))
    elif tc.resume not in ("none", ""):
        resume_from = tc.resume
    run_meta = {"config_fingerprint": resume_fingerprint(cfg), "dataset_hash": dataset_hash, "model": cfg.model.name}
    meta_file = out_dir / "run_meta.json"
    if resume_from and meta_file.exists():
        old = json.loads(meta_file.read_text(encoding="utf-8"))
        diff = {k: (old.get(k), v) for k, v in run_meta.items() if old.get(k) != v}
        if diff:
            raise RuntimeError(
                f"{resume_from} was made with a different config/data/model {diff}. "
                "Use a new train.output_name, or --set train.resume=none to start over."
            )
    meta_file.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    tok = load_tokenizer(cfg.model)
    train_ds = build_dataset(cfg, tok, "train")
    eval_ds = build_dataset(cfg, tok, "val") if tc.eval_strategy != "no" else None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # peak covers weights + activations + optimizer for THIS run
    model = build_for_training(cfg, tok)
    trainable, total = param_counts(model)
    mem_after_load = torch.cuda.max_memory_allocated() / GB

    args = SFTConfig(
        output_dir=str(out_dir),
        seed=cfg.seed,
        data_seed=cfg.seed,
        num_train_epochs=tc.num_train_epochs,
        max_steps=tc.max_steps,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.grad_accum,
        learning_rate=tc.learning_rate,
        lr_scheduler_type=tc.lr_scheduler_type,
        warmup_steps=tc.warmup_ratio,  # float in [0,1) = ratio of total steps (transformers 5)
        max_grad_norm=tc.max_grad_norm,
        weight_decay=tc.weight_decay,
        optim=tc.optim,
        bf16=tc.bf16,
        activation_checkpointing=False,  # set up in model.build_for_training (non-reentrant); v5 name
        eval_strategy=tc.eval_strategy,
        eval_steps=tc.eval_steps,
        save_strategy="steps",
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        logging_steps=tc.logging_steps,
        logging_first_step=True,
        dataloader_num_workers=tc.dataloader_num_workers,
        include_num_input_tokens_seen="non_padding",
        report_to="none",
        max_length=cfg.data.max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        torch_compile=False,
    )
    monitor = telemetry.GpuMonitor(tc.telemetry_interval_s)
    state: dict[str, Any] = {}
    best_file = best_dir / "best.json"
    if resume_from and best_file.exists():  # keep the pre-interruption best; only a better eval replaces it
        prev = json.loads(best_file.read_text(encoding="utf-8"))
        state.update(best_eval_loss=prev["eval_loss"], best_step=prev["step"], best_epoch=prev["epoch"])
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tok.pad_token_id, pad_to_multiple_of=tc.pad_to_multiple_of
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tok,
        data_collator=collator, callbacks=_callbacks(cfg, tok, monitor, best_dir, state),
    )  # fmt: skip

    run, fresh = _mlflow_run(cfg, out_dir, stage, dataset_hash, resuming=resume_from is not None)
    try:
        monitor.start()
        if fresh:
            mlflow.log_params(telemetry.flatten(cfg.to_dict()))
            mlflow.log_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), "resolved_config.yaml")
            mlflow.log_params({"grad_accum": tc.grad_accum, "train_rows": len(train_ds),
                               "val_rows": len(eval_ds) if eval_ds is not None else 0})  # fmt: skip
        mlflow.log_metrics({
            "params.trainable": trainable, "params.total": total,
            "params.trainable_pct": 100 * trainable / total, "mem.after_load_gb": mem_after_load,
        })  # fmt: skip

        t0 = time.time()
        result = trainer.train(resume_from_checkpoint=resume_from)
        wall = time.time() - t0
        monitor.stop()

        peak_alloc = torch.cuda.max_memory_allocated() / GB
        peak_reserved = torch.cuda.max_memory_reserved() / GB
        tokens = trainer.state.num_input_tokens_seen - state.get("tokens_at_start", 0)
        start_step = state.get("step_at_start", 0)
        eval_time = sum(h.get("eval_runtime", 0) for h in trainer.state.log_history if h.get("step", 0) > start_step)
        train_time = max(wall - eval_time, 1e-9)
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]

        if eval_ds is None:  # no eval -> final weights are the result
            save_adapter_or_model(trainer.model, tok, best_dir)
        elif not best_file.exists():
            raise RuntimeError("training with eval finished but no best adapter was saved")
        summary = {
            "output_name": tc.output_name, "method": cfg.method, "model": cfg.model.name, "stage": stage,
            "mlflow_run_id": run.info.run_id, "git_sha": telemetry.git_sha(), "dataset_hash": dataset_hash,
            "resumed_from": resume_from, "global_steps": trainer.state.global_step,
            "trainable_params": trainable, "total_params": total, "trainable_pct": 100 * trainable / total,
            "peak_mem_allocated_gb": round(peak_alloc, 3), "peak_mem_reserved_gb": round(peak_reserved, 3),
            "mem_after_load_gb": round(mem_after_load, 3),
            "wall_time_s": round(wall, 1), "train_runtime_s": result.metrics.get("train_runtime"),
            "tokens_seen_this_session": tokens, "tokens_per_sec": round(tokens / train_time, 1),
            "eval_time_s": round(eval_time, 1),
            "train_loss_first": losses[0] if losses else None, "train_loss_last": losses[-1] if losses else None,
            "best_eval_loss": state.get("best_eval_loss"), "best_step": state.get("best_step"),
            "best_val_perplexity": math.exp(state["best_eval_loss"]) if "best_eval_loss" in state else None,
            "adapter_dir": str(best_dir), "loss_history": losses,
            **monitor.summary(),
        }  # fmt: skip
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_metrics({k: float(v) for k, v in summary.items()
                            if isinstance(v, int | float) and not isinstance(v, bool)})  # fmt: skip
        mlflow.log_dict(summary, "summary.json")
        mlflow.log_artifacts(str(best_dir), artifact_path="adapter")
        return summary
    finally:
        monitor.stop()
        mlflow.end_run()


def smoke_verdict(losses: list[float], k: int = 5) -> tuple[bool, float, float]:
    """PASS if the mean of the last k logged losses is below the mean of the first k."""
    if len(losses) < 2 * k:
        raise ValueError(f"need >= {2 * k} logged losses, got {len(losses)}")
    first, last = sum(losses[:k]) / k, sum(losses[-k:]) / k
    return last < first, first, last
