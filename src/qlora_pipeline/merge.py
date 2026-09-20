"""Phase 5: merge the QLoRA adapter into the fp16 base ON CPU, validate, register in the MLflow Model Registry.

1. Pre-flight (C: + commit headroom, job "merge"), then load the base in fp16 on CPU (NOT 4-bit), add the same
   zero-initialised <pad> rows as training, attach the adapter, merge_and_unload(safe_merge=True), save
   safetensors + tokenizer to artifacts/merged/<name>. The merge process's peak commit is recorded.
2. Validate in separate processes: greedy ROUGE-L on the first N test rows for the adapter model (4-bit base +
   adapter, as trained) and the merged model. The fp16 merged model (13.5 GB) does not fit the 14 GB VRAM cap
   with a KV cache, so it is validated loaded in 4-bit NF4 (how Phase 6's fallback serves it).
3. Only if |delta ROUGE-L| <= tolerance: log adapter as an MLflow artifact, create a registered-model version
   whose source is that artifact (merged model referenced by local path in tags), and set the alias.
"""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from qlora_pipeline import telemetry
from qlora_pipeline.config import Config
from qlora_pipeline.paths import ensure_dir, logs_dir, merged_dir, reports_dir, runs_dir


def _dir_sha(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(d).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def merge_to_disk(cfg: Config, adapter: Path, out: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from qlora_pipeline.audit import process_commit_gb
    from qlora_pipeline.data import load_tokenizer
    from qlora_pipeline.model import model_path, resize_for_pad

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.merge.dtype]
    tok = load_tokenizer(cfg.model)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(model_path(cfg.model), dtype=dtype, device_map={"": "cpu"})
    rows_before = base.get_input_embeddings().weight.shape[0]
    resize_for_pad(base, tok)  # identical zero-init pad rows as in training / eval
    t_load = time.time() - t0
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    merged = model.merge_and_unload(safe_merge=True)
    t_merge = time.time() - t0 - t_load
    for n, prm in merged.named_parameters():
        if not torch.isfinite(prm).all():
            raise RuntimeError(f"non-finite values in merged parameter {n}")
    leftover = [n for n, _ in merged.named_modules() if "lora_" in n]
    if leftover:
        raise RuntimeError(f"LoRA modules left after merge: {leftover[:5]}")
    merged.config.use_cache = True
    ensure_dir(out)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out)
    t_save = time.time() - t0 - t_load - t_merge
    info = {
        "base_model": cfg.model.name, "base_revision": cfg.model.revision, "dtype": cfg.merge.dtype, "device": "cpu",
        "adapter_dir": str(adapter), "adapter_sha256": _dir_sha(adapter), "embedding_rows": [rows_before,
        merged.get_input_embeddings().weight.shape[0]], "pad_token_id": tok.pad_token_id,
        "load_s": round(t_load, 1), "merge_s": round(t_merge, 1), "save_s": round(t_save, 1),
        "process_peak_commit_gb": round(process_commit_gb().get("peak_private_gb", float("nan")), 2),
        "git_sha": telemetry.git_sha(),
        "files": {f.name: f.stat().st_size for f in sorted(out.iterdir()) if f.is_file()},
    }  # fmt: skip
    (out / "merge_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    del merged, model, base
    gc.collect()
    return info


def _validate(cfg_name: str, tag: str, weights: Path, n: int, log: Path) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "qlora_pipeline.cli", "eval", "-c", cfg_name, "--tag", tag, "--skip-lm-eval",
           "--no-record-peak", "--set", "eval.compare_base=false", "--set", f"eval.test_limit={n}",
           "--set", f"eval.weights={weights.as_posix()}", "--set", "eval.quantized=true"]  # fmt: skip
    print("running:", " ".join(cmd[3:]), f"(log: {log})", flush=True)
    with log.open("w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"validation eval {tag} failed (exit {rc}); see {log}")
    return json.loads((reports_dir() / f"eval_{tag}.json").read_text(encoding="utf-8"))


def _agreement(tag_a: str, tag_b: str) -> dict[str, float]:
    import pandas as pd

    a = pd.read_csv(reports_dir() / f"eval_{tag_a}_predictions.csv", encoding="utf-8")
    b = pd.read_csv(reports_dir() / f"eval_{tag_b}_predictions.csv", encoding="utf-8")
    cols = ["id", "pred_tuned", "rougeL_tuned"]
    m = a[cols].merge(b[cols], on="id", suffixes=("_a", "_b"))
    same = (m["pred_tuned_a"].fillna("") == m["pred_tuned_b"].fillna("")).mean()
    diff = float((m["rougeL_tuned_a"] - m["rougeL_tuned_b"]).abs().mean())
    return {"rows": len(m), "identical_outputs_pct": round(100 * float(same), 1),
            "mean_abs_row_rouge_diff": round(diff, 2)}  # fmt: skip


def run_merge(cfg: Config, cfg_name: str, skip_merge_if_present: bool = True) -> dict[str, Any]:
    import mlflow

    from qlora_pipeline.evaluate import resolve_weights
    from qlora_pipeline.model import preflight, record_peak

    name = cfg.train.output_name
    adapter = resolve_weights(cfg)
    out = merged_dir() / name
    logs = ensure_dir(logs_dir())
    telemetry.require_mlflow()

    existing = out / "merge_info.json"
    if skip_merge_if_present and existing.exists() and json.loads(existing.read_text(encoding="utf-8")).get(
        "adapter_sha256") == _dir_sha(adapter):  # fmt: skip
        info = json.loads(existing.read_text(encoding="utf-8"))
        print(f"merge: reusing {out} (same adapter sha256)", flush=True)
    else:
        preflight(cfg, f"merge {name} (CPU, {cfg.merge.dtype})", job="merge")
        info = merge_to_disk(cfg, adapter, out)
        record_peak(cfg, "merge", f"CPU {cfg.merge.dtype} merge {name}")
        print(f"merged -> {out}  (load {info['load_s']}s, merge {info['merge_s']}s, save {info['save_s']}s, "
              f"process peak commit {info['process_peak_commit_gb']} GB)", flush=True)  # fmt: skip

    n = cfg.merge.validate_n
    v_adapter = _validate(cfg_name, f"merge-validate-adapter-{name}", adapter, n, logs / "merge_validate_adapter.log")
    v_merged = _validate(cfg_name, f"merge-validate-merged-{name}", out, n, logs / "merge_validate_merged.log")
    r_a, r_m = v_adapter["tuned"]["rougeL"], v_merged["tuned"]["rougeL"]
    delta = r_m - r_a
    ok = abs(delta) <= cfg.merge.rouge_tolerance
    agree = _agreement(f"merge-validate-adapter-{name}", f"merge-validate-merged-{name}")
    validation = {"n": n, "rougeL_adapter": r_a, "rougeL_merged": r_m, "delta": round(delta, 3),
                  "tolerance": cfg.merge.rouge_tolerance, "pass": ok, **agree,
                  "merged_loaded_as": "4-bit NF4 (fp16 merge does not fit the 14 GB VRAM cap with a KV cache)",
                  "adapter_loaded_as": "4-bit NF4 base + LoRA adapter (as trained)"}  # fmt: skip
    print(f"validation: ROUGE-L merged {r_m:.2f} vs adapter {r_a:.2f} -> delta {delta:+.2f} "
          f"(tolerance ±{cfg.merge.rouge_tolerance}) {'PASS' if ok else 'FAIL'}; {agree}", flush=True)  # fmt: skip

    train_run = (runs_dir() / name / "mlflow_run_id.txt").read_text(encoding="utf-8").strip()
    registered: dict[str, Any] = {}
    with telemetry.start_run(cfg, f"merge-{name}", {"stage": "merge", "train_run_id": train_run,
                                                    "merged_path": str(out)}) as run:  # fmt: skip
        mlflow.log_params({"merge.dtype": cfg.merge.dtype, "merge.device": "cpu", "merge.validate_n": n,
                           "merge.rouge_tolerance": cfg.merge.rouge_tolerance})  # fmt: skip
        mlflow.log_metrics({"validate.rougeL_adapter": r_a, "validate.rougeL_merged": r_m,
                            "validate.delta": delta, "validate.pass": float(ok),
                            "validate.identical_outputs_pct": agree["identical_outputs_pct"],
                            "merge.process_peak_commit_gb": info["process_peak_commit_gb"]})  # fmt: skip
        mlflow.log_dict(info, "merge_info.json")
        mlflow.log_dict(validation, "validation.json")
        mlflow.log_artifacts(str(adapter), artifact_path="adapter")
        run_id = run.info.run_id
        if ok:
            client = mlflow.MlflowClient()
            try:
                client.create_registered_model(
                    cfg.merge.registry_name,
                    description="Llama-2-7b-chat QLoRA adapter for e-commerce customer support (Bitext).")
            except mlflow.exceptions.MlflowException as e:
                if "RESOURCE_ALREADY_EXISTS" not in str(e) and "already exists" not in str(e).lower():
                    raise
            src = f"{run.info.artifact_uri}/adapter"
            tags = {"merged_path": str(out), "base_model": cfg.model.name, "base_revision": cfg.model.revision,
                    "adapter_sha256": info["adapter_sha256"], "dataset_hash": json.loads((runs_dir() / name /
                    "summary.json").read_text(encoding="utf-8"))["dataset_hash"], "train_run_id": train_run,
                    "merge_run_id": run_id, "git_sha": telemetry.git_sha(),
                    "validation_rougeL_delta": f"{delta:+.3f}", "quantization_at_serve": "nf4 4-bit"}  # fmt: skip
            mv = client.create_model_version(cfg.merge.registry_name, source=src, run_id=run_id, tags=tags,
                                             description=f"adapter + merged fp16 at {out}")  # fmt: skip
            client.set_registered_model_alias(cfg.merge.registry_name, cfg.merge.alias, mv.version)
            registered = {"name": cfg.merge.registry_name, "version": mv.version, "alias": cfg.merge.alias,
                          "source": src, "merged_path": str(out)}  # fmt: skip
            mlflow.set_tags({"registered_model": cfg.merge.registry_name, "registered_version": mv.version})
    summary = {"merged_dir": str(out), "merge_info": info, "validation": validation, "mlflow_run_id": run_id,
               "registered": registered}  # fmt: skip
    (reports_dir() / f"merge_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
