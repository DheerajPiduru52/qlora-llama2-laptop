"""MLflow + provenance helpers (GPU telemetry sampling is added in Phase 2)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from qlora_pipeline.paths import repo_dir


def git_sha() -> str:
    """HEAD SHA, suffixed with -dirty if the working tree has uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir(), text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_dir(), text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def require_mlflow(timeout: float = 3.0) -> None:
    import httpx

    uri = tracking_uri()
    try:
        r = httpx.get(uri.rstrip("/") + "/health", timeout=timeout)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"MLflow server not reachable at {uri} ({e}). Start it in another shell: `qlora mlflow`."
        ) from e


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v if not isinstance(v, list) else ",".join(map(str, v))
    return out


def start_run(cfg, run_name: str, tags: dict[str, str] | None = None):
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(cfg.experiment)
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tags({"git_sha": git_sha(), "config_fingerprint": cfg.fingerprint(), **(tags or {})})
    return run


def log_data_run(cfg, stats: dict, splits: dict, card: Path, manifest: Path) -> str:
    import mlflow

    tags = {"stage": "data", "dataset_hash": stats["dataset_hash"]}
    with start_run(cfg, f"data-{stats['dataset_hash'][:8]}", tags) as run:
        params = flatten({"data": cfg.data.model_dump(mode="json")})
        params.update({"seed": cfg.seed, "tokenizer": cfg.model.tokenizer_name, "dataset_hash": stats["dataset_hash"]})
        mlflow.log_params(params)
        metrics = {f"rows.{k}": v for k, v in stats["funnel"].items()}
        metrics.update({f"leak.{k}": v for k, v in stats["leakage"].items()})
        for name, pct in stats["token_lengths"].items():
            metrics.update({f"tokens.{name}.{k}": v for k, v in pct.items()})
        mlflow.log_metrics(metrics)
        for s, df in splits.items():
            d = cfg.data
            src = f"https://huggingface.co/datasets/{d.hf_dataset}/resolve/{d.hf_revision or 'main'}/{d.hf_filename}"
            ds = mlflow.data.from_pandas(df, source=src, name=f"{cfg.data.processed_name}-{s}")
            mlflow.log_input(ds, context=s)
        mlflow.log_artifact(str(card))
        mlflow.log_artifact(str(manifest))
        return run.info.run_id
