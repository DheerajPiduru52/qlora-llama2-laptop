"""MLflow + provenance helpers and NVML GPU telemetry."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Any

from qlora_pipeline.paths import repo_dir


class GpuMonitor:
    """Background NVML sampler: current + extreme temperature, power, SM clock, utilisation, throttle flags."""

    def __init__(self, interval_s: float = 2.0, index: int = 0) -> None:
        self.interval_s = interval_s
        self.index = index
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._cur: dict[str, float] = {}
        self._agg: dict[str, float] = {}
        self._thread: threading.Thread | None = None

    def _sample(self, nv, h) -> dict[str, float]:
        s = {
            "temp_c": float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
            "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
            "sm_clock_mhz": float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
            "util_pct": float(nv.nvmlDeviceGetUtilizationRates(h).gpu),
        }
        try:  # thermal / power throttling active right now (bitmask of slowdown reasons)
            reasons = nv.nvmlDeviceGetCurrentClocksEventReasons(h)
            thermal = nv.nvmlClocksEventReasonSwThermalSlowdown | nv.nvmlClocksEventReasonHwThermalSlowdown
            power = nv.nvmlClocksEventReasonSwPowerCap | nv.nvmlClocksEventReasonHwPowerBrakeSlowdown
            s["throttle_thermal"] = float(bool(reasons & thermal))
            s["throttle_power"] = float(bool(reasons & power))
        except Exception:  # noqa: BLE001 - older drivers / bindings
            pass
        return s

    def _run(self) -> None:
        import pynvml as nv

        nv.nvmlInit()
        try:
            h = nv.nvmlDeviceGetHandleByIndex(self.index)
            while not self._stop.is_set():
                try:
                    s = self._sample(nv, h)
                except Exception:  # noqa: BLE001 - one bad NVML read must not kill the sampler
                    self._stop.wait(self.interval_s)
                    continue
                with self._lock:
                    self._cur = s
                    a = self._agg
                    a["temp_max_c"] = max(a.get("temp_max_c", 0), s["temp_c"])
                    a["power_max_w"] = max(a.get("power_max_w", 0), s["power_w"])
                    a["sm_clock_min_mhz"] = min(a.get("sm_clock_min_mhz", 1e9), s["sm_clock_mhz"])
                    a["throttle_thermal_samples"] = a.get("throttle_thermal_samples", 0) + s.get("throttle_thermal", 0)
                    a["throttle_power_samples"] = a.get("throttle_power_samples", 0) + s.get("throttle_power", 0)
                    a["samples"] = a.get("samples", 0) + 1
                    a["power_sum_w"] = a.get("power_sum_w", 0) + s["power_w"]
                self._stop.wait(self.interval_s)
        finally:
            nv.nvmlShutdown()

    def start(self) -> GpuMonitor:
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def current(self) -> dict[str, float]:
        with self._lock:
            return {f"gpu.{k}": v for k, v in self._cur.items()}

    def summary(self) -> dict[str, float]:
        with self._lock:
            a = dict(self._agg)
        n = max(a.pop("samples", 0), 1)
        a["power_mean_w"] = a.pop("power_sum_w", 0) / n
        a["throttle_thermal_pct"] = 100 * a.pop("throttle_thermal_samples", 0) / n
        a["throttle_power_pct"] = 100 * a.pop("throttle_power_samples", 0) / n
        return {f"gpu.{k}": round(v, 2) for k, v in a.items()}



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
