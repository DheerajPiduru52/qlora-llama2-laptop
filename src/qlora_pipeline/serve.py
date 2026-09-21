"""Phase 6: local HTTP serving with the SageMaker inference contract, plus a chat REPL and a load test.

Contract (so a later cloud move needs no code change):
    GET  /ping          -> 200 once the model is loaded (503 while loading or if loading failed)
    POST /invocations   {"inputs": "<customer message>", "parameters": {"max_new_tokens": .., "temperature": ..}}
                        -> {"generated_text": "..."}
Extra, local-only: GET /stats, POST /stats/reset-peak (peak VRAM for the load test).

The model is the 4-bit NF4 base + the adapter behind the MLflow registry alias (default "champion"). If that
fails and serve.fallback_to_merged is set, the merged model is loaded in 4-bit instead. The prompt template,
pad handling and suppressed tokens are the same as training and eval.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from qlora_pipeline import telemetry
from qlora_pipeline.config import Config

GB = 1024**3


# --------------------------------------------------------------------------- request / response schema


class Parameters(BaseModel):
    model_config = ConfigDict(extra="ignore")  # tolerate other SageMaker/HF params (top_k, do_sample, ...)
    max_new_tokens: int | None = Field(None, ge=1)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)


class InvocationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    inputs: str = Field(..., min_length=1)
    parameters: Parameters = Field(default_factory=Parameters)


class InvocationResponse(BaseModel):
    generated_text: str


class InputTooLongError(ValueError):
    """The only engine error that is the client's fault (-> HTTP 400)."""


class Engine(Protocol):
    info: dict[str, Any]

    def generate(self, text: str, max_new_tokens: int, temperature: float, top_p: float) -> tuple[str, int]: ...

    def stats(self) -> dict[str, Any]: ...

    def reset_peak(self) -> None: ...


# --------------------------------------------------------------------------- real engine (GPU)


class HFEngine:
    def __init__(self, cfg: Config, model, tok, info: dict[str, Any]) -> None:
        from qlora_pipeline.model import suppressed_token_ids

        self.cfg, self.model, self.tok, self.info = cfg, model, tok, info
        self.suppress = suppressed_token_ids(model, tok)

    def generate(self, text: str, max_new_tokens: int, temperature: float, top_p: float) -> tuple[str, int]:
        import torch
        from transformers import GenerationConfig

        from qlora_pipeline.data import encode_prompt

        ids = encode_prompt(self.tok, self.cfg.data.system_prompt, text)
        if len(ids) > self.cfg.data.max_seq_length:
            raise InputTooLongError(f"input is {len(ids)} tokens; the limit is {self.cfg.data.max_seq_length}")
        sample = temperature > 0
        gc = GenerationConfig(
            do_sample=sample, num_beams=1, max_new_tokens=max_new_tokens, repetition_penalty=1.0,
            temperature=temperature if sample else None, top_p=top_p if sample else None, top_k=None,
            eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id, suppress_tokens=self.suppress,
        )  # fmt: skip
        x = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            out = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=gc)
        new = out[0, len(ids):].tolist()
        n = new.index(self.tok.eos_token_id) + 1 if self.tok.eos_token_id in new else len(new)
        del out, x
        return self.tok.decode(new[:n], skip_special_tokens=True).strip(), n

    def stats(self) -> dict[str, Any]:
        import torch

        return {"peak_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 3),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
                "reserved_gb": round(torch.cuda.memory_reserved() / GB, 3),
                "physical_vram_gb": round(telemetry.physical_vram_gb(), 2)}  # fmt: skip

    def reset_peak(self) -> None:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def resolve_champion(cfg: Config) -> tuple[Path, dict[str, Any]]:
    """Registry alias -> downloaded adapter dir (under QLORA_ROOT), verified against the registered sha256."""
    import mlflow

    from qlora_pipeline.merge import _dir_sha
    from qlora_pipeline.paths import ensure_dir, tmp_dir

    mlflow.set_tracking_uri(telemetry.tracking_uri())
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(cfg.merge.registry_name, cfg.merge.alias)
    dst = ensure_dir(tmp_dir() / "serve_cache" / f"{cfg.merge.registry_name}-v{mv.version}")
    path = Path(mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=str(dst)))
    want = mv.tags.get("adapter_sha256")
    got = _dir_sha(path)
    if want and got != want:
        raise RuntimeError(f"downloaded adapter sha256 {got[:12]} != registered {want[:12]}")
    meta = {"registry_name": mv.name, "version": mv.version, "alias": cfg.merge.alias, "source": mv.source,
            "adapter_sha256": got, "tags": dict(mv.tags)}  # fmt: skip
    return path, meta


def load_engine(cfg: Config) -> HFEngine:
    from qlora_pipeline.data import load_tokenizer
    from qlora_pipeline.model import load_for_inference, preflight
    from qlora_pipeline.paths import ensure_under_root, merged_dir

    preflight(cfg, "serve", job="serve")
    cap = telemetry.cap_vram(cfg.serve.max_reserved_gb)
    tok = load_tokenizer(cfg.model, padding_side="left")
    attempts: list[str] = []
    meta: dict[str, Any] = {}
    if cfg.serve.source == "registry":
        try:
            adapter, meta = resolve_champion(cfg)
            model = load_for_inference(cfg, tok, adapter, quantized=True)
            info = {"loaded": "registry", "weights": str(adapter), "quantization": "nf4 4-bit base + LoRA adapter",
                    "vram_cap_gb": round(cap, 2), **{k: meta[k] for k in ("registry_name", "version", "alias")}}  # fmt: skip
            return HFEngine(cfg, model, tok, info)
        except Exception as e:  # noqa: BLE001 - reported, then optional fallback
            attempts.append(f"registry: {type(e).__name__}: {e}")
            print(f"registry load failed: {attempts[-1]}", flush=True)
            if not cfg.serve.fallback_to_merged:
                raise
    merged = cfg.serve.merged_path or meta.get("tags", {}).get("merged_path") or str(merged_dir() / cfg.train.output_name)
    merged_p = ensure_under_root(merged)
    model = load_for_inference(cfg, tok, merged_p, quantized=True)
    info = {"loaded": "merged", "weights": str(merged_p), "quantization": "nf4 4-bit merged model",
            "vram_cap_gb": round(cap, 2), "fallback_reason": attempts}  # fmt: skip
    return HFEngine(cfg, model, tok, info)


# --------------------------------------------------------------------------- app


def create_app(cfg: Config, engine: Engine | None = None) -> FastAPI:
    """engine=None loads the real model in a background thread at startup (/ping answers 503 until done)."""
    state: dict[str, Any] = {"engine": engine, "error": None, "requests": 0, "started": time.time()}
    lock = threading.Lock()  # one generation at a time on the single GPU

    def _load() -> None:
        try:
            state["engine"] = load_engine(cfg)
            print(f"model ready: {state['engine'].info}", flush=True)
        except Exception as e:  # noqa: BLE001
            state["error"] = f"{type(e).__name__}: {e}"
            print(f"MODEL LOAD FAILED: {state['error']}", flush=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["engine"] is None:
            threading.Thread(target=_load, name="model-load", daemon=True).start()
        yield
        if isinstance(state["engine"], HFEngine):  # only a real GPU engine has a meaningful commit peak
            from qlora_pipeline.model import record_peak

            record_peak(cfg, "serve", f"server {state['requests']} requests")

    app = FastAPI(title="qlora-cs", lifespan=lifespan)

    @app.get("/ping")
    async def ping():
        if state["engine"] is not None:
            return JSONResponse({"status": "ok"}, status_code=200)
        return JSONResponse({"status": "error" if state["error"] else "loading", "error": state["error"]},
                            status_code=503)  # fmt: skip

    @app.post("/invocations", response_model=InvocationResponse)
    async def invocations(req: InvocationRequest):
        eng = state["engine"]
        if eng is None:
            return JSONResponse({"error": "model not loaded", "detail": state["error"]}, status_code=503)
        p, s = req.parameters, cfg.serve
        max_new = p.max_new_tokens or s.default_max_new_tokens
        if max_new > s.max_new_tokens_limit:
            return JSONResponse({"error": f"max_new_tokens {max_new} > limit {s.max_new_tokens_limit}"}, status_code=400)
        temp = s.default_temperature if p.temperature is None else p.temperature
        top_p = 1.0 if p.top_p is None else p.top_p

        def _run():
            with lock:
                t0 = time.perf_counter()
                text, n = eng.generate(req.inputs, max_new, temp, top_p)
                return text, n, (time.perf_counter() - t0) * 1000

        try:
            text, n, ms = await asyncio.get_running_loop().run_in_executor(None, _run)
        except InputTooLongError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:  # noqa: BLE001 - e.g. CUDA OOM: report as JSON, keep serving
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
        state["requests"] += 1
        return JSONResponse({"generated_text": text},
                            headers={"X-Generated-Tokens": str(n), "X-Inference-Ms": f"{ms:.1f}"})  # fmt: skip

    @app.get("/stats")
    async def stats():
        eng = state["engine"]
        body = {"loaded": eng is not None, "error": state["error"], "requests": state["requests"],
                "uptime_s": round(time.time() - state["started"], 1)}  # fmt: skip
        if eng is not None:
            body.update(info=eng.info, **eng.stats())
        return body

    @app.post("/stats/reset-peak")
    async def reset_peak():
        eng = state["engine"]
        if eng is None:
            return JSONResponse({"error": "model not loaded"}, status_code=503)
        def _reset():
            with lock:
                eng.reset_peak()

        await asyncio.get_running_loop().run_in_executor(None, _reset)
        return {"ok": True}

    return app


def serve(cfg: Config, host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    uvicorn.run(create_app(cfg), host=host or cfg.serve.host, port=port or cfg.serve.port, workers=1,
                log_level="info")  # fmt: skip


# --------------------------------------------------------------------------- clients


def base_url(cfg: Config, url: str | None) -> str:
    return (url or f"http://{cfg.serve.host}:{cfg.serve.port}").rstrip("/")


def wait_ready(url: str, timeout_s: float = 900) -> None:
    import httpx

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(f"{url}/ping", timeout=5)
            if r.status_code == 200:
                return
            if r.json().get("status") == "error":
                raise RuntimeError(f"server failed to load the model: {r.json().get('error')}")
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError(f"{url}/ping not ready after {timeout_s}s")


def chat_repl(cfg: Config, url: str | None = None, temperature: float | None = None) -> None:
    import httpx

    u = base_url(cfg, url)
    wait_ready(u)
    print(f"Connected to {u}. Type a customer message; /quit to exit.", flush=True)
    while True:
        try:
            msg = input("customer> ").strip()
        except EOFError:
            break
        if not msg:
            continue
        if msg in ("/quit", "/exit"):
            break
        params: dict[str, Any] = {}
        if temperature is not None:
            params["temperature"] = temperature
        try:
            r = httpx.post(f"{u}/invocations", json={"inputs": msg, "parameters": params},
                           timeout=cfg.serve.request_timeout_s)  # fmt: skip
        except httpx.HTTPError as e:
            print(f"[request failed] {type(e).__name__}: {e}", flush=True)
            continue
        if r.status_code != 200:
            print(f"[error {r.status_code}] {r.text}", flush=True)
            continue
        print(f"agent> {r.json()['generated_text']}\n   ({r.headers.get('X-Generated-Tokens')} tokens, "
              f"{r.headers.get('X-Inference-Ms')} ms)", flush=True)  # fmt: skip


def _pct(xs: list[float], q: float) -> float:
    import numpy as np

    return float(np.percentile(xs, q)) if xs else math.nan


def load_test(cfg: Config, url: str | None = None, n: int | None = None, concurrency: int | None = None) -> dict:
    """n requests (first n test-split messages), greedy, at the given concurrency. Logged to MLflow."""
    import concurrent.futures as cf

    import httpx
    import mlflow

    from qlora_pipeline.data import load_split
    from qlora_pipeline.paths import ensure_dir, reports_dir

    u = base_url(cfg, url)
    n = n or cfg.serve.load_test_n
    conc = concurrency or cfg.serve.load_test_concurrency
    wait_ready(u)
    rows = load_split(cfg, "test")[:n]
    n = len(rows)
    body = {"temperature": 0.0, "max_new_tokens": cfg.serve.load_test_max_new_tokens}
    # one uncounted warm-up request so CUDA/kernel warm-up does not land in the measured latencies
    httpx.post(f"{u}/invocations", json={"inputs": "Hello, I need help with my order.",
                                         "parameters": {**body, "max_new_tokens": 16}},
               timeout=cfg.serve.request_timeout_s)  # fmt: skip
    server_before = httpx.get(f"{u}/stats", timeout=30).json()
    httpx.post(f"{u}/stats/reset-peak", timeout=60).raise_for_status()

    def one(r: dict) -> dict:
        t0 = time.perf_counter()
        try:
            resp = httpx.post(f"{u}/invocations", json={"inputs": r["instruction"], "parameters": body},
                              timeout=cfg.serve.request_timeout_s)  # fmt: skip
        except httpx.HTTPError as e:
            return {"ok": False, "status": None, "error": f"{type(e).__name__}: {e}",
                    "latency_s": time.perf_counter() - t0, "tokens": 0, "server_ms": None}  # fmt: skip
        lat = time.perf_counter() - t0
        ok = resp.status_code == 200
        return {"ok": ok, "status": resp.status_code, "latency_s": lat,
                "tokens": int(resp.headers.get("X-Generated-Tokens", 0)) if ok else 0,
                "server_ms": float(resp.headers.get("X-Inference-Ms", "nan")) if ok else None}  # fmt: skip

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        results = list(ex.map(one, rows))
    wall = time.perf_counter() - t0
    server_after = httpx.get(f"{u}/stats", timeout=30).json()

    good = [r for r in results if r["ok"]]
    lats = [r["latency_s"] for r in good]
    toks = sum(r["tokens"] for r in good)
    per_req_tps = [r["tokens"] / (r["server_ms"] / 1000) for r in good if r["server_ms"] and r["server_ms"] > 0]
    summary = {
        "url": u, "n": n, "concurrency": conc, "ok": len(good), "errors": len(results) - len(good),
        "latency_p50_s": round(_pct(lats, 50), 3), "latency_p95_s": round(_pct(lats, 95), 3),
        "latency_mean_s": round(sum(lats) / len(lats), 3) if lats else math.nan,
        "tokens_total": toks, "wall_s": round(wall, 2),
        "throughput_tokens_per_s": round(toks / wall, 1) if wall else math.nan,
        "per_request_tokens_per_s_p50": round(_pct(per_req_tps, 50), 1),
        "mean_generated_tokens": round(toks / len(good), 1) if good else math.nan,
        "peak_vram_reserved_gb": server_after.get("peak_reserved_gb"),
        "peak_vram_allocated_gb": server_after.get("peak_allocated_gb"),
        "physical_vram_gb": server_after.get("physical_vram_gb"),
        "server": server_before.get("info"), "generation": body, "warmup_request": "1 uncounted (16 tokens)",
        "requests": results,
    }  # fmt: skip
    summary = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in summary.items()}
    out = ensure_dir(reports_dir()) / "load_test.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tags = {"stage": "load-test", "server_url": u, "model_source": str((summary["server"] or {}).get("loaded"))}
    with telemetry.start_run(cfg, "load-test", tags) as run:
        mlflow.log_params({"n": n, "concurrency": conc, "max_new_tokens": body["max_new_tokens"], "temperature": 0.0})
        mlflow.log_metrics({k: float(v) for k, v in summary.items()
                            if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)})  # fmt: skip
        mlflow.log_dict(summary, "load_test.json")
        summary["mlflow_run_id"] = run.info.run_id
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
