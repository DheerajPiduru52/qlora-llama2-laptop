"""Windows commit-charge diagnostic for the 7B eval path, stage by stage.

Measures, per stage: this process's private commit (PrivateUsage) and its peak, plus system-wide commit,
sampled every 0.2 s so short spikes are caught. A watchdog aborts if C: drops below --c-floor GB free (default 15, the
pre-flight minimum), bounding any transient pagefile growth.

Finding (2026-11-19): on this Windows/WDDM setup every byte of VRAM reserved by PyTorch is charged 1:1 to
process commit, so commit ~= VRAM reserved + ~2 GB CUDA runtime. With ~43 GB already committed by other
apps, even a bare 7B 4-bit load makes Windows grow the system-managed C: pagefile.

    python scripts/commit_diag.py --label before
    python scripts/commit_diag.py --label after --lm-eval-limit 5
"""

from __future__ import annotations

from qlora_pipeline.bootstrap import load_env

load_env()

import argparse  # noqa: E402
import ctypes  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from ctypes import wintypes  # noqa: E402

from qlora_pipeline.audit import c_free_gb, commit_stats  # noqa: E402

GB = 1024**3
C_FLOOR = 15.0


class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]  # fmt: skip


def proc_mem() -> dict[str, float]:
    c = PMC()
    c.cb = ctypes.sizeof(c)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError(ctypes.get_last_error(), "K32GetProcessMemoryInfo failed")
    return {"private_gb": c.PrivateUsage / GB, "peak_private_gb": c.PeakPagefileUsage / GB, "ws_gb": c.WorkingSetSize / GB}


class Sampler:
    def __init__(self) -> None:
        self.stage = "start"
        self.peaks: dict[str, dict[str, float]] = {}
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                pm, cs = proc_mem(), commit_stats()
                cs["commit_total_gb"], cs["commit_limit_gb"]  # noqa: B018 - KeyError -> skip this sample
            except Exception:  # noqa: BLE001 - never let the watchdog thread die silently
                time.sleep(0.2)
                continue
            p = self.peaks.setdefault(self.stage, {"proc_private_peak_gb": 0, "sys_commit_peak_gb": 0})
            p["proc_private_peak_gb"] = max(p["proc_private_peak_gb"], pm["private_gb"])
            p["sys_commit_peak_gb"] = max(p["sys_commit_peak_gb"], cs["commit_total_gb"])
            headroom = cs["commit_limit_gb"] - cs["commit_total_gb"]  # noqa: F841 - kept for the abort message
            if c_free_gb() < C_FLOOR:
                print(f"WATCHDOG ABORT in stage {self.stage}: commit headroom {headroom:.1f} GB, "
                      f"C: free {c_free_gb():.1f} GB", flush=True)  # fmt: skip
                os._exit(3)
            time.sleep(0.2)

    def enter(self, stage: str) -> None:
        self.stage = stage

    def start(self) -> Sampler:
        self.t.start()
        return self


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", default="llama2_7b_qlora.yaml")
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--lm-eval-limit", type=int, default=5)
    ap.add_argument("--set", dest="set_", action="append", default=[])
    ap.add_argument("--c-floor", type=float, default=15.0)
    ap.add_argument("--empty-cache", action="store_true", help="torch.cuda.empty_cache() after every stage")
    a = ap.parse_args()
    global C_FLOOR
    C_FLOOR = a.c_floor

    s = Sampler().start()
    rows: list[dict] = []

    def mark(stage: str, t0: float) -> None:
        import sys as _sys

        vram_peak = vram_end = 0.0
        _t = _sys.modules.get("torch")
        if _t is not None and _t.cuda.is_initialized():
            vram_peak = _t.cuda.max_memory_reserved() / GB
            if a.empty_cache:
                _t.cuda.empty_cache()
            vram_end = _t.cuda.memory_reserved() / GB
            _t.cuda.reset_peak_memory_stats()
        time.sleep(0.5)  # let the sampler record the stage tail
        pm, cs = proc_mem(), commit_stats()
        pk = s.peaks.get(stage, {})
        rows.append({"stage": stage, "secs": round(time.time() - t0, 1),
                     "proc_private_end_gb": round(pm["private_gb"], 2),
                     "proc_private_peak_gb": round(pk.get("proc_private_peak_gb", 0), 2),
                     "proc_ws_end_gb": round(pm["ws_gb"], 2),
                     "sys_commit_peak_gb": round(pk.get("sys_commit_peak_gb", 0), 2),
                     "vram_reserved_peak_gb": round(vram_peak, 2), "vram_reserved_end_gb": round(vram_end, 2),
                     "sys_commit_limit_gb": cs["commit_limit_gb"]})  # fmt: skip
        r = rows[-1]
        print(f"{stage:28} {r['secs']:7.1f}s  proc private end {r['proc_private_end_gb']:6.2f} GB "
              f"peak {r['proc_private_peak_gb']:6.2f} GB  vram peak {r['vram_reserved_peak_gb']:5.2f} "
              f"end {r['vram_reserved_end_gb']:5.2f}  "
              f"sys commit peak {r['sys_commit_peak_gb']:6.2f}/{r['sys_commit_limit_gb']:.2f} GB", flush=True)  # fmt: skip

    t = time.time(); s.enter("0 python + project imports"); mark("0 python + project imports", t)  # noqa: E702

    t = time.time(); s.enter("1 import torch/transformers")  # noqa: E702
    import torch
    import transformers  # noqa: F401
    mark("1 import torch/transformers", t)

    t = time.time(); s.enter("2 CUDA init + matmul")  # noqa: E702
    x = torch.randn(1024, 1024, device="cuda")
    (x @ x).sum().item()
    del x
    mark("2 CUDA init + matmul", t)

    from qlora_pipeline.config import load_config
    from qlora_pipeline.data import load_split, load_tokenizer
    from qlora_pipeline.evaluate import generate, generation_config, lm_eval_regression, perplexity, resolve_weights
    from qlora_pipeline.model import load_base, preflight

    cfg = load_config(a.config, [f"eval.lm_eval_limit={a.lm_eval_limit}", f"eval.gen_batch_size={a.gen_batch}",
                                 f"eval.max_new_tokens={a.max_new_tokens}", *a.set_])  # fmt: skip
    preflight(cfg, f"commit diagnostic ({a.label})")
    tok = load_tokenizer(cfg.model, padding_side="left")

    t = time.time(); s.enter("3 load 4-bit base")  # noqa: E702
    model = load_base(cfg, tok, quantized=True)
    mark("3 load 4-bit base", t)

    t = time.time(); s.enter("4 attach LoRA adapter")  # noqa: E702
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, resolve_weights(cfg), is_trainable=False)
    model.config.use_cache = True
    model.eval()
    mark("4 attach LoRA adapter", t)

    test_rows = load_split(cfg, "test")
    test_rows = sorted(test_rows, key=lambda r: -len(r["instruction"]))[: a.gen_batch]
    gen_cfg = generation_config(cfg, tok, model)

    t = time.time(); s.enter("5 generate tuned (1 batch)")  # noqa: E702
    generate(model, tok, cfg, test_rows, gen_cfg)
    mark("5 generate tuned (1 batch)", t)

    t = time.time(); s.enter("6 generate base via disable_adapter")  # noqa: E702
    with model.disable_adapter():
        generate(model, tok, cfg, test_rows, gen_cfg)
    mark("6 generate base via disable_adapter", t)

    t = time.time(); s.enter("7 perplexity (16 val rows)")  # noqa: E702
    perplexity(model, tok, cfg, load_split(cfg, "val")[:16])
    mark("7 perplexity (16 val rows)", t)

    t = time.time(); s.enter("8 lm-eval on loaded model")  # noqa: E702
    lm_eval_regression(model, tok, cfg)
    mark("8 lm-eval on loaded model", t)

    t = time.time(); s.enter("9 lm-eval base (disable_adapter)")  # noqa: E702
    with model.disable_adapter():
        lm_eval_regression(model, tok, cfg)
    mark("9 lm-eval base (disable_adapter)", t)

    s._stop.set()
    out = {"label": a.label, "config": a.config, "empty_cache": a.empty_cache,
           "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
           "overrides": a.set_, "gen_batch": a.gen_batch,
           "max_new_tokens": a.max_new_tokens, "lm_eval_limit": a.lm_eval_limit,
           "process_peak_private_gb_lifetime": round(proc_mem()["peak_private_gb"], 2), "stages": rows}  # fmt: skip
    from qlora_pipeline.paths import ensure_dir, reports_dir

    p = ensure_dir(reports_dir()) / f"commit_diag_{a.label}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"process lifetime peak private commit: {out['process_peak_private_gb_lifetime']} GB -> {p}")


if __name__ == "__main__":
    main()
