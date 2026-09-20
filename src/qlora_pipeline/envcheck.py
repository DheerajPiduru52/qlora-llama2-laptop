"""`qlora env-check` implementation. Imported only after bootstrap.load_env() has run."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from qlora_pipeline.paths import ensure_dir, ensure_under_root, qlora_root, tmp_dir

GATED_REPO = "meta-llama/Llama-2-7b-chat-hf"
MIN_FREE_ROOT_GB = 60.0

# Keys in qlora.env that are paths and must resolve under QLORA_ROOT.
NON_PATH_KEYS = {"HF_HUB_DISABLE_SYMLINKS_WARNING", "UV_LINK_MODE", "MLFLOW_TRACKING_URI", "PYTHONUTF8",
                 "UV_PYTHON_INSTALL_REGISTRY", "PYTORCH_CUDA_ALLOC_CONF"}

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


class Report:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, fn: Callable[[], str]) -> bool:
        print(f"\n=== {name} ===", flush=True)
        try:
            detail = fn()
            ok = True
        except Exception as e:  # noqa: BLE001 - every failure is reported, not raised
            traceback.print_exc()
            detail = f"{type(e).__name__}: {e}"
            ok = False
        print(f"{GREEN + 'PASS' if ok else RED + 'FAIL'}{RESET}  {name}: {detail}", flush=True)
        self.results.append((name, ok, detail))
        return ok

    def summary(self) -> bool:
        print("\n=== SUMMARY ===")
        for name, ok, detail in self.results:
            print(f"  {GREEN + 'PASS' if ok else RED + 'FAIL'}{RESET}  {name:28} {detail}")
        all_ok = all(ok for _, ok, _ in self.results)
        print(f"\nenv-check: {GREEN + 'ALL GREEN' if all_ok else RED + 'NOT GREEN'}{RESET}")
        return all_ok


def _gpu_and_torch() -> str:
    import torch

    print(f"python          {sys.version.split()[0]}  ({sys.executable})")
    print(f"torch           {torch.__version__}")
    print(f"torch CUDA      {torch.version.cuda}")
    print(f"cuDNN           {torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    arches = torch.cuda.get_arch_list()
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU             {name}")
    print(f"compute cap     sm_{cc[0]}{cc[1]}")
    print(f"VRAM            {total:.1f} GB")
    print(f"torch arch list {arches}")
    print(f"bf16 supported  {torch.cuda.is_bf16_supported()}")
    want = f"sm_{cc[0]}{cc[1]}"
    if want not in arches:
        raise RuntimeError(f"torch build has no kernels for {want}")
    major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if (major, minor) < (2, 7):
        raise RuntimeError("PyTorch >= 2.7 required for Blackwell")
    # A real kernel launch, not just a device query.
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    (a @ a).sum().item()
    return f"{name}, {want}, torch {torch.__version__}, CUDA {torch.version.cuda}"


def _nvml() -> str:
    import pynvml

    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        drv = pynvml.nvmlSystemGetDriverVersion()
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
        sm_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        return f"driver {drv}, {temp} C, {power:.1f} W, SM clock {sm_clock} MHz"
    finally:
        pynvml.nvmlShutdown()


def _bnb_diag() -> str:
    import bitsandbytes as bnb

    print(f"bitsandbytes {bnb.__version__}  ({Path(bnb.__file__).parent})")
    proc = subprocess.run(
        [sys.executable, "-m", "bitsandbytes"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    )  # fmt: skip
    out = (proc.stdout + proc.stderr).strip().splitlines()
    print("--- python -m bitsandbytes (last 30 lines) ---")
    for line in out[-30:]:
        print("  " + line)
    if proc.returncode != 0:
        raise RuntimeError(f"`python -m bitsandbytes` exited {proc.returncode}")
    text = "\n".join(out).lower()
    if "success" not in text:
        raise RuntimeError("bitsandbytes self-test did not report SUCCESS")
    return f"bitsandbytes {bnb.__version__}, self-test SUCCESS"


def _qlora_fwd_bwd() -> str:
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    work = ensure_dir(tmp_dir() / "envcheck_tiny_llama")
    cfg = LlamaConfig(
        vocab_size=1024, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
    )  # fmt: skip
    LlamaForCausalLM(cfg).save_pretrained(work, safe_serialization=True)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        work, quantization_config=bnb_cfg, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    )
    n4 = sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
    if n4 == 0:
        raise RuntimeError("no Linear4bit modules after 4-bit load")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, activation_checkpointing_kwargs={"use_reentrant": False}
    )
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )  # fmt: skip
    model = get_peft_model(model, lora)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(trainable, lr=1e-3)

    ids = torch.randint(0, cfg.vocab_size, (2, 64), device="cuda")
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        with_grad = sum(p.grad is not None and p.grad.abs().sum().item() > 0 for p in trainable)
        if with_grad == 0:
            raise RuntimeError("backward produced no non-zero LoRA grads")
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(out.loss.item())
    if not all(map(lambda x: x == x and x != float("inf"), losses)):
        raise RuntimeError(f"non-finite loss: {losses}")
    peak = torch.cuda.max_memory_allocated() / 1024**2
    shutil.rmtree(work, ignore_errors=True)
    return (
        f"{n4} Linear4bit layers, {len(trainable)} LoRA tensors got grads, "
        f"loss {losses[0]:.4f} -> {losses[-1]:.4f} over 3 paged_adamw_8bit steps, peak {peak:.0f} MiB"
    )


def _hf_access() -> str:
    import huggingface_hub
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(f"HF_TOKEN not set. Put it in {qlora_root() / 'repo' / '.env'} (see .env.example).")
    api = HfApi(token=token)
    info = api.whoami()  # never print `token` itself
    who = info.get("name", "?")
    tok = (info.get("auth") or {}).get("accessToken") or {}
    role = tok.get("role", "unknown")  # read | write | fineGrained
    tok_name = tok.get("displayName", "unnamed")
    print(f"HF account      {who}")
    print(f"token type      {role} (token name: '{tok_name}')")
    huggingface_hub.auth_check(GATED_REPO, token=token)  # raises GatedRepoError if license not accepted
    files = [s.rfilename for s in api.model_info(GATED_REPO, files_metadata=False).siblings]
    wanted = [f for f in files if f.endswith((".safetensors", ".json")) or f == "tokenizer.model"]
    return f"user '{who}' ({role} token) has access to {GATED_REPO} ({len(wanted)} files match allow_patterns)"


def _disk() -> str:
    root = qlora_root()
    c = shutil.disk_usage(os.environ.get("SystemDrive", "C:") + "\\")
    g = shutil.disk_usage(root.anchor)
    c_gb, g_gb = c.free / 1024**3, g.free / 1024**3
    print(f"C: free         {c_gb:.1f} GB")
    print(f"{root.anchor:15} free {g_gb:.1f} GB (QLORA_ROOT drive)")
    if g_gb < MIN_FREE_ROOT_GB:
        raise RuntimeError(f"QLORA_ROOT drive has {g_gb:.1f} GB free, under {MIN_FREE_ROOT_GB} GB")
    return f"C: {c_gb:.1f} GB free, {root.anchor} {g_gb:.1f} GB free"


def _cache_paths() -> str:
    root = qlora_root()
    keys = []
    for line in (root / "qlora.env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.append(line.split("=", 1)[0].strip())
    bad = []
    for k in keys:
        v = os.environ.get(k)
        if k in NON_PATH_KEYS:
            print(f"  {k:32} = {v}")
            continue
        try:
            resolved = ensure_under_root(v)
            print(f"  {k:32} = {resolved}")
        except Exception as e:  # noqa: BLE001
            bad.append(k)
            print(f"  {k:32} = {v}  <-- {e}")

    # What the libraries actually resolve to (proves the env loaded before import).
    import datasets
    import huggingface_hub.constants as hfc
    import torch

    effective = {
        "huggingface_hub HF_HUB_CACHE": hfc.HF_HUB_CACHE,
        "huggingface_hub token path": hfc.HF_TOKEN_PATH,
        "datasets HF_DATASETS_CACHE": str(datasets.config.HF_DATASETS_CACHE),
        "torch.hub.get_dir()": torch.hub.get_dir(),
        "tempfile.gettempdir()": tempfile.gettempdir(),
    }
    print("  -- effective library paths --")
    for k, v in effective.items():
        try:
            ensure_under_root(v)
            print(f"  {k:32} = {v}")
        except Exception:  # noqa: BLE001
            bad.append(k)
            print(f"  {k:32} = {v}  <-- OUTSIDE QLORA_ROOT")
    if bad:
        raise RuntimeError(f"outside QLORA_ROOT: {bad}")
    return f"{len(keys)} env vars + {len(effective)} library paths all under {root}"


def run_env_check(skip_hf: bool = False) -> bool:
    t0 = time.time()
    r = Report()
    r.check("cache env vars", _cache_paths)
    r.check("disk space", _disk)
    r.check("GPU + torch", _gpu_and_torch)
    r.check("NVML telemetry", _nvml)
    r.check("bitsandbytes diagnostics", _bnb_diag)
    r.check("4-bit fwd+bwd (tiny Llama)", _qlora_fwd_bwd)
    if not skip_hf:
        r.check("HF gated access", _hf_access)
    ok = r.summary()
    print(f"({time.time() - t0:.1f}s)")
    return ok
