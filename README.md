# QLoRA fine-tuning pipeline (local, Windows, everything on G:)

Config-driven QLoRA fine-tuning of `meta-llama/Llama-2-7b-chat-hf` (4-bit NF4 + LoRA) for an e-commerce
customer-support assistant, on one laptop GPU (RTX 5080 Laptop, 16 GB, sm_120). Every download, cache, venv,
dataset, checkpoint, model and MLflow record lives under `QLORA_ROOT` (here `G:\QLoRA`). The local server speaks
the SageMaker inference contract, so a later cloud move needs no code change.

## Results at a glance

| | |
|---|---|
| Data | Bitext customer support, 6,000-row subsample stratified by intent -> 5,408 / 296 / 296 ([data card](../artifacts/reports/data_card.md)) |
| 7B QLoRA training | 3 epochs, best val loss 0.5655 (epoch 2), peak 7.99 GB allocated / 14.31 GB reserved |
| 7B eval (296 test rows) | ROUGE-L base 24.13 -> tuned **44.08**; slot recall 3.7% -> 68.7%; no forgetting on arc_easy / hellaswag |
| Memory claim (proxy) | QLoRA peak allocated -16.8% vs bf16 LoRA, -43.9% vs full FT (~-57% incl. paged optimizer, est.); does **not** hold on reserved |
| Quality parity (proxy) | QLoRA - full FT = -0.23 ROUGE-L, 95% CI [-1.36, +0.89]: PASS on point estimate (tol. ±1.0) |
| Merge | fp16 CPU merge; merged vs adapter on 50 rows +0.40 ROUGE-L (tol. ±0.5) -> registered as `champion` |

Reports: `G:\QLoRA\artifacts\reports\` (`eval_llama2-7b-qlora.md`, `benchmark.md`, `merge_llama2-7b-qlora.json`,
`load_test.json`).

## Setup from zero (Windows 11, nothing on C:)

Requirements: NVIDIA driver with CUDA 12.8+ support (no CUDA toolkit, no Visual Studio), Git, ~60 GB free on G:.

1. Create the root and put the env files there (`qlora.env`, `env.ps1`, `env.sh`; `qlora.env` is the only file
   that names the drive letter).
2. Put the standalone `uv.exe` in `G:\QLoRA\tools\uv` (from the uv GitHub release zip; not the installer script,
   which writes to C:).
3. Bootstrap: takes the C: baseline first, then installs Python 3.11 and the locked stack into `repo\.venv`:
   ```powershell
   powershell -ExecutionPolicy Bypass -File G:\QLoRA\repo\scripts\bootstrap.ps1
   ```
4. Hugging Face token with Llama 2 access: copy `repo\.env.example` to `repo\.env` and set `HF_TOKEN`.
5. Windows settings this build needed (yours to set; the pipeline never changes system settings):
   - NVIDIA Control Panel -> Manage 3D Settings -> **CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback**,
     globally and for `G:\QLoRA\tools\python\cpython-3.11.15-windows-x86_64-none\python.exe` (the real interpreter
     behind the venv launcher). Otherwise VRAM overflow silently spills into system RAM.
   - A fixed C: pagefile (here 4-16 GB). On WDDM, reserved VRAM is charged 1:1 to Windows commit; a
     system-managed pagefile then grows on C: during 7B jobs.

Every command below assumes the env is active in that shell: `. G:\QLoRA\env.ps1` (PowerShell) or
`source /g/QLoRA/env.sh` (Git Bash). `repo\run.ps1 <command>` activates and runs in one step.

## One command per stage

| Stage | Command |
|---|---|
| 0 Environment check (GPU, bitsandbytes, real 4-bit fwd+bwd, HF access, cache paths) | `qlora env-check` |
| C: footprint audit (run after every phase) | `qlora audit-c` |
| MLflow tracking server (keep it running in its own shell) | `qlora mlflow` |
| 1 Data | `qlora data` |
| 2 Smoke test (proxy QLoRA, 50 steps, loss must drop) | `qlora smoke` |
| 2 Train 7B QLoRA (auto-resumes from the last checkpoint) | `qlora train -c llama2_7b_qlora.yaml` |
| 3 Evaluate base vs tuned | `qlora eval -c llama2_7b_qlora.yaml` |
| 4 Benchmark full FT vs LoRA vs QLoRA (proxy) + 7B estimates | `qlora bench` |
| 5 Merge (fp16, CPU), validate, register `champion` | `qlora merge` |
| 6 Serve on 127.0.0.1:8080 | `qlora serve` |
| 6 Chat REPL against the server | `qlora chat` |
| 6 Load test (20 requests) | `qlora load-test` |

Any config value can be overridden: `--set data.subsample_size=3000 --set train.learning_rate=1.0e-4`.
Configs are in `configs/` (`base.yaml`, `llama2_7b_qlora.yaml`, `proxy_*.yaml`, `serve.yaml`).

Large-model GPU jobs run a pre-flight first. It refuses to start if C: has under 15 GB free, if Windows commit
headroom (RAM + pagefile max - commit in use) is below the job's measured peak + 3 GB (`G:\QLoRA\logs\commit_peaks.json`),
or if C: cannot absorb the pagefile growing to its cap + 10 GB. When it refuses, it lists the top processes by commit.

## MLflow UI

Start `qlora mlflow` (sqlite at `G:\QLoRA\mlflow\mlflow.db`, artifacts in `G:\QLoRA\mlflow\artifacts`) and open
<http://127.0.0.1:5000>. Experiment `qlora-cs-practice` holds the data, train, eval, bench, merge and load-test
runs. The Models tab shows `qlora-cs-llama2-7b-chat` with alias `champion` (adapter artifact; the merged model's
local path is in the version tags).

## Serving and the /invocations API

`qlora serve` loads the 4-bit base plus the `champion` adapter from the MLflow registry (it needs the MLflow
server running). If that fails and `serve.fallback_to_merged` is true (it is in `serve.yaml`), it loads the merged
model in 4-bit instead. `GET /ping` returns 503 while loading and 200 once ready.

```bash
curl -s http://127.0.0.1:8080/ping
```

```bash
curl -s -X POST http://127.0.0.1:8080/invocations -H "Content-Type: application/json" -d '{"inputs": "I want to cancel my order, how do I do that?", "parameters": {"max_new_tokens": 200, "temperature": 0.0}}'
```

Response: `{"generated_text": "..."}`, with `X-Generated-Tokens` and `X-Inference-Ms` headers. `temperature`
0 means greedy (the eval setting); `max_new_tokens` is capped at `serve.max_new_tokens_limit` (512).
`GET /stats` reports the loaded model and peak VRAM.

## Docker (written, not built)

`Dockerfile` builds a CUDA 12.8 image that serves the merged model from `/opt/ml/model` on port 8080 with
the same routes. It was not built here because Docker Desktop stores its disk image on C:. `uv.lock` is resolved
for Windows only, so re-lock for linux before relying on exact pins in the image.

## Layout

```
G:\QLoRA\  qlora.env  env.ps1  env.sh
  tools\uv  tools\python            uv + uv-managed Python 3.11
  repo\                             this repo (venv in repo\.venv)
  cache\                            HF, torch, triton, pip/uv, ... (all redirected by qlora.env)
  data\raw  data\processed          source CSV; jsonl splits + manifest (dataset hash)
  runs\                             trainer checkpoints + summary.json per run
  artifacts\adapters  merged  reports
  mlflow\  logs\  tmp\
```

Code lives in `src/qlora_pipeline/`: `cli.py`, `config.py`, `paths.py` (everything resolves under `QLORA_ROOT`),
`data.py`, `model.py`, `train.py`, `evaluate.py`, `benchmark.py`, `merge.py`, `serve.py`, `telemetry.py`,
`audit.py`. Tests: `uv run pytest` (data, label masking, prompt format, config, paths, pre-flight, serve API).
