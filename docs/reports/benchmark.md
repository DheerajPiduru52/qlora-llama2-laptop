# Benchmark: QLoRA vs bf16 LoRA vs full fine-tuning

Proxy: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` @ `59f6f375b26bde864a6ca194a9a3044570490064` (Llama 2 architecture and tokenizer).
Dataset hash `ea5cd40d8a4579f5897b68452f94115f30d6200fce0d219c23bc88055bbb9b51`. GPU: RTX 5080 Laptop, 15.92 GB physical VRAM. NVIDIA sysmem fallback was
verified OFF in a separate process (allocating physical VRAM + 1 GB must raise OOM) before each training and eval
subprocess when it ran (timestamps below; logs/bench.log). Every peak below is <= physical VRAM.
`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6` for all three runs.

## Proxy results (MEASURED, 1 epoch = 169 optimizer steps each)

| method | lr | trainable | peak allocated GB* | peak reserved GB* | tokens/s | wall min | best val loss (trainer) | val ppl (eval) | test ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 2e-05 | 100.00% | 6.75 | 12.19 | 1733 | 9.8 | 0.7631 | 2.100 | 37.82 |
| lora | 0.0002 | 1.13% | 4.55 | 13.30 | 2240 | 7.6 | 0.7482 | 2.073 | 37.59 |
| qlora | 0.0002 | 1.13% | 3.79 | 12.47 | 1839 | 9.3 | 0.7585 | 2.094 | 37.58 |

(*) Peak memory = `torch.cuda.max_memory_allocated` / `max_memory_reserved`, reset before model load: weights,
gradients, activations, eval and any optimizer state held by PyTorch's allocator. **Not included: the paged
8-bit AdamW state**, which bitsandbytes keeps in CUDA managed memory outside PyTorch's counters. That matters only
for full FT: 2 bytes x 1.10B params = **2.05 GB ESTIMATED**, so full FT's true peak is
about 8.80 GB (ESTIMATED). LoRA/QLoRA optimizer state is ~0.023 GB.

Tokens/s = non-padding tokens / training wall time excluding eval. "best val loss (trainer)" is the Trainer's
eval_loss (mean of per-batch losses); "val ppl (eval)" is exp of the token-weighted response NLL from `qlora eval`,
so exp(loss) and ppl differ slightly by construction. ROUGE-L on the 296-example test
split, greedy, same generation config.

GPU conditions (one run per method, so throughput differences are confounded by throttling):

| method | power-throttled samples | min SM clock MHz | max temp C | mean power W | training | fallback check before training |
|---|---:|---:|---:|---:|---|---|
| full | 76% | 180 | 69 | 59.2 | reused | backfilled from logs/bench.log: passed before this subprocess in the first bench invocation, 2026-11-19 ~14:52-15:31 (exact start not recorded) |
| lora | 100% | 180 | 74 | 78.8 | reused | backfilled from logs/bench.log: passed before this subprocess in the first bench invocation, 2026-11-19 ~14:52-15:31 (exact start not recorded) |
| qlora | 100% | 937 | 75 | 78.7 | reused | backfilled from logs/bench.log: passed before this subprocess in the first bench invocation, 2026-11-19 ~14:52-15:31 (exact start not recorded) |


### Claim 1: memory (proxy, measured)

| QLoRA reduction vs | peak allocated | peak reserved |
|---|---:|---:|
| bf16 LoRA | **16.8%** | 6.2% |
| full FT | **43.9%** | -2.3% |
| full FT incl. paged optimizer state (ESTIMATED) | ~57.0% | n/a |

**Claim 1 on peak allocated (memory the run actually needs): HELD**:
QLoRA is lower than both alternatives.
**On peak reserved (allocator cache included): DID NOT HOLD**:
QLoRA is not lower than both alternatives
(6.2% vs bf16 LoRA, -2.3% vs full FT; negative = QLoRA reserved more).

Likely reason the proxy gap is small (reasoning, not separately measured): on a 1.1B model, weights are only ~2.1 GB in bf16, so 4-bit quantization saves
at most ~1.5 GB; activations and the fp32 logits over the 32,064-token vocabulary dominate the peak and are the
same for every method. Reserved memory adds PyTorch's cached blocks, which depend on allocation order and
fragmentation rather than on what each method needs, so it is a noisy basis for comparing methods. The savings
should grow with model size, since weights are the part quantization shrinks; the 7B section supports this with one
measurement against estimates, not a sweep.

### Claim 2: quality parity (proxy, measured)

QLoRA ROUGE-L 37.58 vs full FT 37.82:
difference **-0.23** points (paired over 296 test rows, 95% bootstrap CI
[-1.36, +0.89]); tolerance ±1.0 → **PASS** on the point estimate.
The CI extends beyond the tolerance, so this sample cannot rule out a gap larger than 1.0 point(s).
Caveats: one seed per method; full FT used lr 2e-5 vs 2e-4. Read this as "within tolerance, not distinguishable
from noise at this sample size", not as proof of equality. (bf16 LoRA: 37.59.)


## Llama-2-7B

| method | peak GB | source |
|---|---:|---|
| QLoRA | 7.99 allocated / 14.31 reserved | **MEASURED** (Phase 2, `runs/llama2-7b-qlora`) |
| bf16 LoRA | 16.2 | **ESTIMATED** |
| full FT, paged 8-bit AdamW | 41.3 | **ESTIMATED** |
| full FT, standard fp32 AdamW | 104.0 | **ESTIMATED** |

P = 6.738B parameters (from config.json). Activations + LoRA optimizer state =
3.64 GB, taken from the measured 7B QLoRA run (peak allocated
7.99 GB minus 4.35 GB after load) and assumed equal for the other
methods (same architecture, batch 4, seq <= 1024, gradient checkpointing). "After load" is
`max_memory_allocated` right after loading, so it includes load transients and fp32-upcast norms/embeddings.
Anchor check on the proxy, where bf16 LoRA was measured: the same formula predicts
4.74 GB vs 4.55 GB measured (+4%). Treat the 7B
estimates as approximate, about ±1-2 GB.
- bf16 LoRA = bf16 weights 2P + (activations + LoRA grads/optimizer, from the measured 7B QLoRA run) = 12.55 + 3.64 = 16.19 GB
- full FT (8-bit) = bf16 weights 2P + bf16 grads 2P + paged 8-bit AdamW states 2P (2 x 1 byte) + activations = 41.3 GB
- full FT (fp32 Adam) = bf16 weights 2P + bf16 grads 2P + fp32 master weights 4P + fp32 Adam m,v 8P + activations = 104.05 GB

7B QLoRA vs ESTIMATED bf16 LoRA: 50.6% lower; vs ESTIMATED full FT (8-bit Adam):
80.7% lower (fp32 Adam: 92.3%). These compare QLoRA's peak *allocated*;
on *reserved*, measured 7B QLoRA used 14.31 GB, only
11.6% below the bf16 LoRA estimate
(the proxy's reserved-memory caveat applies). bf16 LoRA at ~16.2 GB ESTIMATED is at or just above the
15.9 GB card (within estimate error); full FT is far beyond it. That is why both are estimated, not measured.

## Identical across the three proxy runs (asserted)

| setting | value |
|---|---|
| `model.name` | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| `model.revision` | `59f6f375b26bde864a6ca194a9a3044570490064` |
| `data.processed_name` | `bitext_cs` |
| `data.max_seq_length` | `1024` |
| `data.system_prompt` | `You are a helpful, concise customer support agent for an online store.` |
| `train.effective_batch_size` | `32` |
| `train.per_device_train_batch_size` | `4` |
| `train.num_train_epochs` | `1.0` |
| `train.max_steps` | `-1` |
| `seed` | `42` |
| `train.gradient_checkpointing` | `True` |
| `train.optim` | `paged_adamw_8bit` |
| `train.bf16` | `True` |
| `train.lr_scheduler_type` | `cosine` |
| `train.warmup_ratio` | `0.03` |
| `train.max_grad_norm` | `0.3` |

Learning rate is the one deliberate difference: 2e-4 for LoRA/QLoRA (the QLoRA default), 2e-5 for full FT
(2e-4 is far too high for updating every weight).
