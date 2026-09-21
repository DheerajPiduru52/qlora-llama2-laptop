# Evaluation: llama2-7b-qlora

- Weights: `G:\QLoRA\artifacts\adapters\llama2-7b-qlora`; base model `meta-llama/Llama-2-7b-chat-hf` @ `f5db02db724555f92da89c216ac04704f23d4590`; 4-bit: True
- Test split: 296 examples; val split for perplexity. Same prompt template as training.
- Fixed generation config: greedy (`do_sample=False`, `num_beams=1`), `max_new_tokens=512`,
  `repetition_penalty=1.0`, pad and embedding-padding ids suppressed.
- ROUGE-L = rouge_score F-measure x100 with stemming. Caveat: Bitext responses are templated per intent,
  so absolute ROUGE-L is high for any model that learns the house style; use it comparatively.

## Overall

| metric | base | tuned |
|---|---:|---:|
| ROUGE-L (mean, test) | 24.13 | 44.08 |
| val perplexity (response tokens) | 5.003 | 1.733 |
| slot recall % | 3.7 | 68.7 |
| rows with all slots kept % | 2.4 | 56.3 |
| invented slots / row | 0.037 | 0.321 |
| mean generated tokens | 149.9 | 139.0 |
| hit max_new_tokens % | 0.0 | 0.0 |
| generation tokens/s (batched) | 120.3 | 106.4 |

## ROUGE-L per category

| category | n | base | tuned |
|---|---:|---:|---:|
| ACCOUNT | 66 | 23.20 | 44.62 |
| CANCEL | 12 | 20.16 | 33.11 |
| CONTACT | 24 | 25.04 | 40.90 |
| DELIVERY | 20 | 25.88 | 48.04 |
| FEEDBACK | 24 | 20.90 | 37.66 |
| INVOICE | 22 | 22.87 | 42.88 |
| ORDER | 37 | 24.02 | 45.79 |
| PAYMENT | 24 | 26.84 | 48.96 |
| REFUND | 31 | 23.43 | 43.75 |
| SHIPPING | 24 | 27.19 | 49.47 |
| SUBSCRIPTION | 12 | 27.90 | 41.99 |

## Forgetting check (lm-evaluation-harness, limit=200 per task, ± = stderr)

| task | metric | base | tuned |
|---|---|---:|---:|
| arc_easy | acc | 65.5 ± 3.4 | 67.0 ± 3.3 |
| arc_easy | acc_norm | 64.5 ± 3.4 | 64.0 ± 3.4 |
| hellaswag | acc | 53.5 ± 3.5 | 53.0 ± 3.5 |
| hellaswag | acc_norm | 68.0 ± 3.3 | 67.0 ± 3.3 |

Tuned minus base: arc_easy acc_norm -0.5 (±4.8), hellaswag acc_norm -1.0 (±4.7) points. Rule: flag if drop > 2.0 pts and > 1.5x combined stderr: **no forgetting beyond tolerance**.

## First 3 test examples (full table: predictions CSV / MLflow `predictions.json`)

**[ORDER / cancel_order]** question about cancelling order {{Order Number}}

- **base** (ROUGE-L 24.5): Hello! Thank you for reaching out to us. I'd be happy to help you with your question. Can you please provide me with the order number you'd like to cancel? Additionally, could you please let me know the reason why you'd like to cancel the order? This information will help me assist you more efficiently.
- **tuned** (ROUGE-L 37.6): I'm happy to help! I'm here to assist you with any questions you may have regarding cancelling order {{Order Number}}. Please let me know what specific information or guidance you need, and I'll be more than happy to provide it to you.
- **reference**: I've understood you have a question regarding canceling order {{Order Number}}, and I'm here to provide you with the information you need. Please go ahead and ask your question, and I'll do my best to assist you.

**[ORDER / cancel_order]** I bought some articles, help me to cancel order {{Order Number}}

- **base** (ROUGE-L 20.8): Of course, I'd be happy to help you with that! Can you please provide me with your order number so I can assist you in canceling your order? You can find your order number in the confirmation email sent to you after placing your order.

Once I have your order number, I can guide you through the process of canceling your order. Please let me know if you have any other questions or concerns.
- **tuned** (ROUGE-L 63.6): I've understood you're seeking assistance with canceling your order with the order number {{Order Number}}. I apologize for any inconvenience caused. To cancel your order, please follow these steps:

1. Sign in to your account: Access our platform by logging in to your {{Online Company Portal Info}}.
2. Navigate to your order history: Once logged in, go to the '{{Online Order Interaction}}' or '{{Online Order Interaction}}' section.
3. Locate the specific order: Look for the order with the numbe
- **reference**: I've noticed that you purchased some articles and now need assistance with canceling order {{Order Number}}. Your satisfaction is our top priority, and I'm here to guide you through the cancellation process. Here are the steps to cancel your order:

1. Sign in to Your Account: Visit our {{Online Company Portal Info}} and log in using your credentials.
2. Access Your Orders: Once you're logged in, navigate to the '{{Online Order Interaction}}' or '{{Online Order Interaction}}' section.
3. Locate 

**[ORDER / cancel_order]** I want assistance canceling purchase {{Order Number}}

- **base** (ROUGE-L 12.0): Of course, I'd be happy to help you cancel your purchase! Can you please provide me with your order number so I can assist you?
- **tuned** (ROUGE-L 57.8): I've decoded that you're seeking assistance with canceling your purchase with the order number {{Order Number}}. I'm here to help you navigate through the cancellation process and ensure a smooth experience. To cancel your purchase, please follow these steps:

1. Sign in to your {{Online Company Portal Info}} using your credentials.
2. Navigate to the '{{Online Order Interaction}}' or '{{Online Order Interaction}}' section, where you can find a record of your previous purchases.
3. Locate the pu
- **reference**: I can sense that you're seeking assistance with canceling purchase {{Order Number}}. We're here to help! To cancel your purchase, please follow these steps:

1. Log into Your Account: Visit our {{Online Company Portal Info}} and sign in using your credentials.
2. Locate Your Purchases: Once signed in, navigate to the '{{Online Order Interaction}}' or '{{Online Order Interaction}}' section.
3. Find the Relevant Purchase: Look for the purchase linked to the order number {{Order Number}} and click 


## GPU memory finding (added 2026-11-19, after the run; results above unaffected)

This run exceeded physical VRAM. PyTorch's peak reserved memory was **26.01 GB (tuned pass)** and
**21.66 GB (base pass)** on a **16 GB** RTX 5080 Laptop GPU. The NVIDIA driver's *CUDA Sysmem Fallback
Policy* was at its default (fallback allowed), so allocations past 16 GB silently spilled into shared system
memory instead of raising out-of-memory.

Consequences, measured by a 1 s system sampler (logs/watch_eval_7b.csv, 522 samples):
- eval process peak private commit **30.16 GB** (on Windows/WDDM, reserved VRAM is charged 1:1 to commit; the
  spill added system-memory commit on top);
- system commit peaked at **47.25 GB against a 47.37 GB limit** (0.12 GB from allocation failures);
- the fixed C: pagefile grew to its **16,384 MB cap**; minimum C: free 22.96 GB.

Spilling slows kernels but does not change numerics, so the metrics in this report stand.

Follow-up (corrected 2026-11-19): per-stage peak VRAM logging added. A 20-row probe found the overflow in
**generation** (batch 16: 16.79 GB reserved vs 6.05 GB allocated, KV-cache fragmentation); perplexity (7.06 GB)
and lm-eval (~4.45 GB per task) were fine. Fix: generation batch 8, cache release between generation batches,
and a hard PyTorch cap of 14 GB reserved (`set_per_process_memory_fraction`). After the fix every stage peaked
<= 7.06 GB reserved and process commit fell from 30.16 to 15.48 GB. Note: a later check showed the NVIDIA
"Prefer No Sysmem Fallback" setting was **not** in effect for this python.exe at that time; the 14 GB bound
comes from PyTorch's own cap, which holds regardless of the driver setting.
