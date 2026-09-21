# Data card: bitext_cs

- Source: `bitext/Bitext-customer-support-llm-chatbot-training-dataset` @ `430d1a89bd93bd1fa23c16f29dd53e73f0087443` / `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`
- Source file sha256: `6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434`
- **Dataset hash (train+val+test jsonl): `ea5cd40d8a4579f5897b68452f94115f30d6200fce0d219c23bc88055bbb9b51`**
- Tokenizer: `meta-llama/Llama-2-7b-chat-hf` (pad token `<pad>` id 32000, eos id 2)
- Seed 42; subsample 6000 stratified by `intent`;
  split 0.9/0.05/0.05 stratified by `intent`
- System prompt: "You are a helpful, concise customer support agent for an online store."
- Responses with `{{template slots}}`: 2594 of 6000 (kept verbatim)

## Funnel

| stage | rows |
|---|---:|
| raw | 26872 |
| after_validation | 26862 |
| after_normalize | 26862 |
| after_dedupe | 24066 |
| after_length_filter | 24066 |
| after_subsample | 6000 |
| split_train | 5408 |
| split_val | 296 |
| split_test | 296 |

Dropped at validation: {"empty_instruction": 0, "empty_output": 0, "empty_intent": 0, "empty_category": 0, "malformed_slot_braces": 10, "prompt_control_tokens": 0, "empty_after_normalize": 0}.
Dedupe: {"exact_duplicates": 0, "normalized_instruction_duplicates": 2796}.

## Split sizes

train 5408 / val 296 / test 296;
27 intents, 11 categories.

## Counts per category

| category | train | val | test | total |
|---|---:|---:|---:|---:|
| ACCOUNT | 1191 | 66 | 66 | 1323 |
| ORDER | 678 | 37 | 37 | 752 |
| REFUND | 576 | 31 | 31 | 638 |
| FEEDBACK | 447 | 24 | 24 | 495 |
| PAYMENT | 445 | 24 | 24 | 493 |
| SHIPPING | 438 | 24 | 24 | 486 |
| CONTACT | 437 | 24 | 24 | 485 |
| INVOICE | 401 | 22 | 22 | 445 |
| DELIVERY | 366 | 20 | 20 | 406 |
| SUBSCRIPTION | 221 | 12 | 12 | 245 |
| CANCEL | 208 | 12 | 12 | 232 |

## Token lengths (prompt + response + EOS)

| | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| corpus_before_filter | 58 | 161 | 285 | 337 | 480 | 620 |
| train | 59 | 161 | 287 | 339 | 486 | 582 |
| val | 84 | 163 | 291 | 356 | 459 | 501 |
| test | 68 | 158 | 280 | 327 | 488 | 528 |
| prompt_only | 38 | 47 | 51 | 52 | 56 | 63 |

p99 = 480 tokens <= max_seq_length = 1024: no change needed.

## Leakage check (normalized text)

| check | count |
|---|---:|
| val_instruction_in_train | 0 |
| val_pair_in_train | 0 |
| test_instruction_in_train | 0 |
| test_pair_in_train | 0 |
| val_test_instruction_overlap | 0 |

Response overlap with train (informational): {"val_response_exact_in_train_pct": 0.0, "val_response_normalized_in_train_pct": 0.0, "test_response_exact_in_train_pct": 0.0, "test_response_normalized_in_train_pct": 0.0}.
Bitext responses are templated per intent, so eval responses can be near-copies of train responses.
This inflates absolute ROUGE-L for any model trained on this data; compare base vs tuned and
QLoRA vs full FT on the same split rather than reading absolute scores.

## Label-mask samples

Loss is computed on response tokens only. System and user tokens get label -100; EOS is kept so the
model learns to stop.

### Sample 1

**category** `REFUND` / **intent** `check_refund_policy` / 445 tokens (49 masked prompt, 396 trained)

Masked prompt (labels = -100):
```text
<s>[INST] <<SYS>>
You are a helpful, concise customer support agent for an online store.
<</SYS>>

need assistance to check in what cases can i request refunds [/INST]
```
Trained response (labels = input ids, incl. EOS):
```text
I hear you loud and clear your request for assistance in understanding the cases in which you can request refunds. Here are some common situations where you may be eligible for a refund:

1. **Product/Service Defect:** If the product or service you received has a defect or does not meet the description provided, you can typically request a refund.
2. **Cancellation within Grace Period:** If you change your mind and decide to cancel your purchase within the specified grace period, you should be able to request a refund.
3. **Unauthorized Charges:** If you notice any charges on your account that you did not authorize, we can investigate the issue and assist you with requesting a refund.
4. **Event Cancellation:** If an event you purchased tickets for gets canceled, and there is no rescheduled date, you may be eligible for a refund.
5. **Duplicate Charges:** If you notice duplicate charges for the same item on your billing statement, please let us know, and we will help resolve the issue and process a refund if necessary.
6. **Non-receipt of Goods:** If you did not receive the product you ordered within the expected delivery timeframe, we can assist you in requesting a refund.

It's important to note that refund policies can vary depending on the vendor or service provider. I recommend reviewing their specific terms and conditions or reaching out to them directly for further information. If you have any specific concerns or need assistance with a particular order, please provide me with the relevant details, such as the order number, and I'll be glad to assist you further.

Feel free to ask any other questions or let me know if there's anything else I can assist you with.</s>
```
Token boundary:

| pos | token | id | label |
|---|---|---|---|
| 45 | `▁[` | 518 | -100 (masked) |
| 46 | `/` | 29914 | -100 (masked) |
| 47 | `INST` | 25580 | -100 (masked) |
| 48 | `]` | 29962 | -100 (masked) |
| 49 | `▁I` | 306 | 306 |
| 50 | `▁hear` | 8293 | 8293 |
| 51 | `▁you` | 366 | 366 |
| 52 | `▁loud` | 22526 | 22526 |
| 53 | `▁and` | 322 | 322 |
| ... | | | |
| 444 | `</s>` | 2 | 2 (EOS kept) |


### Sample 2

**category** `ORDER` / **intent** `cancel_order` / 296 tokens (47 masked prompt, 249 trained)

Masked prompt (labels = -100):
```text
<s>[INST] <<SYS>>
You are a helpful, concise customer support agent for an online store.
<</SYS>>

can you help me cancel order {{Order Number}}? [/INST]
```
Trained response (labels = input ids, incl. EOS):
```text
Of course, I'm here to assist you in canceling your order with the number {{Order Number}}. Let's get this sorted out for you.

Please follow these steps:

1. Sign in to your {{Online Company Portal Info}} using your credentials.
2. Go to your '{{Online Order Interaction}}' or '{{Online Order Interaction}}' section, where you can find a record of all your purchases.
3. Locate the order associated with the number {{Order Number}} and click on it to view the details.
4. Look for the option to '{{Online Order Interaction}}' and select it to initiate the cancellation process.
5. Follow any additional instructions or provide any necessary feedback as prompted by the system.

Should you encounter any difficulties or have further questions, our dedicated support team is available to assist you. You can reach us during {{Customer Support Hours}} at {{Customer Support Phone Number}}, or you can connect with us using the Live Chat feature on our website at {{Website URL}}. We are committed to providing you with a satisfactory resolution and ensuring your utmost satisfaction.</s>
```
Token boundary:

| pos | token | id | label |
|---|---|---|---|
| 43 | `▁[` | 518 | -100 (masked) |
| 44 | `/` | 29914 | -100 (masked) |
| 45 | `INST` | 25580 | -100 (masked) |
| 46 | `]` | 29962 | -100 (masked) |
| 47 | `▁Of` | 4587 | 4587 |
| 48 | `▁course` | 3236 | 3236 |
| 49 | `,` | 29892 | 29892 |
| 50 | `▁I` | 306 | 306 |
| 51 | `'` | 29915 | 29915 |
| ... | | | |
| 295 | `</s>` | 2 | 2 (EOS kept) |


### Sample 3

**category** `DELIVERY` / **intent** `delivery_options` / 180 tokens (46 masked prompt, 134 trained)

Masked prompt (labels = -100):
```text
<s>[INST] <<SYS>>
You are a helpful, concise customer support agent for an online store.
<</SYS>>

do you ship to {{Delivery City}}? [/INST]
```
Trained response (labels = input ids, incl. EOS):
```text
Unquestionably! We offer shipping to {{Delivery City}}. Whether it's a small package or a larger shipment, we've got you covered. Our delivery options are designed to provide convenience and flexibility. You can choose from our standard shipping, expedited shipping, or even opt for overnight shipping for urgent needs. We want to ensure that your items reach you in a timely manner and in perfect condition. If you have any specific concerns or if there's anything else I can assist you with, please feel free to let me know. Shipping to {{Delivery City}} is effortless with us!</s>
```
Token boundary:

| pos | token | id | label |
|---|---|---|---|
| 42 | `▁[` | 518 | -100 (masked) |
| 43 | `/` | 29914 | -100 (masked) |
| 44 | `INST` | 25580 | -100 (masked) |
| 45 | `]` | 29962 | -100 (masked) |
| 46 | `▁Un` | 853 | 853 |
| 47 | `question` | 12470 | 12470 |
| 48 | `ably` | 2197 | 2197 |
| 49 | `!` | 29991 | 29991 |
| 50 | `▁We` | 1334 | 1334 |
| ... | | | |
| 179 | `</s>` | 2 | 2 (EOS kept) |

