# AdapterForge run-v1 model card

## Purpose and scope

Local LoRA adapter for structured extraction from short English support requests. It predicts `intent`, `urgency`, and a literal `order_id` or null. This is a small reproducible fine-tuning experiment on self-authored synthetic data, not a deployed customer-support product.

Base: [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), Apache-2.0, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The adapter requires this base and tokenizer. Its known model-card limitations remain relevant. No model API or proprietary customer dataset was used.

## Actual training run

Verified locally on 2026-11-15 with Python 3.11.9, PyTorch 2.10.0+cu128 and NVIDIA GeForce RTX 3050 6GB Laptop GPU.

- 512 train / 64 validation / 64 test requests; 8 separately authored challenges.
- LoRA rank 16, alpha 32, dropout 0.05; `q_proj` and `v_proj`; completion-only causal language-model objective.
- AdamW, learning rate 0.001, weight decay 0.01; microbatch 4, accumulation 4, 10 epochs, seed 27, 320 optimizer steps.
- Float32 stored weights with bfloat16 autocast. No quantization or QLoRA claim.
- 921,600 trainable parameters out of 135,436,608 including adapters (0.6805%).
- Training took 512.72 seconds; peak PyTorch allocated CUDA memory was 1,700,862,976 bytes. This excludes memory reserved by the allocator, driver and other applications.
- Epoch 10 selected by minimum validation completion-token loss: 0.00049091. Test/challenge generation was inspected only after selection.
- Absolute LoRA weight change was nonzero; a frozen embedding sample was unchanged. The architecture integration test separately checks every frozen parameter on a tiny random model. The full pretrained model's frozen weights were not all hashed before/after training.

Seed/configuration and exact environment are recorded; bit-identical results across GPU/library versions are not guaranteed.

## Measured results

Frozen full report: [benchmarks/run-v1.json](benchmarks/run-v1.json). Same prompts, greedy generation and 96-new-token cap for base and adapter. Base is the same loaded model with adapters disabled. Rules are a separate handcrafted reference, not an output repair or fallback.

| Split | Variant | Schema valid | Exact match | Non-null field micro-F1 |
|---|---|---:|---:|---:|
| Held-out templates, n=64 | Base | 0/64 | 0/64 | 0.0000 |
| Held-out templates, n=64 | Rules | 64/64 | 59/64 | 0.9688 |
| Held-out templates, n=64 | LoRA | 64/64 | 64/64 | 1.0000 |
| Authored challenges, n=8 | Base | 0/8 | 0/8 | 0.0000 |
| Authored challenges, n=8 | Rules | 8/8 | 5/8 | 0.8571 |
| Authored challenges, n=8 | LoRA | 6/8 | 6/8 | 0.8947 |

Adapter raw JSON was valid on 7/8 challenges, but one of those JSON objects violated the field schema. Invalid schema counts as a missing prediction for all fields. The base's zero score reflects failure to follow this strict raw JSON contract under the fixed prompt/token budget; it does not establish zero general language understanding.

The held-out set uses disjoint wrapper-template groups/order IDs but shares task vocabulary and issue phrases with training. Its 100% result is not evidence of 100% real-world accuracy. Eight authored challenges are too few for broad robustness or security claims.

## Preserved adapter failures

1. An ambiguous request mentioning `ORD-9992` and `ORD-9993` produced malformed JSON listing both IDs instead of returning null.
2. A stock-availability question produced `"order_id":"normal"` instead of null; JSON parsed but the schema validator rejected it.

Neither output was repaired and no retraining was performed after observing these failures. Rejecting an output is not equivalent to answering it correctly. New experiments need a new version and a new independent assessment after test-informed development.

## Timing interpretation

On 64 held-out examples, warmed batched generation (batch 8) took 79.44 seconds for base and 26.51 seconds for adapter. Adapter amortized generation time was 0.414 seconds per case; this is not single-request latency. Sequence lengths differ, so this is not proof of a universal serving speedup. No production load test was performed.

## Use limitations

Human review is required for downstream use. Schema validation cannot prove semantic correctness. This version has not established multilingual support, independent real-customer quality, resistance to malicious instructions, or suitability for financial/medical/legal decisions. The local API has no authentication and is intended for loopback use. Docker and hosted CI configuration are provided but were not executed as part of this local verification.

Adapter SHA-256: `eca4415251abead0c9a20172c415b408428fbc7f50ada2417b3c08f179901d44`.
