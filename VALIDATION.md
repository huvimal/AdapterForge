# Local verification — 2026-11-15

## Checks completed

- Dependency consistency: `python -m pip check` passed.
- 23 unit/integration tests passed: strict JSON/schema/grounding and scoring, split disjointness/reproducibility, completion masking/padding, one real PEFT optimizer step on a tiny random model, API input/concurrency behavior, and UI behavior for an invalid output.
- Actual pretrained-model LoRA training completed on CUDA: 10 epochs, 320 optimizer steps; best epoch 10.
- Actual saved adapter reloaded for base/adapter/rules evaluation, with matching dataset, challenge and model revision checks.
- `verify_local.py` passed with the actual saved adapter: API readiness, HTTP 200 extraction through the in-process test client, HTTP 422 blank-input rejection, and the same raw model output in Streamlit's application test runner.

The actual smoke request was `Please urgently track my shipment for ORD-3197.` Both application paths produced:

```json
{"intent":"delivery","urgency":"urgent","order_id":"ORD-3197"}
```

This one-request smoke check confirms application integration, not general accuracy. The unit tests inject fake engines for API/UI behavior; their count is not a model performance score.

## Results and evidence

See [MODEL_CARD.md](MODEL_CARD.md) for the measured comparison and failures. [benchmarks/run-v1.json](benchmarks/run-v1.json) preserves the frozen original synthetic report, including raw generations, failed examples, training history and hashes.

Local artifacts:

- `private/base-12fd25f77366/source-manifest.json`: pinned model files/checksums.
- `private/run-v1/training.json`: actual configuration, training history and hardware evidence.
- `private/run-v1/best/adapter_model.safetensors`: learned LoRA weights, 3,702,168 bytes.
- `reports/comparison.json`: original full evaluation.
- `reports/local-smoke.json`: actual API/UI smoke output.
- `requirements-lock-windows.txt`: exact local package versions.

## Not established

Docker execution, hosted GitHub Actions, live-browser visual appearance, real HTTP socket transport, deployment, production concurrency/performance, and independent customer data quality were not verified. These are separate from the successful local training and application-runner checks.

## Recheck

From the project directory after setup:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 evaluate.py --adapter private/run-v1/best --output reports/recheck.json
.\.venv\Scripts\python.exe -X utf8 verify_local.py
```

The original public report remains frozen. Do not replace it with a result obtained after changing the dataset or adapting to the test failures.
