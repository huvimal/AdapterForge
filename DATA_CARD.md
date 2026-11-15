# Dataset card

Self-authored synthetic English support requests. There is no real customer data and no third-party customer dataset. Labels describe intent (refund/delivery/account/other), explicit urgency (normal/urgent), and a literal order ID or null.

The generated splits contain 512 train examples from eight wrapper-template groups, 64 validation examples from two other groups, and 64 test examples from two additional groups. Exact request text, group IDs and non-null order IDs are disjoint across those splits. The task vocabulary and short issue phrases are intentionally shared: this is not a test of independent real-world language or domain generalization.

Eight separately authored challenge requests cover semantic paraphrases, negation, ambiguous order references and an instruction-shaped input. Their hash is frozen in the training manifest. They are not used in optimization or checkpoint selection. A small challenge set cannot establish robustness or security.

`data.py` generates the three principal JSONL splits and manifest deterministically. `data/challenge.jsonl` is authored separately. Do not alter any test/challenge reference after inspecting its predictions; create a new dataset version and preserve previous results when changing the task.

Schema-valid output is not sufficient for correctness. Non-null order IDs must occur as a unique literal reference in the request. The validator does not infer intent or urgency and does not repair model output. Null output can still miss an available reference and is penalized by field/exact-match metrics.

This first version does not claim Vietnamese support, broad extraction quality, production suitability or business impact. The UI/API are demonstrations for authorized/synthetic requests only.
