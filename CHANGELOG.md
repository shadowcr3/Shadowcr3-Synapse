# Changelog

## 1.1.0

- UI redesigned from scratch as a lab notebook: paper, ink and a serif, numbered figures with captions, fill-in-the-line forms, ledger tables. The network figure encodes signal as dot area with curved pen-line connections, isolates a hovered unit's strongest connections in both signal and error views, and labels the digit you drew.
- Rate limiting per client (infer / read / write / job / static buckets, 429 with Retry-After, X-Forwarded-For ignored unless `--trust-proxy`).
- Fine-tuning: the held-out set of your own drawings is now a deterministic, stratified split and is required. Fine-tuning needs at least 30 drawings with at least 8 held back over at least 4 digits (before, 5 to 9 drawings trained on everything with no independent check).
- Promotion gate: fine-tuned candidates carry their held-out evidence in their metadata, so promoting from the Model page applies the same check as the job; they must also descend from the active model.
- All comments and docstrings removed from the code (behaviour verified unchanged: identical syntax trees apart from docstrings, full test suite).
- The release ZIP is named after the app version (`app/version.py`), not the model version.

## 1.0.0

First complete release.

- Engine: from-scratch NumPy MLP (784-128-64-9), ReLU + softmax, fused softmax cross-entropy, Adam/AdamW, gradient checking.
- Training: float32, on-the-fly augmentation, cosine schedule, weight decay, selectable checkpoint metric, early stopping, cancellable, hooks for live observation. Fixed `data.split_seed` so every model shares one validation set.
- Evaluation: test set evaluated once per release run; reject threshold from validation; out-of-distribution probes (letters, symbols, scribbles, shapes, noise, blank, zero); drawing-pipeline robustness; JS/Python parity.
- App: Flask API served by waitress; canvas UI with live neuron visualisation (forward activations, backward error signals), training dashboard, architecture comparison, fine-tuning on the user's own drawings, versioned models with a validation gate and rollback.
- Release engineering: per-run directories, `verify_release` (weights/metadata/log/report/charts/evaluations must describe one run), packaged ZIP with a SHA-256 manifest and generated release notes.
