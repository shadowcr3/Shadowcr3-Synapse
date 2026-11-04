# Changelog

## 1.0.0

First complete release.

- Engine: from-scratch NumPy MLP (784-128-64-9), ReLU + softmax, fused softmax cross-entropy, Adam/AdamW, gradient checking.
- Training: float32, on-the-fly augmentation, cosine schedule, weight decay, selectable checkpoint metric, early stopping, cancellable, hooks for live observation. Fixed `data.split_seed` so every model shares one validation set.
- Evaluation: test set evaluated once per release run; reject threshold from validation; out-of-distribution probes (letters, symbols, scribbles, shapes, noise, blank, zero); drawing-pipeline robustness; JS/Python parity.
- App: Flask API served by waitress; canvas UI with live neuron visualisation (forward activations, backward error signals), training dashboard, architecture comparison, fine-tuning on the user's own drawings, versioned models with a validation gate and rollback.
- Release engineering: per-run directories, `verify_release` (weights/metadata/log/report/charts/evaluations must describe one run), packaged ZIP with a SHA-256 manifest and generated release notes.
