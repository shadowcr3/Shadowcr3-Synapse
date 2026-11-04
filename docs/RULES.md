# Engineering rules (enforced where a test exists)

1. **No frameworks in the core.** `neural_network/` uses NumPy only.
2. **Gradients are verified** against finite differences in tests and before every training run (float64 copy).
3. **Never fabricate performance.** Every number in reports and release notes is produced by code; `verify_release` compares reports with metadata.
4. **Reproducibility.** One master seed; fixed split seed; configs recorded in every model's metadata; same seed gives identical weights on the same machine.
5. **The test set is sacred.** The trainer cannot see it; it is evaluated once per release run; tuning runs use `--skip-test`; UI-trained models never use it.
6. **Training never touches the served model.** Jobs train copies; promotion goes through a validation gate; every promotion is reversible.
7. **Inference is independent of training** (`app/inference.py`, `app/preprocessing.py` do not import `training`).
8. **Untrusted input is validated at the boundary**: strict `parse_pixels`, body-size limit, JSON-only, whitelisted static files, bounded job parameters, label and id validation.
9. **Model files load safely**: `allow_pickle=False`, SHA-256, tensor names, shapes and finiteness checked; candidate ids match `^[a-z0-9][a-z0-9_-]{0,63}$`.
10. **Offline.** No outbound network from the server; the UI loads nothing from other origins (tested). MNIST download exists only in `tools/fetch_mnist.py` and `train.py`, hash-verified.
11. **Reject, don't guess.** Blank/tiny drawings are refused; low confidence is reported as "not sure" instead of a digit.
12. **A release is one run.** Weights, metadata, log, report, charts and evaluation reports must agree (`tools/verify_release.py`), and the ZIP carries a manifest.
