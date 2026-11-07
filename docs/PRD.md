# Product requirements

**Product:** Synapse, a local, offline handwritten-digit (1-9) recogniser whose neural network is implemented from scratch in NumPy, with a UI that shows how the network works and lets the user train it.

## Goals

1. Recognise digits 1-9 drawn on a canvas, reporting confidence and refusing when unsure.
2. Make the network visible: every neuron's activation as a drawing passes through, how errors propagate backwards, how weights, loss and accuracy change during training, and what changing hidden-layer sizes does.
3. Let the user improve the network with their own drawings, safely (validated, reversible).
4. Run entirely offline on one machine, reproducibly, with measured (never claimed) quality.

## Non-goals

No cloud service, accounts, or multi-user training; no recognition of letters or multi-digit numbers; no deep-learning frameworks in the core; no promise of accuracy on arbitrary handwriting.

## Requirements and where they are met

| # | Requirement | Implementation | Verified by |
|---|---|---|---|
| R1 | Draw -> predict, confidence, reject low confidence | canvas, `/predict`, reject threshold from validation | `test_api`, UI test |
| R2 | Live neuron activations while drawing | `/trace` + `NetworkView` (Live toggle) | UI test |
| R3 | Backward error propagation view | `introspection.trace` deltas = true backprop quantities | `test_introspection` (equals `backward()`) |
| R4 | Weights, loss, accuracy, gradient norms live during training | job events, charts, weight tiles, probe digit | `test_jobs`, UI test |
| R5 | Hidden-layer size experiments | `sweep` job, comparison table/charts | `test_jobs`, UI test |
| R6 | More training data | augmentation (fresh variants each epoch), user drawings, MNIST replay | `test_augmentation`, `test_jobs` |
| R7 | Train from the UI, alongside use | train / fine-tune jobs on a private copy, atomic swap | `test_jobs`, `test_models` |
| R8 | Safe model changes | validation gate (including a stratified held-out set of your own drawings for fine-tuned models), rollback, no force | `test_models`, `test_jobs`, `test_api` |
| R12 | Abuse resistance on a local server | per-client rate limits, Origin/Host checks, size limits | `test_ratelimit`, `test_api` |
| R9 | Efficient | float32, fused ops, in-place Adam, prefetch | `docs/PERFORMANCE.md` |
| R10 | Production hardening | see `docs/SECURITY.md`, `docs/OPERATIONS.md` | `test_api` |
| R11 | Consistent, verifiable release | `verify_release`, manifest | `test_release` |

## Quality targets (release model; measured values are in `RELEASE_NOTES.md`)

Test accuracy >= 97%; single-sample latency < 100 ms; model < 5 MB; offline; reproducible training.
