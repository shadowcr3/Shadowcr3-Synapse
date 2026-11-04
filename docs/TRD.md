# Technical design

## Layers and dependency rule

    neural_network/   engine: layers, activations, loss, optimisers, backprop, introspection, save/load   (NumPy only)
    training/         dataset, augmentation, trainer, evaluator, metrics, reporting, config               (imports neural_network)
    app/              preprocessing, inference, models (versions), feedback, jobs, routes, server        (inference.py and
                                                                                                        preprocessing.py import neural_network only)
    frontend/         classic scripts, no framework, no build step; talks to the API only
    tools/            evaluation, benchmark, release verification/packaging (not imported by the app)

Inference is independent of training (enforced by a test). Only `jobs.py` brings training into the server.

## Data flow: a drawing

canvas 280x280 (white strokes on black, 20 px brush) -> `preprocess.js`: bounding box of ink -> crop -> scale max side to 20 px with an
area filter -> stretch peak to 255 -> centre in 28x28 -> translate so the centre of mass is at (13.5, 13.5) -> 784 uint8 -> `POST /trace`
-> `parse_pixels` (strict validation, blank rejection) -> `/255` -> forward pass -> softmax -> reject if max < threshold.
`app/preprocessing.py:mnist_style_normalize` is the reference implementation; a test proves the JS port matches pixel for pixel.

## Model and training

784-128-64-9, ReLU hidden layers, softmax output, He init (hidden) / Glorot (output), float32. Adam (AdamW weight decay on weights only),
cosine learning-rate schedule, batch 64, 30 epochs, augmentation on (rotation +-12 deg, scale +-10%, shear +-0.1, shift +-2 px, 20% stroke
thickening). The checkpoint with the lowest validation loss is restored. Seeds: one master seed spawns independent streams for init, shuffling
and augmentation; the train/validation split uses `data.split_seed` so every model shares one validation set.

## Evaluation discipline

Validation decides everything (checkpoint, reject threshold, hyper-parameters). The official test set is evaluated exactly once, at the end of a
release run (`--skip-test` for tuning runs). Models trained in the UI never see the test set. The reject threshold accepts 95% of in-distribution
validation digits; digit 0 is used only to measure rejection, not to set it.

## Serving and model lifecycle

`ModelStore` keeps the release model and candidates (`models/candidates/<id>`), the active id and a rollback history in `models/registry.json`
(atomic writes). `/predict` reads the active `Predictor` through one reference, so promotion is an atomic swap. A candidate is promoted only if
(a) it was evaluated on the same validation split, (b) its validation accuracy is within 0.003 of the active model's or better, and (c) accuracy on
the user's held-out drawings did not get worse. Fine-tuning mixes the user's drawings (augmented, repeated up to 20x) with a replay of MNIST so the
network does not forget; before/after accuracy on MNIST validation and on held-out drawings is reported.

## Jobs

One worker thread, event ring buffer, cooperative cancellation through trainer hooks, wall-clock limit, parameter caps. Clients poll
`/jobs/status?since=N` (no websockets: simpler, proxy-friendly, no idle connections held open).

## Known design limits

- The reject rule is max-softmax confidence. It is not an out-of-distribution detector for letters and symbols (see `experiments/ood_eval.md`).
- The server is unauthenticated by design; it is a local tool. Training state is in memory; a restart loses the live event log (models and drawings persist).
- Dropout, batch-norm and convolutions are out of scope.
