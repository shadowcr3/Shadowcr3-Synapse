# Shadowcr3 Synapse

A handwritten-digit (1-9) recogniser whose neural network is implemented **from scratch in NumPy** (no PyTorch/TensorFlow/Keras), wrapped in an offline web app that
shows the network thinking, lets you train it, and lets it learn from your own drawings.

![status](https://img.shields.io/badge/status-release%201.0.0-blue)

## Quick start

    pip install -r requirements.txt
    python -m app.server                      # open http://127.0.0.1:5000

Draw a digit. With **Live** on, every neuron lights up as you draw. Pick the true digit to watch the error signal flow backwards.
The **Train** tab trains new networks, compares hidden-layer sizes and fine-tunes on your saved drawings; the **Model** tab shows measured metrics, versions and rollback.
Training jobs need MNIST on disk: `python tools/fetch_mnist.py` (hash-verified; already included in the release ZIP).

## What you can see and do

| | Where |
|---|---|
| Neurons activating as an image passes through (every neuron of every layer, strongest connections) | Predict, "The network, live" |
| How errors propagate backwards (per-neuron delta = dLoss/dz, error edges, true-digit ring) | same panel, "Backward" |
| What each neuron looks for (its 784 weights as an image; hover any neuron) | inspector; Model tab gallery |
| How weights change during training (live tiles, change-since-start, per-layer movement) | Train, "Live training" |
| How errors/gradients behave per layer; loss and accuracy per epoch with live batch loss | Train |
| What happens when hidden-layer sizes change | Train, "What if the hidden layers change?" |
| More training data | augmentation (new variants every epoch) + your own drawings + MNIST replay |
| Safe model changes | validation gate, rollback, versions table |

## Layout

    neural_network/   engine (NumPy only): layers, activations, loss, Adam, backprop, introspection, save/load
    training/         dataset, augmentation, trainer, evaluator, metrics, config, reporting
    app/              preprocessing, inference, model store, feedback store, job manager, Flask routes, server
    frontend/         UI (plain JS, no build step, no external assets)
    tools/            OOD + pipeline evaluation, benchmark, release verify/package, MNIST fetch
    tests/            Python tests, JS tests, end-to-end UI test (jsdom)
    docs/             PRD, TRD, API, SECURITY, OPERATIONS, MATHEMATICS, PERFORMANCE, RULES
    train.py          reproducible training + evaluation of the release model

## Results

See `RELEASE_NOTES.md` (generated from the shipped files) and `experiments/runs/<run_id>/report.md`. Read the caveats there: accuracy is for MNIST digits;
rejection of non-digit input is **not universal** (`experiments/ood_eval.md`: letters and symbols are often accepted).

## Tests

    python -m unittest discover -s tests -t .
    cd tests/ui && npm install               # enables the UI end-to-end test (jsdom + canvas)

## Docs

`docs/API.md`, `docs/TRD.md`, `docs/SECURITY.md`, `docs/OPERATIONS.md`, `docs/MATHEMATICS.md`, `docs/PERFORMANCE.md`, `docs/RULES.md`, `docs/PRD.md`.

## License

GPL-3.0

