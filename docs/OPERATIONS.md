# Operations

## Run

    pip install -r requirements.txt
    python tools/fetch_mnist.py            # only if data/mnist is missing (hash-verified); needed for UI training jobs
    python -m app.server                   # http://127.0.0.1:5000  (waitress, 8 threads)
    python -m app.server --read-only       # inference only
    python -m app.server --dev             # Flask development server

Options: `--model-dir`, `--model-name`, `--host`, `--port`, `--max-job-seconds`, `--read-only`, `--no-rate-limit`, `--trust-proxy N` (only behind N proxies you control). Exit code 1 and a clear message if the model fails to load or verify.

## Fine-tuning on your drawings

Save drawings on the Draw page (pick the true digit, then Save). Fine-tuning unlocks at 30 drawings, with at least 8 held back for checking across at least 4 digits, so save a few of each digit, ideally 5 or more. The held-back drawings are never trained on and decide whether the result is accepted.

## Tested versions

Python 3.12, NumPy 2.4.4, Flask 3.1.3, waitress, matplotlib 3.10.8; Node 22 with jsdom 27 and canvas 3 for the UI test.

## State on disk

| Path | What | Back up? |
|---|---|---|
| `models/synapse_v1.{npz,json}` | release model (immutable; produced by `train.py`) | yes (ships with the release) |
| `models/candidates/` | models trained in the UI (max 30 kept; the active one and rollback targets are never pruned) | optional |
| `models/registry.json` | active model + rollback history (atomic writes) | yes if you use candidates |
| `data/user_samples.jsonl` | your labelled drawings (0600) | yes, it is your data |
| `experiments/runs/<run_id>/` | log, report, charts of the release run | ships with the release |

## Recover

- Wrong model active: UI Model tab -> "Roll back", or `POST /model/rollback`. Delete `models/registry.json` to return to the release model.
- Corrupt candidate: it is skipped in listings and never auto-loaded; the registry falls back to the release model if its active entry is unusable.
- Delete your drawings: UI Train tab, or `DELETE /feedback?confirm=yes`, or remove the file.

## Retrain the release

    python train.py                        # writes models/ and experiments/runs/<run_id>/
    python tools/evaluate_ood.py && python tools/evaluate_pipeline.py
    python tools/verify_release.py         # must print "release consistent"
    python tools/package_release.py        # dist/Shadowcr3-Synapse-<version>.zip (+ manifest, release notes)

Tuning runs: `python train.py --skip-test --out-root /tmp/try --set training.weight_decay=0.0003` never touches the test set or `models/`.

## Tests

    python -m unittest discover -s tests -t . -v
    cd tests/ui && npm install              # once: enables the browser-less UI test (jsdom + canvas)

## Docker (untested here)

    docker build -t synapse .
    docker run -p 127.0.0.1:5000:5000 -v synapse-data:/app/data -v synapse-models:/app/models synapse

## Manual QA checklist (cannot be automated here)

- [ ] Draw each digit 1-9 with a mouse, a trackpad and (if available) touch; note misreads.
- [ ] Draw thin, thick, tiny, off-centre and slanted digits; confirm the 28x28 preview looks like a centred MNIST digit.
- [ ] Draw a letter, a scribble, a circle: confirm "not sure" appears (many letters will still be accepted; see the OOD table).
- [ ] Save 10+ drawings, fine-tune, confirm the before/after report, then Undo.
- [ ] Train a small network, cancel mid-way, confirm the active model is unchanged.
- [ ] Test in Chrome, Firefox and Safari; check keyboard operation and reduced-motion.
