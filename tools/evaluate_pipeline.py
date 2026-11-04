#!/usr/bin/env python3
"""End-to-end robustness of the drawing pipeline (canvas -> 28x28 -> model).

MNIST test digits are re-rendered as canvas drawings (random size and position, several stroke
thicknesses), normalised exactly like the browser, and classified. Also checks that the
JavaScript implementation agrees with the Python reference on whole predictions, and that the
HTTP API returns the same numbers as direct model calls.

Limits: these are *re-rendered MNIST digits*, so they test geometry and stroke-width invariance,
not how real people draw. Collect real drawings (UI: Teach the network) for that.

    python tools/evaluate_pipeline.py [--samples 1500]  ->  experiments/pipeline_eval.{json,md}
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import create_app                                           # noqa: E402
from app.inference import Predictor                                  # noqa: E402
from app.preprocessing import mnist_style_normalize                  # noqa: E402
from app.server import LOOPBACK_HOSTS                                # noqa: E402
from neural_network.activations import softmax                       # noqa: E402
from tools import canvas_sim as sim                                  # noqa: E402
from training.dataset import load_mnist                              # noqa: E402

THICKNESS = (0, 2, 4, 6, 8)


def score(pred: Predictor, X28_uint8: np.ndarray, y_digit: np.ndarray) -> dict:
    probs = softmax(pred.network.predict_logits(X28_uint8.reshape(len(X28_uint8), -1).astype(np.float64) / 255.0))
    top = np.array(pred.digits)[probs.argmax(1)]
    conf = probs.max(1)
    acc_mask = conf >= (pred.threshold or 0.0)
    return {"n": int(len(y_digit)), "accuracy": float(np.mean(top == y_digit)),
            "accept_rate": float(acc_mask.mean()),
            "accuracy_on_accepted": float(np.mean(top[acc_mask] == y_digit[acc_mask])) if acc_mask.any() else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--model-name", default="synapse_v1")
    ap.add_argument("--data-dir", default="data/mnist")
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--js-samples", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args(argv)

    pred = Predictor.from_dir(args.model_dir, args.model_name)
    sha = hashlib.sha256((Path(args.model_dir) / f"{args.model_name}.npz").read_bytes()).hexdigest()
    rng = np.random.default_rng(args.seed)
    raw = load_mnist(args.data_dir, download=False)
    idx = np.flatnonzero(raw["y_test"] != 0)
    idx = idx[rng.permutation(len(idx))[:args.samples]]
    X28, y = raw["x_test"][idx], raw["y_test"][idx].astype(int)

    report = {"model_sha256": sha, "run_id": pred.metadata.get("run_id"), "threshold": pred.threshold,
              "samples": int(len(idx)), "seed": args.seed}
    report["native_mnist_28x28"] = score(pred, X28, y)

    report["canvas_pipeline"] = {}
    canvases_by_t = {}
    for t in THICKNESS:
        canv = [sim.digit_to_canvas(x, rng, thickness=t) for x in X28]
        canvases_by_t[t] = canv
        X = np.stack([mnist_style_normalize(c) for c in canv])
        report["canvas_pipeline"][f"extra_stroke_width_{2 * t}px"] = score(pred, X, y)

    # JavaScript implementation vs Python reference, on whole predictions
    node = shutil.which("node")
    if node and args.js_samples:
        k = args.js_samples
        canv = canvases_by_t[2][:k]
        req = {"width": sim.SIZE, "height": sim.SIZE,
               "images_b64": [base64.b64encode(np.clip(c, 0, 255).astype(np.uint8).tobytes()).decode() for c in canv]}
        res = subprocess.run([node, str(ROOT / "tools" / "js_preprocess_cli.js")], input=json.dumps(req), capture_output=True,
                             text=True, check=True, timeout=600)
        js = [np.array(r["pixels"], dtype=np.uint8).reshape(28, 28) for r in json.loads(res.stdout)]
        py = [mnist_style_normalize(np.clip(c, 0, 255).astype(np.uint8)) for c in canv]
        pj = score(pred, np.stack(js), y[:k]); pp = score(pred, np.stack(py), y[:k])
        top = lambda arr: np.array(pred.digits)[pred.network.predict_logits(np.stack(arr).reshape(k, -1) / 255.0).argmax(1)]
        report["js_vs_python"] = {"n": k, "max_pixel_difference": int(max(np.abs(a.astype(int) - b.astype(int)).max() for a, b in zip(js, py))),
                                  "prediction_agreement": float(np.mean(top(js) == top(py))),
                                  "js_accuracy": pj["accuracy"], "python_accuracy": pp["accuracy"]}

    # HTTP API vs direct model call
    app = create_app(args.model_dir, args.model_name, LOOPBACK_HOSTS)
    client = app.test_client()
    maxdiff, n_api = 0.0, 60
    for c in canvases_by_t[0][:n_api]:
        px = mnist_style_normalize(c)
        r = client.post("/predict", json={"pixels": [int(v) for v in px.reshape(-1)]}).json
        direct = softmax(pred.network.predict_logits(px.reshape(1, -1) / 255.0))[0]
        maxdiff = max(maxdiff, float(np.abs(np.array(r["scores"]) - direct).max()))
    report["api_vs_direct"] = {"n": n_api, "max_score_difference": maxdiff}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline_eval.json").write_text(json.dumps(report, indent=2))
    n = report["native_mnist_28x28"]
    lines = ["# Drawing-pipeline evaluation", "",
             f"Model run `{report['run_id']}` (npz sha256 `{sha[:16]}…`), {report['samples']} MNIST test digits (1-9), seed {args.seed}. "
             f"Reject threshold {pred.threshold:.4f}.", "",
             "Digits are re-rendered as 280x280 canvas drawings (random size 100-240 px, random position) with extra stroke thickening, "
             "then normalised like the browser does. **This tests geometry and stroke-width invariance, not real handwriting.**", "",
             "| Input | Accuracy | Accepted | Accuracy when accepted |", "|---|---|---|---|",
             f"| Native MNIST 28x28 (no pipeline) | {n['accuracy']:.2%} | {n['accept_rate']:.1%} | {n['accuracy_on_accepted']:.2%} |"]
    for name, s in report["canvas_pipeline"].items():
        lines.append(f"| Canvas, {name.replace('extra_stroke_width_', '+').replace('px', ' px stroke')} | {s['accuracy']:.2%} | {s['accept_rate']:.1%} | {s['accuracy_on_accepted']:.2%} |")
    if "js_vs_python" in report:
        j = report["js_vs_python"]
        lines += ["", f"JavaScript vs Python pipeline on {j['n']} canvases: max pixel difference {j['max_pixel_difference']}, "
                      f"prediction agreement {j['prediction_agreement']:.1%} (JS accuracy {j['js_accuracy']:.2%}, Python {j['python_accuracy']:.2%})."]
    lines += ["", f"HTTP /predict vs direct model call on {n_api} drawings: max score difference {report['api_vs_direct']['max_score_difference']:.2e}.", ""]
    (out / "pipeline_eval.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
