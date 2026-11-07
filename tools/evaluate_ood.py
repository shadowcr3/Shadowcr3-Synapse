#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference import Predictor
from app.preprocessing import BlankInputError, mnist_style_normalize
from neural_network.activations import softmax
from tools import canvas_sim as sim
from training.dataset import load_mnist
from training.metrics import roc_auc

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
SYMBOLS = list("+-=#@%&?!$*~/\\()[]{}<>^_:;|")


def classify(pred: Predictor, canvases):
    kept, out = [], [None] * len(canvases)
    for i, c in enumerate(canvases):
        try:
            kept.append((i, mnist_style_normalize(c)))
        except BlankInputError:
            out[i] = (True, None, False, None)
    if kept:
        X = np.stack([k[1].reshape(-1) for k in kept]).astype(np.float64) / 255.0
        probs = softmax(pred.network.predict_logits(X))
        for (i, _), p in zip(kept, probs):
            conf, top = float(p.max()), pred.digits[int(p.argmax())]
            out[i] = (False, conf, pred.threshold is None or conf >= pred.threshold, top)
    return out


def summarise(results) -> dict:
    n = len(results)
    blocked = sum(r[0] for r in results)
    model_rejected = sum((not r[0]) and (not r[2]) for r in results)
    accepted = sum(r[2] for r in results)
    return {"n": n, "blocked_before_model": blocked, "rejected_by_model": model_rejected, "false_accepts": accepted,
            "reject_rate": (blocked + model_rejected) / n, "accept_rate": accepted / n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure how the release model handles input that is not a digit 1-9.", epilog="writes experiments/ood_eval.json and ood_eval.md",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--model-name", default="synapse_v1")
    ap.add_argument("--data-dir", default="data/mnist")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args(argv)

    pred = Predictor.from_dir(args.model_dir, args.model_name)
    sha = hashlib.sha256((Path(args.model_dir) / f"{args.model_name}.npz").read_bytes()).hexdigest()
    rng = np.random.default_rng(args.seed)
    n = args.samples

    raw = load_mnist(args.data_dir, download=False)
    xt, yt = raw["x_test"], raw["y_test"]
    pick = lambda digit_mask, k: xt[np.flatnonzero(digit_mask)[rng.permutation(int(digit_mask.sum()))[:k]]]
    valid = pick(yt != 0, n)
    zeros = pick(yt == 0, n)

    cats = {
        "valid digits 1-9 (reference)": [sim.digit_to_canvas(x, rng) for x in valid],
        "digit 0 (MNIST)": [sim.digit_to_canvas(x, rng) for x in zeros],
        "blank canvas": [np.zeros((sim.SIZE, sim.SIZE)) for _ in range(20)],
        "tiny mark / dot": [sim.dot(rng) for _ in range(n)],
        "random scribbles": [sim.scribble(rng) for _ in range(n)],
        "geometric shapes": [sim.shape(sim.SHAPES[i % len(sim.SHAPES)], rng) for i in range(n)],
        "speckle noise": [sim.noise(rng) for _ in range(n)],
    }
    letters = [(ch, f) for ch in LETTERS for f in sim.FONTS]
    cats["letters A-Z a-z (6 fonts)"] = [sim.glyph(ch, f, rng) for ch, f in letters]
    cats["symbols (+ - = # @ % & ? ! ...)"] = [sim.glyph(ch, f, rng) for ch in SYMBOLS for f in sim.FONTS]

    results = {name: classify(pred, canv) for name, canv in cats.items()}
    summary = {name: summarise(r) for name, r in results.items()}

    ref_conf = np.array([r[1] for r in results["valid digits 1-9 (reference)"] if r[1] is not None])
    for name, r in results.items():
        confs = np.array([x[1] for x in r if x[1] is not None])
        summary[name]["auroc_vs_valid"] = (roc_auc(ref_conf, confs) if len(confs) >= 5 and name != "valid digits 1-9 (reference)" else None)

    per_char = {}
    for (name, chars) in (("letters", LETTERS), ("symbols", SYMBOLS)):
        res = results["letters A-Z a-z (6 fonts)" if name == "letters" else "symbols (+ - = # @ % & ? ! ...)"]
        k = len(sim.FONTS)
        rows = []
        for i, ch in enumerate(chars):
            chunk = res[i * k:(i + 1) * k]
            acc = [c for c in chunk if c[2]]
            rows.append({"char": ch, "accepted_in_fonts": len(acc), "of": k,
                         "mostly_read_as": (max(set(c[3] for c in acc), key=[c[3] for c in acc].count) if acc else None)})
        per_char[name] = sorted(rows, key=lambda r: -r["accepted_in_fonts"])

    report = {"model_sha256": sha, "run_id": pred.metadata.get("run_id"), "threshold": pred.threshold,
              "samples_per_category": n, "seed": args.seed, "categories": summary, "per_character": per_char}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "ood_eval.json").write_text(json.dumps(report, indent=2))

    ref = summary["valid digits 1-9 (reference)"]
    lines = ["# Out-of-distribution evaluation", "",
             f"Model run `{report['run_id']}` (npz sha256 `{sha[:16]}…`), reject threshold {pred.threshold:.4f}. "
             f"{n} drawings per category unless noted; every drawing is generated as a 280x280 canvas and normalised exactly as the browser does. "
             "**Reject** = stopped before the network (blank / too small) or below the confidence threshold. "
             "**False accept** = the model confidently names a digit.", "",
             "| Category | n | Blocked early | Rejected by model | **False accepts** | Reject rate | AUROC vs valid |", "|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        au = "–" if s["auroc_vs_valid"] is None else f"{s['auroc_vs_valid']:.3f}"
        bold = "" if name.startswith("valid") else "**"
        lines.append(f"| {name} | {s['n']} | {s['blocked_before_model']} | {s['rejected_by_model']} | {bold}{s['false_accepts']}{bold} | {s['reject_rate']:.1%} | {au} |")
    lines += ["", f"Reference: {ref['accept_rate']:.1%} of valid drawings are accepted (the rest are rejected as low confidence).", "",
              "## Characters most often accepted as a digit", ""]
    for name in ("letters", "symbols"):
        top = [r for r in per_char[name] if r["accepted_in_fonts"] >= 3][:12]
        lines.append(f"- {name}: " + (", ".join(f"`{r['char']}`→{r['mostly_read_as']} ({r['accepted_in_fonts']}/{r['of']})" for r in top) or "none accepted in half or more of the fonts"))
    lines += ["", "## Reading this", "",
              "- A closed-set classifier has no 'none of the above' output; the reject threshold is a confidence heuristic, not a detector for non-digits.",
              "- Characters that really look like a digit (`I`, `l`, `|` ~ 1; `O` ~ 0; `Z` ~ 2; `S` ~ 5; `B` ~ 8) are expected to be accepted and cannot be separated by any digit model.",
              "- Fonts here are printed DejaVu faces, not handwriting; scribbles, shapes and noise are procedural. These are probes, not a complete definition of 'not a digit'.", ""]
    (out / "ood_eval.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
