#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHARTS = ("training_curves.png", "confusion_matrix.png", "reliability.png")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path, problems: list, what: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        problems.append(f"{what}: cannot read {path.name} ({exc})")
        return None


def verify(root: Path, name: str = "synapse_v1", strict: bool = True) -> list[str]:
    root = Path(root)
    problems: list[str] = []
    models = root / "models"
    npz, meta_path = models / f"{name}.npz", models / f"{name}.json"
    if not npz.exists() or not meta_path.exists():
        return [f"model files {name}.npz/.json are missing from {models}"]
    meta = _load(meta_path, problems, "model metadata")
    if meta is None:
        return problems
    npz_hash = sha256(npz)
    if meta.get("npz_sha256") != npz_hash:
        problems.append("model metadata npz_sha256 does not match the weights file")
    run_id = meta.get("run_id")
    if not run_id:
        return problems + ["model metadata has no run_id"]
    if not meta.get("test_evaluated"):
        problems.append("the release model was not evaluated on the test set (a --skip-test run?)")

    run_dir = root / "experiments" / "runs" / run_id
    if not run_dir.is_dir():
        problems.append(f"run directory experiments/runs/{run_id} is missing")
    else:
        log = _load(run_dir / "log.json", problems, "run log")
        report = (run_dir / "report.md").read_text(encoding="utf-8") if (run_dir / "report.md").exists() else None
        if report is None:
            problems.append("run report.md is missing")
        for c in CHARTS:
            if not (run_dir / "charts" / c).exists():
                problems.append(f"chart {c} is missing from the run directory")
        if log:
            if log.get("run_id") != run_id:
                problems.append(f"run log is for {log.get('run_id')}, model is {run_id}")
            for key in ("test_accuracy", "validation_accuracy", "test_loss", "test_macro_f1"):
                if log.get("evaluation", {}).get(key) != meta.get("evaluation", {}).get(key):
                    problems.append(f"run log and model metadata disagree on {key}")
            if log.get("evaluation", {}).get("latency") != meta.get("evaluation", {}).get("latency"):
                problems.append("run log and model metadata disagree on latency")
            if log.get("created_utc") != meta.get("created_utc"):
                problems.append("run log and model metadata have different creation times")
        if report:
            if run_id not in report:
                problems.append("report.md does not mention the run id")
            ev = meta.get("evaluation", {})
            row = re.search(r"^\| Test accuracy \| ([^|]+?) \|", report, re.M)
            if ev.get("test_accuracy") is not None and (row is None or row.group(1) != f"{ev['test_accuracy']:.4%}"):
                problems.append(f"report.md test accuracy row ({row.group(1) if row else 'missing'}) != model's ({ev['test_accuracy']:.4%})")
            lat = ev.get("latency", {})
            lrow = re.search(r"^\| Single-sample latency \(median / p95\) \| ([^|]+?) \|", report, re.M)
            if lat and (lrow is None or lrow.group(1) != f"{lat['median_ms']:.3f} / {lat['p95_ms']:.3f} ms"):
                problems.append("report.md latency row differs from the model metadata")

    cfg_path = root / "configs" / "default.json"
    if cfg_path.exists():
        cfg = _load(cfg_path, problems, "config")
        if cfg is not None and cfg != meta.get("training_config"):
            problems.append("configs/default.json differs from the config that trained this model")
    else:
        problems.append("configs/default.json is missing")

    exp = root / "experiments"
    for fname, label in (("ood_eval.json", "OOD evaluation"), ("pipeline_eval.json", "pipeline evaluation")):
        p = exp / fname
        if not p.exists():
            problems.append(f"{label} ({fname}) is missing")
            continue
        rep = _load(p, problems, label)
        if rep:
            if rep.get("model_sha256") != npz_hash:
                problems.append(f"{label} was computed for a different model (hash mismatch)")
            if rep.get("run_id") != run_id:
                problems.append(f"{label} is for run {rep.get('run_id')}, model is {run_id}")
        if not p.with_suffix(".md").exists():
            problems.append(f"{label} markdown report is missing")

    if strict:
        runs = [d.name for d in (exp / "runs").iterdir()] if (exp / "runs").is_dir() else []
        extra = [r for r in runs if r != run_id]
        if extra:
            problems.append(f"leftover run directories from other runs: {sorted(extra)}")
        for stray in ("models/candidates", "models/registry.json", "data/user_samples.jsonl"):
            if (root / stray).exists():
                problems.append(f"{stray} must not ship in a release")
        extra_models = [p.name for p in models.glob("*") if p.name not in (f"{name}.npz", f"{name}.json")]
        if extra_models:
            problems.append(f"unexpected files in models/: {sorted(extra_models)}")

    mnist = root / "data" / "mnist"
    if mnist.exists():
        from training.dataset import MNIST_FILES
        for filename, expected in MNIST_FILES.values():
            f = mnist / filename
            if f.exists() and sha256(f) != expected:
                problems.append(f"data/mnist/{filename} does not match its pinned SHA-256")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check that weights, metadata, run log, report, charts and evaluations describe one run.", epilog="exit code 0 means consistent, 1 means the problems were listed",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--name", default="synapse_v1")
    ap.add_argument("--lenient", action="store_true", help="do not fail on leftover runs/candidates")
    args = ap.parse_args(argv)
    problems = verify(Path(args.root), args.name, strict=not args.lenient)
    if problems:
        print("RELEASE INCONSISTENT:")
        for p in problems:
            print("  -", p)
        return 1
    meta = json.loads((Path(args.root) / "models" / f"{args.name}.json").read_text())
    print(f"release consistent: run {meta['run_id']}, npz sha256 {meta['npz_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
