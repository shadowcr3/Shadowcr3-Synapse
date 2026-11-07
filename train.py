#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neural_network import SGD, Adam, Network, gradient_check
from training import evaluator
from training.augmentation import AugmentConfig
from training.config import apply_overrides, load_config
from training.dataset import normalize_pixels, prepare_data
from training.reporting import build_report, metrics_report
from training.trainer import TrainSettings, train

GRADCHECK_TOLERANCE = 1e-5


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def make_optimizer(tr: dict):
    if tr["optimizer"] == "adam":
        a = tr["adam"]
        return Adam(tr["learning_rate"], a["beta1"], a["beta2"], a["epsilon"], tr["weight_decay"])
    return SGD(tr["learning_rate"])


def make_settings(tr: dict) -> TrainSettings:
    return TrainSettings(
        epochs=tr["epochs"], batch_size=tr["batch_size"], learning_rate=tr["learning_rate"],
        lr_schedule=tr["lr_schedule"], lr_min_factor=tr["lr_min_factor"],
        selection_metric=tr["selection_metric"], patience=tr["early_stopping_patience"],
        augment=AugmentConfig.from_dict(tr["augmentation"]))


def gradient_preflight(net: Network, X_uint8, y, seed: int = 0) -> list[dict]:
    check = Network(net.layer_sizes, seed=0, dtype=np.float64)
    check.set_state([p.astype(np.float64) for p in net.parameters()])
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(X_uint8), size=16, replace=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = gradient_check(check, normalize_pixels(X_uint8[pick]), y[pick], max_per_tensor=25, rng=rng)
    for w in caught:
        print(f"  warning: {w.message}")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train and evaluate the Synapse release model.", epilog="examples:\n  python train.py\n  python train.py --set training.epochs=2\n  python train.py --skip-test --out-root /tmp/try",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out-root", help="write models/ and runs/ under this directory instead")
    ap.add_argument("--no-download", action="store_true", help="fail instead of downloading MNIST")
    ap.add_argument("--skip-gradcheck", action="store_true")
    ap.add_argument("--skip-test", action="store_true", help="do not evaluate the test set (tuning runs)")
    args = ap.parse_args(argv)

    sets = list(args.set)
    if args.epochs is not None:
        sets.append(f"training.epochs={args.epochs}")
    if args.seed is not None:
        sets.append(f"training.seed={args.seed}")
    if args.out_root:
        sets += [f"output.models_dir={Path(args.out_root) / 'models'}", f"output.runs_dir={Path(args.out_root) / 'runs'}"]
    if args.no_download:
        sets.append("data.download=false")
    cfg = apply_overrides(load_config(args.config), sets)

    tr, seed = cfg["training"], cfg["training"]["seed"]
    init_ss, shuffle_ss, augment_ss = np.random.SeedSequence(seed).spawn(3)
    created = datetime.now(timezone.utc)
    run_id = f"{cfg['model']['name']}_seed{seed}_{created.strftime('%Y%m%dT%H%M%SZ')}"
    print(f"run {run_id}")

    ds = prepare_data(cfg["data"], split_seed=cfg["data"]["split_seed"])
    print(f"data: {ds.info['n_train']:,} train / {ds.info['n_val']:,} val / {ds.info['n_test']:,} test "
          f"(classes {list(ds.classes)}; OOD probes {ds.info['n_ood_val']:,} val / {ds.info['n_ood_test']:,} test)")

    net = Network(cfg["model"]["layer_sizes"], seed=init_ss, dtype=cfg["model"]["dtype"])
    print(f"network {net.layer_sizes}: {net.num_parameters():,} trainable parameters ({net.dtype.name})")

    if not args.skip_gradcheck:
        worst = max(c["relative_error"] for c in gradient_preflight(net, ds.X_train, ds.y_train))
        print(f"gradient pre-flight (float64 copy): worst relative error {worst:.2e} (tolerance {GRADCHECK_TOLERANCE:.0e})")
        if worst > GRADCHECK_TOLERANCE:
            print("gradient check FAILED; refusing to train", file=sys.stderr)
            return 2

    history = train(net, make_optimizer(tr), ds.X_train, ds.y_train, ds.X_val, ds.y_val,
                    settings=make_settings(tr), shuffle_seed=shuffle_ss, augment_seed=augment_ss)
    print(f"restored parameters from epoch {history['best_epoch']} (best {tr['selection_metric']})")

    val = evaluator.evaluate_classifier(net, ds.X_val, ds.y_val, ds.classes)
    reject_cfg, reject_val, reject_test, test = cfg["reject"], None, None, None
    if reject_cfg["enabled"]:
        thr = evaluator.select_reject_threshold(evaluator.max_softmax_scores(net, ds.X_val),
                                                reject_cfg["target_in_distribution_accept"])
        reject_val = evaluator.evaluate_reject(net, ds.X_val, ds.y_val, ds.X_ood_val, thr)
    if not args.skip_test:
        test = evaluator.evaluate_classifier(net, ds.X_test, ds.y_test, ds.classes)
        if reject_cfg["enabled"]:
            reject_test = evaluator.evaluate_reject(net, ds.X_test, ds.y_test, ds.X_ood_test, thr)
    latency = evaluator.measure_latency(net, ds.X_val)

    report_blob = metrics_report(history, val, test, reject_val, reject_test, latency, ds)
    metadata = {
        "model_name": cfg["model"]["name"], "model_version": cfg["model"]["version"], "run_id": run_id,
        "created_utc": created.isoformat(), "test_evaluated": test is not None,
        "class_mapping": {str(i): int(c) for i, c in enumerate(ds.classes)},
        "preprocessing": {"input_shape": [28, 28], "flatten": "row-major", "scale": "uint8 / 255.0",
                          "browser_pipeline": "bbox crop -> fit 20x20 -> centre of mass in 28x28"},
        "reject": ({"method": "max_softmax", "threshold": reject_val["threshold"],
                    "target_in_distribution_accept": reject_cfg["target_in_distribution_accept"],
                    "selected_on": "in-distribution validation split"} if reject_cfg["enabled"] else None),
        "training_config": cfg, "best_epoch": history["best_epoch"], "dataset": ds.info,
        "evaluation": {"validation_accuracy": val["accuracy"], "validation_loss": val["loss"],
                       "test_accuracy": test["accuracy"] if test else None,
                       "test_macro_f1": test["macro_f1"] if test else None,
                       "test_ece": test["ece"] if test else None,
                       "test_loss": test["loss"] if test else None,
                       "latency": latency, "reject_validation": reject_val, "reject_test": reject_test},
        "metrics_report": report_blob,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
    }
    out = cfg["output"]
    npz_path, json_path = net.save(out["models_dir"], cfg["model"]["name"], metadata)
    size_bytes = npz_path.stat().st_size

    run_dir = Path(out["runs_dir"]) / run_id
    write_json(run_dir / "log.json", {**metadata, "history": history, "validation": val, "test": test})
    report = build_report(run_id, cfg, ds, history, val, test, reject_val, reject_test, latency,
                          size_bytes, net.num_parameters())
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    charts = evaluator.save_charts(history, test, list(ds.classes), run_dir / "charts") if test else []
    print("\n" + report)
    print(f"saved: {npz_path}, {json_path}, {run_dir}/ (log.json, report.md, {len(charts)} chart(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
