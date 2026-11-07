
from __future__ import annotations


def metrics_report(history, val, test, reject_val, reject_test, latency, ds) -> dict:
    keys = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_seconds")
    rep = {"history": {k: history[k] for k in keys}, "best_epoch": history["best_epoch"],
           "epochs_run": history["epochs_run"], "stopped_early": history["stopped_early"],
           "validation": {k: val[k] for k in ("n", "accuracy", "loss", "macro_f1", "ece")},
           "latency": latency,
           "dataset": {k: ds.info[k] for k in ("n_train", "n_val", "n_test", "classes", "excluded_digits")},
           "reject_validation": reject_val}
    if test is not None:
        rep["test"] = {k: test[k] for k in ("n", "accuracy", "loss", "macro_f1", "ece",
                                            "per_class", "confusion_matrix")}
        rep["reject_test"] = reject_test
    return rep


def build_report(run_id, cfg, ds, history, val, test, reject_val, reject_test, latency,
                 size_bytes, param_count) -> str:
    t = cfg["targets"]
    size_mb = size_bytes / 1e6
    ok = lambda b: "met" if b else "NOT met"
    tr = cfg["training"]
    lines = [
        f"# Evaluation report - {run_id}", "",
        "All values below were measured by `train.py`; none are estimated.", "",
        "## Setup", "",
        f"- Classes {list(ds.classes)}; architecture {cfg['model']['layer_sizes']} ({param_count:,} parameters, {cfg['model']['dtype']})",
        f"- Split: {ds.info['n_train']:,} train / {ds.info['n_val']:,} validation / {ds.info['n_test']:,} test "
        "(official MNIST test set, filtered)",
        f"- Optimizer {tr['optimizer']} lr {tr['learning_rate']} ({tr['lr_schedule']}), weight decay {tr['weight_decay']}, "
        f"batch {tr['batch_size']}, epochs {history['epochs_run']}/{tr['epochs']}, seed {tr['seed']}, "
        f"augmentation {'on' if tr['augmentation']['enabled'] else 'off'}",
        f"- Checkpoint: epoch {history['best_epoch']} by {tr['selection_metric']}"
        + (" (early stopped)" if history["stopped_early"] else ""), "",
        "## Results", "", "| Metric | Measured | Target | Status |", "|---|---|---|---|",
    ]
    if test is not None:
        lines.append(f"| Test accuracy | {test['accuracy']:.4%} | >= {t['test_accuracy']:.0%} | {ok(test['accuracy'] >= t['test_accuracy'])} |")
    else:
        lines.append("| Test accuracy | not evaluated (`--skip-test` run) | | |")
    lines += [
        f"| Single-sample latency (median / p95) | {latency['median_ms']:.3f} / {latency['p95_ms']:.3f} ms | < {t['latency_ms']:g} ms | {ok(latency['p95_ms'] < t['latency_ms'])} |",
        f"| Model size (.npz) | {size_mb:.3f} MB | < {t['model_size_mb']:g} MB | {ok(size_mb < t['model_size_mb'])} |", "",
        f"Validation accuracy {val['accuracy']:.4%}, loss {val['loss']:.4f}, ECE {val['ece']:.4f}."]
    if test is not None:
        lines += [f"Test loss {test['loss']:.4f}, macro F1 {test['macro_f1']:.4f}, ECE (15 bins) {test['ece']:.4f}.", "",
                  "## Per-digit test metrics", "", "| Digit | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for c, m in test["per_class"].items():
            lines.append(f"| {c} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
    if reject_val:
        lines += ["", "## Reject rule (max softmax probability)", "",
                  f"Threshold {reject_val['threshold']:.6f}, chosen on in-distribution validation data so that "
                  f"{cfg['reject']['target_in_distribution_accept']:.0%} of them are accepted. Held-out digit(s) "
                  f"{ds.info['excluded_digits']} are one proxy for invalid input; letters, symbols, scribbles and blank "
                  "canvases are measured separately by `tools/evaluate_ood.py`.", "",
                  "| Set | In-dist. accept | Accuracy on accepted | OOD reject | AUROC |", "|---|---|---|---|---|"]
        rows = [("Validation", reject_val)] + ([("Test", reject_test)] if reject_test else [])
        for name, r in rows:
            lines.append(f"| {name} | {r['in_distribution_accept_rate']:.4%} | {r['accuracy_on_accepted']:.4%} | "
                         f"{r['out_of_distribution_reject_rate']:.4%} | {r['auroc_in_vs_out']:.4f} |")
    lines += ["", "## Caveats", "",
              "- Accuracy is for MNIST digits, not arbitrary handwriting or canvas drawings (see `tools/evaluate_pipeline.py`).",
              "- Latency is one-sample normalise + forward + softmax on this machine; it varies run to run and excludes HTTP and canvas preprocessing.",
              "- Results can differ slightly across BLAS libraries and CPUs even with identical seeds.", ""]
    return "\n".join(lines)
