
from __future__ import annotations

import copy
import json
from pathlib import Path

_AUG = {"enabled": bool, "rotation_deg": float, "scale": float, "shear": float,
        "translate_px": float, "thicken_prob": float}
_ADAM = {"beta1": float, "beta2": float, "epsilon": float}
_SCHEMA = {
    "model": {"name": str, "version": str, "layer_sizes": list, "dtype": str},
    "data": {"classes": list, "val_fraction": float, "split_seed": int, "data_dir": str, "download": bool},
    "training": {"epochs": int, "batch_size": int, "learning_rate": float, "optimizer": str,
                 "adam": dict, "weight_decay": float, "lr_schedule": str, "lr_min_factor": float,
                 "selection_metric": str, "early_stopping_patience": int, "seed": int,
                 "augmentation": dict},
    "reject": {"enabled": bool, "score": str, "target_in_distribution_accept": float},
    "targets": {"test_accuracy": float, "latency_ms": float, "model_size_mb": float},
    "output": {"models_dir": str, "runs_dir": str},
}


def _is(v, typ) -> bool:
    if typ is float:
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if typ is int:
        return isinstance(v, int) and not isinstance(v, bool)
    return isinstance(v, typ)


def _check_types(section: str, values: dict, schema: dict) -> None:
    unknown, missing = set(values) - set(schema), set(schema) - set(values)
    if unknown or missing:
        raise ValueError(f"config[{section!r}]: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    for key, typ in schema.items():
        if not _is(values[key], typ):
            raise ValueError(f"config[{section!r}][{key!r}] must be {typ.__name__}, got {type(values[key]).__name__}")


def validate_config(cfg: dict) -> dict:
    if set(cfg) != set(_SCHEMA):
        raise ValueError(f"config sections must be exactly {sorted(_SCHEMA)}")
    for section, schema in _SCHEMA.items():
        _check_types(section, cfg[section], schema)
    _check_types("training.adam", cfg["training"]["adam"], _ADAM)
    _check_types("training.augmentation", cfg["training"]["augmentation"], _AUG)

    m, d, t, r = cfg["model"], cfg["data"], cfg["training"], cfg["reject"]
    classes, sizes = d["classes"], m["layer_sizes"]
    if not classes or any(not _is(c, int) or not 0 <= c <= 9 for c in classes) or len(set(classes)) != len(classes):
        raise ValueError("data.classes must be unique integers in 0..9")
    if len(sizes) < 2 or any(not _is(s, int) or s < 1 for s in sizes):
        raise ValueError("model.layer_sizes must be positive integers")
    if sizes[0] != 784:
        raise ValueError("model.layer_sizes[0] must be 784 (28x28 pixels)")
    if sizes[-1] != len(classes):
        raise ValueError("model.layer_sizes[-1] must equal the number of classes")
    if m["dtype"] not in ("float32", "float64"):
        raise ValueError("model.dtype must be 'float32' or 'float64'")
    if not 0.0 < d["val_fraction"] < 0.5:
        raise ValueError("data.val_fraction must be in (0, 0.5)")
    if t["epochs"] < 1 or t["batch_size"] < 1 or t["learning_rate"] <= 0:
        raise ValueError("epochs and batch_size must be >= 1 and learning_rate > 0")
    if t["optimizer"] not in ("adam", "sgd"):
        raise ValueError("training.optimizer must be 'adam' or 'sgd'")
    if t["weight_decay"] < 0 or (t["weight_decay"] > 0 and t["optimizer"] != "adam"):
        raise ValueError("weight_decay must be >= 0 and requires the adam optimizer")
    if t["lr_schedule"] not in ("constant", "cosine"):
        raise ValueError("training.lr_schedule must be 'constant' or 'cosine'")
    if not 0.0 <= t["lr_min_factor"] <= 1.0:
        raise ValueError("training.lr_min_factor must be in [0, 1]")
    if t["selection_metric"] not in ("val_loss", "val_acc"):
        raise ValueError("training.selection_metric must be 'val_loss' or 'val_acc'")
    if t["early_stopping_patience"] < 0:
        raise ValueError("training.early_stopping_patience must be >= 0")
    a = t["augmentation"]
    if a["rotation_deg"] < 0 or not 0 <= a["scale"] < 1 or a["shear"] < 0 or a["translate_px"] < 0 \
            or not 0 <= a["thicken_prob"] <= 1:
        raise ValueError("training.augmentation values out of range")
    if r["score"] != "max_softmax":
        raise ValueError("reject.score must be 'max_softmax'")
    if not 0.0 < r["target_in_distribution_accept"] <= 1.0:
        raise ValueError("reject.target_in_distribution_accept must be in (0, 1]")
    if r["enabled"] and len(classes) == 10:
        raise ValueError("reject.enabled needs at least one excluded digit as out-of-distribution data")
    return cfg


def load_config(path) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return validate_config(json.load(f))


def apply_overrides(cfg: dict, assignments) -> dict:
    new = copy.deepcopy(cfg)
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        path, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = new
        keys = path.split(".")
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                raise ValueError(f"unknown config path {path!r}")
            node = node[k]
        if not isinstance(node, dict) or keys[-1] not in node:
            raise ValueError(f"unknown config path {path!r}")
        node[keys[-1]] = value
    return validate_config(new)
