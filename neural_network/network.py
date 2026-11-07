
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .activations import relu, softmax
from .layers import Dense, he_normal, xavier_normal

FORMAT_VERSION = 1


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Network:
    def __init__(self, layer_sizes: Sequence[int], seed=42, dtype=np.float64) -> None:
        sizes = [int(s) for s in layer_sizes]
        if len(sizes) < 2 or any(s < 1 for s in sizes):
            raise ValueError("layer_sizes needs at least two positive integers")
        self.layer_sizes = tuple(sizes)
        self.dtype = np.dtype(dtype)
        if self.dtype.kind != "f":
            raise ValueError("dtype must be a floating-point type")
        rng = np.random.default_rng(seed)
        n_layers = len(sizes) - 1
        self.layers = [
            Dense(fan_in, fan_out, he_normal if i < n_layers - 1 else xavier_normal, rng, self.dtype)
            for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:]))
        ]

    def parameters(self) -> list[np.ndarray]:
        return [a for layer in self.layers for a in (layer.W, layer.b)]

    def parameter_names(self) -> list[str]:
        return [f"{kind}{i}" for i in range(len(self.layers)) for kind in ("W", "b")]

    def num_parameters(self) -> int:
        return int(sum(p.size for p in self.parameters()))

    def get_state(self) -> list[np.ndarray]:
        return [p.copy() for p in self.parameters()]

    def set_state(self, state: Sequence[np.ndarray]) -> None:
        params = self.parameters()
        if len(state) != len(params):
            raise ValueError("state has the wrong number of tensors")
        for p, s in zip(params, state):
            if p.shape != s.shape:
                raise ValueError(f"state shape {s.shape} != parameter shape {p.shape}")
        for p, s in zip(params, state):
            p[...] = s

    def _prepare(self, X) -> np.ndarray:
        X = np.asarray(X)
        if X.dtype.kind not in "fiu":
            raise TypeError(f"input must be numeric, got dtype {X.dtype}")
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"expected input of shape (N, {self.layer_sizes[0]}), got {X.shape}")
        X = X.astype(self.dtype, copy=False)
        if not np.all(np.isfinite(X)):
            raise ValueError("input contains NaN or infinity")
        return X

    def forward(self, X):
        a = self._prepare(X)
        cache = []
        z = a
        for i, layer in enumerate(self.layers):
            z = layer.forward(a)
            cache.append((a, z))
            a = relu(z) if i < len(self.layers) - 1 else z
        return z, cache

    def predict_logits(self, X, batch_size: int = 4096) -> np.ndarray:
        X = self._prepare(X)
        out = np.empty((X.shape[0], self.layer_sizes[-1]), dtype=self.dtype)
        for start in range(0, X.shape[0], batch_size):
            out[start:start + batch_size] = self.forward(X[start:start + batch_size])[0]
        return out

    def predict_proba(self, X, batch_size: int = 4096) -> np.ndarray:
        return softmax(self.predict_logits(X, batch_size))

    def predict(self, X, batch_size: int = 4096) -> np.ndarray:
        return np.argmax(self.predict_logits(X, batch_size), axis=1)

    def save(self, directory, name: str, extra_metadata: dict | None = None) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        npz_path, json_path = directory / f"{name}.npz", directory / f"{name}.json"

        arrays = dict(zip(self.parameter_names(), self.parameters()))
        fd, tmp_npz = tempfile.mkstemp(dir=directory, suffix=".npz")
        os.close(fd)
        np.savez_compressed(tmp_npz, **arrays)
        os.chmod(tmp_npz, 0o644)
        os.replace(tmp_npz, npz_path)

        metadata = {
            "format_version": FORMAT_VERSION,
            "name": name,
            "architecture": {
                "layer_sizes": list(self.layer_sizes),
                "hidden_activation": "relu",
                "output_activation": "softmax",
                "dtype": self.dtype.name,
            },
            "parameter_count": self.num_parameters(),
            "npz_sha256": _sha256_file(npz_path),
            **(extra_metadata or {}),
        }
        fd, tmp_json = tempfile.mkstemp(dir=directory, suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        os.chmod(tmp_json, 0o644)
        os.replace(tmp_json, json_path)
        return npz_path, json_path

    @classmethod
    def load(cls, directory, name: str):
        directory = Path(directory)
        npz_path, json_path = directory / f"{name}.npz", directory / f"{name}.json"
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported model format {metadata.get('format_version')!r}")
        if _sha256_file(npz_path) != metadata.get("npz_sha256"):
            raise ValueError("model weights do not match the hash recorded in metadata")

        arch = metadata["architecture"]
        net = cls(arch["layer_sizes"], seed=0, dtype=np.dtype(arch["dtype"]))
        expected = net.parameter_names()
        with np.load(npz_path, allow_pickle=False) as data:
            if sorted(data.files) != sorted(expected):
                raise ValueError("npz tensors do not match the declared architecture")
            state = [data[k] for k in expected]
        for s in state:
            if s.dtype.kind != "f" or not np.all(np.isfinite(s)):
                raise ValueError("model contains non-float or non-finite values")
        net.set_state(state)
        return net, metadata
