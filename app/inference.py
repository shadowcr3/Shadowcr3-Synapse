"""Inference engine: loads a trained model and turns 784 validated pixels into a prediction.

Independent of the training pipeline: depends only on `neural_network` and NumPy.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import base64

from neural_network import Network
from neural_network.introspection import layer_names, quantize_rows_int8, trace as _trace
from neural_network.activations import softmax

from .preprocessing import N_PIXELS, to_model_input


class ModelError(RuntimeError):
    pass


class Predictor:
    def __init__(self, network: Network, metadata: dict, version_id: str = "release") -> None:
        self.version_id = version_id
        if network.layer_sizes[0] != N_PIXELS:
            raise ModelError(f"model expects {network.layer_sizes[0]} inputs, not {N_PIXELS}")
        mapping = metadata.get("class_mapping")
        n_out = network.layer_sizes[-1]
        if not isinstance(mapping, dict) or sorted(mapping, key=int) != [str(i) for i in range(n_out)]:
            raise ModelError("metadata class_mapping does not match the model's outputs")
        self.network = network
        self.metadata = metadata
        self.digits = [int(mapping[str(i)]) for i in range(n_out)]
        reject = metadata.get("reject")
        self.threshold = None
        if reject:
            thr = reject.get("threshold")
            if not isinstance(thr, (int, float)) or not 0.0 < thr <= 1.0:
                raise ModelError("invalid reject threshold in metadata")
            self.threshold = float(thr)

    @classmethod
    def from_dir(cls, model_dir, name: str, version_id: str | None = None) -> "Predictor":
        try:
            network, metadata = Network.load(Path(model_dir), name)
        except (OSError, ValueError, KeyError) as exc:
            raise ModelError(f"cannot load model {name!r} from {model_dir}: {exc}") from exc
        return cls(network, metadata, version_id or name)

    def predict(self, pixels: np.ndarray) -> dict:
        """pixels: validated uint8 array (784,). Returns a JSON-serialisable dict."""
        t0 = time.perf_counter()
        probs = softmax(self.network.predict_logits(to_model_input(pixels)))[0]
        best = int(np.argmax(probs))
        confidence = float(probs[best])
        accepted = self.threshold is None or confidence >= self.threshold
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "ok" if accepted else "low_confidence",
            "digit": self.digits[best] if accepted else None,
            "top_guess": self.digits[best],
            "confidence": confidence,
            "threshold": self.threshold,
            "digits": self.digits,
            "scores": [float(p) for p in probs],
            "processing_ms": elapsed_ms,
        }

    # ----- metadata views exposed by the API (whitelisted; no file paths) -----------
    def info(self) -> dict:
        m = self.metadata
        ev = m.get("evaluation", {})
        return {
            "version_id": self.version_id, "source": m.get("source", "train.py"), "parent": m.get("parent"),
            "test_evaluated": bool(m.get("test_evaluated")),
            "name": m.get("model_name"), "version": m.get("model_version"),
            "run_id": m.get("run_id"), "created_utc": m.get("created_utc"),
            "architecture": m.get("architecture"), "parameter_count": m.get("parameter_count"),
            "digits": self.digits, "preprocessing": m.get("preprocessing"),
            "reject": m.get("reject"),
            "evaluation": {k: ev.get(k) for k in
                           ("validation_accuracy", "validation_loss", "test_accuracy", "test_macro_f1", "test_ece", "test_loss")},
        }

    def metrics(self) -> dict | None:
        return self.metadata.get("metrics_report")

    def trace(self, pixels: np.ndarray, label_digit: int | None = None, k_edges: int = 80) -> dict:
        """Prediction plus per-neuron activations, strongest edges and (given a label) the
        backward error signals. label_digit is a digit, not a class index."""
        target = None
        if label_digit is not None:
            if label_digit not in self.digits:
                raise ValueError(f"label must be one of {self.digits}")
            target = self.digits.index(label_digit)
        out = _trace(self.network, to_model_input(pixels)[0], target=target, k_edges=k_edges)
        out["digits"] = self.digits
        return out

    def weights(self, layer: int) -> dict:
        """Layer `layer` (0 = first hidden) as int8 + per-neuron scale, plus biases."""
        layers = self.network.layers
        if not 0 <= layer < len(layers):
            raise ValueError("no such layer")
        q, scale = quantize_rows_int8(layers[layer].W)
        names = layer_names(self.network)
        return {"layer": layer, "name": names[layer], "shape": list(q.shape),
                "scales": [float(x) for x in scale], "data": base64.b64encode(q.tobytes()).decode("ascii"),
                "bias": [round(float(b), 5) for b in layers[layer].b]}
