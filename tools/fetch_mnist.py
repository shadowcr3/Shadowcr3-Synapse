#!/usr/bin/env python3
"""Download MNIST into data/mnist and verify every file against pinned SHA-256 hashes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.dataset import load_mnist  # noqa: E402

target = sys.argv[1] if len(sys.argv) > 1 else "data/mnist"
raw = load_mnist(target, download=True)
print(f"MNIST ready in {target}: {len(raw['y_train'])} train / {len(raw['y_test'])} test images, hashes verified.")
