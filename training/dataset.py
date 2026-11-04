"""MNIST loading with integrity checks, class filtering, and stratified splitting.

Images are kept as uint8 (N, 784) in memory and converted to float64 in [0, 1] per batch
with normalize_pixels(); this keeps memory small and the conversion exact (k / 255).
"""

from __future__ import annotations

import gzip
import hashlib
import os
import struct
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# SHA-256 of the four canonical MNIST archives.
MNIST_FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"),
}
# Every download is verified against the hashes above, so the mirror need not be trusted.
MIRRORS = (
    "https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_IDX_BYTES = 100 * 1024 * 1024
IMAGE_SIDE = 28
N_FEATURES = IMAGE_SIDE * IMAGE_SIDE


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_idx(path: Path) -> np.ndarray:
    """Parse a gzip-compressed IDX file of unsigned bytes, validating the header."""
    with gzip.open(path, "rb") as f:
        raw = f.read(MAX_IDX_BYTES + 1)
    if len(raw) > MAX_IDX_BYTES:
        raise ValueError(f"{path}: decompressed size exceeds {MAX_IDX_BYTES} bytes")
    if len(raw) < 4 or raw[0] != 0 or raw[1] != 0:
        raise ValueError(f"{path}: not an IDX file (bad magic)")
    if raw[2] != 0x08:
        raise ValueError(f"{path}: unsupported IDX dtype 0x{raw[2]:02x} (expected unsigned byte)")
    ndim = raw[3]
    if not 1 <= ndim <= 3:
        raise ValueError(f"{path}: unsupported IDX rank {ndim}")
    header_end = 4 + 4 * ndim
    if len(raw) < header_end:
        raise ValueError(f"{path}: truncated IDX header")
    dims = struct.unpack(">" + "I" * ndim, raw[4:header_end])
    if len(raw) - header_end != int(np.prod(dims, dtype=np.uint64)):
        raise ValueError(f"{path}: payload size does not match header dimensions {dims}")
    return np.frombuffer(raw, dtype=np.uint8, offset=header_end).reshape(dims)


def _download(filename: str, expected_sha: str, dest: Path, mirrors=MIRRORS) -> None:
    errors = []
    for base in mirrors:
        if not base.startswith("https://"):
            errors.append(f"{base}: refusing non-HTTPS mirror")
            continue
        url = base + filename
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read(MAX_DOWNLOAD_BYTES + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {exc}")
            continue
        if len(data) > MAX_DOWNLOAD_BYTES:
            errors.append(f"{url}: response too large")
            continue
        if hashlib.sha256(data).hexdigest() != expected_sha:
            errors.append(f"{url}: SHA-256 mismatch")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
        return
    raise RuntimeError(f"could not download {filename}:\n  " + "\n  ".join(errors))


def load_mnist(data_dir, download: bool = True) -> dict:
    """Returns {'x_train','y_train','x_test','y_test','file_sha256'}; images are (N, 28, 28) uint8."""
    data_dir = Path(data_dir)
    hashes = {}
    for key, (filename, expected) in MNIST_FILES.items():
        path = data_dir / filename
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"{path} is missing and download is disabled")
            _download(filename, expected, path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{path}: SHA-256 mismatch (got {actual}). Delete the file and retry.")
        hashes[filename] = actual
    x_train = read_idx(data_dir / MNIST_FILES["train_images"][0])
    y_train = read_idx(data_dir / MNIST_FILES["train_labels"][0])
    x_test = read_idx(data_dir / MNIST_FILES["test_images"][0])
    y_test = read_idx(data_dir / MNIST_FILES["test_labels"][0])
    for x, y in ((x_train, y_train), (x_test, y_test)):
        if x.shape[1:] != (IMAGE_SIDE, IMAGE_SIDE) or x.shape[0] != y.shape[0]:
            raise ValueError("unexpected MNIST array shapes")
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
            "file_sha256": hashes}


def normalize_pixels(x_uint8: np.ndarray, dtype=np.float64) -> np.ndarray:
    """uint8 [0, 255] -> [0, 1] in `dtype` (exact k/255 in float64)."""
    dtype = np.dtype(dtype)
    return x_uint8.astype(dtype) / dtype.type(255)


def stratified_split(labels: np.ndarray, val_fraction: float, rng: np.random.Generator):
    """Per-class random split. Returns sorted (train_idx, val_idx); disjoint and exhaustive."""
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be in (0, 0.5)")
    train_parts, val_parts = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_ood_val: np.ndarray    # excluded digits (e.g. 0) from the training pool: reject-rule tuning only
    X_ood_test: np.ndarray   # excluded digits from the official test set: final reporting only
    classes: tuple
    info: dict = field(default_factory=dict)


def prepare_data(data_cfg: dict, split_seed, raw: dict | None = None) -> Dataset:
    """Pipeline: load -> keep configured digits -> stratified train/val split -> flatten
    -> labels to indices 0..K-1. The official test set is only filtered, never split or
    used for fitting. Excluded digits are kept aside as out-of-distribution (OOD) samples
    for evaluating the reject rule; they never enter training."""
    raw = raw if raw is not None else load_mnist(data_cfg["data_dir"], data_cfg.get("download", True))
    classes = tuple(int(c) for c in data_cfg["classes"])
    excluded = [d for d in range(10) if d not in classes]
    lookup = np.full(10, -1, dtype=np.int64)
    lookup[list(classes)] = np.arange(len(classes))

    y_tr_all, y_te_all = raw["y_train"].astype(np.int64), raw["y_test"].astype(np.int64)
    if y_tr_all.max() > 9 or y_te_all.max() > 9:
        raise ValueError("labels outside 0-9")
    keep_tr, keep_te = np.isin(y_tr_all, classes), np.isin(y_te_all, classes)
    x_tr, y_tr = raw["x_train"][keep_tr], y_tr_all[keep_tr]
    x_te, y_te = raw["x_test"][keep_te], y_te_all[keep_te]

    rng = np.random.default_rng(split_seed)   # fixed data.split_seed => identical val set across runs
    tr_idx, va_idx = stratified_split(y_tr, data_cfg["val_fraction"], rng)

    def flat(x):
        return np.ascontiguousarray(x.reshape(x.shape[0], N_FEATURES))

    # OOD pools
    ood_pool_mask = np.isin(y_tr_all, excluded)
    x_ood_pool = raw["x_train"][ood_pool_mask]
    n_ood_val = int(round(len(x_ood_pool) * data_cfg["val_fraction"]))
    ood_val_idx = np.sort(rng.permutation(len(x_ood_pool))[:n_ood_val])
    x_ood_test = raw["x_test"][np.isin(y_te_all, excluded)]

    split_hash = hashlib.sha256(
        tr_idx.astype("<i8").tobytes() + b"|" + va_idx.astype("<i8").tobytes()
        + b"|" + ood_val_idx.astype("<i8").tobytes()).hexdigest()
    ds = Dataset(
        X_train=flat(x_tr[tr_idx]), y_train=lookup[y_tr[tr_idx]],
        X_val=flat(x_tr[va_idx]), y_val=lookup[y_tr[va_idx]],
        X_test=flat(x_te), y_test=lookup[y_te],
        X_ood_val=flat(x_ood_pool[ood_val_idx]), X_ood_test=flat(x_ood_test),
        classes=classes,
    )
    ds.info = {
        "classes": list(classes), "excluded_digits": excluded,
        "val_fraction": data_cfg["val_fraction"],
        "n_train": len(ds.y_train), "n_val": len(ds.y_val), "n_test": len(ds.y_test),
        "n_ood_val": len(ds.X_ood_val), "n_ood_test": len(ds.X_ood_test),
        "split_sha256": split_hash, "file_sha256": raw.get("file_sha256", {}),
    }
    return ds
