import gzip
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training import dataset
from training.dataset import normalize_pixels, prepare_data, read_idx, stratified_split

REPO = Path(__file__).resolve().parent.parent
REAL_DATA = REPO / "data" / "mnist"


def write_idx(path, array, magic_dtype=0x08, magic_prefix=b"\x00\x00"):
    header = magic_prefix + bytes([magic_dtype, array.ndim]) + struct.pack(">" + "I" * array.ndim, *array.shape)
    with gzip.open(path, "wb") as f:
        f.write(header + array.astype(np.uint8).tobytes())


def fake_raw(n_train=1000, n_test=200, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "x_train": rng.integers(0, 256, (n_train, 28, 28), dtype=np.uint8),
        "y_train": (np.arange(n_train) % 10).astype(np.uint8),
        "x_test": rng.integers(0, 256, (n_test, 28, 28), dtype=np.uint8),
        "y_test": (np.arange(n_test) % 10).astype(np.uint8),
    }


CFG = {"classes": list(range(1, 10)), "val_fraction": 0.1, "data_dir": "unused", "download": False}


class TestIdx(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_roundtrip(self):
        arr = np.random.default_rng(0).integers(0, 256, (5, 28, 28)).astype(np.uint8)
        write_idx(self.dir / "a.gz", arr)
        np.testing.assert_array_equal(read_idx(self.dir / "a.gz"), arr)

    def test_rejects_bad_magic_dtype_rank_and_sizes(self):
        arr = np.zeros((3, 4), dtype=np.uint8)
        write_idx(self.dir / "bad_magic.gz", arr, magic_prefix=b"\x01\x00")
        write_idx(self.dir / "bad_dtype.gz", arr, magic_dtype=0x0D)
        for name in ("bad_magic.gz", "bad_dtype.gz"):
            with self.assertRaises(ValueError, msg=name):
                read_idx(self.dir / name)
        with gzip.open(self.dir / "truncated.gz", "wb") as f:     # header claims 12 bytes, has 4
            f.write(b"\x00\x00\x08\x02" + struct.pack(">II", 3, 4) + b"\x01\x02\x03\x04")
        with self.assertRaises(ValueError):
            read_idx(self.dir / "truncated.gz")
        with gzip.open(self.dir / "rank0.gz", "wb") as f:
            f.write(b"\x00\x00\x08\x00")
        with self.assertRaises(ValueError):
            read_idx(self.dir / "rank0.gz")

    def test_load_mnist_rejects_hash_mismatch(self):
        for filename, _ in dataset.MNIST_FILES.values():
            write_idx(self.dir / filename, np.zeros((2, 2), dtype=np.uint8))
        with self.assertRaises(ValueError):
            dataset.load_mnist(self.dir, download=False)

    def test_missing_file_without_download(self):
        with self.assertRaises(FileNotFoundError):
            dataset.load_mnist(self.dir, download=False)


class TestPreparation(unittest.TestCase):
    def test_only_configured_labels_remain_and_zero_is_excluded(self):
        ds = prepare_data(CFG, split_seed=1, raw=fake_raw())
        for y in (ds.y_train, ds.y_val, ds.y_test):
            self.assertGreaterEqual(y.min(), 0)
            self.assertLessEqual(y.max(), 8)
        raw = fake_raw()
        n_zero_train = int((raw["y_train"] == 0).sum())
        self.assertEqual(ds.info["n_train"] + ds.info["n_val"], len(raw["y_train"]) - n_zero_train)
        self.assertEqual(ds.info["n_test"], int((raw["y_test"] != 0).sum()))
        self.assertEqual(ds.info["n_ood_test"], int((raw["y_test"] == 0).sum()))

    def test_label_index_mapping_is_digit_minus_one(self):
        raw = fake_raw()
        ds = prepare_data(CFG, split_seed=1, raw=raw)
        np.testing.assert_array_equal(np.bincount(ds.y_test), np.bincount(raw["y_test"][raw["y_test"] != 0] - 1))

    def test_feature_dimensions(self):
        ds = prepare_data(CFG, split_seed=1, raw=fake_raw())
        for X in (ds.X_train, ds.X_val, ds.X_test, ds.X_ood_val, ds.X_ood_test):
            self.assertEqual(X.shape[1], 784)
            self.assertEqual(X.dtype, np.uint8)

    def test_normalisation_range_and_exactness(self):
        x = np.array([[0, 1, 128, 255]], dtype=np.uint8)
        out = normalize_pixels(x)
        self.assertEqual(out.dtype, np.float64)
        np.testing.assert_array_equal(out, [[0.0, 1 / 255, 128 / 255, 1.0]])

    def test_split_is_deterministic_and_seed_dependent(self):
        a = prepare_data(CFG, split_seed=5, raw=fake_raw())
        b = prepare_data(CFG, split_seed=5, raw=fake_raw())
        c = prepare_data(CFG, split_seed=6, raw=fake_raw())
        self.assertEqual(a.info["split_sha256"], b.info["split_sha256"])
        self.assertNotEqual(a.info["split_sha256"], c.info["split_sha256"])

    def test_stratified_split_properties(self):
        labels = np.repeat(np.arange(9), [100, 120, 80, 95, 110, 105, 90, 99, 101])
        tr, va = stratified_split(labels, 0.1, np.random.default_rng(0))
        self.assertEqual(len(np.intersect1d(tr, va)), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([tr, va])), np.arange(len(labels)))
        for c in range(9):
            self.assertAlmostEqual((labels[va] == c).sum(), 0.1 * (labels == c).sum(), delta=0.5)

    def test_invalid_val_fraction(self):
        with self.assertRaises(ValueError):
            stratified_split(np.zeros(10, dtype=int), 0.0, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            stratified_split(np.zeros(10, dtype=int), 0.5, np.random.default_rng(0))


@unittest.skipUnless((REAL_DATA / dataset.MNIST_FILES["train_images"][0]).exists(), "MNIST files not present")
class TestRealMnist(unittest.TestCase):
    def test_real_mnist_filtering_and_counts(self):
        raw = dataset.load_mnist(REAL_DATA, download=False)
        self.assertEqual(raw["x_train"].shape, (60000, 28, 28))
        self.assertEqual(raw["x_test"].shape, (10000, 28, 28))
        ds = prepare_data({**CFG, "data_dir": str(REAL_DATA)}, split_seed=np.random.SeedSequence(42), raw=raw)
        self.assertEqual(ds.info["n_train"] + ds.info["n_val"], 60000 - int((raw["y_train"] == 0).sum()))
        self.assertEqual(ds.info["n_test"], 10000 - 980)
        self.assertEqual(ds.info["n_ood_test"], 980)
        self.assertEqual(ds.X_train.shape[1], 784)
        self.assertEqual(set(np.unique(ds.y_train)), set(range(9)))


if __name__ == "__main__":
    unittest.main()
