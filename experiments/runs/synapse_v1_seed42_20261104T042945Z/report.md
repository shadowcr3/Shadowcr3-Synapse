# Evaluation report - synapse_v1_seed42_20261104T042945Z

All values below were measured by `train.py`; none are estimated.

## Setup

- Classes [1, 2, 3, 4, 5, 6, 7, 8, 9]; architecture [784, 128, 64, 9] (109,321 parameters, float32)
- Split: 48,670 train / 5,407 validation / 9,020 test (official MNIST test set, filtered)
- Optimizer adam lr 0.001 (cosine), weight decay 0.0001, batch 64, epochs 30/30, seed 42, augmentation on
- Checkpoint: epoch 25 by val_loss

## Results

| Metric | Measured | Target | Status |
|---|---|---|---|
| Test accuracy | 98.7916% | >= 97% | met |
| Single-sample latency (median / p95) | 0.030 / 0.045 ms | < 100 ms | met |
| Model size (.npz) | 0.408 MB | < 5 MB | met |

Validation accuracy 98.9273%, loss 0.0313, ECE 0.0031.
Test loss 0.0386, macro F1 0.9878, ECE (15 bins) 0.0019.

## Per-digit test metrics

| Digit | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 1 | 0.9930 | 0.9956 | 0.9943 | 1135 |
| 2 | 0.9903 | 0.9903 | 0.9903 | 1032 |
| 3 | 0.9871 | 0.9881 | 0.9876 | 1010 |
| 4 | 0.9839 | 0.9949 | 0.9894 | 982 |
| 5 | 0.9855 | 0.9899 | 0.9877 | 892 |
| 6 | 0.9927 | 0.9875 | 0.9901 | 958 |
| 7 | 0.9817 | 0.9903 | 0.9860 | 1028 |
| 8 | 0.9886 | 0.9815 | 0.9851 | 974 |
| 9 | 0.9879 | 0.9722 | 0.9800 | 1009 |

## Reject rule (max softmax probability)

Threshold 0.953218, chosen on in-distribution validation data so that 95% of them are accepted. Held-out digit(s) [0] are one proxy for invalid input; letters, symbols, scribbles and blank canvases are measured separately by `tools/evaluate_ood.py`.

| Set | In-dist. accept | Accuracy on accepted | OOD reject | AUROC |
|---|---|---|---|---|
| Validation | 95.0065% | 99.8832% | 83.4459% | 0.9716 |
| Test | 94.6120% | 99.7774% | 85.4082% | 0.9721 |

## Caveats

- Accuracy is for MNIST digits, not arbitrary handwriting or canvas drawings (see `tools/evaluate_pipeline.py`).
- Latency is one-sample normalise + forward + softmax on this machine; it varies run to run and excludes HTTP and canvas preprocessing.
- Results can differ slightly across BLAS libraries and CPUs even with identical seeds.
