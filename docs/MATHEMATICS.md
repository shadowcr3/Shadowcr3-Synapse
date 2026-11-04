# Synapse - mathematics

Every formula here maps to code in `neural_network/` and is covered by a test in `tests/`.

## Notation

Batch of N samples, K = 9 classes. Layer l has input `A_{l-1}` (N x n_{l-1}), weights `W_l` (n_l x n_{l-1}), bias `b_l` (n_l).
`A_0 = X` (pixels / 255). Labels are indices 0..8 (digit - 1).

## Forward pass (`network.py`, `layers.py`)

    Z_l = A_{l-1} W_l^T + b_l          (batched form of z = Wx + b)
    A_l = ReLU(Z_l)                    hidden layers l = 1, 2
    logits = Z_3                       output layer is linear; softmax is applied after
    P = softmax(logits)

ReLU(x) = max(0, x); its derivative is 1 for x > 0 and 0 otherwise (the subgradient 0 is used at x = 0).

## Softmax, log-softmax (`activations.py`)

    P_i = exp(z_i) / sum_j exp(z_j)

softmax(z) = softmax(z - c) for any constant c, so subtracting max(z) changes nothing mathematically but prevents overflow.
`log_softmax(z) = (z - m) - log(sum_j exp(z_j - m))`, m = max(z), avoids log(0) when a probability underflows.

## Cross-entropy (`loss.py`)

    L = -(1/N) sum_n sum_i y_{n,i} log P_{n,i}  =  -(1/N) sum_n log P_{n, t_n}

where t_n is the true class index.

## Gradient of softmax + cross-entropy

For one sample, with `L = -log P_t` and `P_i = e^{z_i} / S`:

    dL/dz_k = -d(z_t - log S)/dz_k = P_k - 1[k = t]

so for a batch `dL/dlogits = (P - Y) / N` (Y one-hot). The 1/N comes from the mean over the batch and is applied once, here;
the layer gradients below therefore need no further scaling. Each row of (P - Y) sums to 0 (tested).

## Backpropagation (`backpropagation.py`)

Starting from `dZ_3 = (P - Y)/N`, for l = 3, 2, 1:

    dW_l     = dZ_l^T  A_{l-1}                (n_l x n_{l-1})
    db_l     = sum over batch of dZ_l         (n_l)
    dA_{l-1} = dZ_l  W_l                      (N x n_{l-1})
    dZ_{l-1} = dA_{l-1} (elementwise) ReLU'(Z_{l-1})        (for l > 1)

## Gradient checking (`gradient_check`)

Central differences, `g_num = (L(theta + eps) - L(theta - eps)) / (2 eps)`, eps = 1e-6, float64.
Reported per tensor as `||g_ana - g_num|| / (||g_ana|| + ||g_num||)`. Test tolerance: 1e-6 (observed ~1e-10 on small networks,
~1e-8 on the full architecture with sampled entries).

Caveat: ReLU is not differentiable at 0. A hidden pre-activation within ~eps of 0 makes finite differences straddle the kink
and disagree with the subgradient even when backprop is correct. This happens with zero-initialised biases when every unit of
the previous layer is dead for a sample (Z = b = 0 exactly). `gradient_check` emits a `RuntimeWarning` in that case and the
tests use non-zero biases.

## Initialisation (`layers.py`)

- Hidden layers: He normal, `W ~ N(0, 2 / fan_in)` - preserves activation variance through ReLU.
- Output layer: Xavier/Glorot normal, `W ~ N(0, 2 / (fan_in + fan_out))` - small initial logits, near-uniform starting softmax.
- Biases: 0.

## Parameter count

    (784*128 + 128) + (128*64 + 64) + (64*9 + 9) = 100,480 + 8,256 + 585 = 109,321

(The original spec states 109,385; the arithmetic gives 109,321. `tests/test_network.py` asserts 109,321.)

## Adam / AdamW (`optimizers.py`)

    m <- b1 m + (1 - b1) g            v <- b2 v + (1 - b2) g^2
    m_hat = m / (1 - b1^t)            v_hat = v / (1 - b2^t)
    theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)

Implemented in the paper's efficient form, algebraically identical: `lr_t = lr sqrt(1-b2^t)/(1-b1^t)`, `eps_hat = eps sqrt(1-b2^t)`,
`theta <- theta - lr_t m / (sqrt(v) + eps_hat)`. Decoupled weight decay (AdamW): `W <- W (1 - lr * wd)` before the step, on weight matrices only.
Defaults: lr 1e-3, b1 0.9, b2 0.999, eps 1e-8. On step 1, `m_hat = g` and `v_hat = g^2`, so every parameter moves by ~lr (tested).

## Reproducibility (`train.py`)

One master seed -> `SeedSequence(seed).spawn(3)` -> independent streams for (1) weight init, (2) mini-batch shuffling,
(3) train/validation split. Changing one concern (e.g. the split) does not perturb the others. The split is recorded as a SHA-256 of
the index arrays.

## Model selection

After each epoch the validation loss is computed; the parameters from the epoch with the lowest validation loss are restored at the end.
The test set is not accessible to `trainer.train`.

The release model uses float32; float64 is used for gradient checks (finite differences need it).

## Evaluation metrics (`training/metrics.py`)

- Precision_c = TP_c / (TP_c + FP_c), Recall_c = TP_c / (TP_c + FN_c), F1_c = harmonic mean; macro F1 = unweighted mean over classes.
- ECE (15 equal-width bins on max-probability): `sum_b (n_b / N) |acc_b - conf_b|`. Lower is better calibrated.
- AUROC via the rank-sum identity: `P(score_in > score_out) + 0.5 P(tie)`.

## Reject rule

Score s(x) = max_i P_i(x). Accept if `s >= tau`, else report "not a confident digit".
`tau` is the largest value such that >= 95% (configurable) of **in-distribution validation** digits (1-9) are accepted; no out-of-distribution data
chooses it. Digit 0 (excluded from training) is used only to *measure* how many invalid inputs are rejected: validation 0s while tuning the target,
test 0s for the final report.

Limits: max-softmax is a weak out-of-distribution detector, and 0 is just one kind of invalid input. Scribbles, blank canvases and
non-digit shapes are separate cases (blank input is rejected before the network in M5/M6).

## Learning-rate schedule (`trainer.py`)

Cosine: `lr(s) = lr_min + 0.5 (lr_0 - lr_min)(1 + cos(pi s / (S - 1)))` over S total steps, `lr_min = lr_min_factor * lr_0`.

## Augmentation (`training/augmentation.py`)

Per sample: `A = s R(theta) Shear(k)`, sampled `theta ~ U(+-12 deg)`, `s ~ U(0.9, 1.1)`, `k ~ U(+-0.1)`, shift `~ U(+-2 px)`. Each output pixel `p` is read from the source at
`A^-1 (p - c - t) + c` (c = 13.5) with bilinear interpolation and a zero (black) border, so no information is invented and the digit is never cropped by a rolled edge.
With probability 0.2 strokes are thickened by a 4-neighbour maximum.

## Introspection (`neural_network/introspection.py`)

For one sample with pre-activations `z_l` and post-activations `a_l`: `delta_L = P - Y`, `delta_l = (delta_{l+1} W_{l+1}) (elementwise) ReLU'(z_l)`.
These equal the bias gradients of backpropagation with N = 1 (tested), so the backward view shows real gradients.
Edge values: forward `W[j,i] a_i` (how strongly neuron i drives j); backward `W[j,i] delta_j` (error sent from j to i). The API returns the largest by magnitude.
