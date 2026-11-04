# Drawing-pipeline evaluation

Model run `synapse_v1_seed42_20261104T042945Z` (npz sha256 `1bd8fa92fe9976c1…`), 1200 MNIST test digits (1-9), seed 0. Reject threshold 0.9532.

Digits are re-rendered as 280x280 canvas drawings (random size 100-240 px, random position) with extra stroke thickening, then normalised like the browser does. **This tests geometry and stroke-width invariance, not real handwriting.**

| Input | Accuracy | Accepted | Accuracy when accepted |
|---|---|---|---|
| Native MNIST 28x28 (no pipeline) | 98.92% | 94.6% | 99.91% |
| Canvas, +0 px stroke | 98.92% | 93.7% | 99.91% |
| Canvas, +4 px stroke | 98.92% | 94.0% | 99.91% |
| Canvas, +8 px stroke | 98.42% | 93.1% | 99.73% |
| Canvas, +12 px stroke | 96.83% | 89.2% | 99.44% |
| Canvas, +16 px stroke | 94.58% | 83.7% | 98.80% |

JavaScript vs Python pipeline on 150 canvases: max pixel difference 0, prediction agreement 100.0% (JS accuracy 99.33%, Python 99.33%).

HTTP /predict vs direct model call on 60 drawings: max score difference 0.00e+00.
