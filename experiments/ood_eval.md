# Out-of-distribution evaluation

Model run `synapse_v1_seed42_20261104T042945Z` (npz sha256 `1bd8fa92fe9976c1…`), reject threshold 0.9532. 300 drawings per category unless noted; every drawing is generated as a 280x280 canvas and normalised exactly as the browser does. **Reject** = stopped before the network (blank / too small) or below the confidence threshold. **False accept** = the model confidently names a digit.

| Category | n | Blocked early | Rejected by model | **False accepts** | Reject rate | AUROC vs valid |
|---|---|---|---|---|---|---|
| valid digits 1-9 (reference) | 300 | 0 | 21 | 279 | 7.0% | – |
| digit 0 (MNIST) | 300 | 0 | 264 | **36** | 88.0% | 0.966 |
| blank canvas | 20 | 20 | 0 | **0** | 100.0% | – |
| tiny mark / dot | 300 | 300 | 0 | **0** | 100.0% | – |
| random scribbles | 300 | 0 | 215 | **85** | 71.7% | 0.920 |
| geometric shapes | 300 | 0 | 224 | **76** | 74.7% | 0.939 |
| speckle noise | 300 | 0 | 300 | **0** | 100.0% | 0.994 |
| letters A-Z a-z (6 fonts) | 312 | 0 | 147 | **165** | 47.1% | 0.833 |
| symbols (+ - = # @ % & ? ! ...) | 162 | 0 | 84 | **78** | 51.9% | 0.820 |

Reference: 93.0% of valid drawings are accepted (the rest are rejected as low confidence).

## Characters most often accepted as a digit

- letters: `C`→6 (6/6), `S`→5 (6/6), `j`→1 (6/6), `B`→8 (5/6), `G`→6 (5/6), `H`→4 (5/6), `I`→1 (5/6), `J`→1 (5/6), `R`→8 (5/6), `V`→4 (5/6), `Y`→4 (5/6), `Z`→2 (5/6)
- symbols: `/`→1 (6/6), `\`→1 (6/6), `(`→1 (6/6), `)`→1 (6/6), `[`→1 (6/6), `]`→1 (6/6), `{`→1 (6/6), `|`→1 (6/6), `?`→7 (5/6), `!`→1 (5/6), `;`→2 (4/6), `:`→2 (3/6)

## Reading this

- A closed-set classifier has no 'none of the above' output; the reject threshold is a confidence heuristic, not a detector for non-digits.
- Characters that really look like a digit (`I`, `l`, `|` ~ 1; `O` ~ 0; `Z` ~ 2; `S` ~ 5; `B` ~ 8) are expected to be accepted and cannot be separated by any digit model.
- Fonts here are printed DejaVu faces, not handwriting; scribbles, shapes and noise are procedural. These are probes, not a complete definition of 'not a digit'.
