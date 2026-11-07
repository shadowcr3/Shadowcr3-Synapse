# HTTP API

Base URL `http://127.0.0.1:5000`. All bodies are JSON. State-changing requests need `Content-Type: application/json`
(except bodyless ones), a same-origin `Origin` header if one is sent, and are rejected in `--read-only` mode (403).
Errors have one shape: `{"error": {"code": "...", "message": "..."}}`; stack traces are never returned.
Every endpoint is rate limited per client (see `docs/SECURITY.md`): `429` with `Retry-After` and code `rate_limited`.

| Method & path | Purpose | Notes |
|---|---|---|
| `GET /health` | liveness + active model | `app_version`, `active_model`, `read_only`, `training_busy`, `uptime_s` |
| `POST /predict` | classify | body `{"pixels": [784 integers 0-255]}` (28x28, row-major, ink bright on dark, already MNIST-normalised by the browser). 200 with `status` `ok`/`low_confidence`, `digit` (null if rejected), `top_guess`, `confidence`, `threshold`, `digits`, `scores`, `processing_ms`, `model_id`. 400 malformed, 422 blank/too faint, 413 > 16 KB, 415 not JSON |
| `POST /trace` | classify + per-neuron internals | body adds optional `label` (1-9, enables the backward pass) and `edges` (0-200). Returns `prediction`, and `trace`: per-layer `activation`, strongest `edges`, and with a label `backward.deltas` (dLoss/dz per neuron), `delta_norms`, `backward.edges` |
| `GET /model/info` | active model identity, architecture, headline metrics | whitelisted fields; no file paths |
| `GET /model/metrics` | stored training history, per-class metrics, confusion matrix (release model only), reject-rule stats | 404 if absent |
| `GET /model/ood` | measured behaviour on non-digit input | only served if the report is bound (by SHA-256) to the active model, else 404 |
| `GET /model/weights/<layer>` | incoming weights of layer 0.. as int8 + per-neuron scale + biases | for the visualisations |
| `GET /model/versions` | release + candidates, active id, rollback history | |
| `POST /model/promote` | `{"id"}` make a candidate active **if it passes the gate** | 200 passed / 409 with reasons. There is no force option. Fine-tuned candidates also need a meaningful, non-worse held-out check stored in their own metadata, and must descend from the active model |
| `POST /model/rollback` | revert to the previously active model | 409 if none |
| `DELETE /model/versions/<id>` | delete a candidate | not the active model or a rollback target |
| `POST /feedback` | `{"pixels", "label"}` save a labelled drawing | 201 new, 200 duplicate, 409 limit (20,000) |
| `GET /feedback/stats` / `DELETE /feedback?confirm=yes` | counts per digit plus `readiness` (`ready`, `reasons`, how many drawings would be held back) / delete all | |
| `POST /jobs/train` `{hidden_sizes, epochs, batch_size, learning_rate, weight_decay, augmentation, subset, seed}` | train a new candidate | 202; limits below |
| `POST /jobs/finetune` `{epochs, learning_rate, auto_promote}` | continue from the active model on saved drawings + MNIST replay | 400 unless `readiness.ready`: at least 30 drawings, a stratified hold-out of at least 8 drawings covering at least 4 digits |
| `POST /jobs/sweep` `{architectures, epochs, subset, batch_size, learning_rate, augmentation}` | compare hidden-layer sizes | nothing is saved |
| `GET /jobs/status?since=N` | job state + events after sequence N (poll every ~350 ms while busy) | |
| `POST /jobs/cancel` | stop the running job | |
| `GET /` , `GET /static/<file>` | the UI; fixed whitelist of `.js`/`.css` files | |

## Job limits

One job at a time (409 otherwise); epochs <= 60 (sweep <= 20); 1-3 hidden layers of 2-512 neurons; <= 1,500,000 parameters;
<= 8 architectures per sweep; 900 s wall-clock limit (`--max-job-seconds`); `503 data_unavailable` if MNIST is not on disk
(`python tools/fetch_mnist.py`). Jobs train a private copy and never touch the model being served.

## Event stream (`/jobs/status`)

`job_started`, `init` (layer sizes, initial weight tiles, probe digit trace), `batch` (loss, running accuracy, lr, per-tensor gradient norms),
`epoch` (history row, per-tensor weight change, weight tiles, probe trace with backward deltas), `sweep_arch_done`, `job_finished`
(`done` | `cancelled` | `failed`, result incl. candidate id and gate decision). Events are kept in a ring buffer of 6,000; `dropped: true` means you polled too slowly.
