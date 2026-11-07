# Security notes

Synapse is a **local tool** with no authentication. The controls below reduce risk from other things on the same machine or in the same browser; they do not make it safe to expose to the internet.

| Threat | Control |
|---|---|
| Other websites calling the local API (CSRF) | state changes need `application/json` (no CORS, so cross-site requests cannot send it) and an `Origin` check that must equal `Host` |
| DNS rebinding | when bound to loopback, only loopback `Host` headers are accepted |
| XSS in the UI | strict CSP (`default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'`), no inline script/style, all text set with `textContent` (tested: no `innerHTML`/`eval`) |
| Path traversal | static files from a fixed whitelist; model ids validated by regex; no user path ever reaches the filesystem |
| Oversized / malformed input | 16 KB body limit, strict schema, integer-only pixels, label/parameter bounds |
| Resource exhaustion | per-client rate limits (below), one job at a time, parameter/epoch caps, wall-clock limit, sample limit (20,000), event ring buffer |
| Poisoned fine-tuning data | labels validated, local-only store (0600), validation gate (global accuracy plus a stratified held-out set of your own drawings, read from the candidate's metadata so no route skips it), rollback |
| Model tampering | SHA-256 recorded in metadata and checked at load; `allow_pickle=False` |
| Information leaks | errors are generic JSON; `/model/info` is whitelisted and contains no paths |
| Running untrusted code | none: no `pickle`, `eval`, shell-outs or dynamic imports on request paths |

## Rate limiting

Token buckets per client address, enabled by default, in memory (at most 10,000 clients tracked, least recently seen evicted). A refused request gets `429`, a `Retry-After` header and the usual JSON error.

| Bucket | Applies to | Sustained | Burst |
|---|---|---|---|
| infer | `/predict`, `/trace` | 15 / s | 40 |
| read | other GETs, `/jobs/status` | 10 / s | 40 |
| write | feedback, promote, rollback, delete, `/jobs/cancel` | 2 / s | 10 |
| job | starting a training / fine-tune / sweep job | 6 / min | 3 |
| static | `/` and `/static/*` | 30 / s | 120 |

The drawing UI (a read every 110 ms while drawing, a status poll every 400 ms during a job) stays inside these limits; a test checks it. The client key is the socket address. `X-Forwarded-For` is ignored unless you start with `--trust-proxy N` behind N proxies you control; otherwise anyone could dodge the limit by sending that header. `--no-rate-limit` turns it off.

## If you expose it

Keep it local unless you put an authenticating, TLS-terminating reverse proxy in front. Start with `--read-only` (disables training, drawings and model changes) and `--trust-proxy 1` so the rate limiter sees real client addresses. Do not use `--host 0.0.0.0` on an untrusted network.

## Known gaps

No authentication and no TLS (both are left to a reverse proxy; this server is meant to stay on localhost); the limiter is per process and in memory, so it resets on restart and is not shared between workers; the Docker image has not been built or scanned in this environment; dependency versions are ranges, not hashes (use `pip install --require-hashes` with your own lock file for stricter supply-chain control).
