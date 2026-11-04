# Security notes

Synapse is a **local tool** with no authentication. The controls below reduce risk from other things on the same machine or in the same browser; they do not make it safe to expose to the internet.

| Threat | Control |
|---|---|
| Other websites calling the local API (CSRF) | state changes need `application/json` (no CORS, so cross-site requests cannot send it) and an `Origin` check that must equal `Host` |
| DNS rebinding | when bound to loopback, only loopback `Host` headers are accepted |
| XSS in the UI | strict CSP (`default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'`), no inline script/style, all text set with `textContent` (tested: no `innerHTML`/`eval`) |
| Path traversal | static files from a fixed whitelist; model ids validated by regex; no user path ever reaches the filesystem |
| Oversized / malformed input | 16 KB body limit, strict schema, integer-only pixels, label/parameter bounds |
| Resource exhaustion | one job at a time, parameter/epoch caps, wall-clock limit, sample limit (20,000), event ring buffer |
| Poisoned fine-tuning data | labels validated, local-only store (0600), validation gate (global accuracy + held-out own drawings), rollback |
| Model tampering | SHA-256 recorded in metadata and checked at load; `allow_pickle=False` |
| Information leaks | errors are generic JSON; `/model/info` is whitelisted and contains no paths |
| Running untrusted code | none: no `pickle`, `eval`, shell-outs or dynamic imports on request paths |

## If you expose it

Bind to loopback and put an authenticating reverse proxy in front, and start with `--read-only` (disables training, drawings and model changes). Do not use `--host 0.0.0.0` on an untrusted network. Add rate limiting at the proxy.

## Known gaps

No authentication or TLS; no rate limiting; the Docker image has not been built or scanned in this environment; dependency versions are ranges, not hashes (use `pip install --require-hashes` with your own lock file for stricter supply-chain control).
