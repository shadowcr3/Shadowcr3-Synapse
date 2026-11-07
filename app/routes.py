
from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from .feedback import FeedbackError, FeedbackFull
from .jobs import DataUnavailable, JobBusy, JobError
from .models import StoreError
from .preprocessing import InputError, parse_pixels
from .version import __version__

bp = Blueprint("synapse", __name__)


def _ext():
    return current_app.extensions["synapse"]


def error_response(status: int, code: str, message: str, **extra):
    resp = jsonify({"error": {"code": code, "message": message, **extra}})
    resp.status_code = status
    return resp


def _json_body():
    if not request.is_json:
        return None, error_response(415, "unsupported_media_type", "Content-Type must be application/json")
    payload = request.get_json(silent=True)
    if payload is None:
        return None, error_response(400, "invalid_json", "request body is not valid JSON")
    return payload, None


def _pixels_from(payload):
    try:
        return parse_pixels({"pixels": payload.get("pixels")} if isinstance(payload, dict) else payload), None
    except InputError as exc:
        return None, error_response(422 if exc.code == "blank_input" else 400, exc.code, str(exc))


@bp.get("/")
def index():
    return send_from_directory(_ext()["frontend_dir"], "index.html")


@bp.get("/static/<name>")
def static_file(name: str):
    if name not in _ext()["static_files"]:
        return error_response(404, "not_found", "no such resource")
    return send_from_directory(_ext()["frontend_dir"], name)


@bp.get("/health")
def health():
    import time
    ext = _ext()
    info = ext["store"].predictor.info()
    return jsonify({"status": "ok", "app_version": __version__, "model_loaded": True, "active_model": ext["store"].active_id,
                    "model": {"name": info["name"], "version": info["version"]},
                    "read_only": ext["read_only"], "training_busy": ext["jobs"].status()["busy"],
                    "uptime_s": round(time.monotonic() - ext["started"], 1)})


@bp.get("/model/info")
def model_info():
    return jsonify(_ext()["store"].predictor.info())


@bp.get("/model/metrics")
def model_metrics():
    report = _ext()["store"].predictor.metrics()
    if report is None:
        return error_response(404, "metrics_unavailable", "this model has no stored training metrics")
    return jsonify(report)


@bp.get("/model/ood")
def model_ood():
    import json
    pred = _ext()["store"].predictor
    path = _ext()["experiments_dir"] / "ood_eval.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = None
    if not isinstance(report, dict) or report.get("model_sha256") != pred.metadata.get("npz_sha256"):
        return error_response(404, "ood_unavailable", "no out-of-distribution evaluation exists for the active model")
    return jsonify(report)


@bp.get("/model/weights/<int:layer>")
def model_weights(layer: int):
    try:
        return jsonify(_ext()["store"].predictor.weights(layer))
    except ValueError as exc:
        return error_response(404, "not_found", str(exc))


@bp.get("/model/versions")
def model_versions():
    store = _ext()["store"]
    return jsonify({"active": store.active_id, "history": store.history(), "versions": store.list_versions()})


@bp.post("/model/promote")
def model_promote():
    payload, err = _json_body()
    if err:
        return err
    try:
        decision = _ext()["store"].promote(payload.get("id") if isinstance(payload, dict) else None)
    except StoreError as exc:
        return error_response(400, "invalid_model", str(exc))
    resp = jsonify(decision)
    resp.status_code = 200 if decision["passed"] else 409
    return resp


@bp.post("/model/rollback")
def model_rollback():
    try:
        return jsonify({"active": _ext()["store"].rollback()})
    except StoreError as exc:
        return error_response(409, "nothing_to_roll_back", str(exc))


@bp.delete("/model/versions/<cid>")
def model_delete(cid: str):
    try:
        _ext()["store"].delete(cid)
    except StoreError as exc:
        return error_response(409, "cannot_delete", str(exc))
    return jsonify({"deleted": cid})


@bp.post("/predict")
def predict():
    payload, err = _json_body()
    if err:
        return err
    if not isinstance(payload, dict):
        return error_response(400, "invalid_pixels", 'request body must be a JSON object: {"pixels": [...]}')
    pixels, err = _pixels_from({"pixels": payload.get("pixels")}) if set(payload) <= {"pixels"} else (None, error_response(
        400, "invalid_pixels", f"unexpected field(s): {sorted(set(payload) - {'pixels'})}"))
    if err:
        return err
    out = _ext()["store"].predictor.predict(pixels)
    out["model_id"] = _ext()["store"].active_id
    return jsonify(out)


@bp.post("/trace")
def trace():
    payload, err = _json_body()
    if err:
        return err
    if not isinstance(payload, dict) or not set(payload) <= {"pixels", "label", "edges"}:
        return error_response(400, "invalid_request", 'body must be {"pixels": [...], "label"?: 1-9, "edges"?: 0-200}')
    pixels, err = _pixels_from({"pixels": payload.get("pixels")})
    if err:
        return err
    label, k = payload.get("label"), payload.get("edges", 80)
    if label is not None and (isinstance(label, bool) or not isinstance(label, int)):
        return error_response(400, "invalid_label", "label must be an integer digit")
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= 200:
        return error_response(400, "invalid_request", "edges must be an integer in 0..200")
    predictor = _ext()["store"].predictor
    try:
        tr = predictor.trace(pixels, label, k)
    except ValueError as exc:
        return error_response(400, "invalid_label", str(exc))
    return jsonify({"prediction": predictor.predict(pixels), "trace": tr, "model_id": _ext()["store"].active_id})


@bp.post("/feedback")
def feedback_add():
    payload, err = _json_body()
    if err:
        return err
    if not isinstance(payload, dict) or set(payload) != {"pixels", "label"}:
        return error_response(400, "invalid_request", 'body must be {"pixels": [...], "label": 1-9}')
    pixels, err = _pixels_from({"pixels": payload["pixels"]})
    if err:
        return err
    label = payload["label"]
    if isinstance(label, bool) or not isinstance(label, int):
        return error_response(400, "invalid_label", "label must be an integer digit")
    try:
        res = _ext()["feedback"].add(pixels, label)
    except FeedbackFull as exc:
        return error_response(409, "feedback_full", str(exc))
    except FeedbackError as exc:
        return error_response(400, "invalid_label", str(exc))
    resp = jsonify(res)
    resp.status_code = 200 if res["duplicate"] else 201
    return resp


@bp.get("/feedback/stats")
def feedback_stats():
    return jsonify(_ext()["feedback"].stats())


@bp.delete("/feedback")
def feedback_clear():
    if request.args.get("confirm") != "yes":
        return error_response(400, "confirmation_required", "add ?confirm=yes to delete all saved drawings")
    return jsonify({"deleted": _ext()["feedback"].clear()})


@bp.post("/jobs/<kind>")
def job_start(kind: str):
    payload, err = _json_body()
    if err:
        return err
    try:
        job = _ext()["jobs"].start(kind, payload)
    except JobError as exc:
        return error_response(400, "invalid_job", str(exc))
    except JobBusy as exc:
        return error_response(409, "job_busy", str(exc))
    except DataUnavailable as exc:
        return error_response(503, "data_unavailable", str(exc))
    resp = jsonify(job)
    resp.status_code = 202
    return resp


@bp.get("/jobs/status")
def job_status():
    since = request.args.get("since", "0")
    if not since.isdigit():
        return error_response(400, "invalid_request", "since must be a non-negative integer")
    return jsonify(_ext()["jobs"].status(int(since)))


@bp.post("/jobs/cancel")
def job_cancel():
    return jsonify({"cancelling": _ext()["jobs"].cancel()})
