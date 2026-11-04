"""Synapse application: preprocessing, inference engine, model store, jobs and the Flask API.

`create_app` is exposed lazily so that `python -m app.server` does not import the server module twice.
`inference.py` and `preprocessing.py` import only `neural_network` (never `training`).
"""

__all__ = ["create_app"]


def __getattr__(name):
    if name == "create_app":
        from .server import create_app
        return create_app
    raise AttributeError(f"module 'app' has no attribute {name!r}")
