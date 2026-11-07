
__all__ = ["create_app"]


def __getattr__(name):
    if name == "create_app":
        from .server import create_app
        return create_app
    raise AttributeError(f"module 'app' has no attribute {name!r}")
