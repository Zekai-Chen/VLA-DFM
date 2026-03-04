__all__ = [
    "get_train_strategy",
    "Metrics",
    "VLAMetrics",
]


def __getattr__(name):
    if name == "get_train_strategy":
        from .materialize import get_train_strategy

        globals()["get_train_strategy"] = get_train_strategy
        return get_train_strategy
    if name in ("Metrics", "VLAMetrics"):
        from .metrics import Metrics, VLAMetrics

        globals().update({"Metrics": Metrics, "VLAMetrics": VLAMetrics})
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
