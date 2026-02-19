__all__ = [
    "available_model_names",
    "available_models",
    "get_model_description",
    "load",
    "load_vla",
]


def __getattr__(name):
    if name in __all__:
        from .models import available_model_names, available_models, get_model_description, load, load_vla

        globals().update(
            {
                "available_model_names": available_model_names,
                "available_models": available_models,
                "get_model_description": get_model_description,
                "load": load,
                "load_vla": load_vla,
            }
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
