__all__ = [
    "get_vla_dataset_and_collator",
]


def __getattr__(name):
    if name in __all__:
        from .materialize import get_vla_dataset_and_collator

        globals().update(
            {
                "get_vla_dataset_and_collator": get_vla_dataset_and_collator,
            }
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
