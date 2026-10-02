from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from surpyval.univariate import competing_risks, regression

# Imported on first use, as the models they hold are from ``surpyval``
# (#470); ``surpyval.univariate.regression`` works without an import, as
# it did when ``import surpyval`` imported them.
_SUBPACKAGES = ("competing_risks", "regression")

if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        if name in _SUBPACKAGES:
            from importlib import import_module

            return import_module(f"{__name__}.{name}")
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
