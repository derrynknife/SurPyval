"""Argument checks shared by the test-planning functions, and the wrapper
that makes a function of numbers answer for arrays element by element."""

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from surpyval.utils.validation import check_alpha_ci

_Function = TypeVar("_Function", bound=Callable[..., Any])


def elementwise(function: _Function) -> _Function:
    """``function``, of numbers, answering for each element of the
    arguments given as arrays, broadcast together: an array of its
    answers, or its answer when every argument is a number.

    Its ``alpha_ci``, if it takes one, is checked here, where the user
    called it, so that a level above 0.5 is warned of once."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if "alpha_ci" in bound.arguments:
            for level in np.unique(np.asarray(bound.arguments["alpha_ci"])):
                check_alpha_ci(level.item())
        arrays = {
            name: np.asarray(value)
            for name, value in bound.arguments.items()
            if value is not None and np.ndim(value) > 0
        }
        if not arrays:
            return function(*bound.args, **bound.kwargs)
        shaped = dict(zip(arrays, np.broadcast_arrays(*arrays.values())))
        shape = next(iter(shaped.values())).shape
        answers = [
            function(
                **{
                    **bound.arguments,
                    **{name: a[index].item() for name, a in shaped.items()},
                }
            )
            for index in np.ndindex(shape)
        ]
        return np.array(answers).reshape(shape)

    return wrapper  # type: ignore[return-value]


def is_whole(value: Any) -> bool:
    """Whether ``value`` is a whole number: ``3`` or ``3.0``, but not
    ``True``."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return isinstance(value, (float, np.floating)) and bool(
        float(value).is_integer()
    )


def check_probability(value: Any, name: str) -> float:
    """``value`` as a float, if it is strictly between 0 and 1."""
    try:
        inside = bool(0.0 < value < 1.0)
    except (TypeError, ValueError):
        inside = False
    if not inside:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}.")
    return float(value)


def check_positive(value: Any, name: str) -> float:
    """``value`` as a float, if it is finite and positive."""
    try:
        positive = bool(np.isfinite(value) and value > 0.0)
    except (TypeError, ValueError):
        positive = False
    if not positive:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return float(value)


def check_count(value: Any, name: str = "n", minimum: int = 1) -> int:
    """``value`` as an int, if it is a whole number of at least
    ``minimum``."""
    if not is_whole(value) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(
            f"{name} must be a whole number, {qualifier}, got {value!r}."
        )
    return int(value)


def check_failures(failures: Any) -> int:
    """``failures`` as an int, if it is a whole number, 0 or more."""
    if not is_whole(failures) or failures < 0:
        raise ValueError(
            f"failures must be a whole number, 0 or more, got {failures!r}."
        )
    return int(failures)


def check_flag(value: Any, name: str) -> bool:
    """``value``, if it is True or False."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be True or False, got {value!r}.")
    return bool(value)
