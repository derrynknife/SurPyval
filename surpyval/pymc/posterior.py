"""Posterior draws from PyMC back into a SurPyval model."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

_REF = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*(?:\[([^\]]*)\])?\s*$")

_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda v: v,
    "exp": np.exp,
    "log": np.log,
    "inverse": lambda v: 1.0 / v,
}


def _parse(ref: str, param: str, what: str) -> tuple[str, tuple]:
    """``"log_alpha[1]"`` -> ``("log_alpha", (1,))``."""
    match = _REF.match(ref) if isinstance(ref, str) else None
    if match is None:
        raise ValueError(
            f"{what}: {param}={ref!r} is not a variable of the trace. Give "
            "its name, with integer indices for an element of a vector "
            "variable: 'beta', 'log_alpha[1]', 'b[0, 2]'."
        )
    name, inside = match.groups()
    if inside is None:
        return name, ()
    try:
        index = tuple(int(i) for i in inside.split(","))
    except ValueError as err:
        raise ValueError(
            f"{what}: the index of {ref!r} must be whole numbers, as in "
            "'log_alpha[1]'."
        ) from err
    return name, index


def _variable(trace: Any, name: str, what: str) -> np.ndarray:
    """One variable's draws with the chains joined: shape
    ``(draws, *variable shape)``."""
    posterior = getattr(trace, "posterior", None)
    if posterior is not None:  # ArviZ InferenceData (or DataTree)
        names = list(getattr(posterior, "data_vars", []))
        if name not in names:
            raise ValueError(
                f"{what}: the trace has no variable {name!r}; its posterior "
                f"has {', '.join(names)}."
            )
        values = np.asarray(posterior[name].values, dtype=float)
        return values.reshape((-1,) + values.shape[2:])
    if hasattr(trace, "get_values") and hasattr(trace, "varnames"):
        if name not in trace.varnames:  # a MultiTrace
            raise ValueError(
                f"{what}: the trace has no variable {name!r}; it has "
                f"{', '.join(trace.varnames)}."
            )
        return np.asarray(trace.get_values(name, combine=True), dtype=float)
    raise TypeError(
        f"{what} takes the trace pm.sample returns (an InferenceData, or a "
        f"MultiTrace with return_inferencedata=False), not a "
        f"{type(trace).__name__}."
    )


def _transform(
    transform: Mapping | None, param: str, names: list, what: str
) -> Callable[[np.ndarray], np.ndarray]:
    if transform is None or param not in transform:
        return _TRANSFORMS["identity"]
    t = transform[param]
    if callable(t):
        return t
    if isinstance(t, str) and t in _TRANSFORMS:
        return _TRANSFORMS[t]
    raise ValueError(
        f"{what}: the transform of {param} is {t!r}; use one of "
        f"{', '.join(_TRANSFORMS)} or a function of the draws."
    )


def to_surpyval(
    trace: Any,
    dist: Any,
    *,
    transform: Mapping | None = None,
    data: Any = None,
    priors: Any = None,
    **param_map: str,
) -> Any:
    """
    A SurPyval Bayesian model from PyMC's posterior draws: the
    distribution with each draw's parameters, with ``sf``, ``ff``, ``qf``,
    credible intervals (``cri``, ``quantile_cri``), ``plot`` and the rest
    of :class:`~surpyval.bayes.BayesianParametric`.

    Parameters
    ----------
    trace : InferenceData or MultiTrace
        What ``pm.sample`` returned.
    dist : distribution
        The SurPyval distribution the draws are for, such as
        ``surpyval.Weibull``.
    transform : mapping, optional
        ``{parameter: "exp" | "log" | "inverse" | function}``, applied to
        the draws of a variable that is not the parameter itself (a log
        scale sampled as ``log_alpha``: ``transform={"alpha": "exp"}``).
    data : SurpyvalData or array-like, optional
        The data, for ``plot`` and ``compare``.
    priors : optional
        The priors, kept with the model for its summary.
    **param_map
        Each parameter's variable in the trace, by name, with integer
        indices for an element of a vector (``alpha="log_alpha[1]"``); a
        parameter not given is the variable of its own name.

    Returns
    -------
    BayesianParametric
        Its ``diagnostics`` record the sampler and the variables used.

    Examples
    --------
    >>> import pytest; az = pytest.importorskip("arviz")
    >>> import numpy as np
    >>> import surpyval as sp
    >>> from surpyval.pymc import to_surpyval
    >>> rng = np.random.default_rng(1)
    >>> trace = az.from_dict({"posterior": {
    ...     "log_alpha": rng.normal(np.log(100), 0.05, (2, 1000)),
    ...     "beta": rng.lognormal(np.log(2), 0.05, (2, 1000))}})
    >>> model = to_surpyval(trace, sp.Weibull, alpha="log_alpha",
    ...                     transform={"alpha": "exp"})
    >>> model.params.round(0)
    array([100.,   2.])
    """
    from surpyval.bayes import BayesianParametric
    from surpyval.utils.weights import (
        BAYES_INSTEAD,
        BAYES_WHY,
        refuse_weighted_data,
    )

    what = "to_surpyval"
    refuse_weighted_data(data, what, BAYES_WHY, BAYES_INSTEAD)
    names = list(getattr(dist, "parameter_names", []))
    if not names or not hasattr(dist, "name"):
        raise TypeError(
            f"{what} takes the distribution the draws are for, such as "
            f"surpyval.Weibull, not a {type(dist).__name__}."
        )
    unknown = sorted(set(param_map) - set(names))
    if unknown:
        raise TypeError(
            f"{what}: {dist.name} has no parameter "
            f"{', '.join(repr(u) for u in unknown)}; its parameters are "
            f"{', '.join(names)}."
        )
    if transform is not None:
        stray = sorted(set(transform) - set(names))
        if stray:
            raise ValueError(
                f"{what}: transform names {', '.join(stray)}, which "
                f"{dist.name} does not have ({', '.join(names)})."
            )
    columns = []
    used = {}
    for param in names:
        ref = param_map.get(param, param)
        name, index = _parse(ref, param, what)
        values = _variable(trace, name, what)
        shape = values.shape[1:]
        if len(index) != len(shape):
            raise ValueError(
                f"{what}: {name} has shape {shape}, so {param} needs "
                f"{len(shape)} ind{'ex' if len(shape) == 1 else 'ices'} "
                f"('{name}[0]'); got {ref!r}."
            )
        try:
            column = values[(slice(None),) + index]
        except IndexError as err:
            raise ValueError(
                f"{what}: {ref!r} is outside {name}, of shape {shape}."
            ) from err
        column = np.asarray(
            _transform(transform, param, names, what)(column), dtype=float
        )
        if not np.all(np.isfinite(column)):
            raise ValueError(
                f"{what}: some draws of {param} ({ref}) are not finite "
                "numbers; check the transform and the sampler's warnings."
            )
        columns.append(column)
        used[param] = ref
    lengths = {len(c) for c in columns}
    if len(lengths) != 1:
        raise ValueError(
            f"{what}: the variables have different numbers of draws "
            f"({sorted(lengths)}); take them from one trace."
        )
    for param, column, bounds in zip(names, columns, dist.bounds):
        lo, hi = bounds
        if (lo is not None and np.any(column <= lo)) or (
            hi is not None and np.any(column >= hi)
        ):
            raise ValueError(
                f"{what}: some draws of {param} are outside its bounds "
                f"{bounds}. Is a transform missing (a log scale needs "
                f"transform={{'{param}': 'exp'}})?"
            )
    return BayesianParametric.from_draws(
        dist,
        np.column_stack(columns),
        names,
        data=data,
        priors=priors,
        diagnostics={"sampler": "pymc", "variables": used},
    )
