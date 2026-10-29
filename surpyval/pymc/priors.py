"""PyMC priors from SurPyval: a fitted model, or a plain prior spec."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from surpyval.pymc._optional import pymc, pytensor_tensor


def _unconstrained(bounds: tuple, name: str, what: str) -> str:
    """``"log"`` for a positive parameter, ``"identity"`` for one on the
    real line; others are refused."""
    lo, hi = bounds
    if lo is None and hi is None:
        return "identity"
    if lo == 0 and hi is None:
        return "log"
    raise ValueError(
        f"{what}: the parameter {name} is bounded to {bounds}, which has no "
        "normal approximation on a log or plain scale here. Give it a prior "
        "of your own with prior_from_spec."
    )


def _in_model(pm: Any, what: str) -> None:
    try:
        pm.modelcontext(None)
    except TypeError as err:
        raise TypeError(
            f"{what} adds variables to a PyMC model: call it inside "
            "`with pm.Model():`."
        ) from err


def prior_from_model(
    model: Any, name: str | None = None, weight: float = 1.0
) -> dict:
    """
    A joint PyMC prior from a fitted SurPyval model: its estimates and
    covariance, as a multivariate normal on the unconstrained scale (the
    log of a positive parameter, a real-line parameter as it is).

    In plain words: last time's fit becomes what you believe before the
    new data, as sure of each parameter as that fit was, and keeping how
    the parameters moved together. ``weight`` discounts it: 0.5 counts
    the old fit as half as much information (its covariance is divided
    by 0.5), for data that are only partly like the new.

    Parameters
    ----------
    model : Parametric
        A fitted model with a covariance (``model.covariance()``), such as
        ``surpyval.Weibull.fit(x, c)``; not one with an offset, a
        limited-failure proportion or zero inflation.
    name : str, optional
        The prefix of the variables added; default the distribution's
        name in lower case (``"weibull"``). The prior is ``{name}_z``, the
        unconstrained vector, and ``{name}_{parameter}`` each parameter.
    weight : float, default 1.0
        The share of the old fit's information to keep, in (0, 1].

    Returns
    -------
    dict
        ``{parameter: variable}``, ready for
        :func:`~surpyval.pymc.censored_likelihood` as ``**prior``.

    Examples
    --------
    >>> import pytest; pm = pytest.importorskip("pymc")
    >>> import surpyval as sp
    >>> from surpyval.pymc import prior_from_model
    >>> old = sp.Weibull.fit([310, 520, 640, 790, 905, 1100, 1240])
    >>> with pm.Model() as m:
    ...     prior = prior_from_model(old, name="fleet", weight=0.5)
    >>> sorted(prior)
    ['alpha', 'beta']
    >>> prior["alpha"].name
    'fleet_alpha'
    """
    what = "prior_from_model"
    pm = pymc(what)
    pt = pytensor_tensor(what)
    _in_model(pm, what)
    if not (hasattr(model, "dist") and hasattr(model, "params")):
        raise TypeError(
            f"{what} takes a fitted SurPyval model (such as "
            f"surpyval.Weibull.fit(x, c)), not a {type(model).__name__}."
        )
    for flag, word in (
        ("offset", "an offset (gamma)"),
        ("lfp", "a limited-failure proportion"),
        ("zi", "zero inflation"),
    ):
        if getattr(model, flag, False):
            raise ValueError(
                f"{what}: the model has {word}, which this prior does not "
                "carry. Fit the old data without it, or build the prior "
                "with prior_from_spec."
            )
    weight = float(weight)
    if not (0 < weight <= 1):
        raise ValueError(
            f"{what}: weight is the share of the old fit's information to "
            f"keep, in (0, 1]; got {weight}."
        )
    names = list(model.dist.parameter_names)
    try:
        cov = np.asarray(model.covariance(), dtype=float)
    except ValueError as err:
        raise ValueError(
            f"{what} needs the fit's covariance, and this model has none "
            f"({err}). Use prior_from_spec and say how sure you are."
        ) from err
    params = np.asarray(model.params, dtype=float)
    k = len(names)
    cov = cov[:k, :k]
    scales = [
        _unconstrained(b, p, what) for b, p in zip(model.dist.bounds, names)
    ]
    jac = np.array(
        [1.0 / v if s == "log" else 1.0 for v, s in zip(params, scales)]
    )
    mean = np.array(
        [np.log(v) if s == "log" else v for v, s in zip(params, scales)]
    )
    cov_u = cov * np.outer(jac, jac) / weight
    try:
        np.linalg.cholesky(cov_u)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            f"{what}: the fit's covariance is not positive definite, so it "
            "is no prior. Refit with more data or use prior_from_spec."
        ) from err
    prefix = model.dist.name.lower() if name is None else name
    z = pm.MvNormal(f"{prefix}_z", mu=mean, cov=cov_u)
    out = {}
    for i, (p, s) in enumerate(zip(names, scales)):
        value = pt.exp(z[i]) if s == "log" else z[i]
        out[p] = pm.Deterministic(f"{prefix}_{p}", value)
    return out


# A SurPyval family's name (lower case) -> how to build the PyMC
# variable from its parameters, in SurPyval's order, and its support.
def _pymc_dist(pm: Any, family: str, params: list, var: str) -> Any:
    builders: dict[str, Callable[..., Any]] = {
        "normal": lambda a, b: pm.Normal(var, mu=a, sigma=b),
        "lognormal": lambda a, b: pm.LogNormal(var, mu=a, sigma=b),
        # SurPyval's Gamma: shape alpha, rate beta (as PyMC's)
        "gamma": lambda a, b: pm.Gamma(var, alpha=a, beta=b),
        # SurPyval's Weibull: scale alpha, shape beta (PyMC's are swapped)
        "weibull": lambda a, b: pm.Weibull(var, alpha=b, beta=a),
        "exponential": lambda a: pm.Exponential(var, lam=a),
        "uniform": lambda a, b: pm.Uniform(var, lower=a, upper=b),
        "beta": lambda a, b: pm.Beta(var, alpha=a, beta=b),
        "logistic": lambda a, b: pm.Logistic(var, mu=a, s=b),
    }
    return builders[family](*params)


PRIOR_FAMILIES = (
    "Normal",
    "LogNormal",
    "Gamma",
    "Weibull",
    "Exponential",
    "Uniform",
    "Beta",
    "Logistic",
)
_K = {"exponential": 1}
_REAL_LINE = {"normal", "logistic"}


def _read_spec(entry: Any, param: str, what: str) -> tuple[str, list]:
    """``(family, params)`` from a fitted SurPyval model, a sequence
    ``(family, *params)`` or a dict ``{"dist": family, "params": [...]}``."""
    if hasattr(entry, "dist") and hasattr(entry, "params"):
        family = entry.dist.name
        params = [float(v) for v in entry.params]
    elif isinstance(entry, Mapping):
        if set(entry) != {"dist", "params"}:
            raise ValueError(
                f"{what}: the prior on {param} as a dict has the keys "
                f"'dist' and 'params'; got {sorted(entry)}."
            )
        family, params = entry["dist"], list(entry["params"])
    elif isinstance(entry, (list, tuple)) and entry:
        family, params = entry[0], list(entry[1:])
    else:
        raise TypeError(
            f"{what}: the prior on {param} is a SurPyval distribution "
            "(sp.LogNormal.from_params([mu, sigma])), a tuple such as "
            "('lognormal', mu, sigma), or {'dist': ..., 'params': [...]}; "
            f"got {entry!r}."
        )
    known = {f.lower(): f for f in PRIOR_FAMILIES}
    if not isinstance(family, str) or family.lower() not in known:
        raise ValueError(
            f"{what}: no PyMC prior for {family!r} on {param}. Use one of "
            f"{', '.join(PRIOR_FAMILIES)}, with SurPyval's parameters in "
            "SurPyval's order."
        )
    family = family.lower()
    k = _K.get(family, 2)
    if len(params) != k:
        raise ValueError(
            f"{what}: a {known[family]} prior on {param} takes {k} "
            f"parameter{'s' if k > 1 else ''} in SurPyval's order; got "
            f"{len(params)}."
        )
    return family, [float(v) for v in params]


def prior_from_spec(
    spec: Mapping, dist: Any = None, name: str | None = None
) -> dict:
    """
    Independent PyMC priors, one per parameter, from a plain description
    that can be stored as JSON.

    Each entry is a SurPyval distribution with its parameters
    (``sp.LogNormal.from_params([0.8, 0.3])``), the same as a sequence
    ``("lognormal", 0.8, 0.3)``, or ``{"dist": "lognormal", "params":
    [0.8, 0.3]}``. The parameters are SurPyval's, in SurPyval's order
    (a Weibull prior is ``("weibull", scale, shape)``; a Gamma prior's
    second parameter is its rate), whatever PyMC calls them.

    Parameters
    ----------
    spec : mapping
        ``{parameter: prior}``.
    dist : distribution, optional
        The distribution the priors are for (``surpyval.Weibull``): then
        every name must be one of its parameters, and a prior on a
        positive parameter must be on positive values.
    name : str, optional
        A prefix for the variables' names (``{name}_{parameter}``);
        default none, so each variable is named after its parameter.

    Returns
    -------
    dict
        ``{parameter: variable}``.

    Examples
    --------
    >>> import pytest; pm = pytest.importorskip("pymc")
    >>> import surpyval as sp
    >>> from surpyval.pymc import prior_from_spec
    >>> with pm.Model() as m:
    ...     prior = prior_from_spec(
    ...         {"alpha": ("lognormal", 7.0, 1.0),
    ...          "beta": sp.LogNormal.from_params([0.8, 0.3])},
    ...         dist=sp.Weibull)
    >>> sorted(v.name for v in m.free_RVs)
    ['alpha', 'beta']
    """
    what = "prior_from_spec"
    pm = pymc(what)
    _in_model(pm, what)
    if not isinstance(spec, Mapping) or not spec:
        raise TypeError(
            f"{what} takes {{parameter: prior}}, such as "
            "{'beta': ('lognormal', 0.8, 0.3)}."
        )
    bounds = {}
    if dist is not None:
        names = list(dist.parameter_names)
        unknown = sorted(set(spec) - set(names))
        if unknown:
            raise ValueError(
                f"{what}: {dist.name} has no parameter "
                f"{', '.join(repr(u) for u in unknown)}; its parameters "
                f"are {', '.join(names)}."
            )
        bounds = dict(zip(names, dist.bounds))
    out = {}
    for param, entry in spec.items():
        family, params = _read_spec(entry, param, what)
        lo = bounds.get(param, (None, None))[0]
        below = family in _REAL_LINE or (family == "uniform" and params[0] < 0)
        if lo is not None and below:
            raise ValueError(
                f"{what}: {param} must be above {lo}, and a {family} prior "
                "puts mass below it. Use a LogNormal or Gamma prior, or put "
                f"a normal prior on log({param}) and pass its exp."
            )
        var = param if name is None else f"{name}_{param}"
        out[param] = _pymc_dist(pm, family, params, var)
    return out
