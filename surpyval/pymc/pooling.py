"""Groups sharing one population distribution: hierarchical pooling in
PyMC with SurPyval's likelihood.

Several groups (sites, fleets, batches, hospitals) each have their own
life parameters, but those parameters are alike: they are drawn from one
population distribution, which is estimated too. A group with few
failures then borrows strength from the others (its estimate is pulled
towards the population), while a group with plenty of data keeps its
own. This is partial pooling: between fitting every group alone (no
pooling) and fitting them all as one (complete pooling).

:func:`pooled_likelihood` builds it inside a PyMC model, for one
distribution in :data:`~surpyval.pymc.SUPPORTED`:

- each pooled parameter ``p`` is, on its unconstrained scale (the log of
  a positive parameter), ``mu + tau * z_g`` for group ``g``, with
  ``z_g ~ Normal(0, 1)`` (the non-centred form, which samples well when
  groups have little data), ``mu`` the population's centre and ``tau``
  its spread between groups;
- the other parameters are shared by every group (PyMC variables or
  numbers the caller gives);
- the likelihood is SurPyval's exact one for any censoring and
  truncation, each row at its group's parameters.

:func:`group_models` turns the draws back into SurPyval models: one per
group, and one for a new group not yet seen (its parameters drawn from
the population), each a :class:`~surpyval.bayes.BayesianParametric`.

PyMC is imported only when a helper is called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from surpyval.pymc._optional import pymc, pytensor_tensor


@dataclass
class Pooling:
    """What :func:`pooled_likelihood` built: the groups (in the order of
    the vectors' entries), each pooled parameter's variable names, and
    each shared parameter's (or its value)."""

    dist: Any
    groups: list
    index: np.ndarray
    pooled: dict = field(default_factory=dict)
    shared: dict = field(default_factory=dict)
    data: Any = None

    @property
    def counts(self) -> dict:
        """Rows and failures by group."""
        out = {}
        c = np.asarray(self.data.c)
        n = np.asarray(self.data.n)
        for g, label in enumerate(self.groups):
            rows = self.index == g
            out[label] = {
                "units": float(n[rows].sum()),
                "failures": float(n[rows & (c == 0)].sum()),
            }
        return out


def _scale(bounds: tuple, name: str, what: str) -> str:
    lo, hi = bounds
    if lo is None and hi is None:
        return "identity"
    if lo == 0 and hi is None:
        return "log"
    if lo == 0 and hi == 1:
        return "logit"
    raise ValueError(
        f"{what}: {name} is bounded to {bounds}, which pooling does not "
        "take; share it instead (shared=)."
    )


def pooled_likelihood(
    dist: Any,
    data: Any = None,
    *,
    group: Any,
    pool: dict,
    shared: dict | None = None,
    tau_scale: float = 1.0,
    name: str = "groups",
    x: Any = None,
    c: Any = None,
    n: Any = None,
    t: Any = None,
    xl: Any = None,
    xr: Any = None,
    tl: Any = None,
    tr: Any = None,
) -> Pooling:
    """
    Add a hierarchical model of groups sharing one population distribution
    to the open PyMC model, with SurPyval's likelihood of the data.

    Parameters
    ----------
    dist : distribution
        One of :data:`~surpyval.pymc.SUPPORTED`, such as
        ``surpyval.Weibull``.
    data, x, c, n, t, xl, xr, tl, tr
        The data, as for :func:`~surpyval.pymc.log_likelihood`.
    group : array-like
        Each row's group label (a site, a fleet), one per data row.
    pool : dict
        ``{parameter: (centre, spread)}`` for each parameter that varies
        by group: a Normal prior on the population's centre ``mu``, on
        the parameter's unconstrained scale (the log of a positive
        parameter: ``{"alpha": (np.log(1000), 1.0)}`` says the typical
        group's alpha is about 1000, within a factor of e or so).
    shared : dict, optional
        The parameters every group shares, each a PyMC variable or a
        number (``{"beta": beta}``). Every parameter of ``dist`` is in
        ``pool`` or ``shared``.
    tau_scale : float, default 1.0
        The scale of the half-normal prior on each pooled parameter's
        spread between groups ``tau``, on the unconstrained scale: 1.0
        lets groups differ by a factor of a few in a positive parameter;
        smaller pools harder.
    name : str, default "groups"
        The prefix of the variables: ``{name}_{p}_mu``, ``{name}_{p}_tau``
        and ``{name}_{p}_z`` (one per group), and the likelihood term
        ``{name}``.

    Returns
    -------
    Pooling
        The groups in their order and the variables' names, for
        :func:`group_models`.

    Examples
    --------
    >>> import pytest; pm = pytest.importorskip("pymc")
    >>> import numpy as np
    >>> import surpyval as sp
    >>> from surpyval.pymc import pooled_likelihood
    >>> x = np.r_[np.linspace(500, 1500, 10), np.linspace(800, 2000, 4)]
    >>> sites = ["north"] * 10 + ["south"] * 4
    >>> with pm.Model():
    ...     beta = pm.LogNormal("beta", mu=np.log(2.0), sigma=0.5)
    ...     pooling = pooled_likelihood(
    ...         sp.Weibull, x=x, group=sites,
    ...         pool={"alpha": (np.log(1000.0), 1.0)},
    ...         shared={"beta": beta})
    >>> pooling.groups
    ['north', 'south']
    """
    from surpyval.pymc.likelihood import _build, _family, as_surpyval_data

    what = "pooled_likelihood"
    pm = pymc(what)
    pt = pytensor_tensor(what)
    try:
        pm.modelcontext(None)
    except TypeError as err:
        raise TypeError(
            f"{what} adds a model to PyMC: call it inside "
            "`with pm.Model():`."
        ) from err
    family = _family(dist, what)
    data = as_surpyval_data(
        what, data, x=x, c=c, n=n, t=t, xl=xl, xr=xr, tl=tl, tr=tr
    )
    labels = np.asarray(group)
    rows = len(data.c)
    if labels.shape != (rows,):
        raise ValueError(
            f"{what}: group must have one label per data row ({rows}); "
            f"it has shape {labels.shape}."
        )
    groups, index = np.unique(labels, return_inverse=True)
    if len(groups) < 2:
        raise ValueError(
            f"{what}: the data have one group ({groups[0]!r}), so there is "
            "nothing to pool; fit it alone (Weibull.fit_bayes, or "
            "censored_likelihood)."
        )
    if not isinstance(pool, dict) or not pool:
        raise TypeError(
            f"{what}: pool is a dict of the parameters that vary by group, "
            "each with (centre, spread) for its population centre's prior: "
            "pool={'alpha': (np.log(1000), 1.0)}."
        )
    shared = dict(shared or {})
    names = list(dist.parameter_names)
    both = sorted(set(pool) & set(shared))
    stray = sorted((set(pool) | set(shared)) - set(names))
    missing = [p for p in names if p not in pool and p not in shared]
    if both or stray or missing:
        raise TypeError(
            f"{what}: every parameter of {dist.name} ({', '.join(names)}) "
            "goes in pool or in shared, once"
            + (f"; in both: {', '.join(both)}" if both else "")
            + (f"; not a parameter: {', '.join(stray)}" if stray else "")
            + (f"; missing: {', '.join(missing)}" if missing else "")
            + "."
        )
    if not (np.isfinite(tau_scale) and tau_scale > 0):
        raise ValueError(f"{what}: tau_scale must be positive.")
    c_arr = np.asarray(data.c)
    quiet = [
        str(g)
        for k, g in enumerate(groups)
        if not np.any((index == k) & (c_arr != 1))
    ]
    if quiet:
        import warnings

        from surpyval.utils.warnings import caller_stacklevel

        warnings.warn(
            f"{what}: {len(quiet)} of the {len(groups)} groups "
            f"({', '.join(quiet[:5])}{', ...' if len(quiet) > 5 else ''}) "
            "have no failures; they still inform the population through "
            "their survivors, and their own parameters rest on it.",
            stacklevel=caller_stacklevel(),
        )
    G = len(groups)
    params: dict = {}
    pooled: dict = {}
    for p, prior in pool.items():
        try:
            centre, spread = (float(v) for v in prior)
        except (TypeError, ValueError) as err:
            raise TypeError(
                f"{what}: pool[{p!r}] is (centre, spread) of the Normal "
                "prior on the population centre, on the parameter's "
                "unconstrained scale."
            ) from err
        scale = _scale(dist.bounds[names.index(p)], p, what)
        mu = pm.Normal(f"{name}_{p}_mu", mu=centre, sigma=spread)
        tau = pm.HalfNormal(f"{name}_{p}_tau", sigma=tau_scale)
        z = pm.Normal(f"{name}_{p}_z", mu=0.0, sigma=1.0, shape=G)
        u = mu + tau * z
        if scale == "log":
            value = pt.exp(u)
        elif scale == "logit":
            value = pm.math.sigmoid(u)
        else:
            value = u
        params[p] = value[index]
        pooled[p] = {
            "mu": f"{name}_{p}_mu",
            "tau": f"{name}_{p}_tau",
            "z": f"{name}_{p}_z",
            "scale": scale,
        }
    out_shared = {}
    for p, v in shared.items():
        params[p] = v
        out_shared[p] = getattr(v, "name", None) or float(v)
    pm.Potential(name, _build(pt, dist, family, data, params, what))
    return Pooling(
        dist=dist,
        groups=[g.item() if hasattr(g, "item") else g for g in groups],
        index=index,
        pooled=pooled,
        shared=out_shared,
        data=data,
    )


def _back(u: np.ndarray, scale: str) -> np.ndarray:
    if scale == "log":
        return np.exp(u)
    if scale == "logit":
        return 1.0 / (1.0 + np.exp(-u))
    return u


def group_models(
    trace: Any, pooling: Pooling, random_state: Any = None
) -> dict:
    """
    SurPyval models from the draws of a :func:`pooled_likelihood` model:
    ``{group: model}`` for each group, and ``"new group"`` for a group not
    yet seen (each draw's population, ``mu + tau * z`` with a fresh
    standard normal ``z``, so its spread includes the variation between
    groups). Each is a :class:`~surpyval.bayes.BayesianParametric`, with
    ``sf``, ``qf``, credible intervals and plots.

    ``random_state`` seeds the new group's ``z`` (by SurPyval's seed rule).

    Examples
    --------
    >>> import pytest; az = pytest.importorskip("arviz")
    >>> import numpy as np
    >>> import surpyval as sp
    >>> from surpyval.pymc.pooling import Pooling, group_models
    >>> rng = np.random.default_rng(1)
    >>> trace = az.from_dict({"posterior": {
    ...     "g_alpha_mu": rng.normal(np.log(100), 0.05, (2, 500)),
    ...     "g_alpha_tau": np.abs(rng.normal(0.2, 0.02, (2, 500))),
    ...     "g_alpha_z": rng.normal(0, 1, (2, 500, 2)),
    ...     "beta": rng.lognormal(np.log(2), 0.05, (2, 500))}})
    >>> pooling = Pooling(sp.Weibull, ["a", "b"], np.array([0, 1]),
    ...     pooled={"alpha": {"mu": "g_alpha_mu", "tau": "g_alpha_tau",
    ...                       "z": "g_alpha_z", "scale": "log"}},
    ...     shared={"beta": "beta"})
    >>> sorted(group_models(trace, pooling, random_state=1))
    ['a', 'b', 'new group']
    """
    from surpyval.bayes import BayesianParametric
    from surpyval.pymc.posterior import _variable
    from surpyval.utils.rng import as_generator

    what = "group_models"
    if not isinstance(pooling, Pooling):
        raise TypeError(
            f"{what} takes the Pooling that pooled_likelihood returned."
        )
    dist = pooling.dist
    names = list(dist.parameter_names)
    columns: dict = {}
    new: dict = {}
    rng = as_generator(random_state)
    S = None
    for p, v in pooling.pooled.items():
        mu = _variable(trace, v["mu"], what)
        tau = _variable(trace, v["tau"], what)
        z = _variable(trace, v["z"], what)
        S = len(mu)
        columns[p] = _back(mu[:, None] + tau[:, None] * z, v["scale"])
        new[p] = _back(mu + tau * rng.standard_normal(S), v["scale"])
    assert S is not None
    shared: dict = {}
    for p, v in pooling.shared.items():
        if isinstance(v, str):
            shared[p] = _variable(trace, v, what).reshape(S)
        else:
            shared[p] = np.full(S, float(v))
    out = {}

    def model(values: dict, rows: Any) -> Any:
        draws = np.column_stack([values[p] for p in names])
        data = None
        if rows is not None and pooling.data is not None:
            d = pooling.data
            data = (
                np.asarray(d.x)[rows],
                np.asarray(d.c)[rows],
                np.asarray(d.n)[rows],
                np.asarray(d.t)[rows],
            )
        return BayesianParametric.from_draws(
            dist,
            draws,
            names,
            data=data,
            diagnostics={"sampler": "pymc", "pooling": "partial"},
        )

    for g, label in enumerate(pooling.groups):
        values = {p: columns[p][:, g] for p in columns}
        values.update(shared)
        out[label] = model(values, pooling.index == g)
    out["new group"] = model({**new, **shared}, None)
    return out
