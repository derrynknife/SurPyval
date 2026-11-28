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
    #: ``"life"`` (a distribution's lives, ``index`` a row's group) or
    #: ``"nhpp"`` (event histories of systems, ``index`` a system's
    #: group, in the order of ``data.ids``).
    kind: str = "life"
    centred: bool = False

    def _rows(self, g: int) -> np.ndarray:
        """The data rows of group ``g``."""
        if self.kind == "nhpp":
            _, inverse = self.data.item_rows()
            return np.flatnonzero(self.index[inverse] == g)
        return np.flatnonzero(self.index == g)

    @property
    def counts(self) -> dict:
        """Rows and failures by group (systems and events for an
        NHPP)."""
        out = {}
        if self.kind == "nhpp":
            for g, label in enumerate(self.groups):
                sub = self.data[self._rows(g)]
                c = np.asarray(sub.c)
                n = np.asarray(sub.n, dtype=float)
                out[label] = {
                    "systems": float(len(sub.ids)),
                    "events": float(
                        np.sum(c == 0) + n[np.isin(c, (-1, 2))].sum()
                    ),
                }
            return out
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
    group: Any = None,
    pool: dict,
    shared: dict | None = None,
    tau_scale: float = 1.0,
    name: str = "groups",
    centred: bool = False,
    x: Any = None,
    i: Any = None,
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
    dist : distribution or CrowAMSAA
        One of :data:`~surpyval.pymc.SUPPORTED`, such as
        ``surpyval.Weibull``; or ``surpyval.recurrent.CrowAMSAA``, the
        power-law NHPP, for systems' event histories (a fleet whose
        systems share a population of growth rates ``beta``).
    data, x, c, n, t, xl, xr, tl, tr
        The data, as for :func:`~surpyval.pymc.log_likelihood`; for the
        NHPP, ``x``, ``i``, ``c``, ``n``, ``t``, ``tl``, ``tr`` as for
        ``CrowAMSAA.fit`` (or an ``EventHistoryData`` as ``data``).
    group : array-like or dict
        Each row's group label (a site, a fleet), one per data row. For
        the NHPP a group is a set of whole systems (a label a row of
        ``i``, or ``{system: group}``), and by default each system is
        its own group.
    i : array-like, optional
        The NHPP's systems, one a data row.
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
    centred : bool, default False
        The centred form, ``u_g ~ Normal(mu, tau)`` (one variable
        ``{name}_{p}`` a group), in place of the non-centred ``mu + tau
        * z_g``. The non-centred form samples well when the groups have
        little data each (the usual case for pooling); the centred one
        when every group has plenty, where the non-centred form's
        ``z`` and ``tau`` are tied together and it samples worse.

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
    from surpyval.pymc.nhpp import is_power_law

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
    if is_power_law(dist):
        if any(v is not None for v in (xl, xr)):
            raise TypeError(
                f"{what}: an NHPP's data are x, i, c, n, t, tl and tr, as "
                "for CrowAMSAA.fit."
            )
        return _pooled_nhpp(
            pm,
            pt,
            dist,
            data if data is not None else x,
            i,
            c,
            n,
            t,
            tl,
            tr,
            group,
            pool,
            shared,
            tau_scale,
            name,
            centred,
            what,
        )
    if i is not None:
        raise TypeError(
            f"{what}: i (systems) is for an NHPP's event histories "
            "(dist=CrowAMSAA); lives are grouped by group=."
        )
    if group is None:
        raise TypeError(
            f"{what}: group is each row's group label (a site, a fleet)."
        )
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
        scale = _scale(dist.bounds[names.index(p)], p, what)
        value, pooled[p] = _population(
            pm, pt, name, p, prior, scale, G, tau_scale, centred, what
        )
        params[p] = value[index]
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
        centred=bool(centred),
    )


def _population(
    pm: Any,
    pt: Any,
    name: str,
    p: str,
    prior: Any,
    scale: str,
    G: int,
    tau_scale: float,
    centred: bool,
    what: str,
) -> tuple[Any, dict]:
    """The population of one pooled parameter: its value a group (on
    the parameter's own scale) and its variables' names."""
    try:
        centre, spread = (float(v) for v in prior)
    except (TypeError, ValueError) as err:
        raise TypeError(
            f"{what}: pool[{p!r}] is (centre, spread) of the Normal "
            "prior on the population centre, on the parameter's "
            "unconstrained scale."
        ) from err
    mu = pm.Normal(f"{name}_{p}_mu", mu=centre, sigma=spread)
    tau = pm.HalfNormal(f"{name}_{p}_tau", sigma=tau_scale)
    names = {
        "mu": f"{name}_{p}_mu",
        "tau": f"{name}_{p}_tau",
        "scale": scale,
    }
    if centred:
        u = pm.Normal(f"{name}_{p}", mu=mu, sigma=tau, shape=G)
        names["u"] = f"{name}_{p}"
    else:
        z = pm.Normal(f"{name}_{p}_z", mu=0.0, sigma=1.0, shape=G)
        u = mu + tau * z
        names["z"] = f"{name}_{p}_z"
    if scale == "log":
        value = pt.exp(u)
    elif scale == "logit":
        value = pm.math.sigmoid(u)
    else:
        value = u
    return value, names


def _pooled_nhpp(
    pm: Any,
    pt: Any,
    dist: Any,
    x: Any,
    i: Any,
    c: Any,
    n: Any,
    t: Any,
    tl: Any,
    tr: Any,
    group: Any,
    pool: dict,
    shared: dict | None,
    tau_scale: float,
    name: str,
    centred: bool,
    what: str,
) -> Pooling:
    """:func:`pooled_likelihood` for the power-law NHPP: groups of whole
    systems, each with its own parameters drawn from the population."""
    from surpyval.pymc.nhpp import (
        _group_pieces,
        _item_groups,
        _log_lik,
        event_data,
    )

    data, _ = event_data(what, x, i, c, n, t, tl, tr)
    labels, item_group = _item_groups(what, data, group, i)
    G = len(labels)
    if G < 2:
        raise ValueError(
            f"{what}: the data have one group ({labels[0]!r}), so there is "
            "nothing to pool; fit it alone (CrowAMSAA.fit_bayes, or "
            "nhpp_likelihood)."
        )
    _check_parameters(dist, pool, shared, tau_scale, what)
    values: dict = {}
    pooled: dict = {}
    for p, prior in pool.items():
        values[p], pooled[p] = _population(
            pm, pt, name, p, prior, "log", G, tau_scale, centred, what
        )
    out_shared = {}
    for p, v in dict(shared or {}).items():
        values[p] = pt.as_tensor_variable(
            v if hasattr(v, "type") else np.float64(v)
        ) * pt.ones(G)
        out_shared[p] = getattr(v, "name", None) or float(v)
    built = _group_pieces(data, item_group, G)
    pm.Potential(
        name,
        _log_lik(pt, built, pt.log(values["alpha"]), values["beta"], True),
    )
    return Pooling(
        dist=dist,
        groups=labels,
        index=item_group,
        pooled=pooled,
        shared=out_shared,
        data=data,
        kind="nhpp",
        centred=bool(centred),
    )


def _check_parameters(
    dist: Any, pool: Any, shared: Any, tau_scale: float, what: str
) -> None:
    if not isinstance(pool, dict) or not pool:
        raise TypeError(
            f"{what}: pool is a dict of the parameters that vary by group, "
            "each with (centre, spread) for its population centre's prior: "
            "pool={'beta': (np.log(0.8), 0.5)}."
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
        mu, tau, u = _draws(trace, v, what)
        S = len(mu)
        columns[p] = _back(u, v["scale"])
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
        if pooling.kind == "nhpp":
            sub = pooling.data if rows is None else pooling.data[rows]
            return _recurrence_from_draws(dist, draws, sub)
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
        out[label] = model(values, pooling._rows(g))
    out["new group"] = model({**new, **shared}, None)
    return out


def _draws(trace: Any, v: dict, what: str) -> tuple:
    """A pooled parameter's population centre and spread, and its value
    a group on the unconstrained scale, ``(S,)``, ``(S,)``, ``(S, G)``."""
    from surpyval.pymc.posterior import _variable

    mu = _variable(trace, v["mu"], what)
    tau = _variable(trace, v["tau"], what)
    if "u" in v:
        u = _variable(trace, v["u"], what)
    else:
        u = mu[:, None] + tau[:, None] * _variable(trace, v["z"], what)
    return mu, tau, u


def _recurrence_from_draws(dist: Any, draws: np.ndarray, data: Any) -> Any:
    """A :class:`~surpyval.recurrent.bayes.BayesianRecurrence` of the
    power-law process from PyMC's draws of ``(alpha, beta)``, on the
    systems ``data`` (for a new system, the whole fleet's: its ``end``
    is the latest time any system was observed to)."""
    from surpyval.bayes.posterior import Posterior
    from surpyval.bayes.prior import PriorSpec
    from surpyval.bayes.transforms import Transform
    from surpyval.recurrent.bayes import BayesianRecurrence, RecurrenceTarget

    target = RecurrenceTarget(dist, data, None)
    names = list(dist.parameter_names)
    S = len(draws)
    post = Posterior(
        names,
        names,
        np.asarray(draws, dtype=float),
        np.full(S, 1.0 / S),
        "draws",
        transforms=[Transform.for_bounds(b) for b in dist.bounds],
    )
    return BayesianRecurrence(
        target,
        post,
        PriorSpec(),
        {},
        {"method": "pymc", "sampler": "pymc", "pooling": "partial"},
    )


def shrinkage(trace: Any, pooling: Pooling) -> Any:
    """
    How far pooling pulled each group, as a table: for every pooled
    parameter and group, the group's own maximum-likelihood estimate (its
    data alone), its pooled posterior median, and the population's
    median, with ``pulled``: the share of the way from the group's own
    estimate to the population's that pooling moved it, on the
    parameter's unconstrained scale (0 is no pooling, 1 complete
    pooling). A group with little data is pulled far; one with plenty
    keeps its own.

    Parameters
    ----------
    trace
        What ``pm.sample`` returned for a :func:`pooled_likelihood`
        model.
    pooling : Pooling
        What :func:`pooled_likelihood` returned.

    Returns
    -------
    pandas.DataFrame
        One row a group and pooled parameter: ``group``, ``parameter``,
        the group's data (``units`` and ``failures``, or ``systems`` and
        ``events``), ``alone``, ``pooled``, ``population`` and
        ``pulled``. ``alone`` is ``NaN`` where the group's data alone
        have no maximum-likelihood fit (no failures, say).

    Examples
    --------
    >>> import pytest; az = pytest.importorskip("arviz")
    >>> import numpy as np
    >>> import surpyval as sp
    >>> from surpyval.pymc.pooling import Pooling, shrinkage
    >>> rng = np.random.default_rng(1)
    >>> x = np.r_[np.linspace(500, 1500, 10), np.linspace(800, 2000, 4)]
    >>> data = sp.SurpyvalData(x=x, group_and_sort=False)
    >>> trace = az.from_dict({"posterior": {
    ...     "g_alpha_mu": rng.normal(np.log(1100), 0.05, (2, 500)),
    ...     "g_alpha_tau": np.abs(rng.normal(0.2, 0.02, (2, 500))),
    ...     "g_alpha_z": rng.normal(0, 1, (2, 500, 2)),
    ...     "beta": rng.lognormal(np.log(2), 0.05, (2, 500))}})
    >>> pooling = Pooling(sp.Weibull, ["a", "b"], np.repeat([0, 1], [10, 4]),
    ...     pooled={"alpha": {"mu": "g_alpha_mu", "tau": "g_alpha_tau",
    ...                       "z": "g_alpha_z", "scale": "log"}},
    ...     shared={"beta": "beta"}, data=data)
    >>> table = shrinkage(trace, pooling)
    >>> list(table.columns[4:])
    ['alone', 'pooled', 'population', 'pulled']
    """
    import pandas as pd

    from surpyval.utils.no_maximum import quiet_maximum_warnings
    from surpyval.utils.warnings import ignoring_warnings

    what = "shrinkage"
    if not isinstance(pooling, Pooling):
        raise TypeError(
            f"{what} takes the Pooling that pooled_likelihood returned."
        )
    if pooling.data is None:
        raise ValueError(
            f"{what} needs the data the groups were fitted to; this "
            "Pooling holds none."
        )
    counts = pooling.counts
    alone: dict = {}
    names = list(pooling.dist.parameter_names)
    for g, label in enumerate(pooling.groups):
        rows = pooling._rows(g)
        try:
            with ignoring_warnings(), quiet_maximum_warnings():
                with np.errstate(all="ignore"):
                    if pooling.kind == "nhpp":
                        sub = pooling.data[rows]
                        fit = pooling.dist.fit(sub)
                    else:
                        d = pooling.data
                        fit = pooling.dist.fit(
                            np.asarray(d.x)[rows],
                            np.asarray(d.c)[rows],
                            np.asarray(d.n)[rows],
                            np.asarray(d.t)[rows],
                        )
            alone[label] = dict(zip(names, np.asarray(fit.params, float)))
        except (ValueError, TypeError, np.linalg.LinAlgError):
            alone[label] = {}
    out = []
    for p, v in pooling.pooled.items():
        mu, _, u = _draws(trace, v, what)
        scale = v["scale"]
        centre = float(np.median(mu))
        for g, label in enumerate(pooling.groups):
            u_g = float(np.median(u[:, g]))
            own = alone[label].get(p, np.nan)
            with np.errstate(all="ignore"):
                own_u = _forward(np.asarray(own, dtype=float), scale)
                pulled = float((own_u - u_g) / (own_u - centre))
            row = {"group": label, "parameter": p}
            row.update(counts[label])
            row.update(
                {
                    "alone": float(own),
                    "pooled": float(_back(np.asarray(u_g), scale)),
                    "population": float(_back(np.asarray(centre), scale)),
                    "pulled": pulled,
                }
            )
            out.append(row)
    return pd.DataFrame(out)


def _forward(value: np.ndarray, scale: str) -> np.ndarray:
    if scale == "log":
        return np.log(value)
    if scale == "logit":
        return np.log(value / (1.0 - value))
    return value
