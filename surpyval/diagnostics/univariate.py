"""``diagnose()`` for a fitted univariate distribution (#875).

A :class:`~surpyval.univariate.parametric.parametric.Parametric` model --
``Weibull.fit(x, c, n, t)`` and the other continuous distributions, with
an offset, a limited failure population, zero inflation or fixed
parameters as fitted -- is checked for whether the distribution fits the
data and whether another model would fit them better. The checks are
listed in :data:`UNIVARIATE` at the bottom.

The checks
----------
``goodness_of_fit``
    The Anderson-Darling, Kolmogorov-Smirnov and Cramer-von Mises
    statistics, with parametric bootstrap p-values. Each unit's time is
    put through the fitted CDF, conditioned on the unit's truncation
    window: ``u = (F(x) - F(tl)) / (F(tr) - F(tl))``, which is uniform on
    (0, 1) when the model is right, and a right-censored ``u`` for a
    right-censored unit. The statistics compare the Kaplan-Meier estimate
    ``F_n`` of the ``u`` with the uniform CDF, up to the largest ``u``
    (to 1 when ``F_n`` reaches 1): ``KS = sup |F_n(u) - u|``,
    ``CvM = N int (F_n(u) - u)^2 du`` and ``AD = N int (F_n(u) - u)^2 /
    (u (1 - u)) du`` over ``N`` units. With complete, untruncated data
    they are the classical statistics (``fitdistrplus::gofstat``'s); with
    censoring, Koziol and Green's (CvM) and their Anderson-Darling and
    Kolmogorov-Smirnov analogues.

    Estimated parameters and censoring change the statistics' null
    distribution, so the p-values come from a parametric bootstrap
    (``n_boot`` refits, :data:`DEFAULT_N_BOOT` by default): each resample
    draws a time for every unit from the fitted model within the unit's
    truncation window; censors it as the data were censored (Davison and
    Hinkley's conditional bootstrap, as the bootstrap bounds of #645 do:
    a censored unit at its own censoring time, a failed one at a time
    drawn from the Kaplan-Meier estimate of the censoring distribution
    beyond its failure, so fixed-time (Type I) and random censoring are
    reproduced exactly); refits the same model -- distribution, ``how``,
    offset, ``lfp``, ``zi`` and fixed parameters -- to the resample; and
    computes the statistics of the refit. The p-value is ``(1 + #{refits
    at least as extreme}) / (1 + refits)``. With complete data from a
    location-scale family (the Weibull, on the log scale) the statistics'
    null distribution does not depend on the parameters, and the test is
    exact: in 400 exponential samples of 30, 9-10% of the p-values were at
    most 0.1. With censoring it is somewhat conservative in small
    samples: 5-9% at most 0.1 in two runs of 400 randomly censored
    exponential samples of 40 (about a third censored), and 5-9% in 150
    truncated, censored Weibull samples of 60 -- much the same whether
    the censoring times are simulated conditionally or from the censoring
    distribution outright. Left- and interval-censored data have no such
    test: a resample would need the inspection times, which the data do
    not record, and the rows say so (and warn).
``probability_plot``
    A plot only: the model's probability plot (``model.plot()``), the
    plotting positions of the failures against the fitted line on the
    distribution's linearising axes.
``alternatives``
    An AIC comparison of the model with the alternatives the advice
    names, each fitted to the same data by maximum likelihood:

    - the best other distribution :func:`~surpyval.fit_best.fit_best`
      ranks (its default candidates);
    - a mixture of two of the model's distribution (``MixtureModel``),
      for a model without an offset, ``lfp``, ``zi`` or fixed
      parameters, with at least :data:`MIXTURE_MIN_FAILURES` (20)
      failures, and not when some units failed at time 0 (a component
      would collapse onto them);
    - the model with an offset (a 3-parameter model), for a
      distribution on (0, inf) without one;
    - the model with a limited failure population (``lfp=True``), when
      some units are right-censored;
    - a zero-inflated model (``zi=True``; the model's distribution, or
      for one not on (0, inf) the best of those that are), when some
      units failed at time 0.

    An alternative whose likelihood has no finite maximum is left out;
    one at an unverified maximum (a limited-failure proportion at 1, say)
    is kept, its AIC then overstated, never understated. A screen, not a
    test: a row warns when the alternative's AIC is at least
    :data:`AIC_MARGIN` (2) below the model's -- "substantially less
    support" for the model on Burnham and Anderson's scale -- and never
    fails. Its ``statistic`` is that difference, the model's AIC less the
    alternative's.

What is diagnosed
-----------------
A continuous distribution fitted to data, by any ``how``. ``diagnose()``
refuses, saying what to do: a model without data (``from_params``, or
restored without them), one whose fit has no finite maximum, a discrete
distribution and a fit by a fitter without the shared optimiser (not yet,
#875), and an event-probability model (``Bernoulli``,
``FixedEventProbability``) or ``ExactEventTime``, which have no
distribution of times to test. A mixture (``MixtureModel``) is a model
class of its own, without ``diagnose()``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .registry import AdviceRule, Check, Context, Family, Panel, register

#: The bootstrap refits of the goodness-of-fit tests when ``diagnose()``
#: is not given ``n_boot``.
DEFAULT_N_BOOT = 200
#: An alternative whose AIC is at least this much below the model's warns.
AIC_MARGIN = 2.0
#: The fewest failures (exact, left- or interval-censored units) for which
#: a two-component mixture, with twice the parameters and a weight, is
#: tried as an alternative.
MIXTURE_MIN_FAILURES = 20
#: The goodness-of-fit tests, in the order of the table: the name in the
#: ``test`` column and the key of :func:`gof_statistics`.
TESTS = (
    ("Anderson-Darling", "AD"),
    ("Kolmogorov-Smirnov", "KS"),
    ("Cramer-von Mises", "CvM"),
)
#: How far ``u`` is kept from 0 and 1 in the Anderson-Darling weight.
_EPS = 1e-12
#: The distributions the advice can name a fit of, as ``surpyval``
#: exports them (a model of another, a custom distribution say, gets the
#: same advice without a named feature).
NAMED = (
    "Beta",
    "Exponential",
    "ExpoWeibull",
    "Gamma",
    "Gumbel",
    "GumbelLEV",
    "Logistic",
    "LogLogistic",
    "LogNormal",
    "Normal",
    "Rayleigh",
    "Weibull",
)


def _fmt(value: float) -> str:
    return "{:.3g}".format(value)


# ---------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------
def _ad_antiderivative(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """An antiderivative in ``u`` of ``(v - u)^2 / (u (1 - u))``, ``-u -
    (1 - v)^2 log(1 - u) + v^2 log(u)``, with ``0 log 0 = 0``."""
    uc = np.clip(u, _EPS, 1 - _EPS)
    with np.errstate(all="ignore"):
        upper = np.where(v >= 1, 0.0, (1 - v) ** 2 * np.log1p(-uc))
        lower = np.where(v <= 0, 0.0, v**2 * np.log(uc))
    return -u - upper + lower


def gof_statistics(
    u: np.ndarray, failed: np.ndarray, counts: "np.ndarray | None" = None
) -> dict:
    """
    The Anderson-Darling (``AD``), Kolmogorov-Smirnov (``KS``) and
    Cramer-von Mises (``CvM``) statistics of a right-censored sample on
    (0, 1) against the uniform distribution.

    ``u`` are the values (a model's CDF at each unit's time), ``failed``
    whether each was observed (True) or right-censored, and ``counts``
    the units each stands for. The statistics compare the Kaplan-Meier
    estimate ``F_n`` with ``u``, integrated exactly over each step of
    ``F_n`` up to the largest value (to 1 when ``F_n`` reaches 1); see
    :mod:`surpyval.diagnostics.univariate`. With no censoring they are the
    classical statistics.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.diagnostics.univariate import gof_statistics
    >>> u = np.array([0.1, 0.3, 0.5, 0.8])
    >>> out = gof_statistics(u, np.ones(4, bool))
    >>> {k: round(v, 4) for k, v in out.items()}
    {'AD': 0.288, 'KS': 0.25, 'CvM': 0.0483}
    >>> i = np.arange(1, 5)  # the classical formula agrees
    >>> round(float(1 / 48 + np.sum((u - (2 * i - 1) / 8) ** 2)), 4)
    0.0483
    """
    u = np.asarray(u, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    w = (
        np.ones(u.size)
        if counts is None
        else np.asarray(counts, dtype=float).reshape(-1)
    )
    N = float(w.sum())
    times, inverse = np.unique(u, return_inverse=True)
    d = np.bincount(inverse, weights=w * failed, minlength=times.size)
    total = np.bincount(inverse, weights=w, minlength=times.size)
    at_risk = N - np.concatenate([[0.0], np.cumsum(total)[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.cumprod(1.0 - np.where(at_risk > 0, d / at_risk, 0.0))
    jumps = d > 0
    t, F = times[jumps], 1.0 - S[jumps]
    end = 1.0 if F.size and F[-1] >= 1.0 - 1e-12 else float(times[-1])
    a = np.concatenate([[0.0], t])
    b = np.concatenate([t, [end]])
    v = np.concatenate([[0.0], F])
    ks = float(np.max(np.maximum(np.abs(v - a), np.abs(v - b))))
    cvm = N * float(np.sum(((b - v) ** 3 - (a - v) ** 3) / 3.0))
    ad = N * float(np.sum(_ad_antiderivative(b, v) - _ad_antiderivative(a, v)))
    return {"AD": ad, "KS": ks, "CvM": cvm}


def _window(model: Any, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The model's CDF at each unit's truncation bounds (0 and 1 where
    there is none)."""
    tl, tr = t[:, 0], t[:, 1]
    with np.errstate(all="ignore"):
        lo = np.where(np.isfinite(tl), model.ff(np.nan_to_num(tl)), 0.0)
        hi = np.where(np.isfinite(tr), model.ff(np.nan_to_num(tr)), 1.0)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def transform(model: Any, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Each unit's ``u``: the model's CDF at its time, conditioned on its
    truncation window, ``(F(x) - F(tl)) / (F(tr) - F(tl))`` (``t`` is
    ``[tl, tr]`` per unit, infinite where there is no truncation).

    Examples
    --------
    >>> import numpy as np
    >>> import surpyval as sp
    >>> from surpyval.diagnostics.univariate import transform
    >>> model = sp.Exponential.from_params([0.1])
    >>> t = np.array([[-np.inf, np.inf], [5.0, np.inf]])
    >>> transform(model, np.array([10.0, 15.0]), t).round(4)
    array([0.6321, 0.6321])
    """
    lo, hi = _window(model, t)
    with np.errstate(all="ignore"):
        u = (np.asarray(model.ff(x), dtype=float) - lo) / (hi - lo)
    return np.clip(np.nan_to_num(u, nan=0.0), 0.0, 1.0)


def _statistics(model: Any, x: Any, c: Any, n: Any, t: Any) -> dict:
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1, 2)
    u = transform(model, x, t)
    return gof_statistics(u, np.asarray(c) == 0, n)


# ---------------------------------------------------------------------------
# The parametric bootstrap
# ---------------------------------------------------------------------------
def _simulable(model: Any) -> bool:
    c = np.asarray(model.data["c"])
    return bool(np.all((c == 0) | (c == 1)))


def _design(model: Any) -> Any:
    """What a resample keeps of the data: each unit's truncation window
    and censoring, with the censoring distribution of the bootstrap
    bounds (``ResampleDesign``; a row with ``n = k`` is ``k`` units)."""
    from surpyval.univariate.regression._bootstrap import ResampleDesign

    data = model.data
    c = np.asarray(data["c"])
    x = np.asarray(data["x"], dtype=float)
    units = np.repeat(np.arange(c.size), np.asarray(data["n"], dtype=int))
    t = np.asarray(data["t"], dtype=float).reshape(-1, 2)[units]
    # Built field by field: its own constructor reads a regression's data
    design = ResampleDesign.__new__(ResampleDesign)
    design.x = x[units]
    design.failed = c[units] == 0
    design.tl, design.tr = t[:, 0], t[:, 1]
    design.truncated = bool(
        np.any(np.isfinite(design.tl)) or np.any(np.isfinite(design.tr))
    )
    design._censoring_distribution()
    return design


def _fixed(model: Any) -> dict:
    """The parameters the user fixed, by name, at their values."""
    names = {i: name for name, i in model.param_map.items()}
    out = {}
    for i in sorted(model._user_fixed_idx()):
        name = names[i]
        if name == "gamma":
            out[name] = float(model.gamma)
        elif name == model.lfp_name:
            out[name] = float(model.lfp_p)
        elif name == "f0":
            out[name] = float(model.f0)
        else:
            out[name] = float(model.params[model.dist.param_map[name]])
    return out


def _refit_options(model: Any) -> dict:
    """The options to refit ``model``'s specification with."""
    options: dict[str, Any] = {
        "how": model.method,
        "offset": bool(model.offset),
        "lfp": bool(model.lfp),
        "zi": bool(model.zi),
    }
    fixed = _fixed(model)
    if fixed:
        options["fixed"] = fixed
    # No start at the estimate: a fit given ``init`` also starts from its
    # default start (principle 13), which doubled the time of a refit.
    return options


def bootstrap(model: Any, n_boot: int, random_state: Any = None) -> dict:
    """
    The goodness-of-fit statistics of ``model`` on its data and their
    parametric bootstrap p-values from ``n_boot`` refits (see
    :mod:`surpyval.diagnostics.univariate` for the scheme).

    Returns a dictionary: ``observed`` and ``p``, each by statistic
    (``"AD"``, ``"KS"``, ``"CvM"``); ``draws``, an array of the refits'
    statistics, one column each; ``failed``, the refits that raised and
    were left out; and ``n_boot``.

    Examples
    --------
    >>> import surpyval as sp
    >>> from surpyval.diagnostics.univariate import bootstrap
    >>> x = sp.Weibull.random(30, 10, 2, random_state=1)
    >>> out = bootstrap(sp.Weibull.fit(x), 20, random_state=2)
    >>> sorted(out["p"]), out["draws"].shape
    (['AD', 'CvM', 'KS'], (20, 3))
    """
    from surpyval.univariate.regression._bootstrap import (
        REFIT_ERRORS,
        check_n_boot,
    )
    from surpyval.utils.no_maximum import quiet_maximum_warnings
    from surpyval.utils.rng import as_generator

    n_boot = check_n_boot(n_boot)
    data = model.data
    observed = _statistics(model, data["x"], data["c"], data["n"], data["t"])
    design = _design(model)
    rng = as_generator(random_state)
    t = np.column_stack([design.tl, design.tr])
    lo, hi = _window(model, t)
    longest = float(np.max(design.x))
    options = _refit_options(model)
    keys = [key for _, key in TESTS]
    draws = []
    failed = 0
    for _ in range(n_boot):
        q = lo + rng.uniform(size=design.x.size) * (hi - lo)
        with np.errstate(all="ignore"):
            T = np.asarray(model.qf(q), dtype=float)
        C = design.censoring(rng)
        censored = ~(T <= C)
        x = np.where(censored, C, T)
        # A unit that never fails (a limited failure population) and is
        # not censored by the censoring distribution is censored at the
        # last time observed.
        x = np.where(np.isfinite(x), x, longest)
        kwargs: dict[str, Any] = {"c": censored.astype(int)}
        if design.truncated:
            kwargs["tl"] = design.tl
            kwargs["tr"] = design.tr
        try:
            with warnings.catch_warnings(), quiet_maximum_warnings():
                warnings.simplefilter("ignore")
                refit = model.dist.fit(x, **kwargs, **options)
                stats = _statistics(refit, x, kwargs["c"], None, t)
        except REFIT_ERRORS:
            failed += 1
            continue
        draws.append([stats[k] for k in keys])
    values = np.array(draws, dtype=float).reshape(len(draws), len(keys))
    p = {}
    for j, key in enumerate(keys):
        if len(draws) < n_boot / 2:
            p[key] = float("nan")
        else:
            extreme = np.sum(values[:, j] >= observed[key] - 1e-12)
            p[key] = float((1 + extreme) / (1 + len(draws)))
    return {
        "observed": observed,
        "p": p,
        "draws": values,
        "failed": failed,
        "n_boot": n_boot,
    }


def _boot(ctx: Context) -> "dict | None":
    """The bootstrap of the report, or None for data it cannot
    simulate."""
    if not _simulable(ctx.model):
        return None
    n_boot = DEFAULT_N_BOOT if ctx.n_boot is None else ctx.n_boot
    return ctx.cached(
        "bootstrap", lambda: bootstrap(ctx.model, n_boot, ctx.random_state)
    )


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------
def _label(model: Any) -> str:
    """The model's distribution and options, as a report names it."""
    extras = [
        name
        for name, on in (
            ("offset", model.offset),
            ("lfp", model.lfp),
            ("zi", model.zi),
        )
        if on
    ]
    if not extras:
        return model.dist.name
    return "{} ({})".format(model.dist.name, ", ".join(extras))


def _gof_rows(ctx: Context) -> list[dict]:
    boot = _boot(ctx)
    rows = []
    if boot is None:
        detail = (
            "not computed: the data are left- or interval-censored, and a "
            "resample would need the inspection times, which they do not "
            "record"
        )
        for test, key in TESTS:
            rows.append(
                {
                    "test": test + " (parametric bootstrap)",
                    "statistic": float("nan"),
                    "p": float("nan"),
                    "detail": detail,
                    "alternative": "",
                    "_key": key,
                }
            )
        return rows
    if boot["failed"]:
        ctx.notes.append(
            "{} of the {} bootstrap refits failed and were left out of the "
            "goodness-of-fit p-values.".format(boot["failed"], boot["n_boot"])
        )
    kept = boot["n_boot"] - boot["failed"]
    for test, key in TESTS:
        p = boot["p"][key]
        if not np.isfinite(p):
            detail = (
                "not computed: {} of the {} bootstrap refits failed".format(
                    boot["failed"], boot["n_boot"]
                )
            )
        elif p < 2 * ctx.alpha:
            detail = "the {} does not fit the data ({} refits)".format(
                _label(ctx.model), kept
            )
        else:
            detail = "no evidence against the {} ({} refits)".format(
                _label(ctx.model), kept
            )
        rows.append(
            {
                "test": test + " (parametric bootstrap)",
                "statistic": boot["observed"][key],
                "p": p,
                "detail": detail,
                "alternative": "",
                "_key": key,
            }
        )
    return rows


def _km(ctx: Context) -> tuple[np.ndarray, np.ndarray]:
    """The Kaplan-Meier estimate of the ``u`` (its jump points and the
    estimate after each), for the P-P plot."""
    data = ctx.model.data
    u = transform(
        ctx.model,
        np.asarray(data["x"], dtype=float),
        np.asarray(data["t"], dtype=float).reshape(-1, 2),
    )
    failed = np.asarray(data["c"]) == 0
    w = np.asarray(data["n"], dtype=float)
    times, inverse = np.unique(u, return_inverse=True)
    d = np.bincount(inverse, weights=w * failed, minlength=times.size)
    total = np.bincount(inverse, weights=w, minlength=times.size)
    at_risk = w.sum() - np.concatenate([[0.0], np.cumsum(total)[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.cumprod(1.0 - np.where(at_risk > 0, d / at_risk, 0.0))
    return times, 1.0 - S


def _gof_panels(ctx: Context) -> list[Panel]:
    boot = _boot(ctx)
    if boot is None:
        return []
    times, F = _km(ctx)

    def pp(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        ax.step(
            np.concatenate([[0.0], times]),
            np.concatenate([[0.0], F]),
            where="post",
            color=PALETTE[0],
            label="Kaplan-Meier",
        )
        ax.plot([0, 1], [0, 1], color=INK["secondary"], ls="--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("fitted CDF, u")
        ax.set_ylabel("estimated CDF of u")

    def histogram(ax: Any, options: Any) -> None:
        from surpyval.plotting import PALETTE

        ax.hist(boot["draws"][:, 0], bins=30, color=PALETTE[0], alpha=0.6)
        ax.axvline(boot["observed"]["AD"], color=PALETTE[1], label="observed")
        ax.set_xlabel("Anderson-Darling statistic (observed: line)")
        ax.set_ylabel("refits")

    return [
        Panel("P-P plot (AD p = {})".format(_fmt(boot["p"]["AD"])), pp),
        Panel("bootstrap null distribution (AD)", histogram),
    ]


# ---------------------------------------------------------------------------
# The probability plot
# ---------------------------------------------------------------------------
def _probability_panels(ctx: Context) -> list[Panel]:
    dist = ctx.model.dist
    if not (
        hasattr(dist, "mpp_y_transform")
        and hasattr(dist, "mpp_inv_y_transform")
    ):
        ctx.notes.append(
            "The {} has no probability plot (no linearising axes); "
            "d.plot('goodness_of_fit') draws its P-P plot.".format(dist.name)
        )
        return []

    def draw(ax: Any, options: Any) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx.model.plot(ax=ax)

    return [Panel("{} probability plot".format(_label(ctx.model)), draw)]


# ---------------------------------------------------------------------------
# Alternatives
# ---------------------------------------------------------------------------
def _xcnt(model: Any) -> dict:
    data = model.data
    return {k: data[k] for k in ("x", "c", "n", "t")}


def _quiet(fit: Any) -> Any:
    """``fit()``, without its warnings; None if it raised or its
    likelihood has no finite maximum. A fit at an unverified maximum (a
    limited-failure proportion at 1, say) is kept: its likelihood is at
    most the maximum's, so its AIC can only understate how much better
    the alternative is."""
    from surpyval.utils.no_maximum import quiet_maximum_warnings

    try:
        with warnings.catch_warnings(), quiet_maximum_warnings():
            warnings.simplefilter("ignore")
            out = fit()
            aic = float(out.aic())
    except Exception:  # an alternative that does not fit is not one
        return None
    if getattr(out, "maximum", "verified") == "no finite maximum":
        return None
    if not np.isfinite(aic):
        return None
    return out


def _baseline_aic(ctx: Context) -> float:
    """The model's AIC; for a fit not by maximum likelihood, that of the
    same model fitted by it (an AIC compares maxima)."""
    model = ctx.model

    def compute() -> float:
        if model.method == "MLE":
            return float(model.aic())
        options = dict(_refit_options(model), how="MLE")
        mle = _quiet(lambda: model.dist.fit(**_xcnt(model), **options))
        if mle is None:
            return float(model.aic())
        ctx.notes.append(
            "The model was fitted by {}; the alternatives are compared "
            "with its maximum-likelihood fit (AIC {}).".format(
                model.method, _fmt(float(mle.aic()))
            )
        )
        return float(mle.aic())

    return ctx.cached("baseline_aic", compute)


def _lifetime(dist: Any) -> bool:
    """Whether the family's support is (0, inf): it takes an offset and
    zero inflation."""
    support = np.asarray(getattr(dist, "support", (np.nan, np.nan)), float)
    return bool(support[0] == 0 and support[1] == np.inf)


def _alternatives(ctx: Context) -> list[dict]:
    """Each alternative fitted: ``kind``, its ``label``, the distribution
    name the advice names (``dist``) and its ``aic``."""
    from surpyval import MixtureModel, fit_best
    from surpyval.fit_best import distributions

    model = ctx.model
    data = _xcnt(model)
    c = np.asarray(data["c"])
    x = np.asarray(data["x"], dtype=float)
    out = []

    # Every other distribution, as fit_best ranks them
    def ranked() -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fit_best(**data, return_table=True)[1]

    try:
        table = ranked()
    except Exception:  # no candidate fits these data
        table = None
    if table is not None:
        others = table[
            table["status"].isin(["chosen", "ranked"])
            & (table["model"] != model.dist.name)
        ]
        if len(others):
            best = others.iloc[0]
            out.append(
                {
                    "kind": "distribution",
                    "test": "AIC: best other distribution (fit_best)",
                    "label": str(best["model"]),
                    "dist": str(best["model"]),
                    "aic": float(best["aic"]),
                }
            )

    zeros = (c == 0) & (x.reshape(c.size, -1)[:, 0] == 0)
    plain = not (model.offset or model.lfp or model.zi or _fixed(model))
    # Not with failures at 0: a component collapses onto them, a
    # likelihood with no finite maximum (and a slow search for it)
    failures = float(np.asarray(data["n"])[c != 1].sum())
    if plain and not np.any(zeros) and failures >= MIXTURE_MIN_FAILURES:
        mix = _quiet(lambda: MixtureModel.fit(**data, dist=model.dist, m=2))
        if mix is not None:
            out.append(
                {
                    "kind": "mixture",
                    "test": "AIC: mixture of two",
                    "label": "MixtureModel({}, 2)".format(model.dist.name),
                    "dist": model.dist.name,
                    "aic": float(mix.aic()),
                }
            )

    base = dict(_refit_options(model), how="MLE")

    def variant(**change: Any) -> Any:
        return _quiet(lambda: model.dist.fit(**data, **dict(base, **change)))

    if not model.offset and _lifetime(model.dist) and not model.zi:
        alt = variant(offset=True)
        if alt is not None:
            out.append(
                {
                    "kind": "offset",
                    "test": "AIC: offset (3-parameter)",
                    "label": "{} offset".format(model.dist.name),
                    "dist": model.dist.name,
                    "aic": float(alt.aic()),
                }
            )
    if not model.lfp and np.any(c == 1):
        alt = variant(lfp=True)
        if alt is not None:
            out.append(
                {
                    "kind": "lfp",
                    "test": "AIC: limited failure population",
                    "label": "{} lfp".format(model.dist.name),
                    "dist": model.dist.name,
                    "aic": float(alt.aic()),
                }
            )
    if not model.zi and np.any(zeros):
        if _lifetime(model.dist):
            candidates = [model.dist]
        else:
            candidates = [d for d in distributions if _lifetime(d)]
        best_zi = None
        for dist in candidates:
            alt = _quiet(lambda: dist.fit(**data, zi=True))
            if alt is not None and (
                best_zi is None or alt.aic() < best_zi.aic()
            ):
                best_zi = alt
        if best_zi is not None:
            out.append(
                {
                    "kind": "zi",
                    "test": "AIC: zero-inflated",
                    "label": "{} zi".format(best_zi.dist.name),
                    "dist": best_zi.dist.name,
                    "aic": float(best_zi.aic()),
                }
            )
    return out


def _alternative_rows(ctx: Context) -> list[dict]:
    model_aic = _baseline_aic(ctx)
    rows = []
    for alt in ctx.cached("alternatives", lambda: _alternatives(ctx)):
        delta = model_aic - alt["aic"]
        better = delta >= AIC_MARGIN
        if better:
            detail = "{}'s AIC is {} lower".format(alt["label"], _fmt(delta))
        elif delta > 0:
            detail = "{}'s AIC is {} lower, under {:g}".format(
                alt["label"], _fmt(delta), AIC_MARGIN
            )
        else:
            detail = "{}'s AIC is {} higher".format(alt["label"], _fmt(-delta))
        rows.append(
            {
                "test": alt["test"],
                "alternative": alt["label"],
                "statistic": delta,
                "verdict": "warn" if better else "ok",
                "detail": detail,
                "_kind": alt["kind"],
                "_dist": alt["dist"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Refusals and notes
# ---------------------------------------------------------------------------
def _validate(model: Any) -> None:
    from surpyval.univariate.parametric import OptimisedFitMixin

    dist = model.dist
    name = getattr(dist, "name", type(dist).__name__)
    if name in ("Bernoulli", "FixedEventProbability"):
        raise ValueError(
            "diagnose() checks a distribution of times, and a {} model has "
            "none: its one parameter is the event probability p, which "
            "fits the proportion of events exactly. Bound it with "
            "param_cb('p') instead.".format(name)
        )
    if name == "ExactEventTime":
        raise ValueError(
            "diagnose() checks a distribution of times, and an "
            "ExactEventTime model is a point mass, fitted to the bracket "
            "of censored times around it; there is no shape to test."
        )
    if getattr(dist, "discrete", False):
        raise NotImplementedError(
            "diagnose() is not yet available for a discrete distribution "
            "({}): the goodness-of-fit tests of a discrete model (on its "
            "counts) are tracked in #875. Compare discrete models with "
            "model.aic() meanwhile.".format(name)
        )
    if not isinstance(dist, OptimisedFitMixin):
        raise NotImplementedError(
            "diagnose() is not yet available for a {} model: its "
            "bootstrap needs the refit of the shared optimiser, which the "
            "{} fitter does not have; tracked in #875.".format(name, name)
        )
    if getattr(model, "data", None) is None:
        raise ValueError(
            "diagnose() needs the data the model was fitted to, which this "
            "model does not have (it was built from parameters, or "
            "restored from a dict saved without its data). Refit it, or "
            "save it with to_dict(with_data=True) to keep them."
        )
    if getattr(model, "maximum", None) == "no finite maximum":
        raise ValueError(
            "diagnose() needs a fit with a finite maximum, and this one "
            "has none (model.maximum is 'no finite maximum', as the fit "
            "warned), so the fitted distribution means nothing to test. "
            "Fit another distribution (fit_best ranks them), or drop the "
            "option that runs off (an offset, say)."
        )


def _notes(ctx: Context) -> list[str]:
    boot = _boot(ctx)
    notes = []
    if boot is not None:
        notes.append(
            "Goodness of fit: the Anderson-Darling, Kolmogorov-Smirnov and "
            "Cramer-von Mises statistics of the Kaplan-Meier estimate of "
            "the fitted CDF at each time (conditioned on its truncation) "
            "against the uniform, with p-values from {} parametric "
            "bootstrap refits: each resample is drawn from the model "
            "within each unit's truncation window, censored as the data "
            "were (a censored unit at its own time, a failed one at a time "
            "drawn from the censoring distribution beyond its failure) and "
            "refitted the same way. diagnose(n_boot=..., random_state=...) "
            "sets their number and seed.".format(boot["n_boot"])
        )
    notes.append(
        "Alternatives are compared by AIC, a screen: a row warns when an "
        "alternative's AIC is at least {:g} below the model's. An offset "
        "is a threshold, whose likelihood is not regular, so its AIC is "
        "a guide only.".format(AIC_MARGIN)
    )
    data = ctx.model.data
    c = np.asarray(data["c"])
    x = np.asarray(data["x"], dtype=float).reshape(c.size, -1)[:, 0]
    n = np.asarray(data["n"])
    # (failures at time 0 are the zero-inflation row's to judge)
    positive = (c == 0) & (x != 0)
    tied = int(n[positive].sum() - np.unique(x[positive]).size)
    if boot is not None and tied > 0:
        notes.append(
            "{} failure{} tie with another; the bootstrap's samples from "
            "the continuous model have no ties, so heavily rounded times "
            "can fail the goodness-of-fit tests for the rounding "
            "alone.".format(tied, "" if tied == 1 else "s")
        )
    return notes


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------
def _best(row: dict, ctx: Context) -> str:
    """A sentence for the alternative that fits best of those that warn,
    when more than one does."""
    warned = [r for r in _alternative_rows(ctx) if r["verdict"] == "warn"]
    if len(warned) < 2:
        return ""
    top = max(warned, key=lambda r: r["statistic"])
    if top["alternative"] != row["alternative"]:
        return ""
    return " Of the alternatives tried, this one fits best."


def _call(row: dict) -> str:
    return "surpyval.{}".format(row["_dist"])


def _distribution_message(row: dict, ctx: Context) -> str:
    return (
        "{alt} fits these data better than the {model} (its AIC is {d} "
        "lower). fit_best(x, c, n, t, return_table=True) fits and ranks "
        "every distribution, with the AIC of each; refit the one it "
        "chooses and diagnose it.".format(
            alt=row["_dist"], model=_label(ctx.model), d=_fmt(row["statistic"])
        )
        + _best(row, ctx)
    )


def _mixture_message(row: dict, ctx: Context) -> str:
    return (
        "A mixture of two {name} distributions fits better than one (its "
        "AIC is {d} lower): the data may come from two populations, or "
        "two failure modes. Fit it with MixtureModel.fit(x, c, n, t, "
        "dist={call}, m=2), or compare it with the single distributions "
        "in fit_best(x, c, n, t, include=[..., MixtureModel({call}, "
        "2)]). If the failure modes are recorded, fit each with the "
        "competing-risks models instead.".format(
            name=row["_dist"], d=_fmt(row["statistic"]), call=_call(row)
        )
        + _best(row, ctx)
    )


def _offset_message(row: dict, ctx: Context) -> str:
    return (
        "An offset (a 3-parameter {name}, with no failures before a "
        "threshold time) fits better (its AIC is {d} lower). Fit it with "
        "{name}.fit(x, c, n, t, offset=True); param_cb('gamma', "
        "method='bootstrap') bounds the offset.".format(
            name=row["_dist"], d=_fmt(row["statistic"])
        )
        + _best(row, ctx)
    )


def _lfp_message(row: dict, ctx: Context) -> str:
    return (
        "A limited failure population (a share of the units that never "
        "fails) fits better (its AIC is {d} lower). Fit it with "
        "{name}.fit(x, c, n, t, lfp=True); its lfp_p is the share that "
        "fails.".format(name=row["_dist"], d=_fmt(row["statistic"]))
        + _best(row, ctx)
    )


def _zi_message(row: dict, ctx: Context) -> str:
    return (
        "Some units failed at time 0, and a zero-inflated {name} (a share "
        "of the units failing at once, the rest by the distribution) fits "
        "better (its AIC is {d} lower). Fit it with {name}.fit(x, c, n, "
        "t, zi=True); its f0 is the share failing at 0.".format(
            name=row["_dist"], d=_fmt(row["statistic"])
        )
        + _best(row, ctx)
    )


def _unexplained(row: dict, ctx: Context) -> bool:
    """A failed goodness-of-fit test that no alternative accounts for."""
    rows = _alternative_rows(ctx)
    return not any(r["verdict"] == "warn" for r in rows)


def _gof_message(row: dict, ctx: Context) -> str:
    return (
        "The {model} does not fit the data (the {test} test fails) and "
        "none of the alternatives tried fits better by AIC. Look at the "
        "probability plot, d.plot('probability_plot'), for where the "
        "points leave the line; try more components, MixtureModel.fit(x, "
        "c, n, t, dist=..., m=3), or a non-parametric estimate, "
        "KaplanMeier.fit (Turnbull.fit for interval-censored data), "
        "which assumes no distribution.".format(
            model=_label(ctx.model), test=row["test"].split(" (")[0]
        )
    )


def _rules(
    kind: str, message: Any, feature: str, *extra: str
) -> list[AdviceRule]:
    """One rule per named distribution (so that each names its own
    ``fit``), and one, naming no ``fit``, for any other distribution."""

    def of(name: str) -> Any:
        return lambda row, ctx: row["_kind"] == kind and row["_dist"] == name

    rules = [
        AdviceRule(
            "alternatives",
            message,
            (feature.format(name), *extra),
            when=of(name),
            on=("warn",),
        )
        for name in NAMED
    ]
    rules.append(
        AdviceRule(
            "alternatives",
            message,
            extra,
            when=lambda row, ctx: row["_kind"] == kind
            and row["_dist"] not in NAMED,
            on=("warn",),
        )
    )
    return rules


def _is_parametric(model: Any) -> bool:
    from surpyval.univariate.parametric import Parametric

    return isinstance(model, Parametric)


#: The univariate family: its checks, in the order of the table, and its
#: advice.
UNIVARIATE = register(
    Family(
        name="Univariate distribution",
        issue=875,
        applies=_is_parametric,
        validate=_validate,
        covariates=lambda model: [],
        columns=("alternative",),
        checks=(
            Check(
                "goodness_of_fit",
                "The data are a sample from the fitted distribution.",
                rows=_gof_rows,
                panels=_gof_panels,
            ),
            Check(
                "probability_plot",
                "The failures fall on the fitted line of the probability "
                "plot.",
                panels=_probability_panels,
            ),
            Check(
                "alternatives",
                "No other model the advice can name fits the data "
                "substantially better by AIC.",
                rows=_alternative_rows,
            ),
        ),
        advice=(
            AdviceRule(
                "alternatives",
                _distribution_message,
                ("fit_best(return_table=)",),
                when=lambda row, ctx: row["_kind"] == "distribution",
                on=("warn",),
            ),
            AdviceRule(
                "alternatives",
                _mixture_message,
                (
                    "MixtureModel.fit(dist=)",
                    "MixtureModel.fit(m=)",
                    "fit_best(include=)",
                ),
                when=lambda row, ctx: row["_kind"] == "mixture",
                on=("warn",),
            ),
            *_rules(
                "offset",
                _offset_message,
                "{}.fit(offset=)",
                "param_cb(method=)",
            ),
            *_rules("lfp", _lfp_message, "{}.fit(lfp=)"),
            *_rules("zi", _zi_message, "{}.fit(zi=)"),
            AdviceRule(
                "goodness_of_fit",
                _gof_message,
                ("MixtureModel.fit(m=)", "KaplanMeier.fit", "Turnbull.fit"),
                when=_unexplained,
            ),
        ),
        notes=_notes,
    )
)
