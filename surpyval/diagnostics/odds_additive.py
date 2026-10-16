"""``diagnose()`` for the proportional odds and additive hazards models
(#870).

Two families of :mod:`surpyval.diagnostics`, registered at the bottom:

- **Proportional odds** (:data:`PROPORTIONAL_ODDS_FAMILY`): the
  semi-parametric ``ProportionalOdds`` (Murphy, Rossini and van der Vaart
  1997) and the parametric ``PO(...)`` models (``WeibullPO``,
  ``LogNormalPO``, ...).
- **Additive hazards** (:data:`ADDITIVE_HAZARDS_FAMILY`): Lin and Ying's
  semi-parametric ``AdditiveHazards`` and the parametric ``AH(...)``
  models (``WeibullAH``, ``ExponentialAH``, ...).

Both ask the same question of each covariate -- does its effect stay the
same over follow-up? -- with the same added term, so they share
everything above the registrations.

The test
--------
Follow-up is split into (up to) :data:`PERIODS` periods at the quartiles
of the event times (each cut an event time, events at a cut in the
period before it), and ``g(t)``, the period a time is in (0, 1, 2, 3), is
the clock. For each covariate the model is refitted with the added term
``theta * u(Z) * g(t)``, where ``u(Z)`` is the covariate less its mean
(one column; for a covariate with two to five values one indicator per
level other than the first, less its mean), and the test is of ``theta =
0``: a coefficient that is the same in every period. The extended model
has a covariate that changes at the cut times, and every one of these
estimators handles that exactly by episode splitting -- each row becomes
one row per period it spans, entering (left-truncated) at the start of
the period -- so the refit is the model's own estimator on the split
rows, with no approximation:

- for the proportional odds models, and the parametric additive hazards
  ones, the **likelihood-ratio test** of the two maximum likelihood fits
  (the semi-parametric proportional odds fit's profile likelihood, as
  Murphy and van der Vaart 2000 justify), ``chi2`` with one degree of
  freedom per added column;
- for Lin and Ying's model, which has no likelihood, the **Wald test** of
  ``theta`` from the closed-form estimating equations with the
  time-varying term, with Lin and Ying's sandwich variance, which is what
  ``timereg::aalen`` with ``const()`` terms gives on the split rows (the
  tests compare the two).

This is the test of a time-by-covariate interaction with a step function
of time, the likelihood-ratio (or Wald) counterpart of the score test
that ``cox.zph`` makes for the Cox model, with the period index as its
time transform. A step clock is used so that the extended model is an
exact fit of the family's own estimator, and its four periods set by the
event times make the test the same whatever the unit of time. The joint
``GLOBAL`` row adds every covariate's term at once.

``u(Z)`` is centred because the proportional odds and the parametric
additive models cannot absorb a function of time alone into their
baseline, so the term ``Z * g(t)`` and ``(Z - mean) * g(t)`` are
different alternatives; centred, the added term leaves an average unit's
hazard as it was. (Lin and Ying's baseline absorbs it either way.)

The checks
----------
``proportional_odds``
    For a binary or categorical covariate (two to five values), whether
    the log-odds of failure of its groups stay parallel; for a continuous
    one, whether its coefficient trends over the periods. Drawn as the
    Kaplan-Meier log-odds of failure, ``log(F / S)``, of each group
    against time on a log scale (parallel curves under proportional
    odds), and, for a continuous covariate, its coefficient in each
    period (refitted with one column per period) with ``1 - alpha``
    Wald intervals, against the fitted constant.
``constant_effects``
    For each covariate, whether the amount it adds to the hazard stays
    the same over the periods. Drawn as Aalen's nonparametric cumulative
    regression coefficient ``B_j(t)`` (Aalen's least-squares estimator,
    as ``timereg::aalen``, with its pointwise ``1 - alpha`` band from the
    optional-variation variance) against the straight line ``beta_j t``
    of the fitted model: a constant effect has a straight cumulative
    coefficient.

A covariate with two distinct values is *binary*, one with three to five
(:data:`~surpyval.diagnostics.cox.CATEGORICAL_LEVELS`) *categorical* and
one with more *continuous*, as for the Cox family.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.stats import chi2, norm

from .cox import CATEGORICAL_LEVELS, _fmt
from .registry import AdviceRule, Check, Context, Family, Panel, register

#: The number of periods follow-up is split into for the trend tests, at
#: the quantiles of the event times.
PERIODS = 4

#: The pre-built parametric fitters, by baseline distribution: the advice
#: names ``<name>PH``, ``<name>AFT`` and ``<name>AH``, which exist for
#: each; for any other baseline it names ``PH(...)``, ``AFT(...)`` and
#: ``AH(...)``.
PREBUILT = (
    "Exponential",
    "Normal",
    "Weibull",
    "Gumbel",
    "Logistic",
    "LogNormal",
    "Gamma",
)


# ---------------------------------------------------------------------------
# What the models give
# ---------------------------------------------------------------------------
def _is_parametric(model: Any) -> bool:
    from surpyval.univariate.regression import ParametricRegressionModel

    return isinstance(model, ParametricRegressionModel)


def _is_lin_ying(model: Any) -> bool:
    from surpyval.univariate.regression import AdditiveHazardsModel

    return isinstance(model, AdditiveHazardsModel)


def _coefficients(model: Any) -> np.ndarray:
    """The covariate coefficients, as ``summary()`` lists them."""
    if _is_parametric(model):
        return np.asarray(model.phi_params, dtype=float)
    return np.asarray(model.beta, dtype=float)


def _names(model: Any) -> list[str]:
    """The covariates' names, as ``summary()`` names the coefficients."""
    if _is_parametric(model):
        return list(model.parameter_names[model.k_dist :])
    return list(model.parameter_names)


def _data(ctx: Context) -> dict:
    """The rows fitted: ``x``, ``c``, ``n``, the entry times ``tl``
    (``-inf`` for none), the covariates ``Z`` as given and the point
    ``center`` the baseline is at (zeros unless fitted with
    ``center=True``)."""

    def compute() -> dict:
        model = ctx.model
        if _is_parametric(model):
            data = model.data
            x, c, n = data.x, data.c, data.n
            tl = np.asarray(data.t, dtype=float)[:, 0]
            # An entry at or before the start of the baseline's support is
            # no truncation (as fit_tvc takes it), and costs the fit time.
            low = float(getattr(model.model, "support", (-np.inf,))[0])
            if low > -np.inf:
                tl = np.where(tl <= low, -np.inf, tl)
            Z = data.Z
            center = getattr(model, "center", None)
        elif _is_lin_ying(model):
            x, c, n, Z = model._fit_data
            tl = np.full(np.shape(x), -np.inf)
            center = None
        else:
            fit = model._fit_data
            x, c, n, Z = fit["x"], fit["c"], fit["n"], fit["Z"]
            tl = fit["tl"]
            tl = np.full(np.shape(x), -np.inf) if tl is None else tl
            center = None
        Z = np.asarray(Z, dtype=float).reshape(np.size(x), -1)
        center = (
            np.zeros(Z.shape[1])
            if center is None
            else np.asarray(center, dtype=float)
        )
        return {
            "x": np.asarray(x, dtype=float),
            "c": np.asarray(c, dtype=int),
            "n": np.asarray(n, dtype=float),
            "tl": np.asarray(tl, dtype=float),
            "Z": Z,
            "center": center,
        }

    return ctx.cached("data", compute)


def _kind(ctx: Context, j: int) -> str:
    """``"binary"``, ``"categorical"`` or ``"continuous"``, by the number
    of distinct values of covariate ``j``."""
    levels = np.unique(_data(ctx)["Z"][:, j]).size
    if levels <= 2:
        return "binary"
    if levels <= CATEGORICAL_LEVELS:
        return "categorical"
    return "continuous"


def _cuts(ctx: Context) -> np.ndarray:
    """The cut times: the ``k / PERIODS`` quantiles of the event times
    (weighted by the counts; each the first event time at which that
    share of the events has happened), without repeats or a cut at or
    after the last event time."""

    def compute() -> np.ndarray:
        d = _data(ctx)
        events = d["c"] == 0
        times, at = np.unique(d["x"][events], return_inverse=True)
        if times.size < 2:
            return np.zeros(0)
        weight = np.bincount(at, weights=d["n"][events], minlength=times.size)
        share = np.cumsum(weight) / weight.sum()
        q = np.arange(1, PERIODS) / PERIODS
        cuts = np.unique(times[np.searchsorted(share, q - 1e-12)])
        return cuts[cuts < times[-1]]

    return ctx.cached("cuts", compute)


def _base(ctx: Context, j: int) -> np.ndarray:
    """``u(Z)`` of covariate ``j``, one column per added term: the
    covariate less its (count-weighted) mean, or, for a categorical
    covariate (three to five values), the indicator of each level after
    the first, less its mean. (For a binary covariate the two are the
    same term, up to a scale the tests do not see.)"""
    d = _data(ctx)
    z = d["Z"][:, j]
    n = d["n"]
    if _kind(ctx, j) == "categorical":
        levels = np.unique(z)[1:]
        U = (z[:, None] == levels[None, :]).astype(float)
    else:
        U = z[:, None]
    return U - (n @ U) / n.sum()


def _split(ctx: Context) -> dict:
    """Each row as one row per period it spans: ``rows`` (the row it came
    from), its entry ``tl`` and exit ``x``, its status ``c`` (censored
    at the end of a period it outlives) and its period ``g``."""

    def compute() -> dict:
        d = _data(ctx)
        cuts = _cuts(ctx)
        lo = np.concatenate([[-np.inf], cuts])
        hi = np.concatenate([cuts, [np.inf]])
        parts: dict[str, list] = {k: [] for k in ("rows", "tl", "x", "c", "g")}
        for k in range(lo.size):
            a = np.maximum(d["tl"], lo[k])
            b = np.minimum(d["x"], hi[k])
            keep = np.flatnonzero(a < b)
            parts["rows"].append(keep)
            parts["tl"].append(a[keep])
            parts["x"].append(b[keep])
            parts["c"].append(np.where(d["x"][keep] <= hi[k], d["c"][keep], 1))
            parts["g"].append(np.full(keep.size, float(k)))
        out = {k: np.concatenate(v) for k, v in parts.items()}
        out["n"] = d["n"][out["rows"]]
        return out

    return ctx.cached("split", compute)


# ---------------------------------------------------------------------------
# The refits
# ---------------------------------------------------------------------------
class _NotComputed(Exception):
    """Why a test could not be computed (its row's detail)."""


def _fit_split(
    ctx: Context,
    design: np.ndarray,
    start: "np.ndarray | None" = None,
    where: "list[int | None] | None" = None,
) -> Any:
    """The model's own estimator on the split rows with covariates
    ``design`` (one row per split row, the model's covariates less its
    centre and then the added columns): the semi-parametric proportional
    odds fit, or the parametric fit started at the fitted model, which
    is the extended model's with the added coefficients at 0 (or at
    ``start``, the coefficients to start from, for another layout, in
    which ``where[j]`` is the column of the model's covariate ``j``)."""
    s = _split(ctx)
    model = ctx.model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if _is_parametric(model):
                beta = _coefficients(model)
                if start is None:
                    start = np.concatenate(
                        [beta, np.zeros(design.shape[1] - beta.size)]
                    )
                init = np.concatenate(
                    [np.asarray(model.params[: model.k_dist], float), start]
                )
                entry = np.column_stack(
                    [s["tl"], np.full(s["x"].size, np.inf)]
                )
                fit = model.model.fit(
                    x=s["x"],
                    Z=design,
                    c=s["c"],
                    n=s["n"],
                    t=entry,
                    init=init,
                    fixed=_fixed(model, where) or None,
                )
            else:
                from surpyval.univariate.regression import ProportionalOdds

                fit = ProportionalOdds.fit(
                    s["x"], design, c=s["c"], n=s["n"], tl=s["tl"], center=True
                )
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as e:
            raise _NotComputed(
                "the model with the added terms could not be fitted "
                "({})".format(str(e).split(".")[0])
            )
    if getattr(fit, "maximum", None) == "no finite maximum":
        raise _NotComputed(
            "the model with the added terms has no finite maximum"
        )
    if not np.all(np.isfinite(_coefficients(fit))):
        raise _NotComputed(
            "the added terms are aliased (a period has too few events)"
        )
    if (
        _is_parametric(model)
        and fit.maximum != "verified"
        and _UNVERIFIED not in ctx.notes
    ):
        ctx.notes.append(_UNVERIFIED)
    return fit


_UNVERIFIED = (
    "A parametric refit behind the trend tests did not reach a verified "
    "maximum (as the additive hazards fits often end on the boundary of "
    "their support); its likelihood-ratio test is approximate."
)


def _fixed(model: Any, where: "list[int | None] | None" = None) -> dict:
    """The model's fixed parameters, named as a fit from arrays names
    them: ``coef_k`` for the coefficient of the model's covariate ``j``,
    ``k = where[j]`` (``j`` by default); ``None`` in ``where`` leaves it
    free."""
    out = {}
    names = list(model.parameter_names)
    for name, value in dict(getattr(model, "fixed", None) or {}).items():
        if name not in names:
            continue
        j = names.index(name) - model.k_dist
        if j < 0:
            out[name] = value
            continue
        k = j if where is None else where[j]
        if k is not None:
            out["coef_{}".format(k)] = value
    return out


def _covariates_split(ctx: Context) -> np.ndarray:
    """The model's covariates on the split rows, less its ``center``."""
    d = _data(ctx)
    return (d["Z"] - d["center"])[_split(ctx)["rows"]]


def _likelihood_ratio(ctx: Context, base: np.ndarray) -> dict:
    s = _split(ctx)
    design = np.column_stack(
        [_covariates_split(ctx), base[s["rows"]] * s["g"][:, None]]
    )
    fit = _fit_split(ctx, design)
    model = ctx.model
    if _is_parametric(model):
        null, full = -float(model._neg_ll), -float(fit._neg_ll)
    else:
        null, full = float(model.log_likelihood), float(fit.log_likelihood)
    # Nested at theta = 0: the alternative is at least as likely.
    statistic = max(0.0, 2.0 * (full - null))
    theta = _coefficients(fit)[-base.shape[1] :]
    return {"statistic": statistic, "df": base.shape[1], "theta": theta}


def _lin_ying(ctx: Context, base: np.ndarray) -> dict:
    """The Wald test of ``theta`` in Lin and Ying's model with the added
    term ``theta' u(Z) g(t)``, ``g`` constant on each interval of the
    risk-set grid (the period of its right end), solved in closed form
    with Lin and Ying's sandwich variance."""
    d = _data(ctx)
    x, c, n, Z = d["x"], d["c"], d["n"], d["Z"]
    U = np.column_stack([Z, base])
    p, q = Z.shape[1], U.shape[1]
    grid = np.unique(x)
    m = grid.size
    bucket = np.searchsorted(grid, x)
    # The period of the interval (grid[j - 1], grid[j]]: the cuts before
    # its right end.
    g = np.searchsorted(_cuts(ctx), grid, side="left").astype(float)
    scale = np.ones((m, q))
    scale[:, p:] = g[:, None]

    def at_risk(values: np.ndarray) -> np.ndarray:
        out = np.zeros((m,) + values.shape[1:])
        np.add.at(out, bucket, values)
        return np.cumsum(out[::-1], axis=0)[::-1]

    S0 = at_risk(n)
    S1 = at_risk(n[:, None] * U)
    S2 = at_risk(n[:, None, None] * U[:, :, None] * U[:, None, :])
    V = S2 - S1[:, :, None] * S1[:, None, :] / S0[:, None, None]
    V = V * scale[:, :, None] * scale[:, None, :]
    widths = np.diff(np.concatenate([[0.0], grid]))
    A = (V * widths[:, None, None]).sum(axis=0)
    events = n * (c == 0)
    R = (U - (S1 / S0[:, None])[bucket]) * scale[bucket]
    b = events @ R
    B = np.einsum("i,ij,ik->jk", events, R, R)
    try:
        A_inv = np.linalg.inv(A)
        estimate = A_inv @ b
        cov = A_inv @ B @ A_inv
        theta = estimate[p:]
        statistic = float(theta @ np.linalg.solve(cov[p:, p:], theta))
    except np.linalg.LinAlgError:
        raise _NotComputed(
            "the added terms are aliased (a period has too few events)"
        )
    if not np.isfinite(statistic):
        raise _NotComputed("the added terms are aliased")
    return {"statistic": statistic, "df": base.shape[1], "theta": theta}


def _trend(ctx: Context, key: str, base: np.ndarray) -> dict:
    """The test of the added terms ``base * g(t)`` (computed once per
    report): ``statistic``, ``df``, ``p``, the added coefficients
    ``theta`` and, when it could not be computed, ``why``."""

    def compute() -> dict:
        try:
            if _cuts(ctx).size == 0:
                raise _NotComputed(
                    "too few distinct event times to split follow-up into "
                    "periods"
                )
            if (
                _is_parametric(ctx.model)
                and ctx.model._is_additive()
                and _cuts(ctx)[0] <= 0
            ):
                raise _NotComputed(
                    "a cut time is at or before 0, where an additive "
                    "hazards model starts its covariates' effect"
                )
            if _is_lin_ying(ctx.model):
                res = _lin_ying(ctx, base)
            else:
                res = _likelihood_ratio(ctx, base)
        except _NotComputed as e:
            return {
                "statistic": float("nan"),
                "df": base.shape[1],
                "p": float("nan"),
                "theta": np.full(base.shape[1], np.nan),
                "why": str(e),
            }
        res["p"] = float(chi2.sf(res["statistic"], res["df"]))
        return res

    return ctx.cached("trend_" + key, compute)


def _test_name(ctx: Context, kind: str) -> str:
    if _is_lin_ying(ctx.model):
        test = "Lin-Ying Wald test"
    else:
        test = "likelihood-ratio test"
    if _is_po(ctx.model) and kind != "continuous":
        return test + " of parallel log-odds across groups"
    if kind == "categorical":
        return test + " of trends in the group effects"
    return test + " of a coefficient trend"


def _trend_rows(ctx: Context) -> list[dict]:
    rows = []
    po = _is_po(ctx.model)
    for j, name in ctx.checked:
        kind = _kind(ctx, j)
        res = _trend(ctx, str(j), _base(ctx, j))
        p = res["p"]
        if "why" in res:
            detail = "the test could not be computed: " + res["why"]
        elif p < 2 * ctx.alpha:
            if res["df"] > 1:
                detail = "the gaps between its groups change with time"
            else:
                detail = "its coefficient {} with time".format(
                    "rises" if res["theta"][0] > 0 else "falls"
                )
            if po and kind != "continuous":
                detail = (
                    "its groups' log-odds of failure are not parallel: "
                    + detail
                )
        else:
            detail = "no evidence that its effect changes with time"
        rows.append(
            {
                "test": _test_name(ctx, kind),
                "covariates": (name,),
                "statistic": res["statistic"],
                "df": res["df"],
                "p": p,
                "detail": detail,
                "_kind": kind,
            }
        )
    every = list(range(len(ctx.names)))
    res = _trend(
        ctx, "global", np.column_stack([_base(ctx, j) for j in every])
    )
    rows.append(
        {
            "test": _test_name(ctx, "continuous").replace(
                "of a coefficient trend", "of every covariate's trend"
            )
            + " (GLOBAL)",
            "covariates": tuple(ctx.names),
            "statistic": res["statistic"],
            "df": res["df"],
            "p": res["p"],
            "detail": (
                "the test could not be computed: " + res["why"]
                if "why" in res
                else "the joint test of every covariate"
            ),
            "_global": True,
        }
    )
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _at_risk_steps(
    x: np.ndarray, c: np.ndarray, n: np.ndarray, tl: np.ndarray
) -> tuple:
    """The event times and the at-risk and event counts there (a row is
    at risk on ``(tl, x]``)."""
    events = c == 0
    times = np.unique(x[events])
    d = np.bincount(
        np.searchsorted(times, x[events]),
        weights=n[events],
        minlength=times.size,
    )
    xs, xo = np.sort(x), np.argsort(x)
    ts, to = np.sort(tl), np.argsort(tl)
    exited = np.concatenate([[0.0], np.cumsum(n[xo])])[
        np.searchsorted(xs, times, side="left")
    ]
    entered = np.concatenate([[0.0], np.cumsum(n[to])])[
        np.searchsorted(ts, times, side="left")
    ]
    return times, entered - exited, d


def _kaplan_meier(
    x: np.ndarray, c: np.ndarray, n: np.ndarray, tl: np.ndarray
) -> tuple:
    times, risk, d = _at_risk_steps(x, c, n, tl)
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.cumprod(1.0 - np.where(risk > 0, d / risk, 0.0))
    return times, S


def _log_odds_panel(ctx: Context, j: int, name: str) -> Panel:
    d = _data(ctx)
    z = d["Z"][:, j]
    res = _trend(ctx, str(j), _base(ctx, j))
    curves = []
    for level in np.unique(z):
        rows = z == level
        t, S = _kaplan_meier(
            d["x"][rows], d["c"][rows], d["n"][rows], d["tl"][rows]
        )
        keep = (S > 0) & (S < 1) & (t > 0)
        curves.append((level, t[keep], np.log((1 - S[keep]) / S[keep])))

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import PALETTE

        for k, (level, t, odds) in enumerate(curves):
            if t.size:
                ax.step(
                    t,
                    odds,
                    where="post",
                    color=PALETTE[k % len(PALETTE)],
                    label="{} = {:g}".format(name, level),
                )
        ax.set_xscale("log")
        ax.set_xlabel("time (log scale)")
        ax.set_ylabel("log-odds of failure, log(F / S)")
        ax.legend(fontsize="small")

    return Panel(
        "{}: log-odds by group, p = {}".format(name, _fmt(res["p"])), draw
    )


def _period_panel(ctx: Context, j: int, name: str) -> Panel:
    """Covariate ``j``'s coefficient in each period: the model refitted
    with one column per period holding the covariate (less the model's
    centre) there and 0 elsewhere."""
    s = _split(ctx)
    d = _data(ctx)
    design = _covariates_split(ctx)
    periods = int(s["g"].max()) + 1 if s["g"].size else 1
    zj = design[:, j]
    pieces = np.column_stack([zj * (s["g"] == k) for k in range(periods)])
    design = np.column_stack([np.delete(design, j, axis=1), pieces])
    res = _trend(ctx, str(j), _base(ctx, j))
    coefficients = _coefficients(ctx.model)
    beta = coefficients[j]
    # Started at the fitted model: beta_j in every period
    start = np.concatenate(
        [np.delete(coefficients, j), np.full(periods, beta)]
    )
    moved = [k - (k > j) if k != j else None for k in range(len(coefficients))]
    try:
        fit = _fit_split(ctx, design, start, moved)
        est = _coefficients(fit)[-periods:]
        se = np.asarray(fit.standard_errors(), dtype=float)[-periods:]
    except _NotComputed:
        est = se = np.full(periods, np.nan)
    events = d["c"] == 0
    t_ev = d["x"][events]
    g_ev = np.searchsorted(_cuts(ctx), t_ev, side="left")
    at = np.array(
        [
            np.median(t_ev[g_ev == k]) if np.any(g_ev == k) else np.nan
            for k in range(periods)
        ]
    )

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        z = norm.isf(options.alpha / 2)
        ax.errorbar(
            at,
            est,
            yerr=z * se,
            fmt="o",
            color=PALETTE[0],
            capsize=3,
            label="by period, {:g}% Wald".format(100 * (1 - options.alpha)),
        )
        ax.axhline(beta, color=PALETTE[1], ls="--", label="fitted")
        for cut in _cuts(ctx):
            ax.axvline(cut, color=INK["axis"], lw=0.8)
        ax.set_xlabel("time (median event time of each period)")
        ax.set_ylabel("coefficient in period")
        ax.legend(fontsize="small")

    return Panel(
        "{}: coefficient by period, p = {}".format(name, _fmt(res["p"])), draw
    )


def _po_panels(ctx: Context) -> list[Panel]:
    panels = []
    for j, name in ctx.checked:
        if _kind(ctx, j) == "continuous":
            panels.append(_period_panel(ctx, j, name))
        else:
            panels.append(_log_odds_panel(ctx, j, name))
    return panels


def aalen_cumulative_coefficients(
    x: Any, Z: Any, c: Any = None, n: Any = None, tl: Any = None
) -> dict:
    """
    Aalen's nonparametric additive hazards model, ``h(t | Z) = b_0(t) +
    b(t)' Z``: the least-squares estimate of the cumulative regression
    coefficients ``B(t) = int_0^t b(s) ds`` at the event times, with
    their optional-variation variance, as R's ``timereg::aalen`` (with
    ``robust = 0``) gives them as ``cum`` and ``var.cum``.

    At each event time the increment is ``(X' Y X)^{-1} X' dN``, ``X`` the
    rows at risk with a leading column of ones and ``Y`` their counts, and
    the variance's ``(X' Y X)^{-1} X' diag(dN) X (X' Y X)^{-1}``. The
    estimate stops before the first event time at which ``X' Y X`` is
    singular (too few rows at risk to separate the covariates).

    Parameters
    ----------
    x : array_like
        The exit times.
    Z : array_like
        The covariates, one row per observation.
    c, n, tl : array_like, optional
        The censoring flags (0 event, 1 right censored), counts and entry
        times (a row is at risk on ``(tl, x]``).

    Returns
    -------
    dict
        ``time`` (the event times), ``cum`` (``B(t)`` there, a column per
        covariate after the intercept's) and ``var`` (their variances).

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.diagnostics.odds_additive import (
    ...     aalen_cumulative_coefficients)
    >>> out = aalen_cumulative_coefficients(
    ...     [1, 2, 3, 4, 5, 6], [[0], [1], [0], [1], [0], [1]],
    ...     c=[0, 0, 0, 0, 1, 0])
    >>> out["time"], out["cum"].round(3)[:, 1]
    (array([1., 2., 3., 4.]), array([-0.333,  0.   , -0.5  ,  0.   ]))

    At time 1 the one event is in group 0, of three at risk in each group:
    the group-1 hazard is a third below group 0's. At time 6 only a row of
    group 1 is at risk, too few to separate the groups, so the estimate
    stops at 4.
    """
    x = np.asarray(x, dtype=float)
    Z = np.asarray(Z, dtype=float).reshape(x.size, -1)
    c = np.zeros(x.size, int) if c is None else np.asarray(c, dtype=int)
    n = np.ones(x.size) if n is None else np.asarray(n, dtype=float)
    tl = np.full(x.size, -np.inf) if tl is None else np.asarray(tl, float)
    X = np.column_stack([np.ones(x.size), Z])
    q = X.shape[1]
    events = c == 0
    times = np.unique(x[events])
    XX = n[:, None, None] * X[:, :, None] * X[:, None, :]

    def summed(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
        # sum of values over the rows with key >= each time
        order = np.argsort(keys)
        total = np.concatenate(
            [np.zeros((1,) + values.shape[1:]), np.cumsum(values[order], 0)]
        )
        return total[-1] - total[np.searchsorted(keys[order], times, "left")]

    # at risk at t: entered before t (tl < t) and not yet out (x >= t)
    S2 = summed(x, XX) - summed(tl, XX)
    bucket = np.searchsorted(times, x[events])
    E1 = np.zeros((times.size, q))
    np.add.at(E1, bucket, (n[events])[:, None] * X[events])
    E2 = np.zeros((times.size, q, q))
    np.add.at(E2, bucket, XX[events])
    dB, dV = [], []
    for k in range(times.size):
        if np.linalg.matrix_rank(S2[k]) < q:
            break
        inv = np.linalg.inv(S2[k])
        dB.append(inv @ E1[k])
        dV.append(np.diag(inv @ E2[k] @ inv))
    kept = len(dB)
    return {
        "time": times[:kept],
        "cum": np.cumsum(np.reshape(dB, (kept, q)), axis=0),
        "var": np.cumsum(np.reshape(dV, (kept, q)), axis=0),
    }


def _aalen(ctx: Context) -> dict:
    def compute() -> dict:
        d = _data(ctx)
        return aalen_cumulative_coefficients(
            d["x"], d["Z"], c=d["c"], n=d["n"], tl=d["tl"]
        )

    return ctx.cached("aalen", compute)


def _ah_panels(ctx: Context) -> list[Panel]:
    fit = _aalen(ctx)
    beta = _coefficients(ctx.model)
    panels = []
    for j, name in ctx.checked:
        res = _trend(ctx, str(j), _base(ctx, j))

        def draw(ax: Any, options: Any, j: int = j) -> None:
            from surpyval.plotting import BAND_ALPHA, PALETTE

            t = np.concatenate([[0.0], fit["time"]])
            B = np.concatenate([[0.0], fit["cum"][:, j + 1]])
            se = np.sqrt(np.concatenate([[0.0], fit["var"][:, j + 1]]))
            z = norm.isf(options.alpha / 2)
            ax.step(t, B, where="post", color=PALETTE[0], label="Aalen B(t)")
            ax.fill_between(
                t,
                B - z * se,
                B + z * se,
                step="post",
                color=PALETTE[0],
                alpha=BAND_ALPHA,
                linewidth=0,
                label="pointwise {:g}%".format(100 * (1 - options.alpha)),
            )
            ax.plot(
                [0.0, t[-1]],
                [0.0, beta[j] * t[-1]],
                color=PALETTE[1],
                ls="--",
                label="fitted beta * t",
            )
            ax.set_xlabel("time")
            ax.set_ylabel("cumulative coefficient")
            ax.legend(fontsize="small")

        panels.append(
            Panel(
                "{}: cumulative effect, p = {}".format(name, _fmt(res["p"])),
                draw,
            )
        )
    return panels


# ---------------------------------------------------------------------------
# Refusals and notes
# ---------------------------------------------------------------------------
def _validate(model: Any) -> None:
    label = "ProportionalOdds" if _is_po(model) else "AdditiveHazards"
    if _is_parametric(model):
        label = "this model"
        if getattr(model, "data", None) is None or getattr(
            model, "_restored", False
        ):
            raise ValueError(
                "diagnose() needs the data the model was fitted to, which "
                "a model restored with from_dict / from_json does not "
                "carry. Call it on the fitted model, or refit."
            )
        c = np.asarray(model.data.c)
        if np.any((c == -1) | (c == 2)):
            raise NotImplementedError(
                "diagnose() is not yet available for a parametric "
                "proportional odds or additive hazards model fitted to "
                "left- or interval-censored data: its trend tests split "
                "each row's follow-up at the cut times, which a censoring "
                "interval cannot be split by; tracked in #870."
            )
        t = np.asarray(model.data.t, dtype=float)
        if np.any(np.isfinite(t[:, 1])):
            raise NotImplementedError(
                "diagnose() is not yet available for a parametric "
                "proportional odds or additive hazards model fitted to "
                "right-truncated data (a finite tr): its trend tests "
                "split each row at the cut times, which a right-truncated "
                "row cannot be split by; tracked in #870."
            )
    elif getattr(model, "_fit_data", None) is None:
        raise ValueError(
            "diagnose() needs the data the model was fitted to, which a "
            "model restored with from_dict / from_json does not carry. "
            "Call it on the fitted model, or refit with {}.fit.".format(label)
        )
    if getattr(model, "maximum", None) == "no finite maximum":
        raise ValueError(
            "diagnose() needs a fit with a finite maximum, and this one "
            "has none (model.maximum is 'no finite maximum': a covariate, "
            "or a combination of them, separates the events from the "
            "survivors, as the fit warned). Drop or merge the separating "
            "covariate and refit."
        )
    beta = _coefficients(model)
    if not np.all(np.isfinite(beta)):
        names = _names(model)
        bad = [names[j] for j in np.flatnonzero(~np.isfinite(beta))]
        raise ValueError(
            "diagnose() needs every coefficient estimated, and {} {} "
            "aliased (nan: constant, or a combination of the other "
            "columns). Refit without {}.".format(
                ", ".join(bad),
                "is" if len(bad) == 1 else "are",
                "it" if len(bad) == 1 else "them",
            )
        )


def _period_note(ctx: Context) -> str:
    cuts = _cuts(ctx)
    if cuts.size == 0:
        return (
            "The trend tests need follow-up split into periods at the "
            "event-time quartiles, and these data have too few distinct "
            "event times to split."
        )
    return (
        "The trend tests split follow-up into {} periods at the "
        "event-time quartiles, t = {}, and test whether each covariate's "
        "coefficient changes from period to period (the added term "
        "(Z - mean) * period).".format(
            cuts.size + 1, ", ".join(_fmt(v) for v in cuts)
        )
    )


_NOT_CHECKED = (
    "Not checked yet for these models: the functional form of the "
    "covariates, influential rows and outliers (#870)."
)


def _po_notes(ctx: Context) -> list[str]:
    return [
        "Proportional odds: for each covariate the likelihood-ratio test "
        "of adding (Z - mean) * period to the model, fitted exactly by "
        "splitting each row at the cut times (one indicator per level "
        "after the first for a covariate with two to five values: whether "
        "its groups' log-odds of failure stay parallel). It is the "
        "likelihood-ratio counterpart of cox.zph's residual trend (score) "
        "test, with the period as the clock.",
        _period_note(ctx),
        "The log-odds plots are the Kaplan-Meier estimates of each group, "
        "not adjusted for the other covariates.",
        _NOT_CHECKED,
    ]


def _ah_notes(ctx: Context) -> list[str]:
    if _is_lin_ying(ctx.model):
        test = (
            "Constant effects: for each covariate the Wald test, with Lin "
            "and Ying's sandwich variance, of adding (Z - mean) * period "
            "to the model, solved in closed form (R's timereg::aalen with "
            "const() terms on the rows split at the cut times gives the "
            "same)."
        )
    else:
        test = (
            "Constant effects: for each covariate the likelihood-ratio "
            "test of adding (Z - mean) * period to the model, fitted "
            "exactly by splitting each row at the cut times."
        )
    return [
        test,
        _period_note(ctx),
        "The plots are Aalen's nonparametric cumulative regression "
        "coefficients (as timereg::aalen), which under the model are "
        "straight lines of slope beta: a bend shows when an effect "
        "changes.",
        _NOT_CHECKED,
    ]


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------
def _distribution(model: Any) -> "str | None":
    if not _is_parametric(model):
        return None
    return str(getattr(model.distribution, "name", ""))


def _failing_name(row: dict) -> str:
    return row["covariates"][0]


def _po_lead(row: dict, ctx: Context) -> str:
    if row.get("_global"):
        return (
            "The joint (GLOBAL) proportional-odds test fails though no "
            "single covariate does: the covariates' effects together "
            "change with time (see d.plot('proportional_odds'))."
        )
    name = _failing_name(row)
    what = (
        "its groups' log-odds of failure are not parallel"
        if row.get("_kind") != "continuous"
        else "its coefficient trends over follow-up"
    )
    return (
        "The effect of {} on the odds of survival changes with time ({}; "
        "see d.plot('proportional_odds')), so the odds are not "
        "proportional.".format(name, what)
    )


def _po_semi_message(row: dict, ctx: Context) -> str:
    return _po_lead(row, ctx) + (
        " Fit a model that does not assume proportional odds: proportional "
        "hazards, with CoxPH.fit (whose own diagnose() checks that "
        "assumption), or an accelerated failure time model, such as "
        "WeibullAFT.fit or LogNormalAFT.fit."
    )


def _po_parametric_message(dist: "str | None") -> Any:
    def message(row: dict, ctx: Context) -> str:
        name = _distribution(ctx.model) if dist is None else dist
        if dist is None:
            ph, aft = "PH({0}).fit".format(name), "AFT({0}).fit".format(name)
        else:
            ph, aft = "{}PH.fit".format(name), "{}AFT.fit".format(name)
        return _po_lead(row, ctx) + (
            " Fit the {name} baseline in a model that does not assume "
            "proportional odds: proportional hazards, {ph}, or an "
            "accelerated failure time model, {aft}; CoxPH.fit leaves the "
            "baseline to the data.".format(name=name, ph=ph, aft=aft)
        )

    return message


def _ah_lead(row: dict, ctx: Context) -> str:
    if row.get("_global"):
        return (
            "The joint (GLOBAL) test of constant effects fails though no "
            "single covariate does: the covariates' effects together "
            "change with time. The covariates whose cumulative "
            "coefficients bend in d.plot('constant_effects') are the "
            "likeliest cause."
        )
    return (
        "The effect of {} on the hazard changes with time (its cumulative "
        "coefficient in d.plot('constant_effects') is not a straight "
        "line).".format(_failing_name(row))
    )


def _tvc_recipe(row: dict, fitter: str) -> str:
    if row.get("_global"):
        whom, name = "them", "the covariate"
    else:
        whom, name = "it", _failing_name(row)
    return (
        " Give {whom} a time-varying effect: split each unit's follow-up at "
        "the time(s) where the cumulative coefficient bends, and fit "
        "{fitter} on the pieces (start-stop rows; {fitter}_from_df from a "
        "DataFrame), with one column per period holding {name} in that "
        "period and 0 elsewhere, so that each period has its own "
        "coefficient.".format(whom=whom, fitter=fitter, name=name)
    )


def _ah_semi_message(row: dict, ctx: Context) -> str:
    return (
        _ah_lead(row, ctx)
        + _tvc_recipe(row, "WeibullAH.fit_tvc")
        + " (AdditiveHazards has no time-varying fit; WeibullAH, or "
        "ExponentialAH.fit_tvc for a constant baseline hazard, is the "
        "additive hazards model with a parametric baseline.)"
    )


def _ah_parametric_message(dist: "str | None") -> Any:
    def message(row: dict, ctx: Context) -> str:
        if dist is None:
            fitter = "AH({}).fit_tvc".format(_distribution(ctx.model))
        else:
            fitter = "{}AH.fit_tvc".format(dist)
        return _ah_lead(row, ctx) + _tvc_recipe(row, fitter)

    return message


def _global_only(row: dict, ctx: Context) -> bool:
    """A GLOBAL row that fails with no single covariate failing."""
    if not row.get("_global"):
        return True
    for j, _ in ctx.checked:
        if _trend(ctx, str(j), _base(ctx, j))["p"] < ctx.alpha:
            return False
    return True


def _when(dist: "str | None", semi: bool) -> Any:
    """A rule's condition: the row calls for advice (a covariate's, or a
    GLOBAL failure alone) and the model is semi-parametric (``semi``) or
    parametric with the baseline ``dist`` (``None``: one that has no
    pre-built fitters)."""

    def when(row: dict, ctx: Context) -> bool:
        if not _global_only(row, ctx):
            return False
        if semi:
            return not _is_parametric(ctx.model)
        name = _distribution(ctx.model)
        if name is None:
            return False
        return name == dist if dist is not None else name not in PREBUILT

    return when


def _po_rules() -> tuple[AdviceRule, ...]:
    rules = [
        AdviceRule(
            "proportional_odds",
            _po_semi_message,
            ("CoxPH.fit", "WeibullAFT.fit", "LogNormalAFT.fit"),
            when=_when(None, semi=True),
        )
    ]
    for dist in PREBUILT:
        rules.append(
            AdviceRule(
                "proportional_odds",
                _po_parametric_message(dist),
                (
                    "{}PH.fit".format(dist),
                    "{}AFT.fit".format(dist),
                    "CoxPH.fit",
                ),
                when=_when(dist, semi=False),
            )
        )
    rules.append(
        AdviceRule(
            "proportional_odds",
            _po_parametric_message(None),
            ("PH", "AFT", "CoxPH.fit"),
            when=_when(None, semi=False),
        )
    )
    return tuple(rules)


def _ah_rules() -> tuple[AdviceRule, ...]:
    rules = [
        AdviceRule(
            "constant_effects",
            _ah_semi_message,
            (
                "WeibullAH.fit_tvc",
                "WeibullAH.fit_tvc_from_df",
                "ExponentialAH.fit_tvc",
            ),
            when=_when(None, semi=True),
        )
    ]
    for dist in PREBUILT:
        rules.append(
            AdviceRule(
                "constant_effects",
                _ah_parametric_message(dist),
                (
                    "{}AH.fit_tvc".format(dist),
                    "{}AH.fit_tvc_from_df".format(dist),
                ),
                when=_when(dist, semi=False),
            )
        )
    rules.append(
        AdviceRule(
            "constant_effects",
            _ah_parametric_message(None),
            ("AH",),
            when=_when(None, semi=False),
        )
    )
    return tuple(rules)


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------
def _is_po(model: Any) -> bool:
    from surpyval.univariate.regression import ProportionalOddsModel
    from surpyval.univariate.regression._kinds import PROPORTIONAL_ODDS

    if isinstance(model, ProportionalOddsModel):
        return True
    return _is_parametric(model) and model.kind == PROPORTIONAL_ODDS


def _is_ah(model: Any) -> bool:
    from surpyval.univariate.regression._kinds import ADDITIVE_HAZARD

    if _is_lin_ying(model):
        return True
    return _is_parametric(model) and model.kind == ADDITIVE_HAZARD


#: The proportional odds family: ``ProportionalOdds`` and ``PO(...)``.
PROPORTIONAL_ODDS_FAMILY = register(
    Family(
        name="Proportional odds",
        issue=870,
        applies=_is_po,
        validate=_validate,
        covariates=_names,
        checks=(
            Check(
                "proportional_odds",
                "Each covariate multiplies the odds of survival by a "
                "factor that does not change with time.",
                rows=_trend_rows,
                panels=_po_panels,
            ),
        ),
        advice=_po_rules(),
        notes=_po_notes,
    )
)

#: The additive hazards family: ``AdditiveHazards`` and ``AH(...)``.
ADDITIVE_HAZARDS_FAMILY = register(
    Family(
        name="Additive hazards",
        issue=870,
        applies=_is_ah,
        validate=_validate,
        covariates=_names,
        checks=(
            Check(
                "constant_effects",
                "Each covariate adds to the hazard an amount that does "
                "not change with time.",
                rows=_trend_rows,
                panels=_ah_panels,
            ),
        ),
        advice=_ah_rules(),
        notes=_ah_notes,
    )
)
