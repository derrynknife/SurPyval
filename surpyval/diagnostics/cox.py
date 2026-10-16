"""``diagnose()`` for the Cox proportional hazards model (#868).

The first family of :mod:`surpyval.diagnostics`, and the model to copy:
its checks, plots and advice are listed in :data:`COX` at the bottom, and
everything above it computes them from the residuals and tests of
:mod:`surpyval.univariate.regression.proportional_hazards.diagnostics`.

The checks
----------
``proportional_hazards``
    The score test of R's ``survival::cox.zph`` (version 3.0 and later) of
    whether each covariate's coefficient changes with time, on the
    Kaplan-Meier (``km``) and the ``rank`` scale of time, and the joint
    ``GLOBAL`` test on each; a covariate fails if it fails on either scale.
    Drawn as the scaled Schoenfeld residuals against time (on the km
    scale), with a lowess trend and its bootstrap band: a flat trend at
    the fitted coefficient is proportional hazards.
``functional_form``
    For each covariate with three or more distinct values, the score test
    of adding its square: whether its effect on the log hazard is linear.
    Drawn as the martingale residuals against the covariate, with a
    lowess trend, which is flat when the form is right.
``influence``
    For each covariate, the largest ``|dfbeta| / se``: the move, in
    standard errors, of the coefficient when one observation is dropped
    (the one-step approximation; a row with count ``n`` is ``n``
    observations). It warns when a single row moves a
    coefficient by more than one standard error; a screen, never a fail.
``outliers``
    The deviance residuals: a row is an outlier when its ``|deviance
    residual|`` exceeds the two-sided Bonferroni normal cutoff for the
    number of observations, ``z_{1 - alpha / (2 N)}`` (3.66 for 200
    observations at alpha 0.05), so that a model that fits flags one about
    ``alpha`` of the time. It warns when any row is flagged; never a fail.
``overall_fit``
    A plot only: the Nelson-Aalen cumulative hazard of the Cox-Snell
    residuals against the unit exponential's, the line ``y = x``. For a
    Cox model the residuals sum to the number of events by construction,
    so they have no useful test.

A covariate with two distinct values is *binary*, one with three to five
*categorical* (a coded factor or a few-level score), and one with more
*continuous*: the advice for a proportional-hazards failure depends on
which.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy.stats import norm

from surpyval.univariate.regression.proportional_hazards.diagnostics import (
    added_covariate_score_test,
    check_ph,
    compute_residuals,
    cox_fit_data,
    zph_times,
)

from .registry import AdviceRule, Check, Context, Family, Panel, register

#: The time scales of the proportional-hazards test.
TRANSFORMS = ("km", "rank")
#: The most distinct values a covariate has and still counts as
#: categorical (one with two is binary); more is continuous.
CATEGORICAL_LEVELS = 5
#: ``|dfbeta| / se`` above this warns: one row moves the coefficient by
#: more than a standard error.
INFLUENCE_LIMIT = 1.0


# ---------------------------------------------------------------------------
# What the checks share
# ---------------------------------------------------------------------------
def _raw_Z(ctx: Context) -> np.ndarray:
    """The covariates as fitted (not centred), one row per fitted row."""
    return np.asarray(ctx.model._fit_data["Z"], dtype=float)


def _kind(ctx: Context, j: int) -> str:
    """``"binary"``, ``"categorical"`` or ``"continuous"``, by the number
    of distinct values of covariate ``j``."""
    levels = np.unique(_raw_Z(ctx)[:, j]).size
    if levels <= 2:
        return "binary"
    if levels <= CATEGORICAL_LEVELS:
        return "categorical"
    return "continuous"


def _source(name: str) -> str:
    """The variable a formula-coded column comes from (``sex`` for
    ``C(sex)[T.male]`` or ``sex[T.male]``), else the name itself."""
    match = re.match(r"^C\(\s*([A-Za-z_][\w.]*)[^)]*\)\[", name)
    if match is None:
        match = re.match(r"^([A-Za-z_][\w.]*)\[", name)
    return match.group(1) if match else name


def _zph(ctx: Context, transform: str) -> dict:
    return ctx.cached(
        "zph_" + transform, lambda: check_ph(ctx.model, transform)
    )


def _residuals(ctx: Context, kind: str) -> np.ndarray:
    return ctx.cached(
        "residuals_" + kind, lambda: compute_residuals(ctx.model, kind)
    )


def _event_rows(ctx: Context) -> np.ndarray:
    return np.flatnonzero(np.asarray(ctx.model._fit_data["c"]) == 0)


def _fmt(value: float) -> str:
    return "{:.3g}".format(value)


# ---------------------------------------------------------------------------
# Proportional hazards
# ---------------------------------------------------------------------------
def _trend(ctx: Context, j: int) -> float:
    """The sign of the slope of covariate ``j``'s scaled Schoenfeld
    residuals on the km scale of time: the direction its coefficient
    moves in."""
    events = _event_rows(ctx)
    g = zph_times(cox_fit_data(ctx.model), "km")[events]
    scaled = _residuals(ctx, "scaled_schoenfeld")[:, j]
    weights = np.asarray(ctx.model._fit_data["n"], dtype=float)[events]
    return float(np.sign(np.cov(g, scaled, aweights=weights)[0, 1]))


def _ph_rows(ctx: Context) -> list[dict]:
    rows = []
    for transform in TRANSFORMS:
        res = _zph(ctx, transform)
        for j, name in ctx.checked:
            entry = res["per_covariate"][j]
            p = entry["p_value"]
            if not np.isfinite(p):
                detail = "the test could not be computed on these data"
            elif p < 2 * ctx.alpha:
                detail = "its coefficient {} with time".format(
                    "rises" if _trend(ctx, j) > 0 else "falls"
                )
            else:
                detail = "no evidence that its coefficient changes with time"
            rows.append(
                {
                    "test": "Grambsch-Therneau score test",
                    "transform": transform,
                    "covariates": (name,),
                    "statistic": entry["statistic"],
                    "df": 1,
                    "p": p,
                    "detail": detail,
                    "_kind": _kind(ctx, j),
                }
            )
        glob = res["global"]
        rows.append(
            {
                "test": "Grambsch-Therneau score test (GLOBAL)",
                "transform": transform,
                "covariates": tuple(ctx.names),
                "statistic": glob["statistic"],
                "df": glob["df"],
                "p": glob["p_value"],
                "detail": "the joint test of every covariate",
                "_global": True,
            }
        )
    return rows


def _time_axis(ax: Any, g: np.ndarray, t: np.ndarray) -> None:
    """Label the km-scale axis with the times it stands for."""
    order = np.argsort(g)
    g, t = g[order], t[order]
    ticks = np.linspace(g[0], g[-1], 5)
    labels = np.interp(ticks, g, t)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["{:.3g}".format(v) for v in labels])
    ax.set_xlabel("time (Kaplan-Meier scale)")


def _ph_panels(ctx: Context) -> list[Panel]:
    from .smooth import bootstrap_band, lowess

    data = cox_fit_data(ctx.model)
    events = _event_rows(ctx)
    g = zph_times(data, "km")[events]
    t = np.asarray(data["x"], dtype=float)[events]
    scaled = _residuals(ctx, "scaled_schoenfeld")
    beta = np.asarray(ctx.model.beta, dtype=float)
    p_km = _zph(ctx, "km")["per_covariate"]
    panels = []
    for j, name in ctx.checked:

        def draw(ax: Any, options: Any, j: int = j) -> None:
            from surpyval.plotting import BAND_ALPHA, INK, PALETTE

            grid = np.linspace(g.min(), g.max(), 60)
            ax.scatter(g, scaled[:, j], s=10, color=INK["muted"], alpha=0.6)
            ax.plot(
                grid,
                lowess(g, scaled[:, j], grid),
                color=PALETTE[0],
                label="lowess",
            )
            if options.n_boot > 0:
                band = bootstrap_band(
                    g,
                    scaled[:, j],
                    grid,
                    options.n_boot,
                    options.rng,
                    1 - options.alpha,
                )
                ax.fill_between(
                    grid,
                    band[:, 0],
                    band[:, 1],
                    color=PALETTE[0],
                    alpha=BAND_ALPHA,
                    linewidth=0,
                    label="bootstrap {:g}%".format(100 * (1 - options.alpha)),
                )
            ax.axhline(beta[j], color=PALETTE[1], ls="--", label="fitted")
            _time_axis(ax, g, t)
            ax.set_ylabel("beta(t), scaled Schoenfeld")

        panels.append(
            Panel(
                "{}: p = {} (km)".format(name, _fmt(p_km[j]["p_value"])),
                draw,
            )
        )
    return panels


# ---------------------------------------------------------------------------
# Functional form
# ---------------------------------------------------------------------------
def _form_columns(ctx: Context) -> list[tuple[int, str]]:
    return [(j, n) for j, n in ctx.checked if _kind(ctx, j) != "binary"]


def _form_rows(ctx: Context) -> list[dict]:
    rows = []
    Z = _raw_Z(ctx)
    skipped = [n for j, n in ctx.checked if _kind(ctx, j) == "binary"]
    if skipped:
        ctx.notes.append(
            "Functional form is tested for covariates with three or more "
            "distinct values; {} {} two, and the effect of a two-valued "
            "covariate is linear by construction.".format(
                ", ".join(skipped), "has" if len(skipped) == 1 else "have"
            )
        )
    for j, name in _form_columns(ctx):
        res = added_covariate_score_test(ctx.model, Z[:, j] ** 2)
        p = res["p_value"]
        if not np.isfinite(p):
            detail = "the test could not be computed"
        elif p < 2 * ctx.alpha:
            detail = "its effect on the log hazard is not linear"
        else:
            detail = "no evidence against a linear effect"
        rows.append(
            {
                "test": "score test of a quadratic term",
                "covariates": (name,),
                "statistic": res["statistic"],
                "df": res["df"],
                "p": p,
                "detail": detail,
            }
        )
    return rows


def _form_panels(ctx: Context) -> list[Panel]:
    from .smooth import lowess

    Z = _raw_Z(ctx)
    martingale = _residuals(ctx, "martingale")
    panels = []
    for j, name in _form_columns(ctx):

        def draw(ax: Any, options: Any, j: int = j, name: str = name) -> None:
            from surpyval.plotting import INK, PALETTE

            z = Z[:, j]
            grid = np.linspace(z.min(), z.max(), 60)
            ax.scatter(z, martingale, s=10, color=INK["muted"], alpha=0.6)
            ax.plot(grid, lowess(z, martingale, grid), color=PALETTE[0])
            ax.axhline(0.0, color=INK["axis"], lw=0.8)
            ax.set_xlabel(name)
            ax.set_ylabel("martingale residual")

        panels.append(Panel("{}: functional form".format(name), draw))
    return panels


# ---------------------------------------------------------------------------
# Influence and outliers
# ---------------------------------------------------------------------------
def _scaled_dfbeta(ctx: Context) -> np.ndarray:
    """Each observation's dfbeta in standard errors. A row with count
    ``n`` stands for ``n`` observations and its dfbeta is theirs together,
    so it is divided by ``n``: the screen is the same for the data written
    out one row per observation (principle 5)."""
    se = np.asarray(ctx.model.standard_errors(), dtype=float)
    n = np.asarray(ctx.model._fit_data["n"], dtype=float)
    return _residuals(ctx, "dfbeta") / n[:, None] / se[None, :]


def _influence_rows(ctx: Context) -> list[dict]:
    scaled = np.abs(_scaled_dfbeta(ctx))
    rows = []
    for j, name in ctx.checked:
        col = scaled[:, j]
        worst = int(np.argmax(col))
        above = np.flatnonzero(col > INFLUENCE_LIMIT)
        if above.size:
            detail = (
                "row {} moves the coefficient by {} standard errors; {} "
                "row{} above {:g}".format(
                    worst,
                    _fmt(col[worst]),
                    above.size,
                    "" if above.size == 1 else "s",
                    INFLUENCE_LIMIT,
                )
            )
        else:
            detail = (
                "no row moves the coefficient by more than {:g} standard "
                "error (largest {}, row {})".format(
                    INFLUENCE_LIMIT, _fmt(col[worst]), worst
                )
            )
        rows.append(
            {
                "test": "largest |dfbeta| / se",
                "covariates": (name,),
                "statistic": float(col[worst]),
                "verdict": "warn" if above.size else "ok",
                "detail": detail,
                "_row": worst,
                "_size": float(col[worst]),
            }
        )
    return rows


def _outlier_cutoff(ctx: Context) -> float:
    n = float(np.asarray(ctx.model._fit_data["n"], dtype=float).sum())
    return float(norm.isf(ctx.alpha / (2.0 * n)))


def _outlier_rows(ctx: Context) -> list[dict]:
    dev = np.abs(_residuals(ctx, "deviance"))
    cutoff = _outlier_cutoff(ctx)
    flagged = np.flatnonzero(dev > cutoff)
    if flagged.size:
        shown = ", ".join(str(int(i)) for i in flagged[:10])
        more = (
            ""
            if flagged.size <= 10
            else " and {} more".format(flagged.size - 10)
        )
        detail = "row{} {}{} above {}".format(
            "" if flagged.size == 1 else "s", shown, more, _fmt(cutoff)
        )
    else:
        detail = "no |deviance residual| above {} (largest {})".format(
            _fmt(cutoff), _fmt(dev.max())
        )
    return [
        {
            "test": "deviance residuals, Bonferroni cutoff",
            "covariates": (),
            "statistic": float(dev.max()),
            "verdict": "warn" if flagged.size else "ok",
            "detail": detail,
            "_rows": flagged.tolist(),
            "_cutoff": cutoff,
        }
    ]


def _influence_panels(ctx: Context) -> list[Panel]:
    scaled = _scaled_dfbeta(ctx)
    panels = []
    for j, name in ctx.checked:

        def draw(ax: Any, options: Any, j: int = j) -> None:
            from surpyval.plotting import INK, PALETTE

            rows = np.arange(scaled.shape[0])
            ax.vlines(rows, 0, scaled[:, j], color=PALETTE[0], lw=0.8)
            for level in (-INFLUENCE_LIMIT, INFLUENCE_LIMIT):
                ax.axhline(level, color=INK["secondary"], ls="--", lw=0.8)
            ax.set_xlabel("row")
            ax.set_ylabel("dfbeta / se")

        panels.append(Panel("{}: influence".format(name), draw))
    return panels


def _outlier_panels(ctx: Context) -> list[Panel]:
    dev = _residuals(ctx, "deviance")
    data = cox_fit_data(ctx.model)
    eta = np.asarray(data["Z"], dtype=float) @ np.asarray(
        ctx.model.beta, dtype=float
    )
    cutoff = _outlier_cutoff(ctx)

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        ax.scatter(eta, dev, s=10, color=PALETTE[0], alpha=0.6)
        for level in (-cutoff, cutoff):
            ax.axhline(level, color=INK["secondary"], ls="--", lw=0.8)
        ax.set_xlabel("linear predictor (centred)")
        ax.set_ylabel("deviance residual")

    return [Panel("outliers", draw)]


def _overall_panels(ctx: Context) -> list[Panel]:
    data = cox_fit_data(ctx.model)
    c = np.asarray(data["c"])
    n = np.asarray(data["n"], dtype=float)
    # Cox-Snell residual: each row's cumulative hazard over its time at
    # risk, delta - martingale.
    r = (c == 0).astype(float) - _residuals(ctx, "martingale")
    order = np.argsort(r, kind="stable")
    r, c, n = r[order], c[order], n[order]
    uniq, first = np.unique(r, return_index=True)
    at_risk = n[::-1].cumsum()[::-1][first]
    deaths = np.bincount(
        np.searchsorted(uniq, r), weights=n * (c == 0), minlength=uniq.size
    )
    H = np.cumsum(deaths / at_risk)
    keep = deaths > 0

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        ax.step(uniq[keep], H[keep], where="post", color=PALETTE[0])
        top = float(max(uniq[keep].max(), H[keep].max())) if keep.any() else 1
        ax.plot([0, top], [0, top], color=INK["secondary"], ls="--", lw=0.8)
        ax.set_xlabel("Cox-Snell residual")
        ax.set_ylabel("cumulative hazard")

    return [Panel("overall fit (Cox-Snell)", draw)]


# ---------------------------------------------------------------------------
# Refusals and notes
# ---------------------------------------------------------------------------
def _validate(model: Any) -> None:
    if getattr(model, "is_stratified", False):
        raise NotImplementedError(
            "diagnose() is not yet available for a stratified Cox model: "
            "the residuals its checks use (compute_residuals) do not "
            "handle strata yet; tracked in #868. Diagnose the model "
            "fitted without strata= to see which covariates break "
            "proportional hazards."
        )
    if getattr(model, "maximum", None) == "no finite maximum":
        raise ValueError(
            "diagnose() needs a fit with a finite maximum, and this one "
            "has none (model.maximum is 'no finite maximum': a covariate, "
            "or a combination of them, separates the events from the "
            "survivors, as the fit warned). Drop or merge the separating "
            "covariate and refit."
        )
    cox_fit_data(model)


def _clustered(ctx: Context) -> int:
    """The number of subjects with more than one row (0 unless the model
    was fitted with ``fit_tvc``, the one Cox fit that knows its
    clusters)."""
    ids = getattr(ctx.model, "tvc_subject_ids", None)
    if not getattr(ctx.model, "is_tvc", False) or ids is None:
        return 0
    _, counts = np.unique(np.asarray(ids), return_counts=True)
    return int((counts > 1).sum())


def _notes(ctx: Context) -> list[str]:
    notes = [
        "Proportional hazards: the score test of R's survival::cox.zph "
        "(version 3.0 and later), as model.check_ph() computes it, on the "
        "km and rank scales of time; a covariate fails if it fails on "
        "either.",
        "Overall fit is a plot only (d.plot('overall_fit')): a Cox model's "
        "Cox-Snell residuals sum to its number of events by construction, "
        "so they have no useful test.",
    ]
    tie_method = str(getattr(ctx.model, "tie_method", "efron")).lower()
    if tie_method not in ("efron", "breslow"):
        notes.append(
            "With tie_method={!r} the residuals and tests use Breslow's "
            "risk sets, as R's cox.zph does for its exact method; the "
            "functional-form test is then an approximation.".format(tie_method)
        )
    if not _clustered(ctx):
        notes.append(
            "Robust standard errors: the model has no clusters (only a "
            "fit_tvc model groups rows, by subject); for clustered rows "
            "summary(robust=True, cluster=...) gives cluster-robust "
            "standard errors."
        )
    return notes


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------
def _ph_name(row: dict) -> str:
    return row["covariates"][0]


def _stratify(row: dict, ctx: Context) -> str:
    name = _ph_name(row)
    var = _source(name)
    return (
        "The effect of {name} changes with time (it fails the "
        "proportional-hazards test). Stratify on {var}: fit with "
        "CoxPH.fit(..., strata=<the {var} column>) and leave {var} out of "
        'Z (from a DataFrame, CoxPH.fit_from_df(..., strata_col="{var}")). '
        "Each level then has its own baseline hazard, and the model no "
        "longer estimates an effect for {var}.".format(name=name, var=var)
    )


def _time_varying(row: dict, ctx: Context) -> str:
    return (
        "The effect of {name} changes with time (it fails the "
        "proportional-hazards test). Give it a time-varying effect: split "
        "each unit's follow-up at one or more times (where its scaled "
        "Schoenfeld residuals in d.plot('proportional_hazards') bend) and "
        "fit CoxPH.fit_tvc on the pieces, with one column per period "
        "holding {name} in that period and 0 elsewhere, so that each "
        "period has its own coefficient. Or bin {name} (with pandas.cut, "
        "say) and stratify on the bins (CoxPH.fit(..., strata=...)). Or "
        "fit a parametric accelerated failure time model, such as "
        "WeibullAFT.fit or LogNormalAFT.fit, which does not assume "
        "proportional hazards.".format(name=_ph_name(row))
    )


def _global_only(row: dict, ctx: Context) -> bool:
    if not row.get("_global"):
        return False
    rows = _ph_rows_failed(ctx, row["transform"])
    return not rows


def _ph_rows_failed(ctx: Context, transform: str) -> list[int]:
    res = _zph(ctx, transform)["per_covariate"]
    return [j for j, _ in ctx.checked if res[j]["p_value"] < ctx.alpha]


def _global_message(row: dict, ctx: Context) -> str:
    return (
        "The joint (GLOBAL) proportional-hazards test fails on the {} "
        "scale though no single covariate does. The covariates whose "
        "scaled Schoenfeld residuals trend in "
        "d.plot('proportional_hazards') are the likeliest cause: stratify "
        "on them (CoxPH.fit(..., strata=...)) or give them a time-varying "
        "effect (CoxPH.fit_tvc).".format(row["transform"])
    )


def _form_message(row: dict, ctx: Context) -> str:
    name = row["covariates"][0]
    return (
        "{name} does not act linearly on the log hazard (its square is "
        "significant; see d.plot('functional_form')). Transform it, for "
        "example with a log in a formula, CoxPH.fit_from_df(..., "
        'formula="np.log({name}) + ..."), or bin it (with pandas.cut, '
        "say) and fit the bins as a categorical covariate, "
        'formula="C({name}_band) + ...".'.format(name=name)
    )


def _influence_message(row: dict, ctx: Context) -> str:
    return (
        "Row {row} has a large influence on the coefficient of {name}: "
        "dropping it would move the coefficient by about {size} standard "
        "errors. Check that row, and refit without it to see how much the "
        "conclusions depend on it; compute_residuals('dfbeta') gives "
        "every row's influence.".format(
            row=row["_row"], name=row["covariates"][0], size=_fmt(row["_size"])
        )
    )


def _outlier_message(row: dict, ctx: Context) -> str:
    rows = row["_rows"]
    shown = ", ".join(str(i) for i in rows[:10])
    if len(rows) > 10:
        shown += " and {} more".format(len(rows) - 10)
    return (
        "The model predicts row{s} {shown} poorly (|deviance residual| "
        "above {cut}). Check {them} for recording errors; "
        "compute_residuals('deviance') gives every row's residual.".format(
            s="" if len(rows) == 1 else "s",
            shown=shown,
            cut=_fmt(row["_cutoff"]),
            them="it" if len(rows) == 1 else "them",
        )
    )


def _cluster_message(row: Any, ctx: Context) -> str:
    return (
        "The rows are grouped by subject ({} subjects have more than one "
        "row), and the model-based standard errors assume independent "
        "rows: summary(robust=True), or robust_summary(), gives the "
        "cluster-robust (Lin-Wei) standard errors, clustered by "
        "subject.".format(_clustered(ctx))
    )


def _is_cox(model: Any) -> bool:
    from surpyval.univariate.regression.semi_parametric_regression_model import (  # noqa: E501
        SemiParametricRegressionModel,
    )

    return isinstance(model, SemiParametricRegressionModel)


#: The Cox family: its checks, in the order of the table, and its advice.
COX = register(
    Family(
        name="Cox proportional hazards",
        issue=868,
        applies=_is_cox,
        validate=_validate,
        columns=("transform",),
        checks=(
            Check(
                "proportional_hazards",
                "Each covariate multiplies the hazard by a factor that "
                "does not change with time.",
                rows=_ph_rows,
                panels=_ph_panels,
            ),
            Check(
                "functional_form",
                "Each covariate acts linearly on the log hazard.",
                rows=_form_rows,
                panels=_form_panels,
            ),
            Check(
                "influence",
                "No single row decides a coefficient.",
                rows=_influence_rows,
                panels=_influence_panels,
            ),
            Check(
                "outliers",
                "No row is far worse predicted than the model allows.",
                rows=_outlier_rows,
                panels=_outlier_panels,
            ),
            Check(
                "overall_fit",
                "The Cox-Snell residuals behave as a censored sample of "
                "the unit exponential.",
                panels=_overall_panels,
            ),
        ),
        advice=(
            AdviceRule(
                "proportional_hazards",
                _stratify,
                ("CoxPH.fit(strata=)", "CoxPH.fit_from_df(strata_col=)"),
                when=lambda row, ctx: row.get("_kind")
                in ("binary", "categorical"),
            ),
            AdviceRule(
                "proportional_hazards",
                _time_varying,
                (
                    "CoxPH.fit_tvc",
                    "CoxPH.fit(strata=)",
                    "WeibullAFT.fit",
                    "LogNormalAFT.fit",
                ),
                when=lambda row, ctx: row.get("_kind") == "continuous",
            ),
            AdviceRule(
                "proportional_hazards",
                _global_message,
                ("CoxPH.fit(strata=)", "CoxPH.fit_tvc"),
                when=_global_only,
            ),
            AdviceRule(
                "functional_form",
                _form_message,
                ("CoxPH.fit_from_df(formula=)",),
            ),
            AdviceRule(
                "influence",
                _influence_message,
                ("compute_residuals(kind=)",),
                on=("warn",),
            ),
            AdviceRule(
                "outliers",
                _outlier_message,
                ("compute_residuals(kind=)",),
                on=("warn",),
            ),
            AdviceRule(
                None,
                _cluster_message,
                ("summary(robust=)", "robust_summary"),
                when=lambda row, ctx: _clustered(ctx) > 0,
            ),
        ),
        notes=_notes,
    )
)
