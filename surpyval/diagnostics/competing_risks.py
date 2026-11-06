"""``diagnose()`` for the competing-risks regression models (#874).

Two families, one for each kind of model a competing-risks regression
fits, both reporting by cause (the table's ``cause`` column, the cause's
label as text):

**Cause-specific Cox** (:data:`CAUSE_SPECIFIC`):
``CompetingRisksProportionalHazards`` with ``model="Cox"``, one Cox model
per cause with the other causes censored. Each cause's model is kept, as
``model.models[cause]`` (a ``CoxPH`` model), and its checks are exactly
those of the Cox family (:mod:`surpyval.diagnostics.cox`) on it:
``proportional_hazards`` (R's ``cox.zph`` score test on the km and rank
scales, with a ``GLOBAL`` row per cause), ``functional_form``,
``influence``, ``outliers`` and the ``overall_fit`` plot, each row and
panel labelled with its cause. The verdicts and the screens' rules are
the Cox family's.

**Fine-Gray** (:data:`FINE_GRAY`): ``FineGrayModel`` (``FineGray.fit``,
one cause) and ``CompetingRisksProportionalHazards`` with
``model="Fine-Gray"`` (one per cause).

``proportional_hazards``
    Proportional subdistribution hazards: for each covariate, the score
    test of adding ``Z_j * g(t)`` to the Fine-Gray weighted partial
    likelihood, at the fitted coefficients and a zero coefficient for the
    added term, with the censoring weights ``G(t-)/G(x_i-)`` held as
    fitted, and the joint ``GLOBAL`` test (:func:`fine_gray_zph_test`).
    It is the test of R's ``survival::cox.zph`` applied to the weighted
    counting-process data on which ``coxph`` reproduces the Fine-Gray fit
    (``survival::finegray``'s construction), a Grambsch-Therneau test of
    the weighted Schoenfeld residuals. ``g`` is the ``cif`` scale, the
    Aalen-Johansen cumulative incidence of the cause just before ``t``
    (the subdistribution's analogue of the Cox ``km`` scale), and the
    ``rank`` scale, the rank of ``t`` among all the rows. A covariate
    fails if it fails on either. Drawn as the scaled weighted Schoenfeld
    residuals against time on the cif scale.
``functional_form``
    For each covariate with three or more distinct values, the score test
    of adding its square to the weighted partial likelihood
    (:func:`fine_gray_added_covariate_test`). Drawn as the
    subdistribution martingale residuals against the covariate.

The p-values use the model-based information, as ``FineGray``'s
standard errors do (not Fine and Gray's robust variance). Influence,
outliers and overall fit are not checked for a Fine-Gray model. The
tests are computed by
:mod:`surpyval.univariate.competing_risks.regression.fine_gray_diagnostics`
(re-exported here), as the Cox family's are by the Cox model's
``proportional_hazards.diagnostics``.

**Parametric** (:data:`PARAMETRIC`): ``ParametricCompetingRisks``, one
distribution per cause, each fitted with the other causes censored (the
likelihood factorises over the causes), kept as ``model.models[cause]``.

``goodness_of_fit``
    For each cause, the Cox-Snell residuals ``r_i = H_k(x_i) - H_k(tl_i)``
    of the cause's fitted distribution (its cumulative hazard over each
    unit's time at risk, from entry), the other causes' failures
    censored, against the unit exponential. It is the univariate
    family's test (:mod:`surpyval.diagnostics.univariate`) on the cause's
    model, which compares ``u = 1 - exp(-r)``, the CDF conditioned on
    entry, with the uniform: the Anderson-Darling, Kolmogorov-Smirnov and
    Cramer-von Mises statistics of the Kaplan-Meier estimate of ``u``,
    with parametric bootstrap p-values (each resample drawn from the
    cause's model and censored as that cause's data were, the other
    causes' failures with the censoring). Drawn as the P-P plot of ``u``
    and the bootstrap null distribution, by cause.
``probability_plot``
    A plot only: each cause's probability plot (``models[cause].plot()``).
``alternatives``
    For each cause, the univariate family's AIC screen of the
    alternatives (the best other distribution ``fit_best`` ranks, a
    mixture of two, an offset, a limited failure population, zero
    inflation) fitted to that cause's data; a row warns when one's AIC is
    at least 2 below the cause's model's, and never fails.
``incidence``
    For each cause, the largest gap between the model's cumulative
    incidence and the Aalen-Johansen estimate (``CompetingRisks.fit`` of
    the same data), ``D_k = sup_t |CIF_k(t) - AJ_k(t)|`` up to the last
    time observed (:func:`incidence_distance`). Its p-value is from a
    parametric bootstrap (:func:`incidence_bootstrap`): each resample
    draws every unit's latent time from each cause's model, keeps the
    earliest and its cause, censors it as the data were (a censored unit
    at its own time, a failed one at a time drawn from the censoring
    distribution beyond its failure, as the univariate bootstrap does),
    refits every cause's model the same way and recomputes ``D_k``.
    Drawn as the two incidences with the Aalen-Johansen pointwise band.
    A cause's incidence depends on every cause's hazard (``CIF_k = int
    h_k S``), so a wrong distribution for one cause can move another
    cause's incidence too; the per-cause goodness of fit says which.
    With delayed entry (``tl``) the Aalen-Johansen estimate counts each
    unit at risk only from its entry, and is conditional on surviving to
    the earliest entry ``s``; the model's incidence it is compared with is
    conditioned the same way, ``(CIF_k(t) - CIF_k(s)) / S(s)`` (``S`` the
    all-cause survival), from ``s`` on. Each resample then draws a unit's
    latent times beyond its own entry, and refits with the same entries.

``diagnose()`` refuses a model restored with ``from_dict`` (it has no
data), one fitted to left- or interval-censored rows (its causes are
fitted jointly and keep no data, and these checks need each failure's
time), one assembled with ``from_fitted`` from models that were not
fitted to the same units or that are not fitted distributions, and a
cause whose model the univariate family refuses (the message names the
cause).

Every report notes that the independence of the causes cannot be tested
from competing-risks data (Tsiatis 1975), and how to choose between the
cause-specific and the Fine-Gray model.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from surpyval.univariate.competing_risks.regression.fine_gray_diagnostics import (  # noqa: E501
    TRANSFORMS as FG_TRANSFORMS,
)
from surpyval.univariate.competing_risks.regression.fine_gray_diagnostics import (  # noqa: E501
    FineGrayData,
    fine_gray_added_covariate_test,
    fine_gray_data,
    fine_gray_martingale_residuals,
    fine_gray_zph_test,
)

from . import cox as _cox
from ._shared import _covariate_kind, _fmt, _note, _source, _trend_sign
from .registry import AdviceRule, Check, Context, Family, Panel, register

__all__ = [
    "fine_gray_zph_test",
    "fine_gray_added_covariate_test",
    "incidence_distance",
    "incidence_bootstrap",
]

_INDEPENDENCE_NOTE = (
    "Independence of the causes (of their latent failure times) cannot be "
    "tested from competing-risks data (Tsiatis 1975), so no check here "
    "tests it. The cause-specific hazards and the cumulative incidences "
    "are identifiable without it; a net survival (one cause's, with the "
    "others removed) is not, and rests on the assumption."
)

_CHOICE_NOTE = (
    "Choosing the model: the cause-specific model (model='Cox') answers "
    "questions of aetiology, how a covariate changes the rate of a cause "
    "among those still at risk; the Fine-Gray model (model='Fine-Gray', "
    "or FineGray.fit) answers questions of prediction, how it changes the "
    "incidence of a cause. Proportional hazards rarely holds exactly for "
    "both at once, so a cause-specific model that passes and a Fine-Gray "
    "model that fails (or the reverse) is expected, not a contradiction."
)


def _label(cause: Any) -> str:
    """A cause's label as the table shows it."""
    return str(cause)


def _cause_name(cause: Any) -> str:
    """A cause's label as the advice writes it in code (``'a'``, ``2``)."""
    if isinstance(cause, (np.generic,)):
        cause = cause.item()
    return repr(cause)


def _crph_names(model: Any) -> list[str]:
    """The covariate names of a ``CompetingRisksProportionalHazards``
    model (one set for every cause), as its ``summary()`` names them
    after the cause."""
    from surpyval.univariate.regression._summary import coefficient_names

    return list(coefficient_names(model, np.shape(model.betas)[1]))


def _crph_class() -> Any:
    from surpyval.univariate.competing_risks import (
        CompetingRisksProportionalHazards,
    )

    return CompetingRisksProportionalHazards


def _fg_class() -> Any:
    from surpyval.univariate.competing_risks.regression.fine_gray import (
        FineGrayModel,
    )

    return FineGrayModel


def _restored(label: str) -> ValueError:
    return ValueError(
        "diagnose() needs the data each cause was fitted to, which a {} "
        "restored with from_dict / from_json does not carry. Call it on "
        "the fitted model, or refit.".format(label)
    )


def _relabel(error: Exception, cause: Any) -> Exception:
    """``error`` again, its message about cause ``cause``."""
    return type(error)(f"Cause {_cause_name(cause)}: {error}")


# ===========================================================================
# Cause-specific Cox: the Cox family's checks, cause by cause
# ===========================================================================
def _cause_contexts(ctx: Context) -> list[tuple[Any, Context]]:
    """Each cause and the Cox context of its model, with the report's
    level and covariates (computed once per report)."""

    def build() -> list[tuple[Any, Context]]:
        return [
            (
                cause,
                Context(
                    model,
                    ctx.alpha,
                    ctx.names,
                    ctx.columns,
                    options=ctx.options,
                ),
            )
            for cause, model in ctx.model.models.items()
        ]

    return ctx.cached("causes", build)


def _context_of(ctx: Context, cause: Any) -> Context:
    for c, sub in _cause_contexts(ctx):
        if c == cause:
            return sub
    raise KeyError(cause)


def _keep_notes(ctx: Context, sub: Context) -> None:
    ctx.notes.extend(n for n in sub.notes if n not in ctx.notes)


def _per_cause_rows(check: Check) -> Any:
    """The rows of the Cox check ``check`` for every cause, each with its
    ``cause``."""

    def rows(ctx: Context) -> list[dict]:
        out = []
        assert check.rows is not None
        for cause, sub in _cause_contexts(ctx):
            for row in check.rows(sub):
                out.append(dict(row, cause=_label(cause), _cause=cause))
            _keep_notes(ctx, sub)
        return out

    return rows


def _per_cause_panels(check: Check) -> Any:
    """The panels of the Cox check ``check`` for every cause, each
    titled with its cause."""

    def panels(ctx: Context) -> list[Panel]:
        out = []
        assert check.panels is not None
        for cause, sub in _cause_contexts(ctx):
            for panel in check.panels(sub):
                out.append(
                    Panel(
                        f"cause {_label(cause)}: {panel.title}",
                        panel.draw,
                    )
                )
        return out

    return panels


def _per_cause_table(check: Check) -> Any:
    """The table of the Cox check ``check`` for every cause, one block
    per cause, indexed by ``(cause, covariate)``: ``.loc[cause]`` is the
    table of that cause's model."""

    def table(ctx: Context) -> Any:
        import pandas as pd

        assert check.table is not None
        blocks = [
            (_label(cause), check.table(sub))
            for cause, sub in _cause_contexts(ctx)
        ]
        out = pd.concat(
            [b for _, b in blocks],
            keys=[label for label, _ in blocks],
            names=["cause"],
        )
        if blocks:
            out.attrs = dict(blocks[0][1].attrs)
        return out

    return table


def _cause_specific_checks() -> tuple[Check, ...]:
    """The Cox family's checks, run cause by cause."""
    return tuple(
        Check(
            c.name,
            "For each cause's own Cox model: " + c.assumption[0].lower()
            # (the assumption, as the Cox family words it)
            + c.assumption[1:],
            rows=None if c.rows is None else _per_cause_rows(c),
            panels=None if c.panels is None else _per_cause_panels(c),
            table=None if c.table is None else _per_cause_table(c),
        )
        for c in _cox.COX.checks
        # Not calibration (#881): a cause's exp(-H_k(t)) is not the
        # probability of being free of that cause when others compete, so
        # its predicted "risk" has no observed counterpart to calibrate.
        if c.name != "calibration"
    )


def _validate_cause_specific(model: Any) -> None:
    models = getattr(model, "models", None)
    if models is None:
        raise _restored("CompetingRisksProportionalHazards model")
    for cause, cause_model in models.items():
        try:
            _cox._validate(cause_model)
        except (ValueError, NotImplementedError) as error:
            raise _relabel(error, cause) from None


# -- advice -----------------------------------------------------------------
def _fit_alone(cause: Any, rest: str = "...") -> str:
    """The call that fits cause ``cause`` on its own, the other causes
    censored (``e`` the causes given to the fit)."""
    return "CoxPH.fit(x, Z, c=np.where(e == {}, 0, 1), {})".format(
        _cause_name(cause), rest
    )


def _cs_stratify(row: dict, ctx: Context) -> str:
    name = row["covariates"][0]
    var = _source(name)
    cause = row["_cause"]
    return (
        "The effect of {name} on the hazard of cause {cause} changes with "
        "time (it fails that cause's proportional-hazards test). Stratify "
        "that cause's model on {var}: CompetingRisksProportionalHazards "
        "takes no strata, so fit the cause on its own, with the other "
        "causes censored, {alone}, with {var} left out of Z "
        "(model.models[{cause}] is that fit without the "
        "strata). Each level then has its own baseline hazard of cause "
        "{cause}, and the model no longer estimates an effect of {var} on "
        "it.".format(
            name=name,
            var=var,
            cause=_cause_name(cause),
            alone=_fit_alone(cause, f"strata=<the {var} column>"),
        )
    )


def _cs_time_varying(row: dict, ctx: Context) -> str:
    name = row["covariates"][0]
    cause = row["_cause"]
    return (
        "The effect of {name} on the hazard of cause {cause} changes with "
        "time (it fails that cause's proportional-hazards test). Fit that "
        "cause on its own, with the other causes censored (c = 0 for its "
        "events, 1 otherwise), and give {name} a time-varying effect: "
        "split each unit's follow-up where its scaled Schoenfeld residuals "
        "in d.plot('proportional_hazards') bend and fit CoxPH.fit_tvc on "
        "the pieces, with one column per period holding {name} in that "
        "period and 0 elsewhere. Or bin {name} (with pandas.cut, say) and "
        "stratify on the bins, {alone}.".format(
            name=name,
            cause=_cause_name(cause),
            alone=_fit_alone(cause, "strata=<the bins>"),
        )
    )


def _cs_global_only(row: dict, ctx: Context) -> bool:
    if not row.get("_global"):
        return False
    sub = _context_of(ctx, row["_cause"])
    return not _cox._ph_rows_failed(sub, row["transform"])


def _cs_global(row: dict, ctx: Context) -> str:
    return (
        "The joint (GLOBAL) proportional-hazards test of cause {cause} "
        "fails on the {tr} scale though no single covariate does. The "
        "covariates whose scaled Schoenfeld residuals trend in "
        "d.plot('proportional_hazards') for that cause are the likeliest "
        "cause: fit the cause on its own, {alone}, stratified on them "
        "(strata=) or with a time-varying effect (CoxPH.fit_tvc).".format(
            cause=_cause_name(row["_cause"]),
            tr=row["transform"],
            alone=_fit_alone(row["_cause"]),
        )
    )


def _any_cause_fails_ph(row: Any, ctx: Context) -> bool:
    for _, sub in _cause_contexts(ctx):
        for transform in _cox._transforms(sub):
            res = _cox._zph(sub, transform)
            if _cox._ph_rows_failed(sub, transform):
                return True
            if res["global"]["p_value"] < ctx.alpha:
                return True
    return False


def _cs_fine_gray(row: Any, ctx: Context) -> str:
    return (
        "If the question is the incidence of a cause (the chance of "
        "failing from it by a time) rather than its rate, model the "
        "subdistribution hazard instead: "
        'CompetingRisksProportionalHazards.fit(..., model="Fine-Gray"), '
        "or FineGray.fit(..., event=...) for one cause. It assumes "
        "proportional subdistribution hazards, which its own diagnose() "
        "tests; a cause that fails here may pass there, and the reverse."
    )


def _cs_form(row: dict, ctx: Context) -> str:
    name = row["covariates"][0]
    return (
        "{name} does not act linearly on the log hazard of cause {cause} "
        "(its square is significant; see d.plot('functional_form')). "
        "Transform it in a formula, for every cause, "
        "CompetingRisksProportionalHazards.fit_from_df(..., "
        'formula="np.log({name}) + ..."), or bin it (with pandas.cut, '
        'say) and fit the bins as a categorical covariate, formula="C('
        '{name}_band) + ...".'.format(
            name=name, cause=_cause_name(row["_cause"])
        )
    )


def _cs_influence(row: dict, ctx: Context) -> str:
    cause = _cause_name(row["_cause"])
    return (
        "Row {r} has a large influence on the coefficient of {name} for "
        "cause {cause}: dropping it would move the coefficient by about "
        "{size} standard errors. Check that row, and refit without it to "
        "see how much the conclusions depend on it; "
        "model.models[{cause}].compute_residuals('dfbeta') gives every "
        "row's influence on that cause.".format(
            r=row["_row"],
            name=row["covariates"][0],
            cause=cause,
            size=_fmt(row["_size"]),
        )
    )


def _cs_outliers(row: dict, ctx: Context) -> str:
    rows = row["_rows"]
    shown = ", ".join(str(i) for i in rows[:10])
    if len(rows) > 10:
        shown += f" and {len(rows) - 10} more"
    cause = _cause_name(row["_cause"])
    return (
        "The model of cause {cause} predicts row{s} {shown} poorly "
        "(|deviance residual| above {cut}). Check {them} for recording "
        "errors; model.models[{cause}].compute_residuals('deviance') gives "
        "every row's residual.".format(
            cause=cause,
            s="" if len(rows) == 1 else "s",
            shown=shown,
            cut=_fmt(row["_cutoff"]),
            them="it" if len(rows) == 1 else "them",
        )
    )


def _cs_notes(ctx: Context) -> list[str]:
    notes = [
        _INDEPENDENCE_NOTE,
        "Each cause's checks are the Cox model's (see the Cox "
        "proportional hazards diagnostics) on that cause's own CoxPH fit, "
        "with the other causes censored: model.models[cause]. "
        "Proportional hazards is the score test of R's survival::cox.zph "
        "on coxph(Surv(time, status == cause) ~ ...), on the {} scales; "
        "a covariate fails for a cause if it fails on any; "
        "d.tables['proportional_hazards'] is the test on the {} scale as "
        "cox.zph prints it, by cause. ".format(
            _cox._scales(ctx), _cox._transform(ctx)
        )
        + "With several causes there are more rows, and so more that fail "
        "by chance at alpha.",
        _CHOICE_NOTE,
        "Overall fit is a plot only (d.plot('overall_fit')): a Cox "
        "model's Cox-Snell residuals sum to its number of events by "
        "construction, so they have no useful test.",
    ]
    ties = {
        str(getattr(m, "tie_method", "efron")).lower()
        for m in ctx.model.models.values()
    }
    if ties - {"efron", "breslow"}:
        notes.append(
            "With tie_method={!r} the residuals and tests use Breslow's "
            "risk sets, as R's cox.zph does for its exact method; the "
            "functional-form test is then an approximation.".format(
                sorted(ties)[0]
            )
        )
    return notes


def _is_cause_specific(model: Any) -> bool:
    return isinstance(model, _crph_class()) and (
        getattr(model, "model", None) == "Cox"
    )


#: The cause-specific Cox family: the Cox family's checks, by cause.
CAUSE_SPECIFIC = register(
    Family(
        name="Competing risks (cause-specific Cox)",
        issue=874,
        applies=_is_cause_specific,
        validate=_validate_cause_specific,
        covariates=_crph_names,
        columns=("cause", "transform"),
        options=("transform",),
        checks=_cause_specific_checks(),
        advice=(
            AdviceRule(
                "proportional_hazards",
                _cs_stratify,
                ("CoxPH.fit(strata=)",),
                when=lambda row, ctx: row.get("_kind")
                in ("binary", "categorical"),
            ),
            AdviceRule(
                "proportional_hazards",
                _cs_time_varying,
                ("CoxPH.fit_tvc", "CoxPH.fit(strata=)"),
                when=lambda row, ctx: row.get("_kind") == "continuous",
            ),
            AdviceRule(
                "proportional_hazards",
                _cs_global,
                ("CoxPH.fit(strata=)", "CoxPH.fit_tvc"),
                when=_cs_global_only,
            ),
            AdviceRule(
                None,
                _cs_fine_gray,
                (
                    "CompetingRisksProportionalHazards.fit(model=)",
                    "FineGray.fit",
                ),
                when=_any_cause_fails_ph,
            ),
            AdviceRule(
                "functional_form",
                _cs_form,
                ("CompetingRisksProportionalHazards.fit_from_df(formula=)",),
            ),
            AdviceRule(
                "influence",
                _cs_influence,
                ("compute_residuals(kind=)",),
                on=("warn",),
            ),
            AdviceRule(
                "outliers",
                _cs_outliers,
                ("compute_residuals(kind=)",),
                on=("warn",),
            ),
        ),
        notes=_cs_notes,
    )
)


# -- the Fine-Gray family's checks ------------------------------------------
def _fg_models(model: Any) -> dict:
    """Each cause's ``FineGrayModel``, by cause."""
    if isinstance(model, _fg_class()):
        return {model.cause: model}
    models = getattr(model, "models", None)
    if models is None:
        raise _restored("CompetingRisksProportionalHazards model")
    return dict(models)


def _fg_causes(ctx: Context) -> list[tuple[Any, FineGrayData]]:
    return ctx.cached(
        "fg_data",
        lambda: [
            (cause, fine_gray_data(m))
            for cause, m in _fg_models(ctx.model).items()
        ],
    )


def _fg_zph_cached(ctx: Context, cause: Any, transform: str) -> dict:
    def compute() -> dict:
        fd = dict(_fg_causes(ctx))[cause]
        return fine_gray_zph_test(fd, transform)

    return ctx.cached(f"fg_zph_{transform}_{cause!r}", compute)


def _fg_kind(fd: FineGrayData, j: int) -> str:
    return _covariate_kind(np.unique(fd.Z_raw[:, j]).size)


def _fg_trend(res: dict, j: int, fd: FineGrayData) -> float:
    g, scaled = res["g"], res["scaled_schoenfeld"][:, j]
    w = fd.n[fd.events]
    return _trend_sign(g, scaled, w)


def _fg_ph_rows(ctx: Context) -> list[dict]:
    rows = []
    for cause, fd in _fg_causes(ctx):
        for transform in FG_TRANSFORMS:
            res = _fg_zph_cached(ctx, cause, transform)
            for j, name in ctx.checked:
                entry = res["per_covariate"][j]
                p = entry["p_value"]
                if not np.isfinite(p):
                    detail = "the test could not be computed on these data"
                elif p < 2 * ctx.alpha:
                    detail = (
                        "its subdistribution coefficient {} with time".format(
                            "rises" if _fg_trend(res, j, fd) > 0 else "falls"
                        )
                    )
                else:
                    detail = (
                        "no evidence that its coefficient changes with time"
                    )
                rows.append(
                    {
                        "test": "weighted Grambsch-Therneau score test",
                        "cause": _label(cause),
                        "transform": transform,
                        "covariates": (name,),
                        "statistic": entry["statistic"],
                        "df": 1,
                        "p": p,
                        "detail": detail,
                        "_cause": cause,
                        "_kind": _fg_kind(fd, j),
                    }
                )
            glob = res["global"]
            rows.append(
                {
                    "test": "weighted Grambsch-Therneau score test (GLOBAL)",
                    "cause": _label(cause),
                    "transform": transform,
                    "covariates": tuple(ctx.names),
                    "statistic": glob["statistic"],
                    "df": glob["df"],
                    "p": glob["p_value"],
                    "detail": "the joint test of every covariate",
                    "_cause": cause,
                    "_global": True,
                }
            )
    return rows


def _cif_axis(ax: Any, g: np.ndarray, t: np.ndarray) -> None:
    _cox._time_axis(ax, g, t)
    ax.set_xlabel("time (cumulative incidence scale)")


def _fg_ph_panels(ctx: Context) -> list[Panel]:
    from .smooth import bootstrap_band, lowess

    panels = []
    for cause, fd in _fg_causes(ctx):
        res = _fg_zph_cached(ctx, cause, "cif")
        g, t = res["g"], res["time"]
        scaled = res["scaled_schoenfeld"]
        for j, name in ctx.checked:

            def draw(
                ax: Any,
                options: Any,
                j: int = j,
                g: np.ndarray = g,
                t: np.ndarray = t,
                scaled: np.ndarray = scaled,
                beta: float = float(fd.beta[j]),
            ) -> None:
                from surpyval.plotting import BAND_ALPHA, INK, PALETTE

                grid = np.linspace(g.min(), g.max(), 60)
                y = scaled[:, j]
                ax.scatter(g, y, s=10, color=INK["muted"], alpha=0.6)
                ax.plot(grid, lowess(g, y, grid), color=PALETTE[0])
                if options.n_boot > 0:
                    band = bootstrap_band(
                        g,
                        y,
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
                    )
                ax.axhline(beta, color=PALETTE[1], ls="--")
                _cif_axis(ax, g, t)
                ax.set_ylabel("beta(t), scaled Schoenfeld")

            p = res["per_covariate"][j]["p_value"]
            panels.append(
                Panel(
                    "cause {}: {}: p = {} (cif)".format(
                        _label(cause), name, _fmt(p)
                    ),
                    draw,
                )
            )
    return panels


def _fg_form_rows(ctx: Context) -> list[dict]:
    rows = []
    for cause, fd in _fg_causes(ctx):
        skipped = [n for j, n in ctx.checked if _fg_kind(fd, j) == "binary"]
        if skipped:
            note = (
                "Functional form is tested for covariates with three or "
                "more distinct values; {} {} two, and the effect of a "
                "two-valued covariate is linear by construction.".format(
                    ", ".join(skipped),
                    "has" if len(skipped) == 1 else "have",
                )
            )
            if note not in ctx.notes:
                ctx.notes.append(note)
        for j, name in ctx.checked:
            if _fg_kind(fd, j) == "binary":
                continue
            res = fine_gray_added_covariate_test(fd, fd.Z_raw[:, j] ** 2)
            p = res["p_value"]
            if not np.isfinite(p):
                detail = "the test could not be computed"
            elif p < 2 * ctx.alpha:
                detail = (
                    "its effect on the log subdistribution hazard is not "
                    "linear"
                )
            else:
                detail = "no evidence against a linear effect"
            rows.append(
                {
                    "test": "score test of a quadratic term",
                    "cause": _label(cause),
                    "covariates": (name,),
                    "statistic": res["statistic"],
                    "df": res["df"],
                    "p": p,
                    "detail": detail,
                    "_cause": cause,
                }
            )
    return rows


def _fg_form_panels(ctx: Context) -> list[Panel]:
    from .smooth import lowess

    panels = []
    for cause, fd in _fg_causes(ctx):
        martingale = ctx.cached(
            f"fg_martingale_{cause!r}",
            lambda: fine_gray_martingale_residuals(fd),
        )
        for j, name in ctx.checked:
            if _fg_kind(fd, j) == "binary":
                continue

            def draw(
                ax: Any,
                options: Any,
                z: np.ndarray = fd.Z_raw[:, j],
                m: np.ndarray = martingale,
                name: str = name,
            ) -> None:
                from surpyval.plotting import INK, PALETTE

                grid = np.linspace(z.min(), z.max(), 60)
                ax.scatter(z, m, s=10, color=INK["muted"], alpha=0.6)
                ax.plot(grid, lowess(z, m, grid), color=PALETTE[0])
                ax.axhline(0.0, color=INK["axis"], lw=0.8)
                ax.set_xlabel(name)
                ax.set_ylabel("subdistribution martingale residual")

            panels.append(
                Panel(
                    "cause {}: {}: functional form".format(
                        _label(cause), name
                    ),
                    draw,
                )
            )
    return panels


def _validate_fine_gray(model: Any) -> None:
    for cause, fg in _fg_models(model).items():
        if getattr(fg, "_fit_data", None) is None:
            raise _restored(
                "Fine-Gray model"
                if isinstance(model, _fg_class())
                else "CompetingRisksProportionalHazards model"
            )
        if getattr(fg, "maximum", None) == "no finite maximum":
            raise _relabel(
                ValueError(
                    "diagnose() needs a fit with a finite maximum, and this "
                    "one has none (a covariate, or a combination of them, "
                    "separates the events of the cause from the rest of "
                    "their risk sets, as the fit warned). Drop or merge "
                    "the separating covariate and refit."
                ),
                cause,
            )
        if np.isnan(np.asarray(fg.beta, dtype=float)).any():
            try:
                fine_gray_data(fg)
            except ValueError as error:
                raise _relabel(error, cause) from None


def _fg_names(model: Any) -> list[str]:
    if isinstance(model, _fg_class()):
        from surpyval.univariate.regression._summary import (
            coefficient_names,
        )

        return list(coefficient_names(model, np.size(model.beta)))
    return _crph_names(model)


# -- advice -----------------------------------------------------------------
def _fg_ph(row: dict, ctx: Context) -> str:
    name = row["covariates"][0]
    cause = _cause_name(row["_cause"])
    return (
        "The subdistribution hazard ratio of {name} for cause {cause} "
        "changes with time (it fails the proportional subdistribution "
        "hazards test): exp(coef) is then a time-averaged effect on the "
        "incidence, and the model's cumulative incidences for different "
        "values of {name} are off where the residuals in "
        "d.plot('proportional_hazards') stray from the fitted line. If "
        "the question is the rate of the cause rather than its incidence, "
        "fit the cause-specific model, CompetingRisksProportionalHazards"
        '.fit(..., model="Cox"), whose own diagnose() checks it and '
        "advises stratification or a time-varying effect. To describe the "
        "incidence by levels of {name} without the assumption, estimate "
        "it non-parametrically with CompetingRisks.fit for each level, "
        "and compare the levels with gray_test.".format(name=name, cause=cause)
    )


def _fg_global_only(row: dict, ctx: Context) -> bool:
    if not row.get("_global"):
        return False
    res = _fg_zph_cached(ctx, row["_cause"], row["transform"])
    return not any(
        res["per_covariate"][j]["p_value"] < ctx.alpha for j, _ in ctx.checked
    )


def _fg_global(row: dict, ctx: Context) -> str:
    return (
        "The joint (GLOBAL) proportional subdistribution hazards test of "
        "cause {cause} fails on the {tr} scale though no single covariate "
        "does. The covariates whose scaled residuals trend in "
        "d.plot('proportional_hazards') are the likeliest cause; if the "
        "question is the cause's rate, the cause-specific model, "
        'CompetingRisksProportionalHazards.fit(..., model="Cox"), '
        "may suit better.".format(
            cause=_cause_name(row["_cause"]), tr=row["transform"]
        )
    )


def _fg_form(row: dict, ctx: Context) -> str:
    name = row["covariates"][0]
    return (
        "{name} does not act linearly on the log subdistribution hazard "
        "of cause {cause} (its square is significant; see "
        "d.plot('functional_form')). Transform it in a formula, "
        "CompetingRisksProportionalHazards.fit_from_df(..., "
        'model="Fine-Gray", formula="np.log({name}) + ..."), or bin it '
        "(with pandas.cut, say) and fit the bins as a categorical "
        'covariate, formula="C({name}_band) + ...".'.format(
            name=name, cause=_cause_name(row["_cause"])
        )
    )


def _fg_notes(ctx: Context) -> list[str]:
    return [
        _INDEPENDENCE_NOTE,
        "Proportional subdistribution hazards: the score test of adding "
        "Z * g(t) to the Fine-Gray weighted partial likelihood, with the "
        "censoring weights G(t-)/G(x_i-) held as fitted, on the cif scale "
        "(g the cause's Aalen-Johansen cumulative incidence just before t) "
        "and the rank scale; a covariate fails if it fails on either. It "
        "is R's survival::cox.zph on the weighted counting-process data "
        "of the fit (survival::finegray's construction), with Breslow's "
        "ties, as the fit has.",
        "The p-values use the model-based information of the weighted "
        "partial likelihood, as the model's standard errors do, not Fine "
        "and Gray's robust variance.",
        _CHOICE_NOTE,
        "Influence, outliers and overall fit are not checked for a "
        "Fine-Gray model; the cause-specific model (model='Cox') has "
        "them.",
    ]


def _is_fine_gray(model: Any) -> bool:
    if isinstance(model, _fg_class()):
        return True
    return isinstance(model, _crph_class()) and (
        getattr(model, "model", None) == "Fine-Gray"
    )


#: The Fine-Gray family: proportional subdistribution hazards and the
#: functional form, by cause.
FINE_GRAY = register(
    Family(
        name="Competing risks (Fine-Gray)",
        issue=874,
        applies=_is_fine_gray,
        validate=_validate_fine_gray,
        covariates=_fg_names,
        columns=("cause", "transform"),
        checks=(
            Check(
                "proportional_hazards",
                "Each covariate multiplies the subdistribution hazard of "
                "each cause by a factor that does not change with time.",
                rows=_fg_ph_rows,
                panels=_fg_ph_panels,
            ),
            Check(
                "functional_form",
                "Each covariate acts linearly on the log subdistribution "
                "hazard.",
                rows=_fg_form_rows,
                panels=_fg_form_panels,
            ),
        ),
        advice=(
            AdviceRule(
                "proportional_hazards",
                _fg_ph,
                (
                    "CompetingRisksProportionalHazards.fit(model=)",
                    "CompetingRisks.fit",
                    "gray_test",
                ),
                when=lambda row, ctx: not row.get("_global"),
            ),
            AdviceRule(
                "proportional_hazards",
                _fg_global,
                ("CompetingRisksProportionalHazards.fit(model=)",),
                when=_fg_global_only,
            ),
            AdviceRule(
                "functional_form",
                _fg_form,
                (
                    "CompetingRisksProportionalHazards.fit_from_df(formula=)",
                    "CompetingRisksProportionalHazards.fit_from_df(model=)",
                ),
            ),
        ),
        notes=_fg_notes,
    )
)


# ===========================================================================
# Parametric: each cause's distribution, and the incidences against the
# Aalen-Johansen estimate
# ===========================================================================
def _univariate() -> Any:
    """The univariate family's module (imported when first needed, so
    that the families register in the order of ``FAMILY_MODULES``)."""
    from . import univariate

    return univariate


def _pcr_class() -> Any:
    from surpyval.univariate.competing_risks import ParametricCompetingRisks

    return ParametricCompetingRisks


def _counts(data: dict, failed: bool) -> dict:
    """The rows of one cause's data that failed (``c == 0``) or were
    censored (``c == 1``), as the units at each ``(x, tl, tr)``."""
    c = np.asarray(data["c"])
    x = np.asarray(data["x"], dtype=float).reshape(c.size, -1)[:, 0]
    n = np.asarray(data["n"], dtype=float)
    t = np.asarray(data["t"], dtype=float).reshape(-1, 2)
    keep = (c == 0) if failed else (c == 1)
    out: dict = {}
    for key in zip(x[keep], t[keep, 0], t[keep, 1], n[keep]):
        where = (float(key[0]), float(key[1]), float(key[2]))
        out[where] = out.get(where, 0.0) + float(key[3])
    return out


def _combine(first: dict, *others: dict, sign: float = 1.0) -> dict:
    out = dict(first)
    for other in others:
        for key, value in other.items():
            out[key] = out.get(key, 0.0) + sign * value
    return {k: v for k, v in out.items() if abs(v) > 1e-9}


def pcr_data(model: Any) -> dict:
    """
    The data a ``ParametricCompetingRisks`` model was fitted to, rebuilt
    from its causes' fits (each keeps the data with the other causes
    censored): ``x``, ``e`` (the cause of each row, ``None`` where
    censored), ``c``, ``n`` and ``t`` (each row's ``[tl, tr]``), sorted by
    ``x``. Raises ``ValueError`` when the causes' models were not fitted
    to the same units (a model assembled with ``from_fitted`` from
    separate fits).

    Examples
    --------
    >>> from surpyval import Exponential
    >>> from surpyval.univariate.competing_risks import (
    ...     ParametricCompetingRisks,
    ... )
    >>> from surpyval.diagnostics.competing_risks import pcr_data
    >>> model = ParametricCompetingRisks.fit(
    ...     [1, 2, 2, 3, 5], ["a", "b", "a", None, "b"], dist=Exponential
    ... )
    >>> data = pcr_data(model)
    >>> data["x"].tolist(), data["e"].tolist()
    ([1.0, 2.0, 2.0, 3.0, 5.0], ['a', 'a', 'b', None, 'b'])
    """
    causes = list(model.causes)
    events = {k: _counts(model.models[k].data, True) for k in causes}
    censored = _combine(
        _counts(model.models[causes[0]].data, False),
        *(events[j] for j in causes[1:]),
        sign=-1.0,
    )
    consistent = all(v > 0 for v in censored.values()) and all(
        not _combine(
            _counts(model.models[k].data, False),
            censored,
            *(events[j] for j in causes if j != k),
            sign=-1.0,
        )
        for k in causes
    )
    if not consistent:
        raise ValueError(
            "diagnose() compares the causes on the units they were fitted "
            "to, and this model's causes were fitted to different data "
            "(assembled with from_fitted from separate fits). Diagnose "
            "each cause's model on its own, model.models[cause].diagnose(), "
            "or refit with ParametricCompetingRisks.fit on the shared data."
        )
    rows = [
        (key, k, 0, count) for k in causes for key, count in events[k].items()
    ]
    rows += [(key, None, 1, count) for key, count in censored.items()]
    rows.sort(key=lambda r: r[0][0])
    e = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        e[i] = row[1]
    return {
        "x": np.array([r[0][0] for r in rows], dtype=float),
        "e": e,
        "c": np.array([r[2] for r in rows], dtype=int),
        "n": np.array([r[3] for r in rows], dtype=float),
        "t": np.array([[r[0][1], r[0][2]] for r in rows], dtype=float).reshape(
            -1, 2
        ),
    }


def _right_truncated(data: dict) -> bool:
    """Whether any row has a finite right-truncation bound (only from
    causes assembled with ``from_fitted`` from fits that had one)."""
    return bool(np.any(np.isfinite(np.asarray(data["t"])[:, 1])))


def _entry(data: dict) -> np.ndarray | None:
    """The rows' entry times (``-inf`` for a row with none), or None when
    no row has one."""
    tl = np.asarray(data["t"], dtype=float)[:, 0]
    return tl if np.any(np.isfinite(tl)) else None


def _origin(tl: Any) -> float | None:
    """The time the Aalen-Johansen estimate of data with entry times
    ``tl`` is conditional on surviving to, the earliest entry, or None
    when it is not conditional (no entry times, a unit with none, or the
    earliest at or before 0)."""
    if tl is None:
        return None
    tl = np.asarray(tl, dtype=float)
    s = float(np.min(tl))
    return s if np.isfinite(s) and s > 0 else None


#: The Gauss-Legendre nodes of :func:`_cif_steps`, on [-1, 1].
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(8)


def _cif_steps(model: Any, times: np.ndarray, cause: Any) -> np.ndarray:
    """``model.cif(times, cause)`` at the increasing ``times``, summed a
    step at a time: on the cause's probability scale, as ``cif``
    integrates, ``CIF_k(t) = int_0^{F_k(t)} prod_{j != k}
    S_j(F_k^{-1}(p)) dp``, with eight Gauss-Legendre nodes between each
    time and the next (and a hundred more steps across their span). The
    integrand is bounded by 1 and monotone, so this agrees with ``cif``
    to about 1e-7 (the tests check it), at a small part of the cost of an
    adaptive quadrature from 0 for every time, which the bootstrap would
    pay on every refit."""
    m = model.models[cause]
    others = [model.models[j] for j in model.causes if j != cause]
    # The steps are at the times and at a hundred more across their span,
    # so that a sparse set of times is integrated as finely as a dense one.
    times = np.asarray(times, dtype=float)
    finer = np.linspace(0.0, float(times.max()), 101)[1:]
    grid, where = np.unique(
        np.concatenate([times, finer]), return_inverse=True
    )
    with np.errstate(all="ignore"):
        top = np.asarray(m.ff(grid), dtype=float).ravel()
    top = np.maximum.accumulate(np.clip(np.nan_to_num(top), 0.0, 1.0))
    lower = np.concatenate([[0.0], top[:-1]])
    half = 0.5 * (top - lower)
    p = (0.5 * (top + lower))[:, None] + half[:, None] * _NODES[None, :]
    with np.errstate(all="ignore"):
        s = np.asarray(m.qf(p.ravel()), dtype=float)
        value = np.ones_like(s)
        for other in others:
            value = value * np.asarray(other.sf(s), dtype=float).ravel()
    # Below a model's support its survival formula can give NaN; it is 1
    # there (as cif takes it).
    value = np.where(np.isfinite(value), value, 1.0).reshape(p.shape)
    return np.cumsum(half * (value @ _WEIGHTS))[where[: times.size]]


def _conditional_cif(
    model: Any, times: np.ndarray, cause: Any, origin: float | None
) -> np.ndarray:
    """The model's cumulative incidence of ``cause`` at ``times`` (by
    :func:`_cif_steps`), conditional on surviving every cause to
    ``origin``, ``(CIF_k(t) - CIF_k(s)) / S(s)``: what the Aalen-Johansen
    estimate of data whose earliest entry is ``s`` estimates. ``origin``
    None is the unconditional incidence."""
    times = np.asarray(times, dtype=float)
    if origin is None:
        return _cif_steps(model, times, cause)
    steps = _cif_steps(model, np.concatenate([[origin], times]), cause)
    with np.errstate(all="ignore"):
        alive = float(
            np.prod(
                [
                    float(np.ravel(model.models[j].sf(origin))[0])
                    for j in model.causes
                ]
            )
        )
    return (steps[1:] - steps[0]) / alive


def incidence_distance(
    model: Any,
    x: Any,
    e: Any,
    c: Any = None,
    n: Any = None,
    tl: Any = None,
) -> dict:
    """
    For each cause of ``model``, the largest gap between its cumulative
    incidence and the Aalen-Johansen estimate of the data ``x``, ``e``,
    ``c``, ``n`` (``CompetingRisks.fit``): ``D_k = sup_t |CIF_k(t) -
    AJ_k(t)|`` from 0 to the last time observed. The estimate is a step
    function and the model's incidence rises, so the supremum is at an
    event time, on one side of its step.

    With entry times ``tl`` (delayed entry; ``-inf`` for a unit with
    none) the estimate counts each unit at risk only from its entry, and
    is conditional on surviving to the earliest entry ``s`` when every
    unit has one after 0; the model's incidence is then conditioned the
    same way, ``(CIF_k(t) - CIF_k(s)) / S(s)``, and the gap taken from
    ``s`` on.

    Returns a dictionary by cause of ``{"D": D_k, "time": t}``, ``t``
    the event time where the gap is largest.

    Examples
    --------
    >>> from surpyval import Exponential
    >>> from surpyval.univariate.competing_risks import (
    ...     ParametricCompetingRisks,
    ... )
    >>> from surpyval.diagnostics.competing_risks import incidence_distance
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> e = ["a", "b", "a", "b", "a", "b", "a", None]
    >>> model = ParametricCompetingRisks.fit(x, e, dist=Exponential)
    >>> out = incidence_distance(model, x, e)
    >>> sorted(out), round(out["a"]["D"], 4)
    (['a', 'b'], 0.1276)
    """
    from surpyval.univariate.competing_risks import CompetingRisks

    x = np.asarray(x, dtype=float)
    e = np.asarray(e, dtype=object)
    entry = {} if tl is None else {"tl": tl}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aj = CompetingRisks.fit(x, e, c=c, n=n, **entry)
    origin = _origin(tl)
    failed = (
        np.array([v is not None for v in e])
        if c is None
        else (np.asarray(c) == 0)
    )
    times = np.unique(np.concatenate([x[failed], [x.max()]]))
    out = {}
    for cause in model.causes:
        a = np.asarray(aj.cif(times, cause), dtype=float)
        before = np.concatenate([[0.0], a[:-1]])
        p = _conditional_cif(model, times, cause, origin)
        gaps = np.maximum(np.abs(p - a), np.abs(p - before))
        j = int(np.argmax(gaps))
        out[cause] = {"D": float(gaps[j]), "time": float(times[j])}
    return out


def incidence_bootstrap(
    model: Any, n_boot: int, random_state: Any = None
) -> dict:
    """
    The gap :func:`incidence_distance` of each cause of ``model`` on its
    data, and its parametric bootstrap p-value from ``n_boot`` refits.

    Each resample draws, for every unit, a latent time from each cause's
    model (through its ``qf``), keeps the earliest and its cause, and
    censors it as the data were (a censored unit at its own time, a
    failed one at a time drawn from the Kaplan-Meier estimate of the
    censoring distribution beyond its failure, as
    :func:`surpyval.diagnostics.univariate.bootstrap` does); every cause's
    model is refitted the same way (distribution, ``how`` and options) and
    the gaps recomputed. The p-value is ``(1 + #{refits at least as
    extreme}) / (1 + refits)``. With delayed entry (``tl``) each unit's
    latent times are drawn beyond its entry (from each cause's model
    conditioned on surviving to it), the causes refitted with the same
    entries, and the gaps are those of :func:`incidence_distance` with
    them. Data with right truncation are refused: the Aalen-Johansen
    estimate does not take it.

    Returns a dictionary: ``observed``, ``time`` and ``p``, each by
    cause; ``draws``, the refits' gaps, a column per cause in the order
    of ``model.causes``; ``failed``, the refits that raised and were left
    out; and ``n_boot``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import Exponential
    >>> from surpyval.univariate.competing_risks import (
    ...     ParametricCompetingRisks,
    ... )
    >>> from surpyval.diagnostics.competing_risks import incidence_bootstrap
    >>> rng = np.random.default_rng(1)
    >>> x = rng.exponential(1.0, 60)
    >>> e = rng.choice(["a", "b"], 60).astype(object)
    >>> model = ParametricCompetingRisks.fit(x, e, dist=Exponential)
    >>> out = incidence_bootstrap(model, 20, random_state=2)
    >>> sorted(out["p"]), out["draws"].shape
    (['a', 'b'], (20, 2))
    """
    from types import SimpleNamespace

    from surpyval.univariate.regression._bootstrap import (
        REFIT_ERRORS,
        check_n_boot,
    )
    from surpyval.utils.no_maximum import quiet_maximum_warnings
    from surpyval.utils.rng import as_generator

    uni = _univariate()
    n_boot = check_n_boot(n_boot)
    data = pcr_data(model)
    if _right_truncated(data):
        raise ValueError(
            "The incidence check needs the Aalen-Johansen estimate of the "
            "data, which does not take right truncation (tr); these data "
            "have it. Check each cause with diagnose()'s goodness_of_fit "
            "rows, which allow for truncation."
        )
    causes = list(model.causes)
    entry = _entry(data)
    found = incidence_distance(
        model, data["x"], data["e"], data["c"], data["n"], tl=entry
    )
    design = uni._design(SimpleNamespace(data=data))
    rng = as_generator(random_state)
    longest = float(np.max(design.x))
    options = [uni._refit_options(model.models[k]) for k in causes]
    size = design.x.size
    below: list[np.ndarray | None] = [None for _ in causes]
    refit_entry: dict[str, Any] = {}
    if entry is not None:
        # Each cause's CDF at each unit's entry: its latent time is drawn
        # beyond it (the unit was seen because it survived every cause to
        # its entry, and the latent times are independent).
        entered = np.isfinite(design.tl)
        with np.errstate(all="ignore"):
            below = [
                np.where(
                    entered,
                    np.asarray(
                        model.models[k].ff(np.nan_to_num(design.tl)),
                        dtype=float,
                    ).ravel(),
                    0.0,
                )
                for k in causes
            ]
        refit_entry = {"tl": design.tl}
    draws = []
    failed = 0
    for _ in range(n_boot):
        with np.errstate(all="ignore"):
            latent = np.column_stack(
                [
                    np.asarray(
                        model.models[k].qf(
                            rng.uniform(size=size)
                            if lo is None
                            else lo + rng.uniform(size=size) * (1.0 - lo)
                        ),
                        dtype=float,
                    ).ravel()
                    for k, lo in zip(causes, below)
                ]
            )
        latent = np.where(np.isnan(latent), np.inf, latent)
        first = np.argmin(latent, axis=1)
        T = latent[np.arange(size), first]
        C = design.censoring(rng)
        censored = ~(T <= C)
        x = np.where(censored, C, T)
        # A unit that never fails (a limited failure population) and is
        # not censored by the censoring distribution is censored at the
        # last time observed.
        x = np.where(np.isfinite(x), x, longest)
        e = first.astype(object)
        e[censored] = None
        try:
            with warnings.catch_warnings(), quiet_maximum_warnings():
                warnings.simplefilter("ignore")
                fits = {
                    j: model.models[k].dist.fit(
                        x,
                        c=np.where((first == j) & ~censored, 0, 1),
                        **refit_entry,
                        **options[j],
                    )
                    for j, k in enumerate(causes)
                }
                refit = _pcr_class().from_fitted(fits)
                gaps = incidence_distance(
                    refit, x, e, censored.astype(int), **refit_entry
                )
        except REFIT_ERRORS:
            failed += 1
            continue
        draws.append([gaps[j]["D"] for j in range(len(causes))])
    values = np.array(draws, dtype=float).reshape(len(draws), len(causes))
    p = {}
    for j, cause in enumerate(causes):
        if len(draws) < n_boot / 2:
            p[cause] = float("nan")
        else:
            extreme = np.sum(values[:, j] >= found[cause]["D"] - 1e-12)
            p[cause] = float((1 + extreme) / (1 + len(draws)))
    return {
        "observed": {k: found[k]["D"] for k in causes},
        "time": {k: found[k]["time"] for k in causes},
        "p": p,
        "draws": values,
        "failed": failed,
        "n_boot": n_boot,
    }


def _pcr_seeds(ctx: Context) -> list[int]:
    """One seed for each cause's goodness-of-fit bootstrap and one for the
    incidence bootstrap, drawn from ``diagnose(random_state=)``."""

    def compute() -> list[int]:
        from surpyval.utils.rng import as_generator

        rng = as_generator(ctx.random_state)
        size = len(ctx.model.causes) + 1
        return [int(s) for s in rng.integers(0, 2**63 - 1, size=size)]

    return ctx.cached("pcr_seeds", compute)


def _pcr_causes(ctx: Context) -> list[tuple[Any, Context]]:
    """Each cause and the univariate context of its model, with the
    report's level, ``n_boot`` and a seed of its own."""

    def build() -> list[tuple[Any, Context]]:
        seeds = _pcr_seeds(ctx)
        return [
            (
                cause,
                Context(
                    ctx.model.models[cause],
                    ctx.alpha,
                    [],
                    [],
                    n_boot=ctx.n_boot,
                    random_state=seed,
                ),
            )
            for cause, seed in zip(ctx.model.causes, seeds)
        ]

    return ctx.cached("pcr_causes", build)


def _pcr_data(ctx: Context) -> dict:
    return ctx.cached("pcr_data", lambda: pcr_data(ctx.model))


def _univariate_check(name: str) -> Check:
    for check in _univariate().UNIVARIATE.checks:
        if check.name == name:
            return check
    raise KeyError(name)


def _by_cause_rows(name: str) -> Any:
    """The rows of the univariate check ``name`` for each cause's model,
    each with its ``cause``."""

    def rows(ctx: Context) -> list[dict]:
        check = _univariate_check(name)
        assert check.rows is not None
        out = []
        for cause, sub in _pcr_causes(ctx):
            for row in check.rows(sub):
                out.append(dict(row, cause=_label(cause), _cause=cause))
        return out

    return rows


def _by_cause_panels(name: str) -> Any:
    """The panels of the univariate check ``name`` for each cause's
    model, each titled with its cause."""

    def panels(ctx: Context) -> list[Panel]:
        check = _univariate_check(name)
        assert check.panels is not None
        out = []
        for cause, sub in _pcr_causes(ctx):
            for panel in check.panels(sub):
                out.append(
                    Panel(f"cause {_label(cause)}: {panel.title}", panel.draw)
                )
        return out

    return panels


def _incidence(ctx: Context) -> dict | None:
    """The incidence bootstrap of the report, or None for data with
    right truncation."""
    if _right_truncated(_pcr_data(ctx)):
        return None
    uni = _univariate()
    n_boot = uni.DEFAULT_N_BOOT if ctx.n_boot is None else ctx.n_boot
    seed = _pcr_seeds(ctx)[-1]
    return ctx.cached(
        "pcr_incidence",
        lambda: incidence_bootstrap(ctx.model, n_boot, seed),
    )


_INCIDENCE_TEST = "sup |CIF - Aalen-Johansen| (parametric bootstrap)"


def _incidence_rows(ctx: Context) -> list[dict]:
    boot = _incidence(ctx)
    rows = []
    for cause in ctx.model.causes:
        row = {
            "test": _INCIDENCE_TEST,
            "cause": _label(cause),
            "alternative": "",
            "_cause": cause,
        }
        if boot is None:
            row.update(
                statistic=float("nan"),
                p=float("nan"),
                detail=(
                    "not computed: the data have right truncation (tr), "
                    "which the Aalen-Johansen estimate does not take"
                ),
            )
            rows.append(row)
            continue
        D, p = boot["observed"][cause], boot["p"][cause]
        kept = boot["n_boot"] - boot["failed"]
        if not np.isfinite(p):
            detail = (
                "not computed: {} of the {} bootstrap refits failed".format(
                    boot["failed"], boot["n_boot"]
                )
            )
        elif p < 2 * ctx.alpha:
            detail = (
                "the parametric incidence is {} from the Aalen-Johansen "
                "estimate at time {} ({} refits)".format(
                    _fmt(D), _fmt(boot["time"][cause]), kept
                )
            )
        else:
            detail = (
                "no evidence against the parametric incidence (largest gap "
                "{} at time {}; {} refits)".format(
                    _fmt(D), _fmt(boot["time"][cause]), kept
                )
            )
        row.update(statistic=D, p=p, detail=detail, _time=boot["time"][cause])
        rows.append(row)
    if boot is not None and boot["failed"]:
        _note(
            ctx,
            "{} of the {} incidence bootstrap refits failed and were left "
            "out of its p-values.".format(boot["failed"], boot["n_boot"]),
        )
    return rows


def _incidence_panels(ctx: Context) -> list[Panel]:
    from surpyval.univariate.competing_risks import CompetingRisks

    data = _pcr_data(ctx)
    if _right_truncated(data):
        return []
    tl = _entry(data)
    entry = {} if tl is None else {"tl": tl}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aj = CompetingRisks.fit(
            data["x"], data["e"], c=data["c"], n=data["n"], **entry
        )
    origin = _origin(tl)
    start = 0.0 if origin is None else origin
    end = float(np.max(data["x"]))
    times = np.unique(
        np.concatenate([[start], data["x"][data["c"] == 0], [end]])
    )
    boot = _incidence(ctx)
    panels = []
    for cause in ctx.model.causes:

        def draw(ax: Any, options: Any, cause: Any = cause) -> None:
            from surpyval.plotting import BAND_ALPHA, INK, PALETTE

            step = np.asarray(aj.cif(times, cause), dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                band = np.asarray(
                    aj.cb(
                        times, event=cause, on="cif", alpha_ci=options.alpha
                    ),
                    dtype=float,
                )
            ax.fill_between(
                times,
                band[:, 0],
                band[:, 1],
                step="post",
                color=INK["muted"],
                alpha=BAND_ALPHA,
                linewidth=0,
            )
            ax.step(
                times,
                step,
                where="post",
                color=INK["secondary"],
                label="Aalen-Johansen",
            )
            grid = np.linspace(start, end, 200)
            if origin is None:
                curve = np.asarray(ctx.model.cif(grid, cause), dtype=float)
            else:
                curve = _conditional_cif(ctx.model, grid, cause, origin)
            ax.plot(
                grid,
                curve,
                color=PALETTE[0],
                label=ctx.model.models[cause].dist.name,
            )
            ax.set_xlabel("time")
            ax.set_ylabel(
                "cumulative incidence"
                if origin is None
                else "cumulative incidence given survival to {}".format(
                    _fmt(origin)
                )
            )
            ax.legend(fontsize="small")

        p = float("nan") if boot is None else boot["p"][cause]
        panels.append(
            Panel(
                f"cause {_label(cause)}: incidence (p = {_fmt(p)})",
                draw,
            )
        )
    return panels


def _validate_parametric(model: Any) -> None:
    from surpyval.univariate.parametric import Parametric

    models = getattr(model, "models", None)
    if not models:
        raise _restored("ParametricCompetingRisks model")
    if getattr(model, "_fitted_jointly", False):
        raise NotImplementedError(
            "diagnose() does not yet check a ParametricCompetingRisks model "
            "fitted to left- or interval-censored rows (c = -1 or 2). Its "
            "causes are fitted together and keep no data of their own, and "
            "every check here (each cause's Cox-Snell residuals, its "
            "probability plot and alternatives, and the Aalen-Johansen "
            "comparison of the incidences) needs each failure's exact "
            "time. Instead, compare distributions by refitting with "
            "another dist= and comparing model.aic(), and compare the "
            "all-cause survival model.sf(t) with the nonparametric "
            "Turnbull estimate of the same data, "
            "surpyval.Turnbull.fit(x, c=c, n=n, tl=tl), which takes the "
            "censored windows (but not the causes)."
        )
    for cause, cause_model in models.items():
        if not isinstance(cause_model, Parametric):
            raise _relabel(
                NotImplementedError(
                    "diagnose() checks each cause's fitted distribution, and "
                    "this cause's model is a {} (assembled with "
                    "from_fitted). Diagnose it on its own where it has "
                    "diagnose(), and compare model.cif with the "
                    "nonparametric CompetingRisks.fit of the same "
                    "data.".format(type(cause_model).__name__)
                ),
                cause,
            )
        if getattr(cause_model, "data", None) is None:
            raise _restored("ParametricCompetingRisks model")
        try:
            _univariate()._validate(cause_model)
        except (ValueError, NotImplementedError) as error:
            raise _relabel(error, cause) from None
    pcr_data(model)


# -- advice -----------------------------------------------------------------
def _dist_of(ctx: Context, cause: Any) -> str:
    return _univariate()._label(ctx.model.models[cause])


def _pcr_distribution(row: dict, ctx: Context) -> str:
    cause = _cause_name(row["_cause"])
    return (
        "For cause {cause}, a {alt} fits better than the {dist} (its AIC "
        "is {d} lower). Fit that cause with it, giving each cause its own "
        "distribution: ParametricCompetingRisks.fit(x, e, dist={{{cause}: "
        "surpyval.{alt}, ...}}), with an entry for every cause. "
        "fit_best(x, c=np.where(e == {cause}, 0, 1), return_table=True) "
        "ranks every distribution for that cause.".format(
            cause=cause,
            alt=row["_dist"],
            dist=_dist_of(ctx, row["_cause"]),
            d=_fmt(row["statistic"]),
        )
    )


_OPTION = {
    "offset": "offset=True",
    "lfp": "lfp=True",
    "zi": "zi=True",
}


def _pcr_option(row: dict, ctx: Context) -> str:
    cause = _cause_name(row["_cause"])
    alone = f"c=np.where(e == {cause}, 0, 1)"
    if row["_kind"] == "mixture":
        call = "MixtureModel.fit(x, {}, dist=surpyval.{}, m=2)".format(
            alone, row["_dist"]
        )
    else:
        call = "surpyval.{}.fit(x, {}, {})".format(
            row["_dist"], alone, _OPTION[row["_kind"]]
        )
    return (
        "For cause {cause}, the {label} fits better than the {dist} (its "
        "AIC is {d} lower). ParametricCompetingRisks.fit takes a "
        "distribution per cause without options, so fit that cause on its "
        "own with the other causes censored, {call}, and assemble the "
        "causes with ParametricCompetingRisks.from_fitted({{{cause}: that "
        "fit, ...}}).".format(
            cause=cause,
            label=row["alternative"],
            dist=_dist_of(ctx, row["_cause"]),
            d=_fmt(row["statistic"]),
            call=call,
        )
    )


def _kind_is(kind: str) -> Callable[[dict, Context], bool]:
    """Whether an alternatives row is of the kind ``kind``."""
    return lambda row, ctx: row["_kind"] == kind


def _pcr_unexplained(row: dict, ctx: Context) -> bool:
    """A cause's failed goodness of fit that no alternative accounts
    for."""
    rows = _by_cause_rows("alternatives")(ctx)
    return not any(
        r.get("verdict") == "warn"
        for r in rows
        if r["_cause"] == row["_cause"]
    )


def _pcr_gof(row: dict, ctx: Context) -> str:
    return (
        "The {dist} does not fit cause {cause} (the {test} test of its "
        "Cox-Snell residuals fails) and none of the alternatives tried "
        "fits better by AIC. Look at d.plot('probability_plot') for where "
        "its points leave the line. To describe the incidences without a "
        "distribution, use the Aalen-Johansen estimate, "
        "CompetingRisks.fit(x, e); with covariates, the cause-specific Cox "
        "model, CompetingRisksProportionalHazards.fit(x, Z, e), leaves each "
        "cause's baseline hazard unspecified.".format(
            dist=_dist_of(ctx, row["_cause"]),
            cause=_cause_name(row["_cause"]),
            test=row["test"].split(" (")[0],
        )
    )


def _pcr_incidence(row: dict, ctx: Context) -> str:
    return (
        "The model's cumulative incidence of cause {cause} departs from "
        "the Aalen-Johansen estimate (by {d} at time {t}; see "
        "d.plot('incidence')). A cause's incidence depends on every "
        "cause's hazard, so first fix the causes whose goodness_of_fit "
        "rows fail; if none does, the causes' distributions are each "
        "plausible but together miss the incidence, and the Aalen-Johansen "
        "estimate, CompetingRisks.fit(x, e), describes it without "
        "them.".format(
            cause=_cause_name(row["_cause"]),
            d=_fmt(row["statistic"]),
            t=_fmt(row["_time"]),
        )
    )


def _pcr_notes(ctx: Context) -> list[str]:
    uni = _univariate()
    notes = [
        _INDEPENDENCE_NOTE,
        "Goodness of fit is the univariate distribution checks (see "
        "the univariate distributions' diagnostics) on each cause's own "
        "fit, with the other causes' failures censored: "
        "model.models[cause]. They test the cause's Cox-Snell residuals, "
        "H_k(x) - H_k(tl), its cumulative hazard over each unit's time at "
        "risk, against the unit exponential, through u = 1 - exp(-r), "
        "which is then uniform. With several causes there are more rows, "
        "and so more that fail by chance at alpha.",
    ]
    per_cause = []
    for cause, sub in _pcr_causes(ctx):
        per_cause.append(
            (cause, [n for n in uni._notes(sub) + sub.notes if n])
        )
    every = [n for n in per_cause[0][1]] if per_cause else []
    shared = [n for n in every if all(n in ns for _, ns in per_cause)]
    notes.extend(n for n in shared if n not in notes)
    for cause, ns in per_cause:
        notes.extend(
            f"Cause {_cause_name(cause)}: {n}" for n in ns if n not in shared
        )
    boot = _incidence(ctx)
    if boot is None:
        notes.append(
            "The incidence check is not computed: the data have right "
            "truncation (tr), which the Aalen-Johansen estimate it "
            "compares with does not take. The goodness-of-fit rows allow "
            "for truncation."
        )
    else:
        notes.append(
            "Incidence: the largest gap between each cause's parametric "
            "cumulative incidence and the Aalen-Johansen estimate "
            "(CompetingRisks.fit of the same data), with a p-value from {} "
            "parametric bootstrap refits: each resample draws every unit's "
            "latent time from each cause's model, keeps the earliest, "
            "censors it as the data were and refits every cause. "
            "diagnose(n_boot=..., random_state=...) sets their number and "
            "seed.".format(boot["n_boot"])
        )
        tl = _entry(_pcr_data(ctx))
        if tl is not None:
            origin = _origin(tl)
            notes.append(
                "With delayed entry (tl), the Aalen-Johansen estimate counts "
                "each unit at risk only from its entry, and each resample "
                "draws a unit's latent times beyond its entry and refits "
                "with the same entries."
                + (
                    ""
                    if origin is None
                    else " Every unit entered at {} or later, so the estimate "
                    "is conditional on surviving to {}, and the model's "
                    "incidence it is compared with is too: (CIF(t) - "
                    "CIF({})) / S({}), S the all-cause survival.".format(
                        *[_fmt(origin)] * 4
                    )
                )
            )
    return notes


def _is_parametric_cr(model: Any) -> bool:
    return isinstance(model, _pcr_class())


#: The parametric competing-risks family: each cause's distribution (the
#: univariate checks, by cause) and the incidences against Aalen-Johansen.
PARAMETRIC = register(
    Family(
        name="Competing risks (parametric)",
        issue=874,
        applies=_is_parametric_cr,
        validate=_validate_parametric,
        covariates=lambda model: [],
        columns=("cause", "alternative"),
        checks=(
            Check(
                "goodness_of_fit",
                "Each cause's Cox-Snell residuals, its cumulative hazard "
                "over each unit's time at risk with the other causes "
                "censored, are a censored sample of the unit exponential.",
                rows=_by_cause_rows("goodness_of_fit"),
                panels=_by_cause_panels("goodness_of_fit"),
            ),
            Check(
                "probability_plot",
                "Each cause's failures fall on the fitted line of its "
                "probability plot.",
                panels=_by_cause_panels("probability_plot"),
            ),
            Check(
                "alternatives",
                "No other model the advice can name fits a cause's data "
                "substantially better by AIC.",
                rows=_by_cause_rows("alternatives"),
            ),
            Check(
                "incidence",
                "Each cause's parametric cumulative incidence follows the "
                "Aalen-Johansen estimate of the same data.",
                rows=_incidence_rows,
                panels=_incidence_panels,
            ),
        ),
        advice=(
            AdviceRule(
                "alternatives",
                _pcr_distribution,
                ("ParametricCompetingRisks.fit(dist=)", "fit_best"),
                when=lambda row, ctx: row["_kind"] == "distribution",
                on=("warn",),
            ),
            AdviceRule(
                "alternatives",
                _pcr_option,
                (
                    "ParametricCompetingRisks.from_fitted",
                    "MixtureModel.fit(m=)",
                ),
                when=lambda row, ctx: row["_kind"] == "mixture",
                on=("warn",),
            ),
            *(
                AdviceRule(
                    "alternatives",
                    _pcr_option,
                    ("ParametricCompetingRisks.from_fitted",),
                    when=_kind_is(kind),
                    on=("warn",),
                )
                for kind in _OPTION
            ),
            AdviceRule(
                "goodness_of_fit",
                _pcr_gof,
                (
                    "CompetingRisks.fit",
                    "CompetingRisksProportionalHazards.fit",
                ),
                when=_pcr_unexplained,
            ),
            AdviceRule(
                "incidence",
                _pcr_incidence,
                ("CompetingRisks.fit",),
            ),
        ),
        notes=_pcr_notes,
    )
)
