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

``ParametricCompetingRisks`` is not built yet: its checks are the
univariate goodness-of-fit checks of #875 for each cause, which do not
exist yet; it refuses, naming #874, through the registry's pending list.

Every report notes that the independence of the causes cannot be tested
from competing-risks data (Tsiatis 1975), and how to choose between the
cause-specific and the Fine-Gray model.
"""

from __future__ import annotations

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
from .registry import AdviceRule, Check, Context, Family, Panel, register

__all__ = ["fine_gray_zph_test", "fine_gray_added_covariate_test"]

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
    var = _cox._source(name)
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


def _cs_failed(sub: Context, transform: str) -> list[int]:
    res = _cox._zph(sub, transform)["per_covariate"]
    return [j for j, _ in sub.checked if res[j]["p_value"] < sub.alpha]


def _cs_global_only(row: dict, ctx: Context) -> bool:
    if not row.get("_global"):
        return False
    sub = _context_of(ctx, row["_cause"])
    return not _cs_failed(sub, row["transform"])


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
            if _cs_failed(sub, transform):
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
            size=_cox._fmt(row["_size"]),
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
            cut=_cox._fmt(row["_cutoff"]),
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
    levels = np.unique(fd.Z_raw[:, j]).size
    if levels <= 2:
        return "binary"
    if levels <= _cox.CATEGORICAL_LEVELS:
        return "categorical"
    return "continuous"


def _fg_trend(res: dict, j: int, fd: FineGrayData) -> float:
    g, scaled = res["g"], res["scaled_schoenfeld"][:, j]
    w = fd.n[fd.events]
    return float(np.sign(np.cov(g, scaled, aweights=w)[0, 1]))


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
                        _label(cause), name, _cox._fmt(p)
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
