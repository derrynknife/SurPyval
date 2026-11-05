"""``diagnose()`` for Aalen's nonparametric additive hazards model
(``AalenAdditive``).

One family of :mod:`surpyval.diagnostics`, registered at the bottom. Aalen's
model, ``h(t | Z) = b_0(t) + b(t)' Z``, lets every effect change with time
and assumes nothing about the shape of the ``b_j``, so its report asks two
questions: does any effect in fact change (if none does, the simpler Lin
and Ying model fits as well), and does the fit give any observed unit a
negative hazard (which the additive form allows and a hazard cannot be)?

The checks
----------
``constant_effects``
    For each covariate, the test that its cumulative regression
    coefficient ``B_j(t)`` is a straight line through 0 (a time-constant
    effect): the model's own ``constant_effects_test()``, the
    Kolmogorov-Smirnov (supremum) distance of ``B_j`` from the line to
    its last value, as ``timereg::aalen``'s test for time-invariant
    effects, with its p-value from ``n_boot`` resampled processes (1000
    by default, as ``timereg``'s ``n.sim``) seeded by ``random_state``.
    The Cramér-von Mises p-value is in the row's detail and both are in
    ``d.tables["constant_effects"]``, with the baseline's row. This is a
    test, but not of an assumption: Aalen's model allows an effect that
    changes, so a small p-value is not a fault. Its rows' verdicts are
    therefore set, not taken from ``p``: ``"ok"`` whatever the p-value
    (``"warn"`` when the test could not be computed). The advice says
    what the result means: with no effect shown to change, the Lin-Ying
    model (``AdditiveHazards``, one coefficient per covariate, with its
    own ``diagnose()``) is the simpler fit; with one that changes, how to
    read it. Drawn as each ``B_j(t)`` with its pointwise ``1 - alpha``
    band and the straight line the test measures the distance from.
``non_negative_hazard``
    Whether the cumulative hazard the fit estimates for each observed
    covariate row, ``H(t | Z) = B_0(t) + B(t)' Z`` at the event times,
    ever falls (a negative hazard, which the additive form allows). Each
    row's largest fall below its running maximum (from 0 at time 0) is
    taken in units of the standard error of ``H`` at the last event time,
    and the statistic is the largest over the rows. Its null is the least
    favourable case, a hazard of 0 where the estimate falls: the same
    drawdowns of the resampled error processes of the constant-effects
    test (``n_boot`` of them), which the estimate's falls cannot exceed
    when every true hazard is at or above 0. The test is therefore
    conservative, and its verdict is the shared rule's. The detail gives
    the share of units (rows, weighted by their counts) whose own fall
    has a p-value below ``alpha``, the share whose estimate falls at all
    and the largest fall. A fall does not reach the predictions --
    :meth:`Hf` holds the estimate at its running maximum -- but a
    significant one means the additive form gives those units a negative
    hazard. Drawn as the estimate and the held prediction of the rows
    whose estimates fall the furthest.

Not checked: the functional form of the covariates, influential rows and
outliers. Weighted fits are refused (the resampled nulls are those of an
unweighted fit).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._shared import _fmt
from .registry import AdviceRule, Check, Context, Family, Panel, register

#: The resampled processes of the constant-effects test when
#: ``diagnose()`` is not given ``n_boot``: ``timereg``'s ``n.sim``.
DEFAULT_N_BOOT = 1000

#: The rows the falls panel draws: those whose estimates fall furthest.
FALLS_DRAWN = 3


# ---------------------------------------------------------------------------
# The constant-effects test
# ---------------------------------------------------------------------------
def _n_sim(ctx: Context) -> int:
    n_boot = DEFAULT_N_BOOT if ctx.n_boot is None else ctx.n_boot
    if (
        isinstance(n_boot, bool)
        or not isinstance(n_boot, (int, np.integer))
        or n_boot < 1
    ):
        raise ValueError(
            "'n_boot' is the number of resampled processes of the "
            "constant-effects test: a whole number, at least 1 (1000 by "
            "default); got {!r}.".format(n_boot)
        )
    return int(n_boot)


def _test(ctx: Context) -> Any:
    """The model's constant-effects test (computed once per report): a
    table with one row per term, the baseline first."""
    return ctx.cached(
        "constant_effects",
        lambda: ctx.model.constant_effects_test(
            n_sim=_n_sim(ctx), random_state=ctx.random_state
        ),
    )


def _constant_rows(ctx: Context) -> list[dict]:
    table = _test(ctx)
    rows = []
    for j, name in ctx.checked:
        ks, ks_p, cvm_p = (
            float(table.iloc[j + 1][k]) for k in ("KS", "KS p", "CvM p")
        )
        if not np.isfinite(ks_p):
            detail = (
                "the test could not be computed: the covariate is aliased "
                "(its cumulative coefficient is not estimated)"
            )
            verdict = "warn"
        elif ks_p < ctx.alpha:
            detail = (
                "its effect changes with time (Cramér-von Mises p = {}); "
                "Aalen's model allows that".format(_fmt(cvm_p))
            )
            verdict = "ok"
        else:
            detail = (
                "no evidence that its effect changes with time "
                "(Cramér-von Mises p = {})".format(_fmt(cvm_p))
            )
            verdict = "ok"
        rows.append(
            {
                "test": "Kolmogorov-Smirnov test of a time-constant effect "
                "(resampled)",
                "covariates": (name,),
                "statistic": ks,
                "p": ks_p,
                "verdict": verdict,
                "detail": detail,
            }
        )
    return rows


def _constant_table(ctx: Context) -> Any:
    return _test(ctx).copy()


def _constant_panels(ctx: Context) -> list[Panel]:
    model = ctx.model
    table = _test(ctx)
    t = np.concatenate([[0.0], model.x])
    panels = []
    for j, name in ctx.checked:
        p = float(table.iloc[j + 1]["KS p"])

        def draw(ax: Any, options: Any, j: int = j) -> None:
            from surpyval.plotting import BAND_ALPHA, INK, PALETTE

            B = np.concatenate([[0.0], model.B[:, j + 1]])
            band = model.cumulative_coefficients_cb(
                t, alpha_ci=options.alpha, terms=[j + 1]
            )[:, 0]
            ax.step(t, B, where="post", color=PALETTE[0], label="B(t)")
            ax.fill_between(
                t,
                band[:, 0],
                band[:, 1],
                step="post",
                color=PALETTE[0],
                alpha=BAND_ALPHA,
                linewidth=0,
                label=f"pointwise {100 * (1 - options.alpha):g}%",
            )
            ax.plot(
                [0.0, t[-1]],
                [0.0, B[-1]],
                color=PALETTE[1],
                ls="--",
                label="constant effect",
            )
            ax.axhline(0.0, color=INK["axis"], lw=0.8)
            ax.set_xlabel("time")
            ax.set_ylabel("cumulative coefficient")
            ax.legend(fontsize="small")

        panels.append(Panel(f"{name}: cumulative effect, p = {_fmt(p)}", draw))
    return panels


# ---------------------------------------------------------------------------
# The test for falling cumulative hazards
# ---------------------------------------------------------------------------
#: The most numbers (draws x event times x covariate rows) the drawdown of
#: the resampled processes is computed on at once.
_CHUNK = 4_000_000


def _drawdown(H: np.ndarray) -> np.ndarray:
    """The largest fall of each path ``H[..., :, r]`` (along the
    event-time axis, ``-2``) below its running maximum, the running
    maximum starting at 0 at time 0."""
    peak = np.maximum.accumulate(np.maximum(H, 0.0), axis=-2)
    return (peak - H).max(axis=-2)


def falls(
    model: Any,
    n_sim: int = 1000,
    random_state: Any = None,
    alpha: float = 0.05,
) -> dict:
    """
    Whether the cumulative hazard an Aalen additive model estimates for
    its observed covariate rows falls (a negative hazard): the test behind
    ``diagnose()``'s ``non_negative_hazard`` check.

    For each distinct covariate row ``Z`` of the fitted data, the estimate
    ``H(t | Z) = B_0(t) + B(t)' Z`` at the event times is compared with
    its running maximum (from 0 at time 0), and its largest fall ``D_Z``
    is taken in units of its standard error at the last event time. The
    statistic is the largest ``D_Z`` over the rows. Its null is that of
    the least favourable case, a hazard of 0: the same drawdowns of
    ``n_sim`` resampled error processes (the processes of
    ``AalenAdditiveModel.constant_effects_test()``),
    which a fall of the estimate cannot exceed when every true hazard is
    at or above 0, so the test is conservative there. A row's own p-value
    compares its fall with the draws' falls for that row.

    Parameters
    ----------
    model : AalenAdditiveModel
        A fitted (unweighted) model that keeps its data.
    n_sim : int, optional
        The number of resampled processes. Default 1000.
    random_state : None, int or numpy.random.Generator, optional
        The seed of the draws.
    alpha : float, optional
        The level a row's own p-value is judged at for ``unit_share``.
        Default 0.05.

    Returns
    -------
    dict
        ``statistic`` and ``p`` (the test over every row), ``unit_share``
        (the share of units, by count, whose own fall has a p-value below
        ``alpha``), ``any_share`` (the share whose estimate falls at all),
        ``largest`` (the largest fall, in units of cumulative hazard),
        ``rows`` (the distinct covariate rows), ``row_p`` (each one's
        p-value), ``H`` (the estimates, one column per row) and
        ``H_star`` (their running maxima).

    Examples
    --------
    >>> from surpyval import AalenAdditive
    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval.diagnostics.aalen_additive import falls
    >>> df = load_rossi_static()
    >>> df["censored"] = 1 - df["arrest"]
    >>> model = AalenAdditive.fit_from_df(
    ...     df, x_col="week", c_col="censored", Z_cols=["fin", "age", "prio"]
    ... )
    >>> out = falls(model, random_state=1)
    >>> round(out["any_share"], 3), round(out["p"], 3)
    (1.0, 0.198)

    Every man's estimate dips somewhere (the least-squares increments are
    noisy), but by no more than chance explains: the largest fall, 2.4
    standard errors, is smaller than a fifth of the resampled ones.
    """
    from surpyval.utils.rng import as_generator

    x, c, n, tl, Z = model._fit_data
    Z = np.asarray(Z, dtype=float).reshape(np.size(x), -1)
    uniq, which = np.unique(Z, axis=0, return_inverse=True)
    which = np.asarray(which).ravel()
    weight = np.bincount(which, weights=n, minlength=uniq.shape[0])
    total = float(weight.sum())
    X = model._rows(uniq)
    B = model._coefficients()
    k = B.shape[0]
    H = B @ X.T  # (event times, rows)
    H_star = np.maximum.accumulate(np.maximum(H, 0.0), axis=0)
    out = {
        "rows": uniq,
        "H": H,
        "H_star": H_star,
        "largest": float((H_star - H).max()) if k else 0.0,
    }
    scale = np.maximum(np.abs(H_star).max(axis=0, initial=0.0), 1.0)
    fell = (H_star - H > 1e-10 * scale).any(axis=0)
    out["any_share"] = float(weight @ fell) / total
    C = np.where(np.isnan(model._B_cov), 0.0, model._B_cov)
    se = np.sqrt(np.einsum("rq,qp,rp->r", X, C[-1], X)) if k else np.zeros(0)
    if k == 0 or not np.all(se > 0):
        why = (
            "no event time at which the fit is estimated"
            if k == 0
            else "a row's cumulative hazard has no variance"
        )
        out.update(
            statistic=float("nan"),
            p=float("nan"),
            unit_share=float("nan"),
            row_p=np.full(uniq.shape[0], np.nan),
            why=why,
        )
        return out
    observed = _drawdown(H) / se
    W = model._resampled(int(n_sim), as_generator(random_state))
    W = np.where(np.isnan(W), 0.0, W)  # (draws, times, terms)
    step = max(1, _CHUNK // max(1, n_sim * k))
    largest = np.zeros(n_sim)
    exceed = np.zeros(uniq.shape[0])
    for lo in range(0, uniq.shape[0], step):
        part = slice(lo, lo + step)
        D = _drawdown(W @ X[part].T) / se[part]  # (draws, rows)
        largest = np.maximum(largest, D.max(axis=1))
        exceed[part] = (D >= observed[part] - 1e-12).mean(axis=0)
    statistic = float(observed.max())
    out.update(
        statistic=statistic,
        p=float(np.mean(largest >= statistic - 1e-12)),
        unit_share=float(weight @ (fell & (exceed < alpha))) / total,
        row_p=exceed,
    )
    return out


def _falls(ctx: Context) -> dict:
    return ctx.cached(
        "falls",
        lambda: falls(ctx.model, _n_sim(ctx), ctx.random_state, ctx.alpha),
    )


def _falls_rows(ctx: Context) -> list[dict]:
    res = _falls(ctx)
    if "why" in res:
        detail = "the test could not be computed: " + res["why"]
    else:
        detail = (
            "{} of units fall by more than chance explains on their own "
            "({} fall at all; the largest fall is {} of cumulative "
            "hazard)".format(
                _pct(res["unit_share"]),
                _pct(res["any_share"]),
                _fmt(res["largest"]),
            )
        )
    return [
        {
            "test": "resampled drawdown test of a non-falling cumulative "
            "hazard",
            "covariates": tuple(ctx.names),
            "statistic": res["statistic"],
            "p": res["p"],
            "detail": detail,
        }
    ]


def _pct(share: float) -> str:
    return f"{100 * share:.3g}%"


def _falls_panels(ctx: Context) -> list[Panel]:
    res = _falls(ctx)
    model = ctx.model
    H, H_star = res["H"], res["H_star"]
    if H.shape[0] == 0:
        return []
    order = np.argsort(-(H_star - H).max(axis=0))[:FALLS_DRAWN]
    t = np.concatenate([[0.0], model.x])

    def label(r: int) -> str:
        values = ", ".join(
            f"{name}={_fmt(v)}" for name, v in zip(ctx.names, res["rows"][r])
        )
        return values if len(values) <= 40 else values[:37] + "..."

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        for k, r in enumerate(order):
            colour = PALETTE[k % len(PALETTE)]
            ax.step(
                t,
                np.concatenate([[0.0], H[:, r]]),
                where="post",
                color=colour,
                label=label(r),
            )
            ax.step(
                t,
                np.concatenate([[0.0], H_star[:, r]]),
                where="post",
                color=colour,
                ls=":",
            )
        ax.axhline(0.0, color=INK["axis"], lw=0.8)
        ax.set_xlabel("time")
        ax.set_ylabel("cumulative hazard (dotted: as predicted)")
        ax.legend(fontsize="small")

    return [
        Panel(
            "Falling cumulative hazards, p = {}".format(_fmt(res["p"])),
            draw,
        )
    ]


# ---------------------------------------------------------------------------
# Refusals and notes
# ---------------------------------------------------------------------------
def _validate(model: Any) -> None:
    if getattr(model, "weighted", False):
        from surpyval.utils.weights import refuse_weighted

        raise refuse_weighted(
            "diagnose() of an Aalen additive model",
            "its tests simulate their nulls from the events of an "
            "unweighted fit, and a version for sampling weights has not "
            "been built or checked",
            "Use the robust bands of model.plot() or "
            "model.cumulative_coefficients_cb() to judge whether each "
            "B_j(t) is a straight line, or diagnose() on the fit without w "
            "as a guide only.",
        )
    if getattr(model, "_fit_data", None) is None:
        raise ValueError(
            "diagnose() needs the data the model was fitted to, which a "
            "model restored with from_dict / from_json does not carry. "
            "Call it on the fitted model, or refit with AalenAdditive.fit."
        )


def _notes(ctx: Context) -> list[str]:
    notes = [
        "Constant effects: for each covariate the Kolmogorov-Smirnov "
        "distance of its cumulative coefficient B_j(t) from the straight "
        "line to its last value, with the p-value from {} resampled "
        "processes (model.constant_effects_test(); the full table, with "
        "the Cramér-von Mises statistics and the baseline, is "
        "d.tables['constant_effects']). Aalen's model allows effects that "
        "change, so these rows inform the choice of model and never "
        "fail.".format(_n_sim(ctx)),
        "The resampled processes multiply each event's contribution by a "
        "normal draw rather than each unit's martingale residuals, as "
        "timereg does: timereg's draws rejected constant effects far "
        "more often than the level with a few hundred units. With about "
        "100 units these p-values are conservative.",
        "Falling cumulative hazards: the largest fall of any observed "
        "covariate row's estimated cumulative hazard below its running "
        "maximum, in standard errors, against the falls of the same "
        "resampled processes (the null of a zero hazard, so the test is "
        "conservative). Predictions hold the estimate at its running "
        "maximum (see model.Hf).",
        "Not checked yet for this model: the functional form of the "
        "covariates, influential rows and outliers.",
    ]
    if ctx.model.penalizer:
        notes.append(
            "The fit is penalised (ridge); the tests take "
            "the penalised increments and their variance as fitted."
        )
    return notes


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------
def _all_constant(row: Any, ctx: Context) -> bool:
    """No checked covariate's effect is shown to change (every p at or
    above alpha, and every test computed), and no fitted hazard is shown
    to be negative (another additive model would not mend that)."""
    if _falls(ctx)["p"] < ctx.alpha:
        return False
    table = _test(ctx)
    p = np.array([float(table.iloc[j + 1]["KS p"]) for j, _ in ctx.checked])
    return bool(p.size) and bool(np.all(np.isfinite(p) & (p >= ctx.alpha)))


def _constant_message(row: Any, ctx: Context) -> str:
    return (
        "No covariate's effect is shown to change with time (every "
        "constant-effects p-value is at or above {}). The Lin-Ying model, "
        "AdditiveHazards.fit, with one coefficient per covariate, is the "
        "simpler fit of the same additive form, and its diagnose() tests "
        "its assumptions.".format(_fmt(ctx.alpha))
    )


def _changes(row: dict, ctx: Context) -> bool:
    return bool(np.isfinite(row["p"]) and row["p"] < ctx.alpha)


def _changing_message(row: dict, ctx: Context) -> str:
    name = row["covariates"][0]
    return (
        "The effect of {name} on the hazard changes with time (its "
        "cumulative coefficient in d.plot('constant_effects') bends), "
        "which Aalen's model allows: read when and how from the slope of "
        "B(t), model.smoothed_coefficients(), or model.plot(smoothed="
        "True). A model with a constant {name} effect (AdditiveHazards, "
        "CoxPH) would misstate it.".format(name=name)
    )


def _falls_message(row: dict, ctx: Context) -> str:
    return (
        "The estimated cumulative hazard of some units falls by more than "
        "chance explains ({} of them on their own): the additive model "
        "gives them a negative hazard over some of follow-up. Their "
        "predictions are held at the "
        "estimate's running maximum (model.Hf), which hides the fall; see "
        "d.plot('non_negative_hazard'). A multiplicative model keeps every "
        "hazard positive: fit CoxPH.fit, and check its own diagnose(). An "
        "additive effect over a wide covariate range is the usual cause, "
        "so a transformed covariate (a log, or groups) may also "
        "help.".format(_pct(_falls(ctx)["unit_share"]))
    )


ADVICE = (
    AdviceRule(
        None,
        _constant_message,
        ("AdditiveHazards.fit",),
        when=_all_constant,
    ),
    AdviceRule(
        "constant_effects",
        _changing_message,
        ("smoothed_coefficients", "plot(smoothed=)"),
        when=_changes,
        on=("ok",),
    ),
    AdviceRule(
        "non_negative_hazard",
        _falls_message,
        ("CoxPH.fit", "Hf"),
    ),
)


# ---------------------------------------------------------------------------
# The family
# ---------------------------------------------------------------------------
def _is_aalen(model: Any) -> bool:
    from surpyval.univariate.regression.additive_hazards.aalen_additive import (  # noqa: E501
        AalenAdditiveModel,
    )

    return isinstance(model, AalenAdditiveModel)


def _names(model: Any) -> list[str]:
    """The covariates' names, as ``summary()`` names them (less the
    baseline)."""
    return list(model.parameter_names)


#: The Aalen additive hazards family: ``AalenAdditive``.
AALEN_ADDITIVE_FAMILY = register(
    Family(
        name="Aalen additive hazards",
        issue=867,
        applies=_is_aalen,
        validate=_validate,
        covariates=_names,
        checks=(
            Check(
                "constant_effects",
                "Whether each covariate's effect on the hazard is constant "
                "in time (Aalen's model allows it to change; a constant "
                "one is the simpler Lin-Ying model).",
                rows=_constant_rows,
                panels=_constant_panels,
                table=_constant_table,
            ),
            Check(
                "non_negative_hazard",
                "Every observed unit's hazard b_0(t) + b(t)'Z stays at or "
                "above 0, so its estimated cumulative hazard never falls.",
                rows=_falls_rows,
                panels=_falls_panels,
            ),
        ),
        advice=ADVICE,
        notes=_notes,
    )
)
