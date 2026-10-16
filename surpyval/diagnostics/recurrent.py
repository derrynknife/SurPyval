"""``diagnose()`` for the recurrent-event models (#873).

Three families share this module, because the three kinds of fitted
recurrent model make different assumptions and so need different checks
(and the report is headed by its family):

- **Poisson process** (:data:`POISSON`): a ``ParametricRecurrenceModel``,
  from ``HPP``, ``CrowAMSAA``, ``Duane`` or ``CoxLewis`` -- minimal repair,
  every item with the one intensity;
- **imperfect repair** (:data:`IMPERFECT_REPAIR`): a ``RenewalModel``, from
  ``GeneralizedRenewal``, ``GeneralizedOneRenewal``, ``ARA`` or ``ARI``,
  whose restoration parameter places the repairs between as good as new
  and as bad as old;
- **proportional intensity** (:data:`PROPORTIONAL_INTENSITY`): a
  ``ProportionalIntensityModel``, from ``ProportionalIntensityHPP`` or
  ``ProportionalIntensityNHPP``, a Poisson process whose intensity each
  item's covariates scale.

They share the trend and fit checks; the repair check is the imperfect-
repair models' and the between-items check the Poisson processes'.

The checks
----------
``trend``
    Three trend tests of the event times and observation windows (not of
    the fitted model): the Laplace and the MIL-HDBK-189C tests, whose null
    is a homogeneous Poisson process (a constant rate), and the
    Lewis-Robinson test, whose null is a renewal process (independent,
    identically distributed times between failures, of any distribution):
    the Laplace statistic divided by the coefficient of variation of the
    times between failures (:func:`lewis_robinson`, which takes each item
    from its first renewal to its last failure). The first two are
    :func:`surpyval.recurrent.laplace` and
    :func:`surpyval.recurrent.mil_hdbk_189c`, run by the model's
    ``trend_test``; each item is tested on its own window. For a model
    that assumes a constant rate (``HPP``, ``ProportionalIntensityHPP``)
    they are tests of that assumption, judged by the shared rule. For the
    others a trend is no fault, and the rows are a *model-choice screen*:
    a row warns when its test does not reject its null (the data are
    consistent with the simpler process the null names, so it may do),
    and is ok when it rejects. It never fails.
``fit``
    The Cramer-von Mises test of the fitted model, the model's own
    ``cramer_von_mises`` (conditionally uniform transforms of the event
    times; a parametric bootstrap of refits for the p-value), with
    :data:`N_BOOT` resamples from the seed :data:`SEED`, so that a report
    is reproducible. Drawn as the P-P plot of the transforms.
``between_items``
    (Poisson processes.) The Pearson dispersion test of the items' event
    counts: under the model each time-truncated item's count is Poisson
    with mean its expected count over its window, ``E_q = Lambda_q(close)
    - Lambda_q(entry)``, so ``sum (N_q - E_q)^2 / E_q`` is about
    chi-squared with ``m - 1 - p`` degrees of freedom (``m`` items, ``p``
    covariate coefficients; the one more is the scale of the intensity,
    which makes the residuals sum to zero). Items differing by more than
    chance fail it. A failure-truncated item (observed up to its last
    event) has a fixed count and is left out.
``repair``
    (Imperfect repair.) The model's ``repair_test``: the likelihood-ratio
    tests of perfect repair (as good as new: Kijima and G1 ``q = 0``, ARA
    ``rho = 1``; for ARI its maximal repair ``rho = 1``) and of minimal
    repair (as bad as old: Kijima ``q = 1``, ARA and ARI ``rho = 0``; G1
    has none). A value on the edge of the parameter's range is tested
    with the 50:50 chi-squared(0) / chi-squared(1) mixture (Self and Liang,
    1987). Neither kind of repair is an assumption of the model, so the
    rows are a model-choice screen as above: a row warns when its repair
    is not rejected (the simpler model fits as well), and is ok when it
    is. Drawn as the profile log-likelihood of the restoration parameter.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.stats import chi2, norm

from .registry import AdviceRule, Check, Context, Family, Panel, register

#: The bootstrap resamples behind the Cramer-von Mises p-value: 99, so
#: that the p-value is a multiple of 0.01. ``model.cramer_von_mises(n_boot=
#: ...)`` gives a finer one.
N_BOOT = 99
#: The seed of those resamples: a report is the same each time it is run.
SEED = 0

_LAPLACE = "Laplace"
_MIL = "MIL-HDBK-189C"
_LR = "Lewis-Robinson"

#: The intensity fitters, by the name their fitted model's ``dist`` has.
_INTENSITY_FITTERS = {
    "Homogeneous Poisson Process": "HPP",
    "Crow-AMSAA": "CrowAMSAA",
    "Duane": "Duane",
    "Cox-Lewis": "CoxLewis",
}

#: The fitter of each imperfect-repair kind.
_RENEWAL_FITTERS = {
    "Generalized Renewal": "GeneralizedRenewal",
    "G1 Renewal": "GeneralizedOneRenewal",
    "ARA Renewal": "ARA",
    "ARI Recurrence": "ARI",
}


# ---------------------------------------------------------------------------
# What the checks share
# ---------------------------------------------------------------------------
def _fmt(value: float) -> str:
    return "{:.3g}".format(value)


def _is_poisson(model: Any) -> bool:
    from surpyval.recurrent.parametric.parametric_recurrence import (
        ParametricRecurrenceModel,
    )

    return isinstance(model, ParametricRecurrenceModel)


def _is_renewal(model: Any) -> bool:
    from surpyval.recurrent import RenewalModel

    return isinstance(model, RenewalModel)


def _is_pi(model: Any) -> bool:
    from surpyval.recurrent.regression.proportional_intensity import (
        ProportionalIntensityModel,
    )

    return isinstance(model, ProportionalIntensityModel)


def _constant_rate(model: Any) -> bool:
    """Whether the model assumes a constant rate of failures: an HPP, or a
    proportional-intensity model with an HPP baseline."""
    if _is_pi(model):
        return getattr(model, "kind", None) == "HPP"
    if _is_poisson(model):
        name = getattr(getattr(model, "dist", None), "name", None)
        return name == "Homogeneous Poisson Process"
    return False


def _process(model: Any) -> str:
    """The fitted process, as the notes and advice name it, with the
    fitter that made it."""
    if _is_renewal(model):
        fitter = _RENEWAL_FITTERS.get(model.kind, model.kind)
        return "{} ({}.fit, {} {})".format(
            model.kind,
            fitter,
            model._dist_label.lower(),
            model.model.dist.name,
        )
    if _is_pi(model):
        fitter = (
            "ProportionalIntensityHPP"
            if model.kind == "HPP"
            else "ProportionalIntensityNHPP"
        )
        if model.kind == "HPP":
            return "proportional intensity, constant baseline ({}.fit)".format(
                fitter
            )
        return "proportional intensity, {} baseline ({}.fit)".format(
            model.dist.name, fitter
        )
    name = model.dist.name
    return "{} ({}.fit)".format(name, _INTENSITY_FITTERS.get(name, name))


def _windows(ctx: Context) -> list:
    """Each item's ``(item, events, entry, close, explicit_close)``."""
    from surpyval.recurrent.diagnostics import item_windows

    return ctx.cached("windows", lambda: item_windows(ctx.model.data))


def _item_cif(ctx: Context) -> Any:
    """The fitted cumulative intensity: one callable, or one per item for
    a proportional-intensity model."""
    if _is_pi(ctx.model):
        return ctx.cached("item_cif", ctx.model._item_cif_map)
    return ctx.model.cif


def _quiet(compute: Any) -> Any:
    """``compute()`` with the warnings of its refits and searches kept
    out of the user's console: the report says what they found."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return compute()


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------
def lewis_robinson(windows: list) -> dict:
    """
    The Lewis-Robinson trend test, whose null is a renewal process: the
    Laplace statistic divided by the coefficient of variation (sample
    standard deviation over mean) of the times between failures, referred
    to the standard normal (Lewis and Robinson, 1974; pooled over the
    items as in Kvaloy and Lindqvist, 1998). For an HPP the coefficient
    of variation is 1 and it is the Laplace test.

    ``windows`` is each item's ``(item, events, entry, close,
    explicit_close)``, as :func:`surpyval.recurrent.diagnostics.
    item_windows` gives them. Each item is taken from its first renewal
    -- its start if it is observed from 0 (new), else its first event --
    to its last event, which closes the window and is not itself counted
    (the failure-truncated form of :func:`surpyval.recurrent.laplace`).
    Under a renewal process the times between failures within that
    window are exchangeable, so the statistic is centred exactly; the
    time-truncated form is not, as the gap the window close censors is
    left out (for regular times between failures, a coefficient of
    variation of 0.36, ten items of twenty failures each, it rejected 30%
    of the time at the 5% level). Those times between failures are the
    ones the coefficient of variation is estimated from.

    Returns ``{"statistic", "p_value", "cv", "direction", "n_gaps"}``;
    raises ``ValueError`` when it cannot be computed.

    Examples
    --------
    Six systems (Mettas and Zhao, 2005), whose times between failures
    vary a little less than exponential ones:

    >>> from surpyval.datasets import load_mettas_and_zhao
    >>> from surpyval.diagnostics.recurrent import lewis_robinson
    >>> from surpyval.recurrent import HPP
    >>> from surpyval.recurrent.diagnostics import item_windows
    >>> df = load_mettas_and_zhao()
    >>> model = HPP.fit(df["x"], df["i"], df["c"])
    >>> out = lewis_robinson(item_windows(model.data))
    >>> round(out["statistic"], 4), round(out["p_value"], 4)
    (1.9304, 0.0536)
    >>> round(out["cv"], 4), out["n_gaps"]
    (0.8957, 27)
    """
    from surpyval.recurrent.tests import laplace

    x: list = []
    i: list = []
    starts: dict = {}
    ends: dict = {}
    gaps: list = []
    for k, (_, events, entry, _, _) in enumerate(windows):
        if entry == 0:
            start, renewals = 0.0, events
        elif events.size:
            start, renewals = float(events[0]), events[1:]
        else:
            continue
        if renewals.size == 0:
            continue
        gaps.append(np.diff(np.concatenate([[start], renewals])))
        end = float(renewals[-1])
        inner = renewals[:-1]
        inner = inner[inner > start]
        if end <= start or inner.size == 0:
            continue
        x.extend(inner)
        i.extend([k] * inner.size)
        starts[k], ends[k] = start, end
    pooled = np.concatenate(gaps) if gaps else np.array([])
    if pooled.size < 2:
        raise ValueError(
            "fewer than two times between failures to estimate their " "spread"
        )
    mean, sd = float(pooled.mean()), float(pooled.std(ddof=1))
    if not (mean > 0 and sd > 0):
        raise ValueError("the times between failures do not vary")
    if len(x) < 2:
        raise ValueError(
            "fewer than two failures fall inside the items' windows from "
            "their first renewal to their last failure"
        )
    res = laplace(np.asarray(x), i=np.asarray(i), T=ends, tl=starts)
    cv = sd / mean
    z = float(res.statistic) / cv
    return {
        "statistic": z,
        "p_value": float(2.0 * norm.sf(abs(z))),
        "cv": cv,
        "direction": res.direction,
        "n_gaps": int(pooled.size),
    }


def _trend_results(ctx: Context) -> dict:
    """``{test: result or error message}`` for the three trend tests."""

    def compute() -> dict:
        out: dict = {}
        for name, key in ((_LAPLACE, "laplace"), (_MIL, "mil_hdbk_189c")):
            try:
                res = ctx.model.trend_test(key, alpha_ci=ctx.alpha)
            except ValueError as error:
                out[name] = str(error)
                continue
            out[name] = {
                "statistic": res.statistic,
                "p_value": res.p_value,
                "df": res.dof,
                "direction": res.direction,
            }
        try:
            out[_LR] = lewis_robinson(_windows(ctx))
        except ValueError as error:
            out[_LR] = str(error)
        return out

    return ctx.cached("trend", compute)


def _rejects(ctx: Context, test: str) -> "bool | None":
    """Whether ``test`` rejects its null at ``ctx.alpha``; ``None`` when it
    could not be computed."""
    res = _trend_results(ctx)[test]
    if isinstance(res, str) or not math.isfinite(res["p_value"]):
        return None
    return bool(res["p_value"] < ctx.alpha)


def _trend_rows(ctx: Context) -> list[dict]:
    constant = _constant_rate(ctx.model)
    rows = []
    nulls = {
        _LAPLACE: "a homogeneous Poisson process",
        _MIL: "a homogeneous Poisson process",
        _LR: "a renewal process",
    }
    for test, res in _trend_results(ctx).items():
        row: dict = {"test": "{} trend test".format(test), "_test": test}
        if isinstance(res, str):
            row.update(
                {
                    "p": float("nan"),
                    "detail": "could not be computed: " + res,
                }
            )
            if not constant:
                row["verdict"] = "warn"
            rows.append(row)
            continue
        p = res["p_value"]
        direction = res["direction"]
        row.update(
            {
                "statistic": res["statistic"],
                "df": res.get("df"),
                "p": p,
                "_direction": direction,
            }
        )
        rate = "the rate of failures {}s with time".format(
            "increase" if direction == "increasing" else "decrease"
        )
        if test == _LR:
            row["detail"] = (
                "no evidence against a renewal process (times between "
                "failures with CV {})".format(_fmt(res["cv"]))
                if p >= ctx.alpha
                else "not a renewal process: " + rate
            )
        else:
            row["detail"] = (
                "no evidence of a trend" if p >= ctx.alpha else rate
            )
        if not constant:
            row["verdict"] = "warn" if p >= ctx.alpha else "ok"
            if p >= ctx.alpha:
                row["detail"] += "; consistent with {}".format(nulls[test])
        rows.append(row)
    return rows


def _mcf_steps(ctx: Context) -> tuple[np.ndarray, np.ndarray, float]:
    """The non-parametric mean cumulative function of the data (events
    over items at risk, each item at risk over its window) and the HPP
    rate, events over total exposure."""
    windows = _windows(ctx)
    events = np.concatenate([w[1] for w in windows])
    entries = np.array([w[2] for w in windows])
    closes = np.array([w[3] for w in windows])
    times = np.unique(events)
    at_risk = np.array(
        [((entries < t) & (closes >= t)).sum() for t in times], dtype=float
    )
    counts = np.array([(events == t).sum() for t in times], dtype=float)
    mcf = np.cumsum(counts / np.maximum(at_risk, 1.0))
    rate = events.size / float((closes - entries).sum())
    return times, mcf, rate


def _trend_panels(ctx: Context) -> list[Panel]:
    if not _windows(ctx):
        return []
    times, mcf, rate = _mcf_steps(ctx)
    res = _trend_results(ctx)[_LAPLACE]
    p = "" if isinstance(res, str) else ": Laplace p = " + _fmt(res["p_value"])

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        ax.step(
            np.r_[0.0, times],
            np.r_[0.0, mcf],
            where="post",
            color=PALETTE[0],
            label="data (mean cumulative function)",
        )
        top = float(times.max()) if times.size else 1.0
        ax.plot(
            [0.0, top],
            [0.0, rate * top],
            color=INK["secondary"],
            ls="--",
            lw=0.8,
            label="constant rate",
        )
        ax.set_xlabel("time")
        ax.set_ylabel("events per item")
        ax.legend(loc="upper left", fontsize="small", frameon=False)

    return [Panel("trend" + p, draw)]


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def _cvm(ctx: Context) -> Any:
    """The model's Cramer-von Mises test with :data:`N_BOOT` resamples
    from :data:`SEED`, or the reason it could not be run; and the number
    of bootstrap refits that failed (and were resampled)."""

    def compute() -> tuple:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with np.errstate(all="ignore"):
                try:
                    res = ctx.model.cramer_von_mises(
                        n_boot=N_BOOT, random_state=SEED
                    )
                except (ValueError, RuntimeError) as error:
                    return str(error), caught
        return res, caught

    return ctx.cached("cvm", compute)


def _fit_rows(ctx: Context) -> list[dict]:
    res, caught = _cvm(ctx)
    covariates = tuple(ctx.names) if _is_pi(ctx.model) else ()
    test = "Cramer-von Mises, parametric bootstrap"
    if isinstance(res, str):
        return [
            {
                "test": test,
                "covariates": covariates,
                "p": float("nan"),
                "detail": "could not be computed: " + res,
            }
        ]
    failed = [w for w in caught if "bootstrap replicate" in str(w.message)]
    if failed:
        ctx.notes.append(
            "Fit: " + str(failed[0].message).rstrip(".") + " (of the "
            "Cramer-von Mises bootstrap refits)."
        )
    p = res.p_value
    if p < ctx.alpha:
        detail = "the event times depart from the fitted intensity"
    elif p < 2 * ctx.alpha:
        detail = "borderline departure from the fitted intensity"
    else:
        detail = "no evidence against the fitted intensity"
    return [
        {
            "test": test,
            "covariates": covariates,
            "statistic": res.statistic,
            "p": p,
            "detail": "{} ({} resamples)".format(detail, res.n_boot),
        }
    ]


def _uniforms(ctx: Context) -> np.ndarray:
    """The conditionally uniform transforms of the event times the
    Cramer-von Mises statistic is computed from."""
    from surpyval.recurrent import diagnostics

    model = ctx.model
    if _is_renewal(model):
        increments = model._fitter._rescaled_increments(model, model.data)
        u, _ = diagnostics._renewal_conditional_uniforms(
            model.data, increments
        )
    else:
        u, _ = diagnostics._conditional_uniforms(model.data, _item_cif(ctx))
    return np.sort(np.asarray(u, dtype=float))


def _fit_panels(ctx: Context) -> list[Panel]:
    try:
        u = _quiet(lambda: _uniforms(ctx))
    except ValueError:
        return []
    res, _ = _cvm(ctx)
    p = "" if isinstance(res, str) else ": p = " + _fmt(res.p_value)

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        m = u.size
        expected = (np.arange(1, m + 1) - 0.5) / m
        ax.plot([0, 1], [0, 1], color=INK["secondary"], ls="--", lw=0.8)
        ax.scatter(expected, u, s=10, color=PALETTE[0], alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("uniform quantile")
        ax.set_ylabel("transformed event time")

    return [Panel("fit (Cramer-von Mises)" + p, draw)]


# ---------------------------------------------------------------------------
# Between items (Poisson processes)
# ---------------------------------------------------------------------------
def _estimated_coefficients(model: Any) -> int:
    if not _is_pi(model):
        return 0
    return int(np.isfinite(np.asarray(model.coeffs, dtype=float)).sum())


def _dispersion(ctx: Context) -> dict:
    """The items' observed and expected counts and the Pearson dispersion
    test of them (see the module docstring)."""

    def compute() -> dict:
        item_cif = _item_cif(ctx)
        labels, observed, expected = [], [], []
        dropped = 0
        for item, events, entry, close, explicit in _windows(ctx):
            if not explicit:
                dropped += 1
                continue
            cif = item_cif[item] if isinstance(item_cif, dict) else item_cif
            labels.append(item)
            observed.append(float(events.size))
            expected.append(float(cif(close) - cif(entry)))
        o, e = np.asarray(observed), np.asarray(expected)
        df = o.size - 1 - _estimated_coefficients(ctx.model)
        out = {
            "items": labels,
            "observed": o,
            "expected": e,
            "dropped": dropped,
            "df": df,
        }
        if df < 1 or not np.all(e > 0):
            out["statistic"] = out["p_value"] = float("nan")
            return out
        stat = float(np.sum((o - e) ** 2 / e))
        out["statistic"] = stat
        out["p_value"] = float(chi2.sf(stat, df))
        return out

    return ctx.cached("dispersion", lambda: _quiet(compute))


def _between_rows(ctx: Context) -> list[dict]:
    res = _dispersion(ctx)
    if res["dropped"]:
        ctx.notes.append(
            "Between items: {} failure-truncated item{} (observed up to "
            "{} last event, so with a fixed count) {} left out of the "
            "dispersion test.".format(
                res["dropped"],
                "" if res["dropped"] == 1 else "s",
                "its" if res["dropped"] == 1 else "their",
                "is" if res["dropped"] == 1 else "are",
            )
        )
    n_items = len(res["items"])
    if n_items < 2:
        ctx.notes.append(
            "Between items: not tested, as it needs two or more items "
            "observed to a fixed time (an end-of-observation, c = 1, "
            "row); the data have {}.".format(n_items)
        )
        return []
    covariates = tuple(ctx.names) if _is_pi(ctx.model) else ()
    test = "Pearson dispersion test of the item counts"
    p = res["p_value"]
    if not math.isfinite(p):
        detail = (
            "could not be computed: {} items leave no degrees of freedom "
            "after the model's {} count parameters".format(
                n_items, n_items - res["df"]
            )
        )
        return [
            {
                "test": test,
                "covariates": covariates,
                "p": p,
                "detail": detail,
            }
        ]
    e = res["expected"]
    if e.min() < 1:
        ctx.notes.append(
            "Between items: some items expect fewer than one event over "
            "their window, where the chi-squared approximation of the "
            "dispersion test is rough."
        )
    z = (res["observed"] - e) / np.sqrt(e)
    worst = int(np.argmax(np.abs(z)))
    if p < 2 * ctx.alpha:
        detail = (
            "the items' counts vary more than one intensity allows "
            "(item {} saw {:g} events, expected {})".format(
                res["items"][worst], res["observed"][worst], _fmt(e[worst])
            )
        )
    else:
        detail = "no evidence that the items differ"
    return [
        {
            "test": test,
            "covariates": covariates,
            "statistic": res["statistic"],
            "df": res["df"],
            "p": p,
            "detail": detail,
        }
    ]


def _between_panels(ctx: Context) -> list[Panel]:
    res = _dispersion(ctx)
    if len(res["items"]) < 2:
        return []
    e = res["expected"]
    z = (res["observed"] - e) / np.sqrt(e)
    labels = [str(item) for item in res["items"]]

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        cut = float(norm.isf(options.alpha / 2.0))
        k = np.arange(z.size)
        ax.vlines(k, 0, z, color=PALETTE[0], lw=1.2)
        ax.scatter(k, z, s=14, color=PALETTE[0], zorder=3)
        for level in (-cut, cut):
            ax.axhline(level, color=INK["secondary"], ls="--", lw=0.8)
        ax.axhline(0.0, color=INK["axis"], lw=0.8)
        if z.size <= 20:
            ax.set_xticks(k)
            ax.set_xticklabels(labels)
        ax.set_xlabel("item")
        ax.set_ylabel("(observed - expected) / sqrt(expected)")

    title = "between items"
    if math.isfinite(res["p_value"]):
        title += ": p = " + _fmt(res["p_value"])
    return [Panel(title, draw)]


# ---------------------------------------------------------------------------
# Repair (imperfect repair)
# ---------------------------------------------------------------------------
def _repair(ctx: Context) -> Any:
    return ctx.cached(
        "repair", lambda: _quiet(lambda: ctx.model.repair_test(ctx.alpha))
    )


def _repair_rows(ctx: Context) -> list[dict]:
    result = _repair(ctx)
    rows = []
    for which, test in (
        ("perfect", result.perfect),
        ("minimal", result.minimal),
    ):
        if test is None:
            ctx.notes.append(
                "Repair: the G1 renewal process has no minimal repair (its "
                "q scales each time between failures by 1 + q), so only "
                "perfect repair is tested."
            )
            continue
        name = "{} = {:g}".format(test.parameter, test.value)
        row: dict = {
            "test": "likelihood-ratio test of {} ({})".format(
                test.hypothesis, name
            ),
            "_which": which,
        }
        estimate = "{} = {}".format(result.parameter, _fmt(result.estimate))
        if not test.available:
            row.update(
                {
                    "p": float("nan"),
                    "verdict": "warn",
                    "detail": "could not be computed: " + test.message,
                }
            )
            rows.append(row)
            continue
        rejected = test.p_value < ctx.alpha
        row.update(
            {
                "statistic": test.statistic,
                "df": test.df,
                "p": test.p_value,
                "verdict": "ok" if rejected else "warn",
                "detail": "{}: {} {}{}".format(
                    estimate,
                    test.hypothesis,
                    "rejected" if rejected else "not rejected",
                    " (on the edge of its range)" if test.boundary else "",
                ),
            }
        )
        rows.append(row)
    return rows


def _profile_grid(model: Any) -> np.ndarray:
    lower, upper = model._restoration_bounds
    estimate = float(model.restoration)
    values = _repair_values(model)
    marks = [v for v in values.values() if v is not None] + [estimate]
    lo = float(lower) if lower is not None else min(marks) - 0.5
    hi = float(upper) if upper is not None else max(marks) + 0.5
    if lower is not None and lower == -1:
        lo = max(lo, min(marks) - 0.5, -0.9)  # G1: q = -1 is degenerate
    grid = np.linspace(lo, hi, 21)
    return np.unique(np.r_[grid, marks])


def _repair_values(model: Any) -> dict:
    from surpyval.recurrent.renewal.renewal_model import REPAIR_VALUES

    values = REPAIR_VALUES[model.kind]
    return {"perfect": values["perfect"], "minimal": values["minimal"]}


def _repair_panels(ctx: Context) -> list[Panel]:
    model = ctx.model
    grid = _profile_grid(model)
    ll = _quiet(lambda: np.array([model._profile_ll(float(v)) for v in grid]))
    ll_hat = float(model.log_likelihood)
    estimate = float(model.restoration)
    values = _repair_values(model)
    name = model._restoration_param_name

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        keep = np.isfinite(ll)
        ax.plot(grid[keep], ll[keep] - ll_hat, color=PALETTE[0])
        crit = float(chi2.isf(options.alpha, 1)) / 2.0
        ax.axhline(
            -crit,
            color=INK["secondary"],
            ls="--",
            lw=0.8,
            label="{:g}% limit".format(100 * (1 - options.alpha)),
        )
        ax.axvline(
            estimate, color=PALETTE[0], ls=":", lw=1.0, label="estimate"
        )
        for k, (label, value) in enumerate(values.items()):
            if value is not None:
                ax.axvline(
                    value,
                    color=PALETTE[1 + k],
                    lw=1.2,
                    label="{} repair".format(label),
                )
        ax.set_xlabel(name)
        ax.set_ylabel("profile log-likelihood (relative)")
        ax.legend(loc="lower right", fontsize="small", frameon=False)

    return [Panel("repair: profile of {}".format(name), draw)]


# ---------------------------------------------------------------------------
# Refusals and notes
# ---------------------------------------------------------------------------
def _validate(model: Any) -> None:
    from surpyval.recurrent.diagnostics import validate_diagnostic_data

    if getattr(model, "data", None) is None:
        raise ValueError(
            "diagnose() needs a model fitted from data, and this one has "
            "none: a model built from its parameters (from_params / "
            "fit_from_parameters) or restored with from_dict / from_json "
            "carries no data. Fit the model to the data and diagnose that "
            "fit."
        )
    if not hasattr(model, "_neg_ll"):
        raise ValueError(
            "diagnose() needs a maximum-likelihood fit, and this model has "
            "no likelihood (a how='MSE' fit minimises squared error on the "
            "mean cumulative function); refit it with how='MLE'."
        )
    if getattr(model, "maximum", None) == "no finite maximum":
        raise ValueError(
            "diagnose() needs a fit with a finite maximum, and this one "
            "has none (model.maximum is 'no finite maximum', as the fit "
            "warned); the tests of its fit and repair have no estimate "
            "to test. Fit a simpler model, or one with more data."
        )
    validate_diagnostic_data(model.data, "diagnose()")


def _notes(ctx: Context) -> list[str]:
    model = ctx.model
    notes = ["Model: " + _process(model) + "."]
    if _constant_rate(model):
        notes.append(
            "Trend: the model assumes a constant rate, so the trend tests "
            "are tests of it. Laplace and MIL-HDBK-189C test against a "
            "homogeneous Poisson process, Lewis-Robinson against any "
            "renewal process (the Laplace statistic over the coefficient "
            "of variation of the times between failures)."
        )
    else:
        notes.append(
            "Trend: the model allows the rate to change, so the trend "
            "tests are a model-choice screen, not tests of the model: a "
            "row warns when its test does not reject its simpler null "
            "(Laplace and MIL-HDBK-189C: a homogeneous Poisson process; "
            "Lewis-Robinson: a renewal process), which may then do."
        )
    notes.append(
        "Fit: the Cramer-von Mises p-value is a parametric bootstrap of "
        "{} refits from the seed {}, so the report is reproducible; "
        "model.cramer_von_mises(n_boot=..., random_state=...) gives a "
        "finer one.".format(N_BOOT, SEED)
    )
    if _is_renewal(model):
        notes.append(
            "Repair: neither kind of repair is an assumption of the model, "
            "so the likelihood-ratio tests are a model-choice screen: a "
            "row warns when its repair is not rejected, and the simpler "
            "model then fits as well. model.repair_test() gives the "
            "restricted fits."
        )
        notes.append(
            "Between items: not tested for an imperfect-repair model, "
            "whose item counts are not Poisson."
        )
    else:
        notes.append(
            "Repair: a Poisson process assumes minimal repair (as bad as "
            "old), which these checks do not test; "
            "GeneralizedRenewal.fit(...).repair_test() tests it against "
            "the Kijima model, of which the power law (CrowAMSAA) with "
            "q = 1 is the minimal-repair case."
        )
    if getattr(model.data, "window_map", None):
        notes.append(
            "Each observation window of an item with gaps in its "
            "observation is counted as an item of its own."
        )
    if len(ctx.columns) != len(ctx.names):
        notes.append(
            "columns= has no effect here: every recurrent check is about "
            "the model as a whole."
        )
    return notes


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------
def _row_test(row: Any) -> str:
    return row.get("_test", "") if row is not None else ""


def _lr_rules_out_renewal(ctx: Context) -> bool:
    """True unless Lewis-Robinson ran and did not reject a renewal
    process."""
    return _rejects(ctx, _LR) is not False


def _direction(ctx: Context) -> str:
    words = {"increasing": "increases", "decreasing": "decreases"}
    for test in (_LAPLACE, _MIL, _LR):
        res = _trend_results(ctx)[test]
        if not isinstance(res, str) and res["direction"] in words:
            return words[res["direction"]]
    return "changes"


def _changing_rate(row: Any, ctx: Context) -> str:
    return (
        "The rate of failures {} with time, which a constant-rate model "
        "(an HPP) cannot describe, and not as a renewal process would "
        "(Lewis-Robinson). Fit a model whose rate changes: an NHPP, "
        "CrowAMSAA.fit (a power law) or CoxLewis.fit (log-linear), or an "
        "imperfect-repair model, GeneralizedRenewal.fit or ARA.fit, whose "
        "repair_test() then says how good the repairs are. Compare the "
        "fits with aic().".format(_direction(ctx))
    )


def _changing_rate_pi(row: Any, ctx: Context) -> str:
    return (
        "The rate of failures {} with time, which the constant baseline "
        "of ProportionalIntensityHPP cannot describe. Give the model a "
        "baseline that changes: ProportionalIntensityNHPP.fit(x, Z, ..., "
        "baseline=CrowAMSAA) (a power law) or baseline=CoxLewis "
        "(log-linear). Compare the fits with aic().".format(_direction(ctx))
    )


def _renewal_like(row: Any, ctx: Context) -> str:
    return (
        "A test against a constant rate ({}) finds a trend, but the "
        "Lewis-Robinson test, which allows for the spread of the times "
        "between failures, does not reject a renewal process: the times "
        "between failures may come from one distribution that is not "
        "exponential, with no trend at all. Compare a renewal process: "
        "GeneralizedRenewal.fit (q = 0 is perfect repair; its "
        "repair_test() tests it).".format(
            " and ".join(t for t in (_LAPLACE, _MIL) if _rejects(ctx, t))
        )
    )


def _hpp_suffices(row: Any, ctx: Context) -> str:
    return (
        "Neither the Laplace nor the MIL-HDBK-189C test finds a trend: a "
        "constant rate, HPP.fit, describes these data with fewer "
        "parameters than {}. Compare the two fits with aic().".format(
            _process(ctx.model)
        )
    )


def _pi_hpp_suffices(row: Any, ctx: Context) -> str:
    return (
        "Neither the Laplace nor the MIL-HDBK-189C test finds a trend: a "
        "constant baseline, ProportionalIntensityHPP.fit, describes these "
        "data with fewer parameters. Compare the two fits with aic()."
    )


def _no_trend(ctx: Context) -> bool:
    return _rejects(ctx, _LAPLACE) is False and _rejects(ctx, _MIL) is False


def _trend_found(ctx: Context) -> bool:
    return bool(_rejects(ctx, _LAPLACE) or _rejects(ctx, _MIL))


def _renewal_not_ruled_out(row: Any, ctx: Context) -> str:
    return (
        "Lewis-Robinson does not reject a renewal process, though there "
        "is a trend against a constant rate: the times between failures "
        "may be identically distributed but not exponential. Compare an "
        "imperfect-repair model, GeneralizedRenewal.fit, whose "
        "repair_test() tests perfect repair (a renewal process) and "
        "minimal repair (an NHPP) against the data."
    )


def _renewal_no_trend(row: Any, ctx: Context) -> str:
    return (
        "No test finds a trend in the rate of failures: the repairs may "
        "be as good as new. The repair rows test that directly "
        "(repair_test()); with exponential times between failures the "
        "process is a constant rate, HPP.fit."
    )


#: The intensities the misfit advice offers, with what each is.
_OTHER_INTENSITIES = (
    ("CrowAMSAA", "a power law"),
    ("CoxLewis", "log-linear"),
    ("Duane", "a power law in Duane's form"),
    ("HPP", "constant"),
)


def _poisson_misfit(row: Any, ctx: Context) -> str:
    own = _INTENSITY_FITTERS.get(ctx.model.dist.name)
    others = ", ".join(
        "{}.fit ({})".format(name, what)
        for name, what in _OTHER_INTENSITIES
        if name != own
        and not (own == "CrowAMSAA" and name == "Duane")
        and not (own == "Duane" and name == "CrowAMSAA")
    )
    return (
        "The event times depart from the fitted {} (the Cramer-von Mises "
        "test; see d.plot('fit')). Try another intensity -- {} -- or a "
        "different repair assumption: GeneralizedRenewal.fit or ARA.fit "
        "(imperfect repair), whose repair_test() says how good the "
        "repairs are. Compare the fits with aic().".format(
            _process(ctx.model), others
        )
    )


def _renewal_misfit(row: Any, ctx: Context) -> str:
    return (
        "The event times depart from the fitted {} (the Cramer-von Mises "
        "test; see d.plot('fit')). Try another lifetime distribution "
        "(GeneralizedRenewal.fit(..., dist=...)), the other Kijima type "
        "(kijima='ii'), a longer memory for ARA or ARI (ARA.fit(..., "
        "m=...)), or another repair model (ARA.fit, ARI.fit with a "
        "baseline=, GeneralizedOneRenewal.fit). Compare the fits with "
        "aic().".format(_process(ctx.model))
    )


def _pi_misfit(row: Any, ctx: Context) -> str:
    return (
        "The event times depart from the fitted {} (the Cramer-von Mises "
        "test; see d.plot('fit')). Try another baseline intensity, "
        "ProportionalIntensityNHPP.fit(x, Z, ..., baseline=...) with "
        "CrowAMSAA, CoxLewis or Duane, or ProportionalIntensityHPP.fit "
        "for a constant one; compare the fits with aic().".format(
            _process(ctx.model)
        )
    )


def _covariates_advice(row: Any, ctx: Context) -> str:
    fitter = _INTENSITY_FITTERS.get(ctx.model.dist.name, "CrowAMSAA")
    if fitter == "HPP":
        how = "ProportionalIntensityHPP.fit(x, Z, i=..., c=...)"
    else:
        how = (
            "ProportionalIntensityNHPP.fit(x, Z, i=..., c=..., "
            "baseline={})".format(fitter)
        )
    return (
        "The items' event counts vary more than one intensity allows "
        "(see d.plot('between_items')). If something known about the "
        "items -- their make, duty or environment -- explains it, give it "
        "as covariates Z in a proportional-intensity model, {}; otherwise "
        "fit the items, or groups of them, separately.".format(how)
    )


def _more_covariates(row: Any, ctx: Context) -> str:
    return (
        "The items' event counts vary more than the covariates explain "
        "(see d.plot('between_items')). Add to Z what else differs "
        "between the items (refit with ProportionalIntensityNHPP.fit or "
        "ProportionalIntensityHPP.fit), or fit groups of them "
        "separately."
    )


def _perfect_repair(row: Any, ctx: Context) -> str:
    model = ctx.model
    name = model._restoration_param_name
    value = _repair_values(model)["perfect"]
    return (
        "The data are consistent with perfect repair ({} = {:g}, as good "
        "as new): a renewal process, whose times between failures all "
        "follow the one lifetime distribution. It is the {} fitted to "
        "the times between failures, model.data.to_gaps() (EventHistoryData"
        ".to_gaps), with one parameter fewer.".format(
            name, value, model.model.dist.name
        )
    )


def _minimal_repair(row: Any, ctx: Context) -> str:
    model = ctx.model
    name = model._restoration_param_name
    value = _repair_values(model)["minimal"]
    return (
        "The data are consistent with minimal repair ({} = {:g}, as bad "
        "as old): a non-homogeneous Poisson process whose cumulative "
        "intensity is the lifetime distribution's cumulative hazard -- "
        "for a Weibull lifetime the power law, CrowAMSAA.fit -- with one "
        "parameter fewer.".format(name, value)
    )


def _ari_minimal(row: Any, ctx: Context) -> str:
    model = ctx.model
    fitter = _INTENSITY_FITTERS.get(model.model.dist.name, "CrowAMSAA")
    return (
        "The data are consistent with minimal repair (rho = 0, as bad as "
        "old): the baseline intensity alone, {}.fit, with one parameter "
        "fewer.".format(fitter)
    )


def _repair_is(which: str) -> Any:
    def when(row: Any, ctx: Context) -> bool:
        return row.get("_which") == which and math.isfinite(row["p"])

    return when


def _ari(ctx: Context) -> bool:
    return getattr(ctx.model, "kind", None) == "ARI Recurrence"


def _ari_baseline(fitter: str) -> Any:
    def when(row: Any, ctx: Context) -> bool:
        return (
            _repair_is("minimal")(row, ctx)
            and _ari(ctx)
            and _INTENSITY_FITTERS.get(ctx.model.model.dist.name) == fitter
        )

    return when


def _covariate_names(model: Any) -> list[str]:
    """The covariates of a proportional-intensity model, as its
    ``summary()`` names its coefficients; none for the others."""
    if _is_pi(model):
        return list(model._coefficient_names())
    return []


_TREND = Check(
    "trend",
    "The rate of failures shows the trend, or lack of one, the model "
    "assumes (Laplace, MIL-HDBK-189C, Lewis-Robinson).",
    rows=_trend_rows,
    panels=_trend_panels,
)
_FIT = Check(
    "fit",
    "The event times follow the fitted intensity (Cramer-von Mises).",
    rows=_fit_rows,
    panels=_fit_panels,
)
_BETWEEN = Check(
    "between_items",
    "The items share one intensity (after their covariates): their event "
    "counts vary no more than a Poisson process allows.",
    rows=_between_rows,
    panels=_between_panels,
)
_REPAIR = Check(
    "repair",
    "Repair is neither as good as new nor as bad as old (where either "
    "fits as well, the simpler model does).",
    rows=_repair_rows,
    panels=_repair_panels,
)


def _is_constant(ctx: Context) -> bool:
    return _constant_rate(ctx.model)


#: The Poisson processes (HPP, Crow-AMSAA, Duane, Cox-Lewis).
POISSON = register(
    Family(
        name="Recurrent events: Poisson process",
        issue=873,
        applies=_is_poisson,
        validate=_validate,
        covariates=_covariate_names,
        checks=(_TREND, _FIT, _BETWEEN),
        advice=(
            AdviceRule(
                "trend",
                _changing_rate,
                (
                    "CrowAMSAA.fit",
                    "CoxLewis.fit",
                    "GeneralizedRenewal.fit",
                    "ARA.fit",
                    "repair_test",
                    "aic",
                ),
                when=lambda row, ctx: _is_constant(ctx)
                and _lr_rules_out_renewal(ctx),
            ),
            AdviceRule(
                "trend",
                _renewal_like,
                ("GeneralizedRenewal.fit", "repair_test"),
                when=lambda row, ctx: _is_constant(ctx)
                and _row_test(row) in (_LAPLACE, _MIL)
                and not _lr_rules_out_renewal(ctx),
            ),
            AdviceRule(
                "trend",
                _hpp_suffices,
                ("HPP.fit", "aic"),
                when=lambda row, ctx: not _is_constant(ctx)
                and _row_test(row) in (_LAPLACE, _MIL)
                and _no_trend(ctx),
                on=("warn",),
            ),
            AdviceRule(
                "trend",
                _renewal_not_ruled_out,
                ("GeneralizedRenewal.fit", "repair_test"),
                when=lambda row, ctx: not _is_constant(ctx)
                and _row_test(row) == _LR
                and _rejects(ctx, _LR) is False
                and _trend_found(ctx),
                on=("warn",),
            ),
            AdviceRule(
                "fit",
                _poisson_misfit,
                (
                    "CrowAMSAA.fit",
                    "CoxLewis.fit",
                    "Duane.fit",
                    "HPP.fit",
                    "GeneralizedRenewal.fit",
                    "ARA.fit",
                    "repair_test",
                    "aic",
                ),
            ),
            AdviceRule(
                "between_items",
                _covariates_advice,
                (
                    "ProportionalIntensityNHPP.fit(baseline=)",
                    "ProportionalIntensityHPP.fit",
                ),
            ),
        ),
        notes=_notes,
    )
)

#: The imperfect-repair models (generalized renewal, G1, ARA, ARI).
IMPERFECT_REPAIR = register(
    Family(
        name="Recurrent events: imperfect repair",
        issue=873,
        applies=_is_renewal,
        validate=_validate,
        covariates=_covariate_names,
        checks=(_TREND, _FIT, _REPAIR),
        advice=(
            AdviceRule(
                "trend",
                _renewal_no_trend,
                ("repair_test", "HPP.fit"),
                when=lambda row, ctx: _row_test(row) in (_LAPLACE, _MIL)
                and _no_trend(ctx),
                on=("warn",),
            ),
            AdviceRule(
                "repair",
                _perfect_repair,
                ("EventHistoryData.to_gaps",),
                when=lambda row, ctx: _repair_is("perfect")(row, ctx)
                and not _ari(ctx),
                on=("warn",),
            ),
            AdviceRule(
                "repair",
                _minimal_repair,
                ("CrowAMSAA.fit",),
                when=lambda row, ctx: _repair_is("minimal")(row, ctx)
                and not _ari(ctx),
                on=("warn",),
            ),
            *(
                AdviceRule(
                    "repair",
                    _ari_minimal,
                    ("{}.fit".format(fitter),),
                    when=_ari_baseline(fitter),
                    on=("warn",),
                )
                for fitter in _INTENSITY_FITTERS.values()
            ),
            AdviceRule(
                "fit",
                _renewal_misfit,
                (
                    "GeneralizedRenewal.fit(dist=)",
                    "GeneralizedRenewal.fit(kijima=)",
                    "ARA.fit(m=)",
                    "ARI.fit(baseline=)",
                    "GeneralizedOneRenewal.fit",
                    "aic",
                ),
            ),
        ),
        notes=_notes,
    )
)

#: The proportional-intensity regressions (PI-HPP, PI-NHPP).
PROPORTIONAL_INTENSITY = register(
    Family(
        name="Recurrent events: proportional intensity",
        issue=873,
        applies=_is_pi,
        validate=_validate,
        covariates=_covariate_names,
        checks=(_TREND, _FIT, _BETWEEN),
        advice=(
            AdviceRule(
                "trend",
                _changing_rate_pi,
                ("ProportionalIntensityNHPP.fit(baseline=)", "aic"),
                when=lambda row, ctx: _is_constant(ctx)
                and _lr_rules_out_renewal(ctx),
            ),
            AdviceRule(
                "trend",
                _renewal_like,
                ("GeneralizedRenewal.fit", "repair_test"),
                when=lambda row, ctx: _is_constant(ctx)
                and _row_test(row) in (_LAPLACE, _MIL)
                and not _lr_rules_out_renewal(ctx),
            ),
            AdviceRule(
                "trend",
                _pi_hpp_suffices,
                ("ProportionalIntensityHPP.fit", "aic"),
                when=lambda row, ctx: not _is_constant(ctx)
                and _row_test(row) in (_LAPLACE, _MIL)
                and _no_trend(ctx),
                on=("warn",),
            ),
            AdviceRule(
                "fit",
                _pi_misfit,
                (
                    "ProportionalIntensityNHPP.fit(baseline=)",
                    "ProportionalIntensityHPP.fit",
                    "aic",
                ),
            ),
            AdviceRule(
                "between_items",
                _more_covariates,
                (
                    "ProportionalIntensityNHPP.fit",
                    "ProportionalIntensityHPP.fit",
                ),
            ),
        ),
        notes=_notes,
    )
)
