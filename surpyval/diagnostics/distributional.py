"""``diagnose()`` for the distributional regressions
(``WeibullDistributional``, ``LogNormalDistributional``, ...,
``Distributional(dist)``).

A distributional regression gives every parameter of its distribution its
own linear predictor, ``g_k(theta_k(z)) = g_k(theta_k) + s_k x_k' b_k``,
so each unit has its own distribution. The checks ask of it what the
parametric PH and AFT families ask of theirs -- does the distribution fit,
does each covariate act linearly, does any one unit decide a coefficient
-- with each covariate checked on each parameter whose predictor holds it.
They share the PH and AFT families' machinery
(:mod:`surpyval.diagnostics.ph_aft`): every test is a likelihood-ratio
test of the fitted model against a larger one, whose likelihood is the
package's own (``regression_neg_ll``, with every kind of censoring and
truncation the fit took), maximised from the fitted parameters.

>>> import numpy as np
>>> from surpyval import WeibullDistributional
>>> rng = np.random.default_rng(3)
>>> load = rng.uniform(0, 1, 300)
>>> x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
>>> d = WeibullDistributional.fit(x, load).diagnose()
>>> sorted(set(d.table["check"]))
['distribution', 'functional_form', 'influence']
>>> list(d.table.loc[d.table["check"] == "influence", "parameter"])
['alpha', 'beta']

The checks
----------
``distribution``
    Drawn as the Cox-Snell residuals ``H(x_i | z_i)``, each unit's fitted
    cumulative hazard at its own parameters, on exponential probability
    paper (truncated at ``H(tl_i | z_i)`` and ``H(tr_i | z_i)``; a left- or
    interval-censored residual is the interval its time is in, and the
    plotting positions are Turnbull's): a model whose distribution fits
    puts them on the line ``y = x``. As for the PH family they have no
    useful test of their own (the parameters were fitted to the same
    data). The screen, never a fail: the AIC of the model against each
    other built-in distribution (:data:`CANDIDATES`) fitted as a
    distributional regression to the same data, with the same covariates
    -- on the candidate's life parameter (:data:`LIFE`: its scale,
    location or rate) the columns of the model's life parameter, on each
    of its other parameters the columns of all the model's others. It
    warns when one's AIC is more than :data:`~.ph_aft.AIC_MARGIN` (2)
    lower; its ``statistic`` is the fitted model's AIC less the best
    other's. A candidate whose fit is not a verified maximum is set aside,
    and a note says so.
``functional_form``
    For each covariate with three or more distinct values and each
    parameter whose predictor holds it, the likelihood-ratio test (1 df)
    of adding its square to that parameter's predictor (as the model's
    covariates enter it, about the model's ``center``): whether it acts
    linearly on that parameter's link scale. The ``parameter`` column
    names the parameter. Drawn as each unit's score residual for the
    parameter, ``d log L_i / d eta_ik`` (the change of its
    log-likelihood with its own linear predictor of parameter ``k``),
    against the covariate, with a lowess trend: a predictor of the right
    form leaves no trend. For the log hazard of a PH model the score
    residual is the martingale residual, which the PH family draws; here
    every row has one, whatever its censoring and truncation.
``influence``
    For each coefficient (a covariate on a parameter), the largest
    ``|dfbeta| / se``: the one-step change ``I^-1 U_i`` of the coefficient
    when one unit is dropped (the observed information ``I`` and the
    unit's score ``U_i``; a row with count ``n`` is ``n`` units), in
    standard errors. It warns when one unit moves a coefficient by more
    than one standard error
    (:data:`~surpyval.diagnostics.cox.INFLUENCE_LIMIT`); a screen, never a
    fail.

The covariates' kinds are the Cox family's: two distinct values is
*binary*, three to five *categorical*, more *continuous*. The report's
covariates are the columns of ``Z`` (``columns=`` names them), and each
row's ``parameter`` says which parameter's coefficient it is about.

A fit with weights ``w`` is refused (its likelihood-ratio statistics are
not chi-squared and its AIC is not comparable), as is a penalised fit (no
p-values), a model restored without its data, one whose fit is not a
verified maximum, and one with aliased coefficients.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ._shared import _covariate_kind, _fmt, _gap, _note
from .cox import INFLUENCE_LIMIT
from .ph_aft import (
    AIC_MARGIN,
    _base,
    _covariance,
    _draw_probability,
    _fitted_null,
    _in_layout,
    _lr_test,
    _neg_ll,
    _view,
)
from .registry import AdviceRule, Check, Context, Family, Panel, register

#: The distributions the ``distribution`` check compares the model with,
#: each with its prebuilt ``<name>Distributional`` fitter.
CANDIDATES = (
    "Exponential",
    "Normal",
    "Weibull",
    "Gumbel",
    "Logistic",
    "LogNormal",
    "Gamma",
    "LogLogistic",
)
#: The parameter that sets where each distribution's lives are (its
#: scale, location or rate): the candidates of the AIC comparison take
#: the columns of the model's own, and their other parameters the columns
#: of the model's others. A distribution not listed (a Beta) gives every
#: candidate parameter every column the model uses.
LIFE = {
    "Exponential": "failure_rate",
    "Rayleigh": "sigma",
    "Weibull": "alpha",
    "Gamma": "beta",
    "LogNormal": "mu",
    "LogLogistic": "alpha",
    "Normal": "mu",
    "Gumbel": "mu",
    "GumbelLEV": "mu",
    "Logistic": "mu",
    "ExpoWeibull": "alpha",
}


# ---------------------------------------------------------------------------
# The model's covariates and coefficients
# ---------------------------------------------------------------------------
def _positions(model: Any) -> list[np.ndarray]:
    """Each parameter's columns of ``Z``."""
    return [np.asarray(p, dtype=int) for p in model.covariate_positions]


def _column_names(model: Any) -> list[str]:
    """The name of each column of ``Z``: as the coefficients name it
    (``alpha_load`` is the column ``load``), else ``coef_<j>``."""
    out = [f"coef_{j}" for j in range(model._n_covariates())]
    params = list(model.distribution.parameter_names)
    coefs = model._coefficient_names()
    owner = [
        params[k]
        for k, cols in enumerate(_positions(model))
        for _ in range(cols.size)
    ]
    columns = np.concatenate([np.zeros(0, dtype=int), *_positions(model)])
    for name, param, j in zip(coefs, owner, columns.tolist()):
        out[j] = name[len(param) + 1 :]
    return out


def _coefficients(ctx: Context) -> list[dict]:
    """Each coefficient of a checked covariate: ``{"k", "parameter", "j",
    "name", "index"}``, ``index`` its place in ``params``."""

    def compute() -> list[dict]:
        m = ctx.model
        params = list(m.distribution.parameter_names)
        out = []
        at = int(m.k_dist)
        for k, cols in enumerate(_positions(m)):
            for j in cols.tolist():
                if j in ctx.columns:
                    out.append(
                        {
                            "k": k,
                            "parameter": params[k],
                            "j": j,
                            "name": ctx.names[j],
                            "index": at,
                        }
                    )
                at += 1
        return out

    return ctx.cached("coefficients", compute)


def _kind(ctx: Context, j: int) -> str:
    return _covariate_kind(int(np.unique(_base(ctx).raw[:, j]).size))


def _own(ctx: Context) -> str:
    """How the advice names the model's fitter: ``WeibullDistributional``,
    or ``Distributional(<name>)`` for a distribution with no prebuilt
    one."""
    name = ctx.model.distribution.name
    if name in CANDIDATES:
        return name + "Distributional"
    return f"Distributional({name})"


# ---------------------------------------------------------------------------
# The distribution
# ---------------------------------------------------------------------------
def _candidate_covariates(model: Any, candidate: str) -> dict:
    """The candidate's covariates, by parameter: the model's life
    parameter's columns on its life parameter, the model's other
    parameters' columns on each of its others."""
    import surpyval

    params = list(model.distribution.parameter_names)
    positions = _positions(model)
    every = sorted({j for p in positions for j in p.tolist()})
    life = LIFE.get(model.distribution.name)
    if life is None:
        life_cols, other_cols = every, every
    else:
        life_cols = positions[params.index(life)].tolist()
        other_cols = sorted(
            {
                j
                for k, p in enumerate(positions)
                if params[k] != life
                for j in p.tolist()
            }
        )
    dist = getattr(surpyval, candidate)
    return {
        p: (life_cols if p == LIFE[candidate] else other_cols)
        for p in dist.parameter_names
    }


def _refit(ctx: Context, candidate: str) -> tuple[float, str, str]:
    """``(aic, status, reason)`` of the candidate refitted to the model's
    data: status ``"ranked"`` (a verified maximum), ``"set aside"`` (not
    one) or ``"failed"``."""
    import surpyval

    b = _base(ctx)
    data = b.data
    fitter = getattr(surpyval, candidate + "Distributional")
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = fitter.fit(
                data.x,
                b.raw,
                c=data.c,
                n=data.n,
                t=data.t,
                covariates=_candidate_covariates(b.model, candidate),
                center=b.model._has_center(),
            )
            aic = float(fitted.aic())
    except Exception as error:  # noqa: BLE001 -- a candidate may fail
        text = str(error).strip().splitlines()
        return (
            float("nan"),
            "failed",
            text[0][:100] if text else type(error).__name__,
        )
    if fitted.maximum != "verified" or not np.isfinite(aic):
        return aic, "set aside", f"not a verified maximum ({fitted.maximum})"
    return aic, "ranked", ""


def _aic_candidates(ctx: Context) -> list[dict]:
    """The model, then each other candidate: ``{"label", "name", "aic",
    "status", "reason"}``."""

    def compute() -> list[dict]:
        model = ctx.model
        own = model.distribution.name
        out = [
            {
                "label": _own(ctx),
                "name": own,
                "aic": float(model.aic()),
                "status": "fitted",
                "reason": "",
            }
        ]
        for name in CANDIDATES:
            if name == own:
                continue
            aic, status, reason = _refit(ctx, name)
            out.append(
                {
                    "label": name + "Distributional",
                    "name": name,
                    "aic": aic,
                    "status": status,
                    "reason": reason,
                }
            )
        return out

    return ctx.cached("aic", compute)


def _distribution_rows(ctx: Context) -> list[dict]:
    fixed = list(ctx.model.fixed)
    if fixed:
        _note(
            ctx,
            "The AIC comparison is left out: the fit holds {} fixed, which "
            "the other distributions cannot hold.".format(", ".join(fixed)),
        )
        return []
    candidates = _aic_candidates(ctx)
    own = candidates[0]
    others = [c for c in candidates[1:] if c["status"] == "ranked"]
    skipped = [c for c in candidates[1:] if c["status"] != "ranked"]
    if skipped:
        _note(
            ctx,
            "AIC comparison: {} left out ({}).".format(
                ", ".join(c["label"] for c in skipped),
                "; ".join(
                    "{}: {}".format(c["label"], c["reason"]) for c in skipped
                ),
            ),
        )
    if not others:
        return []
    best = min(others, key=lambda c: c["aic"])
    delta = own["aic"] - best["aic"]
    if delta > AIC_MARGIN:
        detail = "{} fits better: its AIC is {} lower".format(
            best["label"], _fmt(delta)
        )
    else:
        detail = (
            "no other distribution has an AIC more than {:g} lower (the "
            "best of them, {}, is {})".format(
                AIC_MARGIN, best["label"], _gap(delta)
            )
        )
    return [
        {
            "test": "AIC against the other distributions",
            "covariates": (),
            "statistic": delta,
            "verdict": "warn" if delta > AIC_MARGIN else "ok",
            "detail": detail,
            "_best": best["name"],
            "_label": best["label"],
            "_delta": delta,
        }
    ]


def _aic_panel(ctx: Context) -> Panel:
    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        shown = [
            c
            for c in _aic_candidates(ctx)
            if c["status"] in ("fitted", "ranked")
        ]
        best = min(c["aic"] for c in shown)
        order = sorted(shown, key=lambda c: c["aic"])
        y = np.arange(len(order))
        colours = [
            PALETTE[1] if c["status"] == "fitted" else PALETTE[0]
            for c in order
        ]
        ax.barh(y, [c["aic"] - best for c in order], color=colours)
        ax.set_yticks(y)
        ax.set_yticklabels([c["label"] for c in order])
        ax.invert_yaxis()
        ax.axvline(AIC_MARGIN, color=INK["secondary"], ls="--", lw=0.8)
        ax.set_xlabel("AIC - lowest AIC")

    return Panel("AIC by distribution", draw)


def _distribution_panels(ctx: Context) -> list[Panel]:
    panels = [
        Panel(
            "Cox-Snell residuals (exponential paper)",
            lambda ax, options: _draw_probability(ax, ctx, "PH"),
        )
    ]
    if not ctx.model.fixed:
        panels.append(_aic_panel(ctx))
    return panels


# ---------------------------------------------------------------------------
# Scores: the score residuals and the influence
# ---------------------------------------------------------------------------
def _scores(ctx: Context) -> np.ndarray:
    """Each unit's score, ``d log L_i / d theta`` of every parameter (one
    row per data row; a row with count ``n`` is one of ``n`` identical
    units)."""

    def compute() -> np.ndarray:
        from autograd import grad, jacobian

        b = _base(ctx)
        theta = b.params

        def gradient(w: Any) -> Any:
            return grad(_neg_ll(b.fitter, _view(b, b.Z, w)))(theta)

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The likelihood is linear in the rows' weights, so this is
            # exact: d(grad of -loglik)/d(weight of row i) is -n_i u_i.
            J = np.asarray(jacobian(gradient)(np.ones(b.n.size)), dtype=float)
        return -J.T / b.n[:, None]

    return ctx.cached("scores", compute)


def _score_residuals(ctx: Context) -> np.ndarray:
    """Each unit's score residual for each parameter, ``d log L_i / d
    eta_ik``: the score of the parameter's value at the reference row,
    ``theta_k``, times ``d theta_k / d eta``, ``theta_k`` for a log link
    and 1 for the identity."""
    b = _base(ctx)
    scores = _scores(ctx)
    links = ctx.model.links
    factor = np.array(
        [
            b.params[k] if links[k] == "log" else 1.0
            for k in range(int(b.k_dist))
        ]
    )
    return scores[:, : b.k_dist] * factor[None, :]


def _dfbeta(ctx: Context) -> tuple[np.ndarray, np.ndarray]:
    """``(dfbeta, se)``: each unit's one-step change ``I^-1 U_i`` of every
    coefficient (one column per entry of ``params`` past the
    distribution's), and their standard errors; ``nan`` for a fixed
    coefficient, and everywhere if the information is not positive
    definite."""

    def compute() -> tuple[np.ndarray, np.ndarray]:
        b = _base(ctx)
        idx = np.flatnonzero(b.free)
        f = _neg_ll(b.fitter, _view(b, b.Z))
        cov = _covariance(f, b.params, b.free)
        moved = _scores(ctx)[:, idx] @ cov
        dfbeta = np.full((b.n.size, b.params.size), np.nan)
        se = np.full(b.params.size, np.nan)
        dfbeta[:, idx] = moved
        se[idx] = np.sqrt(np.diag(cov))
        return dfbeta[:, b.k_dist :], se[b.k_dist :]

    return ctx.cached("dfbeta", compute)


# ---------------------------------------------------------------------------
# Functional form
# ---------------------------------------------------------------------------
def _form_coefficients(ctx: Context) -> list[dict]:
    return [c for c in _coefficients(ctx) if _kind(ctx, c["j"]) != "binary"]


def _square_test(ctx: Context, k: int, j: int) -> dict:
    """The likelihood-ratio test of adding the square of column ``j`` to
    the predictor of parameter ``k``."""

    def compute() -> dict:
        b = _base(ctx)
        positions = _positions(b.model)
        width = b.Z.shape[1]
        z = b.Z[:, j]
        scale = float(np.std(z)) or 1.0
        Z = np.column_stack([b.Z, (z / scale) ** 2])
        extended = list(positions)
        extended[k] = np.append(positions[k], width)
        fitter = b.fitter._bound(tuple(extended), width + 1, b.fitter.links)
        at = int(b.k_dist) + sum(p.size for p in positions[: k + 1])
        start = np.insert(b.params, at, 0.0)
        null = np.insert(b.free, at, False)
        alt = np.insert(b.free, at, True)
        bounds = list(b.bounds)
        bounds.insert(at, (None, None))
        f = _neg_ll(fitter, _view(b, Z))
        shared = _in_layout(_fitted_null(ctx), 1, at)
        return _lr_test(f, start, null, alt, bounds, null=shared)

    return ctx.cached(f"square_{k}_{j}", compute)


def _form_rows(ctx: Context) -> list[dict]:
    checked = _coefficients(ctx)
    skipped = sorted(
        {c["name"] for c in checked if _kind(ctx, c["j"]) == "binary"}
    )
    if skipped:
        _note(
            ctx,
            "Functional form is tested for covariates with three or more "
            "distinct values; {} {} two, and the effect of a two-valued "
            "covariate is linear by construction.".format(
                ", ".join(skipped), "has" if len(skipped) == 1 else "have"
            ),
        )
    rows = []
    for c in _form_coefficients(ctx):
        res = _square_test(ctx, c["k"], c["j"])
        p = res["p"]
        if not np.isfinite(p):
            detail = "the test could not be computed: " + res["reason"]
        elif p < 2 * ctx.alpha:
            detail = "its effect on {} is not linear on the link scale".format(
                c["parameter"]
            )
        else:
            detail = "no evidence against a linear effect on {}".format(
                c["parameter"]
            )
        rows.append(
            {
                "test": "likelihood-ratio test of a quadratic term",
                "covariates": (c["name"],),
                "parameter": c["parameter"],
                "statistic": res["statistic"],
                "df": res["df"],
                "p": p,
                "detail": detail,
            }
        )
    return rows


def _form_panels(ctx: Context) -> list[Panel]:
    from .smooth import lowess

    b = _base(ctx)
    panels = []
    for c in _form_coefficients(ctx):

        def draw(ax: Any, options: Any, c: dict = c) -> None:
            from surpyval.plotting import INK, PALETTE

            r = _score_residuals(ctx)[:, c["k"]]
            z = b.raw[:, c["j"]]
            keep = np.isfinite(r)
            grid = np.linspace(z.min(), z.max(), 60)
            ax.scatter(z[keep], r[keep], s=10, color=INK["muted"], alpha=0.6)
            ax.plot(grid, lowess(z[keep], r[keep], grid), color=PALETTE[0])
            ax.axhline(0.0, color=INK["axis"], lw=0.8)
            ax.set_xlabel(c["name"])
            ax.set_ylabel("score residual of {}".format(c["parameter"]))

        panels.append(
            Panel(
                "{} on {}: functional form".format(c["name"], c["parameter"]),
                draw,
            )
        )
    return panels


# ---------------------------------------------------------------------------
# Influence
# ---------------------------------------------------------------------------
def _influence_rows(ctx: Context) -> list[dict]:
    dfbeta, se = _dfbeta(ctx)
    k_dist = int(ctx.model.k_dist)
    rows: list[dict] = []
    for c in _coefficients(ctx):
        i = c["index"] - k_dist
        if ctx.model.parameter_names[c["index"]] in ctx.model.fixed:
            continue
        col = np.abs(dfbeta[:, i] / se[i])
        base = {
            "test": "largest |dfbeta| / se",
            "covariates": (c["name"],),
            "parameter": c["parameter"],
        }
        if not np.all(np.isfinite(col)):
            rows.append(
                dict(
                    base,
                    verdict="warn",
                    detail="the screen could not be computed: the observed "
                    "information is not positive definite",
                )
            )
            continue
        worst = int(np.argmax(col))
        above = np.flatnonzero(col > INFLUENCE_LIMIT)
        if above.size:
            detail = (
                "row {} of model.data moves the coefficient by {} standard "
                "errors; {} row{} above {:g}".format(
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
                "error (largest {}, row {} of model.data)".format(
                    INFLUENCE_LIMIT, _fmt(col[worst]), worst
                )
            )
        rows.append(
            dict(
                base,
                statistic=float(col[worst]),
                verdict="warn" if above.size else "ok",
                detail=detail,
                _row=worst,
                _size=float(col[worst]),
            )
        )
    return rows


def _influence_panels(ctx: Context) -> list[Panel]:
    k_dist = int(ctx.model.k_dist)
    panels = []
    for c in _coefficients(ctx):
        if ctx.model.parameter_names[c["index"]] in ctx.model.fixed:
            continue

        def draw(ax: Any, options: Any, i: int = c["index"] - k_dist) -> None:
            from surpyval.plotting import INK, PALETTE

            dfbeta, se = _dfbeta(ctx)
            scaled = dfbeta[:, i] / se[i]
            rows = np.arange(scaled.size)
            ax.vlines(rows, 0, scaled, color=PALETTE[0], lw=0.8)
            for level in (-INFLUENCE_LIMIT, INFLUENCE_LIMIT):
                ax.axhline(level, color=INK["secondary"], ls="--", lw=0.8)
            ax.set_xlabel("row")
            ax.set_ylabel("dfbeta / se")

        panels.append(
            Panel(
                "{} on {}: influence".format(c["name"], c["parameter"]), draw
            )
        )
    return panels


# ---------------------------------------------------------------------------
# Refusals and notes
# ---------------------------------------------------------------------------
def _validate(model: Any) -> None:
    if (
        getattr(model, "_restored", False)
        or getattr(model, "data", None) is None
    ):
        raise ValueError(
            "diagnose() needs the data the model was fitted to, which a "
            "model restored with from_dict / from_json does not carry. Call "
            "it on the fitted model, or refit."
        )
    from surpyval.univariate.regression._penalised import refuse_if_penalised

    # A penalised fit has no p-values; the message is the one its
    # covariance and bounds give.
    refuse_if_penalised(model)
    if getattr(model, "weighted", False):
        from surpyval.utils.weights import LR_WHY, refuse_weighted

        raise refuse_weighted(
            "diagnose()",
            LR_WHY + ", and a weighted fit's AIC cannot be compared with "
            "another model's",
            "Use diagnose() on the fit without w to check the distribution "
            "and the covariates' form, and summary() (with the robust "
            "covariance of the weighted fit) for the weighted estimates.",
        )
    if model.maximum == "no finite maximum":
        raise ValueError(
            "diagnose() needs a fit with a finite maximum, and this one has "
            "none (model.maximum is 'no finite maximum', as the fit warned: "
            "its likelihood keeps rising as a parameter runs off, most "
            "often because a covariate, or a combination of them, "
            "separates the failures from the survivors). Drop or merge the "
            "separating covariate, or take it off a parameter "
            "(covariates=), and refit."
        )
    if model.maximum != "verified":
        raise ValueError(
            "diagnose() needs a fit at a verified maximum of its likelihood "
            "(its tests compare it with larger models), and this one's "
            "model.maximum is {!r}, as the fit warned. Refit from another "
            "start (init=), or with center=True for covariates far from "
            "0.".format(model.maximum)
        )
    aliased = model.aliased
    if np.size(aliased):
        raise ValueError(
            "diagnose() is not available for a model with aliased "
            "coefficients (column(s) {} of Z, whose coefficients are "
            "nan). Refit without those columns, or without them on the "
            "parameter that cannot use them (covariates=).".format(
                np.asarray(aliased).tolist()
            )
        )


def _notes(ctx: Context) -> list[str]:
    name = ctx.model.distribution.name
    others = [c for c in CANDIDATES if c != name]
    return [
        "Distribution: the Cox-Snell residuals H(x | z), each unit's fitted "
        "cumulative hazard at its own parameters, are drawn on exponential "
        "paper (d.plot('distribution')) and, as for a PH model, have no "
        "useful test of their own. The AIC against each other built-in "
        "distribution ({}) fitted as a distributional regression to the "
        "same data warns when one is more than {:g} lower; each takes the "
        "columns of the model's life parameter on its own life parameter "
        "and the columns of the model's other parameters on each of its "
        "others.".format(
            ", ".join(c + "Distributional" for c in others), AIC_MARGIN
        ),
        "Functional form: the likelihood-ratio test of adding a covariate's "
        "square to the predictor of each parameter that holds it; "
        "d.plot('functional_form') draws each unit's score residual for "
        "that parameter (d log L_i / d eta_i) against the covariate.",
        "Influence is the one-step change I^-1 U_i of each coefficient when "
        "a unit is dropped, not a refit without it.",
        "Not checked: whether a covariate left off a parameter belongs on "
        "it. Refit with it on that parameter (covariates=) and test it with "
        "likelihood_ratio_test(parameter=...).",
    ]


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------
def _on(name: str | None) -> Any:
    """Whether the model's distribution is the prebuilt ``name`` (``None``:
    one with no prebuilt fitter)."""

    def test(ctx: Context) -> bool:
        own = ctx.model.distribution.name
        return own not in CANDIDATES if name is None else own == name

    return test


def _better_rule(name: str) -> AdviceRule:
    def message(row: dict, ctx: Context) -> str:
        return (
            "Another distribution fits markedly better: {label}'s AIC is "
            "{delta} lower than this model's (see d.plot('distribution')). "
            "Refit with {label}.fit (covariates= puts the columns on its "
            "parameters) and diagnose it.".format(
                label=row["_label"], delta=_fmt(row["_delta"])
            )
        )

    return AdviceRule(
        "distribution",
        message,
        (name + "Distributional.fit(covariates=)",),
        when=lambda row, ctx: row.get("_best") == name,
        on=("warn",),
    )


def _form_rule(name: str | None) -> AdviceRule:
    on_dist = _on(name)
    feature = (
        ("Distributional",)
        if name is None
        else (name + "Distributional.fit_from_df(formulas=)",)
    )

    def message(row: dict, ctx: Context) -> str:
        cov, param = row["covariates"][0], row["parameter"]
        return (
            "{cov} does not act linearly on {param} (adding its square to "
            "{param}'s predictor improves the fit; see "
            "d.plot('functional_form')). Transform it on that parameter, "
            "for example with a log in its formula, {own}.fit_from_df(..., "
            'formulas={{"{param}": "np.log({cov}) + ..."}}), or bin it '
            "(with pandas.cut, say) and fit the bins as a categorical "
            'covariate, "C({cov}_band) + ...".'.format(
                cov=cov, param=param, own=_own(ctx)
            )
        )

    return AdviceRule(
        "functional_form",
        message,
        feature,
        when=lambda row, ctx: on_dist(ctx),
    )


def _influence_rule(name: str | None) -> AdviceRule:
    on_dist = _on(name)
    feature = (
        ("Distributional",) if name is None else (name + "Distributional.fit",)
    )

    def message(row: dict, ctx: Context) -> str:
        return (
            "Row {row} of model.data (the fitted rows, sorted) has a large "
            "influence on the coefficient of {cov} on {param}: dropping it "
            "would move the coefficient by about {size} standard errors. "
            "Check that row, and refit ({own}.fit) without it to see how "
            "much the conclusions depend on it.".format(
                row=row["_row"],
                cov=row["covariates"][0],
                param=row["parameter"],
                size=_fmt(row["_size"]),
                own=_own(ctx),
            )
        )

    return AdviceRule(
        "influence",
        message,
        feature,
        when=lambda row, ctx: on_dist(ctx) and "_row" in row,
        on=("warn",),
    )


def _applies(model: Any) -> bool:
    from surpyval.univariate.regression.distributional.model import (
        DistributionalRegressionModel,
    )

    return isinstance(model, DistributionalRegressionModel)


#: The distributional regression family.
DISTRIBUTIONAL_FAMILY = register(
    Family(
        name="Distributional regression",
        issue=867,
        applies=_applies,
        validate=_validate,
        covariates=_column_names,
        checks=(
            Check(
                "distribution",
                "The distribution fits: the Cox-Snell residuals follow the "
                "unit exponential, and no other distribution fits markedly "
                "better.",
                rows=_distribution_rows,
                panels=_distribution_panels,
            ),
            Check(
                "functional_form",
                "Each covariate acts linearly on the link scale of each "
                "parameter whose predictor holds it.",
                rows=_form_rows,
                panels=_form_panels,
            ),
            Check(
                "influence",
                "No single unit decides a coefficient.",
                rows=_influence_rows,
                panels=_influence_panels,
            ),
        ),
        advice=tuple(_better_rule(name) for name in CANDIDATES)
        + tuple(_form_rule(name) for name in list(CANDIDATES) + [None])
        + tuple(_influence_rule(name) for name in list(CANDIDATES) + [None]),
        notes=_notes,
        columns=("parameter",),
    )
)
