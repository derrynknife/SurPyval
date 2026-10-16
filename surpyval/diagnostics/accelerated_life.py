"""``diagnose()`` for accelerated life (life-stress) models (#871).

An accelerated life model, ``AcceleratedLife(dist, life_model).fit(x, Z)``,
makes two assumptions about how failure times change with stress: the
distribution keeps one shape at every stress (only its life moves), and
its life follows the life model, :math:`L(Z)`, from one stress to the
next. A test at a few stress levels can check both, because it fits the
distribution at each level too; neither can be checked at the use
stress, where the model is used, so the report also screens how far the
test supports the extrapolation there. The checks and advice are listed
in :data:`ACCELERATED_LIFE` at the bottom.

A *stress level* is a distinct row of ``Z``. A level enters the two tests
when it has at least :data:`MIN_FAILURES` failures (rows that are not
right censored, with their counts) and, for the shape test, when the
distribution fitted to it alone reaches a verified maximum; the others
are left out, and a note names them. With a continuous stress (each unit
at its own stress) no level has two failures, so neither test can be
computed: their rows warn and say so, and the comparison of life models,
which needs no levels, still runs.

The checks
----------
``common_shape``
    The likelihood-ratio test of each level fitted on its own (its own
    shape and life) against one shape shared by every level, each level
    keeping its own life: :math:`2 (\\sum_i \\ell_i - \\ell_0)` on
    :math:`(k - 1) q` degrees of freedom for :math:`k` levels and
    :math:`q` free shape parameters (``beta`` of a Weibull, ``sigma`` of a
    LogNormal; a parameter the fit held fixed stays fixed). It does not
    involve the life model, so it tests the shape alone. Drawn as a
    probability plot of every level, each with its own fit (solid, from
    the distribution's ``plot``) and the model's (dashed): the solid lines
    are parallel when the shape is common. A distribution with no free
    shape parameter (the Exponential) has no row, and a note says so.
``life_stress``
    The likelihood-ratio test of the life model against a separate life
    at every level (the common-shape model above): :math:`2 (\\ell_0 -
    \\ell)` on :math:`k - m` degrees of freedom, :math:`m` the life
    model's free parameters. It is the lack-of-fit test of the
    relationship; with no more levels than parameters (two levels and a
    two-parameter model) the life model passes through every level's
    life and there is nothing to test: the row warns and says so. Drawn
    as each level's own life against stress with the fitted life model's
    curve (and the best alternative's, dashed).
``life_model_comparison``
    A screen, not a test: every built-in life model for the model's
    number of stresses (:data:`CANDIDATES`), refitted with the same
    distribution to the same data, with its AIC. A candidate's row warns
    when its AIC is lower than the fitted model's by more than
    :data:`DELTA_AIC` (2, the usual threshold of a substantial
    difference); every other row is ok. ``InversePower`` and
    ``InverseExponential`` are reparametrisations of ``Power`` and
    ``Exponential`` (the same likelihood), so only one of each pair is
    compared.
``extrapolation``
    A screen, not a test: for each stress column, the number of distinct
    levels tested. It warns with two or fewer: a two-parameter life model
    then passes through each level's life exactly, so the data cannot
    check its form, and any extrapolation rests on the form alone. With
    ``diagnose(use_stress=...)``, also the distance of the use stress
    from the tested range, as a multiple of the range's width, on the
    scale the life model's log-life is linear in (``log s`` for a power
    term, ``1 / s`` for an exponential one, ``s`` for ``Linear`` and
    ``GeneralLogLinear``): it warns beyond :data:`EXTRAPOLATION_LIMIT`
    (1, an extrapolation longer than the tested range is wide). The
    comparison rows then also give each life model's life at the use
    stress.

The option
----------
``use_stress``, the stress the model is to predict at: a number for a
one-stress model, one per stress column otherwise (in ``Z``'s order). The
model has no notion of it, so the report screens the extrapolation only
when it is given.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable

import numpy as np
from scipy.stats import chi2

from .registry import AdviceRule, Check, Context, Family, Panel, register

#: The fewest failures a stress level needs to enter the shape and
#: lack-of-fit tests (a level with one failure has no shape to compare).
MIN_FAILURES = 2
#: A candidate life model whose AIC is lower than the fitted model's by
#: more than this warns: a substantial difference (Burnham and Anderson).
DELTA_AIC = 2.0
#: The fewest distinct levels of a stress column that do not warn: with
#: this many or fewer, a two-parameter life model fits every level.
FEW_LEVELS = 2
#: The use stress warns when it lies outside the tested range by more
#: than this many times the range's width, on the life model's scale of
#: the stress (``log s`` for a power law, ``1 / s`` for Arrhenius).
EXTRAPOLATION_LIMIT = 1.0

#: The built-in life models the comparison refits, by number of stress
#: columns (names as ``surpyval.life_models`` spells them).
CANDIDATES: dict[int, tuple[str, ...]] = {
    1: (
        "Power",
        "Exponential",
        "Eyring",
        "InverseEyring",
        "Linear",
        "GeneralLogLinear",
    ),
    2: (
        "DualPower",
        "DualExponential",
        "PowerExponential",
        "GeneralLogLinear",
    ),
}
#: Life models that are the same model as a candidate under another
#: parametrisation (``1 / (a V^n)`` is ``a' V^-n``): the same likelihood.
EQUIVALENT = {"InversePower": "Power", "InverseExponential": "Exponential"}
#: The distributions an accelerated life model takes that have a shape
#: parameter (every one but the Exponential), for the advice.
SHAPED = ("Weibull", "Normal", "Gumbel", "Logistic", "LogNormal", "Gamma")


# ---------------------------------------------------------------------------
# What the checks share
# ---------------------------------------------------------------------------
def _fmt(value: float) -> str:
    return "{:.3g}".format(value)


def _level_text(level: np.ndarray) -> str:
    level = np.atleast_1d(level)
    if level.size == 1:
        return _fmt(float(level[0]))
    return "(" + ", ".join(_fmt(float(v)) for v in level) + ")"


def _quiet(compute: Any) -> Any:
    """``compute()``, its warnings silenced, or ``None`` if it raises or
    returns a fit with no finite maximum: a refit made for a check, whose
    warnings would be about the check's model, not the user's."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            out = compute()
        except Exception:  # noqa: BLE001 -- any failure skips the refit
            return None
    if getattr(out, "maximum", None) == "no finite maximum":
        return None
    try:
        if not math.isfinite(float(out.neg_ll())):
            return None
    except Exception:  # noqa: BLE001
        return None
    return out


def _held(model: Any) -> tuple[dict, dict]:
    """The distribution and life-model parameters the fit held fixed, as
    the user fixed them (not the life parameter's placeholder, nor an
    aliased stress effect, which a refit finds again)."""
    fixed = dict(getattr(model, "fixed", {}) or {})
    aliased = set(getattr(model.fixed, "aliased", ()) or ())
    life = model.model.life_parameter
    dist = {
        k: v
        for k, v in fixed.items()
        if k in model.distribution.param_map and k != life
    }
    phi = {
        k: v
        for k, v in fixed.items()
        if k in model.reg_model.phi_param_map and k not in aliased
    }
    return dist, phi


def _shape_names(model: Any) -> list[str]:
    """The distribution's free parameters other than its life."""
    dist_fixed, _ = _held(model)
    return [
        k
        for k in model.distribution.param_map
        if k != model.model.life_parameter and k not in dist_fixed
    ]


def _free_phi(model: Any) -> int:
    """The number of life-model parameters the fit estimated."""
    phi = np.asarray(model.params, dtype=float)[model.k_dist :]
    _, held = _held(model)
    return int(np.isfinite(phi).sum()) - len(held)


def _levels(ctx: Context) -> dict:
    """The stress levels: ``levels`` (one row each), ``index`` (each
    data row's level), ``failures`` per level, and ``tested``, the
    levels with at least :data:`MIN_FAILURES` failures."""

    def compute() -> dict:
        data = ctx.model.data
        Z = np.asarray(data.Z, dtype=float)
        levels, index = np.unique(Z, axis=0, return_inverse=True)
        index = np.asarray(index).ravel()
        failed = (np.asarray(data.c) != 1) * np.asarray(data.n, dtype=float)
        failures = np.bincount(index, weights=failed, minlength=len(levels))
        tested = np.flatnonzero(failures >= MIN_FAILURES)
        return {
            "levels": levels,
            "index": index,
            "failures": failures,
            "tested": tested,
        }

    return ctx.cached("levels", compute)


def _rows_of(ctx: Context, levels: np.ndarray) -> np.ndarray:
    return np.isin(_levels(ctx)["index"], levels)


def _separate(ctx: Context) -> dict:
    """Each tested level's distribution fitted alone: ``{level: model}``,
    for the levels whose fit reached a verified maximum."""

    def compute() -> dict:
        model = ctx.model
        dist_fixed, _ = _held(model)
        out = {}
        for i in _levels(ctx)["tested"]:
            sub = model.data[_levels(ctx)["index"] == i]
            fit = _quiet(
                lambda sub=sub: model.distribution.fit_from_surpyval_data(
                    sub, fixed=dist_fixed or None
                )
            )
            if fit is not None and fit.maximum == "verified":
                out[int(i)] = fit
        return out

    return ctx.cached("separate", compute)


def _common(ctx: Context, levels: np.ndarray) -> Any:
    """The common-shape, separate-life model on the rows at ``levels``:
    the distribution with its life a free value at each level (an
    accelerated life model of ``GeneralLogLinear`` on one indicator
    column per level but the first), its shape shared."""
    key = "common_" + ",".join(str(int(i)) for i in levels)

    def compute() -> Any:
        from surpyval import AcceleratedLife, life_models

        model = ctx.model
        rows = _rows_of(ctx, levels)
        sub = model.data[rows]
        index = _levels(ctx)["index"][rows]
        dummies = (index[:, None] == np.asarray(levels)[None, 1:]).astype(
            float
        )
        sub.add_covariates(dummies)
        dist_fixed, _ = _held(model)
        fitter = AcceleratedLife(
            model.distribution, life_models.GeneralLogLinear
        )
        fit = _quiet(lambda: fitter.fit(sub, fixed=dist_fixed or None))
        if fit is None:
            return None
        lives = fit.phi(
            np.vstack([np.zeros(len(levels) - 1), np.eye(len(levels) - 1)])
        )
        return {"model": fit, "lives": np.asarray(lives, dtype=float)}

    return ctx.cached(key, compute)


def _refit(
    ctx: Context, life_model: Any, rows: "np.ndarray | None" = None
) -> Any:
    """The model's distribution with ``life_model`` refitted to the
    model's data (or the rows ``rows`` of it), the distribution's fixed
    parameters held as the model held them; ``None`` if it fails."""
    from surpyval import AcceleratedLife

    model = ctx.model
    dist_fixed, phi_fixed = _held(model)
    fixed = dict(dist_fixed)
    if life_model is model.reg_model:
        fixed.update(phi_fixed)
    data = model.data if rows is None else model.data[rows]
    fitter = AcceleratedLife(model.distribution, life_model)
    return _quiet(lambda: fitter.fit(data, fixed=fixed or None))


def _failure_note(ctx: Context, what: str, kept: np.ndarray) -> None:
    info = _levels(ctx)
    left = [i for i in range(len(info["levels"])) if i not in set(kept)]
    if not left or len(kept) < 2:
        return
    ctx.notes.append(
        "The {} leaves out {} stress level{} ({}) with fewer than {} "
        "failures{}.".format(
            what,
            len(left),
            "" if len(left) == 1 else "s",
            ", ".join(_level_text(info["levels"][i]) for i in left[:8])
            + (" ..." if len(left) > 8 else ""),
            MIN_FAILURES,
            " or no fit of its own" if what == "common-shape test" else "",
        )
    )


def _too_few_levels(ctx: Context, kept: int) -> str:
    info = _levels(ctx)
    n_levels = len(info["levels"])
    n_rows = len(info["index"])
    if n_levels > n_rows / 2:
        return (
            "the stress is continuous ({} distinct levels in {} rows): the "
            "test needs repeated stress levels, at least two with {} or "
            "more failures; group the stresses into levels to test "
            "it".format(n_levels, n_rows, MIN_FAILURES)
        )
    return (
        "{} stress level{} with {} or more failures{}; the test needs "
        "two".format(
            kept,
            "" if kept == 1 else "s",
            MIN_FAILURES,
            "" if kept else " (none)",
        )
    )


def _use_stress(ctx: Context) -> "np.ndarray | None":
    """The ``use_stress`` option as one row of ``Z``, or ``None``; a
    value the life model cannot take is refused."""

    def compute() -> "np.ndarray | None":
        given = ctx.options.get("use_stress")
        if given is None:
            return None
        model = ctx.model
        p = int(np.asarray(model.data.Z).shape[1])
        try:
            use = np.asarray(given, dtype=float).ravel()
        except (TypeError, ValueError):
            use = np.array([np.nan])
        if use.size != p or not np.all(np.isfinite(use)):
            raise ValueError(
                "'use_stress' must be the stress the model is to predict "
                "at: {} finite number{} (one per stress column of Z, {}); "
                "got {!r}.".format(
                    p, "" if p == 1 else "s", ", ".join(ctx.names), given
                )
            )
        life_model = model.reg_model
        for col in sorted(
            {
                *getattr(life_model, "positive_stress_columns", ()),
                *getattr(life_model, "kelvin_stress_columns", ()),
            }
        ):
            if use[col] <= 0:
                raise ValueError(
                    "'use_stress' must be positive in column {} ({}): the "
                    "{} life model takes its power or logarithm, or reads "
                    "it in kelvin; got {:g}.".format(
                        col, ctx.names[col], life_model.name, use[col]
                    )
                )
        return use.reshape(1, -1)

    return ctx.cached("use_stress", compute)


def _stress_scale(ctx: Context, Z: np.ndarray) -> tuple[np.ndarray, list]:
    """The stresses ``Z`` on the scale the life model's log-life is
    linear in, column by column, and each column's scale's name."""
    life_model = ctx.model.reg_model
    found = None
    try:
        found = life_model._stress_terms(Z)
    except Exception:  # noqa: BLE001 -- a custom model's own terms
        found = None
    if found is not None and np.shape(found[0]) == Z.shape:
        terms = np.asarray(found[0], dtype=float)
        names = []
        for j in range(Z.shape[1]):
            if np.allclose(terms[:, j], np.log(Z[:, j])):
                names.append("log")
            elif np.allclose(terms[:, j], 1.0 / Z[:, j]):
                names.append("1 / stress")
            else:
                names.append("linear")
        return terms, names
    kelvin = getattr(life_model, "kelvin_stress_columns", ())
    positive = getattr(life_model, "positive_stress_columns", ())
    out = np.array(Z, dtype=float)
    names = []
    for j in range(Z.shape[1]):
        if j in kelvin:
            out[:, j] = 1.0 / Z[:, j]
            names.append("1 / stress")
        elif j in positive:
            out[:, j] = np.log(Z[:, j])
            names.append("log")
        else:
            names.append("linear")
    return out, names


def _use_rows(ctx: Context) -> list[dict]:
    use = _use_stress(ctx)
    if use is None:
        return []
    Z = np.asarray(ctx.model.data.Z, dtype=float)
    scaled, scales = _stress_scale(ctx, np.vstack([Z, use]))
    rows = []
    for j, name in ctx.checked:
        tested, u = scaled[:-1, j], scaled[-1, j]
        lo, hi = float(tested.min()), float(tested.max())
        outside = max(lo - u, u - hi, 0.0)
        width = hi - lo
        if outside == 0:
            ratio = 0.0
        else:
            ratio = outside / width if width > 0 else float("inf")
        far = ratio > EXTRAPOLATION_LIMIT
        tested_range = "{} to {}".format(
            _fmt(Z[:, j].min()), _fmt(Z[:, j].max())
        )
        if outside == 0:
            detail = "{} is within the tested range, {}".format(
                _fmt(use[0, j]), tested_range
            )
        else:
            detail = (
                "{} is outside the tested range, {}, by {} times its "
                "width ({} scale of the {} model)".format(
                    _fmt(use[0, j]),
                    tested_range,
                    _fmt(ratio),
                    scales[j],
                    ctx.model.reg_model.name,
                )
            )
        rows.append(
            {
                "test": "use stress vs tested range",
                "covariates": (name,),
                "statistic": ratio,
                "verdict": "warn" if far else "ok",
                "detail": detail,
                "_kind": "use",
                "_use": float(use[0, j]),
                "_range": tested_range,
            }
        )
    return rows


def _life_at_use(fit: Any, use: "np.ndarray | None") -> str:
    if use is None:
        return ""
    with np.errstate(all="ignore"):
        life = float(np.asarray(fit.phi(use), dtype=float).ravel()[0])
    return "; life at the use stress {}".format(_fmt(life))


# ---------------------------------------------------------------------------
# Common shape
# ---------------------------------------------------------------------------
def _shape_rows(ctx: Context) -> list[dict]:
    model = ctx.model
    shapes = _shape_names(model)
    if not shapes:
        ctx.notes.append(
            "Common shape is not tested: the {} has no free parameter "
            "besides its life{}.".format(
                model.distribution.name,
                (
                    ""
                    if model.distribution.name == "Exponential"
                    else " (its shape was held fixed)"
                ),
            )
        )
        return []
    separate = _separate(ctx)
    kept = np.array(sorted(separate), dtype=int)
    row = {
        "test": "likelihood ratio, separate vs common shape",
        "covariates": (),
        "_shapes": shapes,
    }
    if len(kept) < 2:
        return [
            dict(
                row,
                p=float("nan"),
                detail="not computed: " + _too_few_levels(ctx, len(kept)),
            )
        ]
    _failure_note(ctx, "common-shape test", kept)
    common = _common(ctx, kept)
    if common is None:
        return [
            dict(
                row,
                p=float("nan"),
                detail="not computed: the common-shape model with a life "
                "at each level could not be fitted",
            )
        ]
    ll_separate = -sum(float(separate[i].neg_ll()) for i in kept)
    ll_common = -float(common["model"].neg_ll())
    stat = max(0.0, 2.0 * (ll_separate - ll_common))
    df = (len(kept) - 1) * len(shapes)
    p = float(chi2.sf(stat, df))
    first = shapes[0]
    values = [
        float(separate[i].params[model.distribution.param_map[first]])
        for i in kept
    ]
    lo, hi = int(np.argmin(values)), int(np.argmax(values))
    levels = _levels(ctx)["levels"]
    spread = "{} from {} at {} to {} at {}".format(
        first,
        _fmt(values[lo]),
        _level_text(levels[kept[lo]]),
        _fmt(values[hi]),
        _level_text(levels[kept[hi]]),
    )
    if p < 2 * ctx.alpha:
        detail = "the shape differs between the {} levels: {}".format(
            len(kept), spread
        )
    else:
        detail = (
            "no evidence that the shape differs between the {} levels "
            "({})".format(len(kept), spread)
        )
    return [
        dict(
            row,
            statistic=stat,
            df=df,
            p=p,
            detail=detail,
            _spread=spread,
        )
    ]


def _shape_panels(ctx: Context) -> list[Panel]:
    model = ctx.model
    separate = _separate(ctx)
    levels = _levels(ctx)["levels"]
    if not separate:
        return []

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import PALETTE

        for k, (i, fit) in enumerate(sorted(separate.items())):
            color = PALETTE[k % len(PALETTE)]
            fit.plot(
                ax=ax,
                plot_bounds=False,
                color=color,
                label=_level_text(levels[i]),
            )
            xs = np.asarray(fit.data["x"], dtype=float)
            xs = xs[np.isfinite(xs) & (xs > 0)]
            if xs.size:
                grid = np.geomspace(xs.min(), xs.max(), 60)
                ax.plot(
                    grid,
                    model.ff(grid, np.atleast_2d(levels[i])).ravel(),
                    color=color,
                    ls="--",
                    lw=1.0,
                )
        ax.legend(title="stress (dashed: model)", fontsize="x-small")

    title = "common shape"
    p = [r for r in _shape_rows_cached(ctx) if np.isfinite(r.get("p", np.nan))]
    if p:
        title += ": p = {}".format(_fmt(p[0]["p"]))
    return [Panel(title, draw)]


def _shape_rows_cached(ctx: Context) -> list[dict]:
    return ctx.cached("shape_rows", lambda: _shape_rows(ctx))


# ---------------------------------------------------------------------------
# The life-stress relationship
# ---------------------------------------------------------------------------
def _lof(ctx: Context) -> dict:
    """The lack-of-fit test of the life model: the rows' fields."""

    def compute() -> dict:
        model = ctx.model
        info = _levels(ctx)
        kept = info["tested"]
        row: dict = {
            "test": "likelihood ratio, life model vs a life at each level",
            "covariates": (),
            "life_model": model.reg_model.name,
        }
        if len(kept) < 2:
            return dict(
                row,
                p=float("nan"),
                detail="not computed: " + _too_few_levels(ctx, len(kept)),
            )
        m = _free_phi(model)
        df = len(kept) - m
        if df <= 0:
            return dict(
                row,
                df=df,
                p=float("nan"),
                detail="not computed: with {} level{} and {} free "
                "parameters the {} model passes through every level's "
                "life, so the data cannot test it; test at more "
                "levels".format(
                    len(kept),
                    "" if len(kept) == 1 else "s",
                    m,
                    model.reg_model.name,
                ),
            )
        _failure_note(ctx, "life-stress test", kept)
        common = _common(ctx, kept)
        if len(kept) == len(info["levels"]):
            fitted = model
        else:
            fitted = _refit(ctx, model.reg_model, _rows_of(ctx, kept))
        if common is None or fitted is None:
            return dict(
                row,
                df=df,
                p=float("nan"),
                detail="not computed: a model it compares could not be "
                "fitted",
            )
        stat = max(
            0.0,
            2.0 * (float(fitted.neg_ll()) - float(common["model"].neg_ll())),
        )
        p = float(chi2.sf(stat, df))
        if p < 2 * ctx.alpha:
            detail = (
                "the {} model does not follow the lives of the {} levels"
            ).format(model.reg_model.name, len(kept))
        else:
            detail = (
                "no evidence against the {} model at the {} levels"
            ).format(model.reg_model.name, len(kept))
        return dict(row, statistic=stat, df=df, p=p, detail=detail)

    return ctx.cached("lof", compute)


def _comparison(ctx: Context) -> list[dict]:
    """The candidate life models refitted, with their AIC: one dict per
    candidate (the fitted model first), sorted by AIC after it."""

    def compute() -> list[dict]:
        from surpyval import life_models

        model = ctx.model
        own = model.reg_model.name
        n_stresses = int(np.asarray(model.data.Z).shape[1])
        names = [
            n
            for n in CANDIDATES.get(n_stresses, ("GeneralLogLinear",))
            if n != EQUIVALENT.get(own, own)
        ]
        out = [
            {"name": own, "model": model, "aic": float(model.aic())},
        ]
        skipped = []
        for name in names:
            fit = _refit(ctx, getattr(life_models, name))
            if fit is None:
                skipped.append(name)
                continue
            out.append({"name": name, "model": fit, "aic": float(fit.aic())})
        if skipped:
            ctx.notes.append(
                "The life-model comparison could not fit {} to these data "
                "(a stress outside the model's domain, such as a "
                "non-positive one, or no finite maximum).".format(
                    ", ".join(skipped)
                )
            )
        out[1:] = sorted(out[1:], key=lambda r: r["aic"])
        return out

    return ctx.cached("comparison", compute)


def _best_alternative(ctx: Context) -> "dict | None":
    """The candidate with the lowest AIC, if lower than the model's."""
    found = _comparison(ctx)
    if len(found) < 2 or found[1]["aic"] >= found[0]["aic"]:
        return None
    return found[1]


def _lof_failed(ctx: Context) -> bool:
    """Whether the life-stress test failed (its advice then names the
    best alternative, and the comparison's need not)."""
    p = _lof(ctx).get("p")
    return p is not None and math.isfinite(p) and p < ctx.alpha


def _lof_rows(ctx: Context) -> list[dict]:
    row = dict(_lof(ctx))
    best = _best_alternative(ctx)
    row["_best"] = None if best is None else best["name"]
    return [row]


def _comparison_rows(ctx: Context) -> list[dict]:
    found = _comparison(ctx)
    own = found[0]
    use = _use_stress(ctx)
    rows = [
        {
            "test": "AIC",
            "life_model": own["name"],
            "covariates": (),
            "statistic": own["aic"],
            "verdict": "ok",
            "detail": "the fitted life model"
            + _life_at_use(own["model"], use),
        }
    ]
    for cand in found[1:]:
        delta = cand["aic"] - own["aic"]
        better = delta < -DELTA_AIC
        if abs(delta) < 5e-3:
            detail = "the same AIC as {}'s".format(own["name"])
        elif better:
            detail = "AIC lower than {}'s by {}".format(
                own["name"], _fmt(-delta)
            )
        elif delta < 0:
            detail = (
                "AIC lower than {}'s by {}, within {:g}: no substantial "
                "difference".format(own["name"], _fmt(-delta), DELTA_AIC)
            )
        else:
            detail = "AIC higher than {}'s by {}".format(
                own["name"], _fmt(delta)
            )
        rows.append(
            {
                "test": "AIC",
                "life_model": cand["name"],
                "covariates": (),
                "statistic": cand["aic"],
                "verdict": "warn" if better else "ok",
                "detail": detail + _life_at_use(cand["model"], use),
                "_delta": -delta,
                "_best": cand is found[1],
            }
        )
    return rows


def _life_panels(ctx: Context) -> list[Panel]:
    model = ctx.model
    info = _levels(ctx)
    kept = info["tested"]
    common = _common(ctx, kept) if len(kept) >= 2 else None
    levels = info["levels"]
    best = _best_alternative(ctx)
    use = _use_stress(ctx)
    if common is None:
        return []
    lives = common["lives"]
    single = levels.shape[1] == 1

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        if single:
            s = levels[kept, 0]
            lo, hi = float(s.min()), float(s.max())
            if use is not None:
                lo, hi = min(lo, use[0, 0]), max(hi, use[0, 0])
                ax.axvline(
                    use[0, 0],
                    color=INK["secondary"],
                    ls=":",
                    lw=1.0,
                    label="use stress",
                )
            grid = np.linspace(lo, hi, 100)
            ax.scatter(
                s, lives, color=INK["muted"], zorder=3, label="each level"
            )
            ax.plot(
                grid,
                model.phi(grid.reshape(-1, 1)),
                color=PALETTE[0],
                label=model.reg_model.name,
            )
            if best is not None:
                ax.plot(
                    grid,
                    best["model"].phi(grid.reshape(-1, 1)),
                    color=PALETTE[1],
                    ls="--",
                    label=best["name"],
                )
            ax.set_yscale("log")
            ax.set_xlabel(ctx.names[0])
            ax.set_ylabel("life L(Z)")
        else:
            fitted = np.asarray(model.phi(levels[kept]), dtype=float)
            ax.scatter(lives, fitted, color=PALETTE[0], zorder=3)
            both = np.concatenate([lives, fitted])
            both = both[np.isfinite(both) & (both > 0)]
            if both.size:
                ends = [both.min(), both.max()]
                ax.plot(ends, ends, color=INK["secondary"], ls="--", lw=0.8)
            from matplotlib.ticker import NullFormatter

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("life at each level (common shape)")
            ax.set_ylabel(
                "life from the {} model".format(model.reg_model.name)
            )
            return
        ax.legend(fontsize="small")

    p = _lof(ctx).get("p", float("nan"))
    title = "life-stress ({})".format(model.reg_model.name)
    if p is not None and np.isfinite(p):
        title += ": p = {}".format(_fmt(p))
    return [Panel(title, draw)]


# ---------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------
def _extrapolation_rows(ctx: Context) -> list[dict]:
    Z = np.asarray(ctx.model.data.Z, dtype=float)
    rows = []
    for j, name in ctx.checked:
        values = np.unique(Z[:, j])
        few = values.size <= FEW_LEVELS
        if few:
            detail = (
                "{} level{} ({}): a two-parameter life model passes through "
                "each level's life, so nothing checks its form, and an "
                "extrapolation rests on the form alone".format(
                    values.size,
                    "" if values.size == 1 else "s",
                    ", ".join(_fmt(v) for v in values),
                )
            )
        else:
            detail = "{} levels, from {} to {}".format(
                values.size, _fmt(values.min()), _fmt(values.max())
            )
        rows.append(
            {
                "test": "stress levels tested",
                "covariates": (name,),
                "statistic": float(values.size),
                "verdict": "warn" if few else "ok",
                "detail": detail,
                "_kind": "levels",
            }
        )
    return rows + _use_rows(ctx)


# ---------------------------------------------------------------------------
# Refusals and notes
# ---------------------------------------------------------------------------
def _is_accelerated_life(model: Any) -> bool:
    from surpyval.univariate.regression._kinds import ACCELERATED_LIFE
    from surpyval.univariate.regression.parametric_regression_model import (
        ParametricRegressionModel,
    )

    return (
        isinstance(model, ParametricRegressionModel)
        and getattr(model, "kind", None) == ACCELERATED_LIFE
    )


def _validate(model: Any) -> None:
    if getattr(model, "data", None) is None:
        raise ValueError(
            "diagnose() needs the data the accelerated life model was "
            "fitted to, which a model restored with from_dict / from_json "
            "does not carry. Call it on the fitted model, or refit."
        )
    if getattr(model, "maximum", None) == "no finite maximum":
        raise ValueError(
            "diagnose() needs a fit with a finite maximum, and this one "
            "has none (model.maximum is 'no finite maximum', as the fit "
            "warned: a life-model parameter runs off, often because a "
            "stress level has no failures). Refit with more failures at "
            "each level, or fix the runaway parameter with `fixed`."
        )


def _covariates(model: Any) -> list[str]:
    """The stress columns' names: the DataFrame's or formula's columns,
    ``GeneralLogLinear``'s coefficient names, or ``stress`` (one column)
    and ``stress_0``, ``stress_1``, ... ."""
    p = int(np.asarray(model.data.Z).shape[1])
    names = getattr(model, "feature_names", None)
    if names is not None and len(names) == p:
        return [str(n) for n in names]
    columns = model.reg_model.coefficient_columns()
    if len(columns) == p:
        return [n for n, _ in sorted(columns.items(), key=lambda kv: kv[1])]
    if p == 1:
        return ["stress"]
    return ["stress_{}".format(j) for j in range(p)]


def _notes(ctx: Context) -> list[str]:
    if _use_stress(ctx) is not None:
        extrapolation = (
            "Extrapolation: the use stress's distance from the tested "
            "range, in widths of the range, on the scale the life model's "
            "log-life is linear in; beyond {:g} it warns. The data cannot "
            "test the life model there: compare the life models' lives at "
            "the use stress (the life_model_comparison rows), and quote "
            "the life with its confidence bounds (cb).".format(
                EXTRAPOLATION_LIMIT
            )
        )
    else:
        extrapolation = (
            "Extrapolation to the use stress is not screened: the model "
            "does not know its use stress; give it as "
            "diagnose(use_stress=...)."
        )
    return [
        "Common shape: the likelihood-ratio test of each stress level "
        "fitted alone against one shape shared by all, each level keeping "
        "its own life; life-stress: the likelihood-ratio test of the life "
        "model against a separate life at each level (common shape). A "
        "level enters both with {} or more failures.".format(MIN_FAILURES),
        "Life-model comparison: each built-in life model for the model's "
        "number of stresses, refitted to the same data with the same "
        "distribution; InversePower and InverseExponential are the same "
        "models as Power and Exponential. A life model is also a physical "
        "claim (Arrhenius for temperature, the inverse power law for "
        "voltage): prefer a better AIC only where its physics fits.",
        extrapolation,
    ]


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------
def _shape_message(row: dict, ctx: Context) -> str:
    dist = ctx.model.distribution.name
    return (
        "The shape of the {dist} differs between stress levels ({spread}; "
        "see d.plot('common_shape')), and the model assumes one shape. A "
        "shape that changes with stress often means a different failure "
        "mode at the highest (or lowest) stresses: check the failures "
        "there, and if they differ, refit AcceleratedLife({dist}, ...) "
        "without those levels. To see the shapes, fit each level on its "
        "own with {dist}.fit; a model with one shape cannot extrapolate a "
        "shape that drifts with stress.".format(
            dist=dist, spread=row["_spread"]
        )
    )


def _lof_message(name: str) -> Any:
    def message(row: dict, ctx: Context) -> str:
        found = _comparison(ctx)
        return (
            "The {own} life model does not follow the lives at the stress "
            "levels (see d.plot('life_stress')). Of the life models "
            "compared, {name} fits best (AIC lower by {delta}; see the "
            "life_model_comparison rows): refit with "
            "AcceleratedLife({dist}, life_models.{name}) if its physics "
            "suits the stress, and diagnose that fit.".format(
                own=ctx.model.reg_model.name,
                name=name,
                delta=_fmt(found[0]["aic"] - found[1]["aic"]),
                dist=ctx.model.distribution.name,
            )
        )

    return message


def _lof_other(row: dict, ctx: Context) -> str:
    return (
        "The {own} life model does not follow the lives at the stress "
        "levels (see d.plot('life_stress')), and no other built-in life "
        "model fits better by AIC. The stress may act through another "
        "transform, or another stress that varied in the test may "
        "matter: write the relationship as a life model of your own "
        "(subclass life_models.LifeModel), or add the other stress with "
        "a two-stress model such as life_models.DualPower.".format(
            own=ctx.model.reg_model.name
        )
    )


def _better_message(row: dict, ctx: Context) -> str:
    return (
        "The {name} life model fits these data better than {own} (AIC "
        "lower by {delta}). Refit with AcceleratedLife({dist}, "
        "life_models.{name}) and compare their lives at the use stress: "
        "where two plausible life models disagree there, the "
        "extrapolation is uncertain whatever its confidence "
        "bounds.".format(
            name=row["life_model"],
            own=ctx.model.reg_model.name,
            delta=_fmt(row["_delta"]),
            dist=ctx.model.distribution.name,
        )
    )


def _levels_message(row: dict, ctx: Context) -> str:
    return (
        "{name} was tested at {k} level{s}, so the data cannot check the "
        "form of the life-stress relationship, and the life at the use "
        "stress rests on that form alone. Test at three or more levels, "
        "spread over the range the units can take without a new failure "
        "mode, with more units at the lower stresses (nearer the use "
        "stress). To compare plans, simulate each with the fitted model's "
        "random(size, Z=<the planned levels>), refit, and see how wide "
        "the bounds of the life at the use stress are (cb).".format(
            name=row["covariates"][0],
            k=int(row["statistic"]),
            s="" if int(row["statistic"]) == 1 else "s",
        )
    )


def _far_message(row: dict, ctx: Context) -> str:
    return (
        "The use stress, {name} = {use:g}, is far outside the tested range "
        "({tested}; {ratio} times its width away): the life there is the "
        "life model's extrapolation, which the data cannot check. Quote "
        "it with its confidence bounds, cb(x, Z=<the use stress>), and "
        "compare it with the other plausible life models' (the "
        "life_model_comparison rows give each one's life at the use "
        "stress): where they disagree, the bounds understate the "
        "uncertainty. If you can, test nearer the use stress; to compare "
        "plans, simulate each with the fitted model's random(size, "
        "Z=<the planned levels>) and refit.".format(
            name=row["covariates"][0],
            use=row["_use"],
            tested=row["_range"],
            ratio=_fmt(row["statistic"]),
        )
    )


def _for_distribution(dist: str) -> Callable[[Any, Context], bool]:
    def when(row: Any, ctx: Context) -> bool:
        return bool(ctx.model.distribution.name == dist)

    return when


def _best_is(name: str) -> Callable[[Any, Context], bool]:
    def when(row: Any, ctx: Context) -> bool:
        return bool(row.get("_best") == name)

    return when


def _better_is(name: str) -> Callable[[Any, Context], bool]:
    def when(row: Any, ctx: Context) -> bool:
        return (
            row.get("_best") is True
            and row["life_model"] == name
            and not _lof_failed(ctx)
        )

    return when


def _advice() -> tuple[AdviceRule, ...]:
    rules: list[AdviceRule] = [
        AdviceRule(
            "common_shape",
            _shape_message,
            ("AcceleratedLife", dist + ".fit"),
            when=_for_distribution(dist),
        )
        for dist in SHAPED
    ]
    names = sorted({n for group in CANDIDATES.values() for n in group})
    rules.extend(
        AdviceRule(
            "life_stress",
            _lof_message(name),
            ("AcceleratedLife", "life_models." + name),
            when=_best_is(name),
        )
        for name in names
    )
    rules.append(
        AdviceRule(
            "life_stress",
            _lof_other,
            ("life_models.LifeModel", "life_models.DualPower"),
            when=lambda row, ctx: row.get("_best") is None,
        )
    )
    rules.extend(
        AdviceRule(
            "life_model_comparison",
            _better_message,
            ("AcceleratedLife", "life_models." + name),
            when=_better_is(name),
            on=("warn",),
        )
        for name in names
    )
    rules.append(
        AdviceRule(
            "extrapolation",
            _levels_message,
            ("random(Z=)", "cb(Z=)"),
            when=lambda row, ctx: row.get("_kind") == "levels",
            on=("warn",),
        )
    )
    rules.append(
        AdviceRule(
            "extrapolation",
            _far_message,
            ("cb(Z=)", "random(Z=)"),
            when=lambda row, ctx: row.get("_kind") == "use",
            on=("warn",),
        )
    )
    return tuple(rules)


#: The accelerated life family: its checks, in the order of the table,
#: and its advice.
ACCELERATED_LIFE = register(
    Family(
        name="Accelerated life",
        issue=871,
        applies=_is_accelerated_life,
        validate=_validate,
        covariates=_covariates,
        columns=("life_model",),
        options=("use_stress",),
        checks=(
            Check(
                "common_shape",
                "The distribution has one shape at every stress level; "
                "only its life changes with stress.",
                rows=_shape_rows_cached,
                panels=_shape_panels,
            ),
            Check(
                "life_stress",
                "The life at each stress level follows the life model.",
                rows=_lof_rows,
                panels=_life_panels,
            ),
            Check(
                "life_model_comparison",
                "No other built-in life model fits the data substantially "
                "better.",
                rows=_comparison_rows,
            ),
            Check(
                "extrapolation",
                "The test's stress levels can support the life model's "
                "form, and the use stress (if given) is not far outside "
                "them.",
                rows=_extrapolation_rows,
            ),
        ),
        advice=_advice(),
        notes=_notes,
    )
)
