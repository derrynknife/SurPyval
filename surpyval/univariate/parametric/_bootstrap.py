"""The parametric bootstrap bounds of a fitted univariate model
(``method="bootstrap"`` on ``cb``, ``quantile_cb`` and ``param_cb``,
#645).

The Wald bounds of an offset (3-parameter) fit hold the offset at its
estimate: the offset is a threshold, whose likelihood is not regular, so
the fit estimates no standard error for it. Near the offset those bounds
are far too narrow (a 90% bound on a 3-parameter Weibull's B1 life from
30 failures covered 40% of the time, #645). The bootstrap includes the
offset's uncertainty, as it does every other parameter's, and so do the
likelihood-ratio bounds, which profile it.

The model is resampled, not the data, as for the regressions
(``univariate.regression._bootstrap``, #617): each resample draws a
failure time for every unit from the fitted model (within the unit's
truncation window), censors it as the unit was censored -- Davison &
Hinkley's conditional bootstrap, a censored unit at its own censoring
time and a failed one at a time drawn from the censoring distribution
beyond its failure -- and refits the same model to the result, offset and
fixed parameters as fitted. The bounds are the bias-corrected percentile
intervals of the refits (Efron 1987): the percentiles of the refits
moved by the share of them below the estimate. The BCa acceleration the
regressions add needs the covariance of every parameter, and an offset
fit's leaves the offset out. A limited-failure or zero-inflated model's
intervals are BCa (#286), with the jackknife's acceleration: the model
refitted without each unit in turn (a row of count ``n`` is ``n`` units),
which needs no covariance and so holds with ``lfp_p`` or ``f0`` on its
edge.

On #645's design (a 3-parameter Weibull of shape 1.5, 30 failures) the
90% bound on the B1 life covered 63-64% over two runs of 100 samples,
where the Wald bound covered 35%; on B10 82-89% (Wald 69%), on ``sf``
just above the offset 66-68% (Wald 53%). It is still short at the offset
itself when the shape is below 2: the offset's estimate is biased up
(it is at most the first failure), and no interval of the refits makes
up for that -- the percentile interval covered B1 45% and the basic
(reflected) interval 75% but ``sf`` above the offset 56%.

A fit with sampling weights ``w`` is resampled by cases instead: the
model knows nothing of the sampling design the weights describe, so
simulating from it would not reproduce the variability of the estimate.
Each resample draws the units i.i.d. with replacement, each with its own
weight, time, censoring and truncation, and refits the weighted model;
the bounds are the same bias-corrected percentile intervals of those
refits. Nothing is simulated, so left- and interval-censored data are
resampled as they are. The resamples are drawn by position alone, so a
seed gives the same resamples, and the same bounds, at any scale of the
weights.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.univariate.regression._bootstrap import (
    FAILED_SHARE,
    REFIT_ERRORS,
    ResampleDesign,
    bca_bounds,
    check_n_boot,
)
from surpyval.utils.no_maximum import quiet_maximum_warnings
from surpyval.utils.rng import as_generator
from surpyval.utils.warnings import caller_stacklevel, ignoring_warnings


@dataclass
class Jackknife:
    """The leave-one-unit-out refits of a model, for the acceleration of
    its BCa intervals: one per distinct row of the data (a row of count
    ``n`` stands for ``n`` units, each of whose removal is the same refit,
    so it is weighted ``n``)."""

    models: list
    #: Each refit's :func:`_point`, an edge read as for the bootstrap.
    values: npt.NDArray
    #: The number of units each refit stands for.
    weights: npt.NDArray


def jackknife_acceleration(
    values: npt.NDArray, weights: npt.NDArray
) -> npt.NDArray:
    """Efron's (1987) jackknife acceleration of each column of ``values``
    (one leave-one-out estimate per row, each standing for ``weights``
    units): ``sum w d^3 / (6 (sum w d^2)^1.5)`` with ``d`` the mean of the
    estimates less each. 0 (the bias-corrected percentile interval) for a
    column with no spread (every estimate on an edge) or a value that is
    not finite.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.univariate.parametric._bootstrap import (
    ...     jackknife_acceleration,
    ... )
    >>> v = np.array([[1.0], [2.0], [6.0]])
    >>> jackknife_acceleration(v, np.ones(3)).round(4)
    array([-0.0573])
    """
    values = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)[:, None]
    if values.shape[0] < 3:
        return np.zeros(values.shape[1:])
    with np.errstate(all="ignore"):
        d = np.sum(w * values, axis=0) / np.sum(w) - values
        m2 = np.sum(w * d**2, axis=0)
        a = np.sum(w * d**3, axis=0) / (6.0 * m2**1.5)
    return np.where(np.isfinite(a), a, 0.0)


def jackknife(model: Any, fits: Refits) -> Jackknife:
    """The leave-one-unit-out refits of a limited-failure or zero-inflated
    ``model`` (kept on ``fits``): the data less one unit of each distinct
    row in turn, refitted with the model's own options from its
    estimate; a refit that raises is left out."""
    if fits.jack is not None:
        return fits.jack
    data = model.data
    x = np.asarray(data["x"], dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    c = np.asarray(data["c"], dtype=int)
    n = np.asarray(data["n"], dtype=float)
    t = np.asarray(data["t"], dtype=float).reshape(-1, 2)
    truncated = bool(np.isfinite(t).any())
    fixed, init = _refit_options(model)
    models, values, weights = [], [], []
    for i in range(x.size):
        n_i = n.copy()
        n_i[i] -= 1
        keep = n_i > 0
        kwargs: dict[str, Any] = {"c": c[keep], "n": n_i[keep]}
        if truncated:
            kwargs["tl"] = t[keep, 0]
            kwargs["tr"] = t[keep, 1]
        try:
            with ignoring_warnings(), quiet_maximum_warnings():
                refit = model.dist.fit(
                    x[keep],
                    offset=model.offset,
                    lfp=model.lfp,
                    zi=model.zi,
                    fixed=fixed,
                    init=init,
                    **kwargs,
                )
        except REFIT_ERRORS:
            continue
        units = np.repeat(np.arange(keep.sum()), n_i[keep].astype(int))
        edge_kwargs = (
            {"tl": t[keep, 0][units], "tr": t[keep, 1][units]}
            if truncated
            else {}
        )
        models.append(refit)
        values.append(
            _at_edges(
                refit,
                _point(refit),
                x[keep][units],
                c[keep][units],
                edge_kwargs,
            )
        )
        weights.append(n[i])
    fits.jack = Jackknife(
        models=models,
        values=np.array(values, dtype=float).reshape(len(values), -1),
        weights=np.array(weights, dtype=float),
    )
    return fits.jack


@dataclass
class Refits:
    """The refits of one bootstrap of a model: each refit's offset and
    parameters, and how many of the ``n_boot`` refits reached no
    verified maximum or failed outright."""

    #: The model's own ``(gamma, *params)`` the refits are of.
    point: tuple
    n_boot: int
    #: The kept refits, as models (``from_params`` at their estimates).
    models: list
    #: ``(n_kept, 1 + len(params) + extras)``: each kept refit's offset (0
    #: without one), parameters, and ``lfp_p`` and ``f0`` where the model
    #: has them.
    values: npt.NDArray
    no_maximum: int
    unverified: int
    failed: int
    model_maximum: str = "verified"
    #: ``point`` with an ``lfp_p`` or ``f0`` the data put on its edge
    #: there, as the refits' ``values`` are (:func:`_at_edges`).
    estimate: tuple = ()
    #: A limited-failure or zero-inflated model's jackknife, for the BCa
    #: acceleration (:func:`jackknife`): built on first use.
    jack: Jackknife | None = None

    def warn(self, public: bool = False) -> None:
        """One warning: that the model itself has no finite maximum, or
        that more than ``FAILED_SHARE`` of the refits did not reach a
        verified maximum or failed (``public``: worded for the refits of
        ``Parametric.bootstrap`` rather than the bounds)."""
        if self.model_maximum == "no finite maximum":
            warnings.warn(
                "This model's likelihood has no finite maximum (as its fit "
                "warned), so the data simulated from it run off the same "
                "way and the bootstrap "
                + (
                    "refits gather about its meaningless estimate. Their "
                    "spread is not the uncertainty of the fit."
                    if public
                    else "bounds close onto its meaningless estimate. They "
                    "are not a confidence bound."
                ),
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
            return
        bad = self.no_maximum + self.unverified + self.failed
        if bad <= FAILED_SHARE * self.n_boot:
            return
        parts = []
        if self.no_maximum:
            parts.append(f"{self.no_maximum} with no finite maximum")
        if self.unverified:
            parts.append(f"{self.unverified} at an unverified maximum")
        if self.failed:
            parts.append(f"{self.failed} failed and left out")
        warnings.warn(
            f"{bad} of the {self.n_boot} bootstrap refits did not reach a "
            f"verified maximum of the likelihood ({', '.join(parts)}). "
            "A refit with no finite maximum is kept at the estimate it "
            + (
                "reached; with this many, the data barely determine the "
                "model and the refits' spread is a rough guide to its "
                "uncertainty."
                if public
                else "reached, so that it counts at its end of the bounds; "
                "with this many, the data barely determine the model and "
                "the bounds are rough."
            ),
            UserWarning,
            stacklevel=caller_stacklevel(),
        )


#: What a refusal of ``Parametric.bootstrap`` says to use instead: the
#: draws from the covariance, which need no refits.
REFITS_INSTEAD = (
    "To carry the fit's uncertainty forward without refits, draw "
    "parameter vectors from covariance() and rebuild a model from each "
    "with with_params."
)


def check_resamplable(model: Any, public: bool = False) -> None:
    """Refuse, with one message each, what the bootstrap cannot resample
    faithfully (``public``: worded for ``Parametric.bootstrap``)."""
    if getattr(model, "data", None) is None:
        raise ValueError(
            ("bootstrap() needs" if public else "Bootstrap bounds need")
            + " the data the model was fitted to; "
            "this model was built from parameters, or restored from a dict "
            "saved without its data (save it with to_dict(with_data=True) "
            "to keep them). "
            + (REFITS_INSTEAD if public else "Use method='wald'.")
        )
    if getattr(model.dist, "discrete", False):
        raise ValueError(
            ("bootstrap() is" if public else "Bootstrap bounds are")
            + " for continuous distributions; "
            + (
                REFITS_INSTEAD
                if public
                else "for a discrete one use method='wald' or method='lr'."
            )
        )


class _UnivariateDesign(ResampleDesign):
    """What a resample of a univariate model's data keeps: each unit's
    truncation window and censoring (a row with ``n = k`` is ``k``
    units); the censoring distribution is the regressions'
    (``ResampleDesign``).
    """

    def __init__(self, model: Any, public: bool = False) -> None:
        check_resamplable(model, public)
        data = model.data
        c = np.asarray(data["c"])
        if np.any((c != 0) & (c != 1)):
            raise ValueError(
                (
                    "bootstrap() resamples"
                    if public
                    else "Bootstrap bounds resample"
                )
                + " exact and right-censored data (truncated or not); this "
                "model was fitted to left- or interval-censored data, whose "
                "inspection times a resample would need and the data do not "
                "record. "
                + (REFITS_INSTEAD if public else "Use method='wald'.")
            )
        x = np.asarray(data["x"], dtype=float)
        if x.ndim == 2:
            x = x[:, 0]
        units = np.repeat(np.arange(len(c)), np.asarray(data["n"], dtype=int))
        self.x = x[units]
        self.failed = c[units] == 0
        t = np.asarray(data["t"], dtype=float).reshape(-1, 2)[units]
        self.tl, self.tr = t[:, 0], t[:, 1]
        self.truncated = bool(
            np.any(np.isfinite(self.tl)) or np.any(np.isfinite(self.tr))
        )
        self._censoring_distribution()


class _CaseDesign:
    """What a case resample of a weighted model's data draws from: every
    unit (a row with ``n = k`` is ``k`` units) with its time, censoring,
    truncation and weight."""

    def __init__(self, model: Any, public: bool = False) -> None:
        check_resamplable(model, public)
        data = model.data
        n = np.asarray(data["n"], dtype=int)
        units = np.repeat(np.arange(n.size), n)
        x = np.asarray(data["x"], dtype=float)
        self.x = x[units]
        self.c = np.asarray(data["c"])[units]
        t = np.asarray(data["t"], dtype=float).reshape(-1, 2)[units]
        self.tl, self.tr = t[:, 0], t[:, 1]
        self.truncated = bool(
            np.any(np.isfinite(self.tl)) or np.any(np.isfinite(self.tr))
        )
        # The weights scaled to sum to the units, as the fit uses them, so
        # that the refits do not depend on the scale of ``w``.
        w = np.asarray(data["w"], dtype=float)
        self.w = (w * (n.sum() / np.sum(n * w)))[units]

    def resample(self, rng: np.random.Generator) -> dict:
        """The arguments of one refit: the units drawn with replacement."""
        k = rng.integers(0, self.x.shape[0], size=self.x.shape[0])
        out: dict[str, Any] = {"x": self.x[k], "c": self.c[k], "w": self.w[k]}
        if self.truncated:
            out["tl"] = self.tl[k]
            out["tr"] = self.tr[k]
        return out


def _point(model: Any) -> tuple:
    """``(gamma, *params)``, then ``lfp_p`` and ``f0`` where the model has
    them (the order of ``model.param_map`` after the offset)."""
    out = [float(model.gamma), *(float(p) for p in model.params)]
    if model.lfp:
        out.append(float(model.lfp_p))
    if model.zi:
        out.append(float(model.f0))
    return tuple(out)


#: A refit's ``lfp_p`` or ``f0`` is on its closed edge (1, 0) where moving
#: it there, the other parameters held, costs less than this in deviance:
#: the fit stops a hair short of an edge (``lfp_p`` at 1 - 1e-7), and the
#: refits on the edge must count there for an interval to reach it. The
#: likelihood-ratio bounds use the same threshold for an estimate on the
#: edge.
EDGE_DEVIANCE = 1e-3


def _log_likelihood(
    m: Any, x: npt.NDArray, c: npt.NDArray, tl: Any, tr: Any
) -> float:
    """The log-likelihood of exact (``c = 0``) and right-censored data
    under ``m``, each unit in its truncation window ``(tl, tr]`` (or
    none); the density of a failure at 0 under a zero-inflated model is
    its mass there, as the fit takes it."""
    with np.errstate(all="ignore"):
        ll = np.where(c == 0, np.log(m.df(x)), np.log(m.sf(x)))
        if tl is not None:
            lo = np.where(np.isfinite(tl), m.ff(np.nan_to_num(tl)), 0.0)
            hi = np.where(np.isfinite(tr), m.ff(np.nan_to_num(tr)), 1.0)
            ll = ll - np.log(hi - lo)
    return float(np.sum(ll))


def _at_edges(
    refit: Any, point: tuple, x: npt.NDArray, c: npt.NDArray, kwargs: dict
) -> tuple:
    """``point`` (a refit's :func:`_point`) with its ``lfp_p`` at 1 and
    its ``f0`` at 0 where the data put them there (``EDGE_DEVIANCE``)."""
    out = list(point)
    k = 1 + len(refit.params)
    edges = []
    if refit.lfp:
        edges.append((k, "lfp_p", 1.0))
        k += 1
    if refit.zi:
        edges.append((k, "f0", 0.0))
    if not edges:
        return point
    tl, tr = kwargs.get("tl"), kwargs.get("tr")
    ll_hat = _log_likelihood(refit, x, c, tl, tr)
    for idx, name, edge in edges:
        if out[idx] == edge:
            continue
        extras = dict(refit.extras)
        extras[name] = edge
        try:
            at_edge = refit.dist.from_params(list(refit.params), **extras)
        except ValueError:
            continue
        ll_edge = _log_likelihood(at_edge, x, c, tl, tr)
        if 2.0 * (ll_hat - ll_edge) < EDGE_DEVIANCE:
            out[idx] = edge
    return tuple(out)


def _point_index(model: Any, name: str) -> int:
    """The position of the parameter ``name`` in :func:`_point`."""
    if name == "gamma":
        return 0
    if name in model.dist.param_map:
        return 1 + model.dist.param_map[name]
    k = 1 + len(model.params)
    if name == "lfp_p":
        return k
    return k + int(bool(model.lfp))


def refits(
    model: Any, n_boot: Any, random_state: Any, public: bool = False
) -> Refits:
    """The bootstrap refits of ``model``: drawn, or those kept on the
    model for this ``n_boot`` and integer ``random_state`` while its
    parameters are as they are (``random_state=None`` or a ``Generator``
    draws afresh each time, from that stream). ``public``: for
    ``Parametric.bootstrap``, whose refusals say what to use instead of
    refits rather than instead of the bounds; the refits are the same."""
    n_boot = check_n_boot(n_boot)
    weighted = getattr(model, "weighted", False)
    if weighted and (model.lfp or model.zi):
        from surpyval.utils.weights import USE_WALD, refuse_weighted

        # The BCa interval of these models needs the jackknife and the
        # edge checks of the weighted units, which are not built; nor is
        # their case bootstrap, so bootstrap() refuses with the bounds.
        raise refuse_weighted(
            (
                "bootstrap() of a limited-failure or zero-inflated model"
                if public
                else "A bootstrap bound of a limited-failure or "
                "zero-inflated model"
            ),
            "its case bootstrap and the interval (BCa) built on it need a "
            "jackknife of the weighted units, which is not built",
            (
                "Draw parameter vectors from covariance() (the robust "
                "covariance of the weighted fit) and rebuild a model from "
                "each with with_params."
                if public
                else USE_WALD
            ),
        )
    design: Any = (
        _CaseDesign(model, public)
        if weighted
        else _UnivariateDesign(model, public)
    )
    point = _point(model)
    key = None
    if isinstance(random_state, (int, np.integer)) and not isinstance(
        random_state, bool
    ):
        key = (n_boot, int(random_state))
        kept = getattr(model, "_bootstrap_refits", None)
        if kept is not None and key in kept and kept[key].point == point:
            return kept[key]
    out = _draw(model, design, point, n_boot, as_generator(random_state))
    if key is not None:
        if getattr(model, "_bootstrap_refits", None) is None:
            model._bootstrap_refits = {}
        model._bootstrap_refits[key] = out
    return out


def _fixed(model: Any) -> dict:
    """The parameters the user fixed at fit time, by name, at their
    values."""
    names = {i: name for name, i in model.param_map.items()}
    point = _point(model)
    return {
        names[i]: point[_point_index(model, names[i])]
        for i in sorted(model._user_fixed_idx())
    }


def _refit_options(model: Any) -> tuple:
    """``(fixed, init)`` of a refit of ``model``: its user-fixed
    parameters, and a start at its estimate in the fit's own order (the
    offset, the parameters, then ``lfp_p`` and ``f0``, each of those kept
    inside (0, 1) as the fit's default start is: on the edge the
    optimiser's transform of it is infinite)."""
    fixed = _fixed(model) or None
    extras = list(_point(model)[1 + len(model.params) :])
    init = np.array(
        ([model.gamma] if model.offset else [])
        + list(model.params)
        + [float(np.clip(v, 1e-3, 0.999)) for v in extras],
        dtype=float,
    )
    if fixed:
        free = [
            i
            for name, i in sorted(
                model.param_map.items(), key=lambda kv: kv[1]
            )
            if name not in fixed
        ]
        init = init[free]
    return fixed, init


def _draw(
    model: Any,
    design: Any,
    point: tuple,
    n_boot: int,
    rng: np.random.Generator,
) -> Refits:
    """``n_boot`` resamples simulated from ``model`` with ``design`` (or,
    for a weighted fit, drawn from its units) and refitted (see the
    module docstring)."""
    cases = isinstance(design, _CaseDesign)
    with np.errstate(all="ignore"):
        F_lo = np.where(
            np.isfinite(design.tl), model.ff(np.nan_to_num(design.tl)), 0.0
        )
        F_hi = np.where(
            np.isfinite(design.tr), model.ff(np.nan_to_num(design.tr)), 1.0
        )
    fixed, init = _refit_options(model)
    # A unit a limited-failure model draws as never failing (``inf``) is
    # followed no longer than any unit was: where the censoring
    # distribution leaves it uncensored (past the last censoring time),
    # it is censored at the last time in the data.
    end = float(np.max(design.x)) if design.x.size else np.inf
    models, values = [], []
    counts = {"no finite maximum": 0, "unverified": 0, "failed": 0}
    kwargs: dict[str, Any]
    for _ in range(n_boot):
        if cases:
            kwargs = design.resample(rng)
            x = kwargs.pop("x")
        else:
            u = rng.uniform(size=design.x.size)
            with np.errstate(all="ignore"):
                T = np.asarray(model.qf(F_lo + u * (F_hi - F_lo)), dtype=float)
            C = design.censoring(rng)
            C = np.where(np.isinf(T) & np.isinf(C), end, C)
            censored = T > C
            x = np.where(censored, C, T)
            kwargs = {"c": censored.astype(int)}
            if design.truncated:
                kwargs["tl"] = design.tl
                kwargs["tr"] = design.tr
        try:
            with ignoring_warnings(), quiet_maximum_warnings():
                refit = model.dist.fit(
                    x,
                    offset=model.offset,
                    lfp=model.lfp,
                    zi=model.zi,
                    fixed=fixed,
                    # From the estimate: each resample's maximum is within
                    # sampling error of it
                    init=init,
                    **kwargs,
                )
        except REFIT_ERRORS:
            counts["failed"] += 1
            continue
        if refit.maximum in counts:
            counts[refit.maximum] += 1
        models.append(refit)
        values.append(
            _at_edges(
                refit, _point(refit), x, np.asarray(kwargs["c"], int), kwargs
            )
        )
    if cases:
        # A weighted model has no lfp_p or f0 (its bootstrap is refused),
        # so nothing to put on an edge.
        estimate = point
    else:
        original = (
            {"tl": design.tl, "tr": design.tr} if design.truncated else {}
        )
        estimate = _at_edges(
            model, point, design.x, (~design.failed).astype(int), original
        )
    return Refits(
        point=point,
        estimate=estimate,
        n_boot=n_boot,
        models=models,
        values=np.array(values).reshape(len(values), len(point)),
        no_maximum=counts["no finite maximum"],
        unverified=counts["unverified"],
        failed=counts["failed"],
        model_maximum=model.maximum,
    )


def _accelerated(model: Any) -> bool:
    """Whether ``model``'s bootstrap intervals are BCa (a limited-failure
    or zero-inflated model's, #286) rather than bias-corrected
    percentile intervals (the plain and offset models', unchanged)."""
    return bool(model.lfp or model.zi)


def _bounds(
    model: Any,
    fits: Refits,
    value: Callable[[Any], npt.NDArray],
    alpha_ci: float,
    bound: str,
) -> npt.NDArray:
    """The bias-corrected percentile bounds on ``value(m)``, a function
    of a model with one value per query point, over the refits about its
    value at ``model``; for a limited-failure or zero-inflated model the
    BCa bounds, with the jackknife's acceleration (:func:`_accelerated`)."""
    est = np.asarray(value(model), dtype=float)
    shape = est.shape
    draws = np.empty((len(fits.models),) + shape)
    with np.errstate(all="ignore"):
        for i, refit in enumerate(fits.models):
            draws[i] = np.reshape(np.asarray(value(refit), dtype=float), shape)
    a = np.zeros(est.size)
    if _accelerated(model):
        jack = jackknife(model, fits)
        jv = np.empty((len(jack.models), est.size))
        with np.errstate(all="ignore"):
            for i, refit in enumerate(jack.models):
                jv[i] = np.asarray(value(refit), dtype=float).reshape(-1)
        a = jackknife_acceleration(jv, jack.weights)
    cb = bca_bounds(
        draws.reshape(len(draws), -1),
        est.reshape(-1),
        a,
        alpha_ci,
        bound,
    )
    return cb.reshape(shape + cb.shape[1:])


def cb_bootstrap(
    model: Any,
    x: npt.NDArray,
    on: str,
    alpha_ci: float,
    bound: str,
    n_boot: Any,
    random_state: Any,
) -> npt.NDArray:
    """``cb(method="bootstrap")``: the bounds of ``on`` at each ``x``
    over the refits. The percentile interval goes with a monotone
    transform, so those on ``sf``, ``ff`` and ``Hf`` are one interval."""
    on = {"R": "sf", "F": "ff"}.get(on, on)
    fits = refits(model, n_boot, random_state)
    fits.warn()
    t = np.asarray(x, dtype=float)
    # On the cumulative hazard, which increases with every parameter's
    # effect the same way at each x; sf and ff are carried from it.
    fn = {"sf": "Hf", "ff": "Hf"}.get(on, on)
    side = bound
    if on == "sf" and bound != "two-sided":
        side = {"lower": "upper", "upper": "lower"}[bound]
    cb = _bounds(model, fits, lambda m: getattr(m, fn)(t), alpha_ci, side)
    if on == "sf":
        if bound == "two-sided":
            cb = cb[..., ::-1]
        with np.errstate(over="ignore"):
            return np.exp(-cb)
    if on == "ff":
        with np.errstate(over="ignore"):
            return -np.expm1(-cb)
    return cb


def quantile_cb_bootstrap(
    model: Any,
    p: npt.NDArray,
    alpha_ci: float,
    bound: str,
    n_boot: Any,
    random_state: Any,
) -> npt.NDArray:
    """``quantile_cb(method="bootstrap")``: the bounds of the refits'
    quantiles ``qf(p)``."""
    fits = refits(model, n_boot, random_state)
    fits.warn()
    return _bounds(model, fits, lambda m: m.qf(p), alpha_ci, bound)


def mrl_cb_bootstrap(
    model: Any,
    x: npt.NDArray,
    alpha_ci: float,
    bound: str,
    n_boot: Any,
    random_state: Any,
) -> npt.NDArray:
    """``mrl_cb(method="bootstrap")``: the bounds of the refits' mean
    residual lives ``mrl(x)``."""
    fits = refits(model, n_boot, random_state)
    fits.warn()
    return _bounds(model, fits, lambda m: m.mrl(x), alpha_ci, bound)


def rmst_cb_bootstrap(
    model: Any,
    tau: npt.NDArray,
    alpha_ci: float,
    bound: str,
    n_boot: Any,
    random_state: Any,
) -> npt.NDArray:
    """``rmst_cb(method="bootstrap")``: the bounds of the refits'
    restricted means to each ``tau``."""
    fits = refits(model, n_boot, random_state)
    fits.warn()
    return _bounds(model, fits, lambda m: m._rmst_values(tau), alpha_ci, bound)


def param_cb_bootstrap(
    model: Any,
    name: str,
    alpha_ci: float,
    bound: str,
    n_boot: Any,
    random_state: Any,
) -> npt.NDArray:
    """``param_cb(method="bootstrap")``: the bounds of the refits'
    parameter ``name`` -- the offset ``gamma``, a limited-failure model's
    ``lfp_p`` and a zero-inflated one's ``f0`` among them. A parameter
    fixed at fit time is known: its interval is its value."""
    valid = (
        (["gamma"] if model.offset else [])
        + list(model.dist.parameter_names)
        + (["lfp_p"] if model.lfp else [])
        + (["f0"] if model.zi else [])
    )
    if name == "gamma" and not model.offset:
        raise ValueError("'gamma' is only estimated for offset models")
    if name not in valid:
        raise ValueError(
            f"Unknown parameter {name!r}; expected one of {valid}"
        )
    idx = _point_index(model, name)
    if model._is_fixed_param(name):
        value = _point(model)[idx]
        return np.array([value, value] if bound == "two-sided" else [value])
    fits = refits(model, n_boot, random_state)
    fits.warn()
    est = np.array([fits.estimate[idx]])
    a = np.zeros(1)
    if _accelerated(model):
        jack = jackknife(model, fits)
        a = jackknife_acceleration(jack.values[:, idx][:, None], jack.weights)
    cb = bca_bounds(fits.values[:, idx][:, None], est, a, alpha_ci, bound)
    return np.atleast_1d(cb[0])


@dataclass
class ParametricBootstrap:
    """
    The bootstrap refits of a fitted parametric model, from its
    ``bootstrap`` method (``Parametric.bootstrap``): the same model
    refitted to ``n_boot`` resamples of its data, the ones behind its
    ``method="bootstrap"`` bounds.

    Attributes
    ----------
    models : list of Parametric
        The refitted models, one per kept refit, each fitted with the
        model's own options (offset, ``lfp``, ``zi``, fixed parameters).
    params : numpy array
        ``(n_kept, len(names))``: each kept refit's parameters in the
        order of ``names``, the model's ``covariance_names``, so
        ``model.with_params(params[i], gamma=gamma[i])`` rebuilds refit
        ``i``'s predictions.
    gamma : numpy array or None
        Each kept refit's offset, for a model fitted with
        ``offset=True`` (the offset is not in the covariance, so not in
        ``params``); ``None`` otherwise.
    names : list of str
        The names of the columns of ``params``.
    n_boot : int
        The number of refits asked for.
    failed : int
        The refits whose fit raised; they are left out, so
        ``len(models) == n_boot - failed``.
    no_maximum : int
        The kept refits whose likelihood had no finite maximum (kept at
        the estimate they reached, as the bounds keep them).
    unverified : int
        The kept refits whose search did not reach a verified maximum.

    Examples
    --------
    >>> import surpyval as surv
    >>> x = [12, 25, 31, 40, 48, 60, 66, 75, 90, 110]
    >>> boot = surv.Weibull.fit(x).bootstrap(50, random_state=1)
    >>> boot.names, boot.params.shape, boot.failed
    (['alpha', 'beta'], (50, 2), 0)
    >>> sorted(boot.to_dict())[:3]
    ['failed', 'gamma', 'n_boot']
    """

    models: list
    params: npt.NDArray
    gamma: npt.NDArray | None
    names: list
    n_boot: int
    failed: int
    no_maximum: int
    unverified: int

    def __len__(self) -> int:
        return len(self.models)

    def __repr__(self) -> str:
        return (
            f"ParametricBootstrap(n_boot={self.n_boot}, "
            f"kept={len(self.models)}, failed={self.failed}, "
            f"names={self.names!r})"
        )

    def to_dict(self) -> dict:
        """The refits' parameters and counts as a dict of native Python
        values, strict JSON (an infinite or missing number is written as
        ``None`` and recorded under ``"non_finite"``). The refitted models
        are not included: ``model.with_params(row, gamma=g)`` rebuilds
        each one's predictions from its row.

        Examples
        --------
        >>> import json
        >>> import surpyval as surv
        >>> x = [12, 25, 31, 40, 48, 60, 66, 75, 90, 110]
        >>> d = surv.Weibull.fit(x).bootstrap(20, random_state=1).to_dict()
        >>> d["result"], d["names"], len(d["params"])
        ('ParametricBootstrap', ['alpha', 'beta'], 20)
        >>> isinstance(json.dumps(d, allow_nan=False), str)
        True
        """
        from surpyval.serialisation import encode_non_finite, to_native

        out: dict = {
            "result": type(self).__name__,
            "names": list(self.names),
            "params": to_native(np.asarray(self.params, dtype=float)),
            "gamma": (
                None
                if self.gamma is None
                else to_native(np.asarray(self.gamma, dtype=float))
            ),
            "n_boot": int(self.n_boot),
            "failed": int(self.failed),
            "no_maximum": int(self.no_maximum),
            "unverified": int(self.unverified),
        }
        return encode_non_finite(out)


def bootstrap(model: Any, n_boot: Any, random_state: Any) -> Any:
    """``Parametric.bootstrap``: the refits :func:`refits` draws for the
    bounds (the same ones, for the same ``n_boot`` and integer
    ``random_state``), with each kept refit's own estimates in the order
    of ``covariance_names`` -- not moved onto an edge, as the bounds'
    ``values`` are, so that ``with_params`` of a row is that refit."""
    fits = refits(model, n_boot, random_state, public=True)
    fits.warn(public=True)
    names = list(model.covariance_names)
    params = np.array(
        [m._estimates() for m in fits.models], dtype=float
    ).reshape(len(fits.models), len(names))
    gamma = (
        np.array([float(m.gamma) for m in fits.models], dtype=float)
        if model.offset
        else None
    )
    return ParametricBootstrap(
        models=list(fits.models),
        params=params,
        gamma=gamma,
        names=names,
        n_boot=fits.n_boot,
        failed=fits.failed,
        no_maximum=fits.no_maximum,
        unverified=fits.unverified,
    )
