"""The parametric bootstrap bounds of a fitted parametric regression model
(``method="bootstrap"`` on ``cb``, ``param_cb``, ``quantile_cb`` and
``cb_tvc``, #617).

The model is resampled, not the data: each resample simulates a failure
time for every unit from the fitted model at the unit's own covariates
(and within its own truncation window), censors it as the unit was
censored, and refits the same model to the result. The bounds are the
BCa intervals (Efron 1987, "Better bootstrap confidence intervals",
JASA 82) of the refits' functions or parameters: the percentiles of the
refits, moved by a bias correction (from the share of refits below the
estimate) and an acceleration (the skewness of the score of the least
favourable family, from each resample's score at the estimate, so no
refits beyond the ``n_boot``). On #583's accelerated life test (46
failures) the 90% BCa bound covered 0.903, where the percentile
interval, the Wald and the likelihood-ratio bounds covered 0.866 to
0.880 (#617).

The censoring is that of the data, by Davison & Hinkley's conditional
bootstrap (*Bootstrap Methods and their Application*, 1997, Algorithm
3.3): a censored unit is censored at its own censoring time, and a unit
that failed at ``x`` is censored at a time drawn from the product-limit
estimate of the censoring distribution given that it exceeds ``x``
(beyond the last censoring time, never). A test stopped at a common
time (Type I) therefore censors every unit at that time, and data with
no censoring stay uncensored.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import autograd.numpy as np
import numpy.typing as npt
from scipy.special import ndtr, ndtri

from surpyval.utils.no_maximum import quiet_maximum_warnings
from surpyval.utils.rng import as_generator
from surpyval.utils.validation import check_option
from surpyval.utils.warnings import caller_stacklevel

from ._likelihood_ratio import LR_NAMES
from ._prediction import quantiles_by_inversion

#: The share of refits that may fail to reach a verified maximum before
#: the bounds warn: a few in a thousand are expected of small samples, as
#: a resample with no failures at a stress leaves its effect unbounded.
FAILED_SHARE = 0.02

#: What a refit that raises is counted as failing with.
REFIT_ERRORS = (
    ValueError,
    RuntimeError,
    ArithmeticError,
    np.linalg.LinAlgError,
)


def bound_method(method: "str | None") -> str:
    """The bound ``method`` asks for: ``"wald"``, ``"lr"`` (or one of
    its aliases) or ``"bootstrap"``; anything else is refused. Case does
    not matter, and ``None`` is the default, ``"wald"``, as for the
    univariate models (#655)."""
    if method is None:
        return "wald"
    m = str(method).lower()
    if m in LR_NAMES:
        return "lr"
    if m in ("wald", "bootstrap"):
        return m
    check_option(
        "method",
        method,
        ("wald", "lr", "bootstrap"),
        "Case does not matter, and 'likelihood', 'likelihood-ratio' and "
        "'profile' also mean 'lr'.",
    )
    return m  # pragma: no cover (check_option raised)


def check_n_boot(n_boot: Any) -> int:
    """``n_boot`` as an int, or a ``ValueError`` naming it."""
    if (
        isinstance(n_boot, bool)
        or not isinstance(n_boot, (int, np.integer))
        or n_boot < 1
    ):
        raise ValueError(
            "'n_boot' must be a positive integer; got {!r}".format(n_boot)
        )
    return int(n_boot)


@dataclass
class Refits:
    """The refits of one bootstrap of a model: each refit's parameters
    (as the model reports its own, ``_eval_params``) and the covariate
    point its baseline is at, and how many of the ``n_boot`` refits
    reached no verified maximum or failed outright."""

    #: What the refits are of (``_covariance_point`` of the model).
    point: tuple
    n_boot: int
    #: ``(n_kept, len(params))``: the refits that returned a model.
    params: npt.NDArray
    #: The baseline point of each kept refit (the model's ``center``).
    centers: list
    #: ``(n_kept, len(params))``: the score (gradient of the
    #: log-likelihood) of each kept resample at the model's parameters,
    #: from which the BCa interval's acceleration is found.
    scores: npt.NDArray
    #: Kept refits whose likelihood had no finite maximum, or whose
    #: maximum was not verified (the values they reached are kept).
    no_maximum: int
    unverified: int
    #: Refits that raised (left out).
    failed: int
    #: What the model's own fit reached (its ``maximum``).
    model_maximum: str = "verified"

    def warn(self) -> None:
        """One warning: that the model itself has no finite maximum, or
        that more than ``FAILED_SHARE`` of the refits did not reach a
        verified maximum or failed."""
        bad = self.no_maximum + self.unverified + self.failed
        if self.model_maximum == "no finite maximum":
            # On #583's test with 11 failures, the bootstrap bound from
            # such a fit covered 0.004 (the Wald bound, [0, 1] or nan
            # there, 1.0 of those that were not nan; #617).
            warnings.warn(
                "This model's likelihood has no finite maximum (as its fit "
                "warned), so the data simulated from it run off the same "
                "way and the bootstrap bounds close onto its meaningless "
                "estimate. They are not a confidence bound; the data do "
                "not determine the model (too few failures at some "
                "covariate values).",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
            return
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
            "reached, where its parameters run off to their limit, so that "
            "it counts at its end of the bounds; with this many, the data "
            "barely identify the model (too few failures at some covariate "
            "values), and the bounds are rough. Compare them with "
            "method='lr'.",
            UserWarning,
            stacklevel=caller_stacklevel(),
        )


def refits(model: Any, n_boot: Any, random_state: Any) -> Refits:
    """The bootstrap refits of ``model``: drawn, or those kept on the
    model for this ``n_boot`` and integer ``random_state`` while its
    parameters and data are as they are (``random_state=None`` or a
    ``Generator`` draws afresh each time, from that stream)."""
    from ._inference import _same_point

    n_boot = check_n_boot(n_boot)
    design = ResampleDesign(model)
    point = model._covariance_point(model._eval_params(), model.center)
    key = None
    if isinstance(random_state, (int, np.integer)) and not isinstance(
        random_state, bool
    ):
        key = (n_boot, int(random_state))
        kept = model._bootstrap_refits
        if kept is not None and key in kept:
            if _same_point(kept[key].point, point):
                return kept[key]
    out = _draw(model, design, point, n_boot, as_generator(random_state))
    if key is not None:
        if model._bootstrap_refits is None:
            model._bootstrap_refits = {}
        model._bootstrap_refits[key] = out
    return out


class ResampleDesign:
    """What a resample of ``model``'s data keeps: each unit's covariates,
    truncation window and censoring (a row with ``n = k`` is ``k``
    units)."""

    def __init__(self, model: Any) -> None:
        if getattr(model, "data", None) is None or model._restored:
            raise ValueError(
                "Bootstrap bounds need the data the model was fitted to, "
                "which a model restored from a dict does not keep; use "
                "method='wald'."
            )
        if model.is_tvc:
            raise ValueError(
                "Bootstrap bounds are not available for a model fitted to "
                "time-varying covariates: a resample would need each "
                "subject's covariate path, which the fit does not keep. "
                "Use method='wald' or method='lr'."
            )
        data = model.data
        c = np.asarray(data.c)
        if np.any((c != 0) & (c != 1)):
            raise ValueError(
                "Bootstrap bounds resample exact and right-censored data "
                "(truncated or not); this model was fitted to left- or "
                "interval-censored data, whose inspection times a resample "
                "would need and the data do not record. Use method='wald' "
                "or method='lr'."
            )
        # (the counts are whole numbers: the data handler requires it)
        units = np.repeat(np.arange(len(c)), np.asarray(data.n, dtype=int))
        self.x = np.asarray(data.x, dtype=float)[units]
        self.failed = c[units] == 0
        self.Z = np.asarray(data.Z, dtype=float)[units]
        t = np.asarray(data.t, dtype=float)[units]
        self.tl, self.tr = t[:, 0], t[:, 1]
        self.truncated = bool(
            np.any(np.isfinite(self.tl)) or np.any(np.isfinite(self.tr))
        )
        self._censoring_distribution()

    def _censoring_distribution(self) -> None:
        """The product-limit estimate of the censoring distribution (the
        censored units its events, the failed ones censored at their
        failure, each at risk from its entry): its distinct times and the
        probability at each, the rest of the mass at infinity."""
        times = np.unique(self.x[~self.failed])
        mass = np.zeros(times.size)
        surv = 1.0
        for j, s in enumerate(times):
            at_risk = np.sum((self.x >= s) & (self.tl < s))
            d = np.sum((self.x == s) & ~self.failed)
            mass[j] = surv * d / at_risk
            surv *= 1.0 - d / at_risk
        self.censor_times = times
        # The mass at or after each time, and past the last (the tail).
        tail = max(surv, 0.0)
        self.censor_after = np.append(np.cumsum(mass[::-1])[::-1] + tail, tail)
        # Where each failed unit's censoring time may start.
        self.first_later = np.searchsorted(times, self.x[self.failed])

    def censoring(self, rng: np.random.Generator) -> npt.NDArray:
        """A censoring time for every unit: a censored unit's own, and for
        a failed unit one drawn from the censoring distribution given
        that it is at least the unit's failure time (infinite past the
        last censoring time)."""
        C = np.where(self.failed, np.inf, self.x)
        times, after = self.censor_times, self.censor_after
        if not times.size:
            return C
        # Inverse transform on the mass after the failure: the time j
        # with after[j + 1] < w <= after[j], or the tail (j = len(times)).
        w = (1.0 - rng.uniform(size=self.first_later.size)) * after[
            self.first_later
        ]
        j = np.searchsorted(-after, -w, side="right") - 1
        drawn = np.append(times, np.inf)[np.clip(j, 0, times.size)]
        C[self.failed] = drawn
        return C


def _draw(
    model: Any,
    design: ResampleDesign,
    point: tuple,
    n_boot: int,
    rng: np.random.Generator,
) -> Refits:
    """``n_boot`` resamples simulated from ``model`` with ``design`` and
    refitted (see the module docstring)."""
    p_hat = np.asarray(model._eval_params(), dtype=float)
    H = _cumulative_hazard(model, p_hat, model.center)
    H_lo = H(np.where(np.isfinite(design.tl), design.tl, -np.inf), design.Z)
    H_hi = H(np.where(np.isfinite(design.tr), design.tr, np.inf), design.Z)
    kwargs: dict[str, Any] = {
        "fixed": {
            k: v for k, v in model.fixed.items() if k != model.life_parameter
        }
        or None,
        # From the estimate: each resample's maximum is within sampling
        # error of it (a fit from ``init`` that is not verified also
        # searches from the default start).
        "init": None if model.aliased.size else p_hat,
    }
    if not model._is_accelerated_life():
        kwargs["center"] = model._has_center()
    score = _score(model, p_hat)
    params, centers, scores = [], [], []
    counts = {"no finite maximum": 0, "unverified": 0, "failed": 0}
    for _ in range(n_boot):
        x, c, t = _simulate(model, design, p_hat, H_lo, H_hi, rng)
        try:
            with warnings.catch_warnings(), quiet_maximum_warnings():
                warnings.simplefilter("ignore")
                refit = model.model.fit(
                    x,
                    design.Z,
                    c=c,
                    t=t if design.truncated else None,
                    **kwargs,
                )
        except REFIT_ERRORS:
            counts["failed"] += 1
            continue
        if refit.maximum in counts:
            counts[refit.maximum] += 1
        params.append(np.asarray(refit._eval_params(), dtype=float))
        centers.append(refit.center)
        scores.append(score(refit.data))
    return Refits(
        point=point,
        n_boot=n_boot,
        params=np.array(params).reshape(len(params), p_hat.size),
        centers=centers,
        scores=np.array(scores).reshape(len(scores), p_hat.size),
        no_maximum=counts["no finite maximum"],
        unverified=counts["unverified"],
        failed=counts["failed"],
        model_maximum=model.maximum,
    )


def _cumulative_hazard(
    model: Any, params: npt.NDArray, center: Any
) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
    """``H(t, Z)``, the cumulative hazard of ``model``'s family at
    ``params`` (its baseline at ``center``), paired by row: 0 at or below
    the start of the support and infinite at its end."""
    lower, upper = (float(v) for v in model.distribution.support)

    def H(t: npt.NDArray, Z: npt.NDArray) -> npt.NDArray:
        t = np.asarray(t, dtype=float)
        rows = model._centred(Z, center)
        inside = (t > lower) & (t < upper)
        t_in = np.where(inside, t, lower + 1.0 if np.isfinite(lower) else 0)
        with np.errstate(all="ignore"):
            out = np.asarray(model.model.Hf(t_in, rows, *params), dtype=float)
        return np.where(inside, out, np.where(t >= upper, np.inf, 0.0))

    return H


def _simulate(
    model: Any,
    design: ResampleDesign,
    p_hat: npt.NDArray,
    H_lo: npt.NDArray,
    H_hi: npt.NDArray,
    rng: np.random.Generator,
) -> tuple:
    """One resample: a failure time for every unit from the fitted model
    at its covariates, within its truncation window (drawn on the
    cumulative hazard, ``H(T) = H(tl) + E`` with ``E`` exponential given
    ``H(T) < H(tr)``), censored at its censoring time."""
    u = rng.uniform(size=design.x.size)
    with np.errstate(over="ignore", invalid="ignore"):
        # E exponential, conditioned on E < H_hi - H_lo.
        E = -np.log1p(u * np.expm1(-(H_hi - H_lo)))
    target = H_lo + E
    T = _times_at(model, p_hat, model.center, target, design.Z)
    C = design.censoring(rng)
    censored = T > C
    x = np.where(censored, C, T)
    t = np.column_stack([design.tl, design.tr])
    return x, censored.astype(int), t


def _times_at(
    model: Any,
    params: npt.NDArray,
    center: Any,
    H_target: npt.NDArray,
    Z: npt.NDArray,
) -> npt.NDArray:
    """The times at which ``model``'s cumulative hazard at ``params``
    reaches ``H_target``, row by row of ``Z``."""
    rows = np.asarray(model._centred(Z, center), dtype=float)
    with np.errstate(all="ignore"):
        p = -np.expm1(-np.asarray(H_target, dtype=float))
        start = np.asarray(
            model.distribution.qf(p, *params[: model.k_dist]), dtype=float
        ) * np.ones(p.shape)
    return quantiles_by_inversion(
        lambda t, k: model.model.Hf(t, rows[k], *params),
        p,
        model.distribution.support,
        start,
    )


# -- the scores of the resamples ---------------------------------------------


def _score(model: Any, p_hat: npt.NDArray) -> Callable[[Any], npt.NDArray]:
    """``score(data)``: the gradient of the log-likelihood of ``data`` at
    the model's own parameters ``p_hat`` (its baseline at its
    ``center``), by central differences in the steps of its covariance
    (``_hessian_step``); zero for a held parameter."""
    from ._fit_skeleton import centred_copy

    names = model.parameter_names
    held = model._held()
    free = [i for i, nm in enumerate(names) if nm not in held]
    step = model._hessian_step(p_hat)
    center = model.center
    shift = center is not None and bool(np.any(center))

    def score(data: Any) -> npt.NDArray:
        if shift:
            data = centred_copy(data, center)
        out = np.zeros(p_hat.size)
        with np.errstate(all="ignore"):
            for i in free:
                e = np.zeros(p_hat.size)
                e[i] = step[i]
                up = model.model.neg_ll(data, *(p_hat + e))
                down = model.model.neg_ll(data, *(p_hat - e))
                out[i] = -(float(up) - float(down)) / (2.0 * step[i])
        return out

    return score


# -- the bounds --------------------------------------------------------------


def param_cb_bootstrap(
    model: Any,
    idx: int,
    alpha_ci: float,
    bound: str,
    n_boot: Any,
    random_state: Any,
) -> npt.NDArray:
    """``param_cb(method="bootstrap")``: the BCa interval of the refits'
    parameter ``idx``. A held parameter's interval is its value, as for
    the Wald bound."""
    fits = refits(model, n_boot, random_state)
    fits.warn()
    names = model.parameter_names
    if names[idx] in model._held():
        value = float(model.params[idx])
        return np.array([value, value] if bound == "two-sided" else [value])
    p_hat = np.asarray(model._eval_params(), dtype=float)
    grad = np.zeros((1, p_hat.size))
    grad[0, idx] = 1.0
    cb = bca_bounds(
        fits.params[:, idx][:, None],
        p_hat[idx : idx + 1],
        acceleration(model, fits, grad),
        alpha_ci,
        bound,
    )[0]
    return np.atleast_1d(cb)


def cb_bootstrap(
    model: Any,
    x: npt.ArrayLike,
    Z: Any,
    on: str,
    alpha_ci: float,
    bound: str,
    n_boot: Any,
    random_state: Any,
) -> npt.NDArray:
    """``cb(method="bootstrap")``: the BCa interval of ``on`` over the
    refits at each ``x`` and row of ``Z`` (paired as for ``sf``). ``sf``,
    ``ff`` and ``Hf`` are one interval, on the cumulative hazard, so they
    agree exactly; below the support the bound is the estimate."""
    on = {"R": "sf", "F": "ff"}.get(on, on)
    fits = refits(model, n_boot, random_state)
    fits.warn()
    t = np.asarray(x, dtype=float)
    rows = model._prepare_Z(Z)
    lower = float(model.distribution.support[0])
    below = t < lower
    x_in = np.where(below, lower + 1.0 if np.isfinite(lower) else 0.0, t)
    if on in ("hf", "df"):
        fn = model.model.hf if on == "hf" else model.model.df

        def value(p: npt.NDArray, center: Any = None) -> npt.NDArray:
            with np.errstate(all="ignore"):
                v = fn(x_in, model._centred(rows, center), *p)
            v = np.broadcast_to(np.asarray(v, dtype=float), t.shape)
            return np.where(below, 0.0, v)

        return function_bounds(model, fits, value, None, alpha_ci, bound)

    def H_of(p: npt.NDArray, center: Any = None) -> npt.NDArray:
        with np.errstate(all="ignore"):
            H = model.model.Hf(x_in, model._centred(rows, center), *p)
        H = np.broadcast_to(np.asarray(H, dtype=float), t.shape)
        return np.where(below, 0.0, H)

    return function_bounds(model, fits, H_of, on, alpha_ci, bound)


def quantile_cb_bootstrap(
    model: Any,
    p: npt.NDArray,
    rows: npt.NDArray,
    alpha_ci: float,
    bound: str,
    n_boot: Any,
    random_state: Any,
) -> npt.NDArray:
    """``quantile_cb(method="bootstrap")``: the BCa interval of the
    refits' quantiles ``qf(p[i], rows[i])``."""
    fits = refits(model, n_boot, random_state)
    fits.warn()
    H_target = -np.log1p(-p)

    def value(params: npt.NDArray, center: Any = None) -> npt.NDArray:
        center = model.center if center is None else center
        return _times_at(model, params, center, H_target, rows)

    return function_bounds(model, fits, value, None, alpha_ci, bound)


def tvc_refits(model: Any, n_boot: Any, random_state: Any) -> Refits:
    """The refits of ``cb_tvc(method="bootstrap")`` (warned of)."""
    fits = refits(model, n_boot, random_state)
    fits.warn()
    return fits


def function_bounds(
    model: Any,
    fits: Refits,
    value: Callable[..., npt.NDArray],
    on: "str | None",
    alpha_ci: float,
    bound: str,
) -> npt.NDArray:
    """The BCa bounds on ``value(params, center)``, a function of the
    parameters (and of the covariate point the baseline is at, the
    model's by default) with one value per query point: over the refits,
    each at its own baseline point, about its value at the model's
    parameters. With ``on`` (``sf``, ``ff`` or ``Hf``), ``value`` is the
    cumulative hazard, and the one interval on it is carried to ``on``
    (``sf`` decreases in it, so its lower end is the hazard's upper)."""
    p_hat = np.asarray(model._eval_params(), dtype=float)
    est = np.asarray(value(p_hat), dtype=float)
    shape = est.shape
    draws = np.empty((len(fits.centers),) + shape)
    for i, (p, center) in enumerate(zip(fits.params, fits.centers)):
        draws[i] = np.reshape(value(p, center), shape)
    a = acceleration(model, fits, _jacobian(model, value, p_hat))
    flip = on == "sf" and bound != "two-sided"
    side = {"lower": "upper", "upper": "lower"}.get(bound, bound)
    cb = bca_bounds(
        draws.reshape(len(draws), -1),
        est.reshape(-1),
        a,
        alpha_ci,
        side if flip else bound,
    )
    cb = cb.reshape(shape + cb.shape[1:])
    if on == "sf":
        if bound == "two-sided":
            cb = cb[..., ::-1]
        with np.errstate(over="ignore"):
            return np.exp(-cb)
    if on == "ff":
        with np.errstate(over="ignore"):
            return -np.expm1(-cb)
    return cb


def _jacobian(
    model: Any, value: Callable[..., npt.NDArray], p_hat: npt.NDArray
) -> npt.NDArray:
    """``(n_points, len(p_hat))``: the central-difference gradient of
    ``value`` at ``p_hat`` at each query point, in the steps of the
    covariance; zero in a held parameter."""
    names = model.parameter_names
    held = model._held()
    step = model._hessian_step(p_hat)
    size = np.size(value(p_hat))
    cols = []
    with np.errstate(all="ignore"):
        for i, name in enumerate(names):
            if name in held:
                cols.append(np.zeros(size))
                continue
            e = np.zeros(p_hat.size)
            e[i] = step[i]
            up = np.asarray(value(p_hat + e), dtype=float).reshape(-1)
            down = np.asarray(value(p_hat - e), dtype=float).reshape(-1)
            cols.append((up - down) / (2.0 * step[i]))
    return np.stack(cols, axis=-1)


def acceleration(model: Any, fits: Refits, grad: npt.NDArray) -> npt.NDArray:
    """The BCa acceleration of each function whose gradient at the
    estimate is a row of ``grad``: Efron's (1987) parametric
    acceleration, a sixth of the skewness of the score of the least
    favourable family, ``L = grad' I^-1 S``, over the resamples' scores
    ``S`` at the estimate (``I^-1`` the model's covariance). Taken as 0
    (the bias-corrected percentile interval) where the covariance or the
    skewness is not finite."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = np.array(model.covariance(), dtype=float)
    held = model._held()
    out = [i for i, nm in enumerate(model.parameter_names) if nm in held]
    cov[out, :] = 0.0
    cov[:, out] = 0.0
    n_points = grad.shape[0]
    if fits.scores.shape[0] < 3 or not np.all(np.isfinite(cov)):
        return np.zeros(n_points)
    with np.errstate(all="ignore"):
        L = fits.scores @ (np.nan_to_num(grad) @ cov).T
        L = L - L.mean(axis=0)
        m2 = np.mean(L**2, axis=0)
        a = np.mean(L**3, axis=0) / (6.0 * m2**1.5)
    return np.where(np.isfinite(a), a, 0.0)


def bca_bounds(
    draws: npt.NDArray,
    est: npt.NDArray,
    a: npt.NDArray,
    alpha_ci: float,
    bound: str,
) -> npt.NDArray:
    """The BCa bounds (Efron 1987) of each column of ``draws`` (one
    refit per row) about the estimate ``est`` of that column, with
    acceleration ``a``: the quantiles of the draws at ``Phi(z0 + w / (1 -
    a w))``, ``w = z0 + z_alpha``, the bias correction ``z0`` the normal
    quantile of the share of draws below the estimate (ties counted
    half, and the share kept within half a draw of 0 and 1). Where ``1 -
    a w`` is not positive the level is the end of the draws on that
    side. The quantiles interpolate linearly, as ``np.quantile``'s.
    Infinite draws (a function that runs off to a limit) count at their
    end; fewer than two draws give ``nan``."""
    n, m = draws.shape
    sides = (2,) if bound == "two-sided" else ()
    if n < 2:
        return np.full((m,) + sides, np.nan)
    tails = {
        "two-sided": [alpha_ci / 2.0, 1.0 - alpha_ci / 2.0],
        "lower": [alpha_ci],
        "upper": [1.0 - alpha_ci],
    }[bound]
    big = np.finfo(float).max
    ordered = np.sort(np.clip(draws, -big, big), axis=0)
    with np.errstate(invalid="ignore"):
        below = np.sum(draws < est, axis=0)
        tied = np.sum(draws == est, axis=0)
    share = np.clip((below + 0.5 * tied) / n, 0.5 / n, 1.0 - 0.5 / n)
    z0 = ndtri(share)
    out = []
    cols = np.arange(m)
    for tail in tails:
        w = z0 + ndtri(tail)
        denom = 1.0 - a * w
        with np.errstate(divide="ignore", invalid="ignore"):
            adjusted = np.where(denom > 0, z0 + w / denom, np.sign(w) * np.inf)
        pos = ndtr(adjusted) * (n - 1)
        lo = np.clip(np.floor(pos).astype(int), 0, n - 1)
        hi = np.minimum(lo + 1, n - 1)
        frac = pos - lo
        v_lo, v_hi = ordered[lo, cols], ordered[hi, cols]
        with np.errstate(invalid="ignore", over="ignore"):
            v = np.where(frac > 0, v_lo + frac * (v_hi - v_lo), v_lo)
        with np.errstate(invalid="ignore"):
            v = np.where(np.abs(v) >= big, np.sign(v) * np.inf, v)
        # A missing estimate (a nan time) gives nan.
        out.append(np.where(np.isnan(est), np.nan, v))
    return np.stack(out, axis=-1) if bound == "two-sided" else out[0]
