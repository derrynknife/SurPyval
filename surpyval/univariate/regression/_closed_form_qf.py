"""The quantiles of the parametric regressions in closed form, where the
family and its baseline allow it.

Every family's quantile is the time at which its cumulative hazard
reaches ``h = -log(1 - p)``, which :func:`._prediction.quantiles_by_inversion`
finds by a bracketed search. For the log-linear families that time is
the baseline's own inverse cumulative hazard at a target moved by the
covariates:

* proportional hazards, ``H(t | Z) = e^(beta'Z) H0(t)``: ``H0(t) = h
  e^(-beta'Z)``;
* accelerated failure time, ``H(t | Z) = H0(e^(beta'Z) t)``: the
  baseline's time at ``h``, divided by ``e^(beta'Z)``;
* proportional odds, ``F/S = e^(-beta'Z) F0/S0`` (``phi S0 / (F0 + phi
  S0)`` the survival): ``H0(t) = log(1 + e^(beta'Z) (e^h - 1))``;
* additive hazards on an Exponential baseline, ``H(t | Z) = (lambda +
  beta'Z) t``: ``t = h / (lambda + beta'Z)`` where the rate is positive.

The baseline's inverse cumulative hazard is taken from ``log H0``, so a
target too small or too large for a float is still exact: the Weibull's
``alpha (H0)^(1/beta)``, the Gumbel's ``mu + sigma log H0``, the
log-logistic's ``alpha (e^H0 - 1)^(1/beta)``, and the normal and gamma
ones through their inverse distribution functions, each taken on the
side (``F0`` or ``S0``) where the probability is below one half, so it
keeps its relative precision.

A custom covariate link, an accelerated life or distributional model,
an additive model on another baseline and a baseline not listed here
have no closed form: :func:`closed_form_times` returns ``None`` for them,
and their quantiles are found by inversion as before.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import gammainccinv, gammaincinv, ndtri

from surpyval.univariate.parametric.distributions.exponential import (
    Exponential_,
)
from surpyval.univariate.parametric.distributions.gamma import Gamma_
from surpyval.univariate.parametric.distributions.gumbel import Gumbel_
from surpyval.univariate.parametric.distributions.logistic import Logistic_
from surpyval.univariate.parametric.distributions.loglogistic import (
    LogLogistic_,
)
from surpyval.univariate.parametric.distributions.lognormal import (
    LogNormal_,
)
from surpyval.univariate.parametric.distributions.normal import Normal_
from surpyval.univariate.parametric.distributions.weibull import Weibull_

from .kinds import (
    ACCELERATED_FAILURE_TIME,
    ADDITIVE_HAZARD,
    PROPORTIONAL_HAZARD,
    PROPORTIONAL_ODDS,
)

_LOG2 = float(np.log(2.0))


def _log_expm1(H: npt.NDArray) -> npt.NDArray:
    """``log(e^H - 1)``, exact for every ``H > 0``: ``H + log(1 -
    e^-H)`` where ``e^H`` would overflow."""
    with np.errstate(all="ignore"):
        big = H > 1.0
        return np.where(
            big,
            H + np.log1p(-np.exp(-np.where(big, H, 1.0))),
            np.log(np.expm1(np.where(big, 1.0, H))),
        )


def _probit_at(H: npt.NDArray) -> npt.NDArray:
    """The standard normal quantile of ``F = 1 - e^-H``, from the side
    where the probability is below one half."""
    with np.errstate(all="ignore"):
        low = H < _LOG2
        return np.where(low, ndtri(-np.expm1(-H)), -ndtri(np.exp(-H)))


def _weibull(log_H: npt.NDArray, alpha: float, beta: float) -> npt.NDArray:
    return alpha * np.exp(log_H / beta)


def _exponential(log_H: npt.NDArray, rate: float) -> npt.NDArray:
    return np.exp(log_H) / rate


def _gumbel(log_H: npt.NDArray, mu: float, sigma: float) -> npt.NDArray:
    return mu + sigma * log_H


def _loglogistic(log_H: npt.NDArray, alpha: float, beta: float) -> npt.NDArray:
    return alpha * np.exp(_log_expm1(np.exp(log_H)) / beta)


def _logistic(log_H: npt.NDArray, mu: float, sigma: float) -> npt.NDArray:
    return mu + sigma * _log_expm1(np.exp(log_H))


def _normal(log_H: npt.NDArray, mu: float, sigma: float) -> npt.NDArray:
    return mu + sigma * _probit_at(np.exp(log_H))


def _lognormal(log_H: npt.NDArray, mu: float, sigma: float) -> npt.NDArray:
    return np.exp(mu + sigma * _probit_at(np.exp(log_H)))


def _gamma(log_H: npt.NDArray, alpha: float, beta: float) -> npt.NDArray:
    H = np.exp(log_H)
    low = H < _LOG2
    z = np.where(
        low, gammaincinv(alpha, -np.expm1(-H)), gammainccinv(alpha, np.exp(-H))
    )
    return z / beta


#: The baselines whose inverse cumulative hazard is in closed form, by
#: the class of their fitter, as a function of ``log H0`` and the
#: baseline's parameters.
_INVERSE_HF: dict[type, Callable[..., npt.NDArray]] = {
    Weibull_: _weibull,
    Exponential_: _exponential,
    Gumbel_: _gumbel,
    LogLogistic_: _loglogistic,
    Logistic_: _logistic,
    Normal_: _normal,
    LogNormal_: _lognormal,
    Gamma_: _gamma,
}


def baseline_inverse(dist: Any) -> Callable[..., npt.NDArray] | None:
    """The function ``(log_H0, *dist_params) -> t`` with ``H0(t) =
    e^log_H0`` for the baseline ``dist``, or ``None`` where it has no
    closed form here."""
    return _INVERSE_HF.get(type(dist))


def _family(fitter: Any) -> str | None:
    """The family of the covariate model ``fitter`` (a regression's
    ``model``) whose quantile has a closed form, or ``None``."""
    from ._fit_skeleton import LogLinearPhi
    from .accelerated_failure_time.aft_fitter import AFTFitter
    from .additive_hazards.additive_hazards_fitter import (
        AdditiveHazardsFitter,
    )
    from .proportional_hazards.proportional_hazards_fitter import (
        ProportionalHazardsFitter,
    )
    from .proportional_odds.proportional_odds_fitter import (
        ProportionalOddsFitter,
    )

    kind = type(fitter)
    if kind is ProportionalHazardsFitter:
        # Only the log-linear link: a custom phi(Z) is any function.
        if getattr(fitter, "phi", None) is LogLinearPhi.phi:
            return PROPORTIONAL_HAZARD
        return None
    if isinstance(fitter, AFTFitter):
        if type(fitter)._phi is AFTFitter._phi:
            return ACCELERATED_FAILURE_TIME
        return None
    if kind is ProportionalOddsFitter:
        return PROPORTIONAL_ODDS
    if kind is AdditiveHazardsFitter and type(fitter.dist) is Exponential_:
        return ADDITIVE_HAZARD
    return None


def has_closed_form(fitter: Any) -> bool:
    """Whether :func:`closed_form_times` has a closed form for the
    covariate model ``fitter``."""
    family = _family(fitter)
    if family is None:
        return False
    return family == ADDITIVE_HAZARD or (
        baseline_inverse(fitter.dist) is not None
    )


def closed_form_times(
    fitter: Any,
    params: npt.NDArray,
    rows: npt.NDArray,
    h: npt.NDArray,
) -> npt.NDArray | None:
    """The times at which the cumulative hazard of the covariate model
    ``fitter`` at ``params`` and the (centred) covariate ``rows`` (one per
    target) reaches ``h``, in closed form; ``None`` where the family or
    the baseline has none.

    Only the targets strictly inside (0, inf) are computed; the others
    are ``nan`` here, for the caller to fill. A closed form that is not
    finite (a time beyond the floats) is ``nan`` too, for the caller to
    solve by inversion.
    """
    if not has_closed_form(fitter):
        return None
    family = _family(fitter)
    params = np.asarray(params, dtype=float)
    k = int(fitter.k_dist)
    dist_params = tuple(float(v) for v in params[:k])
    beta = params[k:]
    h = np.asarray(h, dtype=float)
    rows = np.asarray(rows, dtype=float).reshape(h.shape[0], -1)
    out = np.full(h.shape, np.nan)
    inside = (h > 0) & (h < np.inf)
    if not inside.any():
        return out
    target = h[inside]
    with np.errstate(all="ignore"):
        eta = rows[inside] @ beta if beta.size else np.zeros(target.shape)
        log_h = np.log(target)
        if family == ADDITIVE_HAZARD:
            rate = dist_params[0] + eta
            t = np.where(rate > 0, target / rate, np.nan)
        else:
            inverse = baseline_inverse(fitter.dist)
            assert inverse is not None
            if family == PROPORTIONAL_HAZARD:
                t = inverse(log_h - eta, *dist_params)
            elif family == ACCELERATED_FAILURE_TIME:
                t = inverse(log_h, *dist_params) / np.exp(eta)
            else:
                # The odds of failure are e^(beta'Z) times the
                # baseline's, so H0 = log(1 + e^a) with a = beta'Z +
                # log(e^h - 1); log H0 = a where log(1 + e^a) = e^a to
                # the last bit.
                a = eta + _log_expm1(target)
                log_H0 = np.where(a < -37.0, a, np.log(np.logaddexp(0.0, a)))
                t = inverse(log_H0, *dist_params)
    t = np.asarray(t, dtype=float)
    out[inside] = np.where(np.isfinite(t), t, np.nan)
    return out
