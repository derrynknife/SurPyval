"""What the software reliability growth models share: the refusal of data
that show no growth.

Each of these models reaches a simpler process at the edge of its
parameters: Goel-Okumoto's ``a (1 - exp(-b t))`` and Musa-Okumoto's
``log(1 + lambda0 theta t) / theta`` become a constant rate (as ``a``
grows and ``b`` falls with ``a b`` held, or as ``theta`` falls to 0), and
Yamada's delayed S-shaped curve becomes an intensity that rises in
proportion to time. Data whose failures do not slow down are fitted best
*at* that edge: the likelihood keeps rising towards it and there is no
maximum, so no finite number of faults (or positive decay) to report. The
fitters here say so, rather than return wherever a search stopped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from surpyval.utils.event_history_data import EventHistoryData
from surpyval.utils.warnings import ignoring_warnings

from .nhpp_fitter import NHPPFitter, nhpp_log_likelihood

#: A fit must beat its model's limiting process by more than this much
#: log-likelihood to be told from it: a search that ran to the edge of
#: the parameters ends within the searches' tolerance (about 1e-8) of the
#: limit's likelihood, on either side of it.
LIMIT_MARGIN = 1e-6


def _power_gain(start: np.ndarray, end: np.ndarray, p: int) -> np.ndarray:
    """``end**p - start**p``."""
    return end**p - start**p


def limit_centroids(s: dict, p: int, m: int = 1) -> tuple[float, float]:
    """The mean time of the events, and the mean time the limiting
    process ``c t**p`` would give them over the same observation.

    ``s`` is the data split by ``split_for_nhpp_likelihood``. An exact
    event counts at its time; a count over ``(l, r]`` at the mean time of
    an event of the limiting process in it, ``p / (p + 1) (r**(p + 1) -
    l**(p + 1)) / (r**p - l**p)`` (the midpoint for ``p = 1``). The
    limiting process's own mean is the same expression over every
    observed stretch. With ``m`` it is the mean of the time to the power
    ``m`` (``p / (p + m)`` and powers ``p + m`` in the expression), for a
    model whose curve bends with ``t**m``: the Rayleigh model's, with
    ``t**2``.

    Examples
    --------
    Events at 1, 2 and 6 of a test from 0 to 10: their mean time is 3,
    and a constant rate would put it at 5.

    >>> import surpyval as sp
    >>> from surpyval.recurrent.parametric.software_growth import (
    ...     limit_centroids,
    ... )
    >>> data = sp.handle_xicn(
    ...     [1.0, 2.0, 6.0, 10.0], c=[0, 0, 0, 1], as_recurrent_data=True
    ... )
    >>> seen, limit = limit_centroids(data.split_for_nhpp_likelihood(), 1)
    >>> float(seen), float(limit)
    (3.0, 5.0)
    """
    q = p + m
    ratio = p / q
    starts = [s["x_left_prev"], s["x_i_l"]]
    ends = [s["x_left"], s["x_i_r"]]
    counts = [s["n_left"], s["n_i"]]
    total = float(np.size(s["x_o"]))
    moment = float(np.sum(np.asarray(s["x_o"], dtype=float) ** m))
    for lo, hi, n in zip(starts, ends, counts):
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        n = np.asarray(n, dtype=float)
        total += float(n.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = ratio * _power_gain(lo, hi, q) / _power_gain(lo, hi, p)
        moment += float(np.sum(np.where(n > 0, n * mean, 0.0)))
    starts += [s["x_o_prev"], s["x_right_prev"]]
    ends += [s["x_o"], s["x_right"]]
    lo = np.concatenate([np.asarray(v, dtype=float) for v in starts])
    hi = np.concatenate([np.asarray(v, dtype=float) for v in ends])
    exposure = float(np.sum(_power_gain(lo, hi, p)))
    limit = ratio * float(np.sum(_power_gain(lo, hi, q))) / exposure
    return moment / total, limit


class GrowthNHPPFitter(NHPPFitter):
    """An NHPP whose intensity falls as faults are found and removed: the
    software reliability growth models. It adds to ``NHPPFitter`` the
    refusal of data that show no growth, which have no maximum-likelihood
    estimate (see this module's docstring)."""

    name: str

    #: The limiting process at the edge of the parameters is ``c t**p``:
    #: this ``p`` (1, a constant rate; 2, a rate rising in proportion to
    #: time), or ``None`` where the edge is not one such process.
    _limit_power: int | None = 1
    #: The failures' times are compared with the limit's in the mean of
    #: this power of the time (the power of ``t`` the curve bends with)
    _limit_moment: int = 1
    #: What the limiting process is, in words, and what is lost there
    _limit_name = "a constant rate"
    _no_estimate = "the expected total number of faults a"
    _edge = "a grows without bound and b falls to 0"
    #: The parameter that is the expected number of faults in all, for a
    #: model that has one (``None``: the faults are not finite).
    total_parameter: str | None = "a"

    def _has_selection(self, s: dict) -> bool:
        return bool(np.size(s["sel_to"]) or np.size(s["gap_to"]))

    def _check_identified(self, data: EventHistoryData) -> None:
        """Refuse data whose failures come no earlier, on average, than
        the limiting process would put them: the likelihood then rises
        all the way to the edge of the parameters (its slope along the
        profile is the difference of the two mean times)."""
        s = data.split_for_nhpp_likelihood()
        if self._limit_power is None or self._has_selection(s):
            return
        seen, limit = limit_centroids(s, self._limit_power, self._limit_moment)
        if not seen < limit:
            raise ValueError(self._no_growth(seen, limit))

    def _no_growth(self, seen: float | None, limit: float | None) -> str:
        if seen is None or limit is None:
            evidence = (
                "the fitted likelihood is no higher than that of the "
                "model's limit, {}".format(self._limit_name)
            )
        else:
            evidence = (
                "the failures' mean {}time, {:.6g}, is not earlier than "
                "the {:.6g} that the model's limit, {}, would give over "
                "the same observation".format(
                    "squared " if self._limit_moment == 2 else "",
                    seen,
                    limit,
                    self._limit_name,
                )
            )
        return (
            "The data show no reliability growth that the {name} model "
            "can fit: {evidence}. The likelihood then keeps rising as "
            "{edge}, so {lost} has no estimate. Fit HPP (a constant rate) "
            "or CrowAMSAA (a power law, which also fits a rising rate) "
            "and compare them with aic(); or test for longer, until the "
            "failures slow.".format(
                name=self.name,
                evidence=evidence,
                edge=self._edge,
                lost=self._no_estimate,
            )
        )

    def _limit_log_likelihoods(self, data: EventHistoryData) -> list[float]:
        """The greatest log-likelihood of each limiting process."""
        s = data.split_for_nhpp_likelihood()
        return [_power_law_limit(s, p) for p in self._limit_powers()]

    def _limit_powers(self) -> tuple[int, ...]:
        return () if self._limit_power is None else (self._limit_power,)

    def fit_from_recurrent_data(
        self,
        data: EventHistoryData,
        how: str = "MLE",
        init: Any = None,
        frailty: str | None = None,
        cluster: Any = None,
    ) -> Any:
        model = super().fit_from_recurrent_data(
            data, how, init, frailty, cluster
        )
        if how == "MLE" and frailty is None and cluster is None:
            # A search that ran to the edge stops at the limit's
            # likelihood (to the searches' tolerance), and a maximum
            # inside is above it.
            fitted = -float(model._neg_ll(np.asarray(model.params)))
            limits = self._limit_log_likelihoods(data)
            if limits and not fitted > max(limits) + LIMIT_MARGIN:
                raise ValueError(self._no_growth(None, None))
        return model

    fit_from_recurrent_data.__doc__ = (
        NHPPFitter.fit_from_recurrent_data.__doc__
    )


def _log_power_slope(x: Any, p: int) -> np.ndarray:
    """The log of the slope of ``t**p`` at ``x``."""
    x = np.asarray(x, dtype=float)
    if p == 1:
        return np.zeros_like(x)
    return np.log(p) + (p - 1) * np.log(x)


def _power_law_limit(s: dict, p: int) -> float:
    """The greatest log-likelihood of the process ``c t**p`` for the data
    split ``s``: in closed form (``c`` is the events over the exposure),
    or searched where right truncation adds its terms."""

    def log_likelihood(log_c: float) -> float:
        with np.errstate(all="ignore"):
            return float(
                nhpp_log_likelihood(
                    lambda x: np.exp(log_c) * np.asarray(x, dtype=float) ** p,
                    lambda x: log_c + _log_power_slope(x, p),
                    s,
                )
            )

    events = np.size(s["x_o"]) + np.sum(s["n_left"]) + np.sum(s["n_i"])
    lo = np.concatenate(
        [s["x_o_prev"], s["x_right_prev"], s["x_left_prev"], s["x_i_l"]]
    ).astype(float)
    hi = np.concatenate(
        [s["x_o"], s["x_right"], s["x_left"], s["x_i_r"]]
    ).astype(float)
    start = np.log(events / np.sum(_power_gain(lo, hi, p)))
    if not (np.size(s["sel_to"]) or np.size(s["gap_to"])):
        return log_likelihood(start)
    with ignoring_warnings():
        res = minimize_scalar(
            lambda u: -log_likelihood(u),
            bracket=(start - 1.0, start + 1.0),
            options={"xtol": 1e-10},
        )
    return -float(res.fun)


def half_time(x_unique: np.ndarray, mcf_hat: np.ndarray) -> tuple:
    """The last value of the non-parametric MCF and the time at which it
    reached half of it (interpolated), for a model's starting values.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.recurrent.parametric.software_growth import half_time
    >>> half_time(np.array([1.0, 2.0, 4.0, 8.0]), np.array([1.0, 2, 3, 4]))
    (4.0, 2.0)
    """
    x = np.asarray(x_unique, dtype=float)
    mcf = np.asarray(mcf_hat, dtype=float)
    total = float(mcf[-1])
    t_half = float(np.interp(0.5 * total, mcf, x))
    return total, t_half
