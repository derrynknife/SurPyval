"""Intensity models on another clock: testing effort in place of time,
and a detection rate that changes at a point in time.

Both are one idea. A growth model's curve ``Lambda(t)`` supposes steady
testing; where the testing is not steady, the same curve holds on a clock
``u(t)`` that runs with the testing:

.. math::
    \\Lambda(t) = \\Lambda_0(u(t)), \\qquad
    \\lambda(t) = \\lambda_0(u(t))\\, u'(t).

- :class:`EffortClock`: ``u`` is the cumulative testing effort ``W(t)``
  (tester hours, test cases run), given as a table of effort per period
  or as a curve (Yamada, Ohtera and Narihisa 1986).
- :class:`ChangePoint`: ``u`` is time itself up to ``tau`` and runs at
  ``ratio`` times the speed after it, so every rate of the model is
  multiplied by ``ratio`` from ``tau`` on (for Goel-Okumoto, Zhao's 1993
  change-point model: ``b`` becomes ``ratio * b``). The curve is
  continuous at ``tau`` and keeps its total.

Each wraps any intensity model (``GoelOkumoto``, ``YamadaDelayedS``,
``MusaOkumoto``, ``CrowAMSAA``, ...) and is itself an intensity fitter,
whose ``fit`` returns the ``ParametricRecurrenceModel`` the others do.
"""

from __future__ import annotations

from typing import Any

import autograd.numpy as np
import numpy as onp
from autograd.numpy.numpy_boxes import ArrayBox
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit, minimize, minimize_scalar
from scipy.stats import chi2

from surpyval.recurrent._bounded import unconstraining_maps
from surpyval.recurrent.inference import check_alpha_ci
from surpyval.recurrent.parametric.counting_process import Boxable
from surpyval.utils.rng import as_generator
from surpyval.utils.surpyval_data import refuses_surpyval_data_with
from surpyval.utils.warnings import ignoring_warnings

from .hpp import HPP
from .nhpp_fitter import NHPPFitter
from .software_growth import (
    LIMIT_MARGIN,
    GrowthNHPPFitter,
    _power_law_limit,
    limit_centroids,
)

#: The entries of the likelihood's split of the data that are times
_TIME_KEYS: tuple[str, ...] = (
    "x_o",
    "x_o_prev",
    "x_right",
    "x_right_prev",
    "x_left",
    "x_left_prev",
    "x_i_l",
    "x_i_r",
    "sel_from",
    "sel_to",
    "gap_from",
    "gap_to",
)


# ---------------------------------------------------------------------------
# Testing effort
# ---------------------------------------------------------------------------
class EffortTable:
    """
    Testing effort by period: ``amounts[k]`` of effort (tester hours,
    test cases run) spent evenly over the period that ends at
    ``ends[k]`` (the first starts at 0). After the last period the
    effort goes on at the last period's rate.

    Parameters
    ----------
    ends : array_like
        The end of each period, increasing, each above 0.
    amounts : array_like
        The effort spent in each period: at least 0 (effort cannot be
        taken back), and not all 0.

    Examples
    --------
    >>> from surpyval.recurrent import EffortTable
    >>> effort = EffortTable([1, 2, 3], [10, 30, 20])
    >>> effort.cumulative([0.5, 2.0, 2.5, 4.0]).tolist()
    [5.0, 40.0, 50.0, 80.0]
    >>> effort.rate([0.5, 1.5]).tolist()
    [10.0, 30.0]
    """

    def __init__(self, ends: ArrayLike, amounts: ArrayLike) -> None:
        self.ends = onp.atleast_1d(onp.asarray(ends, dtype=float))
        self.amounts = onp.atleast_1d(onp.asarray(amounts, dtype=float))
        if self.ends.ndim != 1 or self.ends.shape != self.amounts.shape:
            raise ValueError(
                "ends and amounts must be flat and of one length (a "
                "period's end and its effort); got {} and {}".format(
                    self.ends.shape, self.amounts.shape
                )
            )
        starts = onp.concatenate([[0.0], self.ends[:-1]])
        if not onp.all(onp.isfinite(self.ends) & (self.ends > starts)):
            raise ValueError(
                "ends must be the periods' ends in increasing order, the "
                "first above 0; got {}".format(self.ends[:6])
            )
        bad = ~(onp.isfinite(self.amounts) & (self.amounts >= 0))
        if onp.any(bad):
            raise ValueError(
                "amounts must each be at least 0: effort is spent, never "
                "taken back, so cumulative effort cannot fall. Got {} in "
                "period(s) {}. Give the effort of each period, not a "
                "running total.".format(
                    self.amounts[bad][:5], (onp.flatnonzero(bad) + 1)[:5]
                )
            )
        if not self.amounts.sum() > 0:
            raise ValueError("amounts are all 0: no testing was done.")
        self._edges = onp.concatenate([[0.0], self.ends])
        self._total = onp.concatenate([[0.0], onp.cumsum(self.amounts)])
        self._rates = self.amounts / (self.ends - starts)

    def cumulative(self, t: Any) -> Any:
        """The effort spent by time ``t``."""
        t = onp.asarray(t, dtype=float)
        inside = onp.interp(t, self._edges, self._total)
        beyond = onp.maximum(t - self.ends[-1], 0.0) * self._rates[-1]
        return inside + beyond

    def rate(self, t: Any) -> Any:
        """The effort per unit time at ``t`` (that of the period ``t`` is
        in; a time on a boundary belongs to the period it ends)."""
        t = onp.asarray(t, dtype=float)
        k = onp.searchsorted(self.ends, t, side="left")
        return self._rates[onp.minimum(k, self._rates.size - 1)]

    def inverse(self, w: Any) -> Any:
        """The earliest time by which the effort ``w`` has been spent
        (``inf`` for effort never reached)."""
        w = onp.asarray(w, dtype=float)
        k = onp.searchsorted(self._total, w, side="left")
        k = onp.clip(k, 1, self._rates.size)
        with onp.errstate(divide="ignore", invalid="ignore"):
            t = (
                self._edges[k - 1]
                + (w - self._total[k - 1]) / self._rates[k - 1]
            )
            beyond = self.ends[-1] + (w - self._total[-1]) / self._rates[-1]
        t = onp.where(w > self._total[-1], beyond, t)
        return onp.where(w <= 0, 0.0, t)

    def to_spec(self) -> dict:
        return {
            "kind": "EffortTable",
            "ends": self.ends.tolist(),
            "amounts": self.amounts.tolist(),
        }


#: WeibullEffort.fit given event data: its rows are periods of effort.
_NOT_EVENT_DATA = (
    "WeibullEffort.fit takes a table of testing effort, not event data: "
    "ends, the end time of each period, and amounts, the effort spent in "
    "it (as EffortTable(ends, amounts)). To fit failures on an effort "
    "clock, fit the events with EffortClock(base, effort).fit(data)."
)


class WeibullEffort:
    """
    A Weibull-type curve of cumulative testing effort, ``W(t) = N (1 -
    exp(-beta t**m))``: ``N`` the effort that will be spent in all,
    rising slowly, then fast, then levelling off (for ``m > 1``).

    Parameters
    ----------
    N, beta, m : float
        Each above 0.

    Examples
    --------
    >>> from surpyval.recurrent import WeibullEffort
    >>> effort = WeibullEffort(100.0, 0.02, 2.0)
    >>> effort.cumulative([0.0, 5.0, 10.0]).round(2).tolist()
    [0.0, 39.35, 86.47]
    >>> curve = WeibullEffort.fit([2, 4, 6, 8, 10], [8, 20, 24, 18, 9])
    >>> round(curve.N, 1), round(curve.m, 2)
    (84.0, 2.11)
    """

    def __init__(self, N: float, beta: float, m: float) -> None:
        for name, value in (("N", N), ("beta", beta), ("m", m)):
            if not (onp.isfinite(value) and value > 0):
                raise ValueError(
                    f"{name} must be a number above 0; got {value!r}"
                )
        self.N, self.beta, self.m = float(N), float(beta), float(m)

    @classmethod
    @refuses_surpyval_data_with(_NOT_EVENT_DATA, argument="ends")
    def fit(cls, ends: ArrayLike, amounts: ArrayLike) -> WeibullEffort:
        """The curve through a table of effort by period (as
        :class:`EffortTable`'s), by least squares on the cumulative
        effort at the periods' ends."""
        table = EffortTable(ends, amounts)
        total = table._total[1:]
        span = float(table.ends[-1])

        def curve(t: Any, log_n: float, log_beta: float, log_m: float) -> Any:
            return onp.exp(log_n) * -onp.expm1(
                -onp.exp(log_beta) * (t / span) ** onp.exp(log_m)
            )

        with ignoring_warnings():
            fitted, _ = curve_fit(
                curve,
                table.ends,
                total,
                p0=[onp.log(1.2 * total[-1]), 0.5, 0.3],
                maxfev=20000,
            )
        m = float(onp.exp(fitted[2]))
        return cls(
            float(onp.exp(fitted[0])),
            float(onp.exp(fitted[1])) / span**m,
            m,
        )

    def cumulative(self, t: Any) -> Any:
        t = onp.asarray(t, dtype=float)
        return -self.N * onp.expm1(-self.beta * t**self.m)

    def rate(self, t: Any) -> Any:
        t = onp.asarray(t, dtype=float)
        with onp.errstate(divide="ignore", invalid="ignore"):
            return (
                self.N
                * self.beta
                * self.m
                * t ** (self.m - 1.0)
                * onp.exp(-self.beta * t**self.m)
            )

    def inverse(self, w: Any) -> Any:
        share = onp.asarray(w, dtype=float) / self.N
        with onp.errstate(divide="ignore", invalid="ignore"):
            t = (
                -onp.log1p(-onp.where(share < 1, share, 0.0)) / self.beta
            ) ** (1.0 / self.m)
        return onp.where(share < 1, t, onp.inf)

    def to_spec(self) -> dict:
        return {
            "kind": "WeibullEffort",
            "N": self.N,
            "beta": self.beta,
            "m": self.m,
        }


def _effort_from_spec(spec: dict) -> Any:
    if spec["kind"] == "EffortTable":
        return EffortTable(spec["ends"], spec["amounts"])
    return WeibullEffort(spec["N"], spec["beta"], spec["m"])


def _check_base(base: Any, what: str) -> None:
    needed = ("cif", "iif", "log_iif", "inv_cif", "parameter_names", "bounds")
    if isinstance(base, (EffortClock, ChangePoint)) or not all(
        hasattr(base, a) for a in needed
    ):
        raise ValueError(
            "{} takes an intensity model to put on its clock: GoelOkumoto, "
            "YamadaDelayedS, InflectionS, MusaOkumoto, CrowAMSAA, Duane, "
            "CoxLewis or HPP; got {!r}".format(what, base)
        )


class _ClockFitter(NHPPFitter):
    """An intensity model whose time is another's, through a clock."""

    base: Any

    def _clock(self, x: Any, *extra: Any) -> tuple:
        """``(u(x), u'(x))``."""
        raise NotImplementedError

    def _unclock(self, u: Any, *extra: Any) -> Any:
        raise NotImplementedError

    def _split(self, params: tuple) -> tuple:
        k = len(self.base.parameter_names)
        return params[:k], params[k:]

    def cif(self, x: Boxable, *params: Boxable) -> Boxable:
        own, extra = self._split(params)
        return self.base.cif(self._clock(x, *extra)[0], *own)

    def iif(self, x: Boxable, *params: Boxable) -> Boxable:
        own, extra = self._split(params)
        u, speed = self._clock(x, *extra)
        return self.base.iif(u, *own) * speed

    def log_iif(self, x: Boxable, *params: Boxable) -> Boxable:
        own, extra = self._split(params)
        u, speed = self._clock(x, *extra)
        with onp.errstate(divide="ignore"):
            return self.base.log_iif(u, *own) + np.log(speed)

    def inv_cif(self, N: Boxable, *params: Boxable) -> Boxable:
        own, extra = self._split(params)
        return self._unclock(self.base.inv_cif(N, *own), *extra)


class EffortClock(_ClockFitter):
    """
    An intensity model run on cumulative testing effort in place of
    time: ``Lambda(t) = Lambda_0(W(t))``, where ``Lambda_0`` is the
    model's curve and ``W(t)`` the effort spent by ``t``. With
    ``GoelOkumoto`` this is the testing-effort model of Yamada, Ohtera
    and Narihisa, ``a (1 - exp(-b W(t)))``: ``b`` is then the detection
    rate per unit of effort. The model's parameters keep their names and
    meanings, on the effort scale.

    Use it where the testing was not steady (a team that grew, a holiday,
    a final push): the growth models read a week with little testing as a
    week with few faults left.

    Parameters
    ----------
    base : intensity fitter
        ``GoelOkumoto``, ``YamadaDelayedS``, ``InflectionS``,
        ``MusaOkumoto``, ``CrowAMSAA``, ``Duane``, ``CoxLewis`` or
        ``HPP``.
    effort : EffortTable, WeibullEffort or tuple
        The effort spent over time: a table of effort per period (an
        :class:`EffortTable`, or ``(ends, amounts)`` for one), or a
        :class:`WeibullEffort` curve. Effort cannot fall: a negative
        amount is refused.

    The returned object is a fitter: its ``fit`` takes the data the base
    model's does, in time (not effort), and returns a
    ``ParametricRecurrenceModel`` whose ``cif``, ``iif``, bounds,
    ``remaining_faults``, ``time_to_intensity`` and ``release_time`` are
    in time. An exact failure time in a period with no effort has no
    likelihood, and is refused.

    References
    ----------
    Yamada, S., Ohtera, H. and Narihisa, H. (1986), "Software
    reliability growth models with testing-effort", IEEE Transactions on
    Reliability R-35(1), 19-23.

    Examples
    --------
    Failures counted in six weeks of testing, with the tester hours of
    each week:

    >>> import numpy as np
    >>> from surpyval.recurrent import EffortClock, GoelOkumoto
    >>> weeks = np.arange(6.0)
    >>> x = np.column_stack([weeks, weeks + 1])
    >>> hours = [40, 120, 160, 160, 80, 40]
    >>> on_effort = EffortClock(GoelOkumoto, (weeks + 1, hours))
    >>> model = on_effort.fit(x, c=[2] * 6, n=[6, 15, 14, 9, 3, 1])
    >>> model.params.round(4).tolist()
    [58.8584, 0.0028]
    >>> round(float(model.remaining_faults(6.0)), 1)
    10.9
    """

    def __init__(self, base: Any, effort: Any) -> None:
        _check_base(base, "EffortClock")
        if isinstance(effort, (tuple, list)) and len(effort) == 2:
            effort = EffortTable(*effort)
        if not all(
            hasattr(effort, a) for a in ("cumulative", "rate", "inverse")
        ):
            raise ValueError(
                "effort must be an EffortTable, a WeibullEffort or a pair "
                "(ends, amounts) of a table of effort per period; got "
                "{!r}".format(effort)
            )
        self.base = base
        self.effort = effort
        self.name = f"{base.name} on effort"
        self.parameter_names = list(base.parameter_names)
        self.bounds = tuple(base.bounds)
        self.has_scale = getattr(base, "has_scale", False)
        self.support = (0.0, onp.inf)
        self.total_parameter = getattr(base, "total_parameter", None)

    def _repr_name(self) -> str:
        return f"EffortClock({type(self.base).__name__.rstrip('_')})"

    def _clock(self, x: Any, *extra: Any) -> tuple:
        return self.effort.cumulative(x), self.effort.rate(x)

    def _unclock(self, u: Any, *extra: Any) -> Any:
        return self.effort.inverse(u)

    def to_spec(self) -> dict:
        return {
            "kind": "EffortClock",
            "base": self.base.name,
            "effort": self.effort.to_spec(),
        }

    def _check_identified(self, data: Any) -> None:
        s = data.split_for_nhpp_likelihood()
        idle = onp.asarray(self.effort.rate(s["x_o"]), dtype=float) <= 0
        if onp.any(idle):
            raise ValueError(
                "{} failure time(s) fall where the effort's rate is 0 (the "
                "first at {:g}): a failure cannot be found with no "
                "testing, so the likelihood is 0. Check the effort table "
                "against the failure times, or give that period its "
                "effort.".format(int(idle.sum()), float(s["x_o"][idle][0]))
            )
        # The base model's own rule on growth, on the effort clock
        power = getattr(self.base, "_limit_power", None)
        if not isinstance(self.base, GrowthNHPPFitter) or power is None:
            return
        if onp.size(s["sel_to"]) or onp.size(s["gap_to"]):
            return
        seen, limit = limit_centroids(
            self._on_effort(s), power, self.base._limit_moment
        )
        if not seen < limit:
            raise ValueError(self._no_growth(seen, limit))

    def _on_effort(self, s: dict) -> dict:
        """The likelihood's split of the data with its times as effort."""
        return {
            k: (self.effort.cumulative(v) if k in _TIME_KEYS else v)
            for k, v in s.items()
        }

    def _no_growth(self, seen: float | None, limit: float | None) -> str:
        return (
            self.base._no_growth(seen, limit)
            .replace("mean time", "mean effort")
            .replace("mean squared time", "mean squared effort")
            .replace("test for longer", "test for more effort")
            + " (Here the model runs on testing effort, not time: the "
            "failures are compared by the effort spent when each was "
            "found.)"
        )

    def _default_start(self, data: Any, x_unique: Any, mcf_hat: Any) -> Any:
        if not hasattr(self.base, "_default_start"):
            # HPP: the events over the effort, in place of over the time
            last = float(onp.max(x_unique))
            rate = _plain_fit(self.base, data).params
            return rate * last / float(self.effort.cumulative(last))
        return self.base._default_start(
            data, onp.asarray(self.effort.cumulative(x_unique)), mcf_hat
        )

    def fit_from_recurrent_data(
        self,
        data: Any,
        how: str = "MLE",
        init: Any = None,
        frailty: str | None = None,
        cluster: Any = None,
    ) -> Any:
        model = super().fit_from_recurrent_data(
            data, how, init, frailty, cluster
        )
        growth = isinstance(self.base, GrowthNHPPFitter)
        if growth and how == "MLE" and frailty is None and cluster is None:
            # As the model's own fit: a search that ran to the edge of
            # the parameters stops at the likelihood of the model's
            # limit, here a power of the effort.
            s = data.split_for_nhpp_likelihood()
            with onp.errstate(divide="ignore"):
                speeds = onp.sum(onp.log(self.effort.rate(s["x_o"])))
            limits = [
                _power_law_limit(self._on_effort(s), p) + speeds
                for p in self.base._limit_powers()
            ]
            fitted = _log_likelihood(model)
            if limits and not fitted > max(limits) + LIMIT_MARGIN:
                raise ValueError(self._no_growth(None, None))
        return model


# ---------------------------------------------------------------------------
# A change point
# ---------------------------------------------------------------------------
#: The fewest events on each side of a change point that is estimated,
#: and the most candidate times its profile is first looked at on.
MIN_SIDE = 3
MAX_CANDIDATES = 200
REFINED = 3


class ChangePoint(_ClockFitter):
    """
    An intensity model whose rates change at a time ``tau``: the model's
    clock runs at ``ratio`` times its speed from ``tau`` on,

    .. math::
        \\Lambda(t) = \\Lambda_0(t) \\text{ for } t \\le \\tau, \\qquad
        \\Lambda(t) = \\Lambda_0(\\tau + \\rho\\,(t - \\tau))
        \\text{ after},

    with ``rho`` the parameter ``ratio``. For ``GoelOkumoto`` the
    detection rate ``b`` becomes ``ratio * b`` after ``tau`` (Zhao's
    change-point model); the total number of faults is unchanged and the
    curve is continuous at ``tau``. ``ratio = 1`` is no change. Use it
    for a known change in the testing (a new team, a new test strategy),
    or to look for one.

    Parameters
    ----------
    base : intensity fitter
        ``GoelOkumoto``, ``YamadaDelayedS``, ``InflectionS``,
        ``MusaOkumoto``, ``CrowAMSAA``, ``Duane``, ``CoxLewis`` or
        ``HPP``.
    tau : float, optional
        The time of the change. By default it is estimated: the profile
        likelihood (which jumps at each failure and is smooth between
        them) is worked out for a change just after each failure time
        that leaves at least 3 failures on each side (up to 200 of them,
        evenly spread), and searched within the stretch up to the next
        failure for the best 3. Estimating it needs exact failure times
        (not counts in periods) of one system.

    The returned object is a fitter with the base model's parameters and
    ``ratio``; its ``fit`` returns a ``ParametricRecurrenceModel``, which
    carries ``change_point`` (a dict with ``tau``, and for an estimated
    ``tau`` the candidate times, their log-likelihoods and the
    log-likelihood with no change, ``None`` where the model has no fit
    without one). An estimated ``tau`` is not a parameter of the fitted
    model: its ``covariance()``, Wald bounds and ``aic()`` take ``tau``
    as given, so the bounds are narrower, and the ``aic()`` lower by 2,
    than they would be with ``tau`` counted.
    :func:`change_point_cb` bounds an estimated ``tau`` and
    :func:`change_point_test` tests for no change.

    References
    ----------
    Zhao, M. (1993), "Change-point problems in software and hardware
    reliability", Communications in Statistics: Theory and Methods
    22(3), 757-768.

    Examples
    --------
    >>> from surpyval.recurrent import (
    ...     ChangePoint, GoelOkumoto, change_point_cb)
    >>> truth = ChangePoint(GoelOkumoto, tau=30.0).from_params(
    ...     [80, 0.01, 4.0])
    >>> truth.cif([30.0, 60.0]).round(2)
    array([20.73, 62.15])
    >>> d = truth.time_terminated_simulation_data(100.0, random_state=1)
    >>> model = ChangePoint(GoelOkumoto).fit(d.x, c=d.c)
    >>> round(model.change_point["tau"], 1)
    30.4
    >>> model.params.round(3).tolist()
    [88.937, 0.009, 4.577]
    >>> change_point_cb(model).round(1).tolist()
    [28.9, 30.4]
    """

    def __init__(self, base: Any, tau: float | None = None) -> None:
        _check_base(base, "ChangePoint")
        if tau is not None and not (onp.isfinite(tau) and tau > 0):
            raise ValueError(
                f"tau is the time of the change, above 0; got {tau!r}"
            )
        self.base = base
        self.tau = None if tau is None else float(tau)
        self.name = f"{base.name} with a change point"
        self.parameter_names = [*base.parameter_names, "ratio"]
        self.bounds = (*base.bounds, (0, None))
        self.has_scale = getattr(base, "has_scale", False)
        self.support = (0.0, onp.inf)
        self.total_parameter = getattr(base, "total_parameter", None)

    def _repr_name(self) -> str:
        return f"ChangePoint({type(self.base).__name__.rstrip('_')})"

    def _tau(self) -> float:
        if self.tau is None:
            raise ValueError(
                "This ChangePoint has no tau (it is to be estimated by "
                "fit): give ChangePoint(model, tau=...) to use its "
                "functions or from_params."
            )
        return self.tau

    def _clock(self, x: Any, *extra: Any) -> tuple:
        tau, ratio = self._tau(), extra[0]
        after = x > tau
        # autograd's where only where a derivative is being taken
        boxed = isinstance(x, ArrayBox) or isinstance(ratio, ArrayBox)
        where = np.where if boxed else onp.where
        return (
            where(after, tau + ratio * (x - tau), x),
            where(after, ratio, 1.0),
        )

    def _unclock(self, u: Any, *extra: Any) -> Any:
        tau, ratio = self._tau(), extra[0]
        return np.where(u > tau, tau + (u - tau) / ratio, u)

    def to_spec(self) -> dict:
        return {"kind": "ChangePoint", "base": self.base.name, "tau": self.tau}

    def from_params(self, params: ArrayLike, **kwargs: Any) -> Any:
        self._tau()
        model = super().from_params(params, **kwargs)
        model.change_point = {"tau": self.tau}
        return model

    def _default_start(self, data: Any, x_unique: Any, mcf_hat: Any) -> Any:
        return onp.append(_base_start(self.base, data, x_unique, mcf_hat), 1.0)

    def _closed_form_mle(self, data: Any) -> onp.ndarray | None:
        """The exact MLE of ``ChangePoint(HPP, tau)``, as ``HPP`` has its
        own: on each side of ``tau`` the failures over the time observed
        there,

        .. math::
            \\hat\\lambda = \\frac{n_{\\le\\tau}}{E_{\\le\\tau}}, \\qquad
            \\hat\\rho = \\frac{n_{>\\tau} / E_{>\\tau}}{\\hat\\lambda},

        the likelihood being ``n log(lambda) + n_> log(rho) - lambda
        (E_<= + rho E_>)``. The search agreed with it to only about 1e-5.
        ``None`` for any other base model, and for counts in periods or
        right truncation, whose terms do not split at ``tau`` this way
        (they are searched)."""
        if not isinstance(self.base, type(HPP)):
            return None
        s = data.split_for_nhpp_likelihood()
        if any(
            onp.size(s[k]) for k in ("x_left", "x_i_l", "sel_to", "gap_to")
        ):
            return None
        tau = self._tau()
        ends = (s["x_o"], s["x_right"])
        starts = (s["x_o_prev"], s["x_right_prev"])

        def exposure(part: Any) -> float:
            return float(
                sum(
                    onp.sum(part(onp.asarray(b, dtype=float)))
                    - onp.sum(part(onp.asarray(a, dtype=float)))
                    for a, b in zip(starts, ends)
                )
            )

        before = exposure(lambda t: onp.minimum(t, tau))
        after = exposure(lambda t: onp.maximum(t - tau, 0.0))
        n_after = int(onp.sum(onp.asarray(s["x_o"]) > tau))
        n_before = onp.size(s["x_o"]) - n_after
        if not (n_before and n_after and before > 0 and after > 0):
            return None
        rate = n_before / before
        return onp.array([rate, (n_after / after) / rate])

    def _limit_log_likelihood(self, data: Any, ratio: float) -> float:
        """The greatest log-likelihood of the limit of the model at the
        edge of its parameters: the base model's limit, ``c u**p``, on
        this clock, ``u`` running ``ratio`` times as fast after
        ``tau`` (for ``p = 1``, a constant rate that changes at
        ``tau``). Maximised over ``c`` in closed form and over the
        ratio by a search from ``ratio``."""
        s = data.split_for_nhpp_likelihood()
        late = int(onp.sum(s["x_o"] > self._tau()))
        best = -onp.inf
        for p in self.base._limit_powers():

            def loss(log_ratio: float, p: int = p) -> float:
                rho = float(onp.exp(log_ratio))
                clocked = {
                    k: (self._clock(v, rho)[0] if k in _TIME_KEYS else v)
                    for k, v in s.items()
                }
                with onp.errstate(all="ignore"):
                    value = _power_law_limit(clocked, p) + late * log_ratio
                return -value if onp.isfinite(value) else 1e300

            start = float(onp.log(ratio))
            res = minimize_scalar(
                loss, bracket=(start - 0.5, start + 0.5), tol=1e-10
            )
            best = max(best, -float(min(res.fun, loss(start))))
        return best

    def _check_beats_limit(self, data: Any, model: Any) -> None:
        """Refuse a fit that does not beat the model's limit, as the
        growth models do: the search then ran to the edge of the
        parameters, and stopped anywhere."""
        if not isinstance(self.base, GrowthNHPPFitter):
            return
        if not self.base._limit_powers():
            return
        limit = self._limit_log_likelihood(data, float(model.params[-1]))
        if _log_likelihood(model) > limit + LIMIT_MARGIN:
            return
        raise ValueError(
            self.base._no_growth(None, None)
            + " (With the change at tau = {:g}: the limit is {} on each "
            "side of it, at different levels.{})".format(
                self._tau(),
                self.base._limit_name,
                (
                    " ChangePoint(HPP, tau=...) fits that: a constant "
                    "rate that changes at tau."
                    if self.base._limit_powers() == (1,)
                    else ""
                ),
            )
        )

    def _check_identified(self, data: Any) -> None:
        """Refuse a ``tau`` with no failure on one of its sides: the
        likelihood then has no maximum in ``ratio``."""
        tau = self._tau()
        s = data.split_for_nhpp_likelihood()
        counted = [
            (s["x_left_prev"], s["x_left"], s["n_left"]),
            (s["x_i_l"], s["x_i_r"], s["n_i"]),
        ]
        before = int(onp.sum(s["x_o"] <= tau))
        after = int(onp.sum(s["x_o"] > tau))
        for lo, hi, n in counted:
            n = onp.asarray(n, dtype=float)
            before += int(onp.sum(n[onp.asarray(lo, dtype=float) < tau]))
            after += int(onp.sum(n[onp.asarray(hi, dtype=float) > tau]))
        if before and after:
            return
        last = max(
            (float(onp.max(s[k])) for k in _TIME_KEYS if onp.size(s[k])),
            default=0.0,
        )
        raise ValueError(
            "tau = {:g} has no failure {} it (the data run to {:g}), so "
            "the change in the rate, ratio, has no estimate: {}. Give a "
            "tau with failures on both sides, or fit the {} model "
            "itself.".format(
                tau,
                "after" if before else "before",
                last,
                (
                    "a tau beyond the data leaves the model as it is"
                    if tau >= last
                    else "the likelihood keeps rising as ratio goes to "
                    + ("0" if before else "infinity")
                ),
                self.base.name,
            )
        )

    def fit_from_recurrent_data(
        self,
        data: Any,
        how: str = "MLE",
        init: Any = None,
        frailty: str | None = None,
        cluster: Any = None,
    ) -> Any:
        if frailty is not None or cluster is not None:
            raise ValueError(
                "A change-point fit takes no frailty: fit the model "
                "itself with frailty=, or the change point without it."
            )
        if self.tau is not None:
            model = super().fit_from_recurrent_data(data, how, init)
            if how == "MLE":
                self._check_beats_limit(data, model)
            model.change_point = {"tau": self.tau}
            return model
        if how != "MLE":
            raise ValueError(
                "tau is estimated by maximum likelihood: how='MSE' needs "
                "ChangePoint(model, tau=...)."
            )
        return _estimate_tau(self.base, data, init)


def _candidates(data: Any) -> onp.ndarray:
    """The failure times a change point is tried at: every one that
    leaves ``MIN_SIDE`` failures on each side (evenly thinned to
    ``MAX_CANDIDATES`` of them where there are more)."""
    if onp.ndim(data.x) != 1 or onp.any((data.c != 0) & (data.c != 1)):
        raise ValueError(
            "Estimating tau needs each failure's time: the data are "
            "counts in periods. Give tau (ChangePoint(model, tau=...)), "
            "at the boundary of two periods."
        )
    if onp.unique(data.i).size != 1:
        raise ValueError(
            "Estimating tau is for one system's failures; the data have "
            "{} systems. Give tau, or fit each system.".format(
                onp.unique(data.i).size
            )
        )
    events = onp.unique(onp.asarray(data.x, dtype=float)[data.c == 0])
    if events.size < 2 * MIN_SIDE + 1:
        raise ValueError(
            "Estimating tau needs at least {} failures at different "
            "times ({} on each side of the change); the data have "
            "{}.".format(2 * MIN_SIDE + 1, MIN_SIDE, events.size)
        )
    inner = events[MIN_SIDE - 1 : events.size - MIN_SIDE]
    if inner.size > MAX_CANDIDATES:
        pick = onp.unique(
            onp.round(onp.linspace(0, inner.size - 1, MAX_CANDIDATES))
        ).astype(int)
        inner = inner[pick]
    return inner


class _Profile:
    """The log-likelihood of a change-point model maximised over its
    parameters at a given ``tau``: a plain search from given starts,
    without the checks of a fit (which the fit at the chosen ``tau``
    gets)."""

    def __init__(self, base: Any, data: Any, plain: Any) -> None:
        self.base = base
        s = data.split_for_nhpp_likelihood()
        if onp.size(s["sel_to"]) or onp.size(s["gap_to"]):
            raise ValueError(
                "Estimating tau is not available with right truncation "
                "(tr): give tau (ChangePoint(model, tau=...))."
            )
        # Failure times and the ends of observation only (_candidates):
        # log L = sum(log iif(x_o)) + sum(sign * cif(time))
        self.x_o = s["x_o"]
        self.times = onp.concatenate(
            [s["x_o_prev"], s["x_o"], s["x_right_prev"], s["x_right"]]
        )
        self.signs = onp.concatenate(
            [
                onp.ones_like(s["x_o_prev"]),
                -onp.ones_like(s["x_o"]),
                onp.ones_like(s["x_right_prev"]),
                -onp.ones_like(s["x_right"]),
            ]
        )
        bounds = [*base.bounds, (0, None)]
        self.to_natural, self.to_search = unconstraining_maps(bounds)
        #: No change: the model's own fit, at ratio 1
        self.null = onp.append(onp.asarray(plain, dtype=float), 1.0)

    def log_likelihood(self, fitter: Any, params: Any) -> float:
        with onp.errstate(all="ignore"):
            value = onp.sum(fitter.log_iif(self.x_o, *params)) + onp.sum(
                self.signs * fitter.cif(self.times, *params)
            )
        return float(value) if onp.isfinite(value) else -onp.inf

    def at(self, tau: float, start: Any = None) -> tuple:
        """``(log-likelihood, parameters)`` at ``tau``: searched from
        ``start``, and from no change where that is not given or ends
        below it (no change is in the model at every ``tau``)."""
        fitter = ChangePoint(self.base, tau=tau)

        def loss(u: onp.ndarray) -> float:
            with onp.errstate(all="ignore"):
                value = -self.log_likelihood(fitter, self.to_natural(u))
            return value if onp.isfinite(value) else 1e300

        best: tuple = (self.log_likelihood(fitter, self.null), self.null)
        for begin in (start, self.null):
            if begin is None or (begin is self.null and best[1] is not begin):
                continue
            res = minimize(
                loss,
                self.to_search(onp.asarray(begin, dtype=float)),
                method="Nelder-Mead",
                options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 2000},
            )
            if -res.fun > best[0]:
                best = (-float(res.fun), onp.asarray(self.to_natural(res.x)))
        return best

    def search(self, data: Any, init: Any = None) -> tuple:
        """The best ``tau``: the profile at each candidate failure time
        (a change just after that failure), then, the profile being
        smooth only between failures, a search within the stretch up to
        the next failure for the best few. Returns ``(tau, parameters,
        log-likelihood, candidates, their log-likelihoods)``."""
        taus = _candidates(data)
        events = onp.unique(onp.asarray(data.x, dtype=float)[data.c == 0])
        values, found = [], []
        start = init
        for tau in taus:
            value, params = self.at(float(tau), start)
            values.append(value)
            found.append(params)
            start = params
        if not onp.any(onp.isfinite(values)):
            raise ValueError(
                "No change point could be fitted: the {} model has no "
                "finite likelihood at any candidate time. Fit the model "
                "without a change point to see why.".format(self.base.name)
            )
        k = int(onp.argmax(values))
        best = (float(taus[k]), found[k], values[k])
        for j in onp.argsort(values)[::-1][:REFINED]:
            lo = float(taus[j])
            hi = float(events[onp.searchsorted(events, lo, side="right")])

            def loss(tau: float, j: int = int(j)) -> float:
                return -self.at(tau, found[j])[0]

            res = minimize_scalar(
                loss,
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 1e-4 * (hi - lo)},
            )
            if -res.fun > best[2]:
                value, params = self.at(float(res.x), found[j])
                best = (float(res.x), params, value)
        return (*best, onp.asarray(taus, dtype=float), onp.asarray(values))


def _log_likelihood(model: Any) -> float:
    return -float(model._neg_ll(onp.asarray(model.params, dtype=float)))


def _base_start(base: Any, data: Any, x_unique: Any, mcf_hat: Any) -> Any:
    """Where the base model's own search starts (its fit, for HPP,
    which has no search)."""
    if hasattr(base, "_default_start"):
        return base._default_start(data, x_unique, mcf_hat)
    return onp.asarray(_plain_fit(base, data).params, dtype=float)


def _plain_fit(base: Any, data: Any) -> Any:
    with ignoring_warnings():
        return base.fit_from_recurrent_data(data)


def _estimate_tau(base: Any, data: Any, init: Any) -> Any:
    _candidates(data)
    try:
        plain = _plain_fit(base, data)
        start, no_change = plain.params, _log_likelihood(plain)
    except ValueError:
        # The model has no fit without a change (a detection rate that
        # rose can leave no growth overall): start as its fit does.
        x_unique, r, d = data.to_xrd()
        start = _base_start(base, data, x_unique, onp.cumsum(d / r))
        no_change = None
    if init is not None and onp.size(init) != len(base.parameter_names) + 1:
        raise ValueError(
            "init must have {} values ({}, ratio); got {}.".format(
                len(base.parameter_names) + 1,
                ", ".join(base.parameter_names),
                onp.size(init),
            )
        )
    profile = _Profile(base, data, start)
    tau, params, _, taus, values = profile.search(data, init)
    # the fit at the chosen time, with a fit's checks and warnings
    chosen = ChangePoint(base, tau=tau)
    model = NHPPFitter.fit_from_recurrent_data(chosen, data, "MLE", params)
    chosen._check_beats_limit(data, model)
    model.change_point = {
        "tau": tau,
        "candidates": taus,
        "log_likelihood": values,
        "estimated": True,
        "no_change_log_likelihood": no_change,
    }
    return model


def _estimated(model: Any, what: str) -> dict:
    info = getattr(model, "change_point", None)
    if not info or not info.get("estimated") or not hasattr(model, "data"):
        raise ValueError(
            "{} is for a model fitted with its change point estimated: "
            "ChangePoint(model).fit(x) with no tau. This model's tau was "
            "given (or it was not fitted to data).".format(what)
        )
    return info


def change_point_cb(model: Any, alpha_ci: float = 0.05) -> onp.ndarray:
    """
    Likelihood-ratio bounds on an estimated change point: the earliest
    and the latest of the candidate times (and the estimate) whose
    profile log-likelihood is within half the chi-squared quantile of
    the best. They are read from the failure times the fit looked at, so
    they are as fine as the failures are close. A change point's
    likelihood is not smooth, so the level is approximate; and the times
    within the cut-off need not be one stretch (a second, weaker
    candidate for the change widens the bounds to take it in):
    ``model.change_point`` has the ``candidates`` and their
    ``log_likelihood`` to look at.

    Parameters
    ----------
    model : ParametricRecurrenceModel
        A fit of ``ChangePoint(base).fit(x)`` (``tau`` estimated).
    alpha_ci : float, optional
        The total tail probability. Default 0.05.

    Returns
    -------
    numpy.ndarray
        ``[earliest, latest]``.

    Examples
    --------
    >>> from surpyval.recurrent import (
    ...     ChangePoint, GoelOkumoto, change_point_cb)
    >>> truth = ChangePoint(GoelOkumoto, tau=30.0).from_params(
    ...     [80, 0.01, 4.0])
    >>> d = truth.time_terminated_simulation_data(100.0, random_state=1)
    >>> model = ChangePoint(GoelOkumoto).fit(d.x, c=d.c)
    >>> change_point_cb(model).round(1).tolist()
    [28.9, 30.4]
    """
    check_alpha_ci(alpha_ci)
    info = _estimated(model, "change_point_cb")
    best = _log_likelihood(model)
    inside = info["log_likelihood"] >= best - 0.5 * chi2.ppf(1 - alpha_ci, 1)
    times = onp.append(info["candidates"][inside], info["tau"])
    return onp.array([times.min(), times.max()])


def change_point_test(
    model: Any, n_boot: int = 99, random_state: Any = None
) -> dict:
    """
    A test of "no change" against a change at an unknown time.

    The statistic is twice the gain in log-likelihood of the
    change-point fit over the model without one. With the time of the
    change estimated it does not follow a chi-squared distribution (the
    time is not defined when there is no change), so its p-value is by
    parametric bootstrap: data are simulated from the fit *without* a
    change over the same observation, both models are fitted again as
    they were to the data (the change point estimated afresh), and the
    p-value is the share of simulated statistics at least as large as
    the one observed, counting the observed one.

    Where the model has no fit without a change (a growth model on
    failures that, taken together, show no growth), there is nothing to
    test against, and the call is refused saying so.

    Parameters
    ----------
    model : ParametricRecurrenceModel
        A fit of ``ChangePoint(base).fit(x)`` (``tau`` estimated).
    n_boot : int, optional
        The simulated data sets. Default 99, so the p-value is a
        multiple of 0.01. Each takes as long as the fit did.
    random_state : int or numpy.random.Generator, optional
        The seed, for a reproducible p-value.

    Returns
    -------
    dict
        ``"statistic"``, ``"p_value"``, ``"n_boot"`` (the simulated data
        sets used), ``"skipped"`` (those on which a fit was refused, left
        out), ``"tau"`` and ``"ratio"``.

    Examples
    --------
    The NTDS failures (Jelinski and Moranda 1972): a constant rate, or
    one that changed?

    >>> import numpy as np
    >>> from surpyval.recurrent import HPP, ChangePoint, change_point_test
    >>> days = np.cumsum([9, 12, 11, 4, 7, 2, 5, 8, 5, 7, 1, 6, 1,
    ...                   9, 4, 1, 3, 3, 6, 1, 11, 33, 7, 91, 2, 1])
    >>> model = ChangePoint(HPP).fit(days)
    >>> model.change_point["tau"]
    105.0
    >>> out = change_point_test(model, n_boot=19, random_state=1)
    >>> round(out["statistic"], 2), out["p_value"]
    (13.15, 0.05)
    """
    from surpyval.recurrent import diagnostics

    _estimated(model, "change_point_test")
    if not (isinstance(n_boot, (int, onp.integer)) and n_boot >= 19):
        raise ValueError(
            "n_boot is the number of simulated data sets, a whole number "
            f"of at least 19; got {n_boot!r}"
        )
    base, data = model.dist.base, model.data
    try:
        plain = _plain_fit(base, data)
    except ValueError as err:
        raise ValueError(
            "There is no fit without a change to test against: the {} "
            "model itself is refused on these data ({}) So the data are "
            "not this model's with no change; whether the change-point "
            "model describes them is for aic() against other models (HPP, "
            "CrowAMSAA) and compare_predictions.".format(
                base.name, str(err).split(": ", 1)[0] + "."
            )
        ) from err
    statistic = 2.0 * (_log_likelihood(model) - _log_likelihood(plain))
    rng = as_generator(random_state)
    windows = diagnostics._per_item_windows(data)

    def mass(item: Any, entry: float, close: float) -> tuple:
        start = float(plain.cif(entry))
        return start, float(plain.cif(close)) - start

    larger, used, skipped = 0, 0, 0
    for _ in range(n_boot):
        sim = diagnostics._simulate_windows(
            rng, windows, mass, lambda item: plain.inv_cif
        )
        try:
            null = _plain_fit(base, sim)
            changed = _Profile(base, sim, null.params).search(sim)[2]
        except ValueError:
            skipped += 1
            continue
        used += 1
        larger += 2.0 * (changed - _log_likelihood(null)) >= statistic
    return {
        "statistic": float(statistic),
        "p_value": (1 + larger) / (1 + used),
        "n_boot": used,
        "skipped": skipped,
        "tau": float(model.change_point["tau"]),
        "ratio": float(model.params[-1]),
    }


def clock_from_spec(spec: dict) -> Any:
    """The clock fitter a saved model names (``to_spec``)."""
    from surpyval.recurrent.serialisation import intensity_dist_by_name

    base = intensity_dist_by_name(spec["base"])
    if spec["kind"] == "EffortClock":
        return EffortClock(base, _effort_from_spec(spec["effort"]))
    if spec["kind"] == "ChangePoint":
        return ChangePoint(base, tau=spec["tau"])
    raise ValueError(f"Unknown clock {spec['kind']!r} in a saved model")
