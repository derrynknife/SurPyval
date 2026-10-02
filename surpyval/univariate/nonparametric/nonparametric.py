from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.distribution import NonParametricDistribution
from surpyval.serialisation import SerialisableMixin, stamp_schema
from surpyval.utils.data_summary import data_summary
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape

from ._bands import BandsMixin
from ._support import (
    check_support,
    interp_bound,
    interp_function,
    on_support,
    support_from_dict,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Round-off allowed when ``qf`` compares the estimated CDF with p (see
# there); the absolute tolerance ``_snap`` gives values of order one.
_QF_TOL = 1e-9

# The functions ``cb`` can bound ('R' and 'F' are aliases of 'sf' and 'ff').
_CB_ON = ("sf", "ff", "Hf", "R", "F")
_BOUNDS = ("two-sided", "upper", "lower")


# The ``interp`` values: the step estimate, and the interpolation kinds
# of ``interp_function`` ('cubic' is PCHIP, the rest scipy's interp1d).
_INTERP: tuple[str, ...] = ("step", "linear", "cubic", "nearest")
_INTERP += ("nearest-up", "zero", "slinear", "quadratic", "previous", "next")


def _check_option(name: str, value: Any, accepted: tuple) -> None:
    """Refuse an option ``value`` not in ``accepted``, naming the
    argument and the values it takes (principle 2) -- the one message
    for an unknown option value."""
    if not isinstance(value, str) or value not in accepted:
        raise ValueError(
            "'{}' must be one of {}; got {!r}".format(name, accepted, value)
        )


def _check_bound(bound: str) -> None:
    # An unknown ``bound`` (e.g. 'both') used to reach the statistic's
    # if/elif chain and fail as an UnboundLocalError.
    _check_option("bound", bound, _BOUNDS)


def _check_interp(interp: str) -> None:
    # An unknown ``interp`` used to reach scipy, which raised
    # NotImplementedError (#416).
    _check_option("interp", interp, _INTERP)


class NonParametric(BandsMixin, SerialisableMixin, NonParametricDistribution):
    """
    Result of ``.fit()`` method for every non-parametric
    surpyval distribution. This means that each of the
    methods in this class can be called with a model created
    from the ``NelsonAalen``, ``KaplanMeier``,
    ``FlemingHarrington``, or ``Turnbull`` estimators.

    Examples
    --------
    Ten items, two of them censored (``c=1``):

    >>> import numpy as np
    >>> from surpyval import KaplanMeier
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> c = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    >>> model = KaplanMeier.fit(x, c)
    >>> model
    Non-Parametric SurPyval Model
    =============================
    Model            : Kaplan-Meier
    Data             : 10 units: 8 events at 8 unique times, 2 right censored
    >>> model.sf([2.5, 6]).round(4)
    array([0.8   , 0.4571])
    >>> model.cb([2.5, 6]).round(4)
    array([[0.4087, 0.9459],
           [0.143 , 0.7298]])
    """

    # Attributes populated by the fitter (``NonParametricFitter.fit`` /
    # ``from_xrd`` / ``fit_from_ecdf``). Declared here so static type
    # checkers know their types.
    x: npt.NDArray
    r: npt.NDArray
    d: npt.NDArray
    R: npt.NDArray
    F: npt.NDArray
    H: npt.NDArray
    model: str
    greenwood: npt.NDArray
    data: dict[str, Any]
    #: The ``(lower, upper)`` interval the estimate is defined on, set by
    #: :meth:`set_support`; ``None`` (the default) when it has not been set.
    support: "tuple[float, float] | None" = None
    # The sample size of ``band`` as ``to_dict`` stored it ("band_n"), for
    # a model restored without its data; see ``_band_sample_size``.
    _band_n: "float | None" = None
    # The printout's "Data" line of a model restored without its data
    # (#508).
    _data_summary: "str | None" = None

    def __repr__(self) -> str:
        out = (
            "Non-Parametric SurPyval Model"
            + "\n============================="
            + "\nModel            : {dist}"
        ).format(dist=self.model)

        if hasattr(self, "data"):
            if "estimator" in self.data:
                out += "\nEstimator        : {turnbull}".format(
                    turnbull=self.data["estimator"]
                )
        data_line = self._data_repr()
        if data_line:
            out += "\nData             : " + data_line

        return out

    def _data_repr(self) -> str:
        """The data the estimate was fitted to, in one line, for the
        printout (#508): units weighted by ``n``, by kind of censoring and
        truncation. A model restored without its data gives the line it
        was saved with."""
        data = getattr(self, "data", None)
        if not isinstance(data, dict) or "c" not in data or "x" not in data:
            return self._data_summary or ""
        x = np.asarray(data["x"], dtype=float)
        t = np.asarray(data.get("t", np.empty((0, 2))), dtype=float)
        # Times are non-negative in practice, where a truncation at 0
        # truncates nothing (as for a parametric model on (0, inf)).
        finite = x[np.isfinite(x)]
        lower = 0.0 if finite.size and finite.min() >= 0 else -np.inf
        if t.ndim != 2 or len(t) != len(np.asarray(data["c"])):
            return data_summary(data["c"], data.get("n"), x=x)
        return data_summary(
            data["c"], data.get("n"), t[:, 0], t[:, 1], lower=lower, x=x
        )

    def set_support(self, lower: float, upper: float) -> "NonParametric":
        r"""
        Give the estimate an explicit support, ``[lower, upper]``.

        Without one, the estimate says nothing outside the observed
        values, and what its functions give there is a convention: the
        step functions start at their initial value and hold their last
        one, while the interpolated forms (``interp="linear"`` and so on)
        and the confidence bounds are NaN. With a support set, every
        function and every ``interp`` gives

        - in ``[lower, x[0])``, the value before the first observed value:
          ``sf`` 1, and ``ff``, ``Hf``, ``hf`` and ``df`` 0;
        - in ``(x[-1], upper]``, the value at the last observed value,
          carried (``hf`` and ``df`` keep their convention of the step
          containing x, the last one);
        - outside ``[lower, upper]``, NaN.

        ``cb`` (and ``R_cb`` and ``bootstrap_cb``) follow the same rule,
        collapsing onto the estimate's initial value before the first
        observed value and carrying the bounds at the last one. ``band``
        is unchanged: it is defined only between the first and last
        observed events. The bounds are kept by ``to_dict``.

        The variable need not be time: ``lower`` may be negative, and
        either bound may be infinite (``-np.inf`` or ``np.inf`` for no
        bound on that side).

        Parameters
        ----------

        lower : float
            The lower end of the support; at most the first observed
            value, ``x[0]``.
        upper : float
            The upper end of the support; at least the last observed
            value, ``x[-1]``, and above ``lower``.

        Returns
        -------

        model : NonParametric
            The model itself, so the call can be chained.

        Raises
        ------

        ValueError
            If a bound is NaN or not a number, ``lower`` is not below
            ``upper``, or the bounds do not contain the observed values.

        Examples
        --------
        >>> from surpyval import KaplanMeier
        >>> model = KaplanMeier.fit([1, 2, 3, 4, 5], c=[0, 0, 1, 0, 1])
        >>> model.sf([0.5, 3, 8], interp="linear")
        array([nan, 0.6, nan])
        >>> model = model.set_support(0, 10)
        >>> model.support
        (0.0, 10.0)
        >>> model.sf([-1, 0.5, 3, 8, 11], interp="linear")
        array([nan, 1. , 0.6, 0.3, nan])
        """
        self.support = check_support(
            lower,
            upper,
            float(self.x[0]),
            float(self.x[-1]),
            ("the first value of the estimate", "its last value"),
        )
        return self

    def _within_support(
        self,
        x: npt.ArrayLike,
        f: Callable[[npt.ArrayLike], npt.ArrayLike],
        start: float,
    ) -> npt.NDArray:
        """``f(x)``, restricted to the support when one is set (see
        :meth:`set_support` and :func:`on_support`)."""
        if self.support is None:
            return np.asarray(f(x))
        return on_support(
            self.support, float(self.x[0]), float(self.x[-1]), x, f, start
        )

    def _bounds_within_support(
        self,
        x: npt.ArrayLike,
        f: Callable[[npt.ArrayLike], npt.ArrayLike],
        start: float,
    ) -> npt.NDArray:
        """The confidence bounds ``f(x)``, as :meth:`_within_support`
        gives them with a support set, and without one NaN outside the
        observed values (and at a missing x). ``cb``, ``R_cb`` and
        ``bootstrap_cb`` all go through here so that they agree outside
        the data: ``bootstrap_cb`` used to carry its step convention there
        (1 before the first value, the last bounds after it; #452)."""
        first, last = float(self.x[0]), float(self.x[-1])
        support = (first, last) if self.support is None else self.support
        return on_support(support, first, last, x, f, start)

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, interp: str = "step") -> npt.NDArray:
        r"""

        Survival (or Reliability) function with the
        non-parametric estimates from the data.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the survival
            function will be calculated.

        interp : str, optional
            How to evaluate between the estimate's time points: ``"step"``
            (the default, the right-continuous step function), ``"linear"``,
            ``"cubic"`` (a monotone PCHIP curve) or one of the other
            string kinds of :func:`scipy.interpolate.interp1d`
            (``"nearest"``, ``"nearest-up"``, ``"zero"``, ``"slinear"``,
            ``"quadratic"``, ``"previous"``, ``"next"``); any other value
            raises a ``ValueError``. The interpolated forms return NaN
            outside the range of the data, unless the model has bounds
            (see ``set_support``).

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the survival function at each x


        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.sf(2)
        np.float64(0.6376281516217733)
        >>> model.sf([1., 1.5, 2., 2.5])
        array([0.81873075, 0.81873075, 0.63762815, 0.63762815])
        """
        _check_interp(interp)
        return self._within_support(x, lambda q: self._sf(q, interp), 1.0)

    def _sf(self, x: npt.ArrayLike, interp: str) -> npt.NDArray:
        # ``sf`` without the support (see ``set_support``).
        x = np.atleast_1d(x)
        idx = np.argsort(x)
        rev = np.argsort(idx)
        x = x[idx]
        if interp == "step":
            idx = np.searchsorted(self.x, x, side="right") - 1
            R = self.R[idx]
            R = np.where(idx < 0, 1, R)
            R = np.where(np.isposinf(x), 0, R)
            # A missing time sorts past the last step; it has no value.
            R = np.where(np.isnan(x), np.nan, R)
        else:
            R = interp_function(self.x, self.R, kind=interp)(x)

        R = R[rev]
        return R

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, interp: str = "step") -> npt.NDArray:
        r"""

        CDF (failure or unreliability) function with the
        non-parametric estimates from the data

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which
            the failure function will be calculated.

        interp : str, optional
            How to evaluate between the estimate's time points: ``"step"``
            (the default, the right-continuous step function), ``"linear"``,
            ``"cubic"`` (a monotone PCHIP curve) or one of the other
            string kinds of :func:`scipy.interpolate.interp1d`
            (``"nearest"``, ``"nearest-up"``, ``"zero"``, ``"slinear"``,
            ``"quadratic"``, ``"previous"``, ``"next"``); any other value
            raises a ``ValueError``. The interpolated forms return NaN
            outside the range of the data, unless the model has bounds
            (see ``set_support``).

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at each x


        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.ff(2)
        np.float64(0.36237184837822667)
        >>> model.ff([1., 1.5, 2., 2.5])
        array([0.18126925, 0.18126925, 0.36237185, 0.36237185])
        """
        return 1 - self.sf(x, interp=interp)

    @keeps_query_shape
    def hf(self, x: npt.ArrayLike, interp: str = "step") -> npt.NDArray:
        r"""

        The discrete hazard of the non-parametric estimate: the increment
        of the cumulative hazard ``H`` between consecutive requested points
        (for a single point, the increment of the step it falls in). It is
        a jump size, not an instantaneous rate, so it depends on how finely
        ``x`` is spaced; for a rate, use ``smoothed_hf``.

        Two conventions apply to an array ``x`` (taken in sorted order and
        returned in the order given): the first point has nothing before
        it to difference from and repeats the second point's value, and a
        zero increment (a gap in ``x`` with no failure) is replaced by the
        previous non-zero one. Where there is no previous non-zero
        increment, e.g. before the first failure, the result is NaN. With
        bounds set (see ``set_support``) it is 0 from ``lower`` to the first
        value and NaN outside the bounds, and the points outside take no
        part in the differences.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which
            the hazard increments will be calculated

        interp : str, optional
            How to evaluate between the estimate's time points: ``"step"``
            (the default, the right-continuous step function), ``"linear"``,
            ``"cubic"`` (a monotone PCHIP curve) or one of the other
            string kinds of :func:`scipy.interpolate.interp1d`
            (``"nearest"``, ``"nearest-up"``, ``"zero"``, ``"slinear"``,
            ``"quadratic"``, ``"previous"``, ``"next"``); any other value
            raises a ``ValueError``. The interpolated forms return NaN
            outside the range of the data, unless the model has bounds
            (see ``set_support``).

        Returns
        -------

        hf : scalar or numpy array
            The increment of the cumulative hazard at each x


        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.hf([1.5, 2.5, 3.5])
        array([0.25      , 0.25      , 0.33333333])
        """
        _check_interp(interp)
        return self._jumps_within_support(x, interp)[0]

    def _jumps_within_support(
        self, x: npt.ArrayLike, interp: str
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """``(hf, df)`` at ``x``, restricted to the support when one is set
        (see :meth:`set_support`)."""
        x = np.atleast_1d(x)
        if self.support is None:
            return self._jumps(x, interp)
        # Only the points within the bounds are differenced (``Hf`` is 0
        # from ``lower`` and carried to ``upper``), and the increment is 0
        # before the first value.
        xf = np.asarray(x, dtype=float)
        lower, upper = self.support
        inside = (xf >= lower) & (xf <= upper)
        hf, df = np.full(xf.shape, np.nan), np.full(xf.shape, np.nan)
        if inside.any():
            hf[inside], df[inside] = self._jumps(xf[inside], interp)
        before = inside & (xf < self.x[0])
        hf[before], df[before] = 0.0, 0.0
        return hf, df

    def _jumps(
        self, x: npt.NDArray, interp: str
    ) -> tuple[npt.NDArray, npt.NDArray]:
        # ``hf`` and ``df`` without the bounds (see ``set_support``): the
        # increment of ``Hf`` and the drop in ``sf`` over the same step,
        # the one the conventions of ``hf`` pick for each point.
        if x.size == 0:
            # Nothing to difference (the first point below needs one).
            return np.empty(0), np.empty(0)
        missing = np.isnan(np.asarray(x, dtype=float))
        if missing.any():
            # A missing time has no value (NaN). The increments of the
            # others are the ones they have without it; the forward fill
            # below would otherwise copy a neighbour's increment into it.
            hf, df = np.full(x.shape, np.nan), np.full(x.shape, np.nan)
            if not missing.all():
                hf[~missing], df[~missing] = self._jumps(x[~missing], interp)
            return hf, df
        idx = np.argsort(x)
        rev = np.argsort(idx)
        x = x[idx]
        if x.size == 1:
            # The pairwise-difference construction below yields a single
            # zero for scalar input (nothing to backfill from), so hf()
            # and df() always returned NaN for scalars (#282). Use the
            # model's own step ladder instead: the discrete hazard
            # increment of the step containing x (forward-filled over
            # zero-increment censoring steps), matching what the array
            # path returns for the same point inside a grid.
            xs = np.atleast_1d(self.x)
            H = np.hstack([[0.0], self.Hf(xs, interp=interp)])
            S = np.hstack([[1.0], self.sf(xs, interp=interp)])
            pos = np.searchsorted(xs, x[0], side="right") - 1
        else:
            H = np.hstack(
                [self.Hf(x[0], interp=interp), self.Hf(x, interp=interp)]
            )
            S = np.hstack(
                [self.sf(x[0], interp=interp), self.sf(x, interp=interp)]
            )
        # Once a Kaplan-Meier estimate reaches zero, H is inf at every
        # later time and inf - inf is NaN: no new increment, which the
        # forward fill below treats like a zero one (the hazard of the
        # step containing x, here the infinite jump to zero).
        with np.errstate(invalid="ignore"):
            dH = np.diff(H)
        # step[i] is the step (from H[step] to H[step + 1]) whose increment
        # point i takes: its own, or the last non-zero one before it; -1
        # where there is none (NaN).
        has_jump = ~np.isnan(dH) & (dH != 0)
        if x.size == 1:
            below = np.flatnonzero(has_jump[: max(pos + 1, 0)])
            step = np.array([below[-1] if below.size else -1])
        else:
            candidate = np.where(has_jump, np.arange(dH.size), -1)
            # The first point has nothing before it to difference from
            # (its dH is 0) and repeats the second.
            candidate[0] = candidate[1]
            step = np.maximum.accumulate(candidate)
        ok = step >= 0
        s = np.where(ok, step, 0)
        hf = np.where(ok, dH[s], np.nan)
        # The drop in sf over that step: the probability the estimate puts
        # there. Finite where hf * sf was not: inf * 0 once a Kaplan-Meier
        # reaches zero (#408).
        df = np.where(ok, S[s] - S[s + 1], np.nan)
        return hf[rev], df[rev]

    @keeps_query_shape
    def df(self, x: npt.ArrayLike, interp: str = "step") -> npt.NDArray:
        r"""

        The probability the non-parametric estimate puts in each step of
        ``x``: the drop in ``sf`` over the step whose increment ``hf``
        gives, the one from the previous requested point (with the same
        conventions: the first point repeats the second, and a step with
        no failure takes the last one that had a failure before it). Like
        ``hf`` it is a jump size, not a density per unit of time, so it
        depends on how finely ``x`` is spaced; over a step with increment
        :math:`\Delta H` that starts at survival :math:`S` it is
        :math:`S(1 - e^{-\Delta H})`, which for small steps is close to
        :math:`h(x)e^{-H(x)}`.

        It stays finite where the estimate reaches zero: the step a
        Kaplan-Meier estimate takes to zero has an infinite ``hf``, but its
        probability is the survival just before it. (``df`` used to be
        computed as ``hf * exp(-Hf)``, which is ``inf * 0``, NaN, there.)

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the
            density will be calculated

        interp : str, optional
            How to evaluate between the estimate's time points: ``"step"``
            (the default, the right-continuous step function), ``"linear"``,
            ``"cubic"`` (a monotone PCHIP curve) or one of the other
            string kinds of :func:`scipy.interpolate.interp1d`
            (``"nearest"``, ``"nearest-up"``, ``"zero"``, ``"slinear"``,
            ``"quadratic"``, ``"previous"``, ``"next"``); any other value
            raises a ``ValueError``. The interpolated forms return NaN
            outside the range of the data, unless the model has bounds
            (see ``set_support``).

        Returns
        -------

        df : scalar or numpy array
            The probability of failing in the step of ``x`` at each x.

        Examples
        --------
        >>> from surpyval import KaplanMeier, NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.df([1.5, 2.5, 3.5])
        array([0.18110261, 0.18110261, 0.18074761])
        >>> KaplanMeier.fit([1, 2, 3]).df([1.5, 2.5, 3.5])
        array([0.33333333, 0.33333333, 0.33333333])
        """
        _check_interp(interp)
        return self._jumps_within_support(x, interp)[1]

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, interp: str = "step") -> npt.NDArray:
        r"""

        Cumulative hazard with the non-parametric estimates
        from the data. This is calculated from the survival estimate:

        .. math::
            H(x) = -\ln (R(x))

        For the Nelson-Aalen and Fleming-Harrington estimators, which
        report :math:`R = e^{-H}`, this is exactly their summed hazard.
        For the Kaplan-Meier it is infinite once the estimate reaches zero.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the
            function will be calculated.

        interp : str, optional
            How to evaluate between the estimate's time points: ``"step"``
            (the default, the right-continuous step function), ``"linear"``,
            ``"cubic"`` (a monotone PCHIP curve) or one of the other
            string kinds of :func:`scipy.interpolate.interp1d`
            (``"nearest"``, ``"nearest-up"``, ``"zero"``, ``"slinear"``,
            ``"quadratic"``, ``"previous"``, ``"next"``); any other value
            raises a ``ValueError``. The interpolated forms return NaN
            outside the range of the data, unless the model has bounds
            (see ``set_support``).

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard at x

        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.Hf(2)
        np.float64(0.44999999999999996)
        >>> model.Hf([1., 1.5, 2., 2.5])
        array([0.2 , 0.2 , 0.45, 0.45])
        """
        # Bounded separately so that it starts at 0.0, not -log(1) = -0.0.
        _check_interp(interp)
        return self._within_support(x, lambda q: self._Hf(q, interp), 0.0)

    def _Hf(self, x: npt.ArrayLike, interp: str) -> npt.NDArray:
        # ``Hf`` without the bounds (see ``set_support``).
        sf = self._sf(x, interp)
        # -log(0) = inf is the documented value once sf reaches zero.
        with np.errstate(divide="ignore"):
            return -np.log(sf)

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        on: str = "sf",
        bound: str = "two-sided",
        interp: str = "step",
        alpha_ci: float = 0.05,
        bound_type: str = "exp",
        dist: str = "z",
    ) -> npt.NDArray:
        r"""

        Pointwise confidence bounds of the ``on`` function at the
        ``alpha_ci`` level of significance. Can be the upper,
        lower, or two-sided confidence by changing value of ``bound``.
        The bound type can be the plain normal interval or the
        exponential (log(-log) transformed) one, using the normal ('z')
        statistic.

        The variance used is the one appropriate to the estimator with
        which the model was fitted: Greenwood's formula for Kaplan-Meier,
        Aalen's (Poisson) variance for Nelson-Aalen, and the tie-corrected
        variance for Fleming-Harrington. A Turnbull model uses the formula
        of its ``turnbull_estimator``, on the observed counts for exact,
        right censored and truncated data and on the EM's expected counts
        for left or interval censored data (an approximation; prefer
        ``bootstrap_cb`` there).

        The bounds hold at each ``x`` separately; for a band that holds
        over the whole curve at once use ``band``.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the confidence bounds
            will be calculated
        on : ('sf', 'ff', 'Hf'), optional
            The function on which the confidence bound will be calculated
            ('R' and 'F' are accepted for 'sf' and 'ff'); bounds on the
            density or the hazard rate are not available.
            Defaults to 'sf'. The bounds on 'ff' are one minus those on
            'sf', and those on 'Hf' are minus their logarithm; a two-sided
            result is always ``[lower, upper]`` for the function asked
            about.
        bound : ('two-sided', 'upper', 'lower'), str, optional
            Compute either the two-sided, upper or lower confidence bound(s).
            Defaults to two-sided. A one-sided bound puts all of
            ``alpha_ci`` on one side, so it equals the corresponding end
            of the two-sided interval at ``2 * alpha_ci``.
        interp : ('step', 'linear', 'cubic'), optional
            How to interpolate the values between observations. Survival
            statistics traditionally uses step functions, but can use
            interpolated values if desired. Defaults to step. Takes the
            values of ``sf``'s ``interp``.
        alpha_ci : scalar, optional
            The level of significance at which the bound will be computed.
            Defaults to 0.05.
        bound_type : ('exp', 'normal'), str, optional
            The method with which the bounds will be calculated. Using
            'normal' (i.e. the plain Greenwood-style interval,
            :math:`\hat{R} \pm z \hat{R}\hat{\sigma}`) will allow
            for the bounds to exceed 1 or be less than 0 and tends to
            undercover in small samples. Defaults to 'exp' (the
            log(-log) transformed interval) as this ensures the bounds
            are within 0 and 1 and has better coverage.
        dist : ('z',), str, optional
            The statistic used for the bounds. Only the normal ('z') is
            supported, which is what the asymptotic theory of the estimator
            justifies. For small-sample or Turnbull bounds use
            ``bootstrap_cb``; for a simultaneous band use ``band``.

        Returns
        -------

        cb : scalar or numpy array
            The value(s) of the upper, lower, or both confidence bound(s) of
            the selected function at x. For two-sided bounds the result has
            one row per value of x with the columns being the
            ``[lower, upper]`` bounds of the ``on`` function.

        Raises
        ------

        ValueError
            If ``on``, ``bound``, ``bound_type`` or ``dist`` is not one of
            the values listed above, or the model has no variance estimate
            (``fit_from_ecdf``).

        Notes
        -----

        If the last observation is an event (i.e. there is no right
        censoring) the Kaplan-Meier variance is undefined at, and after,
        that point. The bounds there are filled with the last finite
        upper bound and 0 for the lower bound, rather than NaN, so that
        bounds can be drawn up to the last observation. If no value has a
        finite bound (a single exact observation) the upper bound is the
        estimate itself, 0.

        Where the variance is zero (before the first failure) the bounds
        are the estimate, 1. Below the first and above the last observed
        value the bounds are NaN, unless the model has bounds (see
        ``set_support``): then they are the estimate's initial value from
        ``lower`` to the first value, the bounds at the last value carried
        from there to ``upper``, and NaN outside ``[lower, upper]``.

        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.cb([1., 1.5, 2., 2.5], bound='lower')
        array([0.35485348, 0.35485348, 0.2345113 , 0.2345113 ])
        >>> model.cb([1., 1.5, 2., 2.5])
        array([[0.24175891, 0.97222045],
               [0.24175891, 0.97222045],
               [0.16288538, 0.89441253],
               [0.16288538, 0.89441253]])

        References
        ----------

        Klein, J. P. and Moeschberger, M. L. (2003), "Survival
        Analysis", 2nd ed., Sections 4.2 and 4.3.

        http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis

        """
        # The guard used to test ``on in []`` and so never fired: any other
        # ``on`` (e.g. 'hf') fell through to the survival bounds in
        # ``[upper, lower]`` order, i.e. with the lower above the upper.
        if on not in _CB_ON:
            raise ValueError(
                "'on' must be one of {}; got {!r}. Non-parametric bounds "
                "are not available on the density or the hazard rate "
                "('df', 'hf').".format(_CB_ON, on)
            )
        _check_bound(bound)
        _check_interp(interp)
        # Bounded here too (``R_cb`` is) so that 'ff' and 'Hf' start at
        # 0.0, not at -log(1) = -0.0.
        return self._bounds_within_support(
            x,
            lambda q: self._cb(
                q, on, bound, interp, alpha_ci, bound_type, dist
            ),
            1.0 if on in ("sf", "R") else 0.0,
        )

    def _cb(
        self,
        x: npt.ArrayLike,
        on: str,
        bound: str,
        interp: str,
        alpha_ci: float,
        bound_type: str,
        dist: str,
    ) -> npt.NDArray:
        # ``cb`` without the bounds (see ``set_support``).
        with np.errstate(all="ignore"):

            # Reverse for ff and F
            if on in ["ff", "F", "Hf"] and bound == "lower":
                bound = "upper"
            elif on in ["ff", "F", "Hf"] and bound == "upper":
                bound = "lower"

            cb = self.R_cb(
                x,
                bound=bound,
                interp=interp,
                alpha_ci=alpha_ci,
                bound_type=bound_type,
                dist=dist,
            )

            if (on == "ff") or (on == "F"):
                cb = 1.0 - cb

            elif on == "Hf":
                cb = -np.log(cb)

            elif (on == "sf") or (on == "R"):
                if bound == "two-sided":
                    cb = np.fliplr(cb)

        return cb

    @keeps_query_shape
    def R_cb(
        self,
        x: npt.ArrayLike,
        bound: str = "two-sided",
        interp: str = "step",
        alpha_ci: float = 0.05,
        bound_type: str = "exp",
        dist: str = "z",
    ) -> npt.NDArray:
        r"""
        Confidence bounds of the survival function, as used by ``cb`` and
        ``plot``. Takes the same arguments as ``cb`` (without ``on``), but
        a two-sided result has the columns in ``[upper, lower]`` order;
        ``cb(x, on='sf')`` returns them as ``[lower, upper]`` and is the
        method to call. With a support set (see ``set_support``) they are 1
        from ``lower`` to the first value, the bounds at the last value
        from there to ``upper``, and NaN outside.
        """
        _check_bound(bound)
        _check_interp(interp)
        return self._bounds_within_support(
            x,
            lambda q: self._R_cb(q, bound, interp, alpha_ci, bound_type, dist),
            1.0,
        )

    def _R_cb(
        self,
        x: npt.ArrayLike,
        bound: str,
        interp: str,
        alpha_ci: float,
        bound_type: str,
        dist: str,
    ) -> npt.NDArray:
        # ``R_cb`` without the bounds (see ``set_support``).
        if bound_type not in ["exp", "normal"]:
            raise ValueError("'bound_type' must be in ['exp', 'normal']")
        _check_bound(bound)
        if dist != "z":
            raise ValueError(
                "'dist' must be 'z'. The 't' option (Student-t with the "
                "at-risk count as degrees of freedom) has been removed: it "
                "had no asymptotic justification, was undefined at the last "
                "event, and widened bounds arbitrarily as the risk set "
                "shrank. For small-sample or Turnbull confidence bounds use "
                "`bootstrap_cb`, and for a simultaneous band use `band`."
            )
        if getattr(self, "greenwood", None) is None:
            raise ValueError(
                "Model has no variance estimate so confidence bounds "
                + "cannot be computed. This occurs for models created "
                + "with 'fit_from_ecdf' since the at risk and death "
                + "counts are unknown."
            )

        confidence = 1.0 - alpha_ci

        with np.errstate(all="ignore"):

            x = np.atleast_1d(x)
            if bound in ["upper", "lower"]:
                stat = norm.ppf(1 - confidence, 0, 1)
                if bound == "upper":
                    stat = -stat
            elif bound == "two-sided":
                stat = norm.ppf((1 - confidence) / 2, 0, 1)
                stat = np.array([-1, 1]).reshape(2, 1) * stat

            if bound_type == "exp":
                # Exponential Greenwood confidence
                R_out = self.greenwood * 1.0 / (np.log(self.R) ** 2)
                R_out = np.log(-np.log(self.R)) - stat * np.sqrt(R_out)
                R_out = np.exp(-np.exp(R_out))
                # No variance, no interval: the bounds collapse onto the
                # estimate. (That is 1 before the first event; it is the
                # estimate itself, not 1, where float-noise counts in a
                # Turnbull ladder were snapped to zero events.)
                R_out = np.where(self.greenwood == 0, self.R, R_out)
            else:
                # Normal Greenwood confidence
                R_out = self.R + np.sqrt(self.greenwood * self.R**2) * stat

            # Allows for confidence bound to be estimated up to the last value.
            # only used in event that there is no right censoring. When *no*
            # point on the curve has a finite bound (e.g. a single
            # observation), fall back to the point estimate rather than
            # propagating an all-NaN nanmin (#282).
            def _fill_upper(row: npt.NDArray) -> npt.NDArray:
                finite = np.isfinite(row)
                if finite.any():
                    return np.where(finite, row, row[finite].min())
                return np.asarray(self.R, dtype=float).copy()

            # Where the bounds are defined, before the fill below.
            defined = np.isfinite(R_out)
            if bound == "upper":
                R_out = _fill_upper(R_out)
            elif bound == "lower":
                R_out = np.where(np.isfinite(R_out), R_out, 0)
            else:
                R_out[0, :] = _fill_upper(R_out[0, :])
                R_out[1, :] = np.where(
                    np.isfinite(R_out[1, :]), R_out[1, :], 0
                )

            if interp == "step":
                idx = np.searchsorted(self.x, x, side="right") - 1
                if bound == "two-sided":
                    R_out = R_out[:, idx]
                    R_out = np.where(idx < 0, 1, R_out)
                else:
                    R_out = R_out[idx]
                    R_out = np.where(idx < 0, 1, R_out)

            else:
                unit = bound_type == "exp"
                if bound == "two-sided":
                    R_out = np.vstack(
                        [
                            interp_bound(
                                self.x,
                                self.R,
                                R_out[k],
                                defined[k],
                                x,
                                interp,
                                unit,
                            )
                            for k in (0, 1)
                        ]
                    )
                else:
                    R_out = interp_bound(
                        self.x, self.R, R_out, defined, x, interp, unit
                    )

            # A missing x, or one outside the observed values, is NaN (or
            # set by the support): ``R_cb`` only asks within them (see
            # ``_bounds_within_support``).
            if bound == "two-sided":
                R_out = R_out.T

        return R_out

    def random(
        self, size: int, random_state: int | None = None
    ) -> npt.NDArray:
        r"""
        Draws lifetimes from the fitted estimate. Each draw is
        ``qf(u)`` for one uniform ``u`` in (0, 1]: the first step time at
        which the estimated CDF reaches ``u``. So each step time is drawn
        with the probability the estimate puts there (the drop in ``sf``),
        and the draws follow the model's own ``sf`` exactly.

        Where the estimate does not reach zero (the last observation
        censored, say), the probability it leaves beyond its last time,
        ``sf`` there, is not placed anywhere in the data: those draws are
        ``inf``, a lifetime not observed to end within the data. This is
        the convention of a parametric model with a limited failure
        population, whose never-failing units also draw ``inf``.

        Parameters
        ----------

        size : int or tuple of ints
            The number (or shape) of random samples to draw.
        random_state : int or numpy.random.Generator, optional
            Seed or generator for reproducible sampling. ``None`` (the default)
            seeds from numpy's global RNG, so ``np.random.seed`` controls it.
            Matches the ``random_state`` argument of ``bootstrap_cb`` and
            ``band``.

        Returns
        -------

        random : numpy array
            The drawn lifetimes: step times of the estimate, or ``inf``.

        Examples
        --------
        >>> from surpyval import KaplanMeier
        >>> model = KaplanMeier.fit([1, 2, 3, 4, 5, 6, 7, 8],
        ...                         c=[0, 1, 0, 0, 1, 0, 0, 1])
        >>> round(model.sf(8).item(), 4)
        0.1944
        >>> draws = model.random(10_000, random_state=0)
        >>> float(np.isinf(draws).mean())  # the share left beyond 8
        0.1988
        """
        rng = as_generator(random_state)
        # One uniform per draw, in (0, 1] as qf requires; qf(u) is NaN
        # where the estimated CDF never reaches u, the mass the estimate
        # leaves beyond its last time, which is drawn as inf.
        u = 1.0 - rng.random(size)
        draws = np.asarray(self.qf(np.ravel(u)), dtype=float)
        draws = np.where(np.isnan(draws), np.inf, draws)
        return draws.reshape(np.shape(u))

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike) -> npt.NDArray:
        r"""
        Quantile function of the non-parametric estimate. Returns the
        smallest observed value at which the estimated CDF reaches, or
        exceeds, the probability p. A CDF within 1e-9 of p counts as
        reaching it, so that round-off in the estimate does not move the
        quantile a step late: ``qf(ff(x))`` returns ``x`` at the steps.

        Parameters
        ----------

        p : array like or scalar
            The probabilities at which the quantile will be computed.
            Values must be in (0, 1].

        Returns
        -------

        q : numpy array
            The value(s) of the quantile at each p. NaN where the
            estimated CDF never reaches p (e.g. due to right censoring).

        Examples
        --------
        >>> from surpyval import KaplanMeier
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = KaplanMeier.fit(x)
        >>> model.qf(0.5)
        np.float64(3.0)
        >>> model.qf([0.1, 0.5, 0.9])
        array([1., 3., 5.])
        """
        p = np.atleast_1d(p).astype(float)
        if ((p <= 0) | (p > 1)).any():
            raise ValueError("'p' must be in the range (0, 1]")
        # F is a product (or exponentiated sum) of ratios, so where it
        # should equal p exactly it carries round-off: the Kaplan-Meier F
        # of 1..30 at 15 is 0.4999999999999999, and a Turnbull ladder is
        # only as exact as its EM tolerance. Comparing exactly put the
        # quantile one step late (median 16, not 15), so that qf(ff(x))
        # skipped x. A step within 1e-9 of p counts as reaching it -- the
        # same tolerance ``_snap`` uses for counts, and above the ~4e-10
        # by which a converged Turnbull ladder differs from the
        # Kaplan-Meier. The floor keeps a p below that tolerance from
        # matching the steps where F is still exactly 0.
        target = np.maximum(p - _QF_TOL, np.finfo(float).tiny)
        idx = np.searchsorted(self.F, target, side="left")
        x_padded = np.hstack([self.x.astype(float), [np.nan]])
        return x_padded[np.minimum(idx, len(self.x))]

    @property
    def median(self) -> float:
        r"""
        The median survival time; the smallest observed value at which
        the estimated CDF reaches, or exceeds, 0.5 (up to round-off, as
        for ``qf``). NaN if the estimate never reaches 0.5 (e.g. due to
        right censoring).

        Examples
        --------
        >>> from surpyval import KaplanMeier
        >>> KaplanMeier.fit(np.arange(1, 31)).median
        np.float64(15.0)
        """
        return self.qf(0.5)

    @keeps_query_shape
    def quantile_cb(
        self,
        p: npt.ArrayLike,
        alpha_ci: float = 0.05,
        bound_type: str = "exp",
        dist: str = "z",
    ) -> npt.NDArray:
        r"""
        Two-sided confidence interval of the quantile at each
        probability p using the Brookmeyer-Crowley method: the interval
        is the set of times at which the pointwise confidence interval
        of the survival function contains 1 - p.

        Parameters
        ----------

        p : array like or scalar
            The probabilities at which the quantile interval will be
            computed. Values must be in (0, 1].
        alpha_ci : scalar, optional
            The level of significance at which the interval will be
            computed. Defaults to 0.05.
        bound_type : ('exp', 'normal'), str, optional
            The method for the underlying survival function bounds.
        dist : ('z',), str, optional
            The statistic used in the underlying survival function
            bounds. Only the normal ('z') is supported.

        Returns
        -------

        cb : numpy array
            Array of shape (len(p), 2) with the ``[lower, upper]``
            interval of the quantile for each p. The upper limit is NaN
            where the relevant bound of the survival function never
            crosses 1 - p (i.e. the interval is open to the right).

        Examples
        --------
        >>> from surpyval import KaplanMeier
        >>> model = KaplanMeier.fit([1, 2, 3, 4, 5, 6, 7, 8],
        ...                         c=[0, 1, 0, 0, 1, 0, 0, 1])
        >>> model.quantile_cb([0.25, 0.5])
        array([[ 1.,  6.],
               [ 1., nan]])

        References
        ----------

        Brookmeyer, R. and Crowley, J. (1982), "A confidence interval for
        the median survival time", Biometrics 38, 29-41.
        """
        p = np.atleast_1d(p).astype(float)
        if ((p <= 0) | (p > 1)).any():
            raise ValueError("'p' must be in the range (0, 1]")

        bounds = self.cb(
            self.x,
            on="sf",
            bound="two-sided",
            alpha_ci=alpha_ci,
            bound_type=bound_type,
            dist=dist,
        )
        lower_sf, upper_sf = bounds[:, 0], bounds[:, 1]

        out = np.empty((p.size, 2))
        for i, p_i in enumerate(p):
            level = 1.0 - p_i
            # Times enter the interval when the lower survival bound
            # falls to the level, and leave it once the upper survival
            # bound falls below the level.
            in_lower = lower_sf <= level
            in_upper = upper_sf < level
            out[i, 0] = (
                self.x[np.argmax(in_lower)] if in_lower.any() else np.nan
            )
            out[i, 1] = (
                self.x[np.argmax(in_upper)] if in_upper.any() else np.nan
            )
        return out

    def mean(self, tau: float | None = None) -> float:
        r"""
        The (restricted) mean survival time: the area under the
        estimated survival function from 0 to tau.

        If the survival function reaches zero this is the mean of the
        estimated distribution. With right censoring the survival
        function does not reach zero and the unrestricted mean is
        undefined; the restricted mean up to tau (defaulting to the
        largest observed value) is reported instead, which is the
        standard restricted mean survival time (RMST).

        Parameters
        ----------

        tau : scalar, optional
            The horizon up to which the survival function is
            integrated; must be non-negative. Defaults to the largest
            observed value. If tau is beyond the last observation the
            survival function is extended at its final value.

        Returns
        -------

        mean : float
            The restricted mean survival time.

        Examples
        --------
        >>> from surpyval import KaplanMeier
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = KaplanMeier.fit(x)
        >>> model.mean()
        3.0000000000000004
        """
        if np.min(self.x) < 0:
            raise ValueError(
                "Mean survival time requires non-negative observations"
            )
        if tau is None:
            tau = np.max(self.x)
        # A negative horizon used to come back as the (negative) width of
        # [0, tau], i.e. mean(tau=-1) was -1.
        if not tau >= 0:
            raise ValueError(
                "'tau' must be a non-negative number; got {}".format(tau)
            )

        xs = self.x[self.x < tau].astype(float)
        times = np.hstack([[0.0], xs, [tau]])
        surv = np.hstack([[1.0], self.R[: xs.size]])
        return float(np.sum(np.diff(times) * surv))

    def mean_cb(
        self, tau: float | None = None, alpha_ci: float = 0.05
    ) -> npt.NDArray:
        r"""
        Two-sided confidence interval of the (restricted) mean survival
        time, using the normal approximation with the standard variance
        estimate:

        .. math::
            \widehat{Var}(\hat{\mu}) = \sum_{i: x_i \leq \tau}
                A_i^2 v_i

        where :math:`A_i` is the area under the survival function from
        :math:`x_i` to :math:`\tau` and :math:`v_i` is the variance
        increment of the cumulative hazard at :math:`x_i` (e.g. the
        Greenwood increment for the Kaplan-Meier estimator).

        Parameters
        ----------

        tau : scalar, optional
            The horizon up to which the survival function is
            integrated. Defaults to the largest observed value.
        alpha_ci : scalar, optional
            The level of significance at which the interval will be
            computed. Defaults to 0.05.

        Returns
        -------

        cb : numpy array
            The ``[lower, upper]`` interval of the (restricted) mean.

        Examples
        --------
        >>> from surpyval import KaplanMeier
        >>> model = KaplanMeier.fit([1, 2, 3, 4, 5, 6, 7, 8],
        ...                         c=[0, 1, 0, 0, 1, 0, 0, 1])
        >>> model.mean_cb(tau=6)
        array([3.36776153, 5.92390514])
        """
        r = self.rmst(tau=tau, alpha_ci=alpha_ci)
        return np.array([r["lower"], r["upper"]])

    def _rmst_variance(self, tau: float) -> float:
        r"""Variance of the restricted mean survival time up to ``tau``:

        .. math::
            \widehat{Var}(\hat{\mu}) = \sum_{i: x_i \leq \tau} A_i^2 v_i

        where :math:`A_i` is the area under the survival curve from
        :math:`x_i` to :math:`\tau` and :math:`v_i` the variance increment
        of the cumulative hazard (the Greenwood increment for Kaplan-Meier).
        """
        if getattr(self, "greenwood", None) is None:
            raise ValueError(
                "Model has no variance estimate so confidence bounds "
                + "cannot be computed. This occurs for models created "
                + "with 'fit_from_ecdf' since the at risk and death "
                + "counts are unknown."
            )
        # Area under the survival curve from each observation to tau
        xs = self.x.astype(float)
        upper_t = np.minimum(np.hstack([xs[1:], [np.inf]]), tau)
        widths = np.clip(upper_t - np.minimum(xs, tau), 0, None)
        seg_area = widths * self.R
        # A[i] is the area from x[i] to tau
        A = np.cumsum(seg_area[::-1])[::-1]

        v = np.diff(np.hstack([[0.0], self.greenwood]))
        with np.errstate(all="ignore"):
            terms = np.where(A > 0, A**2 * v, 0.0)
        return float(np.sum(terms))

    def rmst(self, tau: float | None = None, alpha_ci: float = 0.05) -> dict:
        r"""
        Restricted mean survival time up to ``tau`` with inference.

        Returns the RMST (area under the survival curve to ``tau``, as
        ``mean(tau)``), its standard error, and the two-sided normal
        confidence interval ``rmst +- z se``. The variance is

        .. math::
            \widehat{Var}(\hat{\mu}) = \sum_{i: x_i \leq \tau} A_i^2 v_i

        with :math:`A_i` the area under the curve from :math:`x_i` to
        :math:`\tau` and :math:`v_i` the increment of the estimator's
        variance of the cumulative hazard at :math:`x_i` (Greenwood's for
        the Kaplan-Meier). For a Turnbull model with left or interval
        censoring those increments come from the EM's expected counts, so
        the standard error is approximate.

        Parameters
        ----------
        tau : scalar, optional
            Integration horizon; defaults to the largest observed value. A
            ``tau`` beyond it holds the curve at its final value.
        alpha_ci : scalar, optional
            Significance level for the interval (default 0.05).

        Returns
        -------
        dict
            ``{"rmst", "se", "lower", "upper", "tau"}``.

        Examples
        --------
        >>> from surpyval import KaplanMeier
        >>> model = KaplanMeier.fit([1, 2, 3, 4, 5, 6, 7, 8],
        ...                         c=[0, 1, 0, 0, 1, 0, 0, 1])
        >>> res = model.rmst(tau=6)
        >>> print(round(res["rmst"], 4), round(res["se"], 4))
        4.6458 0.6521

        See Also
        --------
        surpyval.rmst_diff : compare the RMST of two groups.
        """
        if tau is None:
            tau = float(np.max(self.x))
        mu = self.mean(tau=tau)
        se = float(np.sqrt(self._rmst_variance(tau)))
        z = norm.ppf(1 - alpha_ci / 2)
        return {
            "rmst": mu,
            "se": se,
            "lower": mu - z * se,
            "upper": mu + z * se,
            "tau": float(tau),
        }

    @keeps_query_shape
    def smoothed_hf(
        self, x: npt.ArrayLike, bandwidth: float | None = None
    ) -> npt.NDArray:
        r"""
        Kernel smoothed estimate of the hazard rate, using an
        Epanechnikov kernel over the increments of the cumulative
        hazard estimate:

        .. math::
            \hat{h}(t) = \frac{1}{b} \sum_{i} K\left (
                \frac{t - x_i}{b} \right ) \Delta \hat{H}(x_i)

        Contributions are renormalised near the boundaries of the
        observed range so the estimate is not biased downward where the
        kernel window extends past the data.

        This is a better estimate of the hazard rate than ``hf()``,
        which simply differences the cumulative hazard between the
        requested points.

        Parameters
        ----------

        x : array like or scalar
            The values at which the hazard rate will be estimated.
        bandwidth : scalar, optional
            The kernel bandwidth in the units of x. Defaults to a rough
            rule of thumb (one eighth of the observed range); for
            serious use choose by inspection or cross-validation. (A
            model with a single distinct value has no range to smooth
            over: the default raises and any bandwidth gives NaN.)

        Returns
        -------

        hf : numpy array
            The estimated hazard rate at each x. NaN outside the
            observed range, whether or not the model has bounds
            (``set_support``). An infinite jump (a Kaplan-Meier falling to
            zero at the last failure) is left out of the sum.

        Examples
        --------
        >>> from surpyval import KaplanMeier
        >>> model = KaplanMeier.fit([1, 2, 3, 4, 5, 6, 7, 8],
        ...                         c=[0, 1, 0, 0, 1, 0, 0, 1])
        >>> model.smoothed_hf([3, 4, 5], bandwidth=2)
        array([0.13112971, 0.13495677, 0.17679619])

        References
        ----------

        Klein, J. P. and Moeschberger, M. L. (2003), "Survival
        Analysis", 2nd ed., Section 6.2.
        """
        x = np.atleast_1d(x).astype(float)

        with np.errstate(all="ignore"):
            dH = np.diff(np.hstack([[0.0], self.H]))
        dH = np.where(np.isfinite(dH), dH, 0.0)

        x_min = self.x.min()
        x_max = self.x.max()
        if bandwidth is None:
            if not x_max > x_min:
                # The default is a fraction of the range, which is zero
                # here; the message used to blame a bandwidth the caller
                # never passed.
                raise ValueError(
                    "The default bandwidth is one eighth of the observed "
                    "range, and this model has a single distinct value, so "
                    "there is no range to smooth over."
                )
            bandwidth = (x_max - x_min) / 8
        if not bandwidth > 0:
            raise ValueError(
                "'bandwidth' must be positive; got {}".format(bandwidth)
            )

        u = (x[:, None] - self.x[None, :]) / bandwidth
        kern = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
        h = (kern * dH).sum(axis=1) / bandwidth

        # Renormalise by the kernel mass that lies within the observed
        # range, correcting the downward bias near the boundaries. The
        # Epanechnikov CDF is (2 + 3v - v^3) / 4 on [-1, 1].
        def epa_cdf(v: npt.NDArray) -> npt.NDArray:
            v = np.clip(v, -1, 1)
            return (2 + 3 * v - v**3) / 4

        lo = (x - x_max) / bandwidth
        hi = (x - x_min) / bandwidth
        mass = epa_cdf(hi) - epa_cdf(lo)
        with np.errstate(all="ignore"):
            h = np.where(mass > 0, h / mass, np.nan)

        h = np.where((x < x_min) | (x > x_max), np.nan, h)
        return h

    def get_plot_data(self, plot_bounds: bool = True, **kwargs: Any) -> dict:
        r"""
        The values ``plot`` draws: the axis limits, the observed values
        ``x_``, the estimates ``R`` and ``F`` there, and the confidence
        bounds ``cbs`` from ``R_cb`` (the keyword arguments are passed to
        it). Returned as a dictionary for custom plotting. With
        ``plot_bounds=False`` the bounds are not computed and ``cbs`` is
        None, which is what a model without a variance estimate
        (``fit_from_ecdf``) needs.

        ``failed`` is a boolean mask over ``x_``, True where a failure is
        recorded (``d > 0``, or for a model from ``fit_from_ecdf``, which
        has no ``d``, where ``F`` steps up), as in a parametric model's
        ``get_plot_data``. ``plot`` draws the curve through every row.
        """
        y_scale_min = 0
        y_scale_max = 1

        # x-axis
        x_min = min(0, np.min(self.x))
        x_max = np.max(self.x)

        diff = (x_max - x_min) / 10
        x_scale_min = x_min
        x_scale_max = x_max + diff

        # Only computed when wanted: a ``fit_from_ecdf`` model has no
        # variance, and ``plot(plot_bounds=False)`` used to raise on it.
        cbs = self.R_cb(self.x, **kwargs) if plot_bounds else None

        d = getattr(self, "d", None)
        if d is not None:
            failed = np.asarray(d) > 0
        else:
            failed = np.diff(np.asarray(self.F, dtype=float), prepend=0) > 0

        return {
            "x_scale_min": x_scale_min,
            "x_scale_max": x_scale_max,
            "y_scale_min": y_scale_min,
            "y_scale_max": y_scale_max,
            "cbs": cbs,
            "x_": self.x,
            "R": self.R,
            "F": self.F,
            "failed": failed,
        }

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        r"""
        Creates a plot of the survival function.

        Two-sided confidence bounds are drawn as a shaded band in the
        same colour as the survival curve, and right censored
        observations are marked with ticks on the curve. Any keyword
        arguments not listed below (e.g. ``color`` or ``label``) are
        passed to the matplotlib plotting call for the survival curve;
        without ``color`` each call takes the next colour of the axes'
        colour cycle, so that several estimates on one axes differ.

        The axes are titled with the estimator (e.g. "Kaplan-Meier
        estimate"), the y axis is labelled "Survival probability", and the
        x axis "Time" unless it already has a label; change any of them
        with ``ax.set_title``, ``ax.set_ylabel`` or ``ax.set_xlabel``.

        Parameters
        ----------

        ax : matplotlib axis, optional
            The axis on which the plot will be drawn. Defaults to the
            current axis.
        plot_bounds : bool, optional
            Whether to draw the confidence bounds. Defaults to True.
        show_censors : bool, optional
            Whether to mark right censored observations on the curve.
            Defaults to marking them when the model holds its data; a
            model from ``from_xrd`` or ``fit_from_ecdf``, or one restored
            from a dictionary written without ``with_data=True``, is drawn
            without them, and ``show_censors=True`` raises a
            ``ValueError`` for it.
        interp : ('step', 'linear', 'cubic'), optional
            How to draw the curve between observations.
        bound, alpha_ci, bound_type, dist : optional
            Passed to the confidence bound calculation; see ``cb()``.

        Returns
        -------

        ax : matplotlib axis

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from surpyval import KaplanMeier
        >>> fig, ax = plt.subplots()
        >>> ax = KaplanMeier.fit([1, 2, 3, 5, 8], c=[0, 1, 0, 0, 1]).plot(
        ...     ax=ax, label="A"
        ... )
        >>> ax = KaplanMeier.fit([2, 4, 6, 9, 12]).plot(ax=ax, label="B")
        >>> ax.get_title(), ax.get_xlabel(), ax.get_ylabel()
        ('Kaplan-Meier estimate', 'Time', 'Survival probability')
        >>> ax.get_legend_handles_labels()[1]
        ['A', 'B']
        >>> plt.close(fig)
        """
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()

        plot_bounds = kwargs.pop("plot_bounds", True)
        show_censors = kwargs.pop("show_censors", None)
        interp = kwargs.pop("interp", "step")
        bound = kwargs.pop("bound", "two-sided")
        alpha_ci = kwargs.pop("alpha_ci", 0.05)
        bound_type = kwargs.pop("bound_type", "exp")
        dist = kwargs.pop("dist", "z")

        _check_bound(bound)
        _check_interp(interp)
        # The censoring marks need the raw data. A restored Turnbull model
        # holds a ``data`` dict with only the estimator settings, so this
        # used to fail as ``KeyError: 'x'``.
        has_data = "x" in (getattr(self, "data", None) or {})
        if show_censors is None:
            show_censors = has_data
        elif show_censors and not has_data:
            raise ValueError(
                "Marking the censored observations needs the data the model "
                "was fitted with, which this model does not hold (a model "
                "from 'from_xrd' or 'fit_from_ecdf', or one restored from a "
                "dictionary written without to_dict(with_data=True)). Pass "
                "show_censors=False, or save the model with the data."
            )

        d = self.get_plot_data(
            plot_bounds=plot_bounds,
            interp=interp,
            bound=bound,
            alpha_ci=alpha_ci,
            bound_type=bound_type,
            dist=dist,
        )
        # MAKE THE PLOT
        # Set the y limits
        ax.set_ylim((d["y_scale_min"], d["y_scale_max"]))

        # Label it (#514): the estimator, and the axes' quantities
        ax.set_title(
            "Survival estimate"
            if self.model == "from_ecdf"
            else f"{self.model} estimate"
        )
        ax.set_ylabel("Survival probability")
        if not ax.get_xlabel():
            ax.set_xlabel("Time")
        if interp != "step":
            (line,) = ax.plot(d["x_"], d["R"], **kwargs)
        else:
            (line,) = ax.step(d["x_"], d["R"], where="post", **kwargs)
        color = line.get_color()

        if plot_bounds:
            cbs = d["cbs"]
            band_kwargs: dict[str, Any] = {
                "alpha": 0.3,
                "color": color,
                "linewidth": 0,
            }
            if interp == "step":
                band_kwargs["step"] = "post"
            if np.ndim(cbs) == 2:
                ax.fill_between(d["x_"], cbs[:, 0], cbs[:, 1], **band_kwargs)
            elif interp == "step":
                ax.step(
                    d["x_"], cbs, where="post", color=color, linestyle="--"
                )
            else:
                ax.plot(d["x_"], cbs, color=color, linestyle="--")

        if show_censors:
            x_data = self.data["x"]
            c_data = self.data["c"]
            if np.ndim(x_data) == 1 and (c_data == 1).any():
                x_cens = x_data[c_data == 1]
                ax.plot(
                    x_cens,
                    self.sf(x_cens, interp=interp),
                    linestyle="",
                    marker="|",
                    markersize=10,
                    markeredgewidth=1.5,
                    color=color,
                )

        return ax

    @classmethod
    def fit_from_ecdf(
        cls, x: npt.ArrayLike, R: npt.ArrayLike
    ) -> "NonParametric":
        r"""
        Wrap an existing survival curve, given as its values and the
        survival at each, as a non-parametric model, so that ``sf``,
        ``ff``, ``Hf``, ``qf``, ``mean`` and so on can be used with it.

        Without the numbers at risk and failing there is no variance, so
        the model has no ``cb``, ``band``, ``rmst`` or ``mean_cb`` (they
        raise a ``ValueError``), and no data for ``bootstrap_cb``. It can
        still be plotted with ``plot(plot_bounds=False)``.

        ``x`` must be increasing (a repeated value is allowed) and ``R``
        non-increasing and within [0, 1]; a ``ValueError`` is raised
        otherwise.

        Parameters
        ----------

        x : array like
            The values at which the survival is known, in increasing
            order.
        R : array like
            The survival at each value of ``x``.

        Returns
        -------

        model : NonParametric
            A model whose ``model`` attribute is ``'from_ecdf'``.

        Examples
        --------
        >>> from surpyval import NonParametric
        >>> model = NonParametric.fit_from_ecdf([1, 2, 3], [0.8, 0.5, 0.1])
        >>> model.sf([1.5, 2.5])
        array([0.8, 0.5])
        >>> model.qf(0.5)
        np.float64(2.0)
        """
        # ``sf`` and ``qf`` search these arrays, so a curve given out of
        # order, or rising, silently gave wrong survival and quantiles.
        x_arr = np.asarray(x, dtype=float)
        R_arr = np.asarray(R, dtype=float)
        if x_arr.ndim != 1 or R_arr.ndim != 1:
            raise ValueError("'x' and 'R' must be one dimensional arrays")
        if x_arr.size == 0 or x_arr.shape != R_arr.shape:
            raise ValueError(
                "'x' and 'R' must be non-empty and the same length"
            )
        if np.isnan(x_arr).any() or np.isnan(R_arr).any():
            raise ValueError("'x' and 'R' cannot contain NaN values")
        if (np.diff(x_arr) < 0).any():
            raise ValueError("'x' must be in increasing order")
        if (np.diff(R_arr) > 0).any():
            raise ValueError(
                "'R' must be non-increasing: a survival curve cannot rise"
            )
        if (R_arr < 0).any() or (R_arr > 1).any():
            raise ValueError("'R' must be within [0, 1]")

        out = cls()
        out.model = "from_ecdf"
        out.R = R_arr
        out.x = x_arr
        out.F = 1 - out.R
        with np.errstate(all="ignore"):
            out.H = -np.log(out.R)
        # Without r and d there is no variance estimate, and therefore
        # no confidence bounds, for the model.
        out.greenwood = None  # type: ignore[assignment]

        return out

    # The estimator ladder and derived curves that fully describe a fitted
    # model; everything the public methods need is a function of these.
    _SERIALIZED_ARRAYS = ("x", "r", "d", "R", "F", "H", "greenwood")

    def to_dict(self, with_data: bool = False) -> dict:
        r"""
        Serialize the fitted non-parametric model to a plain dictionary,
        mirroring the parametric ``to_dict``. The estimator ladder
        (``x``, ``r``, ``d``), the derived curves (``R``, ``F``, ``H``),
        the variance estimate (``greenwood``) and, for Turnbull models, the
        estimator name and the EM's ``tol`` and ``max_iter`` are stored,
        which is everything the model's methods need to be reconstructed
        with :meth:`from_dict`.

        Parameters
        ----------

        with_data : bool, optional
            Also store the raw ``x``/``c``/``n``/``t`` data the model was
            fitted with (needed to reconstruct a model that can call
            :meth:`bootstrap_cb`). Defaults to False.
            ``to_json(path, with_data=True)`` writes this to a file.

        Returns
        -------

        model_dict : dict
            The serialized model. The Turnbull fitting diagnostics
            (``converged``, ``iters``, ``degenerate``, ``npmle``,
            ``exploitable_mass``) and the ``bounds``, ``R_upper`` and
            ``R_lower`` arrays are not stored; the ``support`` set by
            :meth:`set_support` is, when set, and so is the sample size of
            :meth:`band` (``"band_n"``, the number of items fitted) where
            it is not the largest risk set, as with left truncated data.
            Either makes the dictionary schema 2. It is strict JSON: the
            non-finite values (``H`` after the last death, an undefined
            Greenwood term, untruncated bounds in the data) are ``None``,
            recorded under ``"non_finite"`` and restored by
            :meth:`from_dict` (see :doc:`/surpyval.serialisation`).

        Examples
        --------
        >>> import surpyval
        >>> from surpyval import KaplanMeier
        >>> model = KaplanMeier.fit([1, 2, 3, 4, 5], c=[0, 1, 0, 0, 1])
        >>> restored = surpyval.from_dict(model.to_dict())
        >>> restored.sf([2, 4])
        array([0.8       , 0.26666667])
        """
        out: dict[str, Any] = {"parameterization": "non-parametric"}
        out["model"] = self.model
        for attr in self._SERIALIZED_ARRAYS:
            value = getattr(self, attr, None)
            out[attr] = None if value is None else np.asarray(value).tolist()

        # The Turnbull settings travel with the model so that a restored
        # model's ``bootstrap_cb`` refits as the original did.
        for key in ("estimator", "tol", "max_iter"):
            if key in getattr(self, "data", {}):
                out[key] = self.data[key]

        # Only when set: without it the dictionary is readable by v0.20.
        if self.support is not None:
            out["support"] = [float(v) for v in self.support]

        # The sample size of ``band``, only where a reader without it would
        # take a different one (the largest risk set, e.g. 37 for 60 items
        # half of which entered late; #451), as for the support. Up to
        # round-off: a Turnbull EM's largest risk set is N +- 3e-14.
        if getattr(self, "r", None) is not None:
            band_n = self._band_sample_size()
            if not np.isclose(band_n, np.max(self.r), rtol=1e-9, atol=0):
                out["band_n"] = band_n

        if with_data and getattr(self, "data", None) is not None:
            data_dict: dict[str, Any] = {}
            for ch in ["x", "c", "n", "t"]:
                value = self.data.get(ch, None)
                data_dict[ch] = (
                    None if value is None else np.asarray(value).tolist()
                )
            out["data"] = data_dict
        # The printout's "Data" line (#508), so a model restored without
        # its data prints the same.
        if self._data_repr():
            out["data_summary"] = self._data_repr()

        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "NonParametric":
        r"""
        Reconstruct a fitted non-parametric model from a dictionary
        produced by :meth:`to_dict`. ``surpyval.from_dict`` does the same
        without needing to know which class wrote the dictionary.

        Parameters
        ----------

        model_dict : dict
            A dictionary written by :meth:`to_dict`.

        Returns
        -------

        model : NonParametric
            The restored model. Its curve, bounds, bands, quantiles and
            restricted mean match the original's; ``bootstrap_cb`` needs a
            dictionary written with ``with_data=True``.
        """
        if model_dict.get("parameterization") != "non-parametric":
            raise ValueError(
                "Must create a non-parametric model from a non-parametric "
                "model dict"
            )
        out = cls()
        out.model = model_dict["model"]
        for attr in cls._SERIALIZED_ARRAYS:
            value = model_dict.get(attr, None)
            if value is None:
                # ``greenwood`` is legitimately absent (no variance
                # estimate, e.g. ``fit_from_ecdf``); leave it as None so
                # the confidence-bound guards fire as they would on the
                # original model.
                if attr == "greenwood":
                    out.greenwood = None  # type: ignore[assignment]
            else:
                setattr(out, attr, np.asarray(value))

        # Turnbull models serialised before they carried ``H`` stored it
        # as None; derive it as the fitter does, so ``smoothed_hf`` works.
        if getattr(out, "H", None) is None and hasattr(out, "R"):
            with np.errstate(all="ignore"):
                out.H = -np.log(out.R)

        if "data" in model_dict or "estimator" in model_dict:
            data: dict[str, Any] = {}
            raw = model_dict.get("data", {})
            for ch in ["x", "c", "n", "t"]:
                value = raw.get(ch, None)
                if value is not None:
                    data[ch] = np.asarray(value)
            for key in ("estimator", "tol", "max_iter"):
                if key in model_dict:
                    data[key] = model_dict[key]
            out.data = data

        band_n = model_dict.get("band_n")
        if band_n is not None:
            if isinstance(band_n, bool) or not isinstance(
                band_n, numbers.Real
            ):
                raise ValueError(
                    "The serialised 'band_n' must be a number; got "
                    "{!r}.".format(band_n)
                )
            out._band_n = float(band_n)
        out._data_summary = model_dict.get("data_summary")

        support = support_from_dict(model_dict)
        if support is not None:
            out.set_support(*support)

        return out


def rmst_diff(
    model_a: "NonParametric",
    model_b: "NonParametric",
    tau: float | None = None,
    alpha_ci: float = 0.05,
) -> dict:
    """
    Compare the restricted mean survival time (RMST) of two groups.

    The RMST-difference is the standard, assumption-light alternative to the
    hazard ratio when proportional hazards fails: it needs no PH assumption
    and reads directly as a difference in expected event-free time within the
    horizon ``tau``.

    Parameters
    ----------
    model_a, model_b : NonParametric
        Two fitted non-parametric estimators (e.g. ``KaplanMeier.fit`` per
        group). They must carry a variance estimate (Greenwood).
    tau : scalar, optional
        Common horizon. Defaults to the smaller of the two groups' largest
        observed times, so both survival curves are supported by data up to
        ``tau`` (the standard choice). A larger ``tau`` is accepted without
        a warning: a curve is then held at its final value beyond its last
        observation, an extrapolation that is left to the caller to judge.
    alpha_ci : scalar, optional
        Significance level for the interval and test (default 0.05).

    Returns
    -------
    dict
        ``{"difference", "se", "lower", "upper", "p_value", "ratio",
        "rmst_a", "rmst_b", "tau"}``. ``difference`` is ``RMST_a - RMST_b``
        and ``ratio`` is ``RMST_a / RMST_b``; ``se`` is the square root of
        the sum of the two groups' variances (the groups are independent);
        ``lower`` and ``upper`` are ``difference +- z se``; ``p_value`` is
        the two-sided z-test of ``difference == 0``.

    Raises
    ------
    ValueError
        If either model has no variance estimate (``fit_from_ecdf``).

    Examples
    --------
    >>> import surpyval as sp
    >>> a = sp.KaplanMeier.fit([2, 3, 4, 5, 6, 7])
    >>> b = sp.KaplanMeier.fit([1, 2, 2, 3, 4, 5])
    >>> res = sp.rmst_diff(a, b)
    >>> print(res["tau"], round(res["difference"], 4), round(res["se"], 4))
    5.0 1.1667 0.7233
    >>> print(round(res["p_value"], 4))
    0.1067
    """
    if tau is None:
        tau = float(min(np.max(model_a.x), np.max(model_b.x)))

    mu_a = model_a.mean(tau=tau)
    mu_b = model_b.mean(tau=tau)
    var_a = model_a._rmst_variance(tau)
    var_b = model_b._rmst_variance(tau)

    diff = mu_a - mu_b
    se = float(np.sqrt(var_a + var_b))  # groups are independent
    z_crit = norm.ppf(1 - alpha_ci / 2)
    if se > 0:
        p_value = float(2.0 * (1.0 - norm.cdf(abs(diff) / se)))
    else:
        p_value = float("nan")

    return {
        "difference": float(diff),
        "se": se,
        "lower": float(diff - z_crit * se),
        "upper": float(diff + z_crit * se),
        "p_value": p_value,
        "ratio": float(mu_a / mu_b) if mu_b != 0 else float("nan"),
        "rmst_a": float(mu_a),
        "rmst_b": float(mu_b),
        "tau": float(tau),
    }
