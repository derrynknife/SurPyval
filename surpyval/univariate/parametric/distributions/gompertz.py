from __future__ import annotations

from typing import Any

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from autograd.extend import defvjp, primitive
from autograd.tracer import getval
from scipy.special import exp1

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.surpyval_data import SurpyvalData

from ._quantile import broadcast_values
from ._stable import log1mexp, on_support, positive_or_one

#: Below this ``|a x|`` the cumulative hazard ``(b / a)(e^{a x} - 1)`` is
#: ``b x`` times the series of ``expm1(y) / y``, so it and its derivative
#: in the shape are exact through ``a = 0`` (the difference
#: ``-(b/a^2) expm1(a x) + (b x / a) e^{a x}`` cancels there).
_SERIES_FROM = 1e-3
#: :math:`1 / (n + 1)!`, the series of ``expm1(y) / y``.
_E1_SERIES = (1.0, 1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 720)

#: Below this ``r = a / b`` the mean ``(1 / b) z e^z E1(z)``, ``z = 1 / r``,
#: is the asymptotic series ``sum (-1)^n n! r^n`` (twenty terms: the
#: smallest is below 3e-16 there).
_MEAN_SERIES_BELOW = 0.02
_MEAN_SERIES = tuple(
    (-1.0) ** n * float(onp.prod(onp.arange(1, n + 1))) for n in range(20)
)


@primitive
def _z_exp_e1(z: Boxable) -> Boxable:
    r""":math:`z e^{z} E_1(z)` for :math:`z > 0`, which tends to 1 as
    :math:`z \to \infty`; :math:`E_1` the exponential integral."""
    z = onp.asarray(z, dtype=float)
    with onp.errstate(over="ignore", invalid="ignore"):
        direct = z * onp.exp(z) * exp1(z)
    # e^z overflows from z = 709; there the asymptotic series
    big = z > 600
    if onp.any(big):
        r = 1.0 / onp.where(big, z, 1.0)
        series = onp.zeros_like(r)
        for c in reversed(_MEAN_SERIES):
            series = series * r + c
        direct = onp.where(big, series, direct)
    return direct[()] if direct.ndim == 0 else direct


# d/dz [z e^z E1(z)] = (1 + 1/z) z e^z E1(z) - 1
defvjp(_z_exp_e1, lambda ans, z: lambda g: g * ((1.0 + 1.0 / z) * ans - 1.0))


class Gompertz_(OptimisedFitMixin, ParametricFitter):
    r"""
    The Gompertz distribution: a hazard that grows (or shrinks)
    exponentially with age, :math:`h(x) = b e^{a x}`. It is the classic
    model of human and animal adult mortality, and of wear-out where the
    risk compounds at a constant rate.

    The parameters are those of R's ``flexsurv`` (``gompertz``) and
    ``eha``: the ``shape`` :math:`a`, any real number, and the ``rate``
    :math:`b`, positive. On the support :math:`(0, \infty)`,

    .. math::
        R(x) = \exp\left\{-\frac{b}{a}\left(e^{a x} - 1\right)\right\},

    with :math:`R(x) = e^{-b x}` at ``shape = 0``: the exponential.

    **A negative shape is allowed, and means a plateau.** With
    :math:`a < 0` the hazard falls towards 0 and the cumulative hazard
    levels off at :math:`-b/a`, so a fraction :math:`e^{b/a}` of units
    never fails: the survival function has a floor, as in a cure model.
    Then ``mean`` is infinite, ``qf`` is infinite above the fraction that
    fails, and ``random`` draws ``inf`` for a unit that never fails.

    ``fit`` estimates the parameters from data (which may be censored
    and truncated); ``from_params`` builds the model from known values.

    Examples
    --------
    >>> from surpyval import Gompertz
    >>> model = Gompertz.from_params([0.1, 0.01])
    >>> model.sf([10, 20, 40]).round(4)
    array([0.8421, 0.5279, 0.0047])
    """

    # The Parameters entries the method docstrings below repeat. Edit them
    # here and run scripts/sync_docstrings.py --write to update the copies.
    _param_docs = """
        shape : numpy array or scalar
            shape parameter (the hazard's growth rate) for the Gompertz
            distribution
        rate : numpy array or scalar
            rate parameter (the hazard at time 0) for the Gompertz
            distribution
        """

    # How the parameters move with the time (rescale_time): both are rates
    # (per unit of time).
    _time_roles = ("rate", "rate")

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((None, None), (0, None)),
            support=(0, np.inf),
            parameter_names=["shape", "rate"],
            param_map={"shape": 0, "rate": 1},
            plot_x_scale="linear",
        )
        # No linearising probability plot without the answer: the
        # straight-line axis ln(1 - (a/b) ln(1 - F)) needs a / b.
        # ``plot()`` draws on the fitted parameters.
        self.supports_mpp = False

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        if offset:
            return self._offset_seed(data)
        # The probability plot: log H(x) = log(b / a) + log(e^{a x} - 1)
        # is not linear, so seed from the exponential (a = 0) of the data
        # and a growth rate of one over the mean time.
        x = onp.asarray(data.x, dtype=float)
        n = onp.asarray(data.n, dtype=float)
        c = onp.asarray(data.c)
        events = float(n[c == 0].sum()) or 1.0
        total = float(onp.sum(onp.where(onp.isfinite(x), x, 0.0) * n))
        rate = events / total if total > 0 else 1.0
        scale = total / float(n.sum()) if total > 0 else 1.0
        return onp.array([1.0 / scale, rate / 2.0])

    @staticmethod
    def _Hf(x: Boxable, shape: Boxable, rate: Boxable) -> Boxable:
        r""":math:`H = (b/a)(e^{a x} - 1)`, as :math:`b x (1 + y/2 + y^2/6
        + \cdots)` for :math:`|y| = |a x| < 10^{-3}`; at :math:`x = \infty`
        its limit, :math:`\infty`, or :math:`-b/a` for a negative shape."""
        at_inf = onp.isposinf(onp.asarray(getval(x), dtype=float))
        if onp.any(at_inf):
            finite = Gompertz_._Hf(np.where(at_inf, 1.0, x), shape, rate)
            negative = shape < 0
            plateau = -rate / np.where(negative, shape, -1.0)
            limit = np.where(negative, plateau, np.inf)
            return np.where(at_inf, limit, finite)
        y = shape * x
        small = np.abs(y) < _SERIES_FROM
        series: Any = _E1_SERIES[-1]
        if np.all(small):
            for c in reversed(_E1_SERIES[:-1]):
                series = series * y + c
            return rate * x * series
        ys = np.where(small, y, 0.0)
        for c in reversed(_E1_SERIES[:-1]):
            series = series * ys + c
        a_l = np.where(small, 1.0, shape + np.zeros_like(y))
        x_l = np.where(small, 0.0, x + np.zeros_like(y))
        with onp.errstate(over="ignore"):
            direct = rate / a_l * np.expm1(a_l * x_l)
        over = onp.isposinf(onp.asarray(getval(direct), dtype=float))
        if onp.any(over):
            # H is inf there; its gradient, inf - inf, would be nan and
            # leak through every mask downstream, so the inf is put in
            # with a zero gradient and H formed at x = 0 beneath it.
            x_l = np.where(over, 0.0, x_l)
            direct = np.where(over, np.inf, rate / a_l * np.expm1(a_l * x_l))
        return np.where(small, rate * x * series, direct)

    def Hf(self, x: Numeric, shape: Boxable, rate: Boxable) -> Boxable:
        r"""

        Cumulative hazard rate for the Gompertz distribution:

        .. math::
            H(x) = \frac{b}{a}\left(e^{a x} - 1\right)

        (:math:`b x` at :math:`a = 0`).

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        shape : numpy array or scalar
            shape parameter (the hazard's growth rate) for the Gompertz
            distribution
        rate : numpy array or scalar
            rate parameter (the hazard at time 0) for the Gompertz
            distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gompertz
        >>> x = np.array([1, 5, 10, 20])
        >>> Gompertz.Hf(x, 0.1, 0.01)
        array([0.01051709, 0.06487213, 0.17182818, 0.63890561])
        """
        x_pos = positive_or_one(x)
        return on_support(x, self._Hf(x_pos, shape, rate), 0.0)

    def hf(self, x: Numeric, shape: Boxable, rate: Boxable) -> Boxable:
        r"""

        Instantaneous hazard rate for the Gompertz distribution:

        .. math::
            h(x) = b e^{a x}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        shape : numpy array or scalar
            shape parameter (the hazard's growth rate) for the Gompertz
            distribution
        rate : numpy array or scalar
            rate parameter (the hazard at time 0) for the Gompertz
            distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gompertz
        >>> x = np.array([1, 5, 10, 20])
        >>> Gompertz.hf(x, 0.1, 0.01)
        array([0.01105171, 0.01648721, 0.02718282, 0.07389056])
        """
        with onp.errstate(over="ignore"):
            return rate * np.exp(shape * x)

    def sf(self, x: Numeric, shape: Boxable, rate: Boxable) -> Boxable:
        r"""

        Survival (or reliability) function for the Gompertz distribution:

        .. math::
            R(x) = \exp\left\{-\frac{b}{a}\left(e^{a x} - 1\right)\right\}

        which levels off at :math:`e^{b/a}` for a negative shape.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        shape : numpy array or scalar
            shape parameter (the hazard's growth rate) for the Gompertz
            distribution
        rate : numpy array or scalar
            rate parameter (the hazard at time 0) for the Gompertz
            distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gompertz
        >>> x = np.array([1, 5, 10, 20])
        >>> Gompertz.sf(x, 0.1, 0.01)
        array([0.98953802, 0.93718709, 0.84213437, 0.52786622])
        """
        return np.exp(-self.Hf(x, shape, rate))

    def ff(self, x: Numeric, shape: Boxable, rate: Boxable) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the Gompertz
        distribution, :math:`F(x) = 1 - e^{-H(x)}`, from ``expm1`` so that
        it keeps its digits where it is small.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        shape : numpy array or scalar
            shape parameter (the hazard's growth rate) for the Gompertz
            distribution
        rate : numpy array or scalar
            rate parameter (the hazard at time 0) for the Gompertz
            distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gompertz
        >>> x = np.array([1, 5, 10, 20])
        >>> Gompertz.ff(x, 0.1, 0.01)
        array([0.01046198, 0.06281291, 0.15786563, 0.47213378])
        """
        return -np.expm1(-self.Hf(x, shape, rate))

    def df(self, x: Numeric, shape: Boxable, rate: Boxable) -> Boxable:
        r"""

        Density function for the Gompertz distribution:

        .. math::
            f(x) = b e^{a x} \exp\left\{-\frac{b}{a}\left(e^{a x} - 1
            \right)\right\}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        shape : numpy array or scalar
            shape parameter (the hazard's growth rate) for the Gompertz
            distribution
        rate : numpy array or scalar
            rate parameter (the hazard at time 0) for the Gompertz
            distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gompertz
        >>> x = np.array([1, 5, 10, 20])
        >>> Gompertz.df(x, 0.1, 0.01)
        array([0.01093609, 0.01545172, 0.02289167, 0.03900567])
        """
        return np.exp(self.log_df(x, shape, rate))

    def log_df(self, x: Numeric, shape: Boxable, rate: Boxable) -> Boxable:
        # ln b + a x - H. Far out, where H is infinite, -inf (not the
        # inf - inf of the formula); at x = inf, -inf for any shape.
        x_pos = positive_or_one(x)
        at_inf = onp.isposinf(onp.asarray(getval(x_pos), dtype=float))
        if onp.any(at_inf):
            x_pos = np.where(at_inf, 1.0, x_pos)
        H = self._Hf(x_pos, shape, rate)
        gone = onp.isposinf(onp.asarray(getval(H), dtype=float)) | at_inf
        if onp.any(gone):
            x_in = np.where(gone, 1.0, x_pos)
            H_in = np.where(gone, 0.0, H)
            inside = np.where(
                gone, -np.inf, np.log(rate) + shape * x_in - H_in
            )
        else:
            inside = np.log(rate) + shape * x_pos - H
        return on_support(x, inside, lambda: np.log(rate))

    def log_sf(self, x: Numeric, shape: Boxable, rate: Boxable) -> Boxable:
        return -self.Hf(x, shape, rate)

    def log_ff(self, x: Numeric, shape: Boxable, rate: Boxable) -> Boxable:
        x_pos = positive_or_one(x)
        H = self._Hf(x_pos, shape, rate)
        inside, _ = log1mexp(H, np.log(H))
        return on_support(x, inside, -np.inf)

    def _time_at_H(self, H: npt.NDArray, shape: Any, rate: Any) -> Any:
        r"""The time at which the cumulative hazard is ``H``:
        :math:`\ln(1 + a H / b) / a`, as :math:`(H/b) \ln(1+y)/y` with
        :math:`y = a H / b`; infinite past the plateau :math:`-b/a` of a
        negative shape."""
        y = shape * H / rate
        with onp.errstate(divide="ignore", invalid="ignore"):
            ratio = onp.where(
                onp.abs(y) < 1e-8,
                1.0 - y / 2.0 + y * y / 3.0,
                onp.log1p(y) / onp.where(y == 0, 1.0, y),
            )
            t = H / rate * ratio
        # at H = inf (u = 1, s = 0) the ratio is inf / inf or 0 * inf
        return onp.where((y <= -1.0) | onp.isposinf(H), onp.inf, t)

    def qf(self, u: Numeric, shape: Boxable, rate: Boxable) -> Boxable:
        r"""

        Quantile function for the Gompertz distribution:

        .. math::
            q(u) = \frac{1}{a} \ln\left(1 - \frac{a}{b}\ln(1 - u)\right)

        infinite for :math:`u \geq 1 - e^{b/a}` when :math:`a < 0` (past
        the fraction that ever fails).

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        shape : numpy array or scalar
            shape parameter (the hazard's growth rate) for the Gompertz
            distribution
        rate : numpy array or scalar
            rate parameter (the hazard at time 0) for the Gompertz
            distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Gompertz distribution at each value u

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gompertz
        >>> u = np.array([.1, .5, .9])
        >>> Gompertz.qf(u, 0.1, 0.01)
        array([ 7.19596862, 20.70838618, 31.79130373])
        """
        u, a, b = broadcast_values(u, shape, rate)
        H = -onp.log1p(-u)
        return self._time_at_H(H, a, b)

    def isf(self, s: Numeric, shape: Boxable, rate: Boxable) -> Boxable:
        r"""The inverse survival function: the time at which the survival
        function is ``s``, :math:`\ln(1 - (a/b) \ln s) / a`.

        Examples
        --------
        >>> from surpyval import Gompertz
        >>> Gompertz.isf([0.5, 1e-300], 0.1, 0.01)
        array([20.70838618, 88.40544767])
        """
        s, a, b = broadcast_values(s, shape, rate)
        with onp.errstate(divide="ignore"):
            H = -onp.log(s)
        return self._time_at_H(H, a, b)

    def mean(self, shape: Boxable, rate: Boxable) -> Boxable:
        r"""

        The mean of the Gompertz distribution:

        .. math::
            E = \frac{1}{a} e^{b/a} E_1\!\left(\frac{b}{a}\right)

        for :math:`a > 0`, with :math:`E_1` the exponential integral;
        :math:`1/b` at :math:`a = 0`, and infinite for :math:`a < 0`, when
        a fraction never fails.

        Examples
        --------
        >>> from surpyval import Gompertz
        >>> round(float(Gompertz.mean(0.1, 0.01)), 6)
        20.146425
        """
        if getval(shape) < 0:
            return np.inf
        r = shape / rate
        if getval(r) < _MEAN_SERIES_BELOW:
            series: Any = _MEAN_SERIES[-1]
            for c in reversed(_MEAN_SERIES[:-1]):
                series = series * r + c
            return series / rate
        return _z_exp_e1(1.0 / r) / rate

    def moment(self, m: int, shape: Boxable, rate: Boxable) -> Boxable:
        r"""
        The ``m``-th raw moment :math:`E[X^m] = m \int_0^\infty x^{m-1}
        R(x)\,dx`, by quadrature; infinite for a negative shape (a
        fraction never fails).

        Examples
        --------
        >>> from surpyval import Gompertz
        >>> round(float(Gompertz.moment(2, 0.1, 0.01)), 4)
        488.9605
        """
        from scipy.integrate import quad

        a, b = float(getval(shape)), float(getval(rate))
        if a < 0:
            return np.inf
        if m == 0:
            return 1.0
        if a == 0:
            # the exponential's m! / b^m
            return float(onp.prod(onp.arange(1, m + 1))) / b**m
        # on a log axis around the median, where the integrand is smooth
        med = float(self.qf(0.5, a, b))

        def integrand(y: float) -> float:
            # m t^m R(t) on the log axis, in logs: t^m overflows where
            # R(t) has long underflowed
            log_t = onp.log(med) + y
            with onp.errstate(over="ignore"):
                H = float(self._Hf(onp.exp(log_t), a, b))
            return float(onp.exp(onp.log(m) + m * log_t - H))

        lo = quad(integrand, -onp.inf, 0.0, epsabs=0, epsrel=1e-13)[0]
        hi = quad(integrand, 0.0, onp.inf, epsabs=0, epsrel=1e-13)[0]
        return lo + hi

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return x

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        # ln(1 - (a/b) ln(1 - F)) is a times x: a straight line through
        # the origin of slope a, on the fitted a / b
        a, b = (float(getval(p)) for p in params[-2:])
        y = onp.asarray(y, dtype=float)
        out = onp.full_like(y, onp.nan)
        inside = (y > 0) & (y < 1)
        H = -onp.log1p(-y[inside])
        out[inside] = self._time_at_H(H, a, b)
        return out

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        a, b = (float(getval(p)) for p in params[-2:])
        return -onp.expm1(-onp.asarray(self._Hf(onp.asarray(y), a, b)))


Gompertz: Gompertz_ = Gompertz_("Gompertz")
