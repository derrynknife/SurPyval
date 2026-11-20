import math
from typing import Any

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from scipy.special import lambertw

from surpyval.recurrent.parametric.counting_process import Boxable
from surpyval.utils.fitter import singleton_fitter

from .software_growth import GrowthNHPPFitter, half_time

#: ``(-1)**k (k - 1) / k!`` for ``k = 2, 3, ...``: the series of ``1 -
#: (1 + u) exp(-u)`` over ``u**2`` (to 1e-20 below ``u`` of 1/2)
_SERIES = tuple((-1) ** k * (k - 1) / math.factorial(k) for k in range(2, 21))


def _found_share(u: Boxable) -> Boxable:
    """``1 - (1 + u) exp(-u)`` for ``u >= 0``: the share of the faults
    found by ``u = b t``. The direct form cancels for a small ``u``
    (where the share is ``u**2 / 2``), so below 1/2 it is the series.
    Each branch is evaluated only where it is taken."""
    small = u < 0.5
    u_small = np.where(small, u, 0.0)
    series = 0.0
    for coeff in reversed(_SERIES):
        series = series * u_small + coeff
    near = u_small * u_small * series
    # exp(-u) is 0 from 746 on, where (1 + u) exp(-u) would be inf * 0
    u_far = np.minimum(np.where(small, 1.0, u), 750.0)
    far = 1.0 - (1.0 + u_far) * np.exp(-u_far)
    return np.where(small, near, far)


@singleton_fitter
class YamadaDelayedS(GrowthNHPPFitter):
    """
    The delayed S-shaped software reliability growth model of Yamada,
    Ohba and Osaki, a non-homogeneous Poisson process with

    .. math::
        \\Lambda(t) = a \\left(1 - (1 + b t)\\, e^{-b t}\\right), \\qquad
        \\lambda(t) = a b^{2}\\, t\\, e^{-b t}.

    ``a`` is the expected number of faults that would be found in all and
    ``b`` a rate per unit of testing time. The intensity starts at 0,
    rises to its peak at ``t = 1 / b`` and then falls: finding a fault
    takes two steps (a failure, then isolating its cause), so the count
    is slow to start. Fit it to failure times or to counts per period, as
    ``GoelOkumoto``. ``YamadaDelayedS`` is an instance of this class;
    ``fit`` and ``from_params`` return a ``ParametricRecurrenceModel``.

    Data whose failures come no earlier, on average, than an intensity
    still rising in proportion to time would have them (the model before
    its peak) have no finite ``a``, and the fit is refused, saying so.
    ``inv_cif`` is ``inf`` for a count of ``a`` or more.

    References
    ----------
    Yamada, S., Ohba, M. and Osaki, S. (1983), "S-shaped reliability
    growth modeling for software error detection", IEEE Transactions on
    Reliability R-32(5), 475-484.

    Examples
    --------
    >>> from surpyval.recurrent import YamadaDelayedS
    >>> model = YamadaDelayedS.from_params([100, 0.1])
    >>> model.cif([5, 10, 50]).round(3)
    array([ 9.02 , 26.424, 95.957])
    >>> model.iif([5, 10, 50]).round(3)
    array([3.033, 3.679, 0.337])
    >>> model.inv_cif([9.02, 26.424]).round(2)
    array([ 5., 10.])
    """

    _limit_power = 2
    _limit_name = "an intensity that rises in proportion to time"

    def __init__(self) -> None:
        self.name = "Yamada delayed S-shaped"
        self.parameter_names = ["a", "b"]
        self.has_scale = True
        self.bounds = ((0, None), (0, None))
        self.support = (0.0, np.inf)

    def cif(self, x: Boxable, *params: Boxable) -> Boxable:
        return params[0] * _found_share(params[1] * x)

    def iif(self, x: Boxable, *params: Boxable) -> Boxable:
        b = params[1]
        # b t exp(-b t), 0 at an infinite t (not inf * 0)
        u = np.minimum(b * x, 750.0)
        return params[0] * b * u * np.exp(-u)

    def log_iif(self, x: Boxable, *params: Boxable) -> Boxable:
        b = params[1]
        return np.log(params[0]) + 2.0 * np.log(b) + np.log(x) - b * x

    def inv_cif(self, N: Boxable, *params: Boxable) -> Boxable:
        # (1 + u) exp(-u) = 1 - N / a has the root u = -1 - W(-(1 - N / a)
        # / e) on the lower branch of Lambert's W. W loses digits near its
        # branch point (a small N / a), so the root is then polished by
        # Newton's steps on log1p(u) - u = log1p(-N / a).
        a, b = float(params[0]), float(params[1])
        share = onp.asarray(N, dtype=float) / a
        inside = (share > 0.0) & (share < 1.0)
        s = onp.where(inside, share, 0.5)
        target = onp.log1p(-s)
        with onp.errstate(all="ignore"):
            u = -1.0 - lambertw(-onp.exp(target - 1.0), k=-1).real
            # From the series' root where the share is small: W has
            # lost its digits there
            root = onp.sqrt(2.0 * s)
            u = onp.where(s < 1e-3, root * (1.0 + root / 3.0), u)
            for _ in range(6):
                # log1p(u) - u, as its series -u**2 / 2 + u**3 / 3 - ...
                # (to 1e-19 below u of 0.1) where it would cancel
                near = onp.minimum(u, 0.1)
                series = onp.zeros_like(near)
                for k in range(18, 1, -1):
                    series = series * near + (-1.0) ** (k + 1) / k
                gap = onp.where(
                    u < 0.1, near * near * series, onp.log1p(u) - u
                )
                u = u + (gap - target) * (1.0 + u) / u
        out = onp.where(inside, u / b, onp.where(share >= 1.0, onp.inf, 0.0))
        return onp.where(share < 0.0, onp.nan, out)

    def _default_start(
        self, data: Any, x_unique: npt.NDArray, mcf_hat: npt.NDArray
    ) -> npt.NDArray:
        # Two thirds of the faults found by the end (a is the last MCF
        # value scaled up by 1.5); 1 - (1 + u) exp(-u) is a third at u =
        # 1.19, which b puts at the MCF's half-time.
        total, t_half = half_time(x_unique, mcf_hat)
        return np.array([1.5 * total, 1.19 / t_half])
