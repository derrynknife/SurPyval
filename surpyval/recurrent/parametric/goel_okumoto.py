from typing import Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.recurrent.parametric.counting_process import Boxable
from surpyval.utils.fitter import singleton_fitter

from .software_growth import GrowthNHPPFitter, half_time


@singleton_fitter
class GoelOkumoto(GrowthNHPPFitter):
    """
    The Goel-Okumoto software reliability growth model, a non-homogeneous
    Poisson process with

    .. math::
        \\Lambda(t) = a \\left(1 - e^{-b t}\\right), \\qquad
        \\lambda(t) = a b\\, e^{-b t}.

    ``a`` is the expected number of faults that would be found in all
    (the curve levels off at it) and ``b`` the rate at which each is
    found, per unit of testing time; the intensity falls from ``a b`` by
    the same proportion in every unit of time. Fit it to failure times
    (of one system or several) or to the number of failures in each
    period of testing (``c=2`` rows of ``[start, end]`` with their counts
    ``n``). ``GoelOkumoto`` is an instance of this class; ``fit`` and
    ``from_params`` return a ``ParametricRecurrenceModel``.

    Data that show no growth (failures no earlier, on average, than a
    constant rate would have them) have no finite ``a``, and the fit is
    refused, saying so. ``inv_cif`` is ``inf`` for a count of ``a`` or
    more, which the model never expects to reach.

    References
    ----------
    Goel, A. L. and Okumoto, K. (1979), "Time-dependent error-detection
    rate model for software reliability and other performance measures",
    IEEE Transactions on Reliability R-28(3), 206-211.

    Examples
    --------
    The numbers of failures found in each of eight weeks of testing:

    >>> import numpy as np
    >>> from surpyval.recurrent import GoelOkumoto
    >>> weeks = np.arange(8.0)
    >>> x = np.column_stack([weeks, weeks + 1])
    >>> n = [12, 9, 8, 5, 4, 4, 2, 1]
    >>> model = GoelOkumoto.fit(x, c=[2] * 8, n=n)
    >>> model.params.round(3)
    array([50.017,  0.287])
    >>> model.cif([4, 8, 16]).round(2)
    array([34.17, 45.  , 49.51])
    >>> model.iif(8.0).round(3)
    np.float64(1.442)
    """

    def __init__(self) -> None:
        self.name = "Goel-Okumoto"
        self.parameter_names = ["a", "b"]
        self.has_scale = True
        self.bounds = ((0, None), (0, None))
        self.support = (0.0, np.inf)

    def cif(self, x: Boxable, *params: Boxable) -> Boxable:
        # -expm1(-b t), not 1 - exp(-b t), which rounds for a small b t
        return -params[0] * np.expm1(-params[1] * x)

    def iif(self, x: Boxable, *params: Boxable) -> Boxable:
        return params[0] * params[1] * np.exp(-params[1] * x)

    def log_iif(self, x: Boxable, *params: Boxable) -> Boxable:
        return np.log(params[0]) + np.log(params[1]) - params[1] * x

    def inv_cif(self, N: Boxable, *params: Boxable) -> Boxable:
        share = np.asarray(N, dtype=float) / params[0]
        reached = share < 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -np.log1p(-np.where(reached, share, 0.0)) / params[1]
        return np.where(reached, t, np.inf)

    def _default_start(
        self, data: Any, x_unique: npt.NDArray, mcf_hat: npt.NDArray
    ) -> npt.NDArray:
        # Two thirds of the faults found by the end: a is the last MCF
        # value scaled up by 1.5, and b puts a third of a at the time the
        # MCF reached half its last value.
        total, t_half = half_time(x_unique, mcf_hat)
        return np.array([1.5 * total, np.log(1.5) / t_half])
