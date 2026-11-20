from typing import Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.recurrent.parametric.counting_process import Boxable
from surpyval.utils.fitter import singleton_fitter

from .software_growth import GrowthNHPPFitter, half_time


@singleton_fitter
class MusaOkumoto(GrowthNHPPFitter):
    """
    The Musa-Okumoto logarithmic Poisson execution time model, a
    non-homogeneous Poisson process with

    .. math::
        \\Lambda(t) = \\frac{\\log(1 + \\lambda_0 \\theta t)}{\\theta},
        \\qquad
        \\lambda(t) = \\frac{\\lambda_0}{1 + \\lambda_0 \\theta t}
        = \\lambda_0\\, e^{-\\theta \\Lambda(t)}.

    ``lambda0`` is the intensity at the start and ``theta`` its decay:
    each failure met (and its fault removed) lowers the intensity by the
    proportion ``theta``, so early fixes help most. The expected number
    of failures grows without limit (as the log of time), so the model
    has no total number of faults: it is an infinite-failure model, where
    ``GoelOkumoto`` and the S-shaped models are finite. ``MusaOkumoto``
    is an instance of this class; ``fit`` and ``from_params`` return a
    ``ParametricRecurrenceModel``.

    Data that show no growth (failures no earlier, on average, than a
    constant rate would have them) have no positive ``theta``, and the
    fit is refused, saying so.

    References
    ----------
    Musa, J. D. and Okumoto, K. (1984), "A logarithmic Poisson execution
    time model for software reliability measurement", Proceedings of the
    7th International Conference on Software Engineering, 230-238.

    Examples
    --------
    >>> from surpyval.recurrent import MusaOkumoto
    >>> model = MusaOkumoto.from_params([2.0, 0.05])
    >>> model.cif([5, 10, 50]).round(3)
    array([ 8.109, 13.863, 35.835])
    >>> model.iif([5, 10, 50]).round(3)
    array([1.333, 1.   , 0.333])
    >>> model.inv_cif([8.109, 13.863]).round(2)
    array([ 5., 10.])
    """

    _no_estimate = "the decay theta"
    _edge = "theta falls to 0"
    total_parameter = None

    def __init__(self) -> None:
        self.name = "Musa-Okumoto"
        self.parameter_names = ["lambda0", "theta"]
        self.has_scale = True
        self.bounds = ((0, None), (0, None))
        self.support = (0.0, np.inf)

    def cif(self, x: Boxable, *params: Boxable) -> Boxable:
        lambda0, theta = params
        return np.log1p(lambda0 * theta * x) / theta

    def iif(self, x: Boxable, *params: Boxable) -> Boxable:
        lambda0, theta = params
        return lambda0 / (1.0 + lambda0 * theta * x)

    def log_iif(self, x: Boxable, *params: Boxable) -> Boxable:
        lambda0, theta = params
        return np.log(lambda0) - np.log1p(lambda0 * theta * x)

    def inv_cif(self, N: Boxable, *params: Boxable) -> Boxable:
        lambda0, theta = params
        return np.expm1(theta * np.asarray(N, dtype=float)) / (lambda0 * theta)

    def _default_start(
        self, data: Any, x_unique: npt.NDArray, mcf_hat: npt.NDArray
    ) -> npt.NDArray:
        # The curve through the end of the MCF and its half-time: with k
        # = lambda0 theta, log(1 + k T) = 2 log(1 + k t_half), so k = (T -
        # 2 t_half) / t_half**2 (1 / T where the MCF is not that bent).
        total, t_half = half_time(x_unique, mcf_hat)
        span = float(x_unique[-1])
        k = max((span - 2.0 * t_half) / t_half**2, 1.0 / span)
        theta = np.log1p(k * span) / total
        return np.array([k / theta, theta])
