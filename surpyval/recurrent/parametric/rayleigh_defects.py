from typing import Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.recurrent.parametric.counting_process import Boxable
from surpyval.utils.fitter import singleton_fitter

from .software_growth import GrowthNHPPFitter, half_time


@singleton_fitter
class RayleighDefects(GrowthNHPPFitter):
    """
    The Rayleigh defect model, a non-homogeneous Poisson process with

    .. math::
        \\Lambda(t) = K \\left(1 - e^{-t^{2} / (2 c^{2})}\\right), \\qquad
        \\lambda(t) = \\frac{K t}{c^{2}}\\, e^{-t^{2} / (2 c^{2})}.

    ``K`` is the number of defects that will be found in all, and ``c``
    the time at which they are found fastest (the peak of the
    intensity); 39% of them are found by ``c`` and 86% by ``2 c``. It is
    used for the defects found across the phases of a development
    (requirements review, design, code inspection, unit test,
    integration, system test), which rise and then fall along such a
    curve: fitted to the phases done so far, it predicts what the later
    phases and the field will find. Time is the phase number, or the
    calendar if the phases' lengths are known;
    :func:`~surpyval.recurrent.phases_to_data` makes the data from a
    table of counts. ``RayleighDefects`` is an instance of this class;
    ``fit`` and ``from_params`` return a ``ParametricRecurrenceModel``,
    whose ``remaining_faults`` is the defects left for the field and
    ``expected_events`` those of a later phase.

    The total cannot be estimated before the counts turn down: data
    whose failures come no earlier (in the mean of their squared times)
    than an intensity still rising in proportion to time would have them
    have no finite ``K``, and the fit is refused, saying so.

    References
    ----------
    Putnam, L. H. (1978), "A general empirical solution to the macro
    software sizing and estimating problem", IEEE Transactions on
    Software Engineering SE-4(4), 345-361.

    Kan, S. H. (2002), Metrics and Models in Software Quality
    Engineering (2nd ed.), Addison-Wesley, chapter 7.

    Examples
    --------
    The defects found in the first five phases of a project:

    >>> from surpyval.recurrent import RayleighDefects, phases_to_data
    >>> data = phases_to_data([18, 44, 52, 41, 24])
    >>> model = RayleighDefects.fit(**data)
    >>> model.params.round(2)
    array([196.74,   2.28])
    >>> model.expected_events(5, 6).round(1)   # the sixth phase
    np.float64(11.6)
    >>> model.remaining_faults(6).round(1)     # after it, for the field
    np.float64(6.2)
    """

    _limit_power = 2
    _limit_moment = 2
    _limit_name = "an intensity that rises in proportion to time"
    _no_estimate = "the total number of defects K"
    _edge = "K and c grow without bound"
    total_parameter = "K"

    def __init__(self) -> None:
        self.name = "Rayleigh defects"
        self.parameter_names = ["K", "c"]
        self.has_scale = True
        self.bounds = ((0, None), (0, None))
        self.support = (0.0, np.inf)

    def cif(self, x: Boxable, *params: Boxable) -> Boxable:
        K, c = params
        return -K * np.expm1(-0.5 * (x / c) ** 2)

    def iif(self, x: Boxable, *params: Boxable) -> Boxable:
        K, c = params
        # (the exponent first: at an infinite time it is 0, not inf * 0)
        u = np.minimum(x / c, 1e150)
        return K / c * u * np.exp(-0.5 * u**2)

    def log_iif(self, x: Boxable, *params: Boxable) -> Boxable:
        K, c = params
        return np.log(K) + np.log(x) - 2.0 * np.log(c) - 0.5 * (x / c) ** 2

    def inv_cif(self, N: Boxable, *params: Boxable) -> Boxable:
        K, c = params
        share = np.asarray(N, dtype=float) / K
        reached = share < 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = c * np.sqrt(-2.0 * np.log1p(-np.where(reached, share, 0.0)))
        return np.where(reached, t, np.inf)

    def _default_start(
        self, data: Any, x_unique: npt.NDArray, mcf_hat: npt.NDArray
    ) -> npt.NDArray:
        # Two thirds of the defects found by the end (K is the last MCF
        # value scaled up by 1.5); 1 - exp(-u**2 / 2) is a third at u =
        # 0.9, which c puts at the MCF's half-time.
        total, t_half = half_time(x_unique, mcf_hat)
        return np.array([1.5 * total, t_half / 0.9])
