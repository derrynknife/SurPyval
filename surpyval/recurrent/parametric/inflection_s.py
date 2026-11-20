from typing import Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.recurrent.parametric.counting_process import Boxable
from surpyval.utils.fitter import singleton_fitter

from .software_growth import GrowthNHPPFitter, half_time


@singleton_fitter
class InflectionS(GrowthNHPPFitter):
    """
    Ohba's inflection S-shaped software reliability growth model, a
    non-homogeneous Poisson process with

    .. math::
        \\Lambda(t) = a\\, \\frac{1 - e^{-b t}}{1 + \\psi e^{-b t}},
        \\qquad
        \\lambda(t) = \\frac{a b (1 + \\psi)\\, e^{-b t}}
        {\\left(1 + \\psi e^{-b t}\\right)^{2}}.

    ``a`` is the expected number of faults that would be found in all,
    ``b`` a rate per unit of testing time and ``psi`` the inflection
    parameter: Ohba's ``(1 - r) / r``, where ``r`` is the share of the
    faults that can be found from the start (the rest are found only
    once others have been removed). With ``psi = 0`` it is
    ``GoelOkumoto``; with ``psi > 1`` the intensity rises to a peak at
    ``t = log(psi) / b`` before it falls, so the count is S-shaped.
    ``psi`` may be as low as -1 (below 0 the intensity falls faster at
    first than Goel-Okumoto's), so that a Goel-Okumoto shape is inside
    the model's range and not on its edge. ``InflectionS`` is an instance
    of this class; ``fit`` and ``from_params`` return a
    ``ParametricRecurrenceModel``.

    Three parameters need more data than two: with few failures ``psi``
    is poorly determined (see its standard error), and ``GoelOkumoto`` or
    ``YamadaDelayedS`` may do as well (compare ``aic()``). Data that show
    no growth are fitted no better than by a constant or an exponentially
    rising rate, the model's limits, and the fit is refused, saying so.
    ``inv_cif`` is ``inf`` for a count of ``a`` or more.

    References
    ----------
    Ohba, M. (1984), "Inflection S-shaped software reliability growth
    model", in Stochastic Models in Reliability Theory (eds S. Osaki and
    Y. Hatoyama), Springer, 144-162.

    Examples
    --------
    >>> from surpyval.recurrent import InflectionS
    >>> model = InflectionS.from_params([100, 0.1, 4.0])
    >>> model.cif([5, 10, 50]).round(3)
    array([11.48 , 25.593, 96.719])
    >>> model.iif([5, 10, 50]).round(3)
    array([2.584, 3.011, 0.319])
    >>> model.inv_cif([11.48, 25.593]).round(1)
    array([ 5., 10.])
    """

    #: Its limits are a constant rate (a growing without bound as b falls)
    #: and an exponentially rising one (a and psi growing together), so
    #: there is no one mean time to compare the failures' with: the fit
    #: is compared with both limits' instead.
    _limit_power = None
    _limit_name = "a constant or an exponentially rising rate"
    _edge = "a grows without bound"

    def __init__(self) -> None:
        self.name = "Inflection S-shaped"
        self.parameter_names = ["a", "b", "psi"]
        self.has_scale = True
        self.bounds = ((0, None), (0, None), (-1, None))
        self.support = (0.0, np.inf)

    def cif(self, x: Boxable, *params: Boxable) -> Boxable:
        a, b, psi = params
        return -a * np.expm1(-b * x) / (1.0 + psi * np.exp(-b * x))

    def iif(self, x: Boxable, *params: Boxable) -> Boxable:
        a, b, psi = params
        decay = np.exp(-b * x)
        return a * b * (1.0 + psi) * decay / (1.0 + psi * decay) ** 2

    def log_iif(self, x: Boxable, *params: Boxable) -> Boxable:
        a, b, psi = params
        return (
            np.log(a)
            + np.log(b)
            + np.log1p(psi)
            - b * x
            - 2.0 * np.log1p(psi * np.exp(-b * x))
        )

    def inv_cif(self, N: Boxable, *params: Boxable) -> Boxable:
        a, b, psi = params
        share = np.asarray(N, dtype=float) / a
        reached = share < 1.0
        y = np.where(reached, share, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (np.log1p(psi * y) - np.log1p(-y)) / b
        return np.where(reached, t, np.inf)

    def _limit_log_likelihoods(self, data: Any) -> list[float]:
        import warnings

        from .cox_lewis import CoxLewis
        from .software_growth import _power_law_limit

        limits = [_power_law_limit(data.split_for_nhpp_likelihood(), 1)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rising = CoxLewis.fit_from_recurrent_data(data)  # type: ignore
        if rising.params[1] > 0:
            limits.append(-float(rising._neg_ll(rising.params)))
        return limits

    def _default_start(
        self, data: Any, x_unique: npt.NDArray, mcf_hat: npt.NDArray
    ) -> npt.NDArray:
        # From psi = 1, where the share found is tanh(b t / 2): two thirds
        # of the faults found by the end (a is the last MCF value scaled
        # up by 1.5), and a third at the MCF's half-time.
        total, t_half = half_time(x_unique, mcf_hat)
        return np.array([1.5 * total, np.log(2.0) / t_half, 1.0])
