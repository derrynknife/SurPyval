from __future__ import annotations

from typing import Any

import numpy.typing as npt

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
)
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils.surpyval_data import SurpyvalData

from .._fit_skeleton import (
    HazardIdentitiesMixin,
    LogLinearPhi,
    fit_log_linear,
    mirror_distribution,
    optimise_nm_tnc,
    split_log_linear,
)
from .._kinds import ACCELERATED_FAILURE_TIME
from .._likelihood import regression_neg_ll
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import DataFrameRegressionMixin, truncation_window
from .aft_tvc_fit import AFTTVCFitMixin, _aft_tvc_neg_ll


class AFTFitter(
    HazardIdentitiesMixin,
    AFTTVCFitMixin,
    DataFrameRegressionMixin,
):
    """
    Accelerated Failure Time fitter using :math:`e^{\\beta' Z}` as the
    acceleration factor. The covariates rescale time:

    .. math::
        H(x \\mid Z) = H_0\\left(e^{\\beta' Z} x\\right), \\qquad
        R(x \\mid Z) = R_0\\left(e^{\\beta' Z} x\\right).

    A positive beta coefficient means higher covariate values accelerate
    failure (shorter life), consistent with the PH sign convention. (Many
    texts write the AFT model with :math:`e^{-\\beta' Z}`, so their
    coefficients have the opposite sign.)

    Use the pre-built instances (``WeibullAFT``, ``LogNormalAFT``, ...) or
    the ``AFT`` factory.
    """

    #: The ``repr`` (#614)
    fitter_kind = "accelerated failure time fitter"
    name_suffix = "AFT"

    def __init__(self, distribution: Any) -> None:
        mirror_distribution(self, distribution)
        self.Hf_dist = distribution.Hf
        self.hf_dist = distribution.hf
        self.sf_dist = distribution.sf
        self.ff_dist = distribution.ff

    def _phi(self, Z: Numeric, *phi_params: Boxable) -> Boxable:
        return LogLinearPhi.phi(Z, *phi_params)

    def Hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Cumulative hazard :math:`H_0(e^{\\beta' Z} x)` at ``x`` for
        covariates ``Z``; ``params`` are the distribution parameters
        followed by the covariate coefficients.
        """
        x, dist_params, phi = split_log_linear(self, x, Z, params)
        return self.Hf_dist(phi * x, *dist_params)

    def hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Hazard rate :math:`e^{\\beta' Z} h_0(e^{\\beta' Z} x)` at ``x`` for
        covariates ``Z``; ``params`` as for :meth:`Hf`.
        """
        x, dist_params, phi = split_log_linear(self, x, Z, params)
        return phi * self.hf_dist(phi * x, *dist_params)

    def neg_ll(self, data: SurpyvalData, *params: Boxable) -> Boxable:
        return regression_neg_ll(self, data, *params)

    @dataframe_covariates
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        init: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
        center: bool = False,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
    ) -> ParametricRegressionModel:
        """
        Fit the accelerated failure time model by maximum likelihood.

        Parameters
        ----------

        x : array_like
            The observed event times.
        Z : array_like
            The covariate matrix, one row per observation (a 1-D array is
            read as a single covariate). Rows with a missing or infinite
            covariate are dropped, with a warning.
        c : array_like, optional
            The censoring indicators (0 observed, 1 right, -1 left, 2
            interval). Defaults to all observed.
        n : array_like, optional
            The count of observations at each time. Defaults to 1.
        t : array_like, optional
            Truncation bounds: an (N, 2) array of the left and right
            truncation times of each observation.
        tl, tr : array_like or float, optional
            The left / right truncation times of each observation (or one
            for every observation), the columns of ``t``, which they
            replace (#662).
        init : array_like, optional
            Initial parameter values: the distribution parameters followed
            by the covariate coefficients.
        fixed : dict, optional
            Parameters to hold fixed, by name (a distribution parameter
            such as ``"beta"``, or a coefficient: its covariate's column
            name, else ``"coef_0"``, ...).
        center : bool, optional
            ``False`` (the default) reports the baseline at ``Z = 0``.
            ``True`` reports the baseline at the covariate means (stored as
            ``model.center``) instead: the fit runs on ``Z - center``, and
            ``init`` and ``fixed`` are read there too. Use it for
            covariates far from 0 (a year, a date), where the baseline at
            ``Z = 0`` cannot be represented or fitted, which the default
            fit refuses with a ``ValueError`` saying so.

        Returns
        -------

        ParametricRegressionModel
            The fitted model.

        Examples
        --------

        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullAFT
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullAFT.fit(x, Z)
        >>> model.params.round(3)
        array([9.629, 1.751, 0.473])
        """
        t = truncation_window(x, t, tl, tr)
        return fit_log_linear(
            self,
            x,
            Z,
            c,
            n,
            t,
            init,
            fixed,
            center,
            kind=ACCELERATED_FAILURE_TIME,
            optimiser=optimise_nm_tnc,
            reg_model=lambda pmap: LogLinearPhi(LogLinearPhi.NAME_EXP, pmap),
        )


class AFTTVCFitter(AFTFitter):
    """The fitter a ``fit_tvc`` model carries: an :class:`AFTFitter` whose
    ``neg_ll`` is the accumulated-age likelihood of the episodes it holds
    (``_tvc``), so the model's bounds use that likelihood. A class rather
    than a method bound onto one instance, so the model pickles
    (#573)."""

    _tvc: dict
    neg_ll = _aft_tvc_neg_ll  # type: ignore[assignment]


def AFT(distribution: Any) -> "AFTFitter":
    """
    Create an Accelerated Failure Time fitter for the given distribution.

    Uses exp(beta'Z) as the acceleration factor — the standard statistical
    parameterisation for AFT models.

    Parameters
    ----------
    distribution : ParametricFitter
        A surpyval parametric distribution (e.g. ``Weibull``, ``LogNormal``).

    Returns
    -------
    AFTFitter
        A configured fitter with a ``.fit(x, Z, ...)`` method.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import Weibull
    >>> from surpyval import AFT
    >>> np.random.seed(1)
    >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
    >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
    >>> model = AFT(Weibull).fit(x, Z=Z)
    >>> model.params.round(3)
    array([9.629, 1.751, 0.473])
    """
    return AFTFitter(distribution)
