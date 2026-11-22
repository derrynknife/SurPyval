"""
The base classes of the parametric regression fitters.

Every parametric regression fitter -- proportional hazards, accelerated
failure time, proportional odds, additive hazards and accelerated life --
wraps a distribution, reads some of its functions as its baseline, and
maximises the same likelihood (:func:`~._likelihood.regression_neg_ll`):
:class:`RegressionFitter` holds those. The three families whose covariates
act through :math:`e^{\\beta' Z}` (PH, AFT and PO) also share their whole
fit (:func:`~._fit_skeleton.fit_log_linear`), the inversion of the
baseline behind ``random`` and the penalised fit:
:class:`LogLinearRegressionFitter`. Each family keeps its own covariate
functions, and its own ``fit`` signature and docstring, which call
:meth:`LogLinearRegressionFitter._fit`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy.typing as npt

from surpyval.univariate.parametric.parametric_fitter import Boxable
from surpyval.utils.surpyval_data import SurpyvalData

from ._fit_skeleton import (
    BaselineInversionMixin,
    LogLinearPhi,
    MirroredDistributionAttrs,
    fit_log_linear,
    mirror_distribution,
    optimise_nm_tnc,
)
from ._likelihood import regression_neg_ll
from ._penalised_fit import PenalisedFitMixin
from .parametric_regression_model import ParametricRegressionModel
from .regression_data import DataFrameRegressionMixin, truncation_window


class RegressionFitter(MirroredDistributionAttrs, DataFrameRegressionMixin):
    """
    The base of the parametric regression fitters: the distribution a
    family wraps (its metadata mirrored by
    :func:`~._fit_skeleton.mirror_distribution`), the functions of it the
    family reads as its baseline, and the likelihood every family
    maximises.

    A family names the baseline functions it reads in
    ``_baseline_functions``, and its ``__init__`` calls
    :meth:`_take_distribution`, which keeps each as ``<name>_dist``
    (``Hf_dist``, ``hf_dist``, ...).
    """

    #: The distribution's functions the family reads as its baseline, kept
    #: on the fitter as ``<name>_dist``.
    _baseline_functions: tuple[str, ...] = ("Hf", "hf")
    # Declared for the type checker; each is set (by _take_distribution)
    # only on a family that names it in _baseline_functions.
    Hf_dist: Callable[..., Any]
    hf_dist: Callable[..., Any]
    sf_dist: Callable[..., Any]
    ff_dist: Callable[..., Any]
    df_dist: Callable[..., Any]
    log_sf_dist: Callable[..., Any]

    def _take_distribution(self, distribution: Any) -> None:
        """Mirror ``distribution``'s metadata onto this fitter and keep
        its ``_baseline_functions``."""
        mirror_distribution(self, distribution)
        for name in self._baseline_functions:
            setattr(self, name + "_dist", getattr(distribution, name))

    def neg_ll(self, data: SurpyvalData, *params: Boxable) -> Boxable:
        """The negative log-likelihood of ``data`` at ``params`` (the
        distribution parameters followed by the covariate parameters),
        for every kind of censoring and truncation."""
        return regression_neg_ll(self, data, *params)


def _exp_link(pmap: dict[str, int]) -> LogLinearPhi:
    """The covariate link ``exp(beta'Z)`` with the parameter map
    ``pmap``."""
    return LogLinearPhi(LogLinearPhi.NAME_EXP, pmap)


class LogLinearRegressionFitter(
    RegressionFitter, BaselineInversionMixin, PenalisedFitMixin
):
    """
    The base of the PH, AFT and PO fitters, whose covariates act through
    ``exp(beta'Z)`` (or, for a custom PH fitter, its own ``phi``): one
    fit for the three (:meth:`_fit`), of the family's kind
    (``_penalised_kind``, which the penalised fit reads too).
    """

    def _fit_options(self) -> dict[str, Any]:
        """The family's options for
        :func:`~._fit_skeleton.fit_log_linear`: the search and the
        covariate link. The Nelder-Mead and TNC search and the
        ``exp(beta'Z)`` link, unless the family says otherwise."""
        return {"optimiser": optimise_nm_tnc, "reg_model": _exp_link}

    def _fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None,
        n: npt.ArrayLike | None,
        t: npt.ArrayLike | None,
        init: npt.ArrayLike | None,
        fixed: dict[str, float] | None,
        center: bool,
        tl: npt.ArrayLike | None,
        tr: npt.ArrayLike | None,
        w: npt.ArrayLike | float | None,
    ) -> ParametricRegressionModel:
        """The family's ``fit``, after its decorators have read the
        data: the truncation window from ``t``, ``tl`` and ``tr``, and
        the maximum likelihood fit."""
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
            w=w,
            kind=self._penalised_kind,
            **self._fit_options(),
        )
