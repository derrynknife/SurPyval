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

    def fit_bayes(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        *,
        priors: Any = None,
        fixed: dict[str, float] | None = None,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
        importance: bool = False,
        method: str = "auto",
        **options: Any,
    ) -> Any:
        """
        Fit the regression by Bayes' rule: priors on the coefficients and
        on the baseline's parameters times the likelihood ``fit`` uses,
        approximated by a normal at the posterior's peak (the Laplace
        engine, checked against the exact posterior). For the
        proportional hazards, accelerated failure time and accelerated
        life fitters. For the exact posterior, sample it with PyMC
        through ``surpyval.pymc``.

        Parameters
        ----------
        x, Z, c, n, t, tl, tr
            The data, as for ``fit``: any censoring and truncation, and
            one row of covariates per row of ``x``.
        priors : dict, joint prior or list, optional
            A dict gives a prior for each parameter it names; a joint
            prior is a belief about quantities of the regression
            (``surpyval.prior.on_quantities`` with
            ``surpyval.prior.on_function``: a B10 life at a use stress),
            an earlier fit (``surpyval.prior.from_model``) or an earlier
            fit's data (``surpyval.prior.from_fit``); a list combines
            them. In a dict: a coefficient by its
            name (``coef_0``, or the covariate's column for
            ``fit_bayes_from_df``), a baseline parameter by its own
            (``beta``), or an accelerated life model's parameter
            (``a``). Each is what ``Weibull.fit_bayes`` takes for a
            parameter: ``surpyval.prior.around(value, between=(low,
            high))``, ``surpyval.prior.between(low, high)``, a SurPyval
            distribution such as ``surpyval.Normal.from_params([0.5,
            0.2])``, or ``surpyval.prior.fixed(value)``. A coefficient of
            ``exp(beta'Z)`` with no prior gets ``Normal(0, 2.5 / sd)``,
            ``sd`` its covariate's standard deviation in the data; any
            other parameter with none is flat on its unconstrained
            scale, allowed where the data determine it. The defaults
            used are in the result's ``default_priors`` and summary.
        fixed : dict, optional
            Parameters held at a value, as for ``fit``.
        importance : bool, optional
            Reweight the normal's draws towards the exact posterior by
            Pareto-smoothed importance sampling, where its check allows
            (Pareto k up to 0.7). False by default.
        method : {'auto', 'laplace', 'grid'}, optional
            The engine. ``'auto'`` (the default) is the Laplace
            approximation for a regression; ``'grid'`` computes the
            exact posterior of one to three free parameters, at a cost
            of tens of thousands of likelihood evaluations.

        Returns
        -------
        surpyval.bayes.BayesianRegression
            ``param_cri(name)``, ``sf``/``ff``/``qf`` and ``cri`` at
            covariates ``Z``, ``compare()``, ``waic()`` and ``loo()``,
            ``diagnose()`` (with posterior predictive checks), and the
            approximation's checks in ``diagnostics`` (``pareto_k``,
            ``shift``, ``approximation``; a poor one warns).

        Not taken, each refused with what to use: weights ``w``, a
        penalty, time-varying covariates, and the other regression
        families.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> rng = np.random.default_rng(1)
        >>> Z = rng.normal(size=(40, 1))
        >>> x = sp.Weibull.random(40, 50.0, 2.0, random_state=2)
        >>> x = x * np.exp(-0.25 * Z[:, 0])
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, priors={"coef_0": sp.Normal.from_params([0.5, 0.2])})
        >>> type(model).__name__
        'BayesianRegression'
        >>> model.param_cri("coef_0").shape
        (2,)
        """
        from surpyval.bayes.regression import fit as _fit

        return _fit(
            self,
            x,
            Z,
            c,
            n,
            t,
            priors=priors,
            fixed=fixed,
            tl=tl,
            tr=tr,
            importance=importance,
            method=method,
            **options,
        )

    def fit_bayes_from_df(
        self,
        df: Any,
        x_col: str | None = None,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        xl_col: str | None = None,
        xr_col: str | None = None,
        tl_col: str | None = None,
        tr_col: str | None = None,
        *,
        priors: Any = None,
        importance: bool = False,
        method: str = "auto",
        **columns: Any,
    ) -> Any:
        """
        :meth:`fit_bayes` on the columns of a DataFrame, named as
        ``fit_from_df`` names them (``tl_col``, ``tr_col``, ``xl_col``,
        ``xr_col``, ``formula`` and ``fixed`` too). A coefficient is then
        named by its covariate's column, in ``priors`` and in the result,
        whose functions take a DataFrame with those columns.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import surpyval as sp
        >>> rng = np.random.default_rng(1)
        >>> df = pd.DataFrame({"load": rng.normal(size=40)})
        >>> df["hours"] = sp.Weibull.random(40, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes_from_df(
        ...     df, x_col="hours", Z_cols=["load"],
        ...     priors={"load": sp.Normal.from_params([0.0, 0.5])})
        >>> model.parameter_names
        ['alpha', 'beta', 'load']
        """
        from surpyval.bayes.regression import fit_from_df as _fit_from_df

        return _fit_from_df(
            self,
            df,
            priors=priors,
            importance=importance,
            method=method,
            x_col=x_col,
            Z_cols=Z_cols,
            c_col=c_col,
            n_col=n_col,
            **{
                k: v
                for k, v in (
                    ("xl_col", xl_col),
                    ("xr_col", xr_col),
                    ("tl_col", tl_col),
                    ("tr_col", tr_col),
                )
                if v is not None
            },
            **columns,
        )


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
