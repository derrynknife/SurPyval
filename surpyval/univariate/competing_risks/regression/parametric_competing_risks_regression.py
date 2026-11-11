"""
Parametric competing-risks regression.

``ParametricCompetingRisksRegression`` gives every cause of failure its own
parametric regression model of the covariates -- a Weibull proportional
hazards model for one cause, a lognormal accelerated failure time model
for another -- and assembles from them each cause's cumulative incidence
at any covariate row, as a smooth curve that goes on past the last
observation.

It is the regression counterpart of
:class:`~surpyval.univariate.competing_risks.ParametricCompetingRisks`
(one distribution per cause, no covariates) and the parametric
counterpart of the cause-specific Cox model
(:class:`~surpyval.univariate.competing_risks.
CompetingRisksProportionalHazards`), whose cumulative incidences are step
functions that stop at the last event.

The likelihood of cause-specific hazard models **factorises across the
causes**: a unit that failed from cause :math:`k` at :math:`x` contributes
:math:`h_k(x \\mid Z)` times the survival from every cause, and the
survival from every cause is the product of the causes' own
:math:`\\exp\\{-H_j(x \\mid Z)\\}`. So the parameters of cause :math:`k`
are fitted by its own regression with the failures of the other causes
right-censored, exactly. Nothing is assumed about how the causes' latent
failure times depend on each other: the cause-specific hazards, and the
cumulative incidences built from them, are identified by the data either
way (see "Net and crude probabilities" in
:doc:`/Competing Risks Analysis`). Only the *net* functions of one cause
(``sf(x, Z, event=k)``: what would happen were the other causes removed)
need the causes to be independent to mean that.

The cumulative incidence of cause :math:`k` at covariates :math:`Z` is

.. math::
    \\mathrm{CIF}_k(t \\mid Z) = \\int_0^t h_k(u \\mid Z)\\,S(u \\mid Z)\\,du,
    \\qquad S(u \\mid Z) = \\exp\\Big\\{-\\sum_j H_j(u \\mid Z)\\Big\\},

and the causes' incidences sum to the all-cause failure probability
:math:`1 - S(t \\mid Z)`.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad, quad_vec
from scipy.linalg import block_diag
from scipy.optimize import brentq
from scipy.special import ndtri

from surpyval.diagnostics.mixin import DiagnoseMixin
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.competing_risks.labels import (
    label_from_native,
    label_mask,
    ordered_labels,
)
from surpyval.univariate.competing_risks.parametric.parametric_competing_risks import (  # noqa: E501
    _central_differences,
    _checked_sd,
    _ic_terms,
)
from surpyval.univariate.information_criteria import corrected_aic
from surpyval.univariate.regression import WeibullPH
from surpyval.univariate.regression._aliasing import (
    collect_aliased,
    covariate_columns,
    dataframe_covariates,
    warn_collected,
)
from surpyval.univariate.regression.parametric_regression_model import (
    ParametricRegressionModel,
)
from surpyval.univariate.regression.regression_data import (
    DataFrameRegressionMixin,
    design_matrix_from_df,
    restore_covariate_meta,
    serialise_covariate_meta,
)
from surpyval.utils import finite_covariate_mask, is_missing_event
from surpyval.utils.linalg import bound_signs, standard_errors_of
from surpyval.utils.no_maximum import combined_maximum
from surpyval.utils.removed_names import column_arguments
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    flatten_query,
    keeps_query_shape,
)
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import (
    BOUNDS,
    check_alpha_ci,
    check_option,
    unknown_cause_error,
    validate_cif_event,
    warn_outside_unit_interval,
)
from surpyval.utils.warnings import ignoring_warnings

from .competing_risks_proportional_hazard import _entry_or_none
from .fine_gray import _validated

if TYPE_CHECKING:
    import pandas as pd

_CB_ON = ("cif", "sf", "ff", "Hf")
_CB_METHODS = ("wald", "lr", "bootstrap")


def _cause_fitters(model: Any, causes: list) -> dict:
    """The regression fitter of each cause: ``model`` for every one, or
    its entry of a ``{cause: fitter}`` dict; each must be a parametric
    regression fitter."""
    if isinstance(model, dict):
        missing = [k for k in causes if k not in model]
        if missing:
            raise ValueError(
                "`model` has no regression fitter for the cause(s) {}; "
                "give one for every cause {}.".format(missing, causes)
            )
        fitters = {k: model[k] for k in causes}
    else:
        fitters = {k: model for k in causes}
    for k, fitter in fitters.items():
        if not isinstance(fitter, DataFrameRegressionMixin):
            raise ValueError(
                "`model` must be a parametric regression fitter, such as "
                "WeibullPH, ExponentialPH, WeibullAFT, LogNormalAFT or "
                "LogLogisticPO (for every cause, or one per cause in a "
                "dict); got {!r} for cause {!r}. For a semi-parametric "
                "(Cox) model of each cause use "
                "CompetingRisksProportionalHazards.".format(fitter, k)
            )
    return fitters


def _cause_inits(init: Any, causes: list) -> dict:
    """The starting values of each cause's fit: ``init`` for every cause,
    or its entry of a ``{cause: init}`` dict (a cause left out starts at
    its fitter's default)."""
    if isinstance(init, dict):
        unknown = [k for k in init if k not in causes]
        if unknown:
            raise ValueError(
                "`init` has starting values for {}, which are not causes "
                "in the data {}.".format(unknown, causes)
            )
        return {k: init.get(k) for k in causes}
    return {k: init for k in causes}


class ParametricCompetingRisksRegression(DiagnoseMixin, SerialisableMixin):
    """
    A parametric competing-risks regression: one parametric regression
    model per cause, each of that cause's hazard given the covariates,
    combined into cumulative incidence functions of time and covariates.

    Build it from data with :meth:`fit` or :meth:`fit_from_df`. The
    fitted per-cause models are ordinary SurPyval regression models, in
    ``models[cause]``; ``causes`` lists the causes in order. Every
    function takes the times ``x`` and the covariate rows ``Z`` as the
    regressions take them: one row per time, one row used at every time,
    or one time for every row.

    ``event`` picks a cause. Without it ``hf``, ``Hf``, ``sf``, ``ff``
    and ``df`` are the all-cause functions (any failure); with it they are
    the cause's *net* ones (its own model, as if the other causes were
    removed). ``cif(x, Z, event)`` is the cumulative incidence: the
    probability of having failed from that cause by ``x``, with the other
    causes competing.

    ``diagnose()`` on the whole model is not built yet and says so: check
    each cause's own regression, ``models[cause].diagnose()``.
    """

    #: The causes, in sorted order.
    causes: list
    #: Each cause's fitted regression model, by cause.
    models: dict

    # -- covariate metadata, shared with the causes' models ---------------

    @property
    def feature_names(self) -> list | None:
        """The covariates' names (a DataFrame ``Z``, or ``fit_from_df``),
        which name the coefficients and by which a DataFrame of new
        covariates is read; ``None`` for an array ``Z``."""
        return self._first().feature_names

    @feature_names.setter
    def feature_names(self, names: list | None) -> None:
        for model in self.models.values():
            model.feature_names = names

    @property
    def formula(self) -> Any:
        """The formula the covariates were built with (``fit_from_df``
        with ``formula``), else ``None``."""
        return self._first().formula

    @formula.setter
    def formula(self, formula: Any) -> None:
        for model in self.models.values():
            model.formula = formula

    @property
    def _model_spec(self) -> Any:
        return self._first()._model_spec

    @_model_spec.setter
    def _model_spec(self, spec: Any) -> None:
        for model in self.models.values():
            model._model_spec = spec

    def _first(self) -> Any:
        return self.models[self.causes[0]]

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the fitted model to a plain, JSON-serialisable dict:
        the causes and each cause's regression model (its own
        ``to_dict``, with its parameter covariance), and the covariates'
        names and formula. The reloaded model reproduces every
        prediction, and its Wald bounds and summary.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval
        >>> from surpyval import ParametricCompetingRisksRegression
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.weibull(2.0, 200) * 10 * np.exp(-0.4 * Z[:, 0])
        >>> t_b = rng.exponential(20, 200)
        >>> x = np.minimum(t_a, t_b).round(3)
        >>> e = np.where(t_a < t_b, "a", "b")
        >>> model = ParametricCompetingRisksRegression.fit(x, Z, e)
        >>> restored = surpyval.from_dict(model.to_dict())
        >>> bool(np.allclose(restored.cif([5, 9], [1], "a"),
        ...                  model.cif([5, 9], [1], "a")))
        True
        """
        out: dict = {
            "model": "ParametricCompetingRisksRegression",
            "causes": to_native(list(self.causes)),
            "models": [self.models[cause].to_dict() for cause in self.causes],
        }
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> ParametricCompetingRisksRegression:
        """Rebuild a parametric competing-risks regression from a
        :meth:`to_dict` dictionary."""
        require_model_tag(
            model_dict,
            "ParametricCompetingRisksRegression",
            "a parametric competing-risks regression",
        )
        out = cls()
        out.causes = [label_from_native(k) for k in model_dict["causes"]]
        out.models = {
            cause: ParametricRegressionModel.from_dict(sub)
            for cause, sub in zip(out.causes, model_dict["models"])
        }
        restore_covariate_meta(out, model_dict)
        return out

    def __repr__(self) -> str:
        kinds = ", ".join(
            "{}: {} ({})".format(
                k, self.models[k].distribution.name, self.models[k].kind
            )
            for k in self.causes
        )
        return (
            "Parametric Competing Risks Regression SurPyval Model\n"
            "====================================================\n"
            "Causes              : {causes}\n"
            "Cause models        : {kinds}\n"
            "Covariates          : {names}".format(
                causes=list(self.causes),
                kinds=kinds,
                names=", ".join(self._coefficient_names()),
            )
        )

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` whose coefficient the data do not
        determine (a constant column, or a combination of the others):
        their coefficients are ``nan`` in ``params`` and taken as 0 in
        the predictions, as on the causes' own models, whose ``aliased``
        these are (every cause is fitted to the same covariates)."""
        found = [
            np.asarray(self.models[k].aliased, dtype=int).ravel()
            for k in self.causes
        ]
        return np.unique(np.concatenate(found)).astype(int)

    def _coefficient_names(self) -> list[str]:
        return [str(name) for name in self._first()._coefficient_names()]

    # -- helpers -----------------------------------------------------------

    def _check_event(self, event: Any) -> None:
        if event not in self.models:
            raise unknown_cause_error(event, self.causes)

    def _of(self, event: Any) -> list:
        """The models a function is over: every cause's, or ``event``'s."""
        if event is None:
            return [self.models[k] for k in self.causes]
        self._check_event(event)
        return [self.models[event]]

    def _rows(self, Z: Any) -> npt.NDArray:
        """``Z`` as numeric covariate rows ``(m, p)``, read as the
        causes' models read it (a DataFrame by the fitted columns or
        formula)."""
        first = self._first()
        return covariate_rows(first._prepare_Z(Z), first._n_covariates())

    def _pairs(self, x: Any, Z: Any) -> tuple[npt.NDArray, npt.NDArray]:
        """The times and covariate rows paired, each of the same length:
        one row for every time, or one time for every row."""
        t = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        rows = self._rows(Z)
        check_paired_rows(t.size, rows.shape[0], grid=False)
        # (an empty query stays empty)
        m = rows.shape[0] if t.size == 1 else t.size
        return (
            np.array(np.broadcast_to(t, (m,))),
            np.array(np.broadcast_to(rows, (m, rows.shape[1]))),
        )

    # -- cause-specific and all-cause functions ---------------------------

    def Hf(self, x: npt.ArrayLike, Z: Any, event: Any = None) -> npt.NDArray:
        """Cumulative hazard at times ``x`` and covariates ``Z``: the
        all-cause :math:`\\sum_k H_k(x \\mid Z)`, or with ``event=k``
        cause ``k``'s own."""
        return sum(m.Hf(x, Z) for m in self._of(event))

    def hf(self, x: npt.ArrayLike, Z: Any, event: Any = None) -> npt.NDArray:
        """Hazard rate at times ``x`` and covariates ``Z``: the all-cause
        :math:`\\sum_k h_k(x \\mid Z)`, or with ``event=k`` the
        cause-specific hazard of ``k``."""
        return sum(m.hf(x, Z) for m in self._of(event))

    def sf(self, x: npt.ArrayLike, Z: Any, event: Any = None) -> npt.NDArray:
        """
        Survival at times ``x`` and covariates ``Z``. ``event=None`` (the
        default) is the all-cause survival, the probability of no failure
        of any cause by ``x``: :math:`\\prod_k S_k(x \\mid Z)`.
        ``event=k`` is the *net* survival of cause ``k``, its own model's
        (as if the other causes were removed); it is not one minus the
        cumulative incidence, which is :meth:`cif`.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ParametricCompetingRisksRegression
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.weibull(2.0, 200) * 10 * np.exp(-0.4 * Z[:, 0])
        >>> t_b = rng.exponential(20, 200)
        >>> x = np.minimum(t_a, t_b).round(3)
        >>> e = np.where(t_a < t_b, "a", "b")
        >>> model = ParametricCompetingRisksRegression.fit(x, Z, e)
        >>> model.sf([5, 10], [[0], [1]]).round(4)
        array([0.6439, 0.0749])
        >>> model.sf([5, 10], [[0], [1]], event="a").round(4)
        array([0.8033, 0.1246])
        """
        out: Any = 1.0
        for m in self._of(event):
            out = out * m.sf(x, Z)
        return out

    def ff(self, x: npt.ArrayLike, Z: Any, event: Any = None) -> npt.NDArray:
        """Failure probability at times ``x`` and covariates ``Z``: the
        all-cause one, ``1 - sf`` (the sum of the causes' cumulative
        incidences), or with ``event=k`` the net one of cause ``k`` alone
        (its own model's; for the probability of failing from ``k`` with
        the other causes competing, use :meth:`cif`)."""
        if event is not None:
            self._check_event(event)
            return self.models[event].ff(x, Z)
        return 1.0 - self.sf(x, Z)

    def df(self, x: npt.ArrayLike, Z: Any, event: Any = None) -> npt.NDArray:
        """Density at times ``x`` and covariates ``Z``: of the time to
        the first failure of any cause, ``hf * sf``, or with ``event=k``
        the net density of cause ``k`` alone (its own model's). The
        density of failures *from* a cause with the others competing is
        :meth:`iif`."""
        if event is not None:
            self._check_event(event)
            return self.models[event].df(x, Z)
        return self.hf(x, Z) * self.sf(x, Z)

    def iif(self, x: npt.ArrayLike, Z: Any, event: Any) -> npt.NDArray:
        """The instantaneous incidence (the sub-distribution density) of
        cause ``event`` at times ``x`` and covariates ``Z``:
        :math:`h_k(x \\mid Z)\\,S(x \\mid Z)`, the derivative of
        :meth:`cif`."""
        self._check_event(event)
        return self.models[event].hf(x, Z) * self.sf(x, Z)

    @keeps_query_shape
    def cif(self, x: npt.ArrayLike, Z: Any, event: Any = None) -> npt.NDArray:
        """
        Cumulative incidence at times ``x`` and covariates ``Z``: the
        probability of having failed from cause ``event`` by ``x``, with
        the other causes competing. ``event=None`` gives the all-cause
        incidence :math:`1 - S(x \\mid Z)`, the sum of the causes'.

        It is a smooth function of time defined at every time, past the
        last observation too; ``inf`` gives the eventual probability of
        the cause (:meth:`probability_of_cause`). The integral
        :math:`\\int_0^x h_k S\\,du` is taken over the cause's own
        probability scale,

        .. math::
            \\mathrm{CIF}_k(x \\mid Z) = \\int_0^{F_k(x \\mid Z)}
            \\prod_{j \\neq k} S_j\\left(F_k^{-1}(p \\mid Z) \\mid Z
            \\right) dp,

        by adaptive quadrature to a relative accuracy of about
        :math:`10^{-10}`, every (time, row) pair at once. The integrand
        is bounded by 1 and monotone, so the result is as accurate for a
        cause whose density is infinite at 0 (a Weibull shape below 1) or
        heavy-tailed as for any other.

        Parameters
        ----------
        x : array_like or float
            The times.
        Z : array_like or DataFrame
            The covariates: one row used at every time, one row per time,
            or several rows for one time.
        event : optional
            The cause; ``None`` for all causes combined.

        Returns
        -------
        numpy array or scalar
            The cumulative incidence of each (time, row) pair; ``nan``
            where the time or a covariate is missing.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ParametricCompetingRisksRegression
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.weibull(2.0, 200) * 10 * np.exp(-0.4 * Z[:, 0])
        >>> t_b = rng.exponential(20, 200)
        >>> x = np.minimum(t_a, t_b).round(3)
        >>> e = np.where(t_a < t_b, "a", "b")
        >>> model = ParametricCompetingRisksRegression.fit(x, Z, e)
        >>> model.cif([5, 10, 50], [1], "a").round(4)
        array([0.3739, 0.6916, 0.7578])
        >>> model.cif(10, [[0], [1]], "a").round(4)
        array([0.4274, 0.6916])
        """
        if event is None:
            return np.asarray(self.ff(x, Z))
        self._check_event(event)
        t, rows = self._pairs(x, Z)
        own = self.models[event]
        others = [self.models[k] for k in self.causes if k != event]
        with np.errstate(all="ignore"), ignoring_warnings():
            top = np.asarray(own.ff(t, rows), dtype=float).ravel()
        missing = np.isnan(t) | np.isnan(rows).any(axis=1) | np.isnan(top)
        top = np.clip(np.where(missing, 0.0, top), 0.0, 1.0)
        out = np.where(missing, np.nan, 0.0)
        todo = (top > 0) & ~missing
        if todo.any():
            upper, at = top[todo], rows[todo]

            def integrand(u: float) -> npt.NDArray:
                with np.errstate(all="ignore"), ignoring_warnings():
                    s = np.asarray(own.qf(u * upper, at), dtype=float).ravel()
                    val = np.ones_like(s)
                    far = np.isposinf(s)
                    for m in others:
                        sj = np.asarray(m.sf(s, at), dtype=float).ravel()
                        # (no regression family has a cure fraction:
                        # every cause's survival falls to 0)
                        val = val * np.where(far, 0.0, sj)
                return np.where(np.isfinite(val), val, 1.0)

            integral, _ = quad_vec(
                integrand,
                0.0,
                1.0,
                epsabs=1e-13,
                epsrel=1e-11,
                norm="max",
                limit=2000,
            )
            out[todo] = upper * np.asarray(integral, dtype=float)
        return out

    def probability_of_cause(self, Z: Any, event: Any) -> Any:
        """
        The eventual probability that a unit with covariates ``Z`` fails
        from ``event``, :math:`\\mathrm{CIF}_k(\\infty \\mid Z)`: one
        value per row of ``Z`` (a scalar for one row). Over the causes
        they sum to one.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ParametricCompetingRisksRegression
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.weibull(2.0, 200) * 10 * np.exp(-0.4 * Z[:, 0])
        >>> t_b = rng.exponential(20, 200)
        >>> x = np.minimum(t_a, t_b).round(3)
        >>> e = np.where(t_a < t_b, "a", "b")
        >>> model = ParametricCompetingRisksRegression.fit(x, Z, e)
        >>> model.probability_of_cause([[0], [1]], "a").round(4)
        array([0.6372, 0.7578])
        """
        self._check_event(event)
        return self.cif(np.inf, Z, event)

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """
        Quantile of the time to the first failure of any cause at
        covariates ``Z``: the time by which a fraction ``p`` of such
        units has failed (``p = 0.1`` the B10 life), the inverse of the
        all-cause :meth:`ff`. ``Z`` is paired with ``p`` as with the
        times of the other functions.

        Each quantile is found by root finding on the all-cause
        cumulative hazard, between two brackets from the causes' own
        quantiles: the unit fails no later than its earliest cause's
        ``p`` quantile, and no earlier than the earliest of the causes'
        ``1 - (1 - p)^(1/K)`` quantiles (``K`` causes). A probability
        outside [0, 1] gives ``nan``, with one warning.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ParametricCompetingRisksRegression
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.weibull(2.0, 200) * 10 * np.exp(-0.4 * Z[:, 0])
        >>> t_b = rng.exponential(20, 200)
        >>> x = np.minimum(t_a, t_b).round(3)
        >>> e = np.where(t_a < t_b, "a", "b")
        >>> model = ParametricCompetingRisksRegression.fit(x, Z, e)
        >>> model.qf([0.1, 0.5], [1]).round(4)
        array([1.4205, 4.6339])
        """
        probs, rows = self._pairs(p, Z)
        outside = warn_outside_unit_interval(probs)
        missing = outside | np.isnan(probs) | np.isnan(rows).any(axis=1)
        out = np.full(probs.shape, np.nan)
        models = [self.models[k] for k in self.causes]
        ok = np.flatnonzero(~missing)
        if not ok.size:
            return out
        with np.errstate(all="ignore"), ignoring_warnings():
            share = -np.expm1(np.log1p(-probs[ok]) / len(models))
            hi = np.min(
                [np.ravel(m.qf(probs[ok], rows[ok])) for m in models], axis=0
            )
            lo = np.min(
                [np.ravel(m.qf(share, rows[ok])) for m in models], axis=0
            )
        for i, a, b in zip(ok, lo, hi):
            out[i] = self._quantile(probs[i], rows[i : i + 1], a, b)
        return out

    def _quantile(
        self, p: float, row: npt.NDArray, lo: float, hi: float
    ) -> float:
        """The all-cause ``p`` quantile at one covariate ``row``, between
        the brackets ``lo <= hi``."""
        if p >= 1.0 or not np.isfinite(hi):
            return float(hi)
        if not (np.isfinite(lo) and hi > lo):
            return float(hi)
        target = -np.log1p(-p)

        def resid(t: float) -> float:
            with np.errstate(all="ignore"):
                return float(np.ravel(self.Hf(t, row))[0]) - target

        f_lo, f_hi = resid(lo), resid(hi)
        if not (f_lo < 0.0 < f_hi):
            # (a bracket that is the answer to rounding)
            return float(lo if abs(f_lo) <= abs(f_hi) else hi)
        return float(brentq(resid, lo, hi, xtol=1e-300, rtol=1e-13))

    def mean(self, Z: Any) -> Any:
        """
        Mean time to the first failure of any cause at covariates ``Z``
        (the MTTF of a unit there): one value per row of ``Z``, a scalar
        for one row. It is the area under the all-cause survival, taken
        on either side of the median: ``median + integral of sf above it
        - integral of ff below it``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ParametricCompetingRisksRegression
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.weibull(2.0, 200) * 10 * np.exp(-0.4 * Z[:, 0])
        >>> t_b = rng.exponential(20, 200)
        >>> x = np.minimum(t_a, t_b).round(3)
        >>> e = np.where(t_a < t_b, "a", "b")
        >>> model = ParametricCompetingRisksRegression.fit(x, Z, e)
        >>> model.mean([[0], [1]]).round(4)
        array([7.4784, 5.0858])
        """
        rows = self._rows(Z)
        lower = min(
            float(self.models[k].distribution.support[0]) for k in self.causes
        )
        out = np.full(rows.shape[0], np.nan)
        for i in range(rows.shape[0]):
            row = rows[i : i + 1]
            if np.isnan(row).any():
                continue
            with np.errstate(all="ignore"), ignoring_warnings():
                median = float(np.ravel(self.qf(0.5, row))[0])
                above = quad(
                    lambda t: float(np.ravel(self.sf(t, row))[0]),
                    median,
                    np.inf,
                    epsabs=0.0,
                    epsrel=1e-10,
                    limit=500,
                )[0]
                below = quad(
                    lambda t: float(np.ravel(self.ff(t, row))[0]),
                    lower,
                    median,
                    epsabs=0.0,
                    epsrel=1e-10,
                    limit=500,
                )[0]
            out[i] = median + above - below
        single = np.ndim(Z) <= 1 and rows.shape[0] == 1
        return out[0] if single else out

    def random(
        self, size: int, Z: Any, random_state: Any = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Draw ``size`` units at each covariate row of ``Z``: a latent time
        is drawn from each cause's model at the row, and the unit fails
        at the earliest, from that cause (the latent failure time
        picture, with independent causes).

        Parameters
        ----------
        size : int
            The number of units drawn at each row.
        Z : array_like or DataFrame
            The covariate row(s) to draw at.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the draw. ``None`` (the default) draws from
            numpy's global generator, so ``np.random.seed`` reproduces
            it; an int or a ``Generator`` gives a stream of its own.

        Returns
        -------
        draws : numpy structured array
            Fields ``x`` (the time) and ``e`` (the cause), the ``size``
            draws of each row, row by row.
        Z : numpy array
            The covariate row of each draw.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ParametricCompetingRisksRegression
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.weibull(2.0, 200) * 10 * np.exp(-0.4 * Z[:, 0])
        >>> t_b = rng.exponential(20, 200)
        >>> x = np.minimum(t_a, t_b).round(3)
        >>> e = np.where(t_a < t_b, "a", "b")
        >>> model = ParametricCompetingRisksRegression.fit(x, Z, e)
        >>> draws, rows = model.random(3, [[0], [1]], random_state=1)
        >>> draws["x"].round(3)
        array([ 4.359,  2.32 , 11.926,  1.448,  5.984,  6.281])
        >>> list(draws["e"])
        ['b', 'a', 'b', 'a', 'b', 'a']
        >>> rows.ravel()
        array([0., 0., 0., 1., 1., 1.])
        """
        rng = as_generator(random_state)
        rows = self._rows(Z)
        latent, drawn_at = [], rows
        for k in self.causes:
            times, drawn_at = self.models[k].random(
                size, rows, random_state=rng
            )
            latent.append(np.ravel(times))
        stacked = np.column_stack(latent)
        first = np.argmin(stacked, axis=1)
        out = np.empty(stacked.shape[0], dtype=[("x", float), ("e", object)])
        out["x"] = stacked[np.arange(stacked.shape[0]), first]
        out["e"] = np.array([self.causes[i] for i in first], dtype=object)
        return out, np.asarray(drawn_at, dtype=float)

    # -- parameters and their uncertainty ---------------------------------

    @property
    def params(self) -> npt.NDArray:
        """Every cause's fitted parameters, cause by cause in the order
        of ``causes``, named by ``parameter_names``."""
        return np.concatenate(
            [
                np.ravel(np.asarray(self.models[k].params, dtype=float))
                for k in self.causes
            ]
        )

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``, entry by entry: ``"<cause>: <name>"``,
        the cause and its model's own parameter name (a baseline
        parameter, or a covariate's coefficient).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ParametricCompetingRisksRegression
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.weibull(2.0, 200) * 10 * np.exp(-0.4 * Z[:, 0])
        >>> t_b = rng.exponential(20, 200)
        >>> x = np.minimum(t_a, t_b).round(3)
        >>> e = np.where(t_a < t_b, "a", "b")
        >>> model = ParametricCompetingRisksRegression.fit(x, Z, e)
        >>> model.parameter_names
        ['a: alpha', 'a: beta', 'a: coef_0', 'b: alpha', 'b: beta', 'b: coef_0']
        """  # noqa: E501
        return [
            f"{k}: {name}"
            for k in self.causes
            for name in self.models[k].parameter_names
        ]

    @property
    def covariance_names(self) -> list[str]:
        """The names of the rows and columns of :meth:`covariance`:
        ``"<cause>: <name>"`` over each cause's own
        ``covariance_names``."""
        return [
            f"{k}: {name}"
            for k in self.causes
            for name in self.models[k].covariance_names
        ]

    def _estimates(self) -> npt.NDArray:
        """The fitted values in the order of ``covariance_names``."""
        return np.concatenate(
            [
                np.ravel(np.asarray(self.models[k]._estimates(), float))
                for k in self.causes
            ]
        )

    def covariance(self) -> npt.NDArray:
        """
        The covariance of the parameter estimates, named by
        ``covariance_names``: block-diagonal, each block one cause's own
        ``models[cause].covariance()``. The blocks between causes are
        zero, exactly: the likelihood factorises over the causes, so
        their estimates are asymptotically independent.
        """
        return block_diag(
            *[
                np.atleast_2d(np.asarray(self.models[k].covariance(), float))
                for k in self.causes
            ]
        )

    def standard_errors(self) -> npt.NDArray:
        """The standard errors of the estimates, the square roots of the
        diagonal of :meth:`covariance`."""
        return standard_errors_of(self.covariance())

    def summary(self, alpha_ci: float = 0.05) -> pd.DataFrame:
        """
        The parameter table of every cause, stacked: each cause's own
        regression ``summary`` (its baseline parameters and its
        coefficients, with standard errors, a two-sided ``1 - alpha_ci``
        Wald interval, ``z`` and ``p``; ``exp(coef)`` where the
        coefficient is a log ratio), under the cause as the first level
        of the index.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ParametricCompetingRisksRegression
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.weibull(2.0, 200) * 10 * np.exp(-0.4 * Z[:, 0])
        >>> t_b = rng.exponential(20, 200)
        >>> x = np.minimum(t_a, t_b).round(3)
        >>> e = np.where(t_a < t_b, "a", "b")
        >>> model = ParametricCompetingRisksRegression.fit(x, Z, e)
        >>> model.summary()[["coef", "se(coef)"]].round(3)
                                     coef  se(coef)
        cause part         name
        a     baseline     alpha   11.107     0.781
                           beta     1.903     0.117
              coefficients coef_0   0.933     0.178
        b     baseline     alpha   18.672     3.196
                           beta     1.145     0.127
              coefficients coef_0   0.040     0.266
        """
        import pandas as pd

        check_alpha_ci(alpha_ci)
        return pd.concat(
            {k: self.models[k].summary(alpha_ci) for k in self.causes},
            names=["cause"],
        )

    def with_params(
        self, params: npt.ArrayLike
    ) -> ParametricCompetingRisksRegression:
        """
        The same model at other parameter values: ``params`` in the
        order of ``covariance_names`` (every cause's, cause by cause),
        each cause rebuilt by its own model's ``with_params``. The data,
        covariance and likelihood are not carried over: the result
        predicts, and has no uncertainty.
        """
        values = np.ravel(np.asarray(params, dtype=float))
        sizes = [len(self.models[k].covariance_names) for k in self.causes]
        if values.size != sum(sizes):
            raise ValueError(
                "params must have {} values, one for each of {}; got "
                "{}.".format(sum(sizes), self.covariance_names, values.size)
            )
        out = copy.copy(self)
        out.models = {}
        start = 0
        for k, size in zip(self.causes, sizes):
            out.models[k] = self.models[k].with_params(
                values[start : start + size]
            )
            start += size
        return out

    # -- confidence bounds --------------------------------------------------

    def cb(
        self,
        x: npt.ArrayLike,
        Z: Any,
        event: Any = None,
        on: str = "cif",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        r"""
        Pointwise confidence bounds at times ``x`` and covariates ``Z``
        on the cumulative incidence of cause ``event`` (``on="cif"``, the
        default), or on the survival, failure probability or cumulative
        hazard (``on="sf"``, ``"ff"``, ``"Hf"``) of all causes
        (``event=None``) or of one cause's net ones.

        The Wald bounds carry the uncertainty of every cause's parameters
        (:meth:`covariance`) to the function by the delta method, on a
        scale that keeps them in range:

        - the all-cause ``sf``, ``ff`` and ``Hf`` share one bound, on the
          log of the all-cause cumulative hazard. The causes' estimates
          are independent, so :math:`\mathrm{Var}\,H = \sum_k
          \mathrm{Var}\,H_k`, each from its cause's gradient;
        - the cumulative incidence on the logit scale. It depends on
          every cause's parameters through an integral, so its gradient
          is taken by central differences on ``covariance_names`` (each
          step :math:`10^{-5}` of the parameter's value or standard
          error, whichever is larger);
        - one cause's net ``sf``, ``ff`` and ``Hf`` (``event=k``) are
          functions of that cause's parameters alone: the bound is the
          cause's own ``models[k].cb``, which also offers
          ``method="lr"`` and ``"bootstrap"``.

        Parameters
        ----------
        x : array_like
            The times.
        Z : array_like or DataFrame
            The covariates, paired with the times as for :meth:`cif`.
        event : optional
            The cause: required for ``on="cif"``; for the others ``None``
            is all causes.
        on : {"cif", "sf", "ff", "Hf"}, optional
            The function bounded. Default ``"cif"``.
        alpha_ci : float, optional
            One minus the confidence level. Default 0.05.
        bound : {"two-sided", "upper", "lower"}, optional
            Default ``"two-sided"``.
        method : {"wald", "lr", "bootstrap"}, optional
            ``"wald"`` (the default) for every function; ``"lr"`` and
            ``"bootstrap"`` for one cause's net functions only (the
            cumulative incidence and the all-cause functions depend on
            every cause's fit at once, and refuse them, saying so).
        n_boot, random_state
            For ``method="bootstrap"``, as the cause's model's ``cb``.

        Returns
        -------
        numpy array
            The bounds, in the shape of the (time, row) pairs: with a
            last ``[lower, upper]`` axis for a two-sided bound.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ParametricCompetingRisksRegression
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.weibull(2.0, 200) * 10 * np.exp(-0.4 * Z[:, 0])
        >>> t_b = rng.exponential(20, 200)
        >>> x = np.minimum(t_a, t_b).round(3)
        >>> e = np.where(t_a < t_b, "a", "b")
        >>> model = ParametricCompetingRisksRegression.fit(x, Z, e)
        >>> model.cb([5, 10], [1], "a").round(4)
        array([[0.3062, 0.4471],
               [0.6065, 0.7654]])
        >>> model.cb(10, [1], on="sf", bound="lower").round(4)
        np.float64(0.0475)
        """
        check_alpha_ci(alpha_ci)
        check_option("on", on, _CB_ON)
        check_option("bound", bound, BOUNDS)
        method = str(method).lower()
        check_option("method", method, _CB_METHODS)
        if event is not None:
            self._check_event(event)
        if on != "cif" and event is not None:
            # A net function: the cause's own model and its own bounds.
            try:
                return np.asarray(
                    self.models[event].cb(
                        x,
                        Z,
                        on=on,
                        alpha_ci=alpha_ci,
                        bound=bound,
                        method=method,
                        n_boot=n_boot,
                        random_state=random_state,
                    ),
                    dtype=float,
                )
            except ValueError as err:
                raise ValueError(f"cause {event!r}: {err}") from err
        if on == "cif":
            validate_cif_event(event)
        if method != "wald":
            raise ValueError(
                f"method={method!r} is available for one cause's net sf, "
                "ff and Hf (cb(x, Z, event=k, on='sf', method=...)), from "
                "that cause's model; the cumulative incidence and the "
                "all-cause functions depend on every cause's parameters "
                "at once and have Wald bounds only: use method='wald'."
            )
        flat, restore = flatten_query(x)
        t, rows = self._pairs(flat, Z)
        with np.errstate(all="ignore"):
            if on == "cif":
                out = self._cb_cif(t, rows, event, alpha_ci, bound)
            else:
                out = self._cb_all_cause(t, rows, on, alpha_ci, bound)
        return restore(out)

    def _cb_all_cause(
        self,
        t: npt.NDArray,
        rows: npt.NDArray,
        on: str,
        alpha_ci: float,
        bound: str,
    ) -> npt.NDArray:
        """The Wald bound on the all-cause ``sf``, ``ff`` or ``Hf``: one
        bound on ``log H``, ``Var H`` the sum of the causes'."""
        H = np.asarray(self.Hf(t, rows), dtype=float)
        var = np.zeros(t.shape)
        for k in self.causes:
            model = self.models[k]
            cov = np.atleast_2d(np.asarray(model.covariance(), dtype=float))

            def H_of(values: npt.NDArray, model: Any = model) -> npt.NDArray:
                with ignoring_warnings():
                    rebuilt = model.with_params(values)
                    return np.asarray(rebuilt.Hf(t, rows), dtype=float)

            grad = _central_differences(
                H_of, np.ravel(model._estimates()), cov, t.size
            )
            var = var + np.einsum("jt,jk,kt->t", grad, cov, grad)
        ok = np.isfinite(H) & (H > 0)
        sd = np.where(ok, _checked_sd(var, t, "Hf", ok), 0.0) / np.where(
            ok, H, 1.0
        )
        alpha, signs = bound_signs(alpha_ci, bound)
        if on == "sf":
            # sf falls as H rises: its lower bound is at H's upper one.
            signs = -signs
        H_b = H[:, None] * np.exp(
            signs[None, :] * ndtri(1 - alpha) * sd[:, None]
        )
        if on == "Hf":
            out = H_b
        elif on == "sf":
            out = np.exp(-H_b)
        else:
            out = -np.expm1(-H_b)
        return out if bound == "two-sided" else out[:, 0]

    def _cb_cif(
        self,
        t: npt.NDArray,
        rows: npt.NDArray,
        event: Any,
        alpha_ci: float,
        bound: str,
    ) -> npt.NDArray:
        """The Wald bound on cause ``event``'s cumulative incidence, on
        the logit scale, from its numerical gradient."""
        F = np.asarray(self.cif(t, rows, event), dtype=float)
        cov = self.covariance()

        def cif_of(values: npt.NDArray) -> npt.NDArray:
            with ignoring_warnings():
                rebuilt = self.with_params(values)
                return np.asarray(rebuilt.cif(t, rows, event), dtype=float)

        grad = _central_differences(cif_of, self._estimates(), cov, t.size)
        var = np.einsum("jt,jk,kt->t", grad, cov, grad)
        ok = (F > 0) & (F < 1) & np.isfinite(F)
        sd = np.where(ok, _checked_sd(var, t, "cif", ok), 0.0)
        sd_logit = sd / np.where(ok, F * (1 - F), 1.0)
        alpha, signs = bound_signs(alpha_ci, bound)
        centre = np.where(ok, F, 0.5)
        logit_F = np.log(centre) - np.log1p(-centre)
        moved = logit_F[:, None] + signs[None, :] * ndtri(1 - alpha) * (
            sd_logit[:, None]
        )
        out = np.where(ok[:, None], 1.0 / (1.0 + np.exp(-moved)), F[:, None])
        return out if bound == "two-sided" else out[:, 0]

    # -- goodness of fit (the likelihood factorises over causes) ----------

    @property
    def maximum(self) -> str:
        """What the causes' fits reached, one of ``MAXIMUM_STATES``
        (``surpyval.utils.no_maximum``): the worst of the causes' own
        ``maximum``, since the likelihood is maximised a cause at a
        time."""
        return combined_maximum(
            getattr(self.models[k], "maximum", "unknown") for k in self.causes
        )

    def neg_ll(self) -> float:
        """The negative log-likelihood of the whole model: the sum of the
        causes' fits'."""
        return float(sum(self.models[k].neg_ll() for k in self.causes))

    @property
    def log_likelihood(self) -> float:
        """The maximised log-likelihood of the whole model,
        ``-neg_ll()``."""
        return -self.neg_ll()

    def _ic_terms(self) -> tuple[int, float]:
        # The parameters estimated over every cause, and the failures of
        # every cause.
        k_total, n_total = 0, 0.0
        for k in self.causes:
            k_cause, n_cause = _ic_terms(self.models[k])
            k_total += k_cause
            n_total += n_cause
        return k_total, n_total

    def aic(self) -> float:
        """Akaike information criterion of the whole model, ``2 K + 2
        neg_ll`` with ``K`` the parameters estimated over every cause."""
        k_total, _ = self._ic_terms()
        return float(2 * k_total + 2 * self.neg_ll())

    def bic(self) -> float:
        """Bayesian information criterion of the whole model, ``2 neg_ll
        + K ln(n)``: ``K`` the parameters estimated over every cause and
        ``n`` the observed failures of any cause (weighted by their
        counts), as for ``ParametricCompetingRisks``. It is not the sum
        of the causes' own BICs."""
        k_total, n_total = self._ic_terms()
        return float(2 * self.neg_ll() + k_total * np.log(n_total))

    def aic_c(self) -> float:
        """The small-sample corrected AIC of the whole model, with the
        ``K`` and ``n`` of :meth:`bic`."""
        k_total, n_total = self._ic_terms()
        return corrected_aic(self.aic(), k_total, n_total)

    # -- construction -----------------------------------------------------

    @classmethod
    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit(
        cls,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        e: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        tl: npt.ArrayLike | float | None = None,
        model: Any = WeibullPH,
        init: Any = None,
    ) -> ParametricCompetingRisksRegression:
        """
        Fit a parametric regression model to each cause's cause-specific
        hazard.

        Parameters
        ----------
        x : array_like
            Failure or censoring times.
        Z : array_like or DataFrame
            Covariate matrix, one row per observation. Rows with a
            missing (``NaN``) or infinite covariate are dropped, with a
            warning.
        e : array_like
            The cause of each failure: any hashable labels, kept in
            sorted order in ``causes``; ``None`` (or ``NaN``) for a
            right-censored observation.
        c : array_like, optional
            Censoring flags: 0 a failure, 1 right-censored. Derived from
            ``e`` if not given (a missing cause is censored). Left and
            interval censoring are refused: with failures found at
            inspections and no covariates, use
            ``ParametricCompetingRisks``.
        n : array_like, optional
            Counts of each row. Default 1.
        tl : array_like or float, optional
            Entry times (delayed entry, left truncation): a unit is
            observed only from ``tl``, having not failed before. The
            same entry time applies to every cause's fit, which is
            exact, as each cause's likelihood term is conditional on
            surviving that cause to ``tl``.
        model : regression fitter or dict, optional
            The parametric regression fitted to every cause (default
            ``WeibullPH``), or a ``{cause: fitter}`` dict to choose per
            cause: any parametric regression fitter, such as
            ``ExponentialPH``, ``WeibullAFT``, ``LogNormalAFT`` or
            ``LogLogisticPO``.
        init : array_like or dict, optional
            Starting values for the causes' fits, in the order of the
            fitter's parameters: one set used for every cause, or a
            ``{cause: init}`` dict.

        Returns
        -------
        ParametricCompetingRisksRegression
            The fitted model: ``causes``, and each cause's regression
            model in ``models[cause]``.

        Notes
        -----
        Each cause is fitted by its fitter's own ``fit`` with the
        failures of the other causes right-censored. That is the
        maximum-likelihood fit of the whole model, since the likelihood
        factorises over the causes; it does not assume that the causes
        are independent.

        Examples
        --------
        Two causes: the covariate shortens the time to cause ``a`` and
        leaves cause ``b`` alone.

        >>> import numpy as np
        >>> from surpyval import ParametricCompetingRisksRegression
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.weibull(2.0, 200) * 10 * np.exp(-0.4 * Z[:, 0])
        >>> t_b = rng.exponential(20, 200)
        >>> x = np.minimum(t_a, t_b).round(3)
        >>> e = np.where(t_a < t_b, "a", "b")
        >>> model = ParametricCompetingRisksRegression.fit(x, Z, e)
        >>> model.causes
        ['a', 'b']
        >>> model.models["a"].params.round(3)
        array([11.107,  1.903,  0.933])
        >>> model.cif(10, [[0], [1]], "a").round(4)
        array([0.4274, 0.6916])

        A different model for each cause:

        >>> from surpyval import ExponentialPH, WeibullAFT
        >>> mixed = ParametricCompetingRisksRegression.fit(
        ...     x, Z, e, model={"a": WeibullAFT, "b": ExponentialPH}
        ... )
        >>> mixed.cif(10, [[0], [1]], "a").round(4)
        array([0.4212, 0.6892])
        """
        x, Z, e, c, n, _, tl = _validated(
            x, Z, e, c, n, None, _entry_or_none(tl)
        )
        causes = ordered_labels(e)
        if not causes:
            raise ValueError("No observed events: every row is censored.")
        fitters = _cause_fitters(model, causes)
        inits = _cause_inits(init, causes)
        models = {}
        found: list = []
        for k in causes:
            # Cause k is observed where it occurred; a failure from any
            # other cause, like a censoring, is right-censored for it.
            c_k = np.where(label_mask(e, k) & (c == 0), 0, 1)
            extra = {} if tl is None else {"tl": tl}
            with collect_aliased() as aliased:
                models[k] = fitters[k].fit(
                    x, Z, c=c_k, n=n, init=inits[k], **extra
                )
            found += aliased
        warn_collected(found, "in the fit of each cause")
        out = cls()
        out.causes = causes
        out.models = models
        return out

    @classmethod
    @column_arguments("x", "c", "n")
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        e_col: str,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        tl_col: str | None = None,
        formula: str | None = None,
        model: Any = WeibullPH,
        init: Any = None,
    ) -> ParametricCompetingRisksRegression:
        """
        Fit the model from the columns of a :class:`pandas.DataFrame`.

        Parameters
        ----------
        df : pandas.DataFrame
            The data.
        x_col : str
            Column of failure or censoring times.
        e_col : str
            Column of the causes; a blank (``None`` / ``NaN``) cell for a
            censored row.
        Z_cols : str or list of str, optional
            The covariate column(s). Mutually exclusive with ``formula``.
        c_col, n_col, tl_col : str, optional
            Columns of the censoring flags, counts and entry times, as
            ``c``, ``n`` and ``tl`` of :meth:`fit`.
        formula : str, optional
            A ``formulaic`` formula of the covariates, e.g. ``"age +
            C(site)"``, as the single-event regressions take it.
            Mutually exclusive with ``Z_cols``.
        model, init
            As for :meth:`fit`.

        Returns
        -------
        ParametricCompetingRisksRegression
            The fitted model. Its functions take a DataFrame of the raw
            covariate columns (the ``formula`` is applied to it) or a
            covariate array in the fitted column order.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from surpyval import ParametricCompetingRisksRegression
        >>> rng = np.random.default_rng(1)
        >>> g = rng.choice(["a", "b", "c"], 300)
        >>> scale = 10 * np.exp(-np.select([g == "b", g == "c"], [0.5, -0.3]))
        >>> t_1 = rng.weibull(1.5, 300) * scale
        >>> t_2 = rng.exponential(20, 300)
        >>> df = pd.DataFrame({
        ...     "time": np.minimum(t_1, t_2).round(3),
        ...     "cause": np.where(t_1 < t_2, "wear", "shock"),
        ...     "g": g,
        ... })
        >>> model = ParametricCompetingRisksRegression.fit_from_df(
        ...     df, "time", "cause", formula="g"
        ... )
        >>> model.feature_names
        ['g[T.b]', 'g[T.c]']
        >>> new = pd.DataFrame({"g": ["a", "b", "c"]})
        >>> model.cif(np.full(3, 5.0), new, "wear").round(4)
        array([0.2844, 0.5163, 0.1903])
        """
        Z, feature_names, model_spec = design_matrix_from_df(
            df, Z_cols, formula
        )
        mask = finite_covariate_mask(Z)
        Z = Z[mask]
        sub = df.loc[mask]
        x = sub[x_col].values
        # A censored row's cause is ``None``; accept a blank/NaN cell.
        e = sub[e_col].to_numpy(dtype=object).copy()
        e[[is_missing_event(v) for v in e]] = None
        c = sub[c_col].values if c_col is not None else None
        n = sub[n_col].values if n_col is not None else None
        tl = sub[tl_col].values if tl_col is not None else None
        with covariate_columns(feature_names, Z, model_spec):
            out = cls.fit(x, Z, e, c=c, n=n, tl=tl, model=model, init=init)
        out.feature_names = feature_names
        out.formula = formula
        out._model_spec = model_spec
        return out
