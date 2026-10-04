from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from surpyval.recurrent import diagnostics
from surpyval.recurrent.inference import LikelihoodInferenceMixin
from surpyval.recurrent.serialisation import intensity_dist_by_name
from surpyval.recurrent.simulation import RecurrenceSimulationMixin
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils.linalg import delta_method_se, log_transformed_cb
from surpyval.utils.no_maximum import maximum_entry, restored_maximum
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import (
    BOUNDS,
    alpha_ci_error,
    check_option,
    option_error,
)

# The bound on 1 / f that a bound on f gives.
_OPPOSITE = {"two-sided": "two-sided", "lower": "upper", "upper": "lower"}

# How the model was obtained, as the repr reports it.
_FITTED_BY = {
    "MLE": "MLE",
    "MSE": "MSE (least squares on the MCF)",
    "from_params": "given parameters (not fitted)",
}


class ParametricRecurrenceModel(
    SerialisableMixin, RecurrenceSimulationMixin, LikelihoodInferenceMixin
):
    """
    A class for holding the parameters, data, and useful methods for a
    fitted parametric recurrence model. This is the result of the ``fit`` calls
    from the counting distributions.

    When fitted by maximum likelihood the model also carries the likelihood-
    inference behaviour (``log_likelihood``, ``aic``, ``bic``,
    ``standard_errors``) from :class:`LikelihoodInferenceMixin`. Models built
    by ``from_params`` or fitted by ``how="MSE"`` carry no likelihood, so those
    methods raise.

    Example
    -------

    >>> from surpyval import Exponential
    >>> from surpyval.recurrent import HPP
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> x = Exponential.random(10, 1e-3).cumsum()
    >>> model = HPP.fit(x)
    """

    # Populated by the fitters; declared for the type checker.
    dist: Any
    params: "np.ndarray"
    bounds: tuple
    support: tuple
    name: str
    data: Any
    mcf_hat: "np.ndarray"
    how: str
    res: Any
    #: What a maximum-likelihood fit reached, one of ``MAXIMUM_STATES``
    #: (``surpyval.utils.no_maximum``), as its warnings say; ``"not
    #: applicable"`` for a least-squares fit or a model built from its
    #: parameters, ``"unknown"`` for one restored from a dict saved
    #: without it.
    maximum: str = "not applicable"

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted recurrence model to a plain, JSON-serialisable
        dict.

        The intensity model is stateless, so only its name and the fitted
        ``params`` are stored; the reloaded model reproduces ``cif``/``iif``/
        ``mcf``/``inv_cif`` exactly. Likelihood-inference state (the data and
        the ``neg_ll`` closure) is not stored, so a reloaded model behaves like
        a ``from_params`` one for confidence bounds and diagnostics.

        See Also
        --------
        from_dict, to_json, from_json
        """
        return stamp_schema(
            {
                "model": "ParametricRecurrenceModel",
                "dist": self.dist.name,
                "params": np.asarray(self.params, dtype=float).tolist(),
                "how": getattr(self, "how", "from_params"),
                **maximum_entry(self.maximum),
            }
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> "ParametricRecurrenceModel":
        """
        Rebuild a recurrence model from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict, "ParametricRecurrenceModel", "a recurrence model"
        )
        out = cls()
        out.dist = intensity_dist_by_name(model_dict["dist"])
        out.params = np.array(model_dict["params"], dtype=float)
        out.how = model_dict.get("how", "from_params")
        out.maximum = restored_maximum(model_dict)
        return out

    def _parameter_names(self) -> list:
        return list(self.dist.parameter_names)

    def _parameter_bounds(self) -> list:
        return list(self.dist.bounds)

    def __repr__(self) -> str:
        param_string = "\n".join(
            [
                "{:>10}".format(name) + ": " + str(p)
                for p, name in zip(self.params, self.dist.parameter_names)
            ]
        )
        return (
            "Parametric Recurrence SurPyval Model"
            + "\n=================================="
            + f"\nProcess             : {self.dist.name}"
            + "\nFitted by           : "
            + _FITTED_BY.get(getattr(self, "how", "MLE"), "MLE")
            + "\nParameters          :\n"
            + param_string
        )

    # Simulation uses the shared conditional inverse-CIF sampler and the
    # CoxLewis post-processing from RecurrenceSimulationMixin; this model is
    # unconditional, so it needs no extra cif args (_cif_args defaults to ()).

    @keeps_query_shape
    def cif(self, x: ArrayLike) -> np.ndarray:
        """
        Compute the cumulative incidence function (CIF) based on the fitted
        model. No need to pass parameters as it uses the parameters of the
        fitted model.

        Parameters
        ----------

        x: array_like
            Values at which to compute the CIF.

        Returns
        -------

        array_like
            Computed cumulative intensity function values.
        """
        x = np.array(x)
        return self.dist.cif(x, *self.params)

    # Narrows the mixin mcf (no simulation arguments): the fitted
    # model evaluates its own CIF directly.
    def mcf(self, x: ArrayLike) -> np.ndarray:
        """
        The mean cumulative function (MCF). For these counting processes the
        MCF equals the cumulative intensity, so this is a closed-form alias for
        :meth:`cif` (overriding the simulation-based estimate in the mixin).

        Parameters
        ----------

        x: array_like
            Values at which to compute the MCF.

        Returns
        -------

        array_like
            The MCF evaluated at ``x``.
        """
        return self.cif(x)

    @keeps_query_shape
    def iif(self, x: ArrayLike) -> np.ndarray:
        """
        Compute the intensity function based on the fitted model. No need to
        pass parameters as it uses the parameters of the fitted model.

        Parameters
        ----------

        x: array_like
            Values at which to compute the intensity.

        Returns
        -------

        array_like
            Computed instantaneous intensity functions values.
        """
        x = np.array(x)
        return self.dist.iif(x, *self.params)

    def inv_cif(self, x: ArrayLike) -> np.ndarray:
        """
        The inverse of the cumulative intensity function: the time by which
        ``x`` events are expected.

        Parameters
        ----------

        x: array_like
            Expected numbers of events.

        Returns
        -------

        array_like
            The times at which the cumulative intensity reaches ``x``.
        """
        x = np.array(x)
        if hasattr(self.dist, "inv_cif"):
            return self.dist.inv_cif(x, *self.params)
        else:
            raise ValueError(
                "Inverse cif undefined for {}".format(self.dist.name)
            )

    def residuals(self, kind: str = "cumulative_hazard") -> np.ndarray:
        """
        Residual diagnostics for the fitted model, from the time-rescaling
        theorem.

        Parameters
        ----------

        kind: {'cumulative_hazard', 'pit', 'martingale'}, optional
            ``'cumulative_hazard'`` returns the rescaled interarrival times
            ``cif(t_k) - cif(t_{k-1})`` of every observed event (pooled
            across items); see below for how far they are iid Exp(1).
            ``'pit'`` applies the probability integral transform
            ``1 - exp(-e)`` to those residuals (U(0, 1) under the same
            conditions).
            ``'martingale'`` returns one residual per (sorted-unique) item:
            its observed event count minus the count the model expects over
            its observation window; positive values mean the item saw more
            events than predicted.

            Only complete gaps (event to event) are returned. When an
            item's observation ends at a window close rather than at an
            event, its final gap is censored and left out, and that
            selection makes the returned residuals smaller than Exp(1) on
            average -- noticeably so with few events per item (a mean
            near 0.66 with about three events per item). So they are
            exactly iid Exp(1) only for failure-truncated items; otherwise
            read a Q-Q plot against Exp(1) with this downward bias in
            mind, or use ``cramer_von_mises``, which conditions on each
            item's window correctly.

        Returns
        -------

        numpy array
            The residuals.
        """
        self._check_has_data("residuals")
        if kind in ("cumulative_hazard", "pit"):
            e = diagnostics.cumulative_hazard_residuals(self.data, self.cif)
            if kind == "pit":
                return 1.0 - np.exp(-e)
            return e
        elif kind == "martingale":
            return diagnostics.martingale_residuals(self.data, self.cif)
        raise option_error(
            "kind", kind, ("cumulative_hazard", "pit", "martingale")
        )

    def trend_test(
        self,
        test: str = "laplace",
        alternative: str = "two-sided",
        *,
        alpha_ci: float = 0.05,
    ) -> Any:
        """
        Run a trend test on the data this model was fitted to.

        This is a convenience wrapper around the standalone tests in
        ``surpyval.recurrent.tests`` -- the null hypothesis is that the
        events follow a *homogeneous* Poisson process (no trend), so it
        checks whether the data warranted a time-varying intensity at all.
        The model's parameters play no part in the statistic.
        Each item is tested on its own observation window, from its entry
        (``tl``; 0 without one) to its close, so data with delayed entry is
        tested as it was fitted.

        Parameters
        ----------

        test: {'laplace', 'mil_hdbk_189c'}, optional
            The trend test to run. Default is 'laplace'.
        alternative: {'two-sided', 'increasing', 'decreasing'}, optional
            The alternative hypothesis. Default is 'two-sided'.
        alpha_ci: float, optional
            The significance level at which the result's ``trend`` is
            judged (default 0.05, keyword only): a trend is named only when
            ``p_value < alpha_ci``.

        Returns
        -------

        TrendTestResult
            The test result, carrying the statistic, p-value, the
            direction of the statistic and the trend concluded at
            ``alpha_ci``.
        """
        self._check_has_data("trend_test")
        return diagnostics.trend_test(
            self.data, test=test, alternative=alternative, alpha_ci=alpha_ci
        )

    def cramer_von_mises(
        self, n_boot: int = 200, random_state: "int | None" = None
    ) -> Any:
        """
        Cramer-von Mises goodness-of-fit test of the fitted intensity.

        Conditional on the number of events an item shows in its
        observation window, the transformed times ``[cif(t) - cif(entry)] /
        [cif(close) - cif(entry)]`` are iid U(0, 1) when the fitted
        intensity is the true one; the Cramer-von Mises statistic measures
        their departure from uniformity (for the power-law process this is
        the construction behind Crow's goodness-of-fit test). Because the
        parameters were estimated from the same data, the p-value is
        computed by a parametric bootstrap: data is simulated from the
        fitted model over the same observation windows, refitted, and the
        statistic recomputed. Failure-truncated items (no explicit window
        close) are approximated in the bootstrap by a window fixed at their
        last observed event.

        Parameters
        ----------

        n_boot: int, optional
            Number of bootstrap replicates for the p-value. Default is 200.
        random_state: int or numpy.random.Generator, optional
            Seed for a reproducible p-value.

        Returns
        -------

        GoodnessOfFitResult
            The observed statistic and its bootstrap p-value.
        """
        # Data first: a restored or from_params model has neither data nor
        # likelihood, and the missing data is the more useful message; an
        # MSE fit has data but no likelihood to refit by.
        self._check_has_data("cramer_von_mises")
        self._check_fitted()
        return diagnostics.cramer_von_mises(
            self, n_boot=n_boot, random_state=random_state
        )

    @keeps_query_shape
    def cif_cb(
        self,
        x: ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds on the fitted CIF at ``x``, from the delta method.

        The variance of the fitted CIF is propagated from the parameter
        covariance (the inverse observed information) through the CIF's
        gradient, and the bounds are computed on the log scale -- the same
        construction as the default (``bound_type="exp"``) bounds on the
        nonparametric MCF -- so they cannot go negative.

        Parameters
        ----------

        x: array_like
            Values at which to compute the confidence bounds.
        alpha_ci: float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound: {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as an ``(len(x), 2)`` array with
            columns ``[lower, upper]``; one-sided bounds have the shape of
            ``x``.

        Returns
        -------

        numpy array
            The confidence bounds on the CIF.
        """
        self._check_fitted()
        x = np.atleast_1d(np.asarray(x, dtype=float))
        se = delta_method_se(
            lambda params: self.dist.cif(x, *params),
            self._mle,
            self.covariance(),
        )
        return log_transformed_cb(self.cif(x), se, alpha_ci, bound)

    @keeps_query_shape
    def mtbf(self, x: ArrayLike) -> np.ndarray:
        """
        The instantaneous mean time between failures at ``x``,
        ``1 / iif(x)``: the MTBF the process would show from ``x`` on if
        it stopped changing there. At the end of a reliability growth test
        this is the *demonstrated* MTBF. With several items it is the MTBF
        of one item.

        Parameters
        ----------

        x: array_like
            Values at which to compute the MTBF.

        Returns
        -------

        array_like
            The instantaneous MTBF.

        Examples
        --------

        >>> from surpyval.recurrent import CrowAMSAA
        >>> x = [40, 110, 210, 340, 500, 690, 920, 1180, 1480, 1800, 2000]
        >>> c = [0] * 10 + [1]
        >>> model = CrowAMSAA.fit(x, c=c)
        >>> round(float(model.mtbf(2000.0)), 1)
        300.0
        """
        x = np.array(x)
        with np.errstate(divide="ignore"):
            return 1.0 / self.dist.iif(x, *self.params)

    @keeps_query_shape
    def iif_cb(
        self,
        x: ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> np.ndarray:
        """
        Confidence bounds on the fitted intensity (``iif``) at ``x``.

        Parameters
        ----------

        x: array_like
            Values at which to compute the confidence bounds.
        alpha_ci: float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound: {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as an ``(len(x), 2)`` array with
            columns ``[lower, upper]``; one-sided bounds have the shape of
            ``x``.
        method: {'wald', 'crow'}, optional
            ``"wald"`` (default): the delta method on the log of the
            intensity, from the parameter covariance (the inverse observed
            information), as :meth:`cif_cb`; it cannot go negative.
            ``"crow"``: Crow's (1982) exact bounds, for a ``CrowAMSAA``
            model at the end of its test only; the reciprocals of the
            :meth:`mtbf_cb` bounds (see there).

        Returns
        -------

        numpy array
            The confidence bounds on the intensity.

        Examples
        --------

        >>> from surpyval.recurrent import CrowAMSAA
        >>> x = [40, 110, 210, 340, 500, 690, 920, 1180, 1480, 1800, 2000]
        >>> c = [0] * 10 + [1]
        >>> model = CrowAMSAA.fit(x, c=c)
        >>> model.iif_cb(2000.0, alpha_ci=0.2).round(5)
        array([0.00188, 0.00591])
        """
        check_option("method", method, ("wald", "crow"))
        check_option("bound", bound, BOUNDS)
        if method == "crow":
            return self._reciprocal_cb(
                self._crow_mtbf_cb(x, alpha_ci, _OPPOSITE[bound]), bound
            )
        self._check_fitted()
        x = np.atleast_1d(np.asarray(x, dtype=float))
        se = delta_method_se(
            lambda params: self.dist.iif(x, *params),
            self._mle,
            self.covariance(),
        )
        return log_transformed_cb(self.iif(x), se, alpha_ci, bound)

    @keeps_query_shape
    def mtbf_cb(
        self,
        x: ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> np.ndarray:
        """
        Confidence bounds on the instantaneous MTBF (:meth:`mtbf`) at
        ``x``. A lower bound on the MTBF is the reciprocal of an upper
        bound on the intensity, and the other way round; this method does
        the flipping, so ``bound="lower"`` is a lower bound on the MTBF.

        The deliverable of a reliability growth test is usually the lower
        bound on the *demonstrated* MTBF, ``mtbf_cb(T, bound="lower",
        method="crow")`` at the end of the test ``T``, compared with the
        requirement.

        Parameters
        ----------

        x: array_like
            Values at which to compute the confidence bounds. With
            ``method="crow"`` every value must be the end of the test.
        alpha_ci: float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound: {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as an ``(len(x), 2)`` array with
            columns ``[lower, upper]``; one-sided bounds have the shape of
            ``x``.
        method: {'wald', 'crow'}, optional
            ``"wald"`` (default): the reciprocals of the delta-method
            bounds of :meth:`iif_cb`, for any model and any ``x``.

            ``"crow"``: the exact bounds of Crow (1982), which
            MIL-HDBK-189C tabulates, for a ``CrowAMSAA`` model fitted by
            maximum likelihood, at the end of the test only: ``M_hat * L``
            and ``M_hat * U``, with coefficients that depend only on the
            number of failures ``N`` and the level. Two designs have them:

            - *time terminated*: every item observed from 0 to the same
              time ``T`` (each ends in a ``c = 1`` row at ``T``, or the
              data are truncated at ``T``), with ``N >= 1`` failures in
              all. The bounds invert the test of the MTBF conditional on
              the sufficient statistic of the shape, so they are exact in
              the sense of a discrete test: they hold at least their
              level (with few failures, a lower bound at 90% covers about
              92% to 97% of the time). With ``N = 1`` the upper bound is
              infinite.
            - *failure terminated*: one item, observed to its ``N``-th
              failure (``N >= 2``) with no ``c = 1`` row. Then
              ``M_hat / M`` is a pivot (the product of independent
              Gamma(N) and Gamma(N - 1) variables over ``N^2``) and the
              bounds hold their level exactly.

            Other data (delayed entry, different end times, interval or
            left censoring, several failure-terminated items) have no
            exact bound, and ``"crow"`` raises; use ``"wald"``.

        Returns
        -------

        numpy array
            The confidence bounds on the instantaneous MTBF.

        References
        ----------

        Crow, L. H. (1982), "Confidence interval procedures for the Weibull
        process with applications to reliability growth", Technometrics
        24(1), 67-72.

        MIL-HDBK-189C (2011), "Reliability Growth Management", Section 5.

        Examples
        --------

        A growth test of one prototype, stopped at 2000 hours. The 80%
        lower bound on the demonstrated MTBF, against a requirement of
        150 hours:

        >>> from surpyval.recurrent import CrowAMSAA
        >>> x = [40, 110, 210, 340, 500, 690, 920, 1180, 1480, 1800, 2000]
        >>> c = [0] * 10 + [1]
        >>> model = CrowAMSAA.fit(x, c=c)
        >>> lower = model.mtbf_cb(2000.0, alpha_ci=0.2, bound="lower",
        ...                       method="crow")
        >>> round(float(lower), 1)
        196.9
        >>> round(float(model.mtbf_cb(2000.0, alpha_ci=0.2, bound="lower")), 1)
        205.9
        """
        check_option("method", method, ("wald", "crow"))
        check_option("bound", bound, BOUNDS)
        if method == "crow":
            return self._crow_mtbf_cb(x, alpha_ci, bound)
        return self._reciprocal_cb(
            self.iif_cb(x, alpha_ci, _OPPOSITE[bound]), bound
        )

    @staticmethod
    def _reciprocal_cb(cb: np.ndarray, bound: str) -> np.ndarray:
        """The bounds on ``1 / f`` from bounds on a positive ``f``
        (``cb`` computed with the opposite ``bound``): the reciprocals,
        with a two-sided pair's columns swapped."""
        with np.errstate(divide="ignore"):
            out = 1.0 / np.asarray(cb, dtype=float)
        return out[..., ::-1] if bound == "two-sided" else out

    def _crow_mtbf_cb(
        self, x: ArrayLike, alpha_ci: float, bound: str
    ) -> np.ndarray:
        """Crow's (1982) exact bounds on the demonstrated MTBF; see
        :meth:`mtbf_cb`."""
        from surpyval.utils.linalg import bound_signs

        from .crow_amsaa import (
            crow_failure_terminated_coefficients,
            crow_time_terminated_coefficients,
        )

        alpha, signs = bound_signs(alpha_ci, bound)
        if not 0.0 < alpha_ci < 1.0:
            raise alpha_ci_error(alpha_ci)
        if self.dist.name != "Crow-AMSAA":
            raise ValueError(
                "method='crow' gives Crow's exact bounds for a CrowAMSAA "
                "model; this is a {} model. Use method='wald'.".format(
                    self.dist.name
                )
            )
        self._check_has_data("method='crow'")
        self._check_fitted()
        T, n_events, terminated = self._crow_design()
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if not np.all(x == T):
            raise ValueError(
                "method='crow' bounds the demonstrated MTBF at the end of the "
                "test, x = {:g}; for other times use method='wald'.".format(T)
            )
        if terminated == "time":
            L, U = crow_time_terminated_coefficients(n_events, alpha)
        else:
            L, U = crow_failure_terminated_coefficients(n_events, alpha)
        coefficients = np.where(signs < 0, L, U)
        cb = self.mtbf(x)[..., None] * coefficients
        return cb if bound == "two-sided" else cb[..., 0]

    def _crow_design(self, for_projection: bool = False) -> tuple:
        """``(T, N, "time" | "failure")`` for data with an exact Crow
        bound, else a ValueError saying why there is none.

        ``for_projection``: the design for :meth:`CrowAMSAA.projection`,
        which has no ``method=`` and takes a time-terminated test only,
        so its message says that rather than offer ``method='wald'``
        (#663).
        """
        windows = diagnostics.item_windows(self.data)
        entries = {entry for _, _, entry, _, _ in windows}
        closes = {close for _, _, _, close, _ in windows}
        explicit = {flag for _, _, _, _, flag in windows}
        n_events = int(sum(events.size for _, events, _, _, _ in windows))
        reason = None
        if entries != {0.0}:
            reason = "some items enter after time 0 (delayed entry)"
        elif explicit == {True} and len(closes) == 1:
            if n_events >= 1:
                return closes.pop(), n_events, "time"
            reason = "there are no failures"
        elif explicit == {False} and len(windows) == 1:
            if n_events >= 2:
                return closes.pop(), n_events, "failure"
            reason = "a failure-terminated test needs at least 2 failures"
        elif explicit == {False}:
            reason = "several items are failure terminated"
        else:
            reason = "the items' observation does not end at one time"
        if for_projection:
            raise ValueError(
                "A growth projection needs a time-terminated test: every "
                "system run from 0 to the same end of test, T, each "
                "recorded with a c=1 row at T; here {}.".format(reason)
            )
        raise ValueError(
            "Crow's exact bounds (method='crow') need a time-terminated test "
            "(every item observed from 0 to the same time) or a failure-"
            "terminated test of one item; here {}. Use method='wald'.".format(
                reason
            )
        )

    # Narrows the mixin plot (bounds options) -- same known divergence.
    def plot(  # type: ignore[override]
        self,
        ax: Any = None,
        plot_bounds: bool = True,
        *,
        alpha_ci: float = 0.05,
    ) -> Any:
        """
        Plot the fitted CIF over the nonparametric MCF of the data used to
        fit it, with a delta-method confidence band around the fitted curve
        when the model carries a likelihood.

        Parameters
        ----------

        ax: matplotlib axes, optional
            An axes object to draw the plot on. Creates a new one if not
            provided.
        plot_bounds: bool, optional
            Whether to draw the confidence band around the fitted CIF.
            Ignored for models with no likelihood (``how="MSE"`` fits and
            ``from_params`` models). Default is True.
        alpha_ci: float, optional
            The total tail probability of the band: a
            ``1 - alpha_ci`` confidence band. Default is 0.05. Keyword
            only.

        Returns
        -------

        matplotlib axes
            An axes object with the plot.
        """
        self._check_has_data("plot")
        x, r, d = self.data.to_xrd()
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()

        x_plot = np.linspace(0, self.data.x.max(), 1000)

        ax.step(x, (d / r).cumsum(), color="r", where="post")
        ax.plot(x_plot, self.cif(x_plot), color="b")
        if plot_bounds and hasattr(self, "_neg_ll"):
            cb = self.cif_cb(x_plot, alpha_ci=alpha_ci)
            ax.fill_between(
                x_plot,
                cb[:, 0],
                cb[:, 1],
                color="b",
                alpha=0.2,
                label=f"{(1 - alpha_ci) * 100:g}% Confidence Band",
            )
        return ax
