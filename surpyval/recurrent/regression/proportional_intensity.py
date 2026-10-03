import warnings
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
from surpyval.univariate.regression._aliasing import (
    aliased_columns,
    constant_columns,
    warn_aliased,
)
from surpyval.utils.deprecation import REMOVED_IN
from surpyval.utils.linalg import delta_method_se, log_transformed_cb
from surpyval.utils.no_maximum import maximum_entry, restored_maximum
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import option_error


def alias_covariates(Z: ArrayLike, intercept: bool) -> np.ndarray:
    """The columns of ``Z`` whose coefficients a proportional-intensity
    fit cannot determine (#502), with one warning naming them.

    The intensity is :math:`\\Lambda_0(t) e^{\\beta' Z}`, so a column that
    is a linear combination of the others leaves the likelihood flat along
    a combination of their coefficients, and a constant column one along
    its coefficient and the baseline's scale, where the baseline has one
    (``intercept``: the HPP's rate, or an NHPP baseline with
    ``has_scale``); otherwise only a column of zeros is aliased. The check
    is the regressions' (:mod:`surpyval.univariate.regression._aliasing`),
    on the (centred) rows of ``Z``, the later of two collinear columns
    aliased, as in R.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[0] == 0 or Z.shape[1] == 0:
        return np.array([], dtype=int)
    if intercept:
        Zc = Z - Z.mean(axis=0)
        constant = constant_columns(Z)
    else:
        Zc = Z
        constant = np.all(Z == 0, axis=0)
    aliased = aliased_columns(Zc.T @ Zc, Z.shape[0], constant)
    if aliased.size:
        warn_aliased(
            aliased,
            (
                "they are constant (the baseline intensity's scale is the "
                "model's intercept) or a linear combination of the other "
                "columns"
                if intercept
                else "they are all zero or a linear combination of the "
                "other columns"
            ),
        )
    return aliased


class ProportionalIntensityModel(
    SerialisableMixin, RecurrenceSimulationMixin, LikelihoodInferenceMixin
):
    """
    Model to provide methods and attributes when using a fitted proportional
    intensity model.

    Simulation reuses the shared :class:`RecurrenceSimulationMixin` (seeding,
    ``max_events`` backstop and the count/time-terminated drivers); the only
    addition here is the per-item covariate vector ``Z``, which the simulation
    entry points take and thread through to the sampler. When the model was
    fitted by maximum likelihood it also carries the likelihood-inference
    behaviour (``log_likelihood``, ``aic``, ``bic``, ``standard_errors``) from
    :class:`LikelihoodInferenceMixin`.

    ``params`` holds the base-rate parameters and ``coeffs`` the covariate
    coefficients; ``parameter_names`` names both, base rate first, the order
    of :meth:`covariance` and :meth:`standard_errors`, so
    ``parameter_names[:len(params)]`` names ``params``. A coefficient the
    data cannot determine is ``nan`` and listed in :attr:`aliased`.

    Examples
    --------
    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval.recurrent import CrowAMSAA
    >>> from surpyval.recurrent import ProportionalIntensityNHPP
    >>> import numpy as np
    >>> data = load_rossi_static()
    >>> x = data['week'].values
    >>> # ``arrest`` is 1 for an arrest, so the censoring flag is its
    >>> # complement (1 for a subject still free at week 52)
    >>> c = 1 - data['arrest'].values
    >>> i = np.arange(len(x))  # one item per subject
    >>> Z = data[["fin", "age", "race", "wexp", "mar", "paro", "prio"]].values
    >>> model = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=CrowAMSAA)
    >>> type(model).__name__
    'ProportionalIntensityModel'
    >>> model.cif([1, 2, 3], Z.mean(axis=0))
    array([0.00107511, 0.00284451, 0.00502557])
    """

    # Populated by the fitters; declared for the type checker.
    kind: str
    parameterization: str
    support: tuple
    _fitter_dist: Any
    dist: Any
    params: "np.ndarray"
    coeffs: "np.ndarray"
    _rate_names: list
    bounds: tuple
    name: str
    data: Any
    how: str
    res: Any
    #: What the fit reached, one of ``MAXIMUM_STATES``
    #: (``surpyval.utils.no_maximum``), as its warnings say; ``"not
    #: applicable"`` for a model built from its parameters, ``"unknown"``
    #: for one restored from a dict saved without it.
    maximum: str = "not applicable"

    def __repr__(self) -> str:
        out = (
            "Proportional Intensity Recurrence Model"
            + "\n======================================="
            + "\nType                : Proportional Intensity"
            + "\nKind                : {kind}"
            + "\nParameterization    : {parameterization}"
        ).format(kind=self.kind, parameterization=self.parameterization)

        out += f"\nHazard Rate Model   : {self.dist.name}\n"

        out = out + "Base Rate Parameters:\n"
        for i, p in zip(self._rate_names, self.params):
            out += "    {i}  :  {p}\n".format(i=i, p=p)

        out = out + "\nCovariate Coefficients:\n"
        for i, p in enumerate(self.coeffs):
            out += "   beta_{i}  :  {p}\n".format(i=i, p=p)
        return out

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted proportional-intensity model to a plain,
        JSON-serialisable dict.

        Stores the base-rate intensity model's name, the base-rate ``params``
        and the covariate ``coeffs``; the reloaded model reproduces
        ``cif``/``iif`` (``Lambda_0(t; params) * exp(Z . coeffs)``) exactly.
        The likelihood-inference state (data, ``neg_ll`` closure) is not
        stored.

        See Also
        --------
        from_dict, to_json, from_json
        """
        return stamp_schema(
            {
                "model": "ProportionalIntensityModel",
                "kind": self.kind,
                "parameterization": self.parameterization,
                "dist": self.dist.name,
                "param_names": list(self._rate_names),
                "params": np.asarray(self.params, dtype=float).tolist(),
                "coeffs": np.asarray(self.coeffs, dtype=float).tolist(),
                **maximum_entry(self.maximum),
            }
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> "ProportionalIntensityModel":
        """
        Rebuild a proportional-intensity model from a :meth:`to_dict`
        dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict,
            "ProportionalIntensityModel",
            "a proportional-intensity model",
        )
        out = cls()
        out.kind = model_dict["kind"]
        out.parameterization = model_dict["parameterization"]
        if out.kind == "HPP":
            # the constant-rate baseline is the PI-HPP fitter itself
            import surpyval.recurrent as recurrent

            out.dist = recurrent.ProportionalIntensityHPP
            out.bounds = ((0, None),)
            out.support = (-np.inf, np.inf)
        else:
            out.dist = intensity_dist_by_name(model_dict["dist"])
        out._rate_names = list(model_dict["param_names"])
        out.params = np.array(model_dict["params"], dtype=float)
        out.coeffs = np.array(model_dict["coeffs"], dtype=float)
        out.maximum = restored_maximum(model_dict)
        return out

    @property
    def aliased(self) -> np.ndarray:
        """The columns of ``Z`` whose coefficients the data cannot
        determine (#502): a constant column (where the baseline has a
        scale, which is the intercept) or a linear combination of the
        others. Their coefficients are ``nan`` in ``coeffs`` (R's ``NA``),
        as are their standard errors, and predictions take them as 0."""
        return np.flatnonzero(np.isnan(np.asarray(self.coeffs, dtype=float)))

    def _coef(self) -> np.ndarray:
        """``coeffs`` with an aliased coefficient as 0, as the model
        predicts with it."""
        coeffs = np.asarray(self.coeffs, dtype=float)
        return np.where(np.isnan(coeffs), 0.0, coeffs)

    @keeps_query_shape
    def cif(self, x: ArrayLike, Z: ArrayLike) -> np.ndarray:
        """
        Compute the cumulative incidence function of the model with the
        parameters found by the fit method.


        Parameters
        ----------

        x : array_like
            The times to compute the CIF at.

        Z : array_like
            The covariates for the item.
        """
        return self.dist.cif(x, *self.params) * np.exp(Z @ self._coef())

    @keeps_query_shape
    def iif(self, x: ArrayLike, Z: ArrayLike) -> np.ndarray:
        """
        Compute the instantaneous incidence function of the model with the
        parameters found by the fit method.


        Parameters
        ----------

        x : array_like
            The times to at which  to compute the iif.

        Z : array_like
            The covariates for the item.
        """
        return self.dist.iif(x, *self.params) * np.exp(Z @ self._coef())

    def inv_cif(self, x: ArrayLike, Z: ArrayLike) -> np.ndarray:
        if hasattr(self.dist, "inv_cif"):
            return self.dist.inv_cif(
                x / np.exp(self._coef() @ Z), *self.params
            )
        else:
            raise ValueError(
                "Inverse cif undefined for {}".format(self.dist.name)
            )

    def _item_cif_map(self) -> dict:
        # Each item's cumulative intensity is the baseline scaled by its own
        # ``exp(Z'beta)`` factor. The covariates are per item (static), so the
        # item's Z is taken from its first row. Returns ``{item: cif(x)}`` for
        # the recurrence diagnostics.
        data = self.data
        cif_map = {}
        for item in np.unique(data.i):
            Z_item = np.asarray(data.Z[data.i == item][0], dtype=float)
            cif_map[item] = lambda x, Z=Z_item: self.cif(
                np.asarray(x, dtype=float), Z
            )
        return cif_map

    def residuals(self, kind: str = "cumulative_hazard") -> np.ndarray:
        """
        Residual diagnostics for the fitted model, from the time-rescaling
        theorem applied per item (each item's intensity is the baseline
        scaled by its covariate factor ``exp(Z'beta)``).

        Parameters
        ----------

        kind: {'cumulative_hazard', 'pit', 'martingale'}, optional
            ``'cumulative_hazard'`` returns the rescaled interarrival times
            ``cif(t_k) - cif(t_{k-1})`` of every observed event (pooled across
            items); see below for how far they are iid Exp(1). ``'pit'``
            applies the probability integral transform ``1 - exp(-e)`` to
            those residuals (U(0, 1) under the same conditions).
            ``'martingale'`` returns one residual per item: its observed
            event count minus the count the model expects over its
            observation window.

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
        cif_map = self._item_cif_map()
        if kind in ("cumulative_hazard", "pit"):
            e = diagnostics.cumulative_hazard_residuals(self.data, cif_map)
            if kind == "pit":
                return 1.0 - np.exp(-e)
            return e
        elif kind == "martingale":
            return diagnostics.martingale_residuals(self.data, cif_map)
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
        Run a trend test on the data this model was fitted to. The null
        hypothesis is a *homogeneous* Poisson process (no trend); the
        statistic uses only the event times and windows, not the covariates,
        so it checks whether a time-varying intensity was warranted at all.
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
        Cramer-von Mises goodness-of-fit test of the fitted proportional-
        intensity model.

        Conditional on the number of events an item shows in its window, the
        transforms ``[cif(t) - cif(entry)] / [cif(close) - cif(entry)]`` under
        the item's own covariate-scaled intensity ``Lambda_0(t) exp(Z'beta)``
        are iid U(0, 1) when the fitted model is the true one; the statistic
        measures their departure from uniformity. Because the parameters
        (base-rate and coefficients) were estimated from the same data, the
        p-value is a parametric bootstrap: data is simulated from the fitted
        model over the same per-item windows and covariates, the full
        regression model is refitted, and the statistic recomputed.

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
        self._check_has_data("cramer_von_mises")
        return diagnostics.cramer_von_mises_regression(
            self, n_boot=n_boot, random_state=random_state
        )

    @keeps_query_shape
    def cif_cb(
        self,
        x: ArrayLike,
        Z: ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds on the fitted CIF at ``x`` for covariates ``Z``,
        from the delta method.

        The variance of the fitted CIF is propagated from the joint
        covariance of the base-rate parameters and covariate coefficients
        (the inverse observed information) through the CIF's gradient, and
        the bounds are computed on the log scale so they cannot go negative.

        Parameters
        ----------

        x : array_like
            Values at which to compute the confidence bounds.
        Z : array_like
            The covariates for the item.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
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
        Z = np.asarray(Z, dtype=float)
        n_dist_params = len(self.params)

        def cif_at(theta: np.ndarray) -> np.ndarray:
            return self.dist.cif(x, *theta[:n_dist_params]) * np.exp(
                Z @ theta[n_dist_params:]
            )

        # An aliased coefficient is held at 0 and has no variance (#502).
        held = ~self._estimated()
        cov = self.covariance()
        cov[held, :] = 0.0
        cov[:, held] = 0.0
        se = delta_method_se(cif_at, self._mle_values(), cov)
        return log_transformed_cb(self.cif(x, Z), se, alpha_ci, bound)

    # Extends the mixin plot with covariates -- same known divergence.
    def plot(  # type: ignore[override]
        self,
        ax: Any = None,
        plot_bounds: bool = True,
        *,
        alpha_ci: float = 0.05,
    ) -> Any:
        """
        PLots the CIF of the model against the data used to fit it.

        To do this, the plot method takes the average of the covariates, and
        uses them to calculate the CIF of the model. This is then plotted
        against the non-parametric MCF of the raw data. That is, the raw
        MCF is created without considering the covariates. A delta-method
        confidence band is drawn around the fitted CIF.

        Parameters
        ----------

        ax : matplotlib.axes.Axes, optional
            The axes to plot the data on. If None, the current axes will be
            used.
        plot_bounds : bool, optional
            Whether to draw the confidence band around the fitted CIF.
            Default is True.
        alpha_ci : float, optional
            The total tail probability of the band: a
            ``1 - alpha_ci`` confidence band. Default is 0.05. Keyword
            only.

        Returns
        -------

        ax : matplotlib.axes.Axes
            The axes the data was plotted on.
        """
        self._check_has_data("plot")
        x, r, d = self.data.to_xrd()
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()

        x_plot = np.linspace(0, self.data.x.max(), 1000)
        Z_0 = self.data.Z.mean(axis=0)

        ax.step(x, (d / r).cumsum(), color="r", where="post")
        ax.plot(x_plot, self.cif(x_plot, Z_0), color="b")
        if plot_bounds and hasattr(self, "_neg_ll"):
            cb = self.cif_cb(x_plot, Z_0, alpha_ci=alpha_ci)
            ax.fill_between(
                x_plot,
                cb[:, 0],
                cb[:, 1],
                color="b",
                alpha=0.2,
                label=f"{(1 - alpha_ci) * 100:g}% Confidence Band",
            )
        return ax

    def _parameter_names(self) -> list:
        # The base-rate (intensity) parameters lead ``_mle``, followed by the
        # covariate coefficients.
        return [
            *self._rate_names,
            *["beta_{}".format(i) for i in range(len(self.coeffs))],
        ]

    @property
    def param_names(self) -> list:
        """The base-rate parameters' names: deprecated, and removed in
        v0.23. Use ``parameter_names[:len(params)]`` (``parameter_names``
        also names the coefficients)."""
        warnings.warn(
            "ProportionalIntensityModel.param_names is deprecated and will "
            "be removed in v{}; use 'parameter_names', which names the "
            "base-rate parameters and then the coefficients "
            "(parameter_names[:len(params)] names params).".format(REMOVED_IN),
            DeprecationWarning,
            stacklevel=2,
        )
        return list(self._rate_names)

    def _parameter_bounds(self) -> list:
        # The base-rate bounds come from the intensity model (PI-HPP stores
        # them on the fitted model directly; PI-NHPP's live on ``dist``); the
        # covariate coefficients are unbounded.
        dist_bounds = getattr(self, "bounds", None) or self.dist.bounds
        return [*dist_bounds, *[(None, None)] * len(self.coeffs)]

    def _unit_covariates(self, Z: ArrayLike) -> np.ndarray:
        """
        Validate ``Z`` as the covariate vector of one unit, for the
        simulation entry points and :meth:`mcf`.

        ``Z`` describes one unit's covariate history, so a missing value
        raises (the package's missing-value rule, in the Conventions page)
        rather than being simulated: a NaN intensity ran every sequence to
        ``max_events`` and then failed on the NaN event times.
        """
        try:
            Z_arr = np.atleast_1d(np.asarray(Z, dtype=float))
        except (TypeError, ValueError):
            raise ValueError(
                "Z must be one unit's covariate vector of numbers; got "
                "{!r}".format(Z)
            ) from None
        n_coeffs = np.size(self.coeffs)
        # One row (1, p) is the same unit's vector as (p,).
        if Z_arr.ndim > 1 and Z_arr.shape[0] == 1:
            Z_arr = Z_arr.reshape(-1)
        if Z_arr.ndim > 1 or Z_arr.size != n_coeffs:
            raise ValueError(
                "Z must be one unit's covariate vector with {} value(s), "
                "one per coefficient; got shape {}".format(
                    n_coeffs, np.shape(Z_arr)
                )
            )
        if np.isnan(Z_arr).any():
            raise ValueError(
                "Z has a missing (NaN) value; it is one unit's covariate "
                "vector, so every value is needed to simulate its events."
            )
        return Z_arr

    def _cif_args(self) -> tuple:
        # The shared inverse-CIF sampler threads these into cif/inv_cif; the
        # covariate vector for the run is stashed on ``_sim_Z`` by the public
        # simulation entry points below. (CoxLewis post-processing is handled
        # by the shared mixin.)
        return (self._sim_Z,)

    # Extends the mixin signature with the covariate vector ``Z``
    # -- a known signature divergence in the simulation API.
    def count_terminated_simulation(  # type: ignore[override]
        self,
        events: int,
        Z: ArrayLike,
        items: int = 1,
        random_state: "int | None" = None,
    ) -> Any:
        """
        Simulate count-terminated recurrence data based on the fitted model.

        Parameters
        ----------

        events: int
            Each sequence is simulated to its ``events + 1``-th event, and
            the returned MCF is kept only where it is below ``events``.
        Z: array_like
            Covariate vector applied to every simulated sequence. A missing
            (NaN) value raises a ``ValueError``.
        items: int, optional
            Number of items (or sequences) to simulate. Default is 1.
        random_state: int or numpy.random.Generator, optional
            Seed for a reproducible simulation.

        Returns
        -------

        NonParametricCounting
            An NonParametricCounting model built from the simulated data.
        """
        self._sim_Z = self._unit_covariates(Z)
        return super().count_terminated_simulation(
            events, items=items, random_state=random_state
        )

    # Extends the mixin signature with the covariate vector ``Z``
    # -- a known signature divergence in the simulation API.
    def time_terminated_simulation(  # type: ignore[override]
        self,
        T: float,
        Z: ArrayLike,
        items: int = 1,
        tol: float = 1e-8,
        max_events: int = 10_000,
        random_state: "int | None" = None,
    ) -> Any:
        """
        Simulate time-terminated recurrence data based on the fitted model.

        Parameters
        ----------

        T: float
            Time termination value.
        Z: array_like
            Covariate vector applied to every simulated sequence. A missing
            (NaN) value raises a ``ValueError``.
        items: int, optional
            Number of items (or sequences) to simulate. Default is 1.
        tol: float, optional
            Interarrival times below this value end a sequence early (a
            possible asymptote). Default is 1e-8.
        max_events: int, optional
            Hard per-sequence event cap that guarantees termination.
            Default is 10000.
        random_state: int or numpy.random.Generator, optional
            Seed for a reproducible simulation.

        Returns
        -------

        NonParametricCounting
            An NonParametricCounting model built from the simulated data.

        Warnings
        --------

        A sequence is ended early at its last event, which is kept as an
        observed event (no censoring row at ``T``), if an interarrival time
        falls below ``tol`` or it reaches ``max_events`` before T. A warning
        is raised in either case.
        """
        self._sim_Z = self._unit_covariates(Z)
        return super().time_terminated_simulation(
            T,
            items=items,
            tol=tol,
            max_events=max_events,
            random_state=random_state,
        )

    # Extends the mixin signature with the covariate vector ``Z``
    # -- a known signature divergence in the simulation API.
    def count_terminated_simulation_data(  # type: ignore[override]
        self,
        events: int,
        Z: ArrayLike,
        items: int = 1,
        random_state: "int | None" = None,
    ) -> Any:
        """
        Simulate count-terminated recurrence data and return the raw events.
        Like :meth:`count_terminated_simulation` but yields the simulated
        ``RecurrentEventData`` rather than the fitted MCF.
        """
        self._sim_Z = self._unit_covariates(Z)
        return super().count_terminated_simulation_data(
            events, items=items, random_state=random_state
        )

    # Extends the mixin signature with the covariate vector ``Z``
    # -- a known signature divergence in the simulation API.
    def time_terminated_simulation_data(  # type: ignore[override]
        self,
        T: float,
        Z: ArrayLike,
        items: int = 1,
        tol: float = 1e-8,
        max_events: int = 10_000,
        random_state: "int | None" = None,
    ) -> Any:
        """
        Simulate time-terminated recurrence data and return the raw events.
        Like :meth:`time_terminated_simulation` but yields the simulated
        ``RecurrentEventData`` rather than the fitted MCF.
        """
        self._sim_Z = self._unit_covariates(Z)
        return super().time_terminated_simulation_data(
            T,
            items=items,
            tol=tol,
            max_events=max_events,
            random_state=random_state,
        )

    # Extends the mixin signature with the covariate vector ``Z``
    # -- a known signature divergence in the simulation API.
    @keeps_query_shape
    def mcf(
        self,
        x: ArrayLike,
        Z: ArrayLike,
        items: int = 1000,
        random_state: "int | None" = None,
    ) -> Any:
        """
        Estimate the mean cumulative function at ``x`` for covariates ``Z`` by
        simulating ``items`` time-terminated sequences out to ``max(x)``.

        ``Z`` is one unit's covariate vector, so a missing (NaN) value in it
        raises a ``ValueError``, as it does for the simulation methods,
        instead of simulating every sequence to ``max_events``.
        """
        self._sim_Z = self._unit_covariates(Z)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if x.size == 0:
            # Nothing to simulate to (the horizon is the largest time).
            return np.empty(0)
        np_model = self.time_terminated_simulation(
            float(x.max()), Z, items=items, random_state=random_state
        )
        return np_model.mcf(x)
