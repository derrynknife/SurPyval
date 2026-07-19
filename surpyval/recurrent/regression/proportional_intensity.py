import json

import numpy as np
from matplotlib import pyplot as plt

from surpyval.recurrent import diagnostics
from surpyval.recurrent.inference import (
    LikelihoodInferenceMixin,
    delta_method_std_errors,
    log_transformed_cb,
)
from surpyval.recurrent.serialisation import intensity_dist_by_name
from surpyval.recurrent.simulation import RecurrenceSimulationMixin
from surpyval.serialisation import stamp_schema


class ProportionalIntensityModel(
    RecurrenceSimulationMixin, LikelihoodInferenceMixin
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

    Examples
    --------
    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval.recurrent import CrowAMSAA
    >>> from surpyval.recurrent import ProportionalIntensityNHPP
    >>> data = load_rossi_static()
    >>> x = data['week'].values
    >>> c = data['arrest'].values
    >>> Z = data[["fin", "age", "race", "wexp", "mar", "paro", "prio"]].values
    >>> model = ProportionalIntensityNHPP.fit(x, Z, c, dist=CrowAMSAA)
    >>> type(model)
    surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel
    >>> model.cif([1, 2, 3], Z.mean(axis=0))
    array([0.00625402, 0.04304137, 0.13302238])
    """

    def __repr__(self):
        out = (
            "Proportional Intensity Recurrence Model"
            + "\n======================================="
            + "\nType                : Proportional Intensity"
            + "\nKind                : {kind}"
            + "\nParameterization    : {parameterization}"
        ).format(kind=self.kind, parameterization=self.parameterization)

        out += f"\nHazard Rate Model   : {self.dist.name}\n"

        out = out + "Base Rate Parameters:\n"
        for i, p in zip(self.param_names, self.params):
            out += "    {i}  :  {p}\n".format(i=i, p=p)

        out = out + "\nCovariate Coefficients:\n"
        for i, p in enumerate(self.coeffs):
            out += "   beta_{i}  :  {p}\n".format(i=i, p=p)
        return out

    # -- serialisation -----------------------------------------------------

    def to_dict(self):
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
                "param_names": list(self.param_names),
                "params": np.asarray(self.params, dtype=float).tolist(),
                "coeffs": np.asarray(self.coeffs, dtype=float).tolist(),
            }
        )

    def to_json(self, fp):
        """Write :meth:`to_dict` to ``fp`` as JSON."""
        with open(fp, "w+") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, model_dict):
        """
        Rebuild a proportional-intensity model from a :meth:`to_dict`
        dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        if model_dict.get("model") != "ProportionalIntensityModel":
            raise ValueError(
                "Must create a proportional-intensity model from a "
                "ProportionalIntensityModel dict"
            )
        out = cls()
        out.kind = model_dict["kind"]
        out.parameterization = model_dict["parameterization"]
        if out.kind == "HPP":
            # the constant-rate baseline is the PI-HPP fitter itself
            import surpyval.recurrent as recurrent

            out.dist = recurrent.ProportionalIntensityHPP
            out.bounds = ((0, None),)
            out.support = (0.0, np.inf)
        else:
            out.dist = intensity_dist_by_name(model_dict["dist"])
        out.param_names = list(model_dict["param_names"])
        out.params = np.array(model_dict["params"], dtype=float)
        out.coeffs = np.array(model_dict["coeffs"], dtype=float)
        return out

    @classmethod
    def from_json(cls, fp):
        """Load a model from a JSON file written by :meth:`to_json`."""
        with open(fp, "r") as f:
            return cls.from_dict(json.load(f))

    def cif(self, x, Z):
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
        return self.dist.cif(x, *self.params) * np.exp(Z @ self.coeffs)

    def iif(self, x, Z):
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
        return self.dist.iif(x, *self.params) * np.exp(Z @ self.coeffs)

    def inv_cif(self, x, Z):
        if hasattr(self.dist, "inv_cif"):
            return self.dist.inv_cif(x / np.exp(self.coeffs @ Z), *self.params)
        else:
            raise ValueError(
                "Inverse cif undefined for {}".format(self.dist.name)
            )

    def _item_cif_map(self):
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

    def residuals(self, kind="cumulative_hazard"):
        """
        Residual diagnostics for the fitted model, from the time-rescaling
        theorem applied per item (each item's intensity is the baseline
        scaled by its covariate factor ``exp(Z'beta)``).

        Parameters
        ----------

        kind: {'cumulative_hazard', 'pit', 'martingale'}, optional
            ``'cumulative_hazard'`` returns the rescaled interarrival times
            ``cif(t_k) - cif(t_{k-1})`` of every observed event (pooled across
            items), which are iid Exp(1) under the fitted model. ``'pit'``
            applies the probability integral transform ``1 - exp(-e)`` to
            those residuals, giving iid U(0, 1) values. ``'martingale'``
            returns one residual per item: its observed event count minus the
            count the model expects over its observation window.

        Returns
        -------

        numpy array
            The residuals.
        """
        cif_map = self._item_cif_map()
        if kind in ("cumulative_hazard", "pit"):
            e = diagnostics.cumulative_hazard_residuals(self.data, cif_map)
            if kind == "pit":
                return 1.0 - np.exp(-e)
            return e
        elif kind == "martingale":
            return diagnostics.martingale_residuals(self.data, cif_map)
        raise ValueError(
            "`kind` must be 'cumulative_hazard', 'pit' or 'martingale'; "
            "got {!r}".format(kind)
        )

    def trend_test(self, test="laplace", alternative="two-sided"):
        """
        Run a trend test on the data this model was fitted to. The null
        hypothesis is a *homogeneous* Poisson process (no trend); the
        statistic uses only the event times and windows, not the covariates,
        so it checks whether a time-varying intensity was warranted at all.

        Parameters
        ----------

        test: {'laplace', 'mil_hdbk_189c'}, optional
            The trend test to run. Default is 'laplace'.
        alternative: {'two-sided', 'increasing', 'decreasing'}, optional
            The alternative hypothesis. Default is 'two-sided'.

        Returns
        -------

        TrendTestResult
            The test result, carrying the statistic, p-value and suggested
            trend direction.
        """
        return diagnostics.trend_test(
            self.data, test=test, alternative=alternative
        )

    def cramer_von_mises(self, n_boot=200, seed=None):
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
        seed: int or numpy.random.Generator, optional
            Seed for a reproducible p-value.

        Returns
        -------

        GoodnessOfFitResult
            The observed statistic and its bootstrap p-value.
        """
        return diagnostics.cramer_von_mises_regression(
            self, n_boot=n_boot, seed=seed
        )

    def cif_cb(self, x, Z, alpha_ci=0.05, bound="two-sided"):
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

        def cif_at(theta):
            return self.dist.cif(x, *theta[:n_dist_params]) * np.exp(
                Z @ theta[n_dist_params:]
            )

        se = delta_method_std_errors(cif_at, self._mle, self.covariance())
        return log_transformed_cb(self.cif(x, Z), se, alpha_ci, bound)

    def plot(self, ax=None, plot_bounds=True, confidence=0.95):
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
        confidence : float, optional
            The confidence level of the band. Default is 0.95.

        Returns
        -------

        ax : matplotlib.axes.Axes
            The axes the data was plotted on.
        """

        x, r, d = self.data.to_xrd()
        if ax is None:
            ax = plt.gcf().gca()

        x_plot = np.linspace(0, self.data.x.max(), 1000)
        Z_0 = self.data.Z.mean(axis=0)

        ax.step(x, (d / r).cumsum(), color="r", where="post")
        ax.plot(x_plot, self.cif(x_plot, Z_0), color="b")
        if plot_bounds and hasattr(self, "_neg_ll"):
            cb = self.cif_cb(x_plot, Z_0, alpha_ci=1.0 - confidence)
            ax.fill_between(
                x_plot,
                cb[:, 0],
                cb[:, 1],
                color="b",
                alpha=0.2,
                label=f"{confidence * 100}% Confidence Band",
            )
        return ax

    def _parameter_names(self):
        # The base-rate (intensity) parameters lead ``_mle``, followed by the
        # covariate coefficients.
        return [
            *self.param_names,
            *["beta_{}".format(i) for i in range(len(self.coeffs))],
        ]

    def _parameter_bounds(self):
        # The base-rate bounds come from the intensity model (PI-HPP stores
        # them on the fitted model directly; PI-NHPP's live on ``dist``); the
        # covariate coefficients are unbounded.
        dist_bounds = getattr(self, "bounds", None) or self.dist.bounds
        return [*dist_bounds, *[(None, None)] * len(self.coeffs)]

    def _cif_args(self):
        # The shared inverse-CIF sampler threads these into cif/inv_cif; the
        # covariate vector for the run is stashed on ``_sim_Z`` by the public
        # simulation entry points below. (CoxLewis post-processing is handled
        # by the shared mixin.)
        return (self._sim_Z,)

    def count_terminated_simulation(self, events, Z, items=1, seed=None):
        """
        Simulate count-terminated recurrence data based on the fitted model.

        Parameters
        ----------

        events: int
            Number of events to simulate per sequence.
        Z: array_like
            Covariate vector applied to every simulated sequence.
        items: int, optional
            Number of items (or sequences) to simulate. Default is 1.
        seed: int or numpy.random.Generator, optional
            Seed for a reproducible simulation.

        Returns
        -------

        NonParametricCounting
            An NonParametricCounting model built from the simulated data.
        """
        self._sim_Z = np.asarray(Z, dtype=float)
        return super().count_terminated_simulation(
            events, items=items, seed=seed
        )

    def time_terminated_simulation(
        self, T, Z, items=1, tol=1e-8, max_events=10_000, seed=None
    ):
        """
        Simulate time-terminated recurrence data based on the fitted model.

        Parameters
        ----------

        T: float
            Time termination value.
        Z: array_like
            Covariate vector applied to every simulated sequence.
        items: int, optional
            Number of items (or sequences) to simulate. Default is 1.
        tol: float, optional
            Interarrival times below this value end a sequence early (a
            possible asymptote). Default is 1e-8.
        max_events: int, optional
            Hard per-sequence event cap that guarantees termination.
            Default is 10000.
        seed: int or numpy.random.Generator, optional
            Seed for a reproducible simulation.

        Returns
        -------

        NonParametricCounting
            An NonParametricCounting model built from the simulated data.

        Warnings
        --------

        A sequence is terminated early and right-censored at its last event if
        an interarrival time falls below ``tol`` or it reaches ``max_events``
        before T. A warning is raised in either case.
        """
        self._sim_Z = np.asarray(Z, dtype=float)
        return super().time_terminated_simulation(
            T, items=items, tol=tol, max_events=max_events, seed=seed
        )

    def count_terminated_simulation_data(self, events, Z, items=1, seed=None):
        """
        Simulate count-terminated recurrence data and return the raw events.
        Like :meth:`count_terminated_simulation` but yields the simulated
        ``RecurrentEventData`` rather than the fitted MCF.
        """
        self._sim_Z = np.asarray(Z, dtype=float)
        return super().count_terminated_simulation_data(
            events, items=items, seed=seed
        )

    def time_terminated_simulation_data(
        self, T, Z, items=1, tol=1e-8, max_events=10_000, seed=None
    ):
        """
        Simulate time-terminated recurrence data and return the raw events.
        Like :meth:`time_terminated_simulation` but yields the simulated
        ``RecurrentEventData`` rather than the fitted MCF.
        """
        self._sim_Z = np.asarray(Z, dtype=float)
        return super().time_terminated_simulation_data(
            T, items=items, tol=tol, max_events=max_events, seed=seed
        )

    def mcf(self, x, Z, items=1000, seed=None):
        """
        Estimate the mean cumulative function at ``x`` for covariates ``Z`` by
        simulating ``items`` time-terminated sequences out to ``max(x)``.
        """
        self._sim_Z = np.asarray(Z, dtype=float)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        np_model = self.time_terminated_simulation(
            float(x.max()), Z, items=items, seed=seed
        )
        return np_model.mcf(x)
