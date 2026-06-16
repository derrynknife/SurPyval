import numpy as np
from matplotlib import pyplot as plt

from surpyval.recurrent.inference import LikelihoodInferenceMixin
from surpyval.recurrent.simulation import RecurrenceSimulationMixin


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

    def plot(self, ax=None):
        """
        PLots the CIF of the model against the data used to fit it.

        To do this, the plot method takes the average of the covariates, and
        uses them to calculate the CIF of the model. This is then plotted
        against the non-parametric MCF of the raw data. That is, the raw
        MCF is created without considering the covariates.

        Parameters
        ----------

        ax : matplotlib.axes.Axes, optional
            The axes to plot the data on. If None, the current axes will be
            used.

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
        return ax

    def _parameter_names(self):
        # The base-rate (intensity) parameters lead ``_mle``, followed by the
        # covariate coefficients.
        return [
            *self.param_names,
            *["beta_{}".format(i) for i in range(len(self.coeffs))],
        ]

    def _new_sequence_sampler(self):
        # The shared simulation driver requests one of these per item; the
        # covariate vector for the run is stashed on ``_sim_Z`` by the public
        # simulation entry points below.
        Z = self._sim_Z
        x_prev = 0.0

        def sample(ui):
            nonlocal x_prev
            u_adj = ui * np.exp(-self.cif(x_prev, Z))
            xi = self.inv_cif(-np.log(u_adj), Z) - x_prev
            x_prev += xi
            return xi

        return sample

    def _postprocess_simulated_model(self, model):
        if self.dist.name == "CoxLewis":
            model.mcf_hat += np.exp(self.params[0])
        return model

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
