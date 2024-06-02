import numpy as np
from matplotlib import pyplot as plt

from surpyval.recurrence.nonparametric import NonParametricCounting


class ProportionalIntensityModel:
    """
    Model to provide methods and attributes when using a fitted proportional
    intensity model.

    Examples
    --------
    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval import Crow
    >>> from surpyval import ProportionalIntensityNHPP
    >>> data = load_rossi_static()
    >>> x = data['week'].values
    >>> c = data['arrest'].values
    >>> Z = data[["fin", "age", "race", "wexp", "mar", "paro", "prio"]].values
    >>> model = ProportionalIntensityNHPP.fit(x, Z, c, dist=Crow)
    >>> type(model)
    surpyval.recurrence.regression.proportional_intensity.ProportionalIntensityModel
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

    def count_terminated_simulation(self, events, Z, items=1):
        """
        Simulate count-terminated recurrence data based on the fitted model.
        That is, if you want to simulate up to 10 events terminating the
        simulation at 10 events, this is the method to use.

        This method is for use with monte carlo methods that use the NHPP model
        to simulate recurrence data.

        Parameters
        ----------

        events: int
            Number of events to simulate.
        items: int, optional
            Number of items (or sequences) to simulate. Default is 1.

        Returns
        -------

        NonParametricCounting
            An NonParametricCounting model built from the simulated data.
        """
        self.initialize_simulation()

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            j = 0
            x_prev = 0
            for j in range(0, events + 1):
                ui = self.get_uniform_random_number()
                u_adj = ui * np.exp(-self.cif(x_prev, Z))
                xi = self.inv_cif(-np.log(u_adj), Z) - x_prev
                running += xi
                x_prev = running
                xicn["i"].append(i + 1)
                xicn["n"].append(1)
                xicn["x"].append(running)
                xicn["c"].append(0)

        self.clear_simulation()

        model = NonParametricCounting.fit(**xicn)

        if self.dist.name == "CoxLewis":
            model.mcf_hat += np.exp(self.params[0])

        mask = model.mcf_hat <= events
        model.x = model.x[mask]
        model.mcf_hat = model.mcf_hat[mask]
        model.var = None
        return model

    def time_terminated_simulation(self, T, Z, items=1, tol=1e-5):
        """
        Simulate time-terminated recurrence data based on the fitted model. If
        you want to simulate up to some time T, this is the method to use. This
        simulation can run into errors when a sequence approaches very steep
        hazard rates. Warnings are provided if this is likely the case.

        This model is useful for doing monte carlo simulations with the NHPP
        model.

        Parameters
        ----------

        T: float
            Time termination value.
        items: int, optional
            Number of items (or sequences) to simulate. Default is 1.
        tol: float, optional
            Tolerance for interarrival times to stop an individual sequence.

        Returns
        -------

        NonParametricCounting
            An NonParametricCounting model built from the simulated data.

        Warnings
        --------

        If any of the simulated sequences seem to not reach the time
        termination value T due to possible asymptote, a warning message will
        be printed to notify the user about potential convergence problems in
        the simulation.
        """
        self.initialize_simulation()
        convergence_problem = False

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            j = 0
            x_prev = 0
            while True:
                ui = self.get_uniform_random_number()
                u_adj = ui * np.exp(-self.cif(x_prev, Z))
                xi = self.inv_cif(-np.log(u_adj), Z) - x_prev
                running += xi
                x_prev = running
                xicn["i"].append(i + 1)
                xicn["n"].append(1)
                if running > T:
                    xicn["x"].append(T)
                    xicn["c"].append(1)
                    break
                elif xi < tol:
                    convergence_problem = True
                    xicn["x"].append(running)
                    xicn["c"].append(0)
                    break
                else:
                    xicn["x"].append(running)
                    xicn["c"].append(0)
                    j += 1

        self.clear_simulation()

        if convergence_problem:
            print("Maybe...")

        model = NonParametricCounting.fit(**xicn)

        if self.dist.name == "CoxLewis":
            model.mcf_hat += np.exp(self.params[0])
        model.var = None

        return model
