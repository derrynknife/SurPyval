import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import uniform

from surpyval.recurrence.nonparametric import NonParametricCounting


class ParametricRecurrenceModel:
    """
    A class for holding the parameters, data, and usefult methods for a
    fitted parametric recurrence model. This is the result of the ``fit`` calls
    from the counting distributions.

    Example
    -------

    >>> from surpyval import HPP, Exponential
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> x = Exponential.random(10, 1e-3).cumsum()
    >>> model = HPP.fit(x)
    """

    def __repr__(self):
        param_string = "\n".join(
            [
                "{:>10}".format(name) + ": " + str(p)
                for p, name in zip(self.params, self.dist.param_names)
            ]
        )
        return (
            "Parametric Recurrence SurPyval Model"
            + "\n=================================="
            + f"\nProcess             : {self.dist.name}"
            + "\nFitted by           : MLE"
            + "\nParameters          :\n"
            + param_string
        )

    def initialize_simulation(self):
        self.us = uniform.rvs(size=100_000).tolist()

    def clear_simulation(self):
        del self.us

    def get_uniform_random_number(self):
        try:
            return self.us.pop()
        except IndexError:
            self.initialize_simulation()
            return self.us.pop()

    def count_terminated_simulation(self, events, items=1):
        """
        Simulate count-terminated recurrence data based on the fitted model.

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
                u_adj = ui * np.exp(-self.cif(x_prev))
                xi = self.inv_cif(-np.log(u_adj)) - x_prev
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

    def time_terminated_simulation(self, T, items=1, tol=1e-5):
        """
        Simulate time-terminated recurrence data based on the fitted model.

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
                u_adj = ui * np.exp(-self.cif(x_prev))
                xi = self.inv_cif(-np.log(u_adj)) - x_prev
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
            warnings.warng(
                "Some timelines unable to reach T due to possible asymptote"
            )

        model = NonParametricCounting.fit(**xicn)

        if self.dist.name == "CoxLewis":
            model.mcf_hat += np.exp(self.params[0])
        model.var = None

        return model

    def cif(self, x):
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

    def iif(self, x):
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

    def inv_cif(self, x):
        x = np.array(x)
        if hasattr(self.dist, "inv_cif"):
            return self.dist.inv_cif(x, *self.params)
        else:
            raise ValueError(
                "Inverse cif undefined for {}".format(self.dist.name)
            )

    def plot(self, ax=None):
        """
        Compute the inverse of the cumulative incidence function (CIF) based
        on the fitted model, if it's defined for the distribution.

        Parameters
        ----------

        ax: matplotlib axes, optional
            An axes object to draw the plot on. Creates a new one if not
            provided.

        Returns
        -------

        matplotlib axes
            An axes object with the plot.
        """
        x, r, d = self.data.to_xrd()
        if ax is None:
            ax = plt.gcf().gca()

        x_plot = np.linspace(0, self.data.x.max(), 1000)

        ax.step(x, (d / r).cumsum(), color="r", where="post")
        ax.plot(x_plot, self.cif(x_plot), color="b")
        return ax
