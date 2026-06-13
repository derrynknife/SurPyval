import numpy as np
from matplotlib import pyplot as plt

from surpyval.recurrent.simulation import RecurrenceSimulationMixin


class ParametricRecurrenceModel(RecurrenceSimulationMixin):
    """
    A class for holding the parameters, data, and usefult methods for a
    fitted parametric recurrence model. This is the result of the ``fit`` calls
    from the counting distributions.

    Example
    -------

    >>> from surpyval import Exponential
    >>> from surpyval.recurrent import HPP
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

    def _new_sequence_sampler(self):
        x_prev = 0.0

        def sample(ui):
            nonlocal x_prev
            u_adj = ui * np.exp(-self.cif(x_prev))
            xi = self.inv_cif(-np.log(u_adj)) - x_prev
            x_prev += xi
            return xi

        return sample

    def _postprocess_simulated_model(self, model):
        if self.dist.name == "CoxLewis":
            model.mcf_hat += np.exp(self.params[0])
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

    def mcf(self, x):
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
