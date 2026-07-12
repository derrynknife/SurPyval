import numpy as np
from matplotlib import pyplot as plt

from surpyval.recurrent.inference import (
    LikelihoodInferenceMixin,
    delta_method_std_errors,
    log_transformed_cb,
)
from surpyval.recurrent.simulation import RecurrenceSimulationMixin


class ParametricRecurrenceModel(
    RecurrenceSimulationMixin, LikelihoodInferenceMixin
):
    """
    A class for holding the parameters, data, and usefult methods for a
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

    def _parameter_names(self):
        return list(self.dist.param_names)

    def _parameter_bounds(self):
        return list(self.dist.bounds)

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

    def cif_cb(self, x, alpha_ci=0.05, bound="two-sided"):
        """
        Confidence bounds on the fitted CIF at ``x``, from the delta method.

        The variance of the fitted CIF is propagated from the parameter
        covariance (the inverse observed information) through the CIF's
        gradient, and the bounds are computed on the log scale -- the same
        construction as the exponential Greenwood bounds on the nonparametric
        MCF -- so they cannot go negative.

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
        se = delta_method_std_errors(
            lambda params: self.dist.cif(x, *params),
            self._mle,
            self.covariance(),
        )
        return log_transformed_cb(self.cif(x), se, alpha_ci, bound)

    def plot(self, ax=None, plot_bounds=True, confidence=0.95):
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
        confidence: float, optional
            The confidence level of the band. Default is 0.95.

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
        if plot_bounds and hasattr(self, "_neg_ll"):
            cb = self.cif_cb(x_plot, alpha_ci=1.0 - confidence)
            ax.fill_between(
                x_plot,
                cb[:, 0],
                cb[:, 1],
                color="b",
                alpha=0.2,
                label=f"{confidence * 100}% Confidence Band",
            )
        return ax
