import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import uniform

from surpyval import Weibull
from surpyval.recurrent.nonparametric import NonParametricCounting
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    validate_renewal_censoring,
)

DT_WARN = "Small increment encountered, may have trouble reaching T."


class GeneralizedOneRenewal:
    """
    A class to handle the G1 renewal process of Kaminskiy and Krivtsov, in
    which the jth interarrival time is the underlying lifetime distribution
    scaled by ``(1 + q) ** j``.

    Because scaling a random variable by a factor ``cj`` is equivalent to
    evaluating the base distribution on a rescaled time axis
    (``S_j(x) = S0(x / cj)``), the model is well defined for any non-negative
    lifetime distribution and does not need to know which parameter is the
    scale. Distributions whose support includes negative values (e.g. Normal,
    Gumbel) are rejected, as scaled interarrival times would not be guaranteed
    positive.

    Since the Generalised One Renewal Process does not have closed form
    solutions for the instantaneous intensity function and the cumulative
    intensity function these values cannot be calculated directly with this
    class. Instead, the model can be used to simulate recurrence data which is
    fitted to a ``NonParametricCounting`` model. This model can then be used
    to calculate the cumulative intensity function.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.recurrent import GeneralizedOneRenewal
    >>> import numpy as np
    >>>
    >>> x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])
    >>>
    >>> model = GeneralizedOneRenewal.fit(x, dist=Weibull)
    >>> model
    G1 Renewal SurPyval Model
    =========================
    Distribution        : Weibull
    Fitted by           : MLE
    Restoration Factor  : -0.1730184624683848
    Parameters          :
        alpha: 1.3919045967886952
         beta: 5.0088611892336115
    >>>
    >>> np.random.seed(0)
    >>> np_model = model.count_terminated_simulation(len(x), 5000)
    >>> np_model.mcf(np.array([1, 2, 3, 4, 5, 6]))
    array([0.1696    , 1.181     , 2.287     , 3.6694    , 5.58237925,
           8.54474531])
    """

    def __init__(self, model, q):
        self.model = model
        self.q = q

    def __repr__(self):
        out = (
            "G1 Renewal SurPyval Model"
            + "\n========================="
            + f"\nDistribution        : {self.model.dist.name}"
            + "\nFitted by           : MLE"
            + f"\nRestoration Factor  : {self.q}"
        )

        param_string = "\n".join(
            [
                "{:>10}".format(name) + ": " + str(p)
                for p, name in zip(
                    self.model.params, self.model.dist.param_names
                )
            ]
        )

        out = (
            out
            + "\nParameters          :\n"
            + "{params}".format(params=param_string)
        )

        return out

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
        base_params = self.model.params
        q = self.q
        self.initialize_simulation()

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            for j in range(0, events + 1):
                ui = self.get_uniform_random_number()
                # The jth interarrival is the base lifetime scaled by
                # (1 + q) ** j, so its quantiles are the base quantiles
                # multiplied by the same factor.
                cj = (1.0 + q) ** j
                xi = cj * self.model.dist.qf(ui, *base_params)
                running += xi
                xicn["x"].append(running)
                xicn["i"].append(i + 1)
                xicn["c"].append(0)
                xicn["n"].append(1)

        self.clear_simulation()

        model = NonParametricCounting.fit(**xicn)
        mask = model.mcf_hat < events
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
        base_params = self.model.params
        q = self.q
        self.initialize_simulation()
        convergence_problem = False

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            j = 0
            while True:
                ui = self.get_uniform_random_number()
                # The jth interarrival is the base lifetime scaled by
                # (1 + q) ** j, so its quantiles are the base quantiles
                # multiplied by the same factor.
                cj = (1.0 + q) ** j
                xi = cj * self.model.dist.qf(ui, *base_params)
                running += xi
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
            warnings.warn(DT_WARN)

        model = NonParametricCounting.fit(**xicn)
        model.var = None

        return model

    @classmethod
    def create_negll_func(cls, x, i, c, n, dist):
        def negll_func(params):
            ll = 0
            q = params[0]
            dist_params = params[1:]

            for item in set(i):
                mask_item = i == item
                x_item = np.atleast_1d(x[mask_item])
                c_item = np.atleast_1d(c[mask_item])
                n_item = np.atleast_1d(n[mask_item])
                for j in range(0, len(x_item)):
                    # The jth interarrival is the base lifetime scaled by
                    # cj = (1 + q) ** j. Scaling the random variable by cj is
                    # equivalent to evaluating the base distribution on a
                    # rescaled time axis: f_j(x) = f0(x / cj) / cj and
                    # S_j(x) = S0(x / cj).
                    cj = (1.0 + q) ** j
                    xj = x_item[j] / cj
                    if c_item[j] == 0:
                        ll += n_item[j] * (
                            dist.log_df(xj, *dist_params) - np.log(cj)
                        )
                    elif c_item[j] == 1:
                        ll += n_item[j] * dist.log_sf(xj, *dist_params)
            return -ll

        return negll_func

    @staticmethod
    def _check_dist_eligible(dist):
        """
        The G1 renewal process scales interarrival times by ``(1 + q) ** j``.
        For the scaled times to remain valid the base distribution must be a
        non-negative lifetime distribution; distributions with support over
        negative values (e.g. Normal, Gumbel) are not eligible.
        """
        if dist.support[0] < 0:
            raise ValueError(
                "{} has support {} which includes negative values; the G1 "
                "renewal process requires a non-negative lifetime "
                "distribution (e.g. Weibull, Exponential, Gamma, "
                "LogNormal).".format(dist.name, dist.support)
            )

    @classmethod
    def fit_from_recurrent_data(cls, data, dist=Weibull, init=None):
        """
        Fit the generalized renewal model from recurrent data.

        Parameters
        ----------

        data : RecurrentData
            Data containing the recurrence details.
        dist : Distribution, optional
            A surpyval distribution object. Default is Weibull.
        init : list, optional
            Initial parameters for the optimization algorithm.

        Returns
        -------

        GeneralizedOneRenewal
            A fitted GeneralizedOneRenewal object.

        Example
        -------

        >>> from surpyval import handle_xicn
        >>> from surpyval.recurrent import GeneralizedOneRenewal
        >>> import numpy as np
        >>>
        >>> x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
        >>> c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 , 1])
        >>> i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
        >>>
        >>> rec_data = handle_xicn(x, i, c)
        >>>
        >>> model = GeneralizedOneRenewal.fit_from_recurrent_data(rec_data)
        >>> model
        G1 Renewal SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MLE
        Restoration Factor  : 0.4270960618530103
        Parameters          :
            alpha: 1.3494830373118245
             beta: 2.7838386997223212
        """
        cls._check_dist_eligible(dist)
        validate_renewal_censoring(data.c, cls.__name__)
        if init is None:
            dist_params = dist.fit(
                data.interarrival_times, data.c, data.n
            ).params

        neg_ll = cls.create_negll_func(
            data.interarrival_times, data.i, data.c, data.n, dist
        )

        results = []
        # Iterate over different initial values for q
        # result is sensitive to initial value of q
        if init is None:
            for q_init in [0.0001, 1.0, 2.0]:
                init = [q_init, *dist_params]
                res = minimize(
                    neg_ll,
                    init,
                    bounds=[(-1, None), *dist.bounds],
                    method="Nelder-Mead",
                )
                if res.success:
                    results.append(res)

            if results == []:
                raise ValueError(
                    "Could not find a good solution. "
                    + "Try using `init` for better initial guess."
                )
            else:
                res = results[np.argmin([res.fun for res in results])]
        else:
            res = minimize(
                neg_ll,
                init,
                bounds=[(-1, None), *dist.bounds],
                method="Nelder-Mead",
            )
            if not res.success:
                raise ValueError(
                    "Optimization with the provided `init` did not "
                    "converge. Try a different initial guess."
                )

        underlying_model = dist.from_params(list(res.x[1:]))
        q = res.x[0]
        out = cls(underlying_model, q)
        out.res = res
        out.data = data
        return out

    @classmethod
    def fit(cls, x, i=None, c=None, n=None, dist=Weibull, init=None):
        """
        Fit the generalized renewal model.

        Parameters
        ----------

        x : array_like
            An array of event times.
        i : array_like, optional
            An array of item indices.
        c : array_like, optional
            An array of censoring indicators.
        n : array_like, optional
            An array of counts.
        dist : object, optional
            A surpyval distribution object. Default is Weibull.
        init : list, optional
            Initial parameters for the optimization algorithm.

        Returns
        -------

        GeneralizedOneRenewal
            A fitted GeneralizedOneRenewal object.

        Example
        -------

        >>> from surpyval.recurrent import GeneralizedOneRenewal
        >>> import numpy as np
        >>>
        >>> x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
        >>> c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 , 1])
        >>> i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
        >>>
        >>> model = GeneralizedOneRenewal.fit(x, i, c=c)
        >>> model
        G1 Renewal SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MLE
        Restoration Factor  : 0.4270960618530103
        Parameters          :
            alpha: 1.3494830373118245
             beta: 2.7838386997223212
        """
        data = handle_xicn(x, i, c, n, as_recurrent_data=True)
        return cls.fit_from_recurrent_data(data, dist=dist, init=init)

    @classmethod
    def fit_from_parameters(cls, params, q, dist=Weibull):
        """
        Fit the generalized renewal model from given parameters.

        Parameters
        ----------

        params : list
            A list of parameters for the survival analysis distribution.
        q : float
            Restoration factor used in the G1 renewal model.
        dist : object, optional
            A surpyval distribution object. Default is Weibull.

        Returns
        -------

        GeneralizedOneRenewal
            A fitted GeneralizedOneRenewal object.

        Example
        -------

        >>> from surpyval import Weibull
        >>> from surpyval.recurrent import GeneralizedOneRenewal
        >>>
        >>> model = GeneralizedOneRenewal.fit_from_parameters(
            [10, 2],
            0.2,
            dist=Weibull
        )
        >>> model
        G1 Renewal SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MLE
        Restoration Factor  : 0.2
        Parameters          :
             alpha: 10
              beta: 2
        """
        cls._check_dist_eligible(dist)
        model = dist.from_params(params)
        return cls(model, q)
