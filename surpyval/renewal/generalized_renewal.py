import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import uniform

from surpyval import Weibull
from surpyval.recurrence.nonparametric import NonParametricCounting
from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.utils.recurrent_utils import handle_xicn


def kijima_ii_from_prev_interarrival(previous_interarrival_times, q):
    """
    Takes the interarrival times from the previous event for a given item
    and returns the virtual age for each interarrival time.

    Assumes that the virtual age is 0 at the start of the observation and that
    the values are in ascending order.

    The Kijima-II is defined as:
    Vn = q * (Vn-1 + Xn)
    Where Vn is the virtual age at the nth event and Xn is the interarrival
    time between the n-1th and nth event.
    """
    v = 0
    return np.array(
        [v := q * (v + x) for x in previous_interarrival_times]  # noqa
    )


class GeneralizedRenewal:
    """
    A class to handle the generalized renewal process with different Kijima
    models.

    Since the Generalised Renewal Process does not have closed form solutions
    for the instantaneous intensity function and the cumulative intensity
    function these values cannot be calculated directly with this class.
    Instead, the model can be used to simulate recurrence data which is
    fitted to a ``NonParametricCounting`` model. This model can then be used
    to calculate the cumulative intensity function.

    Examples
    --------
    >>> from surpyval import GeneralizedRenewal, Weibull
    >>> import numpy as np
    >>>
    >>> x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])
    >>>
    >>> model = GeneralizedRenewal.fit(x, dist=Weibull)
    >>> model
    Generalized Renewal SurPyval Model
    ==================================
    Distribution        : Weibull
    Fitted by           : MLE
    Kijima Type         : i
    Restoration Factor  : 0.1573211400037486
    Parameters          :
        alpha: 1.261338468404201
        beta: 8.93900788677076
    >>>
    >>> np.random.seed(0)
    >>> np_model = model.count_terminated_simulation(len(x), 5000)
    >>> np_model.mcf(np.array([1, 2, 3, 4, 5, 6]))
    array([0.116     , 1.1804    , 2.4032    , 3.9166    , 5.81163625,
           8.77859347])
    """

    def __init__(self, model, q, kijima_type="i"):
        self.model = model
        self.q = q
        self.kijima_type = kijima_type
        if kijima_type == "i":
            self.virtual_age_function = self.kijima_i
        elif kijima_type == "ii":
            self.virtual_age_function = self.kijima_ii

    def __repr__(self):
        out = (
            "Generalized Renewal SurPyval Model"
            + "\n=================================="
            + f"\nDistribution        : {self.model.dist.name}"
            + "\nFitted by           : MLE"
            + f"\nKijima Type         : {self.kijima_type}"
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

    @classmethod
    def kijima_i(self, v, x, q):
        return v + q * x

    @classmethod
    def kijima_ii(self, v, x, q):
        return q * (v + x)

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
        q = self.q
        self.initialize_simulation()

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            virtual_age = 0
            running = 0
            for j in range(0, events):
                ui = self.get_uniform_random_number()
                u_adj = ui * self.model.sf(virtual_age)
                xi = self.model.qf(1 - u_adj) - virtual_age
                # Update virtual age
                virtual_age = self.virtual_age_function(virtual_age, xi, q)
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

    def time_terminated_simulation(self, T, items=1, tol=1e-2):
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
        q = self.q
        self.initialize_simulation()
        convergence_problem = False

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            virtual_age = 0
            j = 0
            while True:
                ui = self.get_uniform_random_number()
                u_adj = ui * self.model.sf(virtual_age)
                xi = self.model.qf(1 - u_adj) - virtual_age
                # Update virtual age
                virtual_age = self.virtual_age_function(virtual_age, xi, q)
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
            warnings.warn("Warning: Convergence Problem")
        model = NonParametricCounting.fit(**xicn)
        model.var = None
        return model

    @classmethod
    def create_negll_func(cls, data, dist, kijima="i"):
        _, idx = np.unique(data.i, return_index=True)
        c = data.c
        x_interarrival = data.get_interarrival_times()

        if kijima == "i":
            arrival_times = np.split(data.x, idx)[1:]
            cumulative_previous = [
                np.concatenate([[0], arr[:-1]]) for arr in arrival_times
            ]
            cumulative_previous = np.concatenate(cumulative_previous)

        elif kijima == "ii":
            prev_x_interarrival = np.concatenate(
                [
                    np.concatenate([[0], np.atleast_1d(arr)])[:-1]
                    for arr in np.split(x_interarrival, idx)[1:]
                ]
            )

        def negll_func(params):
            q = params[0]
            params = params[1:]

            if kijima == "i":
                # Kijima-I is defined by:
                # Vn+1 = Vn + q * Xn
                # Where Vn is the virtual age at the nth event and Xn is the
                # interarrival time between the n-1th and nth event.
                # Kijima-I is much simpler to implement than Kijima-II
                virtual_ages = q * cumulative_previous
            else:
                virtual_ages = np.concatenate(
                    [
                        kijima_ii_from_prev_interarrival(arr, q)
                        for arr in np.split(prev_x_interarrival, idx)[1:]
                    ]
                )

            x_new = x_interarrival + virtual_ages

            ll_o = dist.log_df(x_new, *params) - dist.log_sf(
                virtual_ages, *params
            )
            ll = np.where(c == 0, ll_o, 0)

            ll_right = dist.log_sf(x_new, *params) - dist.log_sf(
                virtual_ages, *params
            )
            ll = np.where(c == 1, ll_right, ll)

            return -ll.sum()

        return negll_func

    @classmethod
    def fit_from_recurrent_data(
        cls, data, dist=Weibull, kijima="i", init=None
    ):
        """
        Fit the generalized renewal model from recurrent data.

        Parameters
        ----------

        data : RecurrentData
            Data containing the recurrence details.
        dist : Distribution, optional
            A surpyval distribution object. Default is Weibull.
        kijima : str, optional
            Type of Kijima model to use, either "i" or "ii". Default is "i".
        init : list, optional
            Initial parameters for the optimization algorithm.

        Returns
        -------

        GeneralizedRenewal
            A fitted GeneralizedRenewal object.

        Example
        -------

        >>> from surpyval import GeneralizedRenewal, Weibull, handle_xicn
        >>> import numpy as np
        >>>
        >>> x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
        >>> c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 , 1])
        >>> i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
        >>>
        >>> recurrent_data = handle_xicn(x, i, c)
        >>>
        >>> model = GeneralizedRenewal.fit_from_recurrent_data(recurrent_data)
        >>> model
        Generalized Renewal SurPyval Model
        ==================================
        Distribution        : Weibull
        Fitted by           : MLE
        Kijima Type         : i
        Restoration Factor  : 1.594694243423234e-11
        Parameters          :
            alpha: 2.399029078569064
            beta: 2.753920439616154
        """
        first_events = data.get_times_to_first_events()
        if init is None:
            if len(first_events.x) < 2:
                dist_params = None
            else:
                try:
                    dist_params = dist.fit(
                        first_events.x, first_events.c, first_events.n
                    ).params
                    if dist_params.isna().any():
                        dist_params = None
                except Exception:
                    dist_params = None

            if dist_params is None:
                dist_params = dist.fit(
                    data.interarrival_times,
                    data.c,
                    data.n,
                ).params

        param_map = {"q": 0, **dist.param_map}
        transform, inv_trans, _, _, not_fixed = bounds_convert(
            data.x, [(0, None), *dist.bounds], {}, param_map
        )

        neg_ll_bounded = cls.create_negll_func(data, dist, kijima=kijima)
        neg_ll_unbounded = lambda params: neg_ll_bounded(  # noqa: E731
            inv_trans(params)
        )

        if init is None:
            # Iterate over different initial values for q
            # result is (very!!) sensitive to initial value of q
            results = []
            for q_init in [0.0001, 1.0, 2.0]:
                init = transform(np.array([1, *dist_params]))
                res = minimize(
                    neg_ll_unbounded,
                    init,
                    method="Nelder-Mead",
                )
                if res.success:
                    results.append(res)
            res = results[np.argmin([res.fun for res in results])]
        else:
            init = transform(np.array(init))
            res = minimize(
                neg_ll_unbounded,
                init,
                method="Nelder-Mead",
            )

        q, *dist_params = inv_trans(res.x)
        q = q
        model = dist.from_params(list(dist_params))
        out = cls(model, q, kijima)
        out.res = res
        out.data = data

        return out

    @classmethod
    def fit(
        cls, x, i=None, c=None, n=None, dist=Weibull, kijima="i", init=None
    ):
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
        kijima : str, optional
            Type of Kijima model to use, either "i" or "ii". Default is "i".
        init : list, optional
            Initial parameters for the optimization algorithm.

        Returns
        -------

        GeneralizedRenewal
            A fitted GeneralizedRenewal object.

        Example
        -------

        >>> from surpyval import GeneralizedRenewal, Weibull
        >>> import numpy as np
        >>>
        >>> x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
        >>> c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 , 1])
        >>> i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
        >>>
        >>> model = GeneralizedRenewal.fit(x, i, c=c)
        >>> model
        Generalized Renewal SurPyval Model
        ==================================
        Distribution        : Weibull
        Fitted by           : MLE
        Kijima Type         : i
        Restoration Factor  : 1.594694243423234e-11
        Parameters          :
            alpha: 2.399029078569064
            beta: 2.753920439616154
        """
        data = handle_xicn(x, i, c, n, as_recurrent_data=True)
        return cls.fit_from_recurrent_data(data, dist, kijima, init=init)

    @classmethod
    def fit_from_parameters(cls, params, q, kijima="i", dist=Weibull):
        """
        Fit the generalized renewal model from given parameters.

        Parameters
        ----------

        params : list
            A list of parameters for the survival analysis distribution.
        q : float
            Restoration factor used in the Kijima models.
        kijima : str, optional
            Type of Kijima model to use, either "i" or "ii". Default is "i".
        dist : object, optional
            A surpyval distribution object. Default is Weibull.

        Returns
        -------

        GeneralizedRenewal
            A fitted GeneralizedRenewal object.

        Example
        -------

        >>> from surpyval import GeneralizedRenewal, Normal
        >>>
        >>> model = GeneralizedRenewal.fit_from_parameters(
            [10, 2],
            0.2,
            dist=Normal
        )
        >>> model
        Generalized Renewal SurPyval Model
        ==================================
        Distribution        : Normal
        Fitted by           : MLE
        Kijima Type         : i
        Restoration Factor  : 0.2
        Parameters          :
                mu: 10
            sigma: 2
        """
        model = dist.from_params(params)
        return cls(model, q, kijima)
