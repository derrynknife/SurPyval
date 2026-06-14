import numpy as np
from scipy.optimize import minimize

from surpyval import Weibull
from surpyval.recurrent.renewal.renewal_model import RenewalModel
from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    reject_left_truncation,
    validate_renewal_censoring,
)


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
    >>> from surpyval import Weibull
    >>> from surpyval.recurrent import GeneralizedRenewal
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
    array([0.1214   , 1.1772   , 2.406    , 3.919    , 5.804    , 8.6088822])
    """

    @classmethod
    def kijima_i(self, v, x, q):
        return v + q * x

    @classmethod
    def kijima_ii(self, v, x, q):
        return q * (v + x)

    @classmethod
    def _resolve_virtual_age_function(cls, kijima_type):
        if kijima_type == "i":
            return cls.kijima_i
        if kijima_type == "ii":
            return cls.kijima_ii
        raise ValueError(
            "Unknown kijima_type {!r}; must be 'i' or 'ii'".format(kijima_type)
        )

    @staticmethod
    def _build_sampler(model):
        q = model.q
        virtual_age_function = model._virtual_age_function
        virtual_age = 0.0

        def sample(ui):
            nonlocal virtual_age
            u_adj = ui * model.model.sf(virtual_age)
            xi = model.model.qf(1 - u_adj) - virtual_age
            virtual_age = virtual_age_function(virtual_age, xi, q)
            return xi

        return sample

    @classmethod
    def _make_model(cls, underlying_model, q, kijima_type):
        out = RenewalModel(
            underlying_model,
            q,
            "q",
            "Restoration Factor",
            "Generalized Renewal",
            cls._build_sampler,
        )
        out.kijima_type = kijima_type
        out._virtual_age_function = cls._resolve_virtual_age_function(
            kijima_type
        )
        return out

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

        RenewalModel
            A fitted renewal model.

        Example
        -------

        >>> from surpyval import Weibull, handle_xicn
        >>> from surpyval.recurrent import GeneralizedRenewal
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
        validate_renewal_censoring(data.c, cls.__name__)
        reject_left_truncation(data, cls.__name__)
        first_events = data.get_times_to_first_events()
        if init is None:
            if len(first_events.x) < 2:
                dist_params = None
            else:
                try:
                    dist_params = dist.fit(
                        first_events.x, first_events.c, first_events.n
                    ).params
                    if np.isnan(dist_params).any():
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
                init = transform(np.array([q_init, *dist_params]))
                res = minimize(
                    neg_ll_unbounded,
                    init,
                    method="Nelder-Mead",
                )
                if res.success:
                    results.append(res)
            if not results:
                raise ValueError(
                    "Could not find a good solution. "
                    + "Try using `init` for better initial guess."
                )
            else:
                res = results[np.argmin([res.fun for res in results])]
        else:
            init = transform(np.array(init))
            res = minimize(
                neg_ll_unbounded,
                init,
                method="Nelder-Mead",
            )
            if not res.success:
                raise ValueError(
                    "Optimization with the provided `init` did not "
                    "converge. Try a different initial guess."
                )

        q, *dist_params = inv_trans(res.x)
        model = dist.from_params(list(dist_params))
        out = cls._make_model(model, q, kijima)
        out.res = res
        out.data = data
        out._neg_ll = neg_ll_bounded
        out._mle = np.asarray([q, *dist_params], dtype=float)
        out._n_obs = len(data.x)

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

        RenewalModel
            A fitted renewal model.

        Example
        -------

        >>> from surpyval import Weibull
        >>> from surpyval.recurrent import GeneralizedRenewal
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

        RenewalModel
            A fitted renewal model.

        Example
        -------

        >>> from surpyval import Normal
        >>> from surpyval.recurrent import GeneralizedRenewal
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
        return cls._make_model(model, q, kijima)
