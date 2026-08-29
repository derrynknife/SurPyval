from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike

from surpyval import Weibull
from surpyval.recurrent.renewal.fit_mixin import RenewalFitMixin
from surpyval.recurrent.renewal.renewal_model import RenewalModel
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    reject_gapped_observation,
    reject_left_truncation,
    validate_renewal_censoring,
)


def kijima_ii_from_prev_interarrival(
    previous_interarrival_times: np.ndarray, q: float
) -> np.ndarray:
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


@singleton_fitter
class GeneralizedRenewal(RenewalFitMixin):
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
    Restoration Factor  : 0.15732122999163628
    Parameters          :
         alpha: 1.261337933121844
          beta: 8.93902321971521
    >>>
    >>> np.random.seed(0)
    >>> np_model = model.count_terminated_simulation(len(x), 5000)
    >>> np_model.mcf(np.array([1, 2, 3, 4, 5, 6]))
    array([0.1214   , 1.1772   , 2.406    , 3.919    , 5.804    , 8.6088822])
    """

    def kijima_i(self, v: float, x: float, q: float) -> float:
        return v + q * x

    def kijima_ii(self, v: float, x: float, q: float) -> float:
        return q * (v + x)

    def _resolve_virtual_age_function(self, kijima_type: str) -> Callable:
        if kijima_type == "i":
            return self.kijima_i
        if kijima_type == "ii":
            return self.kijima_ii
        raise ValueError(
            "Unknown kijima_type {!r}; must be 'i' or 'ii'".format(kijima_type)
        )

    @staticmethod
    def _build_sampler(model: Any) -> Callable:
        q = model.q
        virtual_age_function = model._virtual_age_function
        virtual_age = 0.0

        def sample(ui: float) -> float:
            nonlocal virtual_age
            u_adj = ui * model.model.sf(virtual_age)
            xi = model.model.qf(1 - u_adj) - virtual_age
            virtual_age = virtual_age_function(virtual_age, xi, q)
            return xi

        return sample

    def _make_model(
        self, underlying_model: Any, q: float, kijima_type: str
    ) -> "RenewalModel":
        out = RenewalModel(
            underlying_model,
            q,
            "q",
            "Restoration Factor",
            "Generalized Renewal",
            self._build_sampler,
            restoration_bounds=(0, None),
        )
        out.kijima_type = kijima_type
        out._virtual_age_function = self._resolve_virtual_age_function(
            kijima_type
        )
        return out

    def _rescaled_increments(self, model: Any, data: Any) -> np.ndarray:
        """
        Per-interval cumulative-hazard increments ``H(v_k + x_k) - H(v_k)``
        (the time-rescaling residuals) for a fitted Kijima renewal model, where
        ``v_k`` is the virtual age at the start of interval ``k`` and ``x_k``
        its interarrival time. Aligned with ``data`` rows. iid Exp(1) over the
        observed intervals under the fitted model.
        """
        q = model.q
        _, idx = np.unique(data.i, return_index=True)
        interarrival = data.get_interarrival_times()
        if model.kijima_type == "i":
            arrival_times = np.split(data.x, idx)[1:]
            cumulative_previous = np.concatenate(
                [np.concatenate([[0], arr[:-1]]) for arr in arrival_times]
            )
            virtual_ages = q * cumulative_previous
        else:
            prev_x_interarrival = np.concatenate(
                [
                    np.concatenate([[0], np.atleast_1d(arr)])[:-1]
                    for arr in np.split(interarrival, idx)[1:]
                ]
            )
            virtual_ages = np.concatenate(
                [
                    kijima_ii_from_prev_interarrival(arr, q)
                    for arr in np.split(prev_x_interarrival, idx)[1:]
                ]
            )
        x_new = interarrival + virtual_ages
        return np.asarray(
            model.model.Hf(x_new) - model.model.Hf(virtual_ages), dtype=float
        )

    def _refit(self, model: Any, data: Any) -> Any:
        """Refit this model family on ``data`` with the same lifetime
        distribution and Kijima type; used by the Cramer-von Mises bootstrap.
        """
        return self.fit_from_recurrent_data(
            data, dist=model.model.dist, kijima=model.kijima_type
        )

    def create_negll_func(
        self, data: Any, dist: Any, kijima: str = "i"
    ) -> Callable:
        _, idx = np.unique(data.i, return_index=True)
        c = data.c
        x_interarrival = data.get_interarrival_times()

        if kijima == "i":
            arrival_times = np.split(data.x, idx)[1:]
            cumulative_previous = np.concatenate(
                [np.concatenate([[0], arr[:-1]]) for arr in arrival_times]
            )

        elif kijima == "ii":
            prev_x_interarrival = np.concatenate(
                [
                    np.concatenate([[0], np.atleast_1d(arr)])[:-1]
                    for arr in np.split(x_interarrival, idx)[1:]
                ]
            )

        def negll_func(params: np.ndarray) -> float:
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

    def fit_from_recurrent_data(
        self,
        data: Any,
        dist: Any = Weibull,
        kijima: str = "i",
        init: "ArrayLike | None" = None,
    ) -> "RenewalModel":
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
        Restoration Factor  : 1.3316262291443964e-16
        Parameters          :
             alpha: 2.399029668688425
              beta: 2.753920042066547
        """
        validate_renewal_censoring(data.c, type(self).__name__)
        reject_left_truncation(data, type(self).__name__)
        reject_gapped_observation(data, type(self).__name__)

        neg_ll = self.create_negll_func(data, dist, kijima=kijima)
        # result is (very!!) sensitive to the initial value of q
        dist_params0 = (
            self._initial_dist_params(data, dist) if init is None else None
        )
        res, params = self._fit_restoration_ml(
            data,
            neg_ll,
            (0, None),
            "q",
            dist,
            (0.0001, 1.0, 2.0),
            dist_params0,
            init,
        )
        q, *dist_params = params
        model = dist.from_params(list(dist_params))
        out = self._make_model(model, q, kijima)
        self._attach_inference(
            out, neg_ll, [q, *dist_params], len(data.x), res, data
        )
        return out

    def fit(
        self,
        x: ArrayLike,
        i: "ArrayLike | None" = None,
        c: "ArrayLike | None" = None,
        n: "ArrayLike | None" = None,
        dist: Any = Weibull,
        kijima: str = "i",
        init: "ArrayLike | None" = None,
    ) -> "RenewalModel":
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
        Restoration Factor  : 1.3316262291443964e-16
        Parameters          :
             alpha: 2.399029668688425
              beta: 2.753920042066547
        """
        data = handle_xicn(x, i, c, n)
        return self.fit_from_recurrent_data(data, dist, kijima, init=init)

    def fit_from_parameters(
        self,
        params: ArrayLike,
        q: float,
        kijima: str = "i",
        dist: Any = Weibull,
    ) -> "RenewalModel":
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
        ...     [10, 2],
        ...     0.2,
        ...     dist=Normal
        ... )
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
        return self._make_model(model, q, kijima)
