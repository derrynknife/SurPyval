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
    validate_memory,
    validate_renewal_censoring,
)


def ara_virtual_ages(
    arrival_times: np.ndarray, rho: float, m: "int | float"
) -> np.ndarray:
    """
    Effective (virtual) age at the start of each interarrival for the
    Arithmetic Reduction of Age model with memory ``m`` (Doyen & Gaudoin,
    2004).

    ``arrival_times`` are the cumulative failure times for one item. The age at
    the start of the i-th interarrival uses the most recent ``min(m, i)``
    failures::

        v_i = T_{i-1} - rho * sum_{j=0}^{min(m, i) - 1} (1 - rho)^j T_{i-1-j}

    with ``v_0 = 0``. ``m = 1`` recovers the Kijima-I virtual age and
    ``m = inf`` recovers Kijima-II, so this generalises both.

    Parameters
    ----------
    arrival_times : array_like
        Cumulative failure times ``T_1, ..., T_L`` for a single item.
    rho : float
        Repair efficiency in ``[0, 1]``. ``rho = 1`` is perfect repair
        (as-good-as-new), ``rho = 0`` is minimal repair (as-bad-as-old).
    m : int or float
        Memory of the model; a positive integer or ``numpy.inf``.

    Returns
    -------
    numpy.ndarray
        The virtual age at the start of each interarrival.
    """
    T = np.asarray(arrival_times, dtype=float)
    length = T.size
    v = np.zeros(length)
    for i in range(1, length):
        upper = i if np.isinf(m) else min(int(m), i)
        j = np.arange(upper)
        v[i] = T[i - 1] - rho * np.sum(((1.0 - rho) ** j) * T[i - 1 - j])
    return v


@singleton_fitter
class ARA(RenewalFitMixin):
    """
    Arithmetic Reduction of Age (ARA) imperfect-repair model of Doyen and
    Gaudoin (2004).

    Each repair removes a fraction of the accumulated virtual age. With memory
    ``m`` the reduction is applied to the most recent ``m`` failure
    contributions, so the model interpolates between the two Kijima models that
    ``GeneralizedRenewal`` already provides: ``m = 1`` is Kijima-I (ARA1) and
    ``m = inf`` is Kijima-II (ARA-infinity). The interesting cases are the
    finite memories ``m >= 2``.

    Like the other renewal models there is no closed-form intensity, so the
    cumulative intensity is obtained by simulation (see ``mcf`` and ``plot``).

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.recurrent import ARA
    >>> import numpy as np
    >>>
    >>> x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11])
    >>> c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    >>> i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    >>>
    >>> model = ARA.fit(x, i, c=c, m=2)
    """

    @staticmethod
    def _build_sampler(model: Any) -> Callable:
        rho = model.rho
        m = model.m
        arrivals: list = []
        running = 0.0

        def sample(ui: float) -> float:
            nonlocal running
            if not arrivals:
                v = 0.0
            else:
                T = np.asarray(arrivals)
                n = T.size
                upper = n if np.isinf(m) else min(int(m), n)
                j = np.arange(upper)
                v = T[-1] - rho * np.sum(((1.0 - rho) ** j) * T[n - 1 - j])
            u_adj = ui * model.model.sf(v)
            xi = model.model.qf(1 - u_adj) - v
            running += xi
            arrivals.append(running)
            return xi

        return sample

    def _make_model(
        self, underlying_model: Any, rho: float, m: "int | float"
    ) -> "RenewalModel":
        out = RenewalModel(
            underlying_model,
            rho,
            "rho",
            "Repair Efficiency",
            "ARA Renewal",
            self._build_sampler,
            restoration_bounds=(0, 1),
        )
        out.m = m
        return out

    def _rescaled_increments(self, model: Any, data: Any) -> np.ndarray:
        """
        Per-interval cumulative-hazard increments ``H(v_k + x_k) - H(v_k)``
        (the time-rescaling residuals) for a fitted ARA model, with ``v_k`` the
        arithmetic-reduction virtual age at the start of interval ``k``.
        Aligned with ``data`` rows; iid Exp(1) over the observed intervals
        under the fitted model.
        """
        _, idx = np.unique(data.i, return_index=True)
        arrival_by_item = np.split(data.x, idx)[1:]
        interarrival = data.get_interarrival_times()
        virtual_ages = np.concatenate(
            [ara_virtual_ages(a, model.rho, model.m) for a in arrival_by_item]
        )
        x_new = interarrival + virtual_ages
        return np.asarray(
            model.model.Hf(x_new) - model.model.Hf(virtual_ages), dtype=float
        )

    def _refit(self, model: Any, data: Any) -> Any:
        """Refit this model family on ``data`` with the same lifetime
        distribution and memory; used by the Cramer-von Mises bootstrap."""
        return self.fit_from_recurrent_data(
            data, dist=model.model.dist, m=model.m
        )

    def create_negll_func(
        self, data: Any, dist: Any, m: "int | float"
    ) -> Callable:
        _, idx = np.unique(data.i, return_index=True)
        arrival_by_item = np.split(data.x, idx)[1:]
        interarrival = data.get_interarrival_times()
        c = data.c

        def negll_func(params: np.ndarray) -> float:
            rho = params[0]
            dist_params = params[1:]

            virtual_ages = np.concatenate(
                [ara_virtual_ages(a, rho, m) for a in arrival_by_item]
            )
            x_new = interarrival + virtual_ages

            ll_o = dist.log_df(x_new, *dist_params) - dist.log_sf(
                virtual_ages, *dist_params
            )
            ll = np.where(c == 0, ll_o, 0.0)

            ll_right = dist.log_sf(x_new, *dist_params) - dist.log_sf(
                virtual_ages, *dist_params
            )
            ll = np.where(c == 1, ll_right, ll)

            return -ll.sum()

        return negll_func

    def fit_from_recurrent_data(
        self,
        data: Any,
        dist: Any = Weibull,
        m: "int | float" = 1,
        init: "ArrayLike | None" = None,
    ) -> "RenewalModel":
        """
        Fit the ARA model from recurrent data.

        Parameters
        ----------

        data : RecurrentData
            Data containing the recurrence details.
        dist : Distribution, optional
            A surpyval distribution object. Default is Weibull.
        m : int or float, optional
            Memory of the ARA model; a positive integer or ``numpy.inf``.
            Default is 1 (equivalent to Kijima-I).
        init : list, optional
            Initial parameters ``[rho, *dist_params]`` for the optimizer.

        Returns
        -------

        RenewalModel
            A fitted renewal model.
        """
        validate_memory(m)
        validate_renewal_censoring(data.c, type(self).__name__)
        reject_left_truncation(data, type(self).__name__)
        reject_gapped_observation(data, type(self).__name__)

        neg_ll = self.create_negll_func(data, dist, m)
        dist_params0 = (
            self._initial_dist_params(data, dist) if init is None else None
        )
        res, params = self._fit_restoration_ml(
            data,
            neg_ll,
            (0, 1),
            "rho",
            dist,
            (0.1, 0.5, 0.9),
            dist_params0,
            init,
        )
        rho, *dist_params = params
        model = dist.from_params(list(dist_params))
        out = self._make_model(model, rho, m)
        self._attach_inference(
            out, neg_ll, [rho, *dist_params], len(data.x), res, data
        )
        return out

    def fit(
        self,
        x: ArrayLike,
        i: "ArrayLike | None" = None,
        c: "ArrayLike | None" = None,
        n: "ArrayLike | None" = None,
        dist: Any = Weibull,
        m: "int | float" = 1,
        init: "ArrayLike | None" = None,
    ) -> "RenewalModel":
        """
        Fit the ARA model.

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
        m : int or float, optional
            Memory of the ARA model; a positive integer or ``numpy.inf``.
            Default is 1 (equivalent to Kijima-I).
        init : list, optional
            Initial parameters ``[rho, *dist_params]`` for the optimizer.

        Returns
        -------

        RenewalModel
            A fitted renewal model.
        """
        data = handle_xicn(x, i, c, n)
        return self.fit_from_recurrent_data(data, dist, m, init=init)

    def fit_from_parameters(
        self,
        params: ArrayLike,
        rho: float,
        m: "int | float" = 1,
        dist: Any = Weibull,
    ) -> "RenewalModel":
        """
        Build an ARA model from given parameters.

        Parameters
        ----------

        params : list
            Parameters for the underlying lifetime distribution.
        rho : float
            Repair efficiency in ``[0, 1]``.
        m : int or float, optional
            Memory of the ARA model; a positive integer or ``numpy.inf``.
            Default is 1.
        dist : object, optional
            A surpyval distribution object. Default is Weibull.

        Returns
        -------

        ARA
            An ARA object built from the supplied parameters.
        """
        validate_memory(m)
        model = dist.from_params(params)
        return self._make_model(model, rho, m)
