import numpy as np
from scipy.optimize import minimize

from surpyval import Weibull
from surpyval.recurrent.inference import LikelihoodInferenceMixin
from surpyval.recurrent.simulation import RecurrenceSimulationMixin
from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    validate_renewal_censoring,
)


def ara_virtual_ages(arrival_times, rho, m):
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


class ARA(RecurrenceSimulationMixin, LikelihoodInferenceMixin):
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

    _restoration_param_name = "rho"

    def __init__(self, model, rho, m=1):
        self.model = model
        self.rho = rho
        self.m = m

    def __repr__(self):
        out = (
            "ARA Renewal SurPyval Model"
            + "\n=========================="
            + f"\nDistribution        : {self.model.dist.name}"
            + "\nFitted by           : MLE"
            + f"\nMemory (m)          : {self.m}"
            + f"\nRepair Efficiency   : {self.rho}"
        )

        param_string = "\n".join(
            [
                "{:>10}".format(name) + ": " + str(p)
                for p, name in zip(
                    self.model.params, self.model.dist.param_names
                )
            ]
        )

        return (
            out
            + "\nParameters          :\n"
            + "{params}".format(params=param_string)
        )

    @staticmethod
    def _validate_memory(m):
        if m == np.inf:
            return
        if not (isinstance(m, (int, np.integer)) and m >= 1):
            raise ValueError(
                "m must be a positive integer or numpy.inf; got {!r}".format(m)
            )

    def _new_sequence_sampler(self):
        rho = self.rho
        m = self.m
        arrivals = []
        running = 0.0

        def sample(ui):
            nonlocal running
            if not arrivals:
                v = 0.0
            else:
                T = np.asarray(arrivals)
                n = T.size
                upper = n if np.isinf(m) else min(int(m), n)
                j = np.arange(upper)
                v = T[-1] - rho * np.sum(((1.0 - rho) ** j) * T[n - 1 - j])
            u_adj = ui * self.model.sf(v)
            xi = self.model.qf(1 - u_adj) - v
            running += xi
            arrivals.append(running)
            return xi

        return sample

    @classmethod
    def create_negll_func(cls, data, dist, m):
        _, idx = np.unique(data.i, return_index=True)
        arrival_by_item = np.split(data.x, idx)[1:]
        interarrival = data.get_interarrival_times()
        c = data.c

        def negll_func(params):
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

    @classmethod
    def fit_from_recurrent_data(cls, data, dist=Weibull, m=1, init=None):
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

        ARA
            A fitted ARA object.
        """
        cls._validate_memory(m)
        validate_renewal_censoring(data.c, cls.__name__)

        if init is None:
            first_events = data.get_times_to_first_events()
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
                    data.interarrival_times, data.c, data.n
                ).params

        param_map = {"rho": 0, **dist.param_map}
        transform, inv_trans, _, _, _ = bounds_convert(
            data.x, [(0, 1), *dist.bounds], {}, param_map
        )

        neg_ll = cls.create_negll_func(data, dist, m)
        neg_ll_unbounded = lambda p: neg_ll(inv_trans(p))  # noqa: E731

        if init is None:
            results = []
            for rho_init in [0.1, 0.5, 0.9]:
                x0 = transform(np.array([rho_init, *dist_params]))
                res = minimize(neg_ll_unbounded, x0, method="Nelder-Mead")
                if res.success:
                    results.append(res)
            if not results:
                raise ValueError(
                    "Could not find a good solution. "
                    + "Try using `init` for better initial guess."
                )
            res = results[np.argmin([r.fun for r in results])]
        else:
            x0 = transform(np.array(init))
            res = minimize(neg_ll_unbounded, x0, method="Nelder-Mead")
            if not res.success:
                raise ValueError(
                    "Optimization with the provided `init` did not "
                    "converge. Try a different initial guess."
                )

        rho, *dist_params = inv_trans(res.x)
        model = dist.from_params(list(dist_params))
        out = cls(model, rho, m)
        out.res = res
        out.data = data
        out._neg_ll = neg_ll
        out._mle = np.asarray([rho, *dist_params], dtype=float)
        out._n_obs = len(data.x)
        return out

    @classmethod
    def fit(cls, x, i=None, c=None, n=None, dist=Weibull, m=1, init=None):
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

        ARA
            A fitted ARA object.
        """
        data = handle_xicn(x, i, c, n, as_recurrent_data=True)
        return cls.fit_from_recurrent_data(data, dist, m, init=init)

    @classmethod
    def fit_from_parameters(cls, params, rho, m=1, dist=Weibull):
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
        cls._validate_memory(m)
        model = dist.from_params(params)
        return cls(model, rho, m)
