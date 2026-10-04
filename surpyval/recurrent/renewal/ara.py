from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike

from surpyval import Weibull
from surpyval.recurrent.renewal.fit_mixin import RenewalFitMixin
from surpyval.recurrent.renewal.renewal_model import (
    DiscountedMemory,
    RenewalModel,
    conditional_gaps,
    event_positions,
    rows_by_position,
)
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.pickling import Rebuilt
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    measure_from_entry,
    reject_gapped_observation,
    validate_lifetime_dist,
    validate_memory,
    validate_renewal_censoring,
    validate_renewal_times,
    validate_restoration,
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
    return ARAVirtualAges(T, np.zeros(T.size, dtype=int), m)(rho)


class ARAVirtualAges:
    """
    ``ara_virtual_ages`` for many items at once, laid end to end.

    ``arrival_times`` holds every item's cumulative event times with the
    rows grouped by item (``item``), as ``handle_xicn`` leaves them. The
    layout depends only on the data, so it is worked out once here and
    the likelihood then calls the object with each trial ``rho``.

    The rows are grouped by how many terms their sum has, ``min(m, i)``
    at event position ``i``: one group per position below ``m`` and one
    for every later position. Each group is a single array operation
    across all the items, so a fit no longer runs a Python step per event
    (#515). Each row's terms are gathered in the order the one-item loop
    took them and summed along a row, so the ages are bit-for-bit those of
    that loop.
    """

    def __init__(
        self, arrival_times: np.ndarray, item: np.ndarray, m: "int | float"
    ) -> None:
        self.T = np.asarray(arrival_times, dtype=float)
        by_position = rows_by_position(event_positions(item))
        if np.isinf(m):
            # Every position has its own number of terms: all of them.
            groups = list(enumerate(by_position))[1:]
        else:
            m = int(m)
            groups = list(enumerate(by_position[:m]))[1:]
            if len(by_position) > m:
                groups.append((m, np.concatenate(by_position[m:])))
        self.groups = groups
        self.max_terms = max((k for k, _ in self.groups), default=0)

    def __call__(self, rho: float) -> np.ndarray:
        T = self.T
        v = np.zeros(T.size)
        weights = (1.0 - rho) ** np.arange(self.max_terms)
        for n_terms, rows in self.groups:
            # Row r's terms are T[r - 1], T[r - 2], ... (newest first).
            lagged = T[rows[:, None] - 1 - np.arange(n_terms)]
            discounted = np.sum(weights[:n_terms] * lagged, axis=1)
            v[rows] = T[rows - 1] - rho * discounted
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

    The repair efficiency ``rho`` lies in ``[0, 1]``: ``rho = 1`` is a
    repair as good as new (whatever the memory ``m``: the virtual age
    returns to 0, an ordinary renewal process) and ``rho = 0`` one as bad as
    old (minimal repair, the non-homogeneous Poisson process of the
    distribution's cumulative hazard); ``rho = 1 - q`` for the Kijima ``q``
    of ``GeneralizedRenewal``. The fitted model prints ``rho`` with its
    standard error and Wald interval, and the conclusion of
    ``repair_test()``, the likelihood-ratio tests of the fit against
    perfect repair (``rho = 1``) and minimal repair (``rho = 0``).

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
    def _build_sampler(model: Any, n: int, state: Any = None) -> Callable:
        if state is not None:
            return ARA._state_sampler(model, state)
        rho = model.rho
        # The arrival times so far, discounted over the last m of them.
        memory = DiscountedMemory(n, rho, model.m)
        last_arrival = np.zeros(n)

        def step(idx: np.ndarray, u: np.ndarray) -> np.ndarray:
            latest = last_arrival[idx]
            age = latest - rho * memory.value(idx)
            gap = conditional_gaps(model.model, age, u)
            arrival = latest + gap
            last_arrival[idx] = arrival
            memory.record(idx, arrival)
            return gap

        return step

    @staticmethod
    def _state_sampler(model: Any, state: Any) -> Callable:
        """The sampler of sequences that start from units' current states
        (``UnitStates``): the memory holds each unit's own failure times,
        and its first gap is its residual life from its virtual age
        now."""
        rho = model.rho
        memory = DiscountedMemory.from_history(state.failures, rho, model.m)
        last_arrival = np.array(state.now - state.since_failure, dtype=float)
        since = np.array(state.since_failure, dtype=float)

        def step(idx: np.ndarray, u: np.ndarray) -> np.ndarray:
            latest = last_arrival[idx] + since[idx]
            age = latest - rho * memory.value(idx)
            gap = conditional_gaps(model.model, age, u)
            arrival = latest + gap
            last_arrival[idx] = arrival
            since[idx] = 0.0
            memory.record(idx, arrival)
            return gap

        return step

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
        interarrival = data.get_interarrival_times()
        virtual_ages = ARAVirtualAges(data.x, data.i, model.m)(model.rho)
        x_new = interarrival + virtual_ages
        # H(0) = 0 exactly, but some distributions take log(0) on the way.
        with np.errstate(divide="ignore"):
            return np.asarray(
                model.model.Hf(x_new) - model.model.Hf(virtual_ages),
                dtype=float,
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
        virtual_ages_at = ARAVirtualAges(data.x, data.i, m)
        interarrival = data.get_interarrival_times()
        c = data.c

        def negll_func(params: np.ndarray) -> float:
            rho = params[0]
            dist_params = params[1:]

            virtual_ages = virtual_ages_at(rho)
            x_new = interarrival + virtual_ages

            # Every item starts at virtual age 0, where some distributions
            # take log(0) on the way to the exact S(0) = 1 (a LogNormal's
            # log(x)); that warned thousands of times per fit.
            with np.errstate(divide="ignore"):
                log_sf_v = dist.log_sf(virtual_ages, *dist_params)
                ll_o = dist.log_df(x_new, *dist_params) - log_sf_v
                ll_right = dist.log_sf(x_new, *dist_params) - log_sf_v
            ll = np.where(c == 0, ll_o, 0.0)
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

        data : RecurrentEventData
            Data containing the recurrence details.
            An item with delayed entry (a ``tl``) is taken to be as
            new at entry, with its times counted from there (see
            :meth:`fit`). A finite right truncation ``tr`` ends an
            item's observation there, as a ``c=1`` row at ``tr``
            would (#624).
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
        validate_lifetime_dist(dist, type(self).__name__)
        validate_memory(m)
        validate_renewal_censoring(data.c, type(self).__name__)
        reject_gapped_observation(data, type(self).__name__)
        # Delayed entry: as new at entry (#615).
        data = measure_from_entry(data, type(self).__name__)
        validate_renewal_times(data, dist, type(self).__name__)

        neg_ll = self.create_negll_func(data, dist, m)
        dist_params0 = self._default_start(
            lambda: self._initial_dist_params(data, dist), init
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
            renewal_restoration=0.99,
        )
        rho, *dist_params = params
        model = dist.from_params(list(dist_params))
        out = self._make_model(model, rho, m)
        # The likelihood kept as what it is built from, so the model
        # pickles (#573).
        neg_ll = Rebuilt(self.create_negll_func, (data, dist, m), built=neg_ll)
        self._attach_inference(out, neg_ll, [rho, *dist_params], res, data)
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
        tl: "ArrayLike | None" = None,
    ) -> "RenewalModel":
        """
        Fit the ARA model.

        Parameters
        ----------

        x : array_like
            The event times, pooled over items (each row belongs to the item
            named in ``i``), measured from the start of each item's life.
        i : array_like, optional
            Identity of the item each row belongs to. Defaults to all rows
            belonging to one item.
        c : array_like, optional
            Censoring indicators: 0 an observed failure, 1 the
            right-censored end of an item's observation. Other codes raise
            a ``ValueError``. Defaults to all observed.
        n : array_like, optional
            The number of events each row stands for. This model takes exact
            events (``c=0``) and end-of-observation rows (``c=1``), each of
            which stands for one, so every ``n`` is 1 (``n > 1`` is refused:
            repeat the row for simultaneous events). Defaults to 1.
        dist : object, optional
            A surpyval distribution object. Default is Weibull.
        m : int or float, optional
            Memory of the ARA model; a positive integer or ``numpy.inf``.
            Default is 1 (equivalent to Kijima-I).
        init : list, optional
            Initial parameters ``[rho, *dist_params]`` for the optimizer.
        tl : array_like or scalar, optional
            Delayed entry: the time each item's observation began, when
            its failures before then were not recorded (a scalar for every
            item, or one value per row, the same on every row of an item).
            The item is taken to be **as new at entry** -- virtual age 0 at
            ``tl``, as after an overhaul -- so its times count from there
            and its history before entry plays no part. That is exact for
            an item renewed at entry and an assumption otherwise; the
            fitted model's ``data`` hold the times from entry.
            A negative ``tl`` (an entry age below 0, almost always a
            data error on an age scale) is used as given, with a
            ``UserWarning``.

        Returns
        -------

        RenewalModel
            A fitted renewal model.

        Examples
        --------
        Two systems observed to t = 60 (the ``c=1`` rows). The fitted
        repair efficiency is 1 -- as good as new -- so the model reduces to
        an ordinary renewal process, with the Weibull fitted to the times
        between failures:

        >>> import numpy as np
        >>> from surpyval.recurrent import ARA
        >>> x = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60])
        >>> i = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        >>> model = ARA.fit(x, i, c=c, m=2)
        >>> model.model.params.round(3)
        array([13.779,  1.917])
        >>> round(float(model.rho), 3)
        1.0
        """
        data = handle_xicn(x, i, c, n, tl=tl)
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

        RenewalModel
            A model built from the supplied parameters, for simulation.
        """
        validate_lifetime_dist(dist, type(self).__name__)
        validate_memory(m)
        validate_restoration(rho, "rho", (0, 1))
        model = dist.from_params(params)
        return self._make_model(model, rho, m)
