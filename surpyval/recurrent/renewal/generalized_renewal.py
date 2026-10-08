from __future__ import annotations

from typing import Any, Callable

import autograd.numpy as anp
import numpy as np
from autograd.extend import defvjp, primitive
from autograd.tracer import getval, isbox
from numpy.typing import ArrayLike

from surpyval import Weibull
from surpyval.recurrent.renewal.fit_mixin import RenewalFitMixin
from surpyval.recurrent.renewal.renewal_model import (
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
    validate_renewal_censoring,
    validate_renewal_times,
    validate_restoration,
)
from surpyval.utils.validation import option_error


def _previous_in_item(values: np.ndarray, item: np.ndarray) -> np.ndarray:
    """Each row's previous value within its item, 0 at an item's first row
    (rows grouped by item)."""
    values = np.asarray(values)
    previous = np.zeros(values.size, dtype=np.result_type(values, 0))
    previous[1:] = values[:-1]
    previous[event_positions(item) == 0] = 0
    return previous


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


#: Below this ratio of a gap to the virtual age it starts from, the
#: survival's drop over the gap is integrated from the hazard
#: (``_accurate_where_aged``).
_AGED = 1e-3


def virtual_age_log_likelihood(
    dist: Any,
    params: Any,
    age: Any,
    gap: np.ndarray,
    c: np.ndarray,
    fresh: np.ndarray,
) -> Any:
    """The log-likelihood of the gaps ``gap`` between events, each from
    the virtual age ``age`` it starts at, for the lifetime ``dist`` with
    ``params``: ``log f(v + x) - log S(v)`` for a failure (``c == 0``) and
    ``log S(v + x) - log S(v)`` for the censored end of an observation
    (``c == 1``), summed over the rows in row order. ``fresh`` marks the
    rows whose age is 0 by construction (an item's first gap).

    The gaps far shorter than the virtual age they start from are
    computed differently (#630). The survival's drop over a gap, ``log
    S(v + x) - log S(v)``, is the difference of two logs of size
    ``H(v)``. Once ``v`` dwarfs ``x`` it is lost to rounding, and ``v +
    x`` itself rounds to ``v`` from ``v / x`` of 1e16: a Kijima-II ``q``
    of 611 ages an item to 1e30 within a dozen failures, every drop read 0
    or a rounding step, and the likelihood appeared to rise without bound
    (-172 at q = 611 against -284 at the maximum, q = 0.95). Below a gap
    of ``_AGED`` of the age, the drop is minus the hazard's integral over
    the gap, taken by Simpson's rule (to a relative ``(x / v)**4``), and
    the density term is the hazard at ``v + x`` times that survival.

    Written for autograd (#710): ``age`` may be traced, and every term is
    evaluated only on the rows that use it, so that a term a row does not
    use cannot put a nan into the gradient (``0 * inf``). The fresh rows
    take their age as the constant 0: the derivative of ``log S(v)`` at
    ``v = 0`` is infinite for a hazard that is there (a Weibull ``beta <
    1``), though the age does not depend on the parameters there; and
    for a lifetime on ``[0, inf)`` their ``log S(0)`` is the 0 it is.
    Each row's term is the same arithmetic as before (#515), bit for bit.
    """
    return VirtualAgeLikelihood(gap, c, fresh)(dist, params, age)


class VirtualAgeLikelihood:
    """``virtual_age_log_likelihood`` for fixed gaps, censoring and fresh
    rows: the likelihood calls the object with each trial's ages. Which
    rows take which term depends on the data and on which gaps are aged;
    that bookkeeping is kept for the last few patterns of aged gaps (at
    most fits' trials, none), so a call is the arithmetic alone."""

    def __init__(
        self, gap: Any, c: Any, fresh: Any, aged: float = _AGED
    ) -> None:
        self.gap = np.asarray(gap, dtype=float)
        self.c = np.asarray(c)
        self.fresh = np.asarray(fresh, dtype=bool)
        # The ratio of gap to age below which the gap is integrated (0:
        # never, as ARA's likelihood has always been computed)
        self.aged = aged
        self._layouts: dict = {}

    def _layout(self, aged: np.ndarray) -> tuple:
        key = aged.tobytes() if aged.any() else b""
        layout = self._layouts.get(key)
        if layout is not None:
            return layout
        c, fresh = self.c, self.fresh
        first = np.flatnonzero(fresh)
        later = np.flatnonzero(~fresh & ~aged)
        main = np.concatenate([first, later])
        failed = np.flatnonzero(c[main] == 0)
        censored = np.flatnonzero(c[main] == 1)
        aged_rows = np.flatnonzero(aged)
        aged_failed = np.flatnonzero(c[aged_rows] == 0)
        aged_censored = np.flatnonzero(c[aged_rows] == 1)
        rows = np.concatenate(
            [
                main[failed],
                main[censored],
                aged_rows[aged_failed],
                aged_rows[aged_censored],
            ]
        )
        # Back in row order, so the sum is the one it always was.
        order = np.argsort(rows, kind="stable")
        layout = (
            np.zeros(first.size),
            later,
            self.gap[main],
            failed,
            censored,
            aged_rows,
            self.gap[aged_rows],
            aged_failed,
            aged_censored,
            order,
        )
        if len(self._layouts) >= 8:
            self._layouts.clear()
        self._layouts[key] = layout
        return layout

    def __call__(self, dist: Any, params: Any, age: Any) -> Any:
        if not (isbox(age) or isbox(params) or any(map(isbox, params))):
            return self._plain(dist, params, age)
        with np.errstate(all="ignore"):
            aged = ~self.fresh & (
                self.gap < self.aged * np.asarray(getval(age), dtype=float)
            )
            (
                zeros,
                later,
                gap_main,
                failed,
                censored,
                aged_rows,
                gap_aged,
                aged_failed,
                aged_censored,
                order,
            ) = self._layout(aged)
            terms = []
            v_later = age[later]
            x_new = gap_main + anp.concatenate([zeros, v_later])
            if dist.support[0] >= 0:
                # S(0) = 1: autograd's derivative of a Weibull's log S
                # at 0 is nan in alpha (0 * inf) for beta < 1.
                log_sf_v = anp.concatenate(
                    [zeros, dist.log_sf(v_later, *params)]
                )
            else:
                log_sf_v = dist.log_sf(
                    anp.concatenate([zeros, v_later]), *params
                )
            if failed.size:
                log_df = dist.log_df(x_new[failed], *params)
                terms.append(log_df - log_sf_v[failed])
            if censored.size:
                log_sf = dist.log_sf(x_new[censored], *params)
                terms.append(log_sf - log_sf_v[censored])
            if aged_rows.size:
                v, x = age[aged_rows], gap_aged
                h0 = dist.hf(v, *params)
                h_mid = dist.hf(v + 0.5 * x, *params)
                h1 = dist.hf(v + x, *params)
                drop = -x / 6.0 * (h0 + 4.0 * h_mid + h1)
                if aged_failed.size:
                    terms.append(anp.log(h1[aged_failed]) + drop[aged_failed])
                if aged_censored.size:
                    terms.append(drop[aged_censored])
        if not terms:
            return 0.0
        return anp.sum(anp.concatenate(terms)[order])

    def _plain(self, dist: Any, params: Any, age: np.ndarray) -> float:
        """The same likelihood where nothing is traced (a derivative-free
        search's trials): each term on every row, then the one each row
        takes, which is fewer and larger array operations than taking
        each term on its own rows. The rows' terms, and their sum, are
        the same, bit for bit."""
        gap, c = self.gap, self.c
        x_new = gap + age
        with np.errstate(all="ignore"):
            log_sf_v = dist.log_sf(age, *params)
            ll_o = dist.log_df(x_new, *params) - log_sf_v
            ll_right = dist.log_sf(x_new, *params) - log_sf_v
            aged = ~self.fresh & (gap < self.aged * age)
            if np.any(aged):
                v, x = age[aged], gap[aged]
                h0 = dist.hf(v, *params)
                h_mid = dist.hf(v + 0.5 * x, *params)
                h1 = dist.hf(v + x, *params)
                drop = -x / 6.0 * (h0 + 4.0 * h_mid + h1)
                ll_o = np.array(ll_o, dtype=float)
                ll_right = np.array(ll_right, dtype=float)
                ll_o[aged] = np.log(h1) + drop
                ll_right[aged] = drop
        ll = np.where(c == 0, ll_o, 0.0)
        ll = np.where(c == 1, ll_right, ll)
        return float(ll.sum())


class KijimaIIVirtualAges:
    """
    The Kijima-II virtual ages ``V_k = q * (V_{k-1} + X_k)`` for many items
    at once, as ``kijima_ii_from_prev_interarrival`` gives them for one.

    ``previous_interarrival`` holds each row's previous interarrival time
    (0 at an item's first row) with the rows grouped by item (``item``), as
    ``handle_xicn`` leaves them. The layout depends only on the data, so it
    is worked out once here and the likelihood then calls the object with
    each trial ``q``.

    The recursion runs along the event positions, each one a single array
    step across every item that has an event there, instead of a Python
    step per item and per event (#515). Once fewer than ``_MIN_VECTOR``
    items are left (the long tail of a few long items, or a single system)
    their remaining events are stepped one at a time, where a scalar step
    is the cheaper one. Both do the same arithmetic in the same order as
    the one-item loop, so the ages are bit-for-bit the same.

    Calling the object is differentiable by autograd (#710): the ages'
    derivatives in ``q`` follow the same recursion (``derivative``), to
    any order, so the likelihood has an exact gradient and Hessian. The
    closed form, ``V_k = sum_j q**(k - j + 1) X_j``, would overflow when
    ``q > 1``.
    """

    #: Fewest items at an event position for a whole-array step there.
    _MIN_VECTOR = 8

    def __init__(
        self, previous_interarrival: np.ndarray, item: np.ndarray
    ) -> None:
        self.x = np.asarray(previous_interarrival, dtype=float)
        position = event_positions(item)
        by_position = rows_by_position(position)
        n_vector = 0
        while (
            n_vector < len(by_position)
            and by_position[n_vector].size >= self._MIN_VECTOR
        ):
            n_vector += 1
        self.vector_steps = by_position[:n_vector]
        # The items still running after those positions, from the row at
        # position ``n_vector`` to the item's last row.
        self.scalar_runs: list = []
        if n_vector < len(by_position):
            first_rows = np.flatnonzero(position == 0)
            ends = np.append(first_rows[1:], position.size)
            for start in by_position[n_vector]:
                end = ends[np.searchsorted(first_rows, start, "right") - 1]
                self.scalar_runs.append((int(start), int(end)))
        self.from_start = n_vector == 0

    def __call__(self, q: Any) -> Any:
        if isbox(q):
            return _kijima_ii_ages(q, self, 0)
        return self._ages(q)

    def derivative(self, q: float, order: int = 0) -> np.ndarray:
        """The ``order``-th derivative of the ages in ``q`` (the ages
        themselves at 0). Differentiating ``V_k = q (V_{k-1} + X_k)``
        ``n`` times gives ``V_k^(n) = q V_{k-1}^(n) + n V_{k-1}^(n-1)``,
        plus ``X_k`` for ``n = 1``, so each derivative runs along the same
        steps as the ages, carrying the lower orders with it."""
        if order == 0:
            return self._ages(q)
        q = float(q)
        x = self.x
        d = np.zeros((order + 1, x.size))
        n = np.arange(2, order + 1)[:, None]

        def step(before: np.ndarray, gap: np.ndarray) -> np.ndarray:
            new = np.empty_like(before)
            new[0] = q * (before[0] + gap)
            new[1] = q * before[1] + before[0] + gap
            new[2:] = q * before[2:] + n * before[1:-1]
            return new

        with np.errstate(over="ignore", invalid="ignore"):
            for k, rows in enumerate(self.vector_steps):
                if k:
                    before = d[:, rows - 1]
                else:
                    before = np.zeros((order + 1, rows.size))
                d[:, rows] = step(before, x[rows])
            # One row at a time in Python floats, where they are cheaper.
            for start, end in self.scalar_runs:
                if self.from_start:
                    state = [0.0] * (order + 1)
                else:
                    state = d[:, start - 1].tolist()
                run = []
                for gap in x[start:end].tolist():
                    state = [
                        q * (state[0] + gap),
                        q * state[1] + state[0] + gap,
                    ] + [
                        q * state[j] + j * state[j - 1]
                        for j in range(2, order + 1)
                    ]
                    run.append(state)
                d[:, start:end] = np.array(run).T
        return d[order]

    def _ages(self, q: float) -> np.ndarray:
        x = self.x
        v = np.empty(x.size)
        for k, rows in enumerate(self.vector_steps):
            before = 0.0 if k == 0 else v[rows - 1]
            v[rows] = q * (before + x[rows])
        q_scalar = float(q)
        for start, end in self.scalar_runs:
            age = 0.0 if self.from_start else float(v[start - 1])
            ages = []
            for gap in x[start:end].tolist():
                age = q_scalar * (age + gap)
                ages.append(age)
            v[start:end] = ages
        return v


@primitive
def _kijima_ii_ages(q: Any, ages: KijimaIIVirtualAges, order: int) -> Any:
    """``ages.derivative(q, order)``, as an autograd primitive whose
    derivative in ``q`` is the next order's: differentiable to any
    order."""
    return ages.derivative(q, order)


defvjp(
    _kijima_ii_ages,
    lambda ans, q, ages, order: lambda g: anp.sum(
        g * _kijima_ii_ages(q, ages, order + 1)
    ),
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

    **The restoration factor q.** Each repair sets the system's *virtual
    age* -- the age its next failure time is drawn at, conditional on
    surviving to it -- from its real age:

    - ``q = 0``: as good as new (every repair is a renewal);
    - ``0 < q < 1``: better than old but worse than new;
    - ``q = 1``: as bad as old (minimal repair): the virtual age is the
      real age, and the process is the non-homogeneous Poisson process
      whose cumulative intensity is the distribution's cumulative hazard
      (for a Weibull, the Crow-AMSAA power law);
    - ``q > 1``: worse than old: each repair ages the system further.

    The Kijima type says what a repair acts on. Kijima I (``kijima="i"``)
    removes a fraction ``1 - q`` of the age gained since the last repair,
    ``v_n = v_{n-1} + q x_n``: damage from before is never repaired.
    Kijima II (``kijima="ii"``) removes that fraction of the whole
    accumulated age, ``v_n = q (v_{n-1} + x_n)``.

    ``q`` is often poorly determined: only the order and spacing of each
    system's failures carry information about it, and a few failures per
    system leave a wide interval. The fitted model prints ``q`` with its
    standard error and Wald interval, and the conclusion of
    :meth:`RenewalModel.repair_test
    <surpyval.recurrent.renewal.renewal_model.RenewalModel.repair_test>`,
    the likelihood-ratio tests of the fit against perfect repair (``q =
    0``) and minimal repair (``q = 1``): "not determined" when the data
    are consistent with both.

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
    Parameters          : Wald 95% intervals
               estimate      se  lower 95%  upper 95%
        q        0.1573 0.03311     0.1041     0.2376
        alpha     1.261  0.1238      1.041      1.529
        beta      8.939   2.499      5.168      15.46
    Repair test: both perfect and minimal repair rejected: q = 0.1573 is
          between perfect and minimal repair (LR tests, q = 0: p =
          0.000108; q = 1: p = 3.59e-05)
    >>>
    >>> np.random.seed(0)
    >>> np_model = model.count_terminated_simulation(len(x), 5000)
    >>> np_model.mcf(np.array([1, 2, 3, 4, 5, 6]))
    array([0.1154    , 1.1696    , 2.4062    , 3.937     , 5.8038    ,
           8.58730072])
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
        raise option_error("kijima_type", kijima_type, ("i", "ii"))

    @staticmethod
    def _build_sampler(model: Any, n: int, state: Any = None) -> Callable:
        q = model.q
        virtual_age_function = model._virtual_age_function
        if state is not None:
            return GeneralizedRenewal._state_sampler(model, state)
        virtual_age = np.zeros(n)

        def step(idx: np.ndarray, u: np.ndarray) -> np.ndarray:
            age = virtual_age[idx]
            gap = conditional_gaps(model.model, age, u)
            virtual_age[idx] = virtual_age_function(age, gap, q)
            return gap

        return step

    @staticmethod
    def _state_sampler(model: Any, state: Any) -> Callable:
        """The sampler of sequences that start from units' current states
        (``UnitStates``): each unit's first gap is its residual life from
        its virtual age now, and the repair after it acts on the whole
        time since the unit's last repair."""
        q = model.q
        virtual_age_function = model._virtual_age_function
        after = np.array(state.after_repair, dtype=float)
        since = np.array(state.since_failure, dtype=float)

        def step(idx: np.ndarray, u: np.ndarray) -> np.ndarray:
            age = after[idx]
            elapsed = since[idx]
            gap = conditional_gaps(model.model, age + elapsed, u)
            after[idx] = virtual_age_function(age, elapsed + gap, q)
            since[idx] = 0.0
            return gap

        return step

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
        interarrival = data.get_interarrival_times()
        if model.kijima_type == "i":
            virtual_ages = q * _previous_in_item(data.x, data.i)
        else:
            virtual_ages = KijimaIIVirtualAges(
                _previous_in_item(interarrival, data.i), data.i
            )(q)
        x_new = interarrival + virtual_ages
        # H(0) = 0 exactly, but some distributions take log(0) on the way.
        with np.errstate(divide="ignore"):
            return np.asarray(
                model.model.Hf(x_new) - model.model.Hf(virtual_ages),
                dtype=float,
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
        c = data.c
        x_interarrival = data.get_interarrival_times()
        # Every item starts at virtual age 0.
        log_likelihood = VirtualAgeLikelihood(
            x_interarrival, c, event_positions(data.i) == 0
        )

        if kijima == "i":
            cumulative_previous = _previous_in_item(data.x, data.i)
        elif kijima == "ii":
            kijima_ii_ages = KijimaIIVirtualAges(
                _previous_in_item(x_interarrival, data.i), data.i
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
                virtual_ages = kijima_ii_ages(q)

            return -log_likelihood(dist, params, virtual_ages)

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

        data : RecurrentEventData
            Data containing the recurrence details.
            An item with delayed entry (a ``tl``) is taken to be as
            new at entry, with its times counted from there (see
            :meth:`fit`). A finite right truncation ``tr`` ends an
            item's observation there, as a ``c=1`` row at ``tr``
            would (#624).
        dist : Distribution, optional
            A surpyval distribution object. Default is Weibull.
        kijima : str, optional
            Type of Kijima model to use, either "i" (a repair acts on the
            age gained since the last one) or "ii" (on the whole
            accumulated age). Default is "i".
        init : list, optional
            Initial parameters for the optimization algorithm.

        Returns
        -------

        RenewalModel
            A fitted renewal model. Its restoration factor ``q`` is 0 for
            a repair as good as new, 1 for one as bad as old (minimal
            repair, the non-homogeneous Poisson process) and above 1 for
            one that leaves the system worse than before it failed (see the
            class docstring); the model prints it with its standard error
            and Wald interval, and ``repair_test()`` tests it against
            perfect and minimal repair.

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
        Restoration Factor  : 7.274462742318132e-17
        Parameters          : Wald 95% intervals
                   estimate     se  lower 95%  upper 95%
            q     7.274e-17    nan          0    0.09235
            alpha     2.399  0.287      1.898      3.033
            beta      2.754 0.6516      1.732      4.379
        Note: q = 7.274e-17 is at the edge of its range, so it has no standard
              error. Its interval is the profile-likelihood one, from the
              edge; the others' are Wald intervals with q held there.
        Repair test: consistent with perfect repair; minimal repair rejected
              (LR tests, q = 0: p = 1; q = 1: p = 0.000669)
        """
        # Resolving the Kijima type first gives the clear error for an
        # unknown one (it used to surface as a NameError from inside the
        # likelihood, or as a starting-value fit failure).
        self._resolve_virtual_age_function(kijima)
        validate_lifetime_dist(dist, type(self).__name__)
        validate_renewal_censoring(data.c, type(self).__name__)
        reject_gapped_observation(data, type(self).__name__)
        # Delayed entry: as new at entry (#615).
        data = measure_from_entry(data, type(self).__name__)
        validate_renewal_times(data, dist, type(self).__name__)

        neg_ll = self.create_negll_func(data, dist, kijima=kijima)
        # result is (very!!) sensitive to the initial value of q
        dist_params0 = self._default_start(
            lambda: self._initial_dist_params(data, dist), init
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
            renewal_restoration=0.0001,
        )
        q, *dist_params = params
        self._warn_if_memoryless(dist, "q")
        model = dist.from_params(list(dist_params))
        out = self._make_model(model, q, kijima)
        # The likelihood kept as what it is built from, so the model
        # pickles (#573).
        neg_ll = Rebuilt(
            self.create_negll_func, (data, dist, kijima), built=neg_ll
        )
        self._attach_inference(out, neg_ll, [q, *dist_params], res, data)
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
        tl: "ArrayLike | None" = None,
    ) -> "RenewalModel":
        """
        Fit the generalized renewal model.

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
        kijima : str, optional
            Type of Kijima model to use, either "i" (a repair acts on the
            age gained since the last one) or "ii" (on the whole
            accumulated age). Default is "i".
        init : list, optional
            Initial parameters for the optimization algorithm.
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
            A fitted renewal model. Its restoration factor ``q`` is 0 for
            a repair as good as new, 1 for one as bad as old (minimal
            repair, the non-homogeneous Poisson process) and above 1 for
            one that leaves the system worse than before it failed (see the
            class docstring); the model prints it with its standard error
            and Wald interval, and ``repair_test()`` tests it against
            perfect and minimal repair.

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
        Restoration Factor  : 7.274462742318132e-17
        Parameters          : Wald 95% intervals
                   estimate     se  lower 95%  upper 95%
            q     7.274e-17    nan          0    0.09235
            alpha     2.399  0.287      1.898      3.033
            beta      2.754 0.6516      1.732      4.379
        Note: q = 7.274e-17 is at the edge of its range, so it has no standard
              error. Its interval is the profile-likelihood one, from the
              edge; the others' are Wald intervals with q held there.
        Repair test: consistent with perfect repair; minimal repair rejected
              (LR tests, q = 0: p = 1; q = 1: p = 0.000669)
        """
        data = handle_xicn(x, i, c, n, tl=tl)
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
        Fitted by           : given parameters (not fitted)
        Kijima Type         : i
        Restoration Factor  : 0.2
        Parameters          :
                mu: 10
            sigma: 2
        Repair test         : not available (no data)
        """
        self._resolve_virtual_age_function(kijima)
        validate_lifetime_dist(dist, type(self).__name__)
        validate_restoration(q, "q", (0, None))
        model = dist.from_params(params)
        return self._make_model(model, q, kijima)
