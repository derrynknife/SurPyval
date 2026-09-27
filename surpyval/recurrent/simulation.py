import warnings
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.typing import ArrayLike

from surpyval.recurrent.inference import require_data
from surpyval.recurrent.nonparametric import NonParametricCounting
from surpyval.utils.rng import as_generator

STALLED_WARNING = (
    "Some sequences produced a near-zero interarrival time (< tol) before "
    "reaching T: their events pile up towards a finite time (a possible "
    "asymptote, such as a G1 process with q < 0), so they were ended early "
    "at their last event."
)
MAX_EVENTS_WARNING = (
    "Some sequences reached max_events ({}) before T; increase max_events or "
    "check the model parameters."
)


class SimulatedSequences:
    """
    The rows produced by :func:`simulate_sequences`, grouped by sequence and
    in time order within each: ``x`` the event (or end-of-observation)
    times, ``i`` the 0-based sequence index, ``c`` 0 for an event and 1 for
    the end-of-observation row at a sequence's ``close``. ``stalled`` and
    ``hit_max_events`` say whether any observed-to-``close`` sequence was
    ended early (see :func:`simulate_sequences`).
    """

    def __init__(
        self,
        x: np.ndarray,
        i: np.ndarray,
        c: np.ndarray,
        stalled: bool,
        hit_max_events: bool,
    ) -> None:
        self.x = x
        self.i = i
        self.c = c
        self.stalled = stalled
        self.hit_max_events = hit_max_events

    def xicn(self) -> dict:
        """The rows as an ``xicn`` dict with 1-based item ids."""
        return {
            "x": self.x,
            "i": self.i + 1,
            "c": self.c,
            "n": np.ones(self.x.size, dtype=int),
        }


def _open_uniforms(rng: np.random.Generator, size: int) -> np.ndarray:
    """``size`` uniforms on the open interval (0, 1): an exact 0 (a
    2**-53 chance per draw) would make ``-log(u)`` infinite."""
    u = rng.random(size)
    zero = u == 0.0
    while zero.any():
        u[zero] = rng.random(int(zero.sum()))
        zero = u == 0.0
    return u


def simulate_sequences(
    step: Callable,
    items: int,
    rng: np.random.Generator,
    close: "np.ndarray | None" = None,
    count: "np.ndarray | None" = None,
    tol: float = 1e-8,
    max_events: int = 10_000,
) -> SimulatedSequences:
    """
    Simulate ``items`` recurrent-event sequences together.

    Every sequence still running gets one event per round: the round draws
    one uniform per running sequence and asks ``step(idx, u)`` (from a
    model's ``_new_batch_sampler``) for their interarrival times at once, so
    a round costs a few array operations whatever the number of sequences.
    Because every running sequence has had the same number of events at the
    start of a round, a sampler can keep its history as one array per round.

    Each sequence is observed one of two ways:

    * to a fixed time, ``close[k]`` finite: events are drawn until one falls
      after ``close[k]``, and the sequence ends with an end-of-observation
      row (``c = 1``) at ``close[k]``. It ends early, at its last event, if
      an interarrival time falls below ``tol`` (the events are piling up
      towards a finite time) or it reaches ``max_events``; ``stalled`` and
      ``hit_max_events`` record that it happened.
    * to a fixed number of events, ``close[k]`` infinite (or ``close`` not
      given): exactly ``count[k]`` events are drawn, all observed.

    The uniforms are assigned round by round, so the draws for a given seed
    depend on how many sequences are simulated together.
    """
    close_arr = (
        np.full(items, np.inf)
        if close is None
        else np.asarray(close, dtype=float)
    )
    timed = np.isfinite(close_arr)
    count_arr = (
        np.zeros(items, dtype=int)
        if count is None
        else np.asarray(count, dtype=int)
    )
    if not np.all(timed | (count_arr >= 0)):
        raise ValueError("every sequence needs a close time or a count")

    running = np.zeros(items)
    n_events = np.zeros(items, dtype=int)
    active = np.flatnonzero(timed | (count_arr > 0))
    stalled = hit_max = False
    xs, ids, cs = [], [], []
    while active.size:
        gap = np.asarray(step(active, _open_uniforms(rng, active.size)))
        t = running[active] + gap
        running[active] = t
        n_events[active] += 1
        is_timed = timed[active]
        past = is_timed & (t > close_arr[active])
        stall = is_timed & ~past & (gap < tol)
        maxed = is_timed & ~past & ~stall & (n_events[active] >= max_events)
        counted = ~is_timed & (n_events[active] >= count_arr[active])
        xs.append(np.where(past, close_arr[active], t))
        ids.append(active)
        cs.append(past.astype(int))
        stalled = stalled or bool(stall.any())
        hit_max = hit_max or bool(maxed.any())
        active = active[~(past | stall | maxed | counted)]

    if not xs:
        empty = np.zeros(0)
        return SimulatedSequences(
            empty, empty.astype(int), empty.astype(int), stalled, hit_max
        )
    i = np.concatenate(ids)
    # Rounds are in time order, so a stable sort by sequence keeps each
    # sequence's rows in time order.
    order = np.argsort(i, kind="stable")
    return SimulatedSequences(
        np.concatenate(xs)[order],
        i[order],
        np.concatenate(cs)[order],
        stalled,
        hit_max,
    )


def _fit_mcf(xicn: dict) -> Any:
    """The nonparametric MCF of simulated sequences, without its variance
    (the simulations return the MCF alone)."""
    from surpyval.utils.recurrent_utils import handle_xicn

    # The singleton fitter instance (mypy sees the decorated class).
    fitter: Any = NonParametricCounting
    return fitter._point_estimate(handle_xicn(**xicn, as_recurrent_data=True))


class RecurrenceSimulationMixin:
    """
    Shared simulation machinery for fitted recurrent-event models.

    Every sequence is simulated by :func:`simulate_sequences`, which
    advances all of them together, one event per round. A model supplies
    the per-round draw through ``_new_batch_sampler``. The intensity models
    share the conditional inverse-CIF draw defined here; the only
    per-family difference is the extra arguments threaded into
    ``cif``/``inv_cif``: unconditional models pass none,
    proportional-intensity models pass the covariate vector (declared via
    ``_cif_args``). The renewal models supply their own sampler.
    """

    if TYPE_CHECKING:
        # The host model supplies these; declared rather than defined so a
        # model that forgets one still gets the AttributeError naming it.
        data: Any
        dist: Any
        params: Any

        def cif(self, x: Any, *args: Any) -> Any: ...
        def inv_cif(self, x: Any, *args: Any) -> Any: ...

    def _cif_args(self) -> tuple:
        """
        Extra positional arguments threaded into ``cif``/``inv_cif`` for each
        simulated sequence. Empty for unconditional models; the covariate
        vector for proportional-intensity models (which override this).
        """
        return ()

    def _new_batch_sampler(self, n: int) -> Callable:
        """
        Return ``step(idx, u) -> gaps`` for ``n`` sequences simulated
        together: it draws the next interarrival time of each sequence in
        ``idx`` from the uniforms ``u``, keeping every sequence's state
        (here the time of its last event) itself. See
        :func:`simulate_sequences` for how it is called.

        The next event is sampled by inverting the cumulative intensity
        conditional on the time of the previous event: a uniform ``u`` maps
        to the next event time ``inv_cif(cif(x_prev) - log(u))``. Any
        per-family arguments (e.g. the covariate vector) come from
        :meth:`_cif_args`.
        """
        cif_args = self._cif_args()
        x_prev = np.zeros(n)

        def step(idx: np.ndarray, u: np.ndarray) -> np.ndarray:
            prev = x_prev[idx]
            # Added on the cumulative-intensity scale. This used to go
            # through u * exp(-cif(x_prev)), which underflows to 0 once
            # the expected count passes about 745, so every later event
            # landed at inv_cif(inf).
            target = np.asarray(self.cif(prev, *cif_args), dtype=float)
            target = target - np.log(u)
            new = np.asarray(self.inv_cif(target, *cif_args), dtype=float)
            new = np.broadcast_to(new, prev.shape)
            x_prev[idx] = new
            return new - prev

        return step

    def _postprocess_simulated_model(self, model: Any) -> Any:
        """
        Adjust the fitted ``NonParametricCounting`` model in place before it is
        returned. A CoxLewis (log-linear) intensity has a non-zero baseline
        rate ``exp(alpha)`` at time zero that the simulated event counts do not
        carry, so it is added back to the MCF here. Other intensity models --
        and renewal models, which carry no top-level ``dist`` -- are left
        unchanged.
        """
        # The former Cox-Lewis mcf_hat correction was dead code (it
        # compared against the wrong name string) and would have been
        # wrong had it fired: cif(0) = 0 and the inverse-CIF sampler is
        # exact, so the simulated MCF needs no baseline offset (#288).
        return model

    def _simulate_count_xicn(
        self, events: int, items: int, seed: "int | None"
    ) -> dict:
        """
        Simulate ``items`` count-terminated sequences and return the raw event
        data as an ``xicn`` dict (``events + 1`` exact events per sequence).
        """
        run = simulate_sequences(
            self._new_batch_sampler(items),
            items,
            as_generator(seed),
            count=np.full(items, events + 1),
        )
        return run.xicn()

    def _simulate_time_xicn(
        self,
        T: float,
        items: int,
        tol: float,
        max_events: int,
        seed: "int | None",
    ) -> dict:
        """
        Simulate ``items`` time-terminated sequences and return the raw event
        data as an ``xicn`` dict. Each sequence ends in a right-censored (c=1)
        row at ``T``, or an observed (c=0) row at its last event if it stalls
        or hits ``max_events``. Warns in the latter cases.
        """
        run = simulate_sequences(
            self._new_batch_sampler(items),
            items,
            as_generator(seed),
            close=np.full(items, float(T)),
            tol=tol,
            max_events=max_events,
        )
        if run.stalled:
            warnings.warn(STALLED_WARNING)
        if run.hit_max_events:
            warnings.warn(MAX_EVENTS_WARNING.format(max_events))
        return run.xicn()

    def count_terminated_simulation_data(
        self, events: int, items: int = 1, seed: "int | None" = None
    ) -> Any:
        """
        Simulate count-terminated recurrence data and return the raw events.

        Unlike :meth:`count_terminated_simulation` (which returns the fitted
        ``NonParametricCounting`` MCF), this returns the simulated event data
        itself, ready to be refitted or inspected via ``.x``/``.i``/``.c``/
        ``.n``.

        Parameters
        ----------

        events: int
            Each sequence is simulated to its ``events + 1``-th event (see
            the notes).
        items: int, optional
            Number of items (or sequences) to simulate. Default is 1.
        seed: int or numpy.random.Generator, optional
            Seed for a reproducible simulation.

        Returns
        -------

        RecurrentEventData
            The simulated recurrence data in xicn format.

        Notes
        -----

        Count termination is a failure-terminated (Type II) scheme: each item
        is observed until its ``events + 1``-th event, so its observation
        window is the random time of that last event and every event is exact
        (``c = 0``). Parametric fits handle this correctly -- the
        interarrival/intensity likelihood ends at the last observed event and
        the MLE is consistent. The nonparametric MCF, however, is only reliable
        up to roughly ``events`` recurrences: beyond that the at-risk set is
        depleted and the curve is biased (which is why
        :meth:`count_terminated_simulation` trims to ``mcf_hat < events``). For
        a fixed-window observation scheme, use
        :meth:`time_terminated_simulation_data`, which right-censors each item
        at ``T``.
        """
        from surpyval.utils.recurrent_utils import handle_xicn

        xicn = self._simulate_count_xicn(events, items, seed)
        return handle_xicn(**xicn)

    def time_terminated_simulation_data(
        self,
        T: float,
        items: int = 1,
        tol: float = 1e-8,
        max_events: int = 10_000,
        seed: "int | None" = None,
    ) -> Any:
        """
        Simulate time-terminated recurrence data and return the raw events.

        Unlike :meth:`time_terminated_simulation` (which returns the fitted
        ``NonParametricCounting`` MCF), this returns the simulated event data
        itself, ready to be refitted or inspected via ``.x``/``.i``/``.c``/
        ``.n``. Each sequence is right-censored at ``T``.

        Parameters
        ----------

        T: float
            Time termination value.
        items: int, optional
            Number of items (or sequences) to simulate. Default is 1.
        tol: float, optional
            Interarrival times below this value end the sequence early.
            Default is 1e-8.
        max_events: int, optional
            Hard per-sequence event cap that guarantees termination.
            Default is 10000.
        seed: int or numpy.random.Generator, optional
            Seed for a reproducible simulation.

        Returns
        -------

        RecurrentEventData
            The simulated recurrence data in xicn format.
        """
        from surpyval.utils.recurrent_utils import handle_xicn

        xicn = self._simulate_time_xicn(T, items, tol, max_events, seed)
        return handle_xicn(**xicn)

    def count_terminated_simulation(
        self, events: int, items: int = 1, seed: "int | None" = None
    ) -> Any:
        """
        Simulate count-terminated recurrence data based on the fitted model.

        Parameters
        ----------

        events: int
            Each sequence is simulated to its ``events + 1``-th event, and
            the returned MCF is kept only where it is below ``events``
            (beyond that the items are dropping out of observation).
        items: int, optional
            Number of items (or sequences) to simulate. Default is 1.
        seed: int or numpy.random.Generator, optional
            Seed for a reproducible simulation. When ``None`` (default) the
            numpy global RNG is used.

        Returns
        -------

        NonParametricCounting
            An NonParametricCounting model built from the simulated data.
        """
        xicn = self._simulate_count_xicn(events, items, seed)

        model = _fit_mcf(xicn)
        self._postprocess_simulated_model(model)
        mask = model.mcf_hat < events
        model.x = model.x[mask]
        model.mcf_hat = model.mcf_hat[mask]
        model.var = None
        return model

    def time_terminated_simulation(
        self,
        T: float,
        items: int = 1,
        tol: float = 1e-8,
        max_events: int = 10_000,
        seed: "int | None" = None,
    ) -> Any:
        """
        Simulate time-terminated recurrence data based on the fitted model.

        Parameters
        ----------

        T: float
            Time termination value.
        items: int, optional
            Number of items (or sequences) to simulate. Default is 1.
        tol: float, optional
            Interarrival times below this value end the sequence early; a tiny
            increment indicates the cumulative time has stalled below T (a
            possible asymptote). Default is 1e-8.
        max_events: int, optional
            Hard cap on the number of events simulated per sequence. This is
            the backstop that guarantees termination for sequences whose
            cumulative time cannot reach T. Default is 10000.
        seed: int or numpy.random.Generator, optional
            Seed for a reproducible simulation. When ``None`` (default) the
            numpy global RNG is used.

        Returns
        -------

        NonParametricCounting
            An NonParametricCounting model built from the simulated data.

        Warnings
        --------

        A sequence is ended early at its last event, which is kept as an
        observed event (no censoring row at ``T``), if an interarrival time
        falls below ``tol`` or it reaches ``max_events`` before T. A warning
        is raised in either case.
        """
        xicn = self._simulate_time_xicn(T, items, tol, max_events, seed)

        model = _fit_mcf(xicn)
        self._postprocess_simulated_model(model)
        model.var = None
        return model

    def mcf(
        self, x: ArrayLike, items: int = 1000, seed: "int | None" = None
    ) -> Any:
        """
        Estimate the mean cumulative function (MCF) at ``x``.

        These models have no closed-form cumulative intensity, so the MCF is
        estimated by simulating ``items`` time-terminated sequences out to
        ``max(x)`` and reading off the nonparametric MCF. Increase ``items``
        for a smoother estimate; pass ``seed`` for reproducibility.

        Parameters
        ----------

        x: array_like
            Times at which to evaluate the MCF.
        items: int, optional
            Number of sequences to simulate. Default is 1000.
        seed: int or numpy.random.Generator, optional
            Seed for a reproducible estimate.

        Returns
        -------

        numpy.ndarray
            The estimated MCF at each value of ``x``.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        np_model = self.time_terminated_simulation(
            float(x.max()), items=items, seed=seed
        )
        return np_model.mcf(x)

    def plot(
        self, ax: Any = None, items: int = 1000, seed: "int | None" = None
    ) -> Any:
        """
        Overlay the simulated MCF on the empirical MCF of the fitted data.

        Parameters
        ----------

        ax: matplotlib axes, optional
            Axes to draw on. A new one is created if not provided.
        items: int, optional
            Number of sequences to simulate for the model MCF. Default is 1000.
        seed: int or numpy.random.Generator, optional
            Seed for a reproducible model curve.

        Returns
        -------

        matplotlib axes
            The axes with the plot.
        """
        require_data(self, "plot")
        x, r, d = self.data.to_xrd()
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()

        x_plot = np.linspace(0, float(self.data.x.max()), 200)
        ax.step(x, (d / r).cumsum(), color="r", where="post")
        ax.plot(x_plot, self.mcf(x_plot, items=items, seed=seed), color="b")
        return ax
