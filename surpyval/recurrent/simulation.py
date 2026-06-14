import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import uniform

from surpyval.recurrent.nonparametric import NonParametricCounting

STALLED_WARNING = (
    "Some sequences produced a near-zero interarrival time (< tol) before "
    "reaching T, indicating a possible asymptote; they were terminated early "
    "at their last event."
)
MAX_EVENTS_WARNING = (
    "Some sequences reached max_events ({}) before T; increase max_events or "
    "check the model parameters."
)


class RecurrenceSimulationMixin:
    """
    Shared simulation machinery for fitted recurrent-event models.

    Subclasses provide the per-event sampling logic by implementing
    ``_new_sequence_sampler``, which returns a callable mapping a uniform
    random number to the next interarrival time and which carries its own
    per-sequence state. Subclasses may optionally override
    ``_postprocess_simulated_model`` to adjust the fitted
    ``NonParametricCounting`` model (e.g. the CoxLewis offset) before it is
    returned.
    """

    def _set_simulation_seed(self, seed):
        # ``None`` defers to numpy's global RNG (so ``np.random.seed`` still
        # controls the stream); an int/Generator gives a reproducible stream
        # that is independent of global state.
        self._sim_random_state = (
            None if seed is None else np.random.default_rng(seed)
        )

    def initialize_simulation(self):
        self.us = uniform.rvs(
            size=100_000,
            random_state=getattr(self, "_sim_random_state", None),
        ).tolist()

    def clear_simulation(self):
        del self.us

    def get_uniform_random_number(self):
        try:
            return self.us.pop()
        except IndexError:
            self.initialize_simulation()
            return self.us.pop()

    def _new_sequence_sampler(self):
        """
        Return a callable ``sample(ui) -> xi`` that draws the next interarrival
        time from a uniform random number, maintaining any per-sequence state
        internally. A fresh sampler is requested for each simulated sequence.
        """
        raise NotImplementedError

    def _postprocess_simulated_model(self, model):
        """
        Hook to adjust the fitted ``NonParametricCounting`` model in place
        before it is returned. Default is a no-op.
        """
        return model

    def _simulate_count_xicn(self, events, items, seed):
        """
        Simulate ``items`` count-terminated sequences and return the raw event
        data as an ``xicn`` dict (``events + 1`` exact events per sequence).
        """
        self._set_simulation_seed(seed)
        self.initialize_simulation()

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            sample = self._new_sequence_sampler()
            for j in range(0, events + 1):
                ui = self.get_uniform_random_number()
                running += sample(ui)
                xicn["x"].append(running)
                xicn["i"].append(i + 1)
                xicn["c"].append(0)
                xicn["n"].append(1)

        self.clear_simulation()
        return xicn

    def _simulate_time_xicn(self, T, items, tol, max_events, seed):
        """
        Simulate ``items`` time-terminated sequences and return the raw event
        data as an ``xicn`` dict. Each sequence ends in a right-censored (c=1)
        row at ``T``, or an observed (c=0) row at its last event if it stalls
        or hits ``max_events``. Warns in the latter cases.
        """
        self._set_simulation_seed(seed)
        self.initialize_simulation()
        stalled = False
        hit_max_events = False

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            n_events = 0
            sample = self._new_sequence_sampler()
            while True:
                ui = self.get_uniform_random_number()
                xi = sample(ui)
                running += xi
                n_events += 1
                xicn["i"].append(i + 1)
                xicn["n"].append(1)
                if running > T:
                    xicn["x"].append(T)
                    xicn["c"].append(1)
                    break
                elif xi < tol:
                    stalled = True
                    xicn["x"].append(running)
                    xicn["c"].append(0)
                    break
                elif n_events >= max_events:
                    hit_max_events = True
                    xicn["x"].append(running)
                    xicn["c"].append(0)
                    break
                else:
                    xicn["x"].append(running)
                    xicn["c"].append(0)

        self.clear_simulation()

        if stalled:
            warnings.warn(STALLED_WARNING)
        if hit_max_events:
            warnings.warn(MAX_EVENTS_WARNING.format(max_events))

        return xicn

    def count_terminated_simulation_data(self, events, items=1, seed=None):
        """
        Simulate count-terminated recurrence data and return the raw events.

        Unlike :meth:`count_terminated_simulation` (which returns the fitted
        ``NonParametricCounting`` MCF), this returns the simulated event data
        itself, ready to be refitted or inspected via ``.x``/``.i``/``.c``/
        ``.n``.

        Parameters
        ----------

        events: int
            Number of events to simulate per sequence.
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
        return handle_xicn(as_recurrent_data=True, **xicn)

    def time_terminated_simulation_data(
        self, T, items=1, tol=1e-8, max_events=10_000, seed=None
    ):
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
        return handle_xicn(as_recurrent_data=True, **xicn)

    def count_terminated_simulation(self, events, items=1, seed=None):
        """
        Simulate count-terminated recurrence data based on the fitted model.

        Parameters
        ----------

        events: int
            Number of events to simulate.
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

        model = NonParametricCounting.fit(**xicn)
        self._postprocess_simulated_model(model)
        mask = model.mcf_hat < events
        model.x = model.x[mask]
        model.mcf_hat = model.mcf_hat[mask]
        model.var = None
        return model

    def time_terminated_simulation(
        self, T, items=1, tol=1e-8, max_events=10_000, seed=None
    ):
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

        A sequence is terminated early and right-censored at its last event if
        an interarrival time falls below ``tol`` or it reaches ``max_events``
        before T. A warning is raised in either case.
        """
        xicn = self._simulate_time_xicn(T, items, tol, max_events, seed)

        model = NonParametricCounting.fit(**xicn)
        self._postprocess_simulated_model(model)
        model.var = None
        return model

    def mcf(self, x, items=1000, seed=None):
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

    def plot(self, ax=None, items=1000, seed=None):
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
        if not hasattr(self, "data"):
            raise ValueError(
                "plot requires a model fitted from data; fit_from_parameters "
                "models carry no data to compare against."
            )
        x, r, d = self.data.to_xrd()
        if ax is None:
            ax = plt.gcf().gca()

        x_plot = np.linspace(0, float(self.data.x.max()), 200)
        ax.step(x, (d / r).cumsum(), color="r", where="post")
        ax.plot(x_plot, self.mcf(x_plot, items=items, seed=seed), color="b")
        return ax
