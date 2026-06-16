"""The proportional-intensity regression model now inherits the shared
``RecurrenceSimulationMixin`` instead of carrying its own stale copy, so it
gains seeding, the ``max_events`` backstop, and the data-returning simulators
-- threading the covariate vector ``Z`` through to the sampler."""

import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")

from surpyval.recurrent import (  # noqa: E402
    CrowAMSAA,
    ProportionalIntensityNHPP,
)
from surpyval.utils.recurrent_event_data import (  # noqa: E402
    RecurrentEventData,
)


def _fit():
    x = [9, 14, 18, 20, 7, 12, 16, 19, 20, 5, 9, 13, 16, 18, 20]
    i = [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    c = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    Z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)
    return ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=CrowAMSAA)


def test_regression_simulation_seed_is_reproducible():
    model = _fit()
    Z = np.array([1.0])
    xs = [5.0, 10.0, 15.0, 20.0]
    a = model.mcf(xs, Z, items=300, seed=11)
    b = model.mcf(xs, Z, items=300, seed=11)
    c = model.mcf(xs, Z, items=300, seed=22)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_regression_count_terminated_simulation():
    model = _fit()
    sim = model.count_terminated_simulation(
        6, Z=np.array([0.0]), items=40, seed=0
    )
    # The trimmed simulated MCF is non-decreasing and stays within the events.
    assert np.all(np.diff(sim.mcf_hat) >= -1e-9)
    assert sim.mcf_hat.max() < 6


def test_regression_data_simulators_return_recurrent_data():
    model = _fit()
    count = model.count_terminated_simulation_data(
        5, Z=np.array([1.0]), items=12, seed=0
    )
    assert isinstance(count, RecurrentEventData)
    assert len(count.x) == 12 * (5 + 1)

    timed = model.time_terminated_simulation_data(
        T=20, Z=np.array([1.0]), items=12, seed=0
    )
    assert isinstance(timed, RecurrentEventData)
    assert (timed.c == 1).any()


def test_regression_time_terminated_max_events_backstop():
    # The covariate that makes the intensity decay can leave a sequence unable
    # to reach a huge T; the inherited max_events cap must still terminate it.
    model = _fit()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.time_terminated_simulation(
            T=1e6, Z=np.array([-5.0]), items=3, max_events=5, seed=0
        )
    assert any("max_events" in str(w.message) for w in caught)
