"""The proportional-intensity regression model now inherits the shared
``RecurrenceSimulationMixin`` instead of carrying its own stale copy, so it
gains seeding, the ``max_events`` backstop, and the data-returning simulators
-- threading the covariate vector ``Z`` through to the sampler."""

import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")

import pytest  # noqa: E402

from surpyval.recurrent import (  # noqa: E402
    CrowAMSAA,
    ProportionalIntensityHPP,
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
    a = model.mcf(xs, Z, items=300, random_state=11)
    b = model.mcf(xs, Z, items=300, random_state=11)
    c = model.mcf(xs, Z, items=300, random_state=22)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_regression_count_terminated_simulation():
    model = _fit()
    sim = model.count_terminated_simulation(
        6, Z=np.array([0.0]), items=40, random_state=0
    )
    # The trimmed simulated MCF is non-decreasing and stays within the events.
    assert np.all(np.diff(sim.mcf_hat) >= -1e-9)
    assert sim.mcf_hat.max() < 6


def test_regression_data_simulators_return_recurrent_data():
    model = _fit()
    count = model.count_terminated_simulation_data(
        5, Z=np.array([1.0]), items=12, random_state=0
    )
    assert isinstance(count, RecurrentEventData)
    assert len(count.x) == 12 * (5 + 1)

    timed = model.time_terminated_simulation_data(
        T=20, Z=np.array([1.0]), items=12, random_state=0
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
            T=1e6, Z=np.array([-5.0]), items=3, max_events=5, random_state=0
        )
    assert any("max_events" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# A missing covariate in a proportional-intensity simulation
# (#375 item 8a). ``mcf(x, Z)`` and the simulation entry points
# take one unit's covariate vector; a NaN in it raises a
# ``ValueError`` naming ``Z`` before anything is simulated.
# ---------------------------------------------------------------------------


def _data():
    rng = np.random.default_rng(2)
    xs, items, Zs = [], [], []
    for k in range(40):
        zk = [rng.normal(), rng.uniform()]
        t = np.cumsum(rng.exponential(5 * np.exp(-0.5 * zk[0]), size=4))
        xs += list(t)
        items += [k] * 4
        Zs += [zk] * 4
    return np.array(xs), np.array(Zs), np.array(items)


@pytest.fixture(scope="module", params=["NHPP", "HPP"])
def model(request):
    x, Z, i = _data()
    if request.param == "NHPP":
        return ProportionalIntensityNHPP.fit(x, Z, i, dist=CrowAMSAA)
    return ProportionalIntensityHPP.fit(x, Z, i)


CALLS = {
    "mcf": lambda m, Z: m.mcf([5.0, 10.0], Z, items=20, random_state=1),
    "time_terminated_simulation": lambda m, Z: m.time_terminated_simulation(
        10.0, Z, items=5, random_state=1
    ),
    "time_terminated_simulation_data": (
        lambda m, Z: m.time_terminated_simulation_data(
            10.0, Z, items=5, random_state=1
        )
    ),
    "count_terminated_simulation": (
        lambda m, Z: m.count_terminated_simulation(
            3, Z, items=5, random_state=1
        )
    ),
    "count_terminated_simulation_data": (
        lambda m, Z: m.count_terminated_simulation_data(
            3, Z, items=5, random_state=1
        )
    ),
}


@pytest.mark.parametrize("call", list(CALLS))
@pytest.mark.parametrize(
    "Z", [[np.nan, 0.5], [0.2, None], [[np.nan, 0.5]]], ids=str
)
def test_missing_Z_raises_naming_Z(model, call, Z):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=r"^Z has a missing \(NaN\)"):
            CALLS[call](model, Z)
    # Nothing was simulated: the old code warned that every sequence had
    # reached max_events before failing on a NaN time
    assert not [w for w in caught if "max_events" in str(w.message)]


@pytest.mark.parametrize("Z", [[0.1, 0.5, 3.0], [[0.1, 0.5], [0.2, 0.3]]])
def test_Z_of_the_wrong_shape_raises_naming_Z(model, Z):
    with pytest.raises(ValueError, match="one unit's covariate vector"):
        model.mcf([5.0, 10.0], Z, items=20, random_state=1)


def test_complete_Z_still_simulates(model):
    flat = model.mcf([5.0, 10.0], [0.1, 0.5], items=200, random_state=1)
    row = model.mcf([5.0, 10.0], [[0.1, 0.5]], items=200, random_state=1)
    np.testing.assert_array_equal(flat, row)
    # The simulated MCF tracks the closed-form cif
    cif = model.cif(np.array([5.0, 10.0]), np.array([0.1, 0.5]))
    np.testing.assert_allclose(flat, cif, rtol=0.25)
