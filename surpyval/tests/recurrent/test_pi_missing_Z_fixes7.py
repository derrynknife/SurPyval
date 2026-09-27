"""
A missing covariate in a proportional-intensity simulation (#375 item 8a).

``mcf(x, Z)`` and the simulation entry points take one unit's covariate
vector. With a NaN in it the intensity was NaN, so every sequence ran to
``max_events`` (with a warning suggesting a larger ``max_events``) and the
call then failed with "Variable 'x' cannot contain NaN values", naming a
*time*. Under the package's missing-value rule an input that describes one
unit raises a ``ValueError`` naming it, before anything is simulated.
"""

import warnings

import numpy as np
import pytest

from surpyval.recurrent import (
    CrowAMSAA,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)


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
    "mcf": lambda m, Z: m.mcf([5.0, 10.0], Z, items=20, seed=1),
    "time_terminated_simulation": lambda m, Z: m.time_terminated_simulation(
        10.0, Z, items=5, seed=1
    ),
    "time_terminated_simulation_data": (
        lambda m, Z: m.time_terminated_simulation_data(
            10.0, Z, items=5, seed=1
        )
    ),
    "count_terminated_simulation": (
        lambda m, Z: m.count_terminated_simulation(3, Z, items=5, seed=1)
    ),
    "count_terminated_simulation_data": (
        lambda m, Z: m.count_terminated_simulation_data(3, Z, items=5, seed=1)
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
        model.mcf([5.0, 10.0], Z, items=20, seed=1)


def test_complete_Z_still_simulates(model):
    flat = model.mcf([5.0, 10.0], [0.1, 0.5], items=200, seed=1)
    row = model.mcf([5.0, 10.0], [[0.1, 0.5]], items=200, seed=1)
    np.testing.assert_array_equal(flat, row)
    # The simulated MCF tracks the closed-form cif
    cif = model.cif(np.array([5.0, 10.0]), np.array([0.1, 0.5]))
    np.testing.assert_allclose(flat, cif, rtol=0.25)
