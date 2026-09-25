"""Cox predictions: the baseline before the first event, and pairing of
times with covariate rows."""

import numpy as np
import pytest

from surpyval import CoxPH


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 1))
    x = rng.exponential(1 / np.exp(0.8 * Z[:, 0])) + 0.5
    return CoxPH.fit(x=x, Z=Z)


def test_nothing_has_happened_before_the_first_event(model):
    before = np.array([0.0, 0.1, model.x[0] - 1e-9])
    assert np.all(model.Hf(before, [[0.3]]) == 0.0)
    assert np.all(model.hf(before, [[0.3]]) == 0.0)
    assert np.all(model.sf(before, [[0.3]]) == 1.0)
    assert np.all(model.ff(before, [[0.3]]) == 0.0)
    # and at the first event the baseline has made its first jump
    assert model.Hf(model.x[0], [[0.0]])[0] == pytest.approx(model.H0[0])


def test_times_pair_with_their_own_covariate_row(model):
    x = np.array([3.0, 1.0, 2.0])
    Z = np.array([[0.0], [2.0], [-1.0]])
    paired = model.Hf(x, Z)
    one_by_one = [model.Hf(t, [z])[0] for t, z in zip(x, Z)]
    assert np.allclose(paired, one_by_one)
    assert np.allclose(
        model.sf(x, Z), [model.sf(t, [z])[0] for t, z in zip(x, Z)]
    )
    assert np.allclose(
        model.hf(x, Z), [model.hf(t, [z])[0] for t, z in zip(x, Z)]
    )


def test_one_covariate_row_is_used_for_every_time_in_the_order_given(
    model,
):
    x = np.array([3.0, 1.0, 0.1, 2.0])
    out = model.Hf(x, [[0.5]])
    assert np.allclose(out, [model.Hf(t, [[0.5]])[0] for t in x])
    assert out[2] == 0.0
    # a step function: non-decreasing in time
    order = np.argsort(x)
    assert np.all(np.diff(out[order]) >= 0)
