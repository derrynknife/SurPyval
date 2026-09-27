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


@pytest.mark.parametrize(
    "name", ["WeibullPH", "WeibullAFT", "WeibullPO", "WeibullAH"]
)
def test_parametric_regression_accepts_a_one_dimensional_Z(name):
    import surpyval

    fitter = getattr(surpyval, name)
    rng = np.random.default_rng(0)
    z = rng.normal(size=50)
    x = rng.weibull(1.5, 50) * 10 * np.exp(-0.5 * z)
    one_d = fitter.fit(x=x, Z=z)
    column = fitter.fit(x=x, Z=z.reshape(-1, 1))
    assert np.allclose(one_d.params, column.params)


def test_ph_random_is_finite_for_a_tiny_hazard_multiplier():
    from surpyval import WeibullPH

    params = [10.0, 2.0, 1.0]  # alpha, beta, coefficient
    for z in (0.0, -30.0, -60.0):
        np.random.seed(1)
        u = np.random.uniform(0, 1, 5)
        np.random.seed(1)
        x, _ = WeibullPH.random(5, [z], *params)
        expected = 10.0 * (-np.log(u) / np.exp(z)) ** 0.5
        assert np.all(np.isfinite(x))
        assert np.allclose(x, expected, rtol=1e-9)
