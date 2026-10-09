"""Cox predictions: the baseline before the first event, and pairing of
times with covariate rows."""

import warnings

import numpy as np
import pytest

from surpyval import CoxPH
from surpyval.tests._helpers import weibull_ph_data


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
    assert model.Hf(model.x[0], [[0.0]]) == pytest.approx(model.H0[0])


def test_times_pair_with_their_own_covariate_row(model):
    x = np.array([3.0, 1.0, 2.0])
    Z = np.array([[0.0], [2.0], [-1.0]])
    paired = model.Hf(x, Z)
    one_by_one = [model.Hf(t, [z]) for t, z in zip(x, Z)]
    assert np.allclose(paired, one_by_one)
    assert np.allclose(
        model.sf(x, Z), [model.sf(t, [z]) for t, z in zip(x, Z)]
    )
    assert np.allclose(
        model.hf(x, Z), [model.hf(t, [z]) for t, z in zip(x, Z)]
    )


def test_one_covariate_row_is_used_for_every_time_in_the_order_given(
    model,
):
    x = np.array([3.0, 1.0, 0.1, 2.0])
    out = model.Hf(x, [[0.5]])
    assert np.allclose(out, [model.Hf(t, [[0.5]]) for t in x])
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


def test_585_ph_random_solves_tiny_multiplier_draws_together(monkeypatch):
    # Draws whose quantile rounds to inf (a tiny hazard multiplier) were
    # solved one brentq each: GammaPH.random(2000) at z = -60 took 20 s.
    # They are now solved together, to the same tolerance.
    import warnings

    from surpyval import GammaPH, LogNormalPH

    params = (2.0, 0.5, 1.0)
    Hf = GammaPH.dist.Hf
    calls = []

    def counted(*args):
        calls.append(1)
        return Hf(*args)

    monkeypatch.setattr(GammaPH.dist, "Hf", counted)
    x, _ = GammaPH.random(500, [[-30.0]], *params, random_state=1)
    assert len(calls) < 300
    monkeypatch.undo()
    u = np.random.default_rng(1).uniform(size=500)
    h = -np.log(u) / np.exp(-30.0)
    assert np.isfinite(x).all()
    np.testing.assert_allclose(Hf(x, *params[:2]), h, rtol=1e-10)
    # a time past the largest float is inf, as qf gives (the search raised)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x, _ = LogNormalPH.random(5, [[-30.0]], 2.0, 1.0, 1.0, random_state=1)
    assert np.isposinf(x).all()


# ---------------------------------------------------------------------------
# A scalar covariate.
# ---------------------------------------------------------------------------


def test_cox_accepts_a_scalar_covariate():
    x, Z = weibull_ph_data()
    model = CoxPH.fit(x, Z)
    for fn in ("sf", "hf", "Hf"):
        np.testing.assert_allclose(
            getattr(model, fn)([3.0], 0.5), getattr(model, fn)([3.0], [0.5])
        )
    np.testing.assert_allclose(model.phi(0.5), np.exp(0.5 * model.beta[0]))


# ---------------------------------------------------------------------------
# The density far in the upper tail (#714).
# ---------------------------------------------------------------------------


def test_cox_density_is_zero_where_the_hazard_step_overflows(model):
    # A risk score far beyond the data's (exp(0.8 * 2000) overflows) makes
    # the hazard step and the cumulative hazard inf after the first event
    # time. The density hf * sf was inf * 0, nan with a raw RuntimeWarning
    # (seen on separated data, whose coefficients run off); it is 0 (the
    # step is at most H, and H e^{-H} -> 0).
    query = np.array([0.1, model.x[0], model.x[5], model.x[-1]])
    Z = np.array([[0.0], [2000.0], [2000.0], [0.5]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isposinf(model.hf(query, Z)[1:3]).all()
        df = model.df(query, Z)
        grid = model.df(query, Z, grid=True)
    np.testing.assert_array_equal(df[1:3], 0.0)
    assert np.isfinite(df).all() and np.isfinite(grid).all()
    np.testing.assert_array_equal(np.diagonal(grid), df)
    # elsewhere it is hf * sf, as it was
    keep = [0, 3]
    np.testing.assert_allclose(
        df[keep], model.hf(query, Z)[keep] * model.sf(query, Z)[keep]
    )
    assert df[3] > 0


@pytest.mark.parametrize("name", ["WeibullPH", "WeibullAFT"])
def test_parametric_density_is_zero_where_the_hazard_overflows(name):
    # A Weibull hazard (shape 5) overflows to inf at 1e80 where e^{-H}
    # underflows to 0: the density was inf * 0, nan with a raw
    # RuntimeWarning (from the generated data of the property tests).
    import surpyval

    fitter = getattr(surpyval, name)
    x = np.array([1.0, 1e80])
    Z = np.zeros((2, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = fitter.df(x, Z, 1.0, 5.0, 0.3)
    assert df[1] == 0.0
    assert df[0] == pytest.approx(5 * np.exp(-1.0))


def test_parametric_ph_risk_score_overflow_keeps_the_baseline_0_and_inf():
    # exp(beta'Z) overflows to inf at beta'Z = 800 (a coefficient running
    # off): against a baseline of exactly 0 (x = 0, a Weibull shape above
    # 1) the hazard and H are 0, not inf * 0; and a risk score that
    # underflowed to 0 against an infinite baseline (x = inf) leaves them
    # inf (#714).
    from surpyval import WeibullPH

    x = np.array([0.0, 1.0, np.inf])
    Z = np.array([[800.0], [800.0], [-800.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        H = WeibullPH.Hf(x, Z, 1.0, 2.0, 1.0)
        h = WeibullPH.hf(x, Z, 1.0, 2.0, 1.0)
        sf = WeibullPH.sf(x, Z, 1.0, 2.0, 1.0)
    np.testing.assert_array_equal(H, [0.0, np.inf, np.inf])
    np.testing.assert_array_equal(h, [0.0, np.inf, np.inf])
    np.testing.assert_array_equal(sf, [1.0, 0.0, 0.0])
