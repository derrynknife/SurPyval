"""Parametric competing risks.

A parametric distribution is fitted to each cause's cause-specific hazard
(with the other causes' events treated as right-censored), and the smooth
cumulative-incidence functions are assembled from them. The tests check that
the per-cause parameters are recovered from latent-minimum data, that the
parametric CIFs agree with the nonparametric estimator, that the cause CIFs
sum to the all-cause failure probability ``1 - S``, and that the input
validation and the per-cause distribution mapping behave.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval import Exponential, LogNormal, Weibull
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    ParametricCompetingRisks,
)


def _simulate(N, seed, cens=60.0):
    """Two-cause latent-minimum data: cause 1 ~ Weibull(30, 2.0), cause 2 ~
    Weibull(45, 1.2), with administrative right-censoring at ``cens``."""
    rng = np.random.default_rng(seed)
    t1 = Weibull.random(N, 30.0, 2.0)
    t2 = Weibull.random(N, 45.0, 1.2)
    t = np.minimum(t1, t2)
    cause = np.where(t1 < t2, 1, 2)
    c = np.where(t > cens, 1, 0)
    x = np.where(c == 1, cens, t)
    e = np.array(
        [None if ci == 1 else ei for ci, ei in zip(c, cause)], dtype=object
    )
    del rng
    return x, e, c


def test_recovers_per_cause_parameters():
    x, e, c = _simulate(6000, 0)
    model = ParametricCompetingRisks.fit(x, e, c=c)
    assert model.causes == [1, 2]
    assert np.allclose(model.models[1].params, [30.0, 2.0], rtol=0.1)
    assert np.allclose(model.models[2].params, [45.0, 1.2], rtol=0.12)


def test_cif_agrees_with_nonparametric():
    x, e, c = _simulate(6000, 1)
    par = ParametricCompetingRisks.fit(x, e, c=c)
    nonp = CompetingRisks.fit(x, e, c=c)
    grid = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    for k in (1, 2):
        assert np.allclose(par.cif(grid, k), nonp.cif(grid, k), atol=0.03)


def test_cifs_sum_to_all_cause_incidence():
    x, e, c = _simulate(4000, 2)
    model = ParametricCompetingRisks.fit(x, e, c=c)
    grid = np.linspace(1.0, 55.0, 25)
    total = sum(model.cif(grid, k) for k in model.causes)
    assert np.allclose(total, model.ff(grid), atol=1e-3)


def test_probability_of_cause_sums_to_one():
    x, e, c = _simulate(4000, 3)
    model = ParametricCompetingRisks.fit(x, e, c=c)
    probs = [model.probability_of_cause(k) for k in model.causes]
    assert np.isclose(sum(probs), 1.0, atol=1e-3)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_all_cause_survival_is_product_of_cause_survivals():
    x, e, c = _simulate(3000, 4)
    model = ParametricCompetingRisks.fit(x, e, c=c)
    grid = np.array([5.0, 15.0, 25.0, 40.0])
    prod = model.models[1].sf(grid) * model.models[2].sf(grid)
    assert np.allclose(model.sf(grid), prod)
    assert np.allclose(model.ff(grid), 1.0 - prod)


def test_iif_is_subdistribution_density():
    x, e, c = _simulate(2000, 5)
    model = ParametricCompetingRisks.fit(x, e, c=c)
    grid = np.array([8.0, 20.0, 35.0])
    # f_1^sub = f_1 * S_2
    expected = model.models[1].df(grid) * model.models[2].sf(grid)
    assert np.allclose(model.iif(grid, 1), expected)


def test_hazards_sum_to_all_cause_hazard():
    x, e, c = _simulate(2000, 6)
    model = ParametricCompetingRisks.fit(x, e, c=c)
    grid = np.array([10.0, 20.0, 30.0])
    assert np.allclose(model.hf(grid), model.hf(grid, 1) + model.hf(grid, 2))
    assert np.allclose(model.Hf(grid), model.Hf(grid, 1) + model.Hf(grid, 2))


def test_per_cause_distribution_mapping():
    x, e, c = _simulate(3000, 7)
    model = ParametricCompetingRisks.fit(
        x, e, c=c, dist={1: Weibull, 2: Exponential}
    )
    assert model.models[1].dist.name == "Weibull"
    assert model.models[2].dist.name == "Exponential"


def test_cif_scalar_and_vector():
    x, e, c = _simulate(1500, 8)
    model = ParametricCompetingRisks.fit(x, e, c=c)
    scalar = model.cif(25.0, 1)
    assert np.ndim(scalar) == 0
    vec = model.cif(np.array([25.0]), 1)
    assert np.ndim(vec) == 1
    assert np.isclose(scalar, vec[0])


def test_random_returns_time_and_cause():
    x, e, c = _simulate(2000, 9)
    model = ParametricCompetingRisks.fit(x, e, c=c)
    draws = model.random(500, random_state=0)
    assert draws.dtype.names == ("x", "e")
    assert np.all(draws["x"] > 0)
    assert set(np.unique(draws["e"])).issubset(set(model.causes))


def test_goodness_of_fit_sums_over_causes():
    x, e, c = _simulate(2000, 10)
    model = ParametricCompetingRisks.fit(x, e, c=c)
    assert np.isclose(
        model.neg_ll(),
        model.models[1].neg_ll() + model.models[2].neg_ll(),
    )
    assert np.isfinite(model.aic())
    assert np.isfinite(model.bic())


def test_fit_from_df():
    x, e, c = _simulate(2000, 11)
    df = pd.DataFrame({"time": x, "cause": e, "cens": c})
    model = ParametricCompetingRisks.fit_from_df(
        df, x_col="time", e_col="cause", c_col="cens"
    )
    assert model.causes == [1, 2]
    assert np.allclose(model.models[1].params, [30.0, 2.0], rtol=0.15)


def test_three_causes():
    rng = np.random.default_rng(12)
    N = 5000
    t1 = Weibull.random(N, 20.0, 1.5)
    t2 = Weibull.random(N, 30.0, 2.0)
    t3 = LogNormal.random(N, 3.5, 0.4)
    stack = np.column_stack([t1, t2, t3])
    idx = np.argmin(stack, axis=1)
    x = stack[np.arange(N), idx]
    e = np.array([idx_i + 1 for idx_i in idx], dtype=object)
    del rng
    model = ParametricCompetingRisks.fit(
        x, e, dist={1: Weibull, 2: Weibull, 3: LogNormal}
    )
    assert model.causes == [1, 2, 3]
    probs = [model.probability_of_cause(k) for k in model.causes]
    assert np.isclose(sum(probs), 1.0, atol=1e-3)


# --- validation -----------------------------------------------------------


def test_rejects_event_on_censored_row():
    # A censored row (c = 1) must carry event None.
    with pytest.raises(ValueError, match="None"):
        ParametricCompetingRisks.fit(
            x=[1.0, 2.0, 3.0], e=[1, 2, 1], c=[0, 0, 1]
        )


def test_rejects_none_on_observed_row():
    with pytest.raises(ValueError, match="None"):
        ParametricCompetingRisks.fit(
            x=[1.0, 2.0, 3.0], e=[1, None, 1], c=[0, 0, 0]
        )


def test_rejects_left_censoring():
    with pytest.raises(ValueError, match="Left or interval"):
        ParametricCompetingRisks.fit(
            x=[1.0, 2.0, 3.0], e=[1, 2, None], c=[0, 0, -1]
        )


def test_rejects_no_events():
    with pytest.raises(ValueError, match="No observed events"):
        ParametricCompetingRisks.fit(x=[1.0, 2.0], e=[None, None], c=[1, 1])


def test_unknown_cause_raises():
    x, e, c = _simulate(1000, 13)
    model = ParametricCompetingRisks.fit(x, e, c=c)
    with pytest.raises(ValueError, match="Unknown cause"):
        model.cif(10.0, 99)


# --- from_fitted: assemble from pre-fitted per-cause models ----------------


def _fit_cause(x, e, c, cause, dist):
    """Fit ``dist`` to one cause's cause-specific hazard (its events observed,
    every other event and every censored unit right-censored)."""
    c_k = np.where((np.asarray(e, dtype=object) == cause) & (c == 0), 0, 1)
    return dist.fit(x=x, c=c_k)


def test_from_fitted_matches_fit():
    # Fitting each cause by hand and assembling with from_fitted must give
    # the same model as fit() doing it internally.
    x, e, c = _simulate(5000, 20)
    m1 = _fit_cause(x, e, c, 1, Weibull)
    m2 = _fit_cause(x, e, c, 2, Weibull)
    assembled = ParametricCompetingRisks.from_fitted({1: m1, 2: m2})
    fitted = ParametricCompetingRisks.fit(x, e, c=c)
    assert assembled.causes == [1, 2]
    grid = np.array([10.0, 25.0, 40.0])
    for k in (1, 2):
        assert np.allclose(assembled.cif(grid, k), fitted.cif(grid, k))


def test_from_fitted_heterogeneous_families():
    # A different distribution family per cause.
    x, e, c = _simulate(5000, 21)
    m1 = _fit_cause(x, e, c, 1, Weibull)
    m2 = _fit_cause(x, e, c, 2, LogNormal)
    model = ParametricCompetingRisks.from_fitted({1: m1, 2: m2})
    assert model.models[1].dist.name == "Weibull"
    assert model.models[2].dist.name == "LogNormal"
    # still a coherent competing-risks model: CIFs sum to 1 - S
    grid = np.linspace(1.0, 55.0, 20)
    total = sum(model.cif(grid, k) for k in model.causes)
    assert np.allclose(total, model.ff(grid), atol=1e-3)


def test_from_fitted_agrees_with_nonparametric():
    x, e, c = _simulate(6000, 22)
    m1 = _fit_cause(x, e, c, 1, Weibull)
    m2 = _fit_cause(x, e, c, 2, LogNormal)
    model = ParametricCompetingRisks.from_fitted({1: m1, 2: m2})
    nonp = CompetingRisks.fit(x, e, c=c)
    grid = np.array([15.0, 25.0, 35.0, 45.0])
    for k in (1, 2):
        assert np.allclose(model.cif(grid, k), nonp.cif(grid, k), atol=0.03)


def test_from_fitted_sequence_gives_positional_causes():
    m0 = Weibull.from_params([30.0, 2.0])
    m1 = LogNormal.from_params([3.6, 0.4])
    model = ParametricCompetingRisks.from_fitted([m0, m1])
    assert model.causes == [0, 1]
    assert model.models[0].dist.name == "Weibull"


def test_from_fitted_all_causes_cured_gives_never_fail_units():
    # Every cause carries a cure fraction, so all-cause survival stays
    # positive: some units never fail, and the cause probabilities sum to
    # 1 - S(inf) < 1 rather than to one.
    a = Weibull.from_params([15.0, 2.5], p=0.6)  # 60% ever fail from a
    b = Weibull.from_params([25.0, 1.5], p=0.5)  # 50% ever fail from b
    model = ParametricCompetingRisks.from_fitted([a, b])
    probs = [model.probability_of_cause(k) for k in model.causes]
    # S(inf) = 0.4 * 0.5 = 0.2, so incidence sums to ~0.8
    assert np.isclose(sum(probs), 0.8, atol=1e-2)
    assert np.isclose(model.ff(1e6), 0.8, atol=1e-3)
    draws = model.random(4000, random_state=0)
    never = ~np.isfinite(draws["x"])
    assert np.isclose(never.mean(), 0.2, atol=0.03)
    # never-fail units carry no cause
    assert all(e is None for e in draws["e"][never])
    assert all(e in model.causes for e in draws["e"][~never])


def test_from_fitted_rejects_empty():
    with pytest.raises(ValueError, match="At least one cause"):
        ParametricCompetingRisks.from_fitted({})


def test_from_fitted_rejects_non_model():
    class NotAModel:
        pass

    with pytest.raises(ValueError, match="does not look like"):
        ParametricCompetingRisks.from_fitted({1: NotAModel()})
