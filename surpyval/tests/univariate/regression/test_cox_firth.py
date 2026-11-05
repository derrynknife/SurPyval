"""Firth's penalised Cox fit, ``CoxPH.fit_firth``: its own properties.
The comparison with R's ``coxphf`` is in
``surpyval/tests/reference/test_firth_cox.py``."""

import json
import pickle
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.univariate.regression.proportional_hazards.cox_firth import (
    _BreslowSums,
    _Firth,
)
from surpyval.univariate.regression.semi_parametric_regression_model import (
    SemiParametricRegressionModel,
)

from ...reference._data import fixture


def _separation():
    d = fixture("separation")
    return d["x"], np.column_stack([d["z1"], d["z2"]]), d["c"]


def _lung():
    d = fixture("lung")
    return d["time"], np.column_stack([d["age"], d["sex"]]), d["c"]


def test_ordinary_fit_names_fit_firth_on_separation():
    x, Z, c = _separation()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.CoxPH.fit(x, Z, c=c, tie_method="breslow")
    messages = [str(w.message) for w in caught]
    assert any(
        m.startswith("No finite maximum") and "CoxPH.fit_firth" in m
        for m in messages
    ), messages


def test_firth_is_finite_under_separation_without_warning():
    x, Z, c = _separation()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = sp.CoxPH.fit_firth(x, Z, c=c)
    assert model.maximum == "verified"
    assert np.all(np.isfinite(model.params))
    assert np.all(np.isfinite(model.firth.lower))
    assert np.all(np.isfinite(model.firth.upper))
    # The profile interval is asymmetric: the likelihood is flat upwards
    beta, firth = model.params[0], model.firth
    assert firth.upper[0] - beta > 2 * (beta - firth.lower[0])


def test_penalised_score_is_zero_at_the_estimate_with_strata():
    # The analytic penalised score (third derivatives of the partial
    # likelihood) against central differences of the penalised
    # log-likelihood, on stratified data with delayed entry and counts.
    rng = np.random.default_rng(11)
    N = 40
    Z = rng.normal(size=(N, 2))
    x = np.ceil(rng.exponential(5, N))
    tl = np.where(rng.uniform(size=N) < 0.3, x / 2, -np.inf)
    c = (rng.uniform(size=N) < 0.3).astype(float)
    n = rng.integers(1, 3, N).astype(float)
    strata = (np.arange(N) % 3).astype(float)
    sums = _BreslowSums(x, c.astype(int), n, tl, Z, strata)
    firth = _Firth(sums, 1e-10)
    beta = np.array([0.3, -0.2])
    _, score, _ = firth.score(beta)
    h = 1e-6
    numeric = [
        (firth.value(beta + h * e) - firth.value(beta - h * e)) / (2 * h)
        for e in np.eye(2)
    ]
    assert_allclose(score, numeric, rtol=1e-6, atol=1e-7)
    model = sp.CoxPH.fit_firth(x, Z, c=c, n=n, tl=tl, strata=strata)
    assert model.maximum == "verified"
    assert model.is_stratified
    assert np.all(model.firth.lower < model.params)
    assert np.all(model.params < model.firth.upper)


def test_one_stratum_is_the_unstratified_fit():
    x, Z, c = _separation()
    plain = sp.CoxPH.fit_firth(x, Z, c=c)
    one = sp.CoxPH.fit_firth(x, Z, c=c, strata=np.zeros(len(x)))
    assert_allclose(one.params, plain.params, atol=1e-9)
    assert_allclose(one.firth.lower, plain.firth.lower, atol=1e-8)
    assert_allclose(one.firth.upper, plain.firth.upper, atol=1e-8)


def test_rescaling_a_covariate_rescales_its_coefficient():
    # Jeffreys' prior is invariant: Z * k gives beta / k, the same tests
    x, Z, c = _separation()
    model = sp.CoxPH.fit_firth(x, Z, c=c)
    scaled = sp.CoxPH.fit_firth(x, Z * [10.0, 0.1], c=c)
    factor = np.array([10.0, 0.1])
    assert_allclose(scaled.params * factor, model.params, rtol=1e-8)
    assert_allclose(scaled.firth.lower * factor, model.firth.lower, rtol=1e-7)
    assert_allclose(scaled.firth.upper * factor, model.firth.upper, rtol=1e-7)
    assert_allclose(scaled.p_values, model.p_values, rtol=1e-7)
    assert_allclose(
        scaled.likelihood_ratio_test()["statistic"],
        model.likelihood_ratio_test()["statistic"],
        rtol=1e-9,
    )


def test_counts_are_copies_and_row_order_does_not_matter():
    x, Z, c = _separation()
    n = np.where(np.arange(len(x)) % 4 == 0, 2.0, 1.0)
    counted = sp.CoxPH.fit_firth(x, Z, c=c, n=n)
    rows = np.repeat(np.arange(len(x)), n.astype(int))
    copies = sp.CoxPH.fit_firth(x[rows], Z[rows], c=c[rows])
    assert_allclose(counted.params, copies.params, atol=1e-9)
    assert_allclose(counted.firth.upper, copies.firth.upper, atol=1e-8)
    order = np.random.default_rng(5).permutation(len(x))
    shuffled = sp.CoxPH.fit_firth(x[order], Z[order], c=c[order])
    plain = sp.CoxPH.fit_firth(x, Z, c=c)
    assert_allclose(shuffled.params, plain.params, atol=1e-10)


def test_splitting_rows_into_start_stop_intervals_changes_nothing():
    x, Z, c = _lung()
    plain = sp.CoxPH.fit_firth(x, Z, c=c, profile=False)
    cut = np.median(x)
    long = x > cut
    # Each long row as (0, cut] censored and (cut, x] with its outcome
    start = np.r_[np.full(len(x), -np.inf), np.full(long.sum(), cut)]
    stop = np.r_[np.where(long, cut, x), x[long]]
    status = np.r_[np.where(long, 1, c), c[long]]
    rows = np.r_[np.arange(len(x)), np.flatnonzero(long)]
    split = sp.CoxPH.fit_firth(
        stop, Z[rows], c=status, tl=start, profile=False
    )
    assert_allclose(split.params, plain.params, atol=1e-9)
    assert_allclose(split.covariance(), plain.covariance(), atol=1e-12)


def test_firth_is_close_to_the_ordinary_fit_with_plenty_of_events():
    x, Z, c = _lung()
    firth = sp.CoxPH.fit_firth(x, Z, c=c, profile=False)
    ordinary = sp.CoxPH.fit(x, Z, c=c, tie_method="breslow")
    # Well within a tenth of a standard error (the bias it removes is
    # of order 1 / events)
    gap = np.abs(firth.params - ordinary.params) / ordinary.standard_errors()
    assert np.all(gap < 0.1), gap
    # and shrunk towards 0
    assert np.all(np.abs(firth.params) < np.abs(ordinary.params))


def test_round_trips_through_to_dict_and_pickle():
    x, Z, c = _separation()
    model = sp.CoxPH.fit_firth(x, Z, c=c)
    text = json.dumps(model.to_dict(), allow_nan=False)
    back = SemiParametricRegressionModel.from_dict(json.loads(text))
    for key in ("lower", "upper", "p_values", "chisq", "wald_p"):
        assert_allclose(getattr(back.firth, key), getattr(model.firth, key))
    assert back.firth.profile and back.firth.alpha_ci == 0.05
    assert back.likelihood_ratio_test() == model.likelihood_ratio_test()
    assert back.summary().equals(model.summary())
    assert_allclose(back.sf([5, 10], [1, 0.2]), model.sf([5, 10], [1, 0.2]))
    assert repr(back) == repr(model)
    again = pickle.loads(pickle.dumps(model))
    assert_allclose(again.firth.upper, model.firth.upper)


def test_printout_and_summary_name_the_intervals():
    x, Z, c = _separation()
    model = sp.CoxPH.fit_firth(x, Z, c=c, alpha_ci=0.1)
    text = repr(model)
    assert "Firth (Jeffreys prior)" in text
    assert "profile penalised-likelihood 90% intervals" in text
    assert "coef lower 90%" in model.summary().columns
    with pytest.raises(ValueError, match="alpha_ci=0.05"):
        model.summary(alpha_ci=0.05)
    wald = sp.CoxPH.fit_firth(x, Z, c=c, profile=False)
    assert "Wald 95% intervals" in repr(wald)
    assert_allclose(wald.p_values, wald.firth.wald_p)


def test_refuses_a_column_with_no_information():
    x, Z, c = _separation()
    with pytest.raises(ValueError, match=r"column\(s\) \[2\].*Remove them"):
        sp.CoxPH.fit_firth(x, np.column_stack([Z, np.ones(len(x))]), c=c)
    with pytest.raises(ValueError, match="alpha_ci"):
        sp.CoxPH.fit_firth(x, Z, c=c, alpha_ci=1.5)
