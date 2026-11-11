"""Parametric competing-risks regression: one parametric regression per
cause, combined into cumulative incidences of time and covariates.

The reductions (each cause's model is its fitter's own fit with the other
causes censored; with no covariate effect the model is
``ParametricCompetingRisks``), a case with a closed form in exact
arithmetic, recovery of simulated parameters, the identities between the
functions, and the refusals.
"""

import json
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import (
    ExponentialPH,
    LogNormalAFT,
    ParametricCompetingRisksRegression,
    WeibullAFT,
    WeibullPH,
)
from surpyval.univariate.competing_risks import ParametricCompetingRisks


def _weibull_ph_causes(N, seed, censor=None):
    """Two causes with Weibull proportional hazards in two covariates:
    cause 1 ``alpha = 10, beta = 2``, coefficients ``(0.6, -0.3)``; cause
    2 ``alpha = 15, beta = 1.2``, coefficients ``(-0.4, 0.5)``; optional
    uniform censoring on ``(0, censor)``."""
    rng = np.random.default_rng(seed)
    Z = np.column_stack([rng.binomial(1, 0.5, N), rng.normal(0, 1, N)])
    truth = {
        1: (10.0, 2.0, np.array([0.6, -0.3])),
        2: (15.0, 1.2, np.array([-0.4, 0.5])),
    }
    latent = np.column_stack(
        [
            alpha * (rng.exponential(size=N) / np.exp(Z @ coef)) ** (1 / beta)
            for alpha, beta, coef in truth.values()
        ]
    )
    x = latent.min(axis=1)
    e = np.array([1, 2], dtype=object)[latent.argmin(axis=1)]
    if censor is not None:
        cens = rng.uniform(0, censor, N)
        e[cens < x] = None
        x = np.minimum(x, cens)
    return x, Z, e, truth


@pytest.fixture(scope="module")
def fitted():
    x, Z, e, truth = _weibull_ph_causes(400, 3, censor=30.0)
    return ParametricCompetingRisksRegression.fit(x, Z, e), (x, Z, e)


# -- reductions ------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        WeibullPH,
        {1: WeibullAFT, 2: LogNormalAFT},
        {1: ExponentialPH, 2: WeibullPH},
    ],
    ids=["one-fitter", "aft-per-cause", "ph-per-cause"],
)
def test_each_cause_is_its_fitters_own_fit(model):
    x, Z, e, _ = _weibull_ph_causes(300, 5, censor=30.0)
    both = ParametricCompetingRisksRegression.fit(x, Z, e, model=model)
    assert both.causes == [1, 2]
    for k in (1, 2):
        fitter = model[k] if isinstance(model, dict) else model
        c_k = np.where(e == k, 0, 1)
        alone = fitter.fit(x, Z, c=c_k)
        np.testing.assert_array_equal(both.models[k].params, alone.params)
        assert both.models[k].neg_ll() == alone.neg_ll()
    assert both.neg_ll() == sum(both.models[k].neg_ll() for k in (1, 2))


def test_entry_times_go_to_every_causes_fit():
    x, Z, e, _ = _weibull_ph_causes(400, 6, censor=30.0)
    rng = np.random.default_rng(0)
    tl = rng.uniform(0, 3, x.size)
    keep = x > tl
    x, Z, e, tl = x[keep], Z[keep], e[keep], tl[keep]
    both = ParametricCompetingRisksRegression.fit(x, Z, e, tl=tl)
    for k in (1, 2):
        alone = WeibullPH.fit(x, Z, c=np.where(e == k, 0, 1), tl=tl)
        np.testing.assert_array_equal(both.models[k].params, alone.params)
    untruncated = ParametricCompetingRisksRegression.fit(x, Z, e)
    assert not np.allclose(both.params, untruncated.params, rtol=1e-3)


def test_no_covariate_effect_is_the_model_without_covariates():
    """A covariate column of zeros has no coefficient to estimate (it is
    aliased, with a warning), and the model is then
    ``ParametricCompetingRisks`` with the same distributions."""
    x, _, e, _ = _weibull_ph_causes(300, 8, censor=30.0)
    with pytest.warns(UserWarning, match="cannot be estimated"):
        reg = ParametricCompetingRisksRegression.fit(
            x, np.zeros((x.size, 1)), e
        )
    plain = ParametricCompetingRisks.fit(x, e, dist=sp.Weibull)
    t = np.array([1.0, 5.0, 12.0, 40.0, np.inf])
    for k in (1, 2):
        np.testing.assert_allclose(
            reg.models[k].params[:2], plain.models[k].params, rtol=1e-5
        )
        np.testing.assert_allclose(
            reg.cif(t, [0.0], k), plain.cif(t, k), rtol=1e-5
        )
    np.testing.assert_allclose(reg.sf(t, [0.0]), plain.sf(t), rtol=1e-5)
    np.testing.assert_allclose(reg.neg_ll(), plain.neg_ll(), rtol=1e-9)


# -- a closed form ---------------------------------------------------------


#: Group 0: times 2, 3, 5, 6 (total 16), failures a, b, a and one censored:
#: rates 2/16 and 1/16. Group 1: times 1, 2, 2, 4, 3 (total 12), failures
#: a, a, b, a and one censored: rates 3/12 and 1/12.
_HAND_RATES = {
    ("a", 0): Fraction(2, 16),
    ("b", 0): Fraction(1, 16),
    ("a", 1): Fraction(3, 12),
    ("b", 1): Fraction(1, 12),
}


def _hand_model():
    x = [2, 3, 5, 6, 1, 2, 2, 4, 3]
    z = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    e = ["a", "b", "a", None, "a", "a", "b", "a", None]
    return ParametricCompetingRisksRegression.fit(
        x, np.array(z, dtype=float)[:, None], e, model=ExponentialPH
    )


@pytest.mark.xfail(
    strict=True,
    reason="the ExponentialPH fits of these nine rows reach the groups' "
    "rates to 1.9e-6 (relative), outside the 1e-6 set for this check "
    "before it was run: cause a's rate in group 0 is 0.12500023 against "
    "2/16",
)
def test_exponential_causes_with_one_binary_covariate_by_hand():
    """Constant hazards and one binary covariate: each cause's model
    (a rate and a coefficient) is saturated, so its fitted rate in each
    group is that group's failures of the cause over its time on test
    (``_HAND_RATES``), and the cumulative incidence is ``(rate_k /
    rate)(1 - exp(-rate t))``."""
    model = _hand_model()
    rate = _HAND_RATES
    t = np.array([1.0, 4.0, 10.0])
    for g in (0, 1):
        total = rate["a", g] + rate["b", g]
        for k in ("a", "b"):
            np.testing.assert_allclose(
                model.hf(t, [g], event=k), float(rate[k, g]), rtol=1e-6
            )
            expected = float(rate[k, g] / total) * -np.expm1(-float(total) * t)
            np.testing.assert_allclose(
                model.cif(t, [g], k), expected, rtol=1e-6
            )
            np.testing.assert_allclose(
                model.probability_of_cause([g], k),
                float(rate[k, g] / total),
                rtol=1e-6,
            )
        np.testing.assert_allclose(
            model.sf(t, [g]), np.exp(-float(total) * t), rtol=1e-6
        )
        np.testing.assert_allclose(
            model.mean([g]), 1 / float(total), rtol=1e-6
        )
        np.testing.assert_allclose(
            model.qf(0.5, [g]), np.log(2) / float(total), rtol=1e-6
        )
    # The coefficient of each cause is its log rate ratio between groups
    np.testing.assert_allclose(
        model.models["a"].params[-1],
        np.log(float(Fraction(3, 12) / Fraction(2, 16))),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        model.models["b"].params[-1],
        np.log(float(Fraction(1, 12) / Fraction(1, 16))),
        rtol=1e-5,
    )


def test_constant_hazards_give_the_closed_forms_of_the_fitted_rates():
    """With constant hazards every function of the model is a formula of
    the causes' fitted rates ``r_k`` at the row, ``r`` their sum: the
    incidence ``(r_k / r)(1 - exp(-r t))``, the eventual share ``r_k /
    r``, the survival ``exp(-r t)``, the mean ``1 / r`` and the median
    ``log(2) / r``. (The rates themselves are the hand case above.)"""
    model = _hand_model()
    t = np.array([1.0, 4.0, 10.0])
    for g in (0, 1):
        r = {k: float(model.hf(1.0, [g], event=k)) for k in ("a", "b")}
        total = r["a"] + r["b"]
        # the fitted rates are the hand rates to the fits' accuracy
        for k in ("a", "b"):
            assert abs(r[k] / float(_HAND_RATES[k, g]) - 1) < 1e-4
            np.testing.assert_allclose(
                model.cif(t, [g], k),
                r[k] / total * -np.expm1(-total * t),
                rtol=1e-9,
            )
            np.testing.assert_allclose(
                model.probability_of_cause([g], k), r[k] / total, rtol=1e-9
            )
        np.testing.assert_allclose(
            model.sf(t, [g]), np.exp(-total * t), rtol=1e-12
        )
        np.testing.assert_allclose(model.mean([g]), 1 / total, rtol=1e-9)
        np.testing.assert_allclose(
            model.qf(0.5, [g]), np.log(2) / total, rtol=1e-9
        )


def test_cif_bound_is_the_delta_method_in_the_closed_form():
    """In the exponential model the cumulative incidence is a formula of
    the four parameters, so its delta-method variance is too: the Wald
    bound on the logit scale from the analytic gradient is ``cb``'s."""
    from scipy.special import ndtri

    x, Z, e, _ = _weibull_ph_causes(300, 4, censor=30.0)
    Z = Z[:, :1]
    model = ParametricCompetingRisksRegression.fit(
        x, Z, e, model=ExponentialPH
    )
    lam1, b1 = model.models[1].params
    lam2, b2 = model.models[2].params
    t, z = 6.0, 1.0
    # ExponentialPH: hazard lambda * exp(b z)
    r1, r2 = lam1 * np.exp(b1 * z), lam2 * np.exp(b2 * z)
    np.testing.assert_allclose(model.hf(t, [z], event=1), r1, rtol=1e-9)
    r = r1 + r2
    F = r1 / r * -np.expm1(-r * t)
    np.testing.assert_allclose(model.cif(t, [z], 1), F, rtol=1e-9)
    # dF/dr1 and dF/dr2, then the chain rule to (lambda, b) of each cause
    grow = -np.expm1(-r * t)
    dF_dr1 = r2 / r**2 * grow + r1 / r * t * np.exp(-r * t)
    dF_dr2 = -r1 / r**2 * grow + r1 / r * t * np.exp(-r * t)
    grad = np.array(
        [
            dF_dr1 * r1 / lam1,
            dF_dr1 * r1 * z,
            dF_dr2 * r2 / lam2,
            dF_dr2 * r2 * z,
        ]
    )
    var = grad @ model.covariance() @ grad
    half = ndtri(0.975) * np.sqrt(var) / (F * (1 - F))
    logit = np.log(F / (1 - F))
    expected = 1 / (1 + np.exp(-(logit + np.array([-half, half]))))
    np.testing.assert_allclose(model.cb(t, [z], 1), expected, rtol=1e-6)
    # (a one-sided bound at alpha is the two-sided one's end at 2 alpha)
    np.testing.assert_allclose(
        model.cb(t, [z], 1, bound="lower", alpha_ci=0.025),
        expected[0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        model.cb(t, [z], 1, bound="upper", alpha_ci=0.025),
        expected[1],
        rtol=1e-6,
    )
    # The all-cause cumulative hazard: r t, on the log scale
    grad_H = np.array([r1 * t / lam1, r1 * t * z, r2 * t / lam2, r2 * t * z])
    sd_log = np.sqrt(grad_H @ model.covariance() @ grad_H) / (r * t)
    H = r * t * np.exp(np.array([-1, 1]) * ndtri(0.975) * sd_log)
    np.testing.assert_allclose(model.cb(t, [z], on="Hf"), H, rtol=1e-6)
    np.testing.assert_allclose(
        model.cb(t, [z], on="sf"), np.exp(-H[::-1]), rtol=1e-6
    )
    np.testing.assert_allclose(
        model.cb(t, [z], on="ff"), -np.expm1(-H), rtol=1e-6
    )


# -- recovery --------------------------------------------------------------


def test_recovers_the_parameters_it_was_simulated_from():
    """2000 units, about a fifth censored: every parameter of both
    causes within three of its standard errors of the truth."""
    x, Z, e, truth = _weibull_ph_causes(2000, 20261111, censor=40.0)
    model = ParametricCompetingRisksRegression.fit(x, Z, e)
    assert model.maximum == "verified"
    for k, (alpha, beta, coef) in truth.items():
        fit = model.models[k]
        error = fit.params - np.r_[alpha, beta, coef]
        assert np.all(np.abs(error) < 3 * fit.standard_errors()), (k, error)
    # and the cumulative incidence at a covariate row: inside its bounds
    row = np.array([1.0, 0.5])
    t = np.array([3.0, 8.0, 20.0])
    exact = _true_cif(truth, t, row, 1)
    bounds = model.cb(t, row, 1, alpha_ci=0.01)
    assert np.all((bounds[:, 0] < exact) & (exact < bounds[:, 1])), bounds


def _true_cif(truth, t, row, cause):
    """The cumulative incidence of the simulating model, by quadrature of
    ``h_k S`` on a fine grid (trapezium rule on 200001 points)."""
    out = []
    for end in t:
        u = np.linspace(0, end, 200001)[1:]
        H = sum(
            (u / alpha) ** beta * np.exp(row @ coef)
            for alpha, beta, coef in truth.values()
        )
        alpha, beta, coef = truth[cause]
        h = beta / alpha * (u / alpha) ** (beta - 1) * np.exp(row @ coef)
        y = h * np.exp(-H)
        out.append(
            np.sum((y[1:] + y[:-1]) / 2) * (u[1] - u[0]) + y[0] * u[0] / 2
        )
    return np.array(out)


# -- identities between the functions --------------------------------------


def test_the_functions_agree_with_each_other(fitted):
    model, _ = fitted
    t = np.array([0.5, 2.0, 6.0, 15.0, 40.0])
    row = np.array([1.0, -0.5])
    total = model.cif(t, row, 1) + model.cif(t, row, 2)
    np.testing.assert_allclose(total, model.ff(t, row), rtol=1e-9)
    np.testing.assert_allclose(model.cif(t, row), model.ff(t, row))
    np.testing.assert_allclose(
        model.sf(t, row), np.exp(-model.Hf(t, row)), rtol=1e-12
    )
    np.testing.assert_allclose(
        model.Hf(t, row), model.Hf(t, row, 1) + model.Hf(t, row, 2)
    )
    np.testing.assert_allclose(
        model.df(t, row), model.hf(t, row) * model.sf(t, row)
    )
    np.testing.assert_allclose(
        model.iif(t, row, 1) + model.iif(t, row, 2), model.df(t, row)
    )
    # the incidence is the integral of iif, where it is still rising
    # (the last time is test_incidence_slope_at_every_time's)
    h = 1e-5
    early = t[:4]
    slope = (model.cif(early + h, row, 1) - model.cif(early - h, row, 1)) / (
        2 * h
    )
    np.testing.assert_allclose(slope, model.iif(early, row, 1), rtol=1e-5)
    # the eventual shares
    np.testing.assert_allclose(
        model.probability_of_cause(row, 1)
        + model.probability_of_cause(row, 2),
        1.0,
        rtol=1e-9,
    )
    # quantile and mean of the whole unit
    p = np.array([0.01, 0.1, 0.5, 0.9, 0.999])
    np.testing.assert_allclose(model.ff(model.qf(p, row), row), p, rtol=1e-9)
    assert model.qf(0.0, row) == 0.0
    assert model.qf(1.0, row) == np.inf
    grid = np.linspace(0, 400, 400001)
    area = np.sum(model.sf(grid, row)[:-1] + model.sf(grid, row)[1:]) / 2
    np.testing.assert_allclose(
        model.mean(row), area * (grid[1] - grid[0]), rtol=1e-6
    )


@pytest.mark.xfail(
    strict=True,
    reason="at t = 40 the incidence has stopped moving in double "
    "precision: its central difference is 0.0 against an iif of 1.8e-15, "
    "outside the relative 1e-5 set for this check before it was run (the "
    "four earlier times agree, in test_the_functions_agree_with_each_other)",
)
def test_incidence_slope_at_every_time(fitted):
    model, _ = fitted
    t = np.array([0.5, 2.0, 6.0, 15.0, 40.0])
    row = np.array([1.0, -0.5])
    h = 1e-5
    slope = (model.cif(t + h, row, 1) - model.cif(t - h, row, 1)) / (2 * h)
    np.testing.assert_allclose(slope, model.iif(t, row, 1), rtol=1e-5)


@pytest.mark.xfail(
    strict=True,
    reason="the incidence is an adaptive quadrature over the rows given "
    "together, so a row evaluated with others or alone differs in the "
    "last bit: 0.02945935164618703 against 0.029459351646187015 (they "
    "agree to the quadrature's accuracy, in "
    "test_rows_are_paired_with_times_and_missing_is_nan)",
)
def test_rows_evaluated_together_or_apart_to_the_last_bit(fitted):
    model, _ = fitted
    rows = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
    t = np.array([2.0, 5.0, 9.0])
    paired = model.cif(t, rows, 1)
    for i in range(3):
        assert paired[i] == model.cif(t[i], rows[i], 1)


def test_rows_are_paired_with_times_and_missing_is_nan(fitted):
    model, _ = fitted
    rows = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
    t = np.array([2.0, 5.0, 9.0])
    # (to the quadrature's accuracy; to the last bit is
    # test_rows_evaluated_together_or_apart_to_the_last_bit's)
    paired = model.cif(t, rows, 1)
    for i in range(3):
        np.testing.assert_allclose(
            paired[i], model.cif(t[i], rows[i], 1), rtol=1e-9
        )
    np.testing.assert_allclose(
        model.cif(5.0, rows, 1),
        [model.cif(5.0, r, 1) for r in rows],
        rtol=1e-9,
    )
    assert np.isscalar(model.cif(5.0, rows[0], 1))
    assert model.cif(t.reshape(3, 1), rows[0], 1).shape == (3, 1)
    assert model.cif([], rows[0], 1).shape == (0,)
    assert model.cb(t, rows, 1).shape == (3, 2)
    assert model.cb(5.0, rows[0], 1).shape == (2,)
    assert model.cb(t, rows, 1, bound="upper").shape == (3,)
    with pytest.raises(ValueError, match="covariate rows"):
        model.cif(t, rows[:2], 1)
    out = model.cif([2.0, np.nan], rows[0], 1)
    assert np.isfinite(out[0]) and np.isnan(out[1])
    out = model.cif(5.0, [[0.0, 0.0], [np.nan, 1.0]], 1)
    assert np.isfinite(out[0]) and np.isnan(out[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(model.qf(1.5, rows[0]))
    with pytest.warns(UserWarning, match="outside"):
        model.qf(1.5, rows[0])


# -- simulation ------------------------------------------------------------


def test_random_draws_the_model(fitted):
    """20000 units at one row: the share failing from cause 1 is the
    eventual probability of cause 1, and the mean time the model's mean
    (four Monte Carlo standard errors)."""
    model, _ = fitted
    row = np.array([[1.0, 0.5]])
    draws, drawn_at = model.random(20000, row, random_state=11)
    assert draws.shape == (20000,) and drawn_at.shape == (20000, 2)
    share = np.mean(draws["e"] == 1)
    p = model.probability_of_cause(row, 1)
    assert abs(share - p) < 4 * np.sqrt(p * (1 - p) / 20000)
    assert abs(draws["x"].mean() - model.mean(row)) < (
        4 * draws["x"].std() / np.sqrt(20000)
    )
    again, _ = model.random(20000, row, random_state=11)
    np.testing.assert_array_equal(again["x"], draws["x"])
    two, at = model.random(3, [[0.0, 0.0], [1.0, 1.0]], random_state=0)
    np.testing.assert_array_equal(
        at, np.repeat([[0.0, 0.0], [1.0, 1.0]], 3, 0)
    )
    assert set(two["e"]) <= {1, 2}


# -- parameters, tables and criteria ---------------------------------------


def test_parameters_covariance_and_criteria(fitted):
    model, (x, Z, e) = fitted
    assert model.parameter_names == [
        f"{k}: {name}"
        for k in (1, 2)
        for name in ("alpha", "beta", "coef_0", "coef_1")
    ]
    assert model.covariance_names == model.parameter_names
    np.testing.assert_array_equal(
        model.params, np.r_[model.models[1].params, model.models[2].params]
    )
    cov = model.covariance()
    assert cov.shape == (8, 8)
    np.testing.assert_array_equal(cov[:4, :4], model.models[1].covariance())
    np.testing.assert_array_equal(cov[4:, 4:], model.models[2].covariance())
    assert not cov[:4, 4:].any()
    np.testing.assert_allclose(model.standard_errors(), np.sqrt(np.diag(cov)))
    table = model.summary()
    assert list(table.index.names)[0] == "cause"
    assert len(table) == 8
    np.testing.assert_allclose(table["coef"].to_numpy(), model.params)
    events = float(np.sum([v is not None for v in e]))
    assert model.log_likelihood == -model.neg_ll()
    assert model.aic() == pytest.approx(16 + 2 * model.neg_ll())
    assert model.bic() == pytest.approx(
        2 * model.neg_ll() + 8 * np.log(events)
    )
    assert model.aic_c() == pytest.approx(
        model.aic() + (2 * 64 + 16) / (events - 9)
    )
    assert model.maximum == "verified"
    moved = model.with_params(model.params * 1.1)
    np.testing.assert_allclose(moved.params, model.params * 1.1)
    assert moved.cif(5.0, [0.0, 0.0], 1) != model.cif(5.0, [0.0, 0.0], 1)
    with pytest.raises(ValueError, match="params must have 8 values"):
        model.with_params([1.0, 2.0])


def test_net_bounds_are_the_causes_own(fitted):
    model, _ = fitted
    t, row = np.array([3.0, 9.0]), np.array([1.0, 0.0])
    for on in ("sf", "ff", "Hf"):
        np.testing.assert_array_equal(
            model.cb(t, row, event=2, on=on),
            model.models[2].cb(t, row, on=on),
        )
    lo, hi = model.cb(5.0, row, on="sf")
    assert lo < model.sf(5.0, row) < hi
    with pytest.raises(ValueError, match="Wald bounds only"):
        model.cb(t, row, 1, method="lr")
    with pytest.raises(ValueError, match="Wald bounds only"):
        model.cb(t, row, on="sf", method="bootstrap")
    with pytest.raises(ValueError):
        model.cb(t, row, on="cif")


# -- DataFrames, formulas and saving ---------------------------------------


def test_fit_from_df_formula_and_round_trip():
    x, Z, e, _ = _weibull_ph_causes(300, 9, censor=30.0)
    df = pd.DataFrame(
        {
            "time": x,
            "cause": e,
            "grp": np.where(Z[:, 0] == 1, "treated", "control"),
            "z": Z[:, 1],
        }
    )
    by_cols = ParametricCompetingRisksRegression.fit_from_df(
        df.assign(g=Z[:, 0]), "time", "cause", Z_cols=["g", "z"]
    )
    arrays = ParametricCompetingRisksRegression.fit(x, Z, e)
    np.testing.assert_allclose(by_cols.params, arrays.params, rtol=1e-9)
    assert by_cols.feature_names == ["g", "z"]
    assert by_cols.parameter_names[2] == "1: g"
    model = ParametricCompetingRisksRegression.fit_from_df(
        df, "time", "cause", formula="grp + z"
    )
    assert model.feature_names == ["grp[T.treated]", "z"]
    np.testing.assert_allclose(model.params, arrays.params, rtol=1e-6)
    new = pd.DataFrame({"z": [0.5, 0.5], "grp": ["control", "treated"]})
    by_frame = model.cif([5.0, 5.0], new, 1)
    np.testing.assert_allclose(
        by_frame,
        arrays.cif([5.0, 5.0], [[0.0, 0.5], [1.0, 0.5]], 1),
        rtol=1e-6,
    )
    restored = sp.from_dict(json.loads(json.dumps(model.to_dict())))
    assert isinstance(restored, ParametricCompetingRisksRegression)
    np.testing.assert_array_equal(restored.cif([5.0, 5.0], new, 1), by_frame)
    np.testing.assert_allclose(
        restored.cb([5.0, 5.0], new, 1),
        model.cb([5.0, 5.0], new, 1),
        rtol=1e-9,
    )
    np.testing.assert_array_equal(
        restored.summary().to_numpy(), model.summary().to_numpy()
    )
    assert repr(restored) == repr(model)
    # a DataFrame Z in fit names the coefficients too
    named = ParametricCompetingRisksRegression.fit(
        x, pd.DataFrame(Z, columns=["g", "z"]), e
    )
    assert named.feature_names == ["g", "z"]
    np.testing.assert_array_equal(named.params, arrays.params)


def test_counts_equal_repeated_rows():
    x, Z, e, _ = _weibull_ph_causes(150, 12, censor=30.0)
    x = np.round(x, 1)
    rng = np.random.default_rng(1)
    n = rng.integers(1, 4, x.size)
    counted = ParametricCompetingRisksRegression.fit(x, Z, e, n=n)
    repeated = ParametricCompetingRisksRegression.fit(
        np.repeat(x, n), np.repeat(Z, n, axis=0), np.repeat(e, n)
    )
    np.testing.assert_allclose(counted.params, repeated.params, rtol=1e-5)
    np.testing.assert_allclose(counted.neg_ll(), repeated.neg_ll(), rtol=1e-9)


# -- refusals --------------------------------------------------------------


def test_refusals_say_what_to_do():
    x, Z, e, _ = _weibull_ph_causes(100, 2, censor=30.0)
    fit = ParametricCompetingRisksRegression.fit
    with pytest.raises(ValueError, match="parametric regression fitter"):
        fit(x, Z, e, model=sp.CoxPH)
    with pytest.raises(ValueError, match="parametric regression fitter"):
        fit(x, Z, e, model=sp.Weibull)
    with pytest.raises(
        ValueError, match=r"no regression fitter for the cause\(s\) \[2\]"
    ):
        fit(x, Z, e, model={1: WeibullPH})
    with pytest.raises(ValueError, match="not causes in the data"):
        fit(x, Z, e, init={3: [1.0, 1.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="every row is censored"):
        fit(x, Z, [None] * x.size)
    c = np.zeros(x.size, dtype=int)
    c[0] = 2
    with pytest.raises(ValueError):
        fit(x, Z, np.where(e == None, 1, e), c=c)  # noqa: E711
    with pytest.raises(ValueError, match="w"):
        fit(x, Z, e, w=np.ones(x.size))
    model = fit(x, Z, e)
    with pytest.raises(ValueError, match="3"):
        model.cif(1.0, [0.0, 0.0], 3)
    with pytest.raises(ValueError, match="3"):
        model.sf(1.0, [0.0, 0.0], event=3)
    with pytest.raises(ValueError):
        model.cif(1.0, [0.0, 0.0, 0.0], 1)


def test_init_starts_each_cause():
    x, Z, e, _ = _weibull_ph_causes(300, 5, censor=30.0)
    default = ParametricCompetingRisksRegression.fit(x, Z, e)
    started = ParametricCompetingRisksRegression.fit(
        x, Z, e, init={1: [9.0, 1.5, 0.0, 0.0]}
    )
    np.testing.assert_allclose(started.params, default.params, rtol=1e-4)
    shared = ParametricCompetingRisksRegression.fit(
        x, Z, e, init=[9.0, 1.5, 0.0, 0.0]
    )
    np.testing.assert_allclose(shared.params, default.params, rtol=1e-4)


def test_fit_takes_a_surpyval_data():
    x, Z, e, _ = _weibull_ph_causes(300, 5, censor=30.0)
    rng = np.random.default_rng(0)
    tl = rng.uniform(0, 0.5, x.size) * x
    c = np.array([1 if v is None else 0 for v in e])
    data = sp.SurpyvalData(x, c=c, tl=tl, Z=Z, e=e)
    held = ParametricCompetingRisksRegression.fit(data)
    arrays = ParametricCompetingRisksRegression.fit(x, Z, e, tl=tl)
    np.testing.assert_array_equal(held.params, arrays.params)
    with pytest.raises(TypeError, match="leave out"):
        ParametricCompetingRisksRegression.fit(data, Z)


def test_diagnose_points_to_the_causes_own_models(fitted):
    model, _ = fitted
    with pytest.raises(NotImplementedError, match=r"models\[cause\]"):
        model.diagnose()
    # each cause's regression has its own
    assert model.models[1].diagnose().table.shape[0] > 0
