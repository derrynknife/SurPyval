"""``LittlewoodVerrall``: exponential gaps whose rates are gamma.

The likelihood against a sum of Pareto densities written out, its limit
as alpha grows, recovery from simulated campaigns (seeds fixed before
the results were seen), the refusal where alpha has no estimate, the
predictions against closed forms, the bounds and saving.

No published fit is checked: none could be read (see DECISIONS.md).
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import gamma

import surpyval as sp
from surpyval.recurrent import (
    JelinskiMoranda,
    LittlewoodVerrall,
    LittlewoodVerrallModel,
    MusaOkumoto,
    compare_predictions,
)

TRUTH = [3.0, 10.0, 2.0]


@pytest.fixture(scope="module")
def gaps():
    return LittlewoodVerrall.from_params(TRUTH).random(80, random_state=1)


@pytest.fixture(scope="module")
def fitted(gaps):
    return LittlewoodVerrall.fit(gaps, gaps=True)


def _direct(gaps, alpha, beta0, beta1, since=0.0, power=1):
    total = 0.0
    for i, t in enumerate(gaps, start=1):
        psi = beta0 + beta1 * i**power
        total += (
            np.log(alpha) + alpha * np.log(psi) - (alpha + 1) * np.log(psi + t)
        )
    psi = beta0 + beta1 * (len(gaps) + 1) ** power
    return total + alpha * np.log(psi / (psi + since))


def test_likelihood_is_the_sum_of_pareto_densities(gaps, fitted):
    assert fitted.log_likelihood == pytest.approx(
        _direct(gaps, *fitted.params), rel=1e-12
    )
    assert fitted.aic() == pytest.approx(6 - 2 * fitted.log_likelihood)
    # with the end of testing after the last failure
    since = 40.0
    later = LittlewoodVerrall.fit(
        np.append(gaps, since), c=[0] * 80 + [1], gaps=True
    )
    assert later.log_likelihood == pytest.approx(
        _direct(gaps, *later.params, since=since), rel=1e-12
    )
    # and as failure times
    times = LittlewoodVerrall.fit(np.cumsum(gaps))
    np.testing.assert_allclose(times.params, fitted.params, rtol=1e-6)
    quadratic = LittlewoodVerrall.fit(gaps, gaps=True, psi="quadratic")
    assert quadratic.log_likelihood == pytest.approx(
        _direct(gaps, *quadratic.params, power=2), rel=1e-12
    )
    assert quadratic.psi_form == "quadratic"


def test_fit_is_a_maximum_of_the_likelihood(gaps, fitted):
    assert fitted.maximum == "verified"
    best = fitted.log_likelihood
    for k in range(3):
        for factor in (0.97, 1.03):
            params = fitted.params.copy()
            params[k] *= factor
            assert _direct(gaps, *params) < best
    # an independent search, in the model's own parameters
    res = minimize(
        lambda p: -_direct(gaps, *p) if p[0] > 0 and p[1] + p[2] > 0 else 1e9,
        fitted.params * 1.2,
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000},
    )
    assert -res.fun == pytest.approx(best, abs=1e-6)


def test_limit_of_large_alpha_is_exponential_gaps(gaps):
    # psi = alpha (b0 + b1 i): the gaps' density tends to an
    # exponential's with mean b0 + b1 i
    b0, b1 = 4.0, 1.5
    mean = b0 + b1 * np.arange(1, 81)
    exponential = np.sum(-np.log(mean) - gaps / mean)
    previous = np.inf
    for alpha in (1e2, 1e4, 1e6):
        gap = abs(_direct(gaps, alpha, alpha * b0, alpha * b1) - exponential)
        assert gap < previous
        previous = gap
    assert previous < 1e-3


@pytest.mark.parametrize("n", [150, 400])
@pytest.mark.parametrize("psi", ["linear", "quadratic"])
def test_recovery_from_simulated_campaigns(n, psi):
    # Seed fixed before the run: each estimate within 3 standard errors.
    params = TRUTH if psi == "linear" else [3.0, 10.0, 0.05]
    truth = LittlewoodVerrall.from_params(params, psi=psi)
    data = truth.random(n, random_state=20261120)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = LittlewoodVerrall.fit(data, gaps=True, psi=psi)
    z = (model.params - params) / model.standard_errors()
    print(n, psi, model.params, z)
    assert np.all(np.abs(z) < 3.0), z


def test_gaps_with_no_extra_scatter_are_refused():
    # exponential gaps with a growing mean: the model's limit
    rng = np.random.default_rng(0)
    gaps = rng.exponential(1.0, 60) * (5 + np.arange(60))
    with pytest.raises(ValueError) as error:
        LittlewoodVerrall.fit(gaps, gaps=True)
    message = str(error.value)
    assert "no more scattered than exponential times" in message
    assert "alpha has no estimate" in message
    assert "keeps rising as alpha grows without bound" in message
    assert "MusaOkumoto or GoelOkumoto" in message
    assert "JelinskiMoranda" in message
    # and gaps with no scatter at all
    with pytest.raises(ValueError, match="alpha has no estimate"):
        LittlewoodVerrall.fit(np.cumsum(5.0 + np.arange(30.0)))


def test_predictions_are_the_closed_forms():
    model = LittlewoodVerrall.from_params(TRUTH, n=20)
    psi = 10.0 + 2.0 * 21
    assert model.psi(21) == psi
    x = np.array([0.0, 5.0, 26.0, 260.0])
    np.testing.assert_allclose(
        model.next_failure_sf(x), (psi / (psi + x)) ** 3.0, rtol=1e-13
    )
    assert model.next_failure_mean() == pytest.approx(psi / 2.0)
    assert model.mtbf() == model.next_failure_mean()
    median = model.next_failure_qf(0.5)
    assert model.next_failure_sf(median) == pytest.approx(0.5, rel=1e-12)
    np.testing.assert_allclose(
        model.next_failure_sf(model.next_failure_qf([0.1, 0.9])),
        [0.9, 0.1],
        rtol=1e-12,
    )
    assert model.current_rate() == pytest.approx(3.0 / psi)
    np.testing.assert_allclose(
        model.current_rate_qf([0.05, 0.95]),
        gamma.ppf([0.05, 0.95], 3.0, scale=1 / psi),
    )
    # the predictive is the exponential averaged over the rate
    rates = gamma.ppf((np.arange(20000) + 0.5) / 20000, 3.0, scale=1 / psi)
    assert model.next_failure_sf(26.0) == pytest.approx(
        np.mean(np.exp(-rates * 26.0)), rel=1e-3
    )
    quadratic = LittlewoodVerrall.from_params(
        [3.0, 10.0, 0.05], n=20, psi="quadratic"
    )
    assert quadratic.psi(21) == pytest.approx(10.0 + 0.05 * 441)


def test_an_infinite_mean_is_refused_with_the_median_to_use():
    model = LittlewoodVerrall.from_params([0.8, 10.0, 2.0], n=5)
    with pytest.raises(ValueError) as error:
        model.next_failure_mean()
    assert "alpha = 0.8 is not above 1" in str(error.value)
    assert "next_failure_qf(0.5)" in str(error.value)
    assert np.isfinite(model.next_failure_qf(0.5))
    with pytest.raises(ValueError, match="infinite"):
        model.mtbf()


def test_alpha_bound_by_the_likelihood_ratio(gaps, fitted):
    lower, upper = fitted.param_cb("alpha", alpha_ci=0.5)
    assert lower < fitted.alpha < upper and np.isfinite(upper)
    best = fitted.log_likelihood
    drop = 0.5 * 0.454936423119572  # chi-squared(1) at 0.5

    def profile(alpha):
        res = minimize(
            lambda u: -_direct(
                gaps,
                alpha,
                *fitted._data.betas(*np.exp(u)),
            ),
            np.log(fitted._data.ends(*fitted.params[1:])),
            method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 5000},
        )
        return -res.fun

    for end in (lower, upper):
        assert best - profile(end) == pytest.approx(drop, abs=1e-5)
    assert fitted.param_cb("alpha", 0.25, "upper") == pytest.approx(upper)
    # at 95% the limit of no scatter is inside: no upper bound
    wide = fitted.param_cb("alpha")
    assert wide[0] < lower and wide[1] == np.inf
    assert fitted.param_cb("alpha", bound="upper") == np.inf


def test_wald_bounds_and_refusals(fitted):
    se = fitted.standard_errors()
    z = 1.959963984540054
    np.testing.assert_allclose(
        fitted.param_cb("beta1"),
        fitted.params[2] + np.array([-z, z]) * se[2],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        fitted.param_cb("alpha", method="wald"),
        fitted.alpha * np.exp(np.array([-z, z]) * se[0] / fitted.alpha),
        rtol=1e-12,
    )
    assert fitted.covariance().shape == (3, 3)
    with pytest.raises(ValueError, match="for alpha only"):
        fitted.param_cb("beta0", method="lr")
    with pytest.raises(ValueError, match="method"):
        fitted.param_cb("alpha", method="bootstrap")
    with pytest.raises(ValueError, match="name"):
        fitted.param_cb("psi")
    given = LittlewoodVerrall.from_params(TRUTH, n=20)
    with pytest.raises(ValueError, match="LittlewoodVerrall.fit"):
        given.param_cb("alpha")
    with pytest.raises(ValueError, match="LittlewoodVerrall.fit"):
        given.aic()


def test_covariance_is_the_inverse_information_in_the_parameters(gaps, fitted):
    from surpyval.utils.linalg import numerical_hessian

    hessian = numerical_hessian(
        lambda p: -_direct(gaps, *p), fitted.params.copy()
    )
    np.testing.assert_allclose(
        fitted.covariance(), np.linalg.inv(hessian), rtol=2e-3
    )


def test_from_params_refusals():
    with pytest.raises(ValueError, match="alpha is a gamma shape"):
        LittlewoodVerrall.from_params([0.0, 10.0, 2.0])
    with pytest.raises(ValueError) as error:
        LittlewoodVerrall.from_params([3.0, 10.0, -1.0], n=12)
    assert "psi(13) = beta0 + beta1 * 13 = -3 is not positive" in str(
        error.value
    )
    with pytest.raises(ValueError, match="three finite numbers"):
        LittlewoodVerrall.from_params([3.0, 10.0])
    with pytest.raises(ValueError, match="psi"):
        LittlewoodVerrall.from_params(TRUTH, psi="cubic")
    with pytest.raises(ValueError, match="whole number of at least 0"):
        LittlewoodVerrall.from_params(TRUTH, n=-1)


def test_data_refusals(gaps):
    with pytest.raises(ValueError, match="at least 5 failures"):
        LittlewoodVerrall.fit([1.0, 3.0, 7.0, 15.0])
    with pytest.raises(ValueError, match="times between failures are 0"):
        LittlewoodVerrall.fit([1.0, 3.0, 3.0, 7.0, 15.0, 40.0])
    with pytest.raises(ValueError, match="Littlewood-Verrall"):
        LittlewoodVerrall.fit([1.0, np.nan, 3.0, 7.0, 15.0, 40.0])
    with pytest.raises(ValueError, match="psi"):
        LittlewoodVerrall.fit(gaps, gaps=True, psi="cubic")
    with pytest.raises(ValueError, match="size is the number of failures"):
        LittlewoodVerrall.from_params(TRUTH).random(0)
    assert (
        repr(LittlewoodVerrall)
        == "LittlewoodVerrall: rate-uncertainty model fitter"
    )


def test_draws_are_seeded_and_follow_the_model():
    truth = LittlewoodVerrall.from_params(TRUTH)
    np.testing.assert_array_equal(
        truth.random(10, random_state=3), truth.random(10, random_state=3)
    )
    # the 30th gap: Pareto with psi(30) = 70; its median over many draws
    draws = np.array(
        [truth.random(30, random_state=s)[-1] for s in range(2000)]
    )
    median = 70.0 * (2 ** (1 / 3.0) - 1)
    assert np.median(draws) == pytest.approx(median, rel=0.1)


def test_save_and_load_and_the_data_frame(gaps, fitted):
    import pandas as pd

    back = LittlewoodVerrallModel.from_dict(fitted.to_dict())
    np.testing.assert_array_equal(back.params, fitted.params)
    assert back.n == 80 and back.psi_form == "linear"
    np.testing.assert_allclose(
        back.next_failure_sf([5.0, 50.0]), fitted.next_failure_sf([5.0, 50.0])
    )
    np.testing.assert_allclose(
        back.param_cb("beta0"), fitted.param_cb("beta0"), rtol=1e-12
    )
    assert isinstance(sp.from_json(fitted.to_json()), LittlewoodVerrallModel)
    assert "Littlewood-Verrall SurPyval Model" in repr(back)
    df = pd.DataFrame({"gap": gaps})
    framed = LittlewoodVerrall.fit_from_df(df, x_col="gap", gaps=True)
    np.testing.assert_array_equal(framed.params, fitted.params)


def test_it_takes_part_in_compare_predictions(gaps):
    result = compare_predictions(
        np.cumsum(gaps), [LittlewoodVerrall, MusaOkumoto, JelinskiMoranda], 40
    )
    made = result.predicted("LittlewoodVerrall")
    assert made.sum() >= 30
    k = int(np.flatnonzero(made)[0])
    i = result.steps[k]
    model = LittlewoodVerrall.fit(np.cumsum(gaps)[:i])
    assert result.u["LittlewoodVerrall"][k] == pytest.approx(
        1 - model.next_failure_sf(result.gaps[k]), rel=1e-10
    )
    psi = model.psi(i + 1)
    assert result.log_density["LittlewoodVerrall"][k] == pytest.approx(
        np.log(model.alpha)
        + model.alpha * np.log(psi)
        - (model.alpha + 1) * np.log(psi + result.gaps[k]),
        rel=1e-10,
    )
