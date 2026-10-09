"""One sample-size rule for BIC and AIC_c across the library.

The sample size ``d`` of BIC's ``k ln d`` penalty and of AIC_c's
``(2k^2 + 2k) / (d - k - 1)`` correction is the number of observed failures
-- exact, left- and interval-censored observations, weighted by their
counts -- falling back to the (weighted) number of observations when there
is none. Recurrent-event models count observed events (a left- or
interval-censored count adds the events it holds); copulas count the rows
in which at least one series failed. Before, the univariate models counted
every non-right-censored unit for BIC but every unit for AIC_c; regression
counted exact failures only; recurrent models counted exact events (NaN
without one); copulas and Royston-Parmar counted every row.
"""

import json
import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval import Exponential, Weibull, WeibullAFT, WeibullPH
from surpyval.multivariate import Clayton
from surpyval.recurrent import HPP
from surpyval.tests._helpers import (
    fitted_accelerated_life_model,
    no_warnings,
    weibull_binary_covariate_data,
)
from surpyval.univariate.parametric.royston_parmar import RoystonParmar
from surpyval.univariate.regression.parametric_regression_model import (
    ParametricRegressionModel,
)

N = 80


def _aic_c(model, k, d):
    return model.aic() + (2 * k**2 + 2 * k) / (d - k - 1)


@pytest.fixture(scope="module")
def mixed():
    """Exact, right-, interval- and left-censored Weibull data: 45 exact,
    20 right-censored, 10 interval-censored and 5 left-censored units, so
    60 observed failures."""
    np.random.seed(0)
    t = Weibull.random(N, 10, 2)
    x = np.round(t, 2)
    c = np.zeros(N, dtype=int)
    c[:20], c[20:30], c[30:35] = 1, 2, -1
    xl, xr = x.copy(), x.copy()
    xl[20:30] = np.floor(t[20:30])
    xr[20:30] = np.floor(t[20:30]) + 1
    return np.column_stack([xl, xr]), c


@pytest.fixture(scope="module")
def right_censored():
    np.random.seed(1)
    x = Weibull.random(60, 10, 2)
    c = (x > 12).astype(int)
    return np.minimum(x, 12), c


# -- the shared helper -------------------------------------------------------


def test_helper_counts_observed_failures_with_weights():
    from surpyval.univariate.information_criteria import ic_sample_size

    assert ic_sample_size([0, 1, 2, -1], [3, 5, 2, 1]) == 6.0
    # no failure: the weighted rows, or the fallback given
    assert ic_sample_size([1, 1], [3, 5]) == 8.0
    assert ic_sample_size([1, 1], [3, 5], n_rows=2) == 2.0
    # joint rows count when any series failed
    c = np.array([[1, 1], [0, 1], [1, 2], [0, 0]])
    assert ic_sample_size(c, [4, 3, 2, 1]) == 6.0
    assert ic_sample_size(np.ones((2, 2), dtype=int), [4, 3]) == 7.0


# -- univariate parametric ---------------------------------------------------


def test_univariate_bic_unchanged_on_exact_and_right_censored(right_censored):
    x, c = right_censored
    model = Weibull.fit(x, c)
    d = int((c == 0).sum())
    assert model.bic() == pytest.approx(2 * np.log(d) + 2 * model.neg_ll())
    # the value this data gave before the change
    assert model.bic() == pytest.approx(307.2914851874891, rel=1e-9)


def test_univariate_aic_c_uses_the_failures(right_censored):
    x, c = right_censored
    model = Weibull.fit(x, c)
    d = int((c == 0).sum())
    assert d < len(x)
    assert model.aic_c() == pytest.approx(_aic_c(model, 2, d))


def test_univariate_interval_only_is_finite_and_counts_failures():
    model = Weibull.fit(xl=np.arange(1, 20), xr=np.arange(2, 21))
    assert model.bic() == pytest.approx(2 * np.log(19) + 2 * model.neg_ll())
    assert model.aic_c() == pytest.approx(_aic_c(model, 2, 19))


def test_univariate_no_failure_falls_back_to_units():
    fitted = Weibull.fit([3, 4, 5, 6, 7, 8, 9, 10], c=[0, 0, 0, 1, 1, 1, 1, 1])
    d = fitted.to_dict(with_data=True)
    d["data"]["c"] = [1] * len(d["data"]["c"])
    del d["ic_n"]
    restored = surv.from_dict(d)
    nll = restored.neg_ll()
    assert restored.bic() == pytest.approx(2 * np.log(8) + 2 * nll)
    assert restored.aic_c() == pytest.approx(_aic_c(restored, 2, 8))


def test_univariate_restored_without_data_keeps_bic_and_aic_c(right_censored):
    x, c = right_censored
    model = Weibull.fit(x, c)
    restored = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.data is None
    assert restored.bic() == pytest.approx(model.bic())
    assert restored.aic_c() == pytest.approx(model.aic_c())
    # a dict written before the sample size was stored needs the data
    old = model.to_dict()
    del old["ic_n"]
    with pytest.raises(ValueError, match="needs the data"):
        surv.from_dict(old).bic()


# -- regression --------------------------------------------------------------


def test_regression_counts_all_observed_failures(mixed):
    X, c = mixed
    Z = np.random.default_rng(3).normal(size=(N, 1))
    model = WeibullPH.fit(x=X, c=c, Z=Z)
    assert model.bic() == pytest.approx(3 * np.log(60) + 2 * model.neg_ll())
    assert model.aic_c() == pytest.approx(_aic_c(model, 3, 60))


def test_regression_with_null_covariate_matches_univariate(mixed):
    X, c = mixed
    uni = Weibull.fit(x=X, c=c)
    # A column of zeros determines no coefficient: it is aliased (#476),
    # with a warning, and not estimated, so it costs no degree of freedom.
    with pytest.warns(UserWarning, match="cannot be estimated"):
        reg = WeibullPH.fit(x=X, c=c, Z=np.zeros((N, 1)))
    assert reg.neg_ll() == pytest.approx(uni.neg_ll(), rel=1e-6)
    # the same d, and the same number of estimated parameters
    assert reg.bic() == pytest.approx(uni.bic(), rel=1e-6)
    assert reg.aic_c() - reg.aic() == pytest.approx(12 / (60 - 3))
    assert uni.aic_c() - uni.aic() == pytest.approx(12 / (60 - 3))
    # A covariate that is estimated costs one.
    Z = np.random.default_rng(0).normal(size=(N, 1))
    reg = WeibullPH.fit(x=X, c=c, Z=Z)
    assert reg.k == uni.k + 1
    assert reg.aic_c() - reg.aic() == pytest.approx(24 / (60 - 4))


def test_regression_without_exact_failures_uses_failures():
    # 40 interval-censored failures and 10 right-censored units: no exact
    # failure, which made d = 0 (and the count fall back to all 50 units).
    np.random.seed(4)
    t = Weibull.random(50, 10, 2)
    X = np.column_stack([np.floor(t), np.floor(t) + 1])
    X[40:] = 15.0
    c = np.r_[np.full(40, 2), np.ones(10, dtype=int)]
    Z = np.random.binomial(1, 0.5, (50, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = WeibullPH.fit(x=X, c=c, Z=Z)
    assert np.isfinite(model.bic())
    assert model.bic() == pytest.approx(3 * np.log(40) + 2 * model.neg_ll())


def test_regression_restored_keeps_bic_and_aic_c(mixed):
    X, c = mixed
    Z = np.random.default_rng(3).normal(size=(N, 1))
    model = WeibullPH.fit(x=X, c=c, Z=Z)
    restored = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.bic() == pytest.approx(model.bic())
    assert restored.aic_c() == pytest.approx(model.aic_c())


def _tvc_data(n_subjects, censor_every=3):
    rng = np.random.default_rng(5)
    Z = rng.normal(size=n_subjects)
    x = 10 * rng.weibull(2, n_subjects) * np.exp(-0.3 * Z)
    i, xl, xr, c, Zr = [], [], [], [], []
    for s in range(n_subjects):
        mid = x[s] * 0.5
        i += [s, s]
        xl += [0.0, mid]
        xr += [mid, x[s]]
        c += [1, 1 if s % censor_every == 0 else 0]
        Zr += [[Z[s]], [Z[s]]]
    return [np.array(a) for a in (i, xl, xr, c, Zr)]


@pytest.mark.parametrize("fitter", [WeibullPH, WeibullAFT])
def test_time_varying_fits_count_failed_subjects(fitter):
    i, xl, xr, c, Zr = _tvc_data(60)
    d = int((c == 0).sum())
    assert d == 40
    model = fitter.fit_tvc(i, xl, xr, c, Zr)
    k = model.k
    assert model.bic() == pytest.approx(k * np.log(d) + 2 * model.neg_ll())
    assert model.aic_c() == pytest.approx(_aic_c(model, k, d))
    restored = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.bic() == pytest.approx(model.bic())
    assert restored.aic_c() == pytest.approx(model.aic_c())


def test_frailty_aic_c_uses_events():
    rng = np.random.default_rng(2)
    z = rng.normal(size=200)
    x = 20 * (-np.log(rng.uniform(size=200)) / np.exp(0.8 * z)) ** (1 / 1.8)
    c = (x > 25).astype(int)
    x = np.minimum(x, 25)
    groups = np.repeat(np.arange(20), 10)
    model = surv.WeibullFrailty.fit(
        x=x, c=c, Z=z.reshape(-1, 1), groups=groups
    )
    d = int((c == 0).sum())
    assert d < 200
    k = model.k
    assert model.bic() == pytest.approx(k * np.log(d) + 2 * model.neg_ll())
    assert model.aic_c() == pytest.approx(_aic_c(model, k, d))
    restored = type(model).from_dict(model.to_dict())
    assert restored.aic_c() == pytest.approx(model.aic_c())


# -- Royston-Parmar ----------------------------------------------------------


def test_royston_parmar_bic_counts_failures(right_censored):
    x, c = right_censored
    model = RoystonParmar.fit(x, c=c, df=1)
    d = int((c == 0).sum())
    assert model.bic() == pytest.approx(2 * np.log(d) + 2 * model.neg_ll())
    # df = 1 on the hazard scale is the Weibull: the same BIC
    assert model.bic() == pytest.approx(Weibull.fit(x, c).bic(), rel=1e-5)
    restored = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.bic() == pytest.approx(model.bic())


def test_royston_parmar_interval_failures_count(mixed):
    X, c = mixed
    model = RoystonParmar.fit(x=X, c=c, df=1)
    assert model.bic() == pytest.approx(2 * np.log(60) + 2 * model.neg_ll())
    # a dict written before ``ic_n`` falls back to the exact failures
    old = model.to_dict()
    del old["ic_n"]
    restored = surv.from_dict(old)
    assert restored.bic() == pytest.approx(2 * np.log(45) + 2 * model.neg_ll())


# -- recurrent events --------------------------------------------------------


def test_recurrent_counts_interval_and_exact_events():
    x = [[0, 10], [10, 20], 25.0, 30.0, 40.0]
    model = HPP.fit(x, i=[1, 1, 1, 1, 1], c=[2, 2, 0, 0, 1], n=[2, 3, 1, 1, 1])
    assert model._n_obs == 7
    assert model.bic() == pytest.approx(np.log(7) - 2 * model.log_likelihood)


def test_recurrent_hpp_matches_univariate_exponential():
    rng = np.random.default_rng(0)
    events = np.cumsum(rng.exponential(2, 30))
    end = events[-1] + 1.5
    hpp = HPP.fit(np.r_[events, end], c=np.r_[np.zeros(30), 1])
    gaps = np.r_[np.diff(np.r_[0, events]), end - events[-1]]
    exp = Exponential.fit(gaps, c=np.r_[np.zeros(30), 1])
    # the same likelihood and the same 30 observed events
    assert hpp.log_likelihood == pytest.approx(-exp.neg_ll(), rel=1e-8)
    assert hpp.bic() == pytest.approx(exp.bic(), rel=1e-8)


# -- copulas -----------------------------------------------------------------


@pytest.fixture(scope="module")
def clayton_sample():
    margins = [Weibull.from_params([10, 2]), Weibull.from_params([20, 3])]
    X = Clayton.from_params([2.0], margins).random(200, random_state=0)
    return X, margins


def test_copula_counts_rows_with_a_failure(clayton_sample):
    X, _ = clayton_sample
    c = np.zeros_like(X, dtype=int)
    c[:50] = 1  # both series right-censored: no failure
    c[50:120, 1] = 1  # one series failed
    model = Clayton.fit(X, c=c, margins=[Weibull, Weibull])
    expected = model.k * np.log(150) + 2 * model.neg_ll()
    assert model.bic() == pytest.approx(expected)
    restored = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.bic() == pytest.approx(model.bic())


def test_copula_no_failure_falls_back_to_rows(clayton_sample):
    X, margins = clayton_sample
    c = np.ones_like(X, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Clayton.fit(X, c=c, margins=margins)
    assert model.bic() == pytest.approx(
        model.k * np.log(200) + 2 * model.neg_ll()
    )


def test_copula_old_dict_restores_with_its_row_count(clayton_sample):
    X, _ = clayton_sample
    c = np.zeros_like(X, dtype=int)
    c[:50] = 1
    model = Clayton.fit(X, c=c, margins=[Weibull, Weibull])
    old = model.to_dict()
    old["n_obs"] = 200.0
    del old["ic_n"]
    restored = surv.from_dict(old)
    assert restored.bic() == pytest.approx(
        model.k * np.log(200) + 2 * model.neg_ll()
    )


# ---------------------------------------------------------------------------
# AIC and BIC count only the estimated parameters.
# ---------------------------------------------------------------------------


def test_information_criteria_exclude_fixed_parameters():
    np.random.seed(0)
    x = surv.Weibull.random(50, 10, 2)
    model = surv.Weibull.fit(x, fixed={"beta": 2})
    n_obs = len(x)
    assert model.aic() == pytest.approx(2 * 1 + 2 * model.neg_ll())
    assert model.bic() == pytest.approx(np.log(n_obs) + 2 * model.neg_ll())
    assert model.aic_c() == pytest.approx(
        model.aic() + (2 * 1 + 2 * 1) / (n_obs - 1 - 1)
    )
    # the fixed-shape Weibull and the Rayleigh are the same model with
    # the same single free parameter, so they score the same
    rayleigh = surv.Rayleigh.fit(x)
    fixed = surv.Weibull.fit(x, fixed={"beta": 2})
    assert fixed.aic() == pytest.approx(rayleigh.aic(), rel=1e-6)
    # a free fit still counts both
    free = surv.Weibull.fit(x)
    assert free.aic() == pytest.approx(2 * 2 + 2 * free.neg_ll())


def test_fixed_offset_and_lfp_are_not_counted_either():
    np.random.seed(0)
    x = surv.Weibull.random(60, 10, 2) + 5
    model = surv.Weibull.fit(x, offset=True, fixed={"gamma": 4.0})
    assert model.aic() == pytest.approx(2 * 2 + 2 * model.neg_ll())


def test_restored_model_keeps_the_estimated_parameter_count():
    np.random.seed(0)
    x = surv.Weibull.random(50, 10, 2)
    model = surv.Weibull.fit(x, fixed={"beta": 2})
    d = model.to_dict(with_data=True)
    assert d["fixed"] == ["beta"]
    restored = surv.from_dict(d)
    assert restored.aic() == pytest.approx(model.aic())
    assert restored.bic() == pytest.approx(model.bic())
    assert restored.to_dict(with_data=True)["fixed"] == ["beta"]
    # no fixed parameters: no key
    assert "fixed" not in surv.Weibull.fit(x).to_dict()


# ---------------------------------------------------------------------------
# BIC without exact failures; AIC_c without enough observations.
# ---------------------------------------------------------------------------


W, E, G = surv.Weibull, surv.Exponential, surv.Geometric


def test_bic_is_finite_without_exact_failures():
    model = W.fit(xl=np.arange(1, 20), xr=np.arange(2, 21))
    bic = no_warnings(model.bic)
    assert np.isfinite(bic)
    assert bic == pytest.approx(2 * np.log(19) + 2 * model.neg_ll())


def test_bic_counts_exact_failures_only_for_exact_and_right_censored():
    model = W.fit([1.0, 2, 3, 4, 5, 6], c=[0, 0, 0, 0, 1, 1])
    assert model.bic() == pytest.approx(2 * np.log(4) + 2 * model.neg_ll())


@pytest.mark.parametrize("x", [[1.0, 2.0, 3.0], [1.0, 2.0]])
def test_aic_c_is_nan_without_enough_observations(x):
    model = W.fit(x) if len(x) == 3 else W.fit(x, fixed={"beta": 2.0})
    if len(x) == 2:
        # k = 1, N = 2: N = k + 1
        assert np.isnan(no_warnings(model.aic_c))
    else:
        # k = 2, N = 3: N = k + 1
        assert np.isnan(no_warnings(model.aic_c))
    assert not model.aic_c() < model.aic()


# ---------------------------------------------------------------------------
# A regression's ``k`` counts the estimated parameters only.
# ---------------------------------------------------------------------------


def test_fixed_parameters_not_counted_in_k():
    x, Z = weibull_binary_covariate_data()
    free = WeibullPH.fit(x, Z)
    fixed = WeibullPH.fit(x, Z, fixed={"beta": 2.0})
    assert free.k == 3
    assert fixed.k == 2
    n = len(x)
    assert fixed.aic() == pytest.approx(2 * 2 + 2 * fixed.neg_ll())
    assert fixed.aic_c() == pytest.approx(
        fixed.aic() + (2 * 2**2 + 2 * 2) / (n - 2 - 1)
    )


def test_restored_model_k_counts_estimated_parameters():
    model = fitted_accelerated_life_model()
    d = model.to_dict()
    d["k"] = 4  # a dict written by a version that counted the placeholder
    restored = ParametricRegressionModel.from_dict(d)
    assert restored.k == 3
    assert restored.aic() == pytest.approx(model.aic())


# ---------------------------------------------------------------------------
# AIC_c and the log-likelihood wherever there is an AIC (#605)
# ---------------------------------------------------------------------------


def _hpp():
    rng = np.random.default_rng(0)
    events = np.cumsum(rng.exponential(2, 30))
    return HPP.fit(np.r_[events, events[-1] + 1.5], c=np.r_[np.zeros(30), 1])


def _parametric_competing_risks():
    from surpyval.univariate.competing_risks import ParametricCompetingRisks

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    e = ["a", "b", "a", None, "a", "b", "a", None, "b", "a", "b", "a"]
    return ParametricCompetingRisks.fit(x, e, dist=Exponential), 10


@pytest.mark.parametrize(
    "fit", ["royston-parmar", "hpp", "copula", "competing-risks"]
)
def test_605_aic_c_and_log_likelihood_wherever_aic(
    fit, right_censored, clayton_sample
):
    if fit == "royston-parmar":
        x, c = right_censored
        model, d = RoystonParmar.fit(x, c=c, df=2), int((c == 0).sum())
        k = 3
    elif fit == "hpp":
        model, d, k = _hpp(), 30, 1
    elif fit == "copula":
        X, _ = clayton_sample
        model = Clayton.fit(X, margins=[Weibull, Weibull])
        d, k = 200, model.k
    else:
        model, d = _parametric_competing_risks()
        k = 2
    assert model.aic_c() == pytest.approx(_aic_c(model, k, d), rel=1e-12)
    assert isinstance(model.log_likelihood, float)
    assert model.log_likelihood == -model.neg_ll()


def test_605_restored_parametric_model_has_its_log_likelihood(
    right_censored,
):
    x, c = right_censored
    model = Weibull.fit(x, c)
    restored = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.log_likelihood == model.log_likelihood == -model.neg_ll()


# -- #711: the degradation and cause-specific recurrent fits ----------------


def _paths(shape=0.2, units=5):
    """Five units read every 10 time units, wear in gamma increments of
    mean 0.5 per unit time (skewed for a small ``shape``)."""
    rng = np.random.default_rng(1)
    t = np.tile(np.arange(0, 110, 10.0), units)
    i = np.repeat(np.arange(units), 11)
    steps = rng.gamma(shape * 10, 0.5 / shape, size=(units, 10))
    y = np.hstack([np.r_[0.0, np.cumsum(s)] for s in steps])
    return t, y, i


def _check_criteria(model, k, n):
    neg_ll = model.neg_ll()
    assert isinstance(neg_ll, float) and np.isfinite(neg_ll)
    assert model.log_likelihood == -neg_ll
    assert model.aic() == pytest.approx(2 * k + 2 * neg_ll, rel=1e-12)
    assert model.bic() == pytest.approx(k * np.log(n) + 2 * neg_ll, rel=1e-12)
    assert model.aic_c() == pytest.approx(_aic_c(model, k, n), rel=1e-12)


def test_711_wiener_and_gamma_process_criteria():
    from scipy import stats

    from surpyval.degradation import GammaProcess, WienerProcess

    t, y, i = _paths()
    dt, dy = np.full(50, 10.0), np.diff(y.reshape(5, 11), axis=1).ravel()
    wiener = WienerProcess.fit(t, y, i, threshold=100)
    gamma = GammaProcess.fit(t, y, i, threshold=100)
    # the likelihood of the increments, independently
    assert wiener.neg_ll() == pytest.approx(
        -stats.norm.logpdf(
            dy, wiener.mu * dt, wiener.sigma * np.sqrt(dt)
        ).sum(),
        rel=1e-10,
    )
    assert gamma.neg_ll() == pytest.approx(
        -stats.gamma.logpdf(dy, gamma.alpha * dt, scale=1 / gamma.beta).sum(),
        rel=1e-10,
    )
    # two parameters each (y0 is not one), 50 increments
    _check_criteria(wiener, 2, 50)
    _check_criteria(gamma, 2, 50)
    # the same increments, so the two rank: skewed wear is a gamma process
    assert gamma.aic() < wiener.aic() - 5
    assert gamma.bic() < wiener.bic() - 5


@pytest.mark.parametrize("process", ["WienerProcess", "GammaProcess"])
def test_711_process_criteria_count_the_stress_coefficients(process):
    import surpyval.degradation as deg

    t, y, i = _paths()
    Z = np.repeat([1.0, 1.0, 2.0, 2.0, 3.0], 11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = getattr(deg, process).fit(t, y, i, threshold=100, Z=Z)
    _check_criteria(model, 3, 50)


def test_711_process_criteria_restored_and_from_params():
    from surpyval.degradation import GammaProcess, GammaProcessModel

    t, y, i = _paths()
    model = GammaProcess.fit(t, y, i, threshold=100)
    restored = GammaProcessModel.from_dict(
        json.loads(json.dumps(model.to_dict()))
    )
    assert restored.aic() == model.aic()
    assert restored.bic() == model.bic()
    with pytest.raises(ValueError, match="has no likelihood"):
        GammaProcessModel(2.0, 4.0, threshold=100).aic()


def test_711_destructive_degradation_criteria():
    from scipy import stats

    from surpyval.degradation import DestructiveDegradation

    rng = np.random.default_rng(1)
    x = np.repeat([10.0, 20.0, 30.0, 40.0], 6)
    y = np.exp(4.0 - 0.02 * x + rng.normal(0, 0.1, 24))
    c = np.zeros(24, int)
    c[[3, 9]] = 1  # did not break at the maximum load: not counted in n
    model = DestructiveDegradation.fit(x, y, c=c, threshold=20)
    mu = model.beta[0] + model.beta[1] * x
    log_lik = np.where(
        c == 0,
        stats.lognorm.logpdf(y, model.sigma, scale=np.exp(mu)),
        stats.lognorm.logsf(y, model.sigma, scale=np.exp(mu)),
    )
    assert model.neg_ll() == pytest.approx(-log_lik.sum(), rel=1e-10)
    # beta_0, beta_1 and sigma; 22 measurements not right-censored
    _check_criteria(model, 3, 22)


def test_711_cause_specific_nhpp_criteria():
    from surpyval.recurrent import CauseSpecificNHPP

    x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
    i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
    model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
    # the likelihood factorises over the causes
    assert model.neg_ll() == pytest.approx(
        sum(model.models[k].neg_ll() for k in model.event_types)
    )
    # two Crow-AMSAA parameters per cause; nine events of either cause
    _check_criteria(model, 4, 9)
    with pytest.raises(ValueError, match="how='MSE'"):
        CauseSpecificNHPP.fit(x, i=i, c=c, e=e, how="MSE").aic()
