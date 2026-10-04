"""Likelihood inference (AIC/BIC/SE/confidence bounds) for the parametric,
regression and renewal recurrent models -- the behaviour provided by
``LikelihoodInferenceMixin`` from ``_neg_ll``/``_mle``/``_n_obs``."""

import matplotlib
import numpy as np
import pytest
from scipy.stats import norm

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from surpyval import Weibull  # noqa: E402
from surpyval.recurrent import (  # noqa: E402
    ARI,
    HPP,
    CoxLewis,
    CrowAMSAA,
    Duane,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)
from surpyval.tests._helpers import (  # noqa: E402
    REPAIR_FLEET_C,
    REPAIR_FLEET_I,
    REPAIR_FLEET_X,
    exponential_event_times,
)


def _assert_information_criteria(model, dist):
    k = model._mle.size
    n = model._n_obs
    ll = model.log_likelihood
    assert np.isclose(ll, -float(model._neg_ll(model._mle)))
    assert np.isclose(model.aic(), 2 * k - 2 * ll)
    assert np.isclose(model.bic(), k * np.log(n) - 2 * ll)
    assert model.parameter_names == list(dist.parameter_names)


@pytest.mark.parametrize("dist", [HPP, CrowAMSAA, Duane])
def test_parametric_intensity_information_criteria(dist):
    # Every MLE-fitted intensity model now exposes a likelihood and the
    # standard information criteria derived from it.
    model = dist.fit(exponential_event_times())
    _assert_information_criteria(model, dist)


def test_cox_lewis_information_criteria():
    # Cox-Lewis needs a log-linear intensity over a bounded window (its
    # exponential CIF overflows on very large cumulative times), so it gets a
    # tailored dataset rather than the shared one.
    rng = np.random.default_rng(0)
    alpha, beta, T = 0.0, 0.3, 20.0
    lam_max = np.exp(alpha + beta * T)
    cand = np.sort(rng.uniform(0, T, rng.poisson(lam_max * T)))
    keep = rng.uniform(0, 1, cand.size) < np.exp(alpha + beta * cand) / lam_max
    model = CoxLewis.fit(cand[keep], tl=0.0, tr=T)
    _assert_information_criteria(model, CoxLewis)


@pytest.mark.parametrize("dist", [HPP, CrowAMSAA, Duane])
def test_parametric_intensity_standard_errors(dist):
    # Standard errors come from the observed information and are finite and
    # positive for these well-identified fits.
    x = exponential_event_times()
    model = dist.fit(x)
    se = model.standard_errors()
    assert se.shape == (model._mle.size,)
    assert np.all(np.isfinite(se)) and np.all(se > 0)


def test_mse_fit_has_no_likelihood():
    # The MSE fit minimises a sum of squares, not a likelihood, so inference
    # must raise rather than report a meaningless AIC.
    x = exponential_event_times()
    model = CrowAMSAA.fit(x, how="MSE")
    with pytest.raises(ValueError, match="fitted from data"):
        model.log_likelihood
    for name in ("aic", "bic"):
        with pytest.raises(ValueError, match="fitted from data"):
            getattr(model, name)()


def test_from_params_has_no_likelihood():
    # A model built directly from parameters carries no data/likelihood.
    model = CrowAMSAA.from_params([1000.0, 1.2])
    with pytest.raises(ValueError, match="fitted from data"):
        model.aic()


def _regression_data():
    x = [9, 14, 18, 20, 7, 12, 16, 19, 20, 5, 9, 13, 16, 18, 20]
    i = [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    c = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    Z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)
    return x, i, c, Z


def test_nhpp_regression_information_criteria():
    x, i, c, Z = _regression_data()
    model = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, baseline=CrowAMSAA)
    k = model._mle.size
    n = model._n_obs
    ll = model.log_likelihood
    assert np.isclose(ll, -float(model._neg_ll(model._mle)))
    assert np.isclose(model.aic(), 2 * k - 2 * ll)
    assert np.isclose(model.bic(), k * np.log(n) - 2 * ll)
    # Base-rate parameters first, then one coefficient per covariate column.
    assert model.parameter_names == ["alpha", "beta", "coef_0"]
    assert np.all(np.isfinite(model.standard_errors()))


def test_hpp_regression_information_criteria():
    x, i, c, Z = _regression_data()
    model = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
    ll = model.log_likelihood
    # ``_mle`` is in natural (rate) space; ``_neg_ll`` must agree there.
    assert np.isclose(ll, -float(model._neg_ll(model._mle)))
    assert model.parameter_names == ["lambda", "coef_0"]
    assert np.isfinite(model.aic()) and np.isfinite(model.bic())
    assert np.all(np.isfinite(model.standard_errors()))


def test_hpp_param_cb_matches_analytic():
    # For an HPP with n events observed to the last event, the observed
    # information gives se = lambda / sqrt(n), so the log-Wald bounds are
    # exactly lambda * exp(+/- z / sqrt(n)).
    x = exponential_event_times()
    model = HPP.fit(x)
    lam, n = model.params[0], len(x)
    z = norm.ppf(0.975)
    expected = lam * np.exp(np.array([-1.0, 1.0]) * z / np.sqrt(n))
    assert np.allclose(model.param_cb("lambda"), expected, rtol=1e-4)


@pytest.mark.parametrize("dist", [HPP, CrowAMSAA, Duane])
def test_param_cb_brackets_mle_and_respects_support(dist):
    x = exponential_event_times()
    model = dist.fit(x)
    for name, (lo, hi), p_hat in zip(
        model.parameter_names, model._parameter_bounds(), model._mle
    ):
        lower, upper = model.param_cb(name)
        assert lower < p_hat < upper
        if lo is not None:
            assert lower > lo
        if hi is not None:
            assert upper < hi
        # One-sided bounds are single values on the matching side of the MLE,
        # and less extreme than the two-sided ones at the same alpha_ci.
        (lower_1s,) = model.param_cb(name, bound="lower")
        (upper_1s,) = model.param_cb(name, bound="upper")
        assert lower < lower_1s < p_hat < upper_1s < upper


def test_param_cb_unknown_name_raises():
    model = CrowAMSAA.fit(exponential_event_times())
    with pytest.raises(ValueError, match="Unknown parameter"):
        model.param_cb("nope")


def test_param_cb_requires_likelihood():
    model = CrowAMSAA.from_params([1000.0, 1.2])
    with pytest.raises(ValueError, match="fitted from data"):
        model.param_cb("alpha")


def test_hpp_cif_cb_matches_analytic():
    # cif = lambda * x, so the delta-method se is x * se(lambda) and the
    # log-transformed band is cif * exp(+/- z * se(lambda) / lambda) -- the
    # relative width is constant in x.
    x = exponential_event_times()
    model = HPP.fit(x)
    lam, n = model.params[0], len(x)
    z = norm.ppf(0.975)
    t = np.array([100.0, 500.0])
    expected = (lam * t)[:, None] * np.exp(
        np.array([-1.0, 1.0]) * z / np.sqrt(n)
    )
    assert np.allclose(model.cif_cb(t), expected, rtol=1e-4)


@pytest.mark.parametrize("dist", [HPP, CrowAMSAA, Duane])
def test_cif_cb_brackets_cif(dist):
    x = exponential_event_times()
    model = dist.fit(x)
    t = np.linspace(0.0, x.max(), 25)
    cb = model.cif_cb(t)
    cif = model.cif(t)
    assert cb.shape == (t.size, 2)
    # The band starts as a point at the origin (cif(0) == 0) and brackets
    # the fitted curve everywhere else.
    assert np.all(cb[0] == 0.0)
    assert np.all(cb[1:, 0] < cif[1:])
    assert np.all(cb[1:, 1] > cif[1:])
    assert np.all(cb >= 0.0)
    # One-sided bounds match the corresponding side's shape.
    assert model.cif_cb(t, bound="lower").shape == t.shape
    assert model.cif_cb(t, bound="upper").shape == t.shape


def test_cif_cb_requires_likelihood():
    model = CrowAMSAA.fit(exponential_event_times(), how="MSE")
    with pytest.raises(ValueError, match="fitted from data"):
        model.cif_cb([1.0, 2.0])


def test_plot_confidence_band():
    model = CrowAMSAA.fit(exponential_event_times())
    # The band is drawn as a fill_between collection for MLE fits...
    ax = model.plot()
    assert len(ax.collections) == 1
    plt.close("all")
    # ...can be turned off...
    ax = model.plot(plot_bounds=False)
    assert len(ax.collections) == 0
    plt.close("all")
    # ...and is skipped (not an error) for MSE fits with no likelihood.
    mse_model = CrowAMSAA.fit(exponential_event_times(), how="MSE")
    ax = mse_model.plot()
    assert len(ax.collections) == 0
    plt.close("all")


def test_regression_param_cb():
    x, i, c, Z = _regression_data()
    model = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, baseline=CrowAMSAA)
    # Positive base-rate parameter: log-Wald bounds stay positive.
    lower, upper = model.param_cb("alpha")
    assert 0 < lower < model.params[0] < upper
    # Unbounded coefficient: plain Wald bounds are symmetric about the MLE.
    cb = model.param_cb("coef_0")
    assert np.isclose(cb.mean(), model.coeffs[0])
    assert cb[0] < model.coeffs[0] < cb[1]


def test_regression_cif_cb_brackets_cif():
    x, i, c, Z = _regression_data()
    Z_0 = np.array([0.5])
    t = np.array([5.0, 10.0, 20.0])
    for model in (
        ProportionalIntensityHPP.fit(x, Z, i=i, c=c),
        ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, baseline=CrowAMSAA),
    ):
        cb = model.cif_cb(t, Z_0)
        cif = model.cif(t, Z_0)
        assert cb.shape == (t.size, 2)
        assert np.all(cb[:, 0] < cif) and np.all(cif < cb[:, 1])
        assert np.all(cb > 0)
        ax = model.plot()
        assert len(ax.collections) == 1
        plt.close("all")


def test_pi_hpp_model_functions_delegate_to_constant_baseline():
    # The PI-HPP fitted model's dist used to be a bare namespace with only a
    # name, so cif/iif/inv_cif (and anything built on them) raised.
    x, i, c, Z = _regression_data()
    model = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
    Z_0 = np.array([0.5])
    lam = model.params[0]
    phi = np.exp(model.coeffs @ Z_0)
    assert np.allclose(
        model.cif([2.0, 4.0], Z_0), lam * phi * np.array([2.0, 4.0])
    )
    assert np.allclose(model.iif([2.0, 4.0], Z_0), lam * phi)
    assert np.allclose(model.inv_cif(model.cif([3.0], Z_0), Z_0), [3.0])
    assert model.dist.name == "Constant"


def test_renewal_param_cb():
    # The renewal models share the same mixin; the restoration parameter's
    # bounds flow through so its confidence bounds respect the support.
    true = GeneralizedRenewal.fit_from_parameters([10, 2.5], 0.3, dist=Weibull)
    data = true.count_terminated_simulation_data(10, items=6, random_state=3)
    model = GeneralizedRenewal.fit_from_recurrent_data(data)
    assert model.parameter_names == ["q", "alpha", "beta"]
    assert model._parameter_bounds() == [(0, None), (0, None), (0, None)]
    lower, upper = model.param_cb("q")
    assert 0 < lower < model.q < upper
    lower, upper = model.param_cb("alpha")
    assert 0 < lower < model.model.params[0] < upper


# ---------------------------------------------------------------------------
# One BIC sample size, the observed events, for every recurrent
# model.
# ---------------------------------------------------------------------------


def test_bic_counts_observed_events_only():
    n_events = sum(ci == 0 for ci in REPAIR_FLEET_C)
    models = [
        HPP.fit(REPAIR_FLEET_X, REPAIR_FLEET_I, REPAIR_FLEET_C),
        CrowAMSAA.fit(REPAIR_FLEET_X, REPAIR_FLEET_I, REPAIR_FLEET_C),
        GeneralizedOneRenewal.fit(
            REPAIR_FLEET_X, REPAIR_FLEET_I, REPAIR_FLEET_C
        ),
        ARI.fit(REPAIR_FLEET_X, REPAIR_FLEET_I, REPAIR_FLEET_C, m=1),
    ]
    for model in models:
        k = model._mle.size
        expected = k * np.log(n_events) - 2 * model.log_likelihood
        assert model.bic() == pytest.approx(expected)


def test_bic_counts_interval_events():
    # Only interval counts: the five events they hold are observed events,
    # so BIC's sample size is 5 (it used to count exact events only, and
    # was NaN here rather than log(0) = -inf).
    model = HPP.fit([[0, 10], [10, 20]], c=[2, 2], n=[2, 3])
    assert model.bic() == pytest.approx(np.log(5) - 2 * model.log_likelihood)
    assert np.isfinite(model.aic())


def test_572_aic_and_bic_are_methods():
    # They were properties here and methods everywhere else; the property
    # spelling, deprecated in v0.23, is gone in v0.24.
    import warnings

    x = np.cumsum(np.random.default_rng(0).exponential(10, 20))
    model = CrowAMSAA.fit(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aic, bic = model.aic(), model.bic()
        assert type(aic) is float and type(bic) is float
        assert model.neg_ll() == -model.log_likelihood
    with pytest.raises(TypeError):
        model.aic + 1
    with pytest.raises(TypeError):
        model.bic < bic + 1


# ----------------------------------------------------------------------------
# #578: bounds on the intensity and the (demonstrated) MTBF
# ----------------------------------------------------------------------------

_GROWTH_X = [40, 110, 210, 340, 500, 690, 920, 1180, 1480, 1800, 2000]
_GROWTH_C = [0] * 10 + [1]


def test_578_iif_cb_is_the_delta_method_on_log_iif():
    # log iif(T) = log(beta) - beta log(alpha) + (beta - 1) log(T): its
    # gradient is (-beta / alpha, 1 / beta - log(alpha) + log(T)).
    model = CrowAMSAA.fit(_GROWTH_X, c=_GROWTH_C)
    alpha, beta = model.params
    T = 2000.0
    grad = np.array([-beta / alpha, 1 / beta - np.log(alpha) + np.log(T)])
    se = np.sqrt(grad @ model.covariance() @ grad)
    z = norm.ppf(0.9)
    expected = model.iif(T) * np.exp(np.array([-1.0, 1.0]) * z * se)
    np.testing.assert_allclose(
        model.iif_cb(T, alpha_ci=0.2), expected, rtol=1e-6
    )
    np.testing.assert_allclose(
        model.iif_cb(T, alpha_ci=0.1, bound="lower"), expected[0], rtol=1e-6
    )


@pytest.mark.parametrize("dist", [HPP, CrowAMSAA, Duane, CoxLewis])
def test_578_mtbf_cb_is_the_reciprocal_of_iif_cb(dist):
    # A lower bound on the MTBF is 1 / the upper bound on the intensity.
    model = dist.fit(exponential_event_times())
    t = np.array([50.0, 300.0, 900.0])
    np.testing.assert_allclose(model.mtbf(t), 1 / model.iif(t))
    iif = model.iif_cb(t, alpha_ci=0.1)
    np.testing.assert_allclose(
        model.mtbf_cb(t, alpha_ci=0.1), 1 / iif[:, ::-1]
    )
    np.testing.assert_allclose(
        model.mtbf_cb(t, alpha_ci=0.1, bound="lower"),
        1 / model.iif_cb(t, alpha_ci=0.1, bound="upper"),
    )
    lower, upper = model.mtbf_cb(t, alpha_ci=0.1).T
    assert np.all(lower < model.mtbf(t)) and np.all(model.mtbf(t) < upper)


def _time_terminated_cdf(n, z):
    # P(N <= n | z), P(N = k | z) = z^k / (k! (k - 1)!) / (sqrt(z)
    # I_1(2 sqrt(z))), written out with the plain Bessel function.
    from math import factorial

    from scipy.special import iv

    terms = [z**k / (factorial(k) * factorial(k - 1)) for k in range(1, n + 1)]
    return sum(terms) / (np.sqrt(z) * iv(1, 2 * np.sqrt(z)))


@pytest.mark.parametrize("n", [1, 2, 5, 12, 40])
def test_578_crow_time_terminated_coefficients(n):
    from surpyval.recurrent.parametric.crow_amsaa import (
        crow_time_terminated_coefficients,
    )

    L, U = crow_time_terminated_coefficients(n, 0.1)
    # L = n^2 / z_U with P(N <= n | z_U) = 0.1; U = n^2 / z_L with
    # P(N >= n | z_L) = 0.1 (infinite for n = 1).
    assert _time_terminated_cdf(n, n**2 / L) == pytest.approx(0.1, rel=1e-8)
    if n == 1:
        assert U == np.inf
    else:
        p_at_least = 1 - _time_terminated_cdf(n - 1, n**2 / U)
        assert p_at_least == pytest.approx(0.1, rel=1e-8)
    assert 0 < L < 1 < U


@pytest.mark.parametrize("n", [2, 3, 8, 30])
def test_578_crow_failure_terminated_coefficients(n):
    from scipy.integrate import quad
    from scipy.special import gammainc
    from scipy.stats import gamma

    from surpyval.recurrent.parametric.crow_amsaa import (
        crow_failure_terminated_coefficients,
    )

    L, U = crow_failure_terminated_coefficients(n, 0.1)

    # M_hat / M = Z G / n^2, Z ~ Gamma(n), G ~ Gamma(n - 1): its CDF,
    # integrating over Z this time.
    def cdf(r):
        return quad(
            lambda z: gammainc(n - 1, r * n**2 / z) * gamma.pdf(z, n),
            0,
            np.inf,
        )[0]

    assert cdf(1 / L) == pytest.approx(0.9, abs=1e-7)
    assert cdf(1 / U) == pytest.approx(0.1, abs=1e-7)


def test_578_crow_coefficients_approach_the_normal_approximation():
    # MIL-HDBK-189C's large-N approximation: L, U ~ (1 -/+ z / sqrt(2N))^-2
    # with z the one-sided normal quantile; both designs tend to it.
    from surpyval.recurrent.parametric.crow_amsaa import (
        crow_failure_terminated_coefficients,
        crow_time_terminated_coefficients,
    )

    n, z = 2000, norm.ppf(0.95)
    approx = [(1 + z / np.sqrt(2 * n)) ** -2, (1 - z / np.sqrt(2 * n)) ** -2]
    for coefficients in (
        crow_time_terminated_coefficients(n, 0.05),
        crow_failure_terminated_coefficients(n, 0.05),
    ):
        np.testing.assert_allclose(coefficients, approx, rtol=3e-3)


def test_578_crow_mtbf_cb_time_terminated_fleet():
    # Three prototypes, each tested to T = 2000 h: the demonstrated MTBF of
    # one prototype is 3 T / (N beta_hat), with beta_hat = N / sum log(T/t).
    from surpyval.recurrent.parametric.crow_amsaa import (
        crow_time_terminated_coefficients,
    )

    rng = np.random.default_rng(578)
    T, beta, x, i, c = 2000.0, 0.55, [], [], []
    for unit in range(3):
        n = rng.poisson(17)
        times = np.sort(T * rng.uniform(size=n) ** (1 / beta))
        x += [*times, T]
        i += [unit] * (n + 1)
        c += [0] * n + [1]
    events = np.array(x)[np.array(c) == 0]
    N = events.size
    beta_hat = N / np.log(T / events).sum()
    m_hat = 3 * T / (N * beta_hat)
    model = CrowAMSAA.fit(x, i=i, c=c)
    assert model.mtbf(T) == pytest.approx(m_hat, rel=1e-4)
    L, U = crow_time_terminated_coefficients(N, 0.1)
    cb = model.mtbf_cb(T, alpha_ci=0.2, method="crow")
    np.testing.assert_allclose(cb, model.mtbf(T) * np.array([L, U]))
    lower = model.mtbf_cb(T, alpha_ci=0.1, bound="lower", method="crow")
    assert lower == cb[0]
    np.testing.assert_allclose(
        model.iif_cb(T, alpha_ci=0.2, method="crow"), 1 / cb[::-1]
    )


def test_578_crow_mtbf_cb_failure_terminated():
    from surpyval.recurrent.parametric.crow_amsaa import (
        crow_failure_terminated_coefficients,
    )

    x = np.array(_GROWTH_X[:-1], dtype=float)  # observed to the 10th failure
    model = CrowAMSAA.fit(x)
    beta_hat = 10 / np.log(x[-1] / x[:-1]).sum()
    assert model.mtbf(x[-1]) == pytest.approx(
        x[-1] / (10 * beta_hat), rel=1e-4
    )
    L, U = crow_failure_terminated_coefficients(10, 0.025)
    np.testing.assert_allclose(
        model.mtbf_cb(x[-1], method="crow"), model.mtbf(x[-1]) * np.r_[L, U]
    )


def test_578_crow_refuses_data_without_an_exact_bound():
    T = 2000.0
    model = CrowAMSAA.fit(_GROWTH_X, c=_GROWTH_C)
    with pytest.raises(ValueError, match="end of the test, x = 2000"):
        model.mtbf_cb(1500.0, method="crow")
    with pytest.raises(ValueError, match="'method' must be one of"):
        model.mtbf_cb(T, method="exact")
    with pytest.raises(ValueError, match="CrowAMSAA model"):
        Duane.fit(_GROWTH_X, c=_GROWTH_C).mtbf_cb(T, method="crow")
    delayed = CrowAMSAA.fit(_GROWTH_X, c=_GROWTH_C, tl=10.0)
    with pytest.raises(ValueError, match="delayed entry"):
        delayed.mtbf_cb(T, method="crow")
    two_ends = CrowAMSAA.fit(
        _GROWTH_X + [100, 900, 1500],
        i=[1] * 11 + [2] * 3,
        c=_GROWTH_C + [0, 0, 1],
    )
    with pytest.raises(ValueError, match="does not end at one time"):
        two_ends.mtbf_cb(T, method="crow")
    with pytest.raises(ValueError, match="fitted from data"):
        CrowAMSAA.fit(_GROWTH_X, c=_GROWTH_C, how="MSE").mtbf_cb(
            T, method="crow"
        )


def test_578_regression_iif_cb_brackets_iif():
    x = [3, 8, 12, 15, 20, 4, 6, 9, 11, 13, 20]
    i = [1] * 5 + [2] * 6
    c = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    Z = [[0.0]] * 5 + [[1.0]] * 6
    model = ProportionalIntensityNHPP.fit(x, Z, i, c)
    t = np.array([2.0, 10.0, 18.0])
    cb = model.iif_cb(t, [1.0])
    iif = model.iif(t, [1.0])
    assert cb.shape == (3, 2)
    assert np.all(cb[:, 0] < iif) and np.all(iif < cb[:, 1])
    # The intensity is the derivative of the CIF, scaled alike by exp(Z b):
    # the relative width does not depend on Z for the HPP baseline.
    hpp = ProportionalIntensityHPP.fit(x, Z, i, c)
    ratio = hpp.iif_cb(t, [0.0]) / hpp.iif(t, [0.0])[:, None]
    np.testing.assert_allclose(ratio, ratio[0] * np.ones((3, 1)))
