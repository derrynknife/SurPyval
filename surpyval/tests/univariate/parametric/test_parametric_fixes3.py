"""Regression tests for the third review of the univariate parametric area.

Each test names the bug it pins down and failed on the code before the fix,
apart from a few parametrised cases that are controls (unit scale, the
unchanged exact-and-right-censored BIC) and the MSE fallback test, which
guards the scale-invariance change itself.
"""

import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval.univariate.parametric import parametric as parametric_module

W, E, G = surv.Weibull, surv.Exponential, surv.Geometric


def _no_warnings(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args, **kwargs)


# -- conditional survival follows the model's own sf ------------------------


def test_cs_counts_the_limited_failure_proportion():
    model = W.from_params([10, 2], p=0.7)
    assert model.cs(5, 10) == pytest.approx(model.sf(15) / model.sf(10))
    assert model.cs(5, 10) == pytest.approx(0.670437654623545)


def test_cs_counts_the_zero_inflation_fraction():
    model = W.from_params([10, 2], p=0.8, f0=0.2)
    assert model.cs(3, 1) == pytest.approx(model.sf(4) / model.sf(1))


@pytest.mark.parametrize(
    "dist, params", [(W, [10, 2]), (W, [10, 1.5]), (surv.Gamma, [2.0, 0.5])]
)
def test_cs_before_the_offset(dist, params):
    model = dist.from_params(params, gamma=5)
    value = _no_warnings(model.cs, 5, 2)
    assert value == pytest.approx(model.sf(7) / model.sf(2))
    assert value < 1


# -- likelihood-ratio bounds ------------------------------------------------


def test_lr_band_does_not_collapse_onto_the_estimate():
    np.random.seed(1000)
    model = G.fit(G.random(30, 0.3))
    lower, upper = model.cb([2.0], method="lr")[0]
    estimate = float(model.sf(2.0))
    assert lower < estimate - 0.05 < estimate < upper
    # The band's lower sf is the sf at the parameter's upper LR bound
    p_hi = model.param_cb("p", method="lr")[1]
    assert lower == pytest.approx((1 - p_hi) ** 2, rel=1e-3)


def test_lr_band_is_nan_with_a_warning_when_every_search_fails(monkeypatch):
    np.random.seed(1)
    model = W.fit(W.random(30, 10, 3))

    class Failed:
        success = False
        x = np.array([np.nan, np.nan])
        fun = np.nan

    monkeypatch.setattr(
        parametric_module, "minimize", lambda *a, **k: Failed()
    )
    with pytest.warns(RuntimeWarning, match="could not be found"):
        band = model.cb([5.0, 10.0], method="lr")
    assert np.isnan(band).all()


def test_lr_param_bound_is_nan_with_a_warning_when_unsolved(monkeypatch):
    np.random.seed(1)
    model = W.fit(W.random(30, 10, 3))
    monkeypatch.setattr(model, "_profile_neg_ll", lambda idx, v: np.nan)
    with pytest.warns(RuntimeWarning, match="could not be found"):
        bound = model.param_cb("beta", method="lr")
    assert np.isnan(bound).all()


def test_lr_bounds_after_restoring_with_the_data():
    np.random.seed(1)
    model = W.fit(W.random(30, 10, 3))
    restored = surv.from_dict(model.to_dict(with_data=True))
    assert np.allclose(
        restored.cb([5.0, 10.0], method="lr"),
        model.cb([5.0, 10.0], method="lr"),
    )
    assert np.allclose(
        restored.param_cb("beta", method="lr"),
        model.param_cb("beta", method="lr"),
    )


def test_param_cb_on_a_fixed_parameter_agrees_between_methods():
    model = W.fit([1.0, 2, 3, 4, 5, 6], fixed={"beta": 2.0})
    assert np.array_equal(model.param_cb("beta"), [2.0, 2.0])
    assert np.array_equal(model.param_cb("beta", method="lr"), [2.0, 2.0])
    assert np.array_equal(
        model.param_cb("beta", bound="lower", method="lr"), [2.0]
    )


def test_cb_rejects_an_unknown_bound():
    model = W.fit([1.0, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="bound must be"):
        model.cb([2.0], bound="both")


# -- mixture EM ---------------------------------------------------------------


def test_mixture_em_on_interval_data_reaches_the_optimum():
    np.random.seed(0)
    x = np.concatenate([W.random(300, 5, 3), W.random(300, 30, 4)])
    mm = surv.MixtureModel(W, 2)
    _no_warnings(mm.fit, xl=np.floor(x), xr=np.floor(x) + 1)
    truth = mm.neg_ll_of(np.array([0.5, 0.5]), np.array([[5, 3], [30, 4.0]]))
    # It stalled 114 units above the truth's negative log-likelihood
    assert mm.loglike <= truth + 1e-6


def test_geometric_mixture_fits_without_warnings():
    np.random.seed(0)
    x = np.concatenate([G.random(300, 0.5), G.random(300, 0.05)])
    mm = surv.MixtureModel(G, 2)
    _no_warnings(mm.fit, x)
    assert sorted(mm.params.ravel()) == pytest.approx([0.05, 0.5], abs=0.03)


def test_restored_mixture_needs_its_data_for_plots_and_takes_lists_in_cs():
    x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
    mm = surv.MixtureModel(W, 2)
    mm.fit(x)
    restored = surv.from_dict(mm.to_dict())
    assert np.allclose(restored.cs([1, 2], 5), mm.cs(np.array([1, 2]), 5))
    for method in (restored.plot, restored.get_plot_data):
        with pytest.raises(ValueError, match="needs the data"):
            method()


# -- moments ------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1e-4, 1.0, 1e4])
def test_expo_weibull_moments_at_any_scale(alpha):
    EW = surv.ExpoWeibull
    mean = _no_warnings(EW.mean, alpha, 2.0, 1.5)
    assert mean == pytest.approx(alpha * 1.0394154617791786, rel=1e-9)
    m2 = _no_warnings(EW.moment, 2, alpha, 2.0, 1.5)
    assert m2 == pytest.approx(alpha**2 * EW.moment(2, 1.0, 2.0, 1.5))
    entropy = _no_warnings(EW.entropy, alpha, 2.0, 1.5)
    assert entropy == pytest.approx(
        EW.entropy(1.0, 2.0, 1.5) + np.log(alpha), abs=1e-9
    )


def test_expo_weibull_moments_match_the_weibull_at_mu_one():
    assert surv.ExpoWeibull.moment(3, 7.0, 1.3, 1.0) == pytest.approx(
        W.moment(3, 7.0, 1.3), rel=1e-10
    )


def _custom_weibull(name="CustomW3"):
    def Hf(x, *params):
        return (x / params[0]) ** params[1]

    return surv.CustomDistribution(
        name, Hf, ["a", "b"], ((0, None), (0, None)), (0, np.inf)
    )


@pytest.mark.parametrize("alpha", [1e-4, 10.0, 1e5])
def test_custom_distribution_moments_at_any_scale(alpha):
    C = _custom_weibull()
    assert _no_warnings(C.mean, alpha, 2.0) == pytest.approx(
        W.mean(alpha, 2.0), rel=1e-9
    )
    assert C.moment(2, alpha, 2.0) == pytest.approx(
        W.moment(2, alpha, 2.0), rel=1e-9
    )


def test_custom_distribution_has_a_quantile_and_random():
    C = _custom_weibull()
    u = np.array([0.1, 0.5, 0.9])
    assert np.allclose(C.qf(u, 10.0, 2.0), W.qf(u, 10.0, 2.0))
    model = C.fit([1.0, 2, 3, 4, 5])
    np.random.seed(1)
    assert model.random(4).shape == (4,)


def test_beta_geometric_higher_moments_are_exact():
    BG = surv.BetaGeometric
    assert BG.moment(3, 5, 3) == pytest.approx(33.25, rel=1e-12)
    assert BG.moment(3, 4, 3) == pytest.approx(92.0, rel=1e-12)
    assert BG.moment(4, 3.5, 2) == np.inf


@pytest.mark.parametrize(
    "dist, params",
    [
        (G, (0.3,)),
        (surv.Poisson, (2.5,)),
        (surv.NegativeBinomial, (3.0, 0.4)),
        (surv.BetaGeometric, (5.0, 3.0)),
    ],
)
def test_discrete_first_moment_is_the_mean(dist, params):
    assert dist.moment(1, *params) == dist.mean(*params)


def test_discrete_moments_are_exact():
    assert G.moment(2, 0.2) == pytest.approx(45.0, rel=1e-14)
    assert G.moment(3, 0.3) == pytest.approx(158.88888888888889, rel=1e-14)
    assert surv.Poisson.moment(3, 2.5) == pytest.approx(36.875, rel=1e-14)
    assert surv.NegativeBinomial.moment(2, 3.0, 0.4) == pytest.approx(
        41.5, rel=1e-14
    )


# -- MPS and MSE are scale invariant ---------------------------------------


@pytest.mark.parametrize("how", ["MPS", "MSE"])
@pytest.mark.parametrize("k", [1e3, 1e-3])
def test_mps_and_mse_are_scale_invariant(how, k):
    np.random.seed(3)
    x = W.random(200, 10, 2)
    base = W.fit(x, how=how).params
    scaled = W.fit(x * k, how=how).params
    assert scaled / [k, 1] == pytest.approx(base, rel=1e-5)


def test_mse_keeps_the_better_optimum_across_its_fallbacks():
    np.random.seed(3)
    x = surv.Normal.random(200, 5.0, 2.0)
    cens = float(np.quantile(x, 0.85))
    c = (x > cens).astype(int)
    xc = np.where(x > cens, cens, x)
    base = surv.Normal.fit(xc, c, how="MSE").params
    small = surv.Normal.fit(xc * 1e-3, c, how="MSE").params
    assert small * 1e3 == pytest.approx(base, rel=1e-5)


# -- information criteria ---------------------------------------------------


def test_bic_is_finite_without_exact_failures():
    model = W.fit(xl=np.arange(1, 20), xr=np.arange(2, 21))
    bic = _no_warnings(model.bic)
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
        assert np.isnan(_no_warnings(model.aic_c))
    else:
        # k = 2, N = 3: N = k + 1
        assert np.isnan(_no_warnings(model.aic_c))
    assert not model.aic_c() < model.aic()


# -- offsets and supports -----------------------------------------------------


def test_exponential_mpp_offset_stays_below_the_first_failure():
    np.random.seed(1)
    x = E.random(500, 0.1) + 5
    model = E.fit(x, offset=True, how="MPP")
    assert model.gamma < x.min()
    assert np.isfinite(model.neg_ll())


def test_log_density_is_minus_infinity_outside_the_support():
    assert E.log_df(-1.0, 0.1) == -np.inf
    assert W.log_df(-1.0, 10.0, 2.0) == -np.inf
    assert surv.Beta.log_df(1.5, 2.0, 3.0) == -np.inf


@pytest.mark.parametrize(
    "dist, params, below, above",
    [
        (E, (0.1,), -1.0, None),
        (W, (10.0, 2.5), -1.0, None),
        (surv.Gamma, (2.0, 1.0), -1.0, None),
        (surv.LogNormal, (0.0, 1.0), -1.0, None),
        (surv.LogLogistic, (1.0, 2.0), -1.0, None),
        (surv.Rayleigh, (1.0,), -1.0, None),
        (surv.ExpoWeibull, (1.0, 2.0, 0.5), -1.0, None),
        (surv.Beta, (2.0, 3.0), -0.5, 1.5),
        (surv.Beta4, (2.0, 3.0, 0.0, 4.0), -1.0, 5.0),
        (surv.Uniform, (0.0, 4.0), -1.0, 5.0),
    ],
)
def test_raw_functions_outside_the_support(dist, params, below, above):
    def at(x):
        return [
            float(np.real(getattr(dist, f)(x, *params)))
            for f in ("sf", "ff", "df", "hf", "Hf")
        ]

    assert _no_warnings(at, below) == [1.0, 0.0, 0.0, 0.0, 0.0]
    if above is not None:
        assert _no_warnings(at, above) == [0.0, 1.0, 0.0, np.inf, np.inf]


def test_discretized_mass_is_zero_below_the_support():
    assert surv.Discretize(W).df(0.0, 10.0, 2.0) == 0.0


def test_right_censoring_below_the_support_does_not_break_the_fit():
    # A unit censored before the support starts carries no information,
    # R = 1; its log-survival was nan and the fit fell back to its start
    # with an "MLE Failed" warning.
    np.random.seed(2)
    x = W.random(100, 10, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = W.fit(np.append(x, -1.0), np.append(np.zeros(100), 1))
    assert model.params == pytest.approx(W.fit(x).params, rel=1e-6)


def test_zero_inflation_mass_arrives_at_zero():
    for dist, params in ((W, [10, 2]), (G, [0.3])):
        model = dist.from_params(params, f0=0.1)
        assert model.ff(-1) == 0.0
        assert model.sf(-1) == 1.0
        assert model.ff(0) == pytest.approx(0.1)


# -- validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: W.from_params([10, 2], p=1.5), "must be in"),
        (lambda: W.from_params([10, 2], f0=-0.1), "must be in"),
        (lambda: W.from_params([10, 2], p=0.3, f0=0.4), "less than p"),
        (lambda: surv.Normal.from_params([1, 2], f0=0.1), "starting at 0"),
        (lambda: surv.Beta4.from_params([2, 3, 5, 1]), "a < b"),
        (lambda: surv.Uniform.from_params([4, 1]), "a < b"),
    ],
)
def test_from_params_validates(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    "dist, x",
    [
        (G, [1.5, 2.2, 3.7, 1.1]),
        (surv.Poisson, [-0.5, 1, 2, 3]),
        (surv.Discretize(W), [1, 2, 2.5, 3]),
    ],
)
def test_discrete_fits_refuse_non_integer_data(dist, x):
    with pytest.raises(ValueError, match="whole numbers"):
        dist.fit(x)


def test_exact_event_time_refuses_contradictory_checks():
    with pytest.raises(ValueError, match="contradict"):
        surv.ExactEventTime.fit([5, 3], [1, -1])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(fixed={"shape": 1}), "Unknown parameter 'shape'"),
        (dict(fixed={"p": 0.5}), "needs lfp=True"),
        (dict(fixed={"gamma": 0.5}), "needs offset=True"),
        (dict(fixed={"alpha": -1}), "Cannot fix alpha"),
        (dict(lfp=True, fixed={"p": 1.5}), "Cannot fix p"),
        (dict(offset=True, fixed={"gamma": 1.5}), "Cannot fix gamma"),
        (dict(init=[1.0]), "`init` has 1 value"),
        (dict(init=[-1.0, 2.0]), "Bad `init`: alpha"),
    ],
)
def test_fixed_and_init_are_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        W.fit([1.0, 2, 3, 4, 5], **kwargs)


def test_init_for_the_free_parameters_only_still_works():
    model = W.fit([1.0, 2, 3, 4, 5], fixed={"alpha": 3.0}, init=[2.0])
    assert model.params[0] == 3.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: surv.Beta.fit([0.2, 0.3, 0.5, 1.5], [0, 0, 0, 1]),
        lambda: surv.Poisson.fit([-2, 1, 2, 3]),
    ],
)
def test_out_of_support_data_is_refused(call):
    with pytest.raises(ValueError, match="outside the support"):
        call()


def test_support_message_is_one_clear_line():
    with pytest.raises(ValueError) as err:
        W.fit([0.0, 1, 2, 3])
    message = str(err.value)
    assert message.startswith("Some of your data")
    assert "(0, inf)" in message and "[0, inf]" not in message


def test_fit_from_ecdf_refuses_distributions_without_a_line():
    for dist in (surv.Gamma, E, surv.Beta):
        with pytest.raises(ValueError, match="cannot be fitted to an ECDF"):
            dist.fit_from_ecdf([1, 2, 3, 4], [0.1, 0.3, 0.6, 0.9])


def test_fit_best_checks_distribution_names():
    x = [1.0, 2, 3, 4, 5, 6]
    assert surv.fit_best(x, include=["weibull"]).dist.name == "Weibull"
    assert surv.fit_best(x, include="Weibull").dist.name == "Weibull"
    with pytest.raises(ValueError, match="Unknown distribution"):
        surv.fit_best(x, include=["Weibul"])
    with pytest.raises(ValueError, match="Unknown distribution"):
        surv.fit_best(x, exclude=["Geometric"])


# -- Uniform MLE ------------------------------------------------------------


def test_uniform_accepts_truncation_beyond_the_data():
    model = surv.Uniform.fit([1.0, 2, 3, 4], tr=100)
    assert np.allclose(model.params, [1, 4])


def test_uniform_censored_at_the_maximum_has_the_closed_form():
    # N = 4 units, k = 1 censored at r = 5, a = 1: b = (N r - k a)/(N - k)
    model = surv.Uniform.fit([1.0, 2, 3, 5], [0, 0, 0, 1])
    assert model.params == pytest.approx([1.0, 19 / 3], rel=1e-8)
    model = surv.Uniform.fit([1.0, 2, 3, 5], [0, 0, 0, 1], tr=100)
    assert model.params == pytest.approx([1.0, 19 / 3], rel=1e-8)


@pytest.mark.parametrize("seed", [0, 134, 287])
def test_uniform_censored_mle_reaches_the_optimum(seed):
    # L-BFGS-B stopped up to 1% short here (b = 10.10 for 9.98)
    np.random.seed(seed)
    x = np.sort(surv.Uniform.random(100, 0, 10))
    n = len(x)
    c = np.zeros(n)
    c[-1] = 1
    b_hat = (n * x[-1] - x[0]) / (n - 1)
    assert surv.Uniform.fit(x, c).params == pytest.approx(
        [x[0], b_hat], rel=1e-12
    )
    c = np.zeros(n)
    c[0] = -1
    a_hat = (n * x[0] - x[-1]) / (n - 1)
    assert surv.Uniform.fit(x, c).params == pytest.approx(
        [a_hat, x[-1]], rel=1e-10, abs=1e-12
    )


def test_uniform_still_refuses_where_no_mle_exists():
    with pytest.raises(ValueError, match="no unique MLE"):
        surv.Uniform.fit([1.0, 2, 3, 5], [0, 0, 0, 1], tr=5.5)


# -- serialisation ----------------------------------------------------------


def test_discretize_round_trips():
    model = surv.Discretize(W).fit([1, 2, 2, 3, 4, 5, 3, 2])
    restored = surv.from_dict(model.to_dict())
    assert restored.dist.name == "Discretize(Weibull)"
    assert np.allclose(restored.sf([1, 3, 5]), model.sf([1, 3, 5]))


def test_custom_distribution_round_trips_through_its_registry():
    C = _custom_weibull("RoundTripW")
    model = C.fit([1.0, 2, 3, 4, 5])
    d = model.to_dict()
    assert d["custom"] is True
    restored = surv.from_dict(d)
    assert restored.dist is C
    assert np.allclose(restored.sf([2, 4]), model.sf([2, 4]))


def test_unregistered_custom_distribution_gives_a_clear_error():
    C = _custom_weibull("NeverRegisteredAgain")
    d = C.fit([1.0, 2, 3, 4, 5]).to_dict()
    d["distribution"] = "SomeOtherCustom"
    with pytest.raises(ValueError, match="Construct it again"):
        surv.from_dict(d)


def test_restored_model_get_plot_data_needs_the_data():
    np.random.seed(1)
    model = W.fit(W.random(30, 10, 3))
    with pytest.raises(ValueError, match="needs the data"):
        surv.from_dict(model.to_dict()).get_plot_data()


def test_custom_distribution_may_name_a_parameter_p():
    def Hf(x, *params):
        return (x / params[0]) ** params[1]

    C = surv.CustomDistribution(
        "CustomWithP", Hf, ["a", "p"], ((0, None), (0, None)), (0, np.inf)
    )
    np.random.seed(2)
    x = W.random(200, 10, 2)
    c = np.zeros(200)
    never = np.random.uniform(size=200) > 0.7
    x[never], c[never] = 30, 1
    model = C.fit(x, c, lfp=True)
    reference = W.fit(x, c, lfp=True)
    assert model.params == pytest.approx(reference.params, rel=1e-4)
    assert model.p == pytest.approx(reference.p, rel=1e-4)


# -- warnings and attributes --------------------------------------------------


def test_lfp_and_zi_without_zeros_fit_quietly():
    _no_warnings(
        W.fit,
        [1.0, 2, 3, 4, 5, 6, 7, 8],
        [0, 0, 0, 0, 0, 1, 1, 1],
        lfp=True,
        zi=True,
    )


def test_binomial_hazard_beyond_n_is_quiet():
    hf = _no_warnings(surv.Binomial.hf, np.array([3.0, 6, 7]), 5, 0.3)
    assert hf[1:].tolist() == [0.0, 0.0]


def test_royston_parmar_edges():
    np.random.seed(1)
    model = surv.RoystonParmar.fit(W.random(50, 10, 2))
    assert _no_warnings(model.sf, -1.0) == 1.0
    assert _no_warnings(model.hf, 0.0) == 0.0
    assert _no_warnings(model.df, np.array([-1.0, 0.0]))[0] == 0.0


@pytest.mark.parametrize("how", ["MPP", "MOM", "MSE", "MPS"])
def test_every_fit_reports_its_optimizer(how):
    np.random.seed(1)
    model = W.fit(W.random(30, 10, 3), how=how)
    assert isinstance(model.optimizer, str) and model.optimizer


def test_closed_form_models_have_the_support_a_restored_one_has():
    for model in (
        surv.Bernoulli.fit([0, 1, 1, 0, 1]),
        surv.FixedEventProbability.fit([0, 1, 1]),
        surv.ExactEventTime.fit([2, 3, 5], [1, 1, -1]),
        surv.Binomial.from_params([5, 0.3]),
    ):
        restored = surv.from_dict(model.to_dict())
        assert np.array_equal(model.support, restored.support)
    assert np.array_equal(
        surv.from_dict(surv.Binomial.from_params([5, 0.3]).to_dict()).support,
        [-1, 6],
    )


def test_fixed_event_probability_mean_is_its_first_moment():
    model = surv.FixedEventProbability.from_params([0.3])
    assert model.mean() == pytest.approx(0.3)
    assert model.mean() == pytest.approx(model.moment(1))
