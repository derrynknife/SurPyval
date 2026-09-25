"""Regression tests for the second round of univariate parametric fixes."""

import warnings

import numpy as np
import pytest
from autograd import numpy as anp
from scipy.stats import gompertz

import surpyval as surv
from surpyval.univariate.parametric.fitters import fallback_minimize

# -- a distribution parameter named ``p`` (Geometric, NegativeBinomial) ---


def test_param_cb_on_a_distribution_parameter_named_p():
    np.random.seed(0)
    model = surv.Geometric.fit(surv.Geometric.random(200, 0.15))
    wald = model.param_cb("p")
    lr = model.param_cb("p", method="lr")
    p_hat = model.params[0]
    for bound in (wald, lr):
        assert bound[0] < p_hat < bound[1]
        assert 0 < bound[0] and bound[1] < 1


@pytest.mark.parametrize(
    "dist, x, c",
    [
        (surv.Geometric, [1, 2, 2, 3, 5, 8, 10], [0, 0, 0, 0, 0, 1, 1]),
        (
            surv.NegativeBinomial,
            [1, 2, 2, 3, 3, 4, 5, 6, 8, 10, 12, 12],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        ),
    ],
)
def test_lfp_fit_for_a_distribution_parameter_named_p(dist, x, c):
    model = dist.fit(x, c=c, lfp=True)
    assert model.lfp_name == "lfp_p"
    assert len(model.params) == dist.k
    assert 0 < model.p < 1
    # the two p's are distinct parameters with distinct bounds
    assert model.param_cb("p")[0] < model.params[dist.param_map["p"]]
    lfp_bound = model.param_cb("lfp_p")
    assert lfp_bound[0] < model.p < lfp_bound[1]
    assert "Max Proportion (lfp_p)" in repr(model)


def test_lfp_proportion_can_be_fixed_by_its_own_name():
    model = surv.Geometric.fit(
        [1, 2, 2, 3, 5, 8, 10],
        c=[0, 0, 0, 0, 0, 1, 1],
        lfp=True,
        fixed={"lfp_p": 0.8},
    )
    assert model.p == pytest.approx(0.8)
    # the Weibull's LFP proportion keeps its usual name
    weibull = surv.Weibull.fit(
        [1, 2, 3, 4, 5, 6], c=[0, 0, 0, 0, 1, 1], lfp=True, fixed={"p": 0.8}
    )
    assert weibull.lfp_name == "p"
    assert weibull.p == pytest.approx(0.8)


# -- offsets are refused for discrete distributions -----------------------


@pytest.mark.parametrize(
    "dist, x",
    [
        (surv.DiscreteWeibull, [6, 7, 7, 8, 9, 11]),
        (surv.Geometric, [6, 7, 7, 8, 9, 11]),
        (surv.Discretize(surv.Weibull), [6, 7, 7, 8, 9, 11]),
    ],
)
def test_discrete_distributions_cannot_be_offset(dist, x):
    with pytest.raises(ValueError, match="discrete distribution"):
        dist.fit(x, offset=True)


# -- Uniform MLE with censoring --------------------------------------------


def test_uniform_censored_mle_is_not_min_max():
    x, c = [0, 9.9, 9.9, 9.9, 10], [0, 1, 1, 1, 0]
    model = surv.Uniform.fit(x, c=c)
    # neg_ll = 2 log b - 3 log((b - 9.9) / b) is minimised at b = 24.75
    np.testing.assert_allclose(model.params, [0.0, 24.75], atol=1e-7)
    assert model.neg_ll() == pytest.approx(7.950127849, abs=1e-8)
    assert model.neg_ll() < 18.42


def test_uniform_censored_mle_matches_the_profile_solution():
    # A tie between an exact and a right-censored maximum: the censored
    # unit has zero probability at b = 10, so b = 40 / 3 (from
    # d/db [log(b - 10) - 4 log b] = 0 with a = 0).
    model = surv.Uniform.fit([0, 10, 10, 5], c=[0, 0, 1, 0])
    np.testing.assert_allclose(model.params, [0.0, 40 / 3], rtol=1e-9)
    # left censoring mirrors it onto a
    model = surv.Uniform.fit([-10, -10, -5, 0], c=[0, -1, 0, 0])
    np.testing.assert_allclose(model.params, [-40 / 3, 0.0], rtol=1e-9)


def test_uniform_complete_data_still_closed_form():
    x = np.array([2.0, 3.5, 7.0, 4.2])
    model = surv.Uniform.fit(x)
    np.testing.assert_array_equal(model.params, [2.0, 7.0])


# -- Beta-Geometric method of moments --------------------------------------


def test_beta_geometric_mom_matches_the_moments():
    np.random.seed(0)
    x = surv.BetaGeometric.random(2000, 5.0, 3.0)
    model = surv.BetaGeometric.fit(x, how="MOM")
    assert not np.allclose(model.params, [1.0, 1.0])
    assert model.params[0] > 2
    np.testing.assert_allclose(model.moment(1), x.mean(), rtol=1e-10)
    np.testing.assert_allclose(model.moment(2), (x**2).mean(), rtol=1e-10)


def test_beta_geometric_second_moment_is_exact():
    a, b = 5.0, 3.0
    c = a + b - 1
    exact = 2 * c * (c - 1) / ((a - 1) * (a - 2)) - c / (a - 1)
    assert surv.BetaGeometric.moment(2, a, b) == pytest.approx(exact)
    assert exact == pytest.approx(5.25)


def test_beta_geometric_mom_refuses_underdispersed_data():
    with pytest.raises(ValueError, match="no Beta-Geometric solution"):
        surv.BetaGeometric.fit([1, 2] * 50, how="MOM")


def test_mom_never_returns_a_nan_objective_start():
    np.random.seed(0)
    x = surv.BetaGeometric.random(500, 5.0, 3.0)
    # the default start a = 1 has no finite mean
    with pytest.raises(ValueError, match="moments are not finite"):
        surv.BetaGeometric.fit(x, how="MOM", fixed={"b": 3.0})
    # a valid start works
    model = surv.BetaGeometric.fit(x, how="MOM", fixed={"b": 3.0}, init=[4.0])
    assert model.moment(1) == pytest.approx(x.mean(), rel=1e-4)


# -- CustomDistribution ----------------------------------------------------


def _gompertz() -> surv.CustomDistribution:
    def Hf(x, *params):
        return params[0] * (anp.exp(params[1] * x) - 1)

    return surv.CustomDistribution(
        "Gompertz", Hf, ["nu", "b"], ((0, None), (0, None)), (0, np.inf)
    )


def test_custom_distribution_refuses_mpp_clearly():
    G = _gompertz()
    assert not G.supports_mpp
    with pytest.raises(ValueError, match="probability plot"):
        G.fit([0.5, 1.0, 1.2, 1.5, 2.0], how="MPP")


def test_custom_distribution_moments_mean_and_var():
    G = _gompertz()
    rv = gompertz(0.1, scale=1 / 0.5)
    model = G.from_params([0.1, 0.5])
    assert model.mean() == pytest.approx(rv.mean(), rel=1e-8)
    assert model.var() == pytest.approx(rv.var(), rel=1e-8)
    assert G.moment(2, 0.1, 0.5) == pytest.approx(rv.moment(2), rel=1e-8)


def test_custom_distribution_moments_on_other_supports():
    def H_shifted(x, *params):
        return ((x - 2) / params[0]) ** params[1]

    W = surv.CustomDistribution(
        "W2", H_shifted, ["a", "k"], ((0, None), (0, None)), (2, np.inf)
    )
    assert W.moment(1, 3.0, 2.0) == pytest.approx(
        2 + surv.Weibull.mean(3.0, 2.0)
    )

    def H_gumbel(x, *params):
        return anp.exp((x - params[0]) / params[1])

    G = surv.CustomDistribution(
        "Gmb",
        H_gumbel,
        ["mu", "s"],
        ((None, None), (0, None)),
        (-np.inf, np.inf),
    )
    assert G.moment(1, -3.0, 2.0) == pytest.approx(surv.Gumbel.mean(-3.0, 2.0))
    assert G.moment(2, -3.0, 2.0) == pytest.approx(
        surv.Gumbel.moment(2, -3.0, 2.0)
    )


def test_custom_distribution_mom_is_fast_and_matches():
    G = _gompertz()
    x = gompertz(0.1, scale=1 / 0.5).rvs(size=100, random_state=0)
    model = G.fit(x, how="MOM")
    np.testing.assert_allclose(model.mean(), x.mean(), rtol=1e-4)
    np.testing.assert_allclose(model.moment(2), (x**2).mean(), rtol=1e-4)


# -- MOM with fixed parameters ---------------------------------------------


def test_mom_with_fixed_matches_only_the_free_moments():
    np.random.seed(1)
    x = surv.Weibull.random(50, 10, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = surv.Weibull.fit(x, how="MOM", fixed={"beta": 2})
    assert model.params[1] == 2
    # one free parameter, one equation: the mean is matched exactly
    assert model.mean() == pytest.approx(x.mean(), rel=1e-4)


def test_mom_search_backs_away_from_missing_moments():
    # With alpha fixed well below the data's scale, matching the mean
    # needs a LogLogistic shape just above 1, next to the region where the
    # mean does not exist. The search used to step in, end on a nan
    # objective and report it; that region now reads as +inf.
    np.random.seed(0)
    x = surv.LogLogistic.random(100, 10, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = surv.LogLogistic.fit(x, how="MOM", fixed={"alpha": 2.0})
    assert 1 < model.params[1] < 1.5
    assert model.mean() == pytest.approx(x.mean(), rel=1e-4)


def test_mom_with_every_parameter_fixed():
    model = surv.Weibull.fit(
        [1.0, 2.0, 3.0], how="MOM", fixed={"alpha": 3.0, "beta": 2.0}
    )
    np.testing.assert_allclose(model.params, [3.0, 2.0])


# -- fallback_minimize ends on Nelder-Mead -------------------------------


def test_fallback_minimize_last_rung_is_nelder_mead():
    # A jacobian that is wrong everywhere makes BFGS fail and a zero
    # hessian skips Newton-CG, so the last rung must finish the job. The
    # objective has a kink at its minimum, where finite-difference BFGS
    # (the old last rung) loses precision and fails too.
    def fun(v):
        return abs(v[0] - 3.0) + abs(v[1] + 1.0)

    def bad_jac(v):
        return np.array([1.0, 1.0])

    def zero_hess(v):
        return np.zeros((2, 2))

    res = fallback_minimize(fun, np.array([0.0, 0.0]), (), bad_jac, zero_hess)
    assert res.success
    # Nelder-Mead reports no gradient; either BFGS would
    assert "jac" not in res
    np.testing.assert_allclose(res.x, [3.0, -1.0], atol=1e-3)


# -- param_cb on the offset ------------------------------------------------


def test_param_cb_gamma_gives_a_clear_error():
    np.random.seed(0)
    x = surv.Weibull.random(50, 10, 2) + 5
    with pytest.raises(ValueError, match="only estimated for offset"):
        surv.Weibull.fit(x).param_cb("gamma")
    with pytest.raises(ValueError, match="threshold parameter"):
        surv.Weibull.fit(x, offset=True).param_cb("gamma")
    with pytest.raises(ValueError, match="Unknown parameter 'shape'"):
        surv.Weibull.fit(x).param_cb("shape")


# -- AIC / BIC count only the estimated parameters -------------------------


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


# -- var() of LFP / zero-inflated models ----------------------------------


def test_var_follows_the_defective_convention_of_mean():
    lfp = surv.Weibull.from_params([10.0, 2.0], p=0.7)
    assert lfp.var() == pytest.approx(lfp.moment(2) - lfp.mean() ** 2)
    zi = surv.Weibull.from_params([10.0, 2.0], f0=0.2)
    assert zi.var() == pytest.approx(zi.moment(2) - zi.mean() ** 2)
    # the zero-inflated variance is the mixture's, checked by simulation
    np.random.seed(0)
    draws = zi.random(400_000)
    assert zi.var() == pytest.approx(np.var(draws), rel=1e-2)
    # plain and offset models are unchanged
    plain = surv.Weibull.from_params([10.0, 3.0])
    assert plain.var() == pytest.approx(10.533288486847923)
    shifted = surv.Weibull.from_params([10.0, 3.0], gamma=5.0)
    assert shifted.var() == pytest.approx(10.533288486847923)


# -- MPS error messages ----------------------------------------------------


def test_mps_truncation_messages_have_no_space_run():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(ValueError) as err:
        surv.Weibull.fit(x, how="MPS", tl=[0, 0, 0.5, 0, 0])
    assert "  " not in str(err.value)
    assert str(err.value) == (
        "Left truncated value can only be single number when using MPS"
    )
    with pytest.raises(ValueError) as err:
        surv.Weibull.fit(x, how="MPS", tr=[10, 10, 10, 10, 9])
    assert "  " not in str(err.value)
