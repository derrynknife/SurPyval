"""``rescale_time(factor)``: a fitted model on another clock.

The conformance module of the same name checks every registered model's
identities and its agreement with a refit; these check what a fit
carries across (the likelihood-ratio and bootstrap bounds, the weighted
fit's covariances, a fixed parameter), a model without its data, the
mixture's covariance in its search coordinates, the unit label, and the
refusals. The data, seeds and tolerances were fixed before the results
were seen.
"""

import math
import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval import Parametric

K = 7.3
T = np.array([20.0, 60.0, 150.0, 300.0])


def _data(seed=20261120):
    rng = np.random.default_rng(seed)
    x = 30 * rng.weibull(1.8, 40)
    c = (x > 45).astype(int)
    return np.minimum(x, 45), c, rng


@pytest.fixture(scope="module")
def plain():
    x, c, _ = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return x, c, surv.Weibull.fit(x, c), surv.Weibull.fit(x * K, c)


# -- the parameters ----------------------------------------------------------


@pytest.mark.parametrize(
    "dist, params, expected",
    [
        # As RePyability moves them for a duty cycle d: R(d t)
        (surv.Weibull, [100.0, 2.0], [100.0 / 0.4, 2.0]),
        (surv.Exponential, [0.01], [0.01 * 0.4]),
        (surv.Gamma, [2.0, 0.05], [2.0, 0.05 * 0.4]),
        (surv.LogNormal, [3.0, 0.5], [3.0 - math.log(0.4), 0.5]),
        (surv.Normal, [100.0, 10.0], [250.0, 25.0]),
        (surv.Uniform, [10.0, 50.0], [25.0, 125.0]),
        (surv.Beta4, [2.0, 3.0, 10.0, 50.0], [2.0, 3.0, 25.0, 125.0]),
        (surv.Hypoexponential, [0.1, 0.3], [0.04, 0.12]),
        (surv.ExactEventTime, [80.0], [200.0]),
    ],
)
def test_parameters_move_by_their_roles(dist, params, expected):
    new = dist.from_params(params).rescale_time(0.4)
    np.testing.assert_allclose(new.params, expected, rtol=1e-15)
    assert new.dist.name == dist.name


def test_an_offset_is_a_time_and_the_proportions_are_kept():
    model = surv.Weibull.from_params([100, 2], gamma=10.0, lfp_p=0.8, f0=0.1)
    new = model.rescale_time(0.5)
    assert new.extras == {"gamma": 20.0, "lfp_p": 0.8, "f0": 0.1}
    t = np.array([5.0, 25.0, 100.0, 400.0])
    np.testing.assert_allclose(new.sf(t), model.sf(0.5 * t), rtol=1e-14)


def test_a_factor_of_one_is_the_same_model(plain):
    _, _, model, _ = plain
    same = model.rescale_time(1)
    np.testing.assert_array_equal(same.params, model.params)
    np.testing.assert_array_equal(same.covariance(), model.covariance())
    assert same.neg_ll() == model.neg_ll()


@pytest.mark.parametrize(
    "factor", [0, -1.0, np.nan, np.inf, True, "2", None, [2.0]]
)
def test_a_factor_must_be_a_finite_number_above_0(factor):
    model = surv.Weibull.from_params([100, 2])
    with pytest.raises(ValueError, match="'factor' must be a finite"):
        model.rescale_time(factor)


# -- the unit label ----------------------------------------------------------


def test_the_unit_is_kept_unless_given():
    model = surv.Weibull.from_params([100, 2], unit="hours")
    assert model.rescale_time(0.25).unit == "hours"
    days = model.rescale_time(24, unit="  days ")
    assert days.unit == "days"
    assert surv.from_dict(days.to_dict()).unit == "days"
    assert surv.Weibull.from_params([1, 2]).rescale_time(2).unit is None
    with pytest.raises(TypeError, match="unit must be text"):
        model.rescale_time(24, unit=24)


# -- what a fit carries ------------------------------------------------------


def test_bounds_of_every_kind_are_the_refits(plain):
    _, _, model, ref = plain
    new = model.rescale_time(1 / K)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("wald", "lr"):
            np.testing.assert_allclose(
                new.cb(T, on="sf", method=method),
                ref.cb(T, on="sf", method=method),
                rtol=1e-6,
                err_msg=method,
            )
        np.testing.assert_allclose(
            new.cb(T, on="sf", method="bootstrap", random_state=7),
            ref.cb(T, on="sf", method="bootstrap", random_state=7),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            new.param_cb("alpha", method="lr"),
            ref.param_cb("alpha", method="lr"),
            rtol=1e-6,
        )
    assert new.aic() == pytest.approx(ref.aic(), rel=1e-10)
    assert new.bic() == pytest.approx(ref.bic(), rel=1e-10)


def test_pivotal_bounds_of_a_plotting_fit_are_the_refits(plain):
    x, _, _, _ = plain
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = surv.Weibull.fit(x, how="MPP")
        ref = surv.Weibull.fit(x * K, how="MPP")
        new = model.rescale_time(1 / K)
        np.testing.assert_allclose(
            new.cb(T, on="sf", method="pivotal", random_state=3),
            ref.cb(T, on="sf", method="pivotal", random_state=3),
            rtol=1e-6,
        )


def test_standard_errors_of_shapes_stay_and_of_times_scale(plain):
    _, _, model, _ = plain
    new = model.rescale_time(24)
    np.testing.assert_allclose(
        new.standard_errors(),
        model.standard_errors() * [1 / 24, 1],
        rtol=1e-14,
    )


def test_a_weighted_fit_keeps_its_robust_covariance():
    x, c, rng = _data()
    w = rng.uniform(0.5, 2.0, x.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = surv.Weibull.fit(x, c, w=w)
        ref = surv.Weibull.fit(x * K, c, w=w)
    new = model.rescale_time(1 / K)
    assert new.weighted and new.effective_n == model.effective_n
    np.testing.assert_allclose(new.covariance(), ref.covariance(), rtol=1e-6)
    np.testing.assert_allclose(new.hess_inv, ref.hess_inv, rtol=1e-6)
    assert new.neg_ll() == pytest.approx(ref.neg_ll(), rel=1e-10)
    np.testing.assert_array_equal(new.data["w"], model.data["w"])


def test_a_fixed_parameter_stays_fixed(plain):
    x, c, _, _ = plain
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = surv.Weibull.fit(x, c, fixed={"beta": 2.0})
        ref = surv.Weibull.fit(x * K, c, fixed={"beta": 2.0})
    new = model.rescale_time(1 / K)
    assert new.to_dict()["fixed"] == ["beta"]
    np.testing.assert_allclose(new.covariance(), ref.covariance(), rtol=1e-6)
    assert new.covariance()[1, 1] == 0


def test_zero_inflation_counts_only_the_densities():
    x, c, _ = _data()
    x0, c0 = np.r_[0.0, 0.0, 0.0, x], np.r_[0, 0, 0, c]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = surv.Weibull.fit(x0, c0, zi=True)
        ref = surv.Weibull.fit(x0 * K, c0, zi=True)
    new = model.rescale_time(1 / K)
    assert new.neg_ll() == pytest.approx(ref.neg_ll(), rel=1e-8)


# -- a model without its data ------------------------------------------------


def test_without_data_the_covariance_comes_but_not_the_likelihood(plain):
    _, _, model, ref = plain
    restored = Parametric.from_dict(model.to_dict())
    new = restored.rescale_time(1 / K)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.testing.assert_allclose(
            new.cb(T, on="sf"), ref.cb(T, on="sf"), rtol=1e-6
        )
    for criterion in (new.neg_ll, new.aic, new.bic):
        with pytest.raises(ValueError, match="No log-likelihood: .*rescale"):
            criterion()
    with pytest.raises(ValueError, match="need the original data"):
        new.cb(T, method="lr")


def test_saved_with_its_data_it_is_the_fitted_model(plain):
    _, _, model, ref = plain
    restored = Parametric.from_dict(model.to_dict(with_data=True))
    new = restored.rescale_time(1 / K)
    assert new.neg_ll() == pytest.approx(ref.neg_ll(), rel=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.testing.assert_allclose(
            new.cb(T, on="sf", method="lr"),
            ref.cb(T, on="sf", method="lr"),
            rtol=1e-6,
        )


def test_a_model_built_from_parameters_stays_one():
    new = surv.Weibull.from_params([100, 2]).rescale_time(2)
    assert new.method == "given parameters"
    with pytest.raises(ValueError, match="Must have been fit with data"):
        new.neg_ll()


# -- mixtures ----------------------------------------------------------------

X_MIX = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]


@pytest.mark.parametrize("dist", [surv.Weibull, surv.Normal, surv.LogNormal])
def test_a_mixtures_covariance_is_moved_exactly(dist):
    """The covariance carried to the new clock is the one worked out
    afresh from the rescaled data at the rescaled parameters."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = surv.MixtureModel.fit(X_MIX, dist=dist, m=2)
        new = model.rescale_time(1 / K)
        carried = new.covariance()
        new.__dict__.pop("_theta_cov")
        afresh = new.covariance()
    np.testing.assert_allclose(
        carried, afresh, rtol=1e-6, atol=1e-6 * np.abs(afresh).max()
    )
    np.testing.assert_array_equal(new.w, model.w)


def test_a_mixture_without_data_has_no_likelihood():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = surv.MixtureModel.fit(X_MIX, dist=surv.Weibull, m=2)
    new = surv.MixtureModel.from_dict(model.to_dict()).rescale_time(2.0)
    with pytest.raises(ValueError, match="No log-likelihood"):
        new.aic()
    assert new.covariance().shape == (6, 6)


def test_an_unfitted_mixture_refuses():
    with pytest.raises(ValueError, match="unfitted mixture"):
        surv.MixtureModel(surv.Weibull, 2).rescale_time(2.0)


# -- refusals ----------------------------------------------------------------


def test_the_beta_points_to_the_beta4():
    model = surv.Beta.from_params([2.0, 3.0])
    with pytest.raises(ValueError, match=r"Beta4\.from_params"):
        model.rescale_time(2.0)
    # ... which is the Beta on the new clock
    beta4 = surv.Beta4.from_params([2.0, 3.0, 0.0, 0.5])
    t = np.array([0.05, 0.2, 0.4])
    np.testing.assert_allclose(beta4.sf(t), model.sf(2.0 * t), rtol=1e-12)


def test_a_count_and_a_custom_distribution_refuse():
    with pytest.raises(ValueError, match="distribution of counts"):
        surv.Poisson.from_params([3.0]).rescale_time(2.0)
    gompertz = surv.CustomDistribution(
        "RescaleGompertz",
        lambda x, a, b: a / b * (surv.np.exp(b * x) - 1),
        ["a", "b"],
        ((0, None), (0, None)),
        (0, np.inf),
    )
    with pytest.raises(ValueError, match="custom distribution"):
        gompertz.from_params([0.01, 0.1]).rescale_time(2.0)


def test_the_point_masses_are_the_same_on_every_clock():
    assert surv.InstantlyOccurs.rescale_time(3.0) is surv.InstantlyOccurs
    with pytest.raises(ValueError, match="no unit"):
        surv.NeverOccurs.rescale_time(3.0, unit="days")
    with pytest.raises(ValueError, match="'factor'"):
        surv.NeverOccurs.rescale_time(0)
