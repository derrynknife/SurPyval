"""Standard errors and confidence bounds of a fitted copula model (#540).

``CopulaModel`` had no ``covariance``, ``param_cb`` or ``cb``: a fitted
copula parameter came with no measure of its uncertainty. These check
the joint-MLE covariance against an independent numerical Hessian, the
two-stage (IFM) Godambe covariance against a bootstrap, the Wald bounds
against their formulas, and the bounds' behaviour (options, units,
saving, boundaries). Their coverage is checked in
``calibration/test_coverage_copula.py``.
"""

import json
import pickle
import warnings

import numdifftools as nd
import numpy as np
import pytest
from scipy.special import expit, logit
from scipy.stats import norm

import surpyval as sp
from surpyval import KaplanMeier, Weibull
from surpyval.multivariate import (
    AMH,
    Clayton,
    CopulaModel,
    Frank,
    Gaussian,
    Independence,
    StudentT,
)
from surpyval.tests._helpers import WEIBULL_MARGINS

Z = norm.ppf(0.975)
PTS = np.array([[5.0, 15.0], [10.0, 20.0], [14.0, 12.0]])


@pytest.fixture(scope="module")
def sample():
    return Clayton.from_params([2.0], WEIBULL_MARGINS).random(
        300, random_state=0
    )


@pytest.fixture(scope="module")
def mle(sample):
    return Clayton.fit(sample, margins=[Weibull, Weibull], how="MLE")


@pytest.fixture(scope="module")
def ifm(sample):
    return Clayton.fit(sample, margins=[Weibull, Weibull])


def _joint_neg_ll(copula, data):
    """The joint negative log-likelihood in (copula, alpha_1, beta_1,
    alpha_2, beta_2), written from the copula's likelihood alone."""
    k = len(copula.parameter_names)

    def f(v):
        dims = [
            copula._prepare_dim(
                Weibull.from_params(v[k + 2 * d : k + 2 * d + 2]),
                *data.dimension(d),
            )
            for d in range(2)
        ]
        return copula.neg_ll(v[:k], dims, data.n)

    return f


def _vector(model):
    return np.r_[model.params, model.margins[0].params, model.margins[1].params]


# -- the joint MLE ----------------------------------------------------------
def test_540_mle_covariance_is_the_inverse_joint_hessian(mle):
    # numdifftools' adaptive Hessian, independent of the central
    # differences the model takes.
    # (A fixed step: the default's largest leaves the parameters' space.)
    H = nd.Hessian(_joint_neg_ll(Clayton, mle.data), step=1e-4)(_vector(mle))
    want = np.linalg.inv(H)
    got = mle.covariance(margins=True)
    assert got.shape == (5, 5)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=1e-6)
    np.testing.assert_allclose(mle.covariance(), want[:1, :1], rtol=2e-3)
    np.testing.assert_allclose(
        mle.standard_errors(margins=True), np.sqrt(np.diag(want)), rtol=1e-3
    )


def test_540_param_cb_is_wald_on_the_log_scale(mle):
    theta = mle.params[0]
    se = np.sqrt(mle.covariance()[0, 0])
    # theta > 0: theta * exp(-+ z se / theta), the univariate models' rule
    want = theta * np.exp(np.array([-1, 1]) * Z * se / theta)
    np.testing.assert_allclose(mle.param_cb("theta"), want, rtol=1e-12)
    lower = mle.param_cb("theta", bound="lower", alpha_ci=0.025)
    np.testing.assert_allclose(lower, want[0], rtol=1e-12)
    # Symmetric about the estimate on the log scale (the transform's
    # round trip)
    logs = np.log(mle.param_cb("theta", alpha_ci=0.2))
    np.testing.assert_allclose(logs.mean(), np.log(theta), rtol=1e-12)


def test_540_param_cb_of_rho_is_on_fishers_z():
    X = Gaussian.from_params([0.6], WEIBULL_MARGINS).random(
        200, random_state=1
    )
    model = Gaussian.fit(X, margins=[Weibull, Weibull], how="MLE")
    rho, se = model.params[0], model.standard_errors()[0]
    z = np.arctanh(rho) + np.array([-1, 1]) * Z * se / (1 - rho**2)
    np.testing.assert_allclose(model.param_cb("rho"), np.tanh(z), rtol=1e-10)
    lo, hi = model.param_cb("rho", alpha_ci=1e-12)
    assert -1 < lo < rho < hi < 1


def test_540_param_cb_of_an_unbounded_parameter_is_plain():
    X = Frank.from_params([5.0], WEIBULL_MARGINS).random(200, random_state=2)
    model = Frank.fit(X, margins=[Weibull, Weibull])
    theta, se = model.params[0], model.standard_errors()[0]
    np.testing.assert_allclose(
        model.param_cb("theta"), theta + np.array([-1, 1]) * Z * se
    )


def test_540_student_t_bounds_each_parameter():
    X = StudentT.from_params([0.6, 5.0], WEIBULL_MARGINS).random(
        300, random_state=3
    )
    model = StudentT.fit(X, margins=[Weibull, Weibull])
    assert model.covariance().shape == (2, 2)
    for name, (low, high) in zip(model.parameter_names, model.copula.bounds):
        lo, hi = model.param_cb(name)
        assert (low is None or low < lo) and (high is None or hi < high)
        assert lo < model.params[model.parameter_names.index(name)] < hi


# -- the two-stage (IFM) fit ------------------------------------------------
def _copula_stage_variance(model):
    """The copula stage's inverse Hessian, the margins held as fitted."""
    dims = [
        model.copula._prepare_dim(model.margins[d], *model.data.dimension(d))
        for d in range(2)
    ]
    H = nd.Hessian(
        lambda t: model.copula.neg_ll(t, dims, model.data.n), step=1e-4
    )(model.params)
    return np.linalg.inv(np.atleast_2d(H))


def test_540_ifm_godambe_matches_the_bootstrap():
    X = Clayton.from_params([2.0], WEIBULL_MARGINS).random(
        150, random_state=7
    )
    model = Clayton.fit(X, margins=[Weibull, Weibull])
    godambe = model.standard_errors()[0]
    naive = np.sqrt(_copula_stage_variance(model)[0, 0])
    rng = np.random.default_rng(1)
    reps = 200
    draws = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(reps):
            rows = rng.integers(0, len(X), len(X))
            refit = Clayton.fit(X[rows], margins=[Weibull, Weibull])
            draws.append(refit.params[0])
    boot = np.std(draws, ddof=1)
    # The bootstrap's standard error has a Monte Carlo error of about
    # 1 / sqrt(2 reps) relative: 5%. Treating the margins as known (the
    # copula stage's Hessian alone) understates it.
    assert abs(godambe / boot - 1) < 3 / np.sqrt(2 * reps)
    assert abs(godambe / boot - 1) < abs(naive / boot - 1)
    assert naive < godambe


def test_540_ifm_with_given_margins_is_the_copula_stage_mle(sample):
    fitted = [Weibull.fit(sample[:, 0]), Weibull.fit(sample[:, 1])]
    model = Clayton.fit(sample, margins=fitted)
    want = _copula_stage_variance(model)
    np.testing.assert_allclose(model.covariance(), want, rtol=2e-3)
    full = model.covariance(margins=True)
    # The margins passed fitted are known: zero rows and columns
    np.testing.assert_array_equal(full[1:, :], 0.0)
    np.testing.assert_array_equal(full[:, 1:], 0.0)


def test_540_ifm_margin_blocks_are_their_sandwich(ifm, sample):
    # With the model right, each margin's sandwich is close to its own
    # univariate fit's inverse information.
    full = ifm.covariance(margins=True)
    for d, sl in ((0, slice(1, 3)), (1, slice(3, 5))):
        own = Weibull.fit(sample[:, d]).covariance()
        np.testing.assert_allclose(full[sl, sl], own, rtol=0.25, atol=0.01)


# -- bounds on the joint functions -----------------------------------------
@pytest.mark.parametrize("on", ["sf", "ff"])
def test_540_cb_is_the_delta_method_on_the_logit(mle, on):
    v0 = _vector(mle)

    def joint(v):
        built = Clayton.from_params(
            v[:1],
            [Weibull.from_params(v[1:3]), Weibull.from_params(v[3:5])],
        )
        return getattr(built, on)(PTS)

    p = joint(v0)
    J = nd.Jacobian(joint, step=1e-5)(v0)
    se = np.sqrt(np.einsum("ij,jk,ik->i", J, mle.covariance(margins=True), J))
    half = Z * se / (p * (1 - p))
    want = expit(logit(p)[:, None] + np.outer(half, [-1, 1]))
    np.testing.assert_allclose(mle.cb(PTS, on=on), want, rtol=1e-4)
    # The margins' uncertainty is in it: wider than the copula's alone
    copula_only = np.sqrt(J[:, 0] ** 2 * mle.covariance()[0, 0])
    assert np.all(se > copula_only)


def test_540_cb_options_and_shapes(ifm):
    two = ifm.cb(PTS)
    assert two.shape == (3, 2)
    assert np.all((two[:, 0] < ifm.sf(PTS)) & (ifm.sf(PTS) < two[:, 1]))
    np.testing.assert_array_equal(ifm.cb(PTS, on="R"), two)
    np.testing.assert_array_equal(ifm.cb(PTS, on="F"), ifm.cb(PTS, on="ff"))
    lower = ifm.cb(PTS, bound="lower", alpha_ci=0.025)
    np.testing.assert_allclose(lower, two[:, 0], rtol=1e-12)
    upper = ifm.cb(PTS, on="ff", bound="upper", alpha_ci=0.025)
    np.testing.assert_allclose(upper, ifm.cb(PTS, on="ff")[:, 1], rtol=1e-12)
    # One point, as a pair: [lower, upper]
    np.testing.assert_allclose(ifm.cb(PTS[1]), two[1], rtol=1e-12)
    assert ifm.cb(PTS[1], bound="lower").shape == ()
    assert ifm.cb(np.empty((0, 2))).shape == (0, 2)
    # A missing coordinate is missing there only, without a warning
    x = PTS.copy()
    x[1, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = ifm.cb(x)
    assert np.isnan(got[1]).all() and not np.isnan(got[[0, 2]]).any()


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.cb(PTS, on="hf"),
        lambda m: m.cb(PTS, bound="both"),
        lambda m: m.cb(PTS, method="lr"),
        lambda m: m.cb(PTS, alpha_ci=1.5),
        lambda m: m.cb([[1.0, 2.0, 3.0]]),
        lambda m: m.param_cb("rho"),
        lambda m: m.param_cb("theta", bound="both"),
        lambda m: m.param_cb("theta", method="lr"),
        lambda m: m.param_cb("theta", alpha_ci=0.0),
    ],
)
def test_540_invalid_options_raise(ifm, call):
    with pytest.raises(ValueError):
        call(ifm)


# -- units, saving -----------------------------------------------------------
def test_540_bounds_do_not_depend_on_the_time_unit(sample):
    # A constant scale on the data changes nothing but the margins' scale
    hours = Clayton.fit(sample, margins=[Weibull, Weibull], how="MLE")
    minutes = Clayton.fit(60 * sample, margins=[Weibull, Weibull], how="MLE")
    np.testing.assert_allclose(
        minutes.param_cb("theta"), hours.param_cb("theta"), rtol=1e-3
    )
    np.testing.assert_allclose(
        minutes.cb(60 * PTS), hours.cb(PTS), rtol=1e-3
    )


@pytest.mark.parametrize("how", ["IFM", "MLE"])
def test_540_bounds_survive_saving(sample, how):
    model = Clayton.fit(sample, margins=[Weibull, Weibull], how=how)
    text = json.dumps(model.to_dict(), allow_nan=False)
    for restored in (
        sp.from_dict(json.loads(text)),
        pickle.loads(pickle.dumps(model)),
    ):
        np.testing.assert_array_equal(
            restored.covariance(margins=True), model.covariance(margins=True)
        )
        np.testing.assert_array_equal(
            restored.param_cb("theta"), model.param_cb("theta")
        )
        np.testing.assert_array_equal(restored.cb(PTS), model.cb(PTS))


def test_540_a_dict_saved_without_the_covariance_says_to_refit(ifm):
    d = ifm.to_dict()
    del d["covariance"]
    restored = CopulaModel.from_dict(d)
    with pytest.raises(ValueError, match="refit"):
        restored.param_cb("theta")
    with pytest.raises(ValueError, match="refit"):
        restored.cb(PTS)


# -- models without a covariance, and boundaries ---------------------------
def test_540_from_params_has_no_covariance():
    model = Clayton.from_params([2.0], WEIBULL_MARGINS)
    for call in (model.covariance, lambda: model.param_cb("theta")):
        with pytest.raises(ValueError, match="no parameter covariance"):
            call()
    assert "covariance" not in model.to_dict()


def test_540_non_parametric_margins_have_no_covariance(sample):
    model = Clayton.fit(sample, margins=[KaplanMeier, Weibull])
    with pytest.raises(ValueError, match="pseudo-likelihood"):
        model.param_cb("theta")
    assert "covariance" not in model.to_dict()


def test_540_independence_bounds_the_margins_only(sample):
    model = Independence.fit(sample, margins=[Weibull, Weibull])
    assert model.covariance().shape == (0, 0)
    assert model.covariance(margins=True).shape == (4, 4)
    sf = model.sf(PTS)
    lo, hi = model.cb(PTS).T
    assert np.all((lo < sf) & (sf < hi))


def test_540_a_parameter_on_its_bound_has_no_wald_bound():
    # Kendall's tau of 0.5, past the AMH's 1/3: its estimate is on its
    # bound, theta = 1, where the likelihood is not regular.
    X = Clayton.from_params([2.0], WEIBULL_MARGINS).random(
        200, random_state=4
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AMH.fit(X, margins=[Weibull, Weibull])
    assert model.params[0] == 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cov = model.covariance(margins=True)
    assert not caught  # nan there by design, documented
    assert np.isnan(cov[0]).all() and np.isfinite(cov[1:, 1:]).all()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bound = model.param_cb("theta")
    assert np.isnan(bound).all()
    assert len(caught) == 1 and "edge of its support" in str(caught[0].message)
    assert caught[0].filename == __file__
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joint = model.cb(PTS)
    assert np.isnan(joint).all()
    assert len(caught) == 1 and caught[0].filename == __file__
