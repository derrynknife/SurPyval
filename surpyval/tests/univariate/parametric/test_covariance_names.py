"""``Parametric.covariance_names``: the name of each row of ``covariance()``.

The covariance of a fit holds the distribution's parameters, then the
limited-failure proportion ``lfp_p`` and the zero-inflation fraction ``f0``
where the model has them; a parameter fixed at fit time keeps a zero row;
an offset ``gamma`` is held at its estimate and has no row. These tests pin
the names to that layout for every option, check that a draw from the
covariance maps onto the named parameters, and check an offset fit's
covariance against a numerical Hessian with ``gamma`` held.
"""

import warnings

import numpy as np
import pytest
from numdifftools import Hessian  # type: ignore
from scipy.special import expit
from scipy.stats import norm

import surpyval as surv
from surpyval.univariate.parametric.parametric import Parametric

RNG = np.random.default_rng(0)
X = RNG.weibull(2, 200) * 100
X_ZI = np.r_[X, np.zeros(20)]
# Censor the largest fifth, so a limited-failure proportion is estimable
C_LFP = (X > np.quantile(X, 0.8)).astype(int)


def _fit(dist, x, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dist.fit(x, **kwargs)


CASES = [
    ("plain", surv.Weibull, X, {}, ["alpha", "beta"]),
    (
        "lfp",
        surv.Weibull,
        X,
        {"c": C_LFP, "lfp": True},
        ["alpha", "beta", "lfp_p"],
    ),
    ("zi", surv.Weibull, X_ZI, {"zi": True}, ["alpha", "beta", "f0"]),
    (
        "lfp+zi",
        surv.Weibull,
        X_ZI,
        {"c": np.r_[C_LFP, np.zeros(20, int)], "lfp": True, "zi": True},
        ["alpha", "beta", "lfp_p", "f0"],
    ),
    ("offset", surv.Weibull, X + 50, {"offset": True}, ["alpha", "beta"]),
    (
        "offset+lfp",
        surv.Weibull,
        X + 50,
        {"c": C_LFP, "offset": True, "lfp": True},
        ["alpha", "beta", "lfp_p"],
    ),
    ("fixed", surv.Weibull, X, {"fixed": {"beta": 2.0}}, ["alpha", "beta"]),
    (
        "fixed+lfp",
        surv.Weibull,
        X,
        {"c": C_LFP, "lfp": True, "fixed": {"alpha": 100.0}},
        ["alpha", "beta", "lfp_p"],
    ),
    (
        "weighted",
        surv.Weibull,
        X,
        {"w": RNG.uniform(0.5, 2, 200)},
        ["alpha", "beta"],
    ),
    ("exponential", surv.Exponential, X, {}, ["failure_rate"]),
    (
        "lognormal+zi",
        surv.LogNormal,
        X_ZI,
        {"zi": True},
        ["mu", "sigma", "f0"],
    ),
    # Geometric's own parameter is p; the proportion is still lfp_p
    (
        "geometric+lfp",
        surv.Geometric,
        np.r_[np.arange(1, 30), np.full(10, 30)],
        {"c": np.r_[np.zeros(29, int), np.ones(10, int)], "lfp": True},
        ["p", "lfp_p"],
    ),
]


@pytest.mark.parametrize(
    "dist, x, kwargs, names",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_names_label_every_row_of_the_covariance(dist, x, kwargs, names):
    model = _fit(dist, x, **kwargs)
    assert model.covariance_names == names
    cov = model.covariance()
    assert cov.shape == (len(names), len(names))
    # The distribution's names lead, as parameter_names (unchanged)
    k = len(model.parameter_names)
    assert model.covariance_names[:k] == model.parameter_names
    assert model.standard_errors().shape == (len(names),)


def test_a_fixed_parameter_keeps_a_zero_row():
    model = _fit(surv.Weibull, X, fixed={"beta": 2.0})
    i = model.covariance_names.index("beta")
    cov = model.covariance()
    assert np.all(cov[i] == 0) and np.all(cov[:, i] == 0)
    assert cov[0, 0] > 0


def test_offset_gamma_is_not_a_row():
    model = _fit(surv.Weibull, X + 50, offset=True)
    assert model.gamma > 0
    assert "gamma" not in model.covariance_names
    # the optimiser's vector leads with gamma; the covariance does not
    assert model.param_map["gamma"] == 0


@pytest.mark.parametrize("how", ["MPS", "MSE", "MPP"])
def test_names_exist_where_there_is_no_covariance(how):
    model = _fit(surv.Weibull, X, how=how)
    assert model.covariance_names == ["alpha", "beta"]
    with pytest.raises(ValueError):
        model.covariance()


def test_names_of_a_model_from_params():
    model = surv.Weibull.from_params([100, 2], gamma=5.0, lfp_p=0.9, f0=0.1)
    assert model.covariance_names == ["alpha", "beta", "lfp_p", "f0"]
    assert model.parameter_names == ["alpha", "beta"]


def test_names_survive_the_dict_round_trip():
    model = _fit(surv.Weibull, X_ZI, zi=True)
    stored = model.to_dict()
    # Derived from the stored options: the dict format is unchanged
    assert "covariance_names" not in stored
    restored = Parametric.from_dict(stored)
    assert restored.covariance_names == model.covariance_names
    assert restored.covariance().shape == model.covariance().shape


def test_a_draw_from_the_covariance_maps_onto_the_named_rows():
    """Draws from N(estimates, covariance()) labelled by covariance_names
    rebuild models: each name's draws centre on that parameter's estimate
    and spread by its standard error. 20,000 draws: the mean of a name's
    draws is within 5 / sqrt(20000) = 0.035 of its standard error from the
    estimate, and the sample SD within 3% of the SE."""
    model = _fit(
        surv.Weibull,
        X_ZI,
        c=np.r_[C_LFP, np.zeros(20, int)],
        lfp=True,
        zi=True,
    )
    names = model.covariance_names
    estimates = {
        **dict(zip(model.parameter_names, model.params)),
        "lfp_p": model.lfp_p,
        "f0": model.f0,
    }
    centre = np.array([estimates[n] for n in names])
    cov = model.covariance()
    draws = np.random.default_rng(1).multivariate_normal(centre, cov, 20000)
    se = np.sqrt(np.diag(cov))
    for j, name in enumerate(names):
        assert abs(draws[:, j].mean() - estimates[name]) < 0.035 * se[j]
        assert abs(draws[:, j].std() / se[j] - 1) < 0.03, name
    # One draw, by name, is a model
    named = dict(zip(names, draws[0]))
    rebuilt = surv.Weibull.from_params(
        [named["alpha"], named["beta"]],
        lfp_p=float(np.clip(named["lfp_p"], 0, 1)),
        f0=float(np.clip(named["f0"], 0, 1)),
    )
    assert rebuilt.covariance_names == names


def test_param_cb_reads_the_named_rows():
    """The Wald bound of each extra parameter uses the row its name
    labels (logit scale for a proportion)."""
    model = _fit(
        surv.Weibull,
        X_ZI,
        c=np.r_[C_LFP, np.zeros(20, int)],
        lfp=True,
        zi=True,
    )
    cov = model.covariance()
    z = norm.ppf(0.975)
    for name, value in (("lfp_p", model.lfp_p), ("f0", model.f0)):
        i = model.covariance_names.index(name)
        half = z * np.sqrt(cov[i, i]) / (value * (1 - value))
        logit = np.log(value / (1 - value))
        expected = expit(logit + np.array([-half, half]))
        np.testing.assert_allclose(model.param_cb(name), expected, rtol=1e-10)


def test_offset_covariance_is_the_inverse_hessian_with_gamma_held():
    """An offset fit's covariance is the inverse of numdifftools' Hessian
    of the negative log-likelihood in (alpha, beta), gamma held at its
    estimate; rtol 1e-3 (finite differences against autograd)."""
    model = _fit(surv.Weibull, X + 50, offset=True)

    def nll(theta):
        return model.dist._neg_ll_func(
            model.surv_data, *theta, model.gamma, 0.0, 1.0
        )

    expected = np.linalg.inv(Hessian(nll)(np.asarray(model.params, float)))
    np.testing.assert_allclose(model.covariance(), expected, rtol=1e-3)
