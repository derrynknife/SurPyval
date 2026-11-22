"""The uncertainty of a copula fitted with non-parametric margins: the
rank-based covariance of the maximum pseudo-likelihood estimate (Genest,
Ghoudi and Rivest 1995, proposition 2.1) on complete data, the case
bootstrap everywhere, and the refusals between them.

The covariance is checked against the proposition written out a second
way: exact derivatives of the log density by autograd (of the family's
own density for the Clayton and Gumbel, of the closed forms written here
for the Frank and Gaussian) where the package takes central differences,
and the indicator matrix of the ranks where it takes cumulative sums.
The tolerances were set before the values were seen: 1e-4 relative
against the second implementation (nested differences of step
``eps**(1/3)`` are good to about 1e-6), and 15% against the standard
deviation of 400 bootstrap refits, whose own Monte Carlo error is about
``1 / sqrt(2 * 400)``, 3.5%.
"""

import json
import warnings

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import elementwise_grad
from scipy.special import ndtri

import surpyval as surv
from surpyval import KaplanMeier, NelsonAalen, Weibull
from surpyval.multivariate import (
    Clayton,
    Frank,
    Gaussian,
    Gumbel,
    Independence,
    StudentT,
)

N = 500
KM = [KaplanMeier, KaplanMeier]
MARGINS = [Weibull.from_params([10.0, 2.0]), Weibull.from_params([20.0, 3.0])]
# family, true parameter, the data's seed
FAMILIES = {
    "Clayton": (Clayton, 2.0, 5201),
    "Gumbel": (Gumbel, 2.0, 5202),
    "Frank": (Frank, 5.0, 5203),
    "Gaussian": (Gaussian, 0.6, 5204),
}
# The bootstraps' seed
BOOT_SEED = 5210


def _sample(name):
    family, truth, seed = FAMILIES[name]
    return family.from_params([truth], MARGINS).random(N, random_state=seed)


@pytest.fixture(scope="module")
def fits():
    return {
        name: FAMILIES[name][0].fit(_sample(name), margins=KM)
        for name in FAMILIES
    }


def _frank_log_density(theta, u, v):
    a = 1.0 - anp.exp(-theta)
    b = (1.0 - anp.exp(-theta * u)) * (1.0 - anp.exp(-theta * v))
    return anp.log(theta) + anp.log(a) - theta * (u + v) - 2.0 * anp.log(a - b)


def _gaussian_log_density(rho, a, b):
    # in the normal scores a, b of u, v
    return -0.5 * anp.log(1.0 - rho**2) - (
        rho**2 * (a**2 + b**2) - 2.0 * rho * a * b
    ) / (2.0 * (1.0 - rho**2))


def _log_density(name):
    """The family's log density for autograd: of ``(theta, u, v)``."""
    if name == "Frank":
        return _frank_log_density
    if name == "Gaussian":
        return None  # in the normal scores: see _score_parts
    family = FAMILIES[name][0]
    return lambda theta, u, v: anp.log(family.pdf(u, v, theta))


def _score_parts(log_density, theta, u, v):
    """``(s, ds/dtheta, ds/du, ds/dv)`` at each row, by autograd."""
    t = np.full(u.size, theta)
    if log_density is None:
        a, b = ndtri(u), ndtri(v)
        s = elementwise_grad(_gaussian_log_density, 0)
        # d/du = d/da / phi(a)
        return (
            s(t, a, b),
            elementwise_grad(s, 0)(t, a, b),
            elementwise_grad(s, 1)(t, a, b) / _phi(a),
            elementwise_grad(s, 2)(t, a, b) / _phi(b),
        )
    s = elementwise_grad(log_density, 0)
    return (
        s(t, u, v),
        elementwise_grad(s, 0)(t, u, v),
        elementwise_grad(s, 1)(t, u, v),
        elementwise_grad(s, 2)(t, u, v),
    )


def _phi(z):
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def _proposition_variance(log_density, theta, x):
    """Proposition 2.1 of Genest, Ghoudi and Rivest written out."""
    n = x.shape[0]
    u, v = (
        (np.argsort(np.argsort(x[:, d])) + 1.0) / (n + 1.0) for d in (0, 1)
    )
    s, s_theta, s_u, s_v = _score_parts(log_density, theta, u, v)
    beta = -np.mean(s_theta)
    w1 = (u[:, None] <= u[None, :]).astype(float) @ s_u / n
    w2 = (v[:, None] <= v[None, :]).astype(float) @ s_v / n
    return np.var(s + w1 + w2, ddof=1) / beta**2 / n


@pytest.mark.parametrize("name", list(FAMILIES))
def test_rank_covariance_is_the_propositions(fits, name):
    model = fits[name]
    expected = _proposition_variance(
        _log_density(name), model.params[0], _sample(name)
    )
    cov = model.covariance()
    assert cov.shape == (1, 1)
    assert cov[0, 0] == pytest.approx(expected, rel=1e-4)
    assert model.standard_errors()[0] == pytest.approx(
        np.sqrt(expected), rel=1e-4
    )
    # The margins have no parameters
    np.testing.assert_array_equal(model.covariance(margins=True), cov)


@pytest.mark.parametrize("name", list(FAMILIES))
def test_rank_standard_error_agrees_with_the_bootstrap(fits, name):
    model = fits[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        boot = model.standard_errors(
            method="bootstrap", n_boot=400, random_state=BOOT_SEED
        )
    assert model.standard_errors()[0] == pytest.approx(boot[0], rel=0.15)


def test_rank_covariance_of_a_rotated_family():
    # A Clayton rotated by 180 degrees on (u, v) is the Clayton on
    # (1 - u, 1 - v), the ranks of the negated data: the same estimate,
    # and the proposition's variance with that density. (The standard
    # error of the fit to the negated data differs by a term of order
    # 1 / n, 0.35% here: the proposition's W counts the row itself among
    # those ranked at or above it, which a reflection moves to the other
    # side.)
    x = _sample("Gumbel")
    rotated = Clayton.fit(x, margins=KM, rotation=180)
    plain = Clayton.fit(-x, margins=KM)
    assert rotated.params[0] == pytest.approx(plain.params[0], rel=1e-6)

    def log_density(theta, u, v):
        return anp.log(Clayton.pdf(1.0 - u, 1.0 - v, theta))

    expected = _proposition_variance(log_density, rotated.params[0], x)
    assert rotated.covariance()[0, 0] == pytest.approx(expected, rel=1e-4)


def test_rank_covariance_of_two_parameters():
    x = StudentT.from_params([0.6, 5.0], MARGINS).random(N, random_state=5205)
    model = StudentT.fit(x, margins=KM)
    cov = model.covariance()
    assert cov.shape == (2, 2)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-8)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    lo, hi = model.param_cb("rho")
    assert -1 < lo < model.params[0] < hi < 1
    lo, hi = model.param_cb("nu")
    assert 0 < lo < model.params[1] < hi


def test_rank_bounds_follow_from_the_covariance(fits):
    model = fits["Clayton"]
    se = model.standard_errors()[0]
    theta = model.params[0]
    # theta > 0: the Wald bound on its log
    expected = theta * np.exp(
        np.array([-1, 1]) * 1.959963984540054 * se / theta
    )
    np.testing.assert_allclose(model.param_cb("theta"), expected, rtol=1e-9)
    one = model.param_cb("theta", alpha_ci=0.025, bound="upper")
    np.testing.assert_allclose(one, expected[1], rtol=1e-9)


# -- what the rank-based covariance refuses -----------------------------------
def _censored(x):
    limit = np.array([14.0, 24.0])
    return np.minimum(x, limit), (x > limit).astype(int)


def _refusal_cases():
    x = _sample("Clayton")
    xc, c = _censored(x)
    window = np.empty(x.shape + (2,))
    window[..., 0], window[..., 1] = 0.5, np.inf
    kept = (x > 0.5).all(axis=1)
    return {
        "censored": (
            lambda: Clayton.fit(xc, c=c, margins=KM),
            "the data are censored",
        ),
        "truncated": (
            lambda: Clayton.fit(x[kept], t=window[kept], margins=KM),
            "the data are truncated",
        ),
        "counts": (
            lambda: Clayton.fit(x, n=np.full(N, 2), margins=KM),
            "counts `n` other than 1",
        ),
        "ties": (
            lambda: Clayton.fit(np.round(x, 0), margins=KM),
            r"series 0 has tied values \(\d+ of its 500 rows",
        ),
        "one parametric margin": (
            lambda: Clayton.fit(x, margins=[Weibull, KaplanMeier]),
            "one margin is parametric and the other is not",
        ),
        "not the empirical CDF": (
            lambda: Clayton.fit(x, margins=[NelsonAalen, NelsonAalen]),
            "margin 0 is not the empirical CDF.*`KaplanMeier`",
        ),
        "margin given fitted": (
            lambda: Clayton.fit(
                x, margins=[KaplanMeier.fit(x[:, 0]), KaplanMeier]
            ),
            "passed already fitted",
        ),
    }


@pytest.mark.parametrize("case", list(_refusal_cases()))
def test_default_refuses_and_names_the_bootstrap(case):
    fit, reason = _refusal_cases()[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit()
    calls = (
        model.covariance,
        model.standard_errors,
        lambda: model.param_cb("theta"),
        model.kendall_tau_cb,
    )
    for call in calls:
        with pytest.raises(ValueError, match=reason) as caught:
            call()
        assert "`method='bootstrap'`" in str(caught.value)
        assert "Genest, Ghoudi and Rivest 1995" in str(caught.value)
    assert "covariance" not in model.to_dict()
    # ... which the bootstrap gives
    lo, hi = model.param_cb(
        "theta", method="bootstrap", n_boot=50, random_state=BOOT_SEED
    )
    assert 0 < lo < model.params[0] < hi


def test_wald_bound_on_a_joint_probability_is_refused(fits):
    model = fits["Clayton"]
    for on in ("sf", "ff"):
        with pytest.raises(ValueError, match="step function") as caught:
            model.cb([8.0, 18.0], on=on)
        assert "`cb(x, method='bootstrap')`" in str(caught.value)


def test_bootstrap_needs_data(fits):
    model = fits["Clayton"]
    restored = surv.from_dict(model.to_dict())
    with pytest.raises(ValueError, match="restored from a dictionary"):
        restored.param_cb("theta", method="bootstrap")
    given = Clayton.from_params([2.0], MARGINS)
    with pytest.raises(ValueError, match="`from_params`"):
        given.cb([8.0, 18.0], method="bootstrap")
    for bad in (1, 0, 2.5, True):
        with pytest.raises(ValueError, match="'n_boot'"):
            model.param_cb("theta", method="bootstrap", n_boot=bad)
    with pytest.raises(ValueError, match="'method'"):
        model.covariance(method="wald")


# -- the case bootstrap ---------------------------------------------------
def test_bootstrap_is_seeded_and_shaped(fits):
    model = fits["Frank"]
    kw = {"method": "bootstrap", "n_boot": 60, "random_state": BOOT_SEED}
    two = model.param_cb("theta", **kw)
    np.testing.assert_array_equal(two, model.param_cb("theta", **kw))
    assert two.shape == (2,) and two[0] < model.params[0] < two[1]
    lower = model.param_cb("theta", bound="lower", alpha_ci=0.025, **kw)
    upper = model.param_cb("theta", bound="upper", alpha_ci=0.025, **kw)
    assert lower.shape == upper.shape == (1,)
    np.testing.assert_array_equal(np.r_[lower, upper], two)
    other = model.param_cb(
        "theta", method="bootstrap", n_boot=60, random_state=BOOT_SEED + 1
    )
    assert not np.array_equal(other, two)
    points = np.array([[8.0, 18.0], [5.0, 15.0], [12.0, 22.0]])
    for on in ("sf", "ff"):
        value = getattr(model, on)(points)
        bounds = model.cb(points, on=on, **kw)
        assert bounds.shape == (3, 2)
        assert np.all((bounds[:, 0] < value) & (value < bounds[:, 1]))
        assert np.all((bounds > 0) & (bounds < 1))
        one = model.cb(points, on=on, bound="lower", alpha_ci=0.025, **kw)
        np.testing.assert_array_equal(one, bounds[:, 0])
        np.testing.assert_array_equal(
            model.cb(points[0], on=on, **kw), bounds[0]
        )
    cov = model.covariance(**kw)
    assert cov.shape == (1, 1) and cov[0, 0] > 0
    np.testing.assert_array_equal(
        model.standard_errors(**kw), np.sqrt(np.diag(cov))
    )


def test_bootstrap_with_one_parametric_margin():
    x = _sample("Clayton")
    model = Clayton.fit(x, margins=[Weibull, KaplanMeier])
    kw = {"method": "bootstrap", "n_boot": 100, "random_state": BOOT_SEED}
    cov = model.covariance(margins=True, **kw)
    # theta, then the Weibull margin's alpha and beta
    assert cov.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    np.testing.assert_array_equal(model.covariance(**kw), cov[:1, :1])
    # The margin's own standard errors, from its own fit, within the
    # bootstrap's error (100 refits: about 7%; 25% allowed, set before
    # the values were seen)
    np.testing.assert_allclose(
        np.sqrt(np.diag(cov))[1:],
        np.sqrt(np.diag(model.margins[0].covariance())),
        rtol=0.25,
    )
    sf = model.sf([8.0, 18.0])
    lo, hi = model.cb([8.0, 18.0], **kw)
    assert 0 < lo < sf < hi < 1


def test_bootstrap_of_censored_data_and_a_known_margin():
    x = _sample("Clayton")
    xc, c = _censored(x)
    known = Weibull.from_params([10.0, 2.0])
    model = Clayton.fit(xc, c=c, margins=[known, KaplanMeier])
    kw = {"method": "bootstrap", "n_boot": 60, "random_state": BOOT_SEED}
    cov = model.covariance(margins=True, **kw)
    # A margin passed already fitted is known: no variance
    assert cov.shape == (3, 3)
    np.testing.assert_array_equal(cov[1:, :], 0.0)
    assert cov[0, 0] > 0


def test_bootstrap_of_a_parametric_fit_agrees_with_its_wald_bound():
    x = _sample("Gumbel")
    for how in ("IFM", "MLE"):
        model = Gumbel.fit(x, margins=[Weibull, Weibull], how=how)
        boot = model.standard_errors(
            margins=True,
            method="bootstrap",
            n_boot=200,
            random_state=BOOT_SEED,
        )
        # 200 refits: about 5%; 20% allowed, set before the values were seen
        np.testing.assert_allclose(
            boot, model.standard_errors(margins=True), rtol=0.2
        )


def test_failed_refits_are_counted_in_a_warning(fits, monkeypatch):
    model = fits["Clayton"]
    family = type(model.copula)
    real = family.fit
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] % 2:
            raise RuntimeError("no fit")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(family, "fit", flaky)
    with pytest.warns(
        surv.utils.no_maximum.BootstrapRefitWarning,
        match="10 of the 20 bootstrap refits",
    ):
        bounds = model.param_cb(
            "theta", method="bootstrap", n_boot=20, random_state=BOOT_SEED
        )
    assert np.all(np.isfinite(bounds))


# -- the dependence measures ----------------------------------------------
@pytest.mark.parametrize("measure", ["kendall_tau", "spearman_rho"])
def test_measure_bound_is_the_parameters_carried_through(fits, measure):
    for name, model in fits.items():
        at = getattr(model.copula, measure)
        expected = [at(b) for b in model.param_cb(model.parameter_names[0])]
        bounds = getattr(model, measure + "_cb")()
        np.testing.assert_allclose(bounds, expected, rtol=1e-12)
        value = getattr(model, measure)()
        assert bounds[0] < value < bounds[1]
        for side, i in (("lower", 0), ("upper", 1)):
            one = getattr(model, measure + "_cb")(alpha_ci=0.025, bound=side)
            assert np.shape(one) == ()
            np.testing.assert_allclose(one, expected[i], rtol=1e-12)


def test_measure_bound_where_the_measure_falls_with_the_parameter():
    x = _sample("Clayton")
    x[:, 1] = -x[:, 1]
    model = Clayton.fit(x, margins=KM, rotation=90)
    tau = model.kendall_tau()
    assert tau < 0
    lo, hi = model.kendall_tau_cb()
    assert -1 < lo < tau < hi < 0
    theta_lo, theta_hi = model.param_cb("theta")
    assert lo == pytest.approx(model.copula.kendall_tau(theta_hi), rel=1e-12)
    assert hi == pytest.approx(model.copula.kendall_tau(theta_lo), rel=1e-12)
    assert model.kendall_tau_cb(alpha_ci=0.025, bound="lower") == (
        pytest.approx(lo, rel=1e-12)
    )


def test_measure_bound_of_two_parameters_and_of_none():
    x = StudentT.from_params([0.6, 5.0], MARGINS).random(N, random_state=5205)
    model = StudentT.fit(x, margins=KM)
    # Kendall's tau of a Student-t is (2 / pi) arcsin(rho): its bound is
    # the delta method's on Fisher's z
    rho = model.params[0]
    tau = 2.0 / np.pi * np.arcsin(rho)
    se = 2.0 / np.pi / np.sqrt(1.0 - rho**2) * model.standard_errors()[0]
    z = np.arctanh(tau) + np.array([-1, 1]) * 1.959963984540054 * se / (
        1.0 - tau**2
    )
    np.testing.assert_allclose(model.kendall_tau_cb(), np.tanh(z), rtol=1e-6)
    lo, hi = model.spearman_rho_cb()
    assert -1 < lo < model.spearman_rho() < hi < 1
    kw = {"method": "bootstrap", "n_boot": 40, "random_state": BOOT_SEED}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo, hi = model.kendall_tau_cb(**kw)
    assert lo < tau < hi
    independent = Independence.fit(x, margins=KM)
    np.testing.assert_array_equal(independent.kendall_tau_cb(), [0.0, 0.0])
    assert independent.spearman_rho_cb(bound="upper") == 0.0
    with pytest.raises(ValueError, match="'method'"):
        model.kendall_tau_cb(method="lr")


def test_measure_bootstrap_bound(fits):
    model = fits["Gaussian"]
    kw = {"method": "bootstrap", "n_boot": 60, "random_state": BOOT_SEED}
    lo, hi = model.kendall_tau_cb(**kw)
    rho = model.param_cb("rho", **kw)
    # A percentile interval is carried through a monotone function
    np.testing.assert_allclose(
        [lo, hi], [model.copula.kendall_tau(r) for r in rho], rtol=1e-12
    )


# -- saving ---------------------------------------------------------------
def test_dict_keeps_the_covariance_and_how_it_was_computed(fits):
    model = fits["Gumbel"]
    saved = json.loads(json.dumps(model.to_dict()))
    assert saved["covariance_method"] == "rank"
    restored = surv.from_dict(saved)
    np.testing.assert_array_equal(restored.covariance(), model.covariance())
    np.testing.assert_array_equal(
        restored.param_cb("theta"), model.param_cb("theta")
    )
    np.testing.assert_array_equal(
        restored.kendall_tau_cb(), model.kendall_tau_cb()
    )
    assert restored.to_dict() == model.to_dict()
    with pytest.raises(ValueError, match="step function"):
        restored.cb([8.0, 18.0])


@pytest.mark.parametrize(
    "how, expected", [("IFM", "godambe"), ("MLE", "information")]
)
def test_dict_names_a_parametric_fits_covariance(how, expected):
    model = Clayton.fit(
        _sample("Clayton"), margins=[Weibull, Weibull], how=how
    )
    saved = model.to_dict()
    assert saved["covariance_method"] == expected
    # A dictionary written before the key was stored
    del saved["covariance_method"]
    restored = surv.from_dict(saved)
    assert restored.to_dict()["covariance_method"] == expected
