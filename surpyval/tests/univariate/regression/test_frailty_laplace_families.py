"""The inverse Gaussian and positive stable shared frailties.

Checked from first principles, with no other package as the reference:

* each family's ``log I(D, H) = log E[u^D exp(-uH)]`` against direct
  numerical integration over the frailty's density (the inverse
  Gaussian's closed form; the positive stable's by Kanter's integral
  representation), and the positive stable's against arbitrary-precision
  derivatives of its Laplace transform ``exp(-s^alpha)`` (mpmath);
* the marginal log-likelihood of a parametric and of a Cox fit against
  brute-force integration over each group's frailty on a small data set;
* ``theta -> 0`` against the fit without frailty (``WeibullPH``,
  ``CoxPH`` with Breslow's ties);
* the Cox fit's EM, run with the gamma family's E-step, against the
  established gamma ``CoxFrailty`` fit (Breslow's ties), and its analytic
  information against a numerical Hessian of the full likelihood.

The tolerances were set before the checks were run.
"""

import json
import warnings

import autograd
import autograd.numpy as anp
import mpmath
import numpy as np
import pytest
from scipy.integrate import quad

import surpyval as sp
from surpyval import CoxFrailty, CoxPH, Frailty, Weibull, WeibullPH
from surpyval.recurrent import CrowAMSAA
from surpyval.univariate.regression import CoxFrailtyModel, FrailtyModel
from surpyval.univariate.regression.frailty import _cox_laplace
from surpyval.univariate.regression.frailty.families import (
    frailty_cv2,
    inverse_gaussian_kendall_tau,
    inverse_gaussian_log_integral,
    kendall_tau,
    laplace_log_integral,
    positive_stable_log_integral,
    posterior_mean,
)

FAMILIES = ("inverse_gaussian", "positive_stable")


# -- the densities, written from their definitions ---------------------------


def ig_density(u, theta):
    """IG(mean 1, lambda = 1 / theta)."""
    lam = 1.0 / theta
    return np.sqrt(lam / (2 * np.pi * u**3)) * np.exp(
        -lam * (u - 1) ** 2 / (2 * u)
    )


def ps_density(u, alpha):
    """The positive stable density with Laplace transform exp(-s^alpha),
    by Kanter's (1975) integral representation."""

    def a(phi):
        return (
            (np.sin(alpha * phi) / np.sin(phi)) ** (1 / (1 - alpha))
            * np.sin((1 - alpha) * phi)
            / np.sin(alpha * phi)
        )

    k = u ** (-alpha / (1 - alpha))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val, _ = quad(
            lambda p: a(p) * np.exp(-k * a(p)),
            0,
            np.pi,
            limit=200,
            epsabs=0,
            epsrel=1e-12,
        )
    return alpha / (1 - alpha) * u ** (-1 / (1 - alpha)) / np.pi * val


_EDGES = [0, 0.01, 0.1, 0.5, 1, 2, 5, 20, 100, np.inf]


def direct(f):
    """``int_0^inf f(u) du``, in pieces."""
    total = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lo, hi in zip(_EDGES[:-1], _EDGES[1:]):
            v, _ = quad(f, lo, hi, epsabs=0, epsrel=1e-11, limit=400)
            total += v
    return total


def density(family, theta):
    if family == "inverse_gaussian":
        return lambda u: ig_density(u, theta)
    return lambda u: ps_density(u, 1 - theta)


CASES = [
    (0, 1.3, 0.5),
    (1, 1.3, 0.5),
    (3, 0.7, 0.2),
    (5, 2.0, 0.6),
    (10, 4.0, 0.1),
    (2, 0.05, 0.3),
]


@pytest.mark.parametrize("D, H, theta", CASES)
def test_inverse_gaussian_integral_against_its_density(D, H, theta):
    value = float(inverse_gaussian_log_integral(D, H, theta))
    g = density("inverse_gaussian", theta)
    reference = np.log(direct(lambda u: u**D * np.exp(-u * H) * g(u)))
    assert value == pytest.approx(reference, rel=1e-9, abs=1e-11)


@pytest.mark.parametrize("D, H, theta", CASES)
def test_positive_stable_integral_against_derivatives_and_density(D, H, theta):
    value = float(positive_stable_log_integral(D, H, theta))
    mpmath.mp.dps = 40
    alpha = mpmath.mpf(1) - mpmath.mpf(theta)
    derivative = mpmath.diff(lambda s: mpmath.exp(-(s**alpha)), H, D)
    exact = float(mpmath.log((-1) ** D * derivative))
    assert value == pytest.approx(exact, rel=1e-12, abs=1e-13)
    g = density("positive_stable", theta)
    by_density = np.log(direct(lambda u: u**D * np.exp(-u * H) * g(u)))
    assert value == pytest.approx(by_density, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("family", FAMILIES)
def test_derivatives_are_the_posterior_moments(family):
    # d/dH log I(D, H) = -I(D + 1, H) / I(D, H), the posterior mean, by
    # autograd through the closed form; and in theta against a central
    # difference.
    D = np.array([0.0, 1, 3, 5, 10, 2])
    H = np.array([1.3, 1.3, 0.7, 2.0, 4.0, 0.05])
    theta = 0.4
    d_h = autograd.elementwise_grad(
        lambda h: laplace_log_integral(family, D, h, theta)
    )(H)
    np.testing.assert_allclose(
        d_h, -posterior_mean(family, D, H, theta), rtol=1e-12
    )

    def total(t):
        return anp.sum(laplace_log_integral(family, D, H, t))

    eps = 1e-6
    central = (total(theta + eps) - total(theta - eps)) / (2 * eps)
    assert float(autograd.grad(total)(theta)) == pytest.approx(
        central, rel=1e-7
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_no_frailty_limit(family):
    D = np.array([0.0, 1, 3, 5, 10, 2])
    H = np.array([1.3, 1.3, 0.7, 2.0, 4.0, 0.05])
    for theta in (1e-12, 0.0):
        np.testing.assert_allclose(
            laplace_log_integral(family, D, H, theta), -H, rtol=0, atol=1e-9
        )


def test_fractional_counts_are_refused():
    with pytest.raises(ValueError, match="whole number of events"):
        inverse_gaussian_log_integral(1.5, 1.0, 0.5)
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(10), 3)
    x = rng.exponential(size=30)
    n = np.full(30, 1.5)
    # (the data handler refuses a fractional count before the family does)
    with pytest.raises(ValueError):
        Frailty(Weibull, family="positive_stable").fit(x, n=n, groups=g)


def test_kendall_tau():
    # The inverse Gaussian's closed form against 4 int s L L'' ds - 1
    def by_quadrature(theta):
        def integrand(v):
            s = np.exp(v)
            logs = inverse_gaussian_log_integral(
                np.array([0.0, 2.0]), s, theta
            )
            return float(np.exp(2 * v + logs.sum()))

        total, _ = quad(integrand, -60, 60, limit=400)
        return 4 * total - 1

    for theta in (0.01, 0.019, 0.021, 0.1, 1.0, 5.0, 50.0):
        assert inverse_gaussian_kendall_tau(theta) == pytest.approx(
            by_quadrature(theta), rel=1e-9
        )

    # The positive stable's tau is theta (= 1 - alpha), by the same
    # integral of its own transform.
    def stable(theta):
        def integrand(v):
            s = np.exp(v)
            logs = positive_stable_log_integral(np.array([0.0, 2.0]), s, theta)
            return float(np.exp(2 * v + logs.sum()))

        total, _ = quad(integrand, -80, 80, limit=400)
        return 4 * total - 1

    for theta in (0.1, 0.3, 0.6):
        assert kendall_tau("positive_stable", theta) == theta
        assert stable(theta) == pytest.approx(theta, rel=1e-8)
    assert frailty_cv2("positive_stable", 0.3) == np.inf
    assert frailty_cv2("inverse_gaussian", 0.3) == 0.3


# -- data --------------------------------------------------------------------


def stable_draw(rng, alpha, size):
    """Positive stable draws (Laplace transform exp(-s^alpha)), by
    Kanter's representation: (A(U) / E)^((1 - alpha) / alpha)."""
    U = rng.uniform(0, np.pi, size)
    E = rng.exponential(1, size)
    A = (
        (np.sin(alpha * U) / np.sin(U)) ** (1 / (1 - alpha))
        * np.sin((1 - alpha) * U)
        / np.sin(alpha * U)
    )
    return (A / E) ** ((1 - alpha) / alpha)


def grouped(seed, family, theta, G=60, per=4, beta=0.5):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(G), per)
    if theta == 0:
        u = np.ones(G)
    elif family == "positive_stable":
        u = stable_draw(rng, 1 - theta, G)
    else:
        u = rng.wald(1.0, 1.0 / theta, G)
    z = rng.binomial(1, 0.5, g.size).astype(float)
    E = rng.exponential(size=g.size)
    t = 10 * (E / (u[g] * np.exp(beta * z))) ** (1 / 1.5)
    c = (t > 30).astype(int)
    return np.minimum(t, 30), z[:, None], c, g


# -- the parametric fit ------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_parametric_marginal_likelihood_by_brute_force(family):
    x, Z, c, g = grouped(5, family, 0.3, G=5, per=3)
    fitter = Frailty(Weibull, family=family)
    nat = np.array([9.0, 1.4, 0.4, 0.35])
    inv = np.unique(g, return_inverse=True)[1]
    value = -fitter._neg_ll_natural(nat, x, c, np.ones(x.size), Z, inv, 1)
    gdens = density(family, nat[-1])
    reference = 0.0
    for k in range(5):
        rows = inv == k
        eta = np.exp(0.4 * Z[rows, 0])
        h = Weibull.hf(x[rows], 9.0, 1.4) * eta
        H = Weibull.Hf(x[rows], 9.0, 1.4) * eta
        ev = c[rows] == 0

        def integrand(u, h=h, H=H, ev=ev):
            return np.prod((u * h)[ev]) * np.exp(-u * H.sum()) * gdens(u)

        reference += np.log(direct(integrand))
    assert value == pytest.approx(reference, rel=1e-8)


@pytest.mark.parametrize("family", FAMILIES)
def test_parametric_no_frailty_is_the_ph_fit(family):
    x, Z, c, g = grouped(3, family, 0.0, G=50)
    with pytest.warns(UserWarning, match="boundary 0"):
        model = Frailty(Weibull, family=family).fit(x, Z=Z, c=c, groups=g)
    ph = WeibullPH.fit(x, Z=Z, c=c)
    assert model.theta == 0.0
    np.testing.assert_allclose(model.params[:-1], ph.params, rtol=1e-5)
    assert model._neg_ll == pytest.approx(ph.neg_ll(), abs=1e-7)
    se = model.standard_errors()
    assert np.isnan(se[-1])
    np.testing.assert_allclose(se[:-1], ph.standard_errors(), rtol=1e-3)
    # the likelihood at a theta of 1e-12, at the PH fit's parameters
    fitter = Frailty(Weibull, family=family)
    inv = np.unique(g, return_inverse=True)[1]
    nat = np.r_[ph.params, 1e-12]
    order = np.lexsort([inv, Z[:, 0], c, x])
    tiny = fitter._neg_ll_natural(
        nat, x[order], c[order], np.ones(x.size), Z[order], inv[order], 1
    )
    assert tiny == pytest.approx(ph.neg_ll(), abs=1e-8)


def test_parametric_positive_stable_predictions():
    x, Z, c, g = grouped(1, "positive_stable", 0.3, G=60)
    model = Frailty(Weibull, family="positive_stable").fit(
        x, Z=Z, c=c, groups=g
    )
    assert 0 < model.theta < 1
    a = 1 - model.theta
    t = np.array([5.0, 10.0, 20.0])
    s = np.exp(model.beta[0]) * Weibull.Hf(t, *model.dist_params)
    # the marginal is proportional hazards: S = exp(-s^a)
    np.testing.assert_allclose(model.sf(t, [1.0]), np.exp(-(s**a)), rtol=1e-12)
    h = np.exp(model.beta[0]) * Weibull.hf(t, *model.dist_params)
    np.testing.assert_allclose(
        model.hf(t, [1.0]), a * s ** (a - 1) * h, rtol=1e-10
    )
    lo, hi = model.param_cb("theta")
    assert 0 < lo < model.theta < hi < 1
    assert model.kendall_tau == model.theta
    assert model.frailty_variance == np.inf
    restored = FrailtyModel.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_allclose(restored.sf(t, [1.0]), model.sf(t, [1.0]))
    assert "positive stable" in repr(model)


def test_penalised_fit_refuses_the_new_families():
    x, Z, c, g = grouped(1, "inverse_gaussian", 0.5)
    with pytest.raises(ValueError, match="penalised frailty fit"):
        Frailty(Weibull, family="inverse_gaussian").fit(
            x, Z=Z, c=c, groups=g, penalizer=0.1
        )


# -- the Cox fit -------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_cox_no_frailty_is_the_breslow_cox_fit(family):
    x, Z, c, g = grouped(3, family, 0.0, G=50)
    with pytest.warns(UserWarning, match="boundary 0"):
        model = CoxFrailty.fit(x, Z=Z, c=c, groups=g, family=family)
    cox = CoxPH.fit(x, Z=Z, c=c, tie_method="breslow")
    assert model.theta == 0.0
    np.testing.assert_allclose(model.beta, cox.beta, rtol=1e-8)
    se = model.standard_errors()
    assert np.isnan(se[-1])
    np.testing.assert_allclose(se[:-1], cox.standard_errors(), rtol=1e-6)
    assert model.log_likelihood == model.log_likelihood_no_frailty


def _gamma_log_integral(family, D, H, theta):
    """The gamma frailty's ``log I(D, H)`` (mean 1, variance theta)."""
    from surpyval.univariate.regression.frailty.frailty_fitter import (
        _group_frailty_ll,
    )

    return _group_frailty_ll(np.asarray(D, float), H, theta, None, "gamma")


def _gamma_posterior(family, D, H, theta):
    return (1.0 + D * theta) / (1.0 + H * theta)


def test_cox_em_reproduces_the_gamma_fit(monkeypatch):
    # The EM of the new families, given the gamma's E-step and integral,
    # is the established gamma CoxFrailty fit (Breslow's ties, where R's
    # integrated likelihood is the marginal likelihood up to a constant).
    x, Z, c, g = grouped(2, "inverse_gaussian", 0.6, G=60)
    gamma = CoxFrailty.fit(x, Z=Z, c=c, groups=g, tie_method="breslow")
    monkeypatch.setattr(
        _cox_laplace, "laplace_log_integral", _gamma_log_integral
    )
    monkeypatch.setattr(_cox_laplace, "posterior_mean", _gamma_posterior)
    monkeypatch.setitem(
        _cox_laplace._LOG_THETA_BOUNDS,
        "inverse_gaussian",
        (np.log(1e-6), np.log(100.0)),
    )
    generic = CoxFrailty.fit(
        x, Z=Z, c=c, groups=g, family="inverse_gaussian", tie_method="breslow"
    )
    assert generic.theta == pytest.approx(gamma.theta, rel=1e-5)
    np.testing.assert_allclose(generic.beta, gamma.beta, rtol=1e-5)
    np.testing.assert_allclose(
        list(generic.frailties.values()),
        list(gamma.frailties.values()),
        rtol=1e-5,
    )


def _full_log_likelihood(family, x, Z, c, inv, params, times):
    """The marginal log-likelihood in (beta, theta, log h), directly."""
    p = Z.shape[1]
    beta, theta, h = params[:p], params[p], np.exp(params[p + 1 :])
    lin = Z @ beta
    k = np.searchsorted(times, x, side="right") - 1
    H0 = np.where(k >= 0, np.cumsum(h)[np.maximum(k, 0)], 0.0)
    event = c == 0
    ke = np.searchsorted(times, x[event])
    G = inv.max() + 1
    D = np.bincount(inv, weights=event.astype(float), minlength=G)
    A = np.bincount(inv, weights=np.exp(lin) * H0, minlength=G)
    return np.sum(np.log(h[ke]) + lin[event]) + np.sum(
        laplace_log_integral(family, D, A, theta)
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_cox_covariance_against_a_numerical_hessian(family):
    from surpyval.utils.linalg import numerical_hessian

    x, Z, c, g = grouped(7, family, 0.3, G=15, per=3)
    model = CoxFrailty.fit(x, Z=Z, c=c, groups=g, family=family, center=True)
    inv = np.unique(g, return_inverse=True)[1]
    times = model.x[model.h0 > 0]
    h = model.h0[model.h0 > 0]
    # the baseline at the covariate mean (center=True), Z centred to match
    Zc = Z - model.center
    point = np.r_[model.beta, model.theta, np.log(h)]

    def nll(v):
        return -_full_log_likelihood(family, x, Zc, c, inv, v, times)

    Hm = numerical_hessian(nll, point, step=1e-4 * np.ones(point.size))
    # (in log h: the Schur complement of (beta, theta) does not depend on
    # how the jumps are parameterised)
    cov = np.linalg.inv(Hm)[:2, :2]
    np.testing.assert_allclose(model.covariance(), cov, rtol=1e-4)
    # and the fit is a maximum of it: the gradient is 0
    eps = 1e-6
    grad = [
        (nll(point + eps * e) - nll(point - eps * e)) / (2 * eps)
        for e in np.eye(point.size)
    ]
    assert np.max(np.abs(grad)) < 1e-4


@pytest.mark.parametrize("family", FAMILIES)
def test_cox_marginal_likelihood_by_brute_force(family):
    # (theta fixed: five groups are too few to estimate it)
    x, Z, c, g = grouped(5, family, 0.3, G=5, per=3)
    model = CoxFrailty.fit(x, Z=Z, c=c, groups=g, family=family, theta=0.3)
    gdens = density(family, model.theta)
    eta = np.exp(model.beta[0] * Z[:, 0])
    H = model._H0(x) * eta
    h = model._h0(x) * eta
    reference = 0.0
    for k in np.unique(g):
        rows = g == k
        ev = c[rows] == 0

        def integrand(u, rows=rows, ev=ev):
            return (
                np.prod((u * h[rows])[ev])
                * np.exp(-u * H[rows].sum())
                * gdens(u)
            )

        reference += np.log(direct(integrand))
    assert model.log_likelihood == pytest.approx(reference, rel=1e-8)


def test_cox_refuses_efron_with_ties_and_theta_out_of_range():
    x, Z, c, g = grouped(1, "inverse_gaussian", 0.5)
    x = np.round(x)
    with pytest.raises(ValueError, match="tie_method='breslow'"):
        CoxFrailty.fit(x, Z=Z, c=c, groups=g, family="inverse_gaussian")
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        CoxFrailty.fit(
            x,
            Z=Z,
            c=c,
            groups=g,
            family="positive_stable",
            tie_method="breslow",
            theta=1.2,
        )


def test_cox_round_trip_and_diagnose_refusal():
    x, Z, c, g = grouped(1, "inverse_gaussian", 0.5)
    model = CoxFrailty.fit(x, Z=Z, c=c, groups=g, family="inverse_gaussian")
    restored = CoxFrailtyModel.from_dict(
        json.loads(json.dumps(model.to_dict()))
    )
    np.testing.assert_allclose(
        restored.sf([5, 10], [1.0]), model.sf([5, 10], [1.0])
    )
    assert restored.family == "inverse_gaussian"
    with pytest.raises(ValueError, match="gamma and log-normal"):
        model.diagnose()


# -- the recurrent (shared frailty Poisson process) fit ----------------------


def recurrent(seed, family, theta, G=60, T=50.0):
    rng = np.random.default_rng(seed)
    if theta == 0:
        u = np.ones(G)
    elif family == "positive_stable":
        u = stable_draw(rng, 1 - theta, G)
    else:
        u = rng.wald(1.0, 1.0 / theta, G)
    xs, ii, cs = [], [], []
    for k in range(G):
        n = rng.poisson(u[k] * (T / 10.0) ** 1.3)
        t = np.sort(T * rng.uniform(size=n) ** (1 / 1.3))
        xs += [*t, T]
        ii += [k] * (n + 1)
        cs += [0] * n + [1]
    return np.array(xs), np.array(ii), np.array(cs)


@pytest.mark.parametrize("family", FAMILIES)
def test_recurrent_no_frailty_is_the_plain_fit(family):
    x, i, c = recurrent(4, family, 0.0)
    with pytest.warns(UserWarning, match="boundary 0"):
        model = CrowAMSAA.fit(x, i, c, frailty=family)
    plain = CrowAMSAA.fit(x, i, c)
    assert model.theta == 0.0
    np.testing.assert_allclose(model.params[:2], plain.params, rtol=1e-5)


def test_recurrent_inverse_gaussian_frailties_and_round_trip():
    from scipy.stats import geninvgauss

    x, i, c = recurrent(2, "inverse_gaussian", 0.6)
    model = CrowAMSAA.fit(x, i, c, frailty="inverse_gaussian")
    table = model.frailties()
    n, A = table["events"].to_numpy(), table["expected"].to_numpy()
    # the posterior mean of the generalised inverse Gaussian, by
    # integrating its density, against the closed-form ratio
    a, b = 1 / model.theta + 2 * A[:3], 1 / model.theta
    for k in range(3):
        dist = geninvgauss(
            n[k] - 0.5, np.sqrt(a[k] * b), scale=np.sqrt(b / a[k])
        )
        assert table["frailty"].iloc[k] == pytest.approx(dist.mean(), rel=1e-8)
    assert np.all(table["lower 95%"] < table["frailty"])
    restored = type(model).from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_allclose(
        restored.cif([10.0, 20.0]), model.cif([10.0, 20.0])
    )


def test_recurrent_positive_stable_refuses_population_means():
    x, i, c = recurrent(2, "positive_stable", 0.3, G=40, T=30.0)
    model = CrowAMSAA.fit(x, i, c, frailty="positive_stable")
    assert 0 < model.theta < 1
    with pytest.raises(ValueError, match="no finite mean"):
        model.cif([10.0])
    assert np.all(np.isfinite(model.cif([10.0], frailty=1.0)))
    assert np.all(np.isnan(model.frailties()["lower 95%"]))
    lo, hi = model.param_cb("theta")
    assert 0 < lo < model.theta < hi < 1


def test_joint_frailty_refuses_the_new_families():
    with pytest.raises(ValueError, match="family"):
        sp.JointFrailty.fit(
            [1.0, 2.0, 3.0], [1, 1, 2], [0, 1, 1], family="inverse_gaussian"
        )


# -- the pending reference from R's frailtyEM --------------------------------

_FRAILTYEM = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "reference"
    / "data"
    / "r_frailtyem.json"
)


@pytest.mark.skipif(
    not _FRAILTYEM.exists(),
    reason="the frailtyEM reference file has not been generated yet "
    "(scripts/reference/reference_r_frailtyem.R)",
)
@pytest.mark.parametrize("family", FAMILIES)
def test_against_r_frailtyem(family):  # pragma: no cover - needs the file
    from surpyval.datasets import load_kidney

    ref = json.loads(_FRAILTYEM.read_text())[family]
    df = load_kidney()
    Z = np.column_stack([df["age"], df["sex"] == 2]).astype(float)
    model = CoxFrailty.fit(
        df["time"],
        Z=Z,
        c=1 - df["status"],
        groups=df["id"],
        family=family,
        tie_method="breslow",
    )
    np.testing.assert_allclose(model.beta, ref["coef"], rtol=1e-3)
    assert model.kendall_tau == pytest.approx(
        kendall_tau(family, ref["theta"]), rel=1e-3
    )
    assert model.log_likelihood - model.log_likelihood_no_frailty == (
        pytest.approx(ref["loglik"] - ref["loglik_no_frailty"], abs=1e-3)
    )
