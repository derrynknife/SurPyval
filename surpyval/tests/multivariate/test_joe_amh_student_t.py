"""The Joe, Ali-Mikhail-Haq and Student-t copulas (#157).

References: vinecopulib (pyvinecopulib 1.0.0, the C++ library of the
VineCopula authors) for the Joe and Student-t primitives at integer
``nu``; R's mvtnorm 1.2.4 ``pmvt(algorithm = TVPACK())`` (Genz's exact
bivariate t, integer ``nu``) and mpmath (30 digits) for the t copula CDF
at non-integer ``nu``; Nelsen (2006) for the AMH closed forms. The
recovery study (200 replications) is in the #157 report.
"""

import json
import warnings

import numpy as np
import pytest
from scipy.integrate import dblquad, quad
from scipy.special import gammaln, stdtrit

import surpyval as surv
from surpyval import LogNormal, Weibull
from surpyval.multivariate import (
    AMH,
    Clayton,
    Copula,
    Gaussian,
    Joe,
    StudentT,
)
from surpyval.multivariate.parametric.data import MultivariateSurpyvalData

MARGINS = [Weibull.from_params([10.0, 2.0]), LogNormal.from_params([2.5, 0.5])]
FAMILIES = [(Joe, [2.86]), (AMH, [-0.7]), (AMH, [1.0]), (StudentT, [0.6, 3.5])]
IDS = ["Joe", "AMH-neg", "AMH-bound", "StudentT"]

# (u, v, cdf, h1 = dC/du, pdf) from vinecopulib
JOE_REF = {
    2.86: [
        (
            0.001,
            0.3,
            0.0006392230619209904,
            0.6390083350795757,
            1.4739182763604843,
        ),
        (0.3, 0.7, 0.2863766693480011, 0.9339520136565735, 0.6044985686998486),
        (0.9, 0.95, 0.8953908105235665, 0.9194275691173983, 4.145663113029136),
        (
            0.99,
            0.05,
            0.04999989999337817,
            2.8601891076424374e-05,
            0.000600961027070218,
        ),
    ],
    8.0: [
        (
            0.001,
            0.3,
            0.0009421613368102433,
            0.9419701658623135,
            0.6629292621889679,
        ),
        (
            0.3,
            0.7,
            0.2999061999150807,
            0.9989969532269337,
            0.026721164955093298,
        ),
        (0.9, 0.95, 0.8999512551188328, 0.996594501539881, 0.5428919471850618),
    ],
}
T_REF = {
    (0.5, 4.0): [
        (
            0.001,
            0.3,
            0.000839358674844297,
            0.8283487001487583,
            0.23533818888987051,
        ),
        (0.3, 0.7, 0.2614278367278644, 0.8310146901493509, 0.8317621445478687),
        (0.9, 0.95, 0.8742134179432227, 0.8896278602291718, 2.568396454334045),
        (
            0.99,
            0.05,
            0.04960997924784135,
            0.0295122574283102,
            0.3930560643291821,
        ),
    ],
    (0.95, 2.0): [
        (
            0.001,
            0.3,
            0.0009929488416042657,
            0.9927349110887886,
            0.0035652791682492823,
        ),
        (
            0.3,
            0.7,
            0.2979414267530941,
            0.9886188216105971,
            0.09260861566765269,
        ),
        (
            0.9,
            0.95,
            0.8977662378288648,
            0.9617050852017421,
            2.6296244771149127,
        ),
    ],
}


@pytest.mark.parametrize("theta", sorted(JOE_REF))
def test_joe_against_vinecopulib(theta):
    for u, v, cdf, h1, pdf in JOE_REF[theta]:
        assert float(Joe.cdf(u, v, theta)) == pytest.approx(cdf, rel=1e-10)
        assert float(Joe.du(u, v, theta)) == pytest.approx(h1, rel=1e-10)
        assert float(Joe.dv(v, u, theta)) == pytest.approx(h1, rel=1e-10)
        assert float(Joe.pdf(u, v, theta)) == pytest.approx(pdf, rel=1e-10)
    tau = {2.86: 0.5004853110353475, 8.0: 0.7832540438417558}[theta]
    upper = {2.86: 0.7257482366866286, 8.0: 0.9094922673347423}[theta]
    assert Joe.kendall_tau(theta) == pytest.approx(tau, rel=1e-13)
    assert Joe.tail_dependence(theta) == pytest.approx((0.0, upper))


def test_joe_tau_through_theta_two():
    # The digamma form is 0/0 at theta = 2, where tau = 2 - pi^2 / 6
    assert Joe.kendall_tau(2.0) == pytest.approx(2 - np.pi**2 / 6, rel=1e-14)
    around = [Joe.kendall_tau(2.0 + d) for d in (-2e-4, -5e-5, 5e-5, 2e-4)]
    assert np.all(np.diff(around) > 0)
    assert Joe.kendall_tau(1.0) == pytest.approx(0.0, abs=1e-15)


def test_joe_lower_corner_keeps_relative_accuracy():
    # C ~ u v near the lower corner: the bracket A = 1 - a b is formed as
    # log1p(-a b), not as 1 minus a number near 1
    for theta in (1.5, 4.0, 30.0):
        c = float(Joe.cdf(1e-9, 2e-9, theta))
        exact = -np.expm1(
            np.log1p(
                -(-np.expm1(theta * np.log1p(-1e-9)))
                * (-np.expm1(theta * np.log1p(-2e-9)))
            )
            / theta
        )
        assert c == pytest.approx(exact, rel=1e-12)
        assert c == pytest.approx(1e-9 * 2e-9 * theta, rel=1e-6)


def test_amh_closed_forms():
    # Nelsen (2006): tau in [-0.1817, 1/3], rho in [-0.2711, 0.4784]
    assert AMH.kendall_tau(1.0) == pytest.approx(1 / 3, rel=1e-14)
    assert AMH.kendall_tau(-1.0) == pytest.approx(
        5 / 3 - 8 / 3 * np.log(2), rel=1e-14
    )
    assert AMH.spearman_rho(1.0) == pytest.approx(
        12 * 2 * np.pi**2 / 6 - 3 * 13, rel=1e-13
    )
    for theta in (-1.0, -0.5, -0.05, 0.0, 0.03, 0.3, 0.95):
        assert AMH.kendall_tau(theta) == pytest.approx(
            Copula.kendall_tau(AMH, theta), abs=1e-12
        )
        assert AMH.spearman_rho(theta) == pytest.approx(
            Copula.spearman_rho(AMH, theta), abs=1e-12
        )


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_amh_series_meets_the_closed_form(side):
    edge = AMH._SERIES_THETA * side
    for f in (AMH.kendall_tau, AMH.spearman_rho):
        below = f(edge * (1 - 1e-9))
        above = f(edge * (1 + 1e-9))
        assert below == pytest.approx(above, rel=1e-8)


def test_amh_tail_dependence_only_at_its_upper_bound():
    assert AMH.tail_dependence(0.99) == (0.0, 0.0)
    assert AMH.tail_dependence(1.0) == (0.5, 0.0)
    # theta = 1 is the Clayton copula with theta = 1
    u, v = np.array([0.01, 0.4, 0.9]), np.array([0.2, 0.5, 0.95])
    np.testing.assert_allclose(AMH.cdf(u, v, 1.0), Clayton.cdf(u, v, 1.0))
    np.testing.assert_allclose(AMH.pdf(u, v, 1.0), Clayton.pdf(u, v, 1.0))


# (nu, rho, u, v, C): mvtnorm TVPACK (integer nu), mpmath (others)
T_CDF = [
    (4.0, 0.7, 0.2, 0.8, 0.19440077048264917),
    (1.0, -0.9, 0.99, 0.999, 0.98904564537532458),
    (30.0, 0.999, 0.001, 0.05, 0.001000000000000061),
    (2.0, 0.95, 0.05, 0.3, 0.049397466890737943),
    (0.6, 0.4, 0.97, 0.99, 0.96652085230536600231),
    (0.6, -0.6, 0.05, 0.3, 0.011353760455675627051),
    (2.5, 0.9, 0.9, 0.2, 0.19933795518588622415),
    (7.3, -0.6, 0.97, 0.99, 0.96003914500480798401),
]


@pytest.mark.parametrize("nu, rho, u, v, expected", T_CDF)
def test_t_cdf_against_exact_values(nu, rho, u, v, expected):
    got = float(StudentT.cdf(u, v, rho, nu))
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert float(StudentT.cdf(v, u, rho, nu)) == pytest.approx(got, abs=1e-12)


@pytest.mark.parametrize("params", sorted(T_REF))
def test_t_primitives_against_vinecopulib(params):
    for u, v, cdf, h1, pdf in T_REF[params]:
        assert float(StudentT.cdf(u, v, *params)) == pytest.approx(
            cdf, rel=1e-9
        )
        assert float(StudentT.du(u, v, *params)) == pytest.approx(
            h1, rel=1e-10
        )
        assert float(StudentT.pdf(u, v, *params)) == pytest.approx(
            pdf, rel=1e-10
        )
    tail = {(0.5, 4.0): 0.2531699951003227, (0.95, 2.0): 0.7995251457913304}
    assert StudentT.tail_dependence(*params) == pytest.approx(
        (tail[params], tail[params]), rel=1e-12
    )
    assert StudentT.kendall_tau(*params) == pytest.approx(
        2 / np.pi * np.arcsin(params[0]), rel=1e-14
    )


def test_t_spearman_rho():
    # 12 int int C - 3 by adaptive quadrature of the CDF
    assert StudentT.spearman_rho(0.5, 4.0) == pytest.approx(
        0.4690201700242338, abs=1e-12
    )
    # the Gaussian copula's closed form in the limit
    assert StudentT.spearman_rho(0.5, 1e9) == pytest.approx(
        Gaussian.spearman_rho(0.5), abs=1e-8
    )


def test_t_density_tends_to_the_gaussian():
    u = np.array([1e-8, 0.13, 0.5, 0.99])
    v = np.array([0.3, 0.86, 0.5, 0.995])
    # (the t tails are heavier by about x^4 / nu: 1e-4 at u = 1e-8)
    for nu, tol in ((1e6, 1e-3), (1.3e8, 1e-5), (1e12, 1e-9)):
        np.testing.assert_allclose(
            StudentT.pdf(u, v, 0.63, nu), Gaussian.pdf(u, v, 0.63), rtol=tol
        )


@pytest.mark.parametrize("family, params", FAMILIES, ids=IDS)
def test_copula_identities(family, params):
    g = np.r_[1e-6, np.linspace(0.02, 0.98, 49), 1 - 1e-6]
    U, V = np.meshgrid(g, g, indexing="ij")
    C = np.asarray(family.cdf(U.ravel(), V.ravel(), *params)).reshape(U.shape)
    # 2-increasing: every rectangle has non-negative mass
    volume = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
    assert volume.min() > -1e-12
    # Frechet bounds and the margins
    assert np.all(C <= np.minimum(U, V) + 1e-12)
    assert np.all(C >= np.maximum(U + V - 1, 0) - 1e-12)
    inner = np.linspace(0.05, 0.95, 7)
    one = np.full_like(inner, 1 - 1e-12)
    np.testing.assert_allclose(
        family.cdf(inner, one, *params), inner, atol=1e-10
    )
    np.testing.assert_allclose(
        family.cdf(one, inner, *params), inner, atol=1e-10
    )
    # h-functions and density are the derivatives of the CDF
    P, Q = np.meshgrid(inner, inner, indexing="ij")
    p, q = P.ravel(), Q.ravel()
    h = 1e-6
    fd_u = (family.cdf(p + h, q, *params) - family.cdf(p - h, q, *params)) / (
        2 * h
    )
    np.testing.assert_allclose(family.du(p, q, *params), fd_u, atol=1e-7)
    fd_c = (family.du(p, q + h, *params) - family.du(p, q - h, *params)) / (
        2 * h
    )
    np.testing.assert_allclose(family.pdf(p, q, *params), fd_c, rtol=1e-6)
    np.testing.assert_allclose(
        family.dv(p, q, *params), family.du(q, p, *params), rtol=1e-12
    )


@pytest.mark.parametrize("family, params", FAMILIES, ids=IDS)
def test_conditional_sampling_matches_the_copula(family, params):
    u, v = family.sample_uv(40_000, np.asarray(params), random_state=7)
    grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    for a in grid:
        for b in grid:
            emp = np.mean((u <= a) & (v <= b))
            assert emp == pytest.approx(
                float(family.cdf(a, b, *params)), abs=0.008
            )
    tau = family.kendall_tau(*params)
    from scipy.stats import kendalltau

    assert kendalltau(u[:8000], v[:8000]).statistic == pytest.approx(
        tau, abs=0.02
    )


def _uspace_dims(rows, us):
    """Prepared dimensions with uniform margins (u-space data)."""
    dims = []
    for d in range(2):
        c = np.array([r[d] for r in rows])
        u = np.array([us[r[d]][d] for r in rows], dtype=float)
        lo = np.array([0.2, 0.25])[d] * np.ones(len(rows))
        hi = np.array([0.6, 0.7])[d] * np.ones(len(rows))
        dims.append(
            {
                "c": c,
                "u": u,
                "ulo": lo,
                "uhi": hi,
                "logf": np.zeros(len(rows)),
                "ul": np.zeros(len(rows)),
                "ur": np.ones(len(rows)),
                "has_trunc": False,
            }
        )
    return dims


def _density_in_scale(family, params):
    """The copula's density in the scale integrated, with the map from
    the copula scale: u itself, except for the t copula, whose density is
    singular at the corners of the unit square and which is integrated
    instead as the bivariate t density in the t scale (smooth, so the
    adaptive rules stay fast)."""
    if family is not StudentT:
        return (lambda x, y: float(family.pdf(x, y, *params))), (lambda u: u)
    rho, nu = params
    const = np.exp(gammaln((nu + 2) / 2) - gammaln(nu / 2)) / (
        nu * np.pi * np.sqrt(1 - rho**2)
    )

    def joint(x, y):
        quad_form = (x * x + y * y - 2 * rho * x * y) / (nu * (1 - rho**2))
        return const * (1 + quad_form) ** (-(nu + 2) / 2)

    def to_scale(u):
        return np.inf if u >= 1 else -np.inf if u <= 0 else stdtrit(nu, u)

    return joint, to_scale


@pytest.mark.parametrize("family, params", FAMILIES, ids=IDS)
def test_censored_likelihood_is_the_integral_of_the_density(family, params):
    # every one of the 16 censoring combinations against the copula
    # density integrated over the row's region (u-space margins)
    point = (0.4, 0.55)
    region = {
        1: lambda d: (point[d], 1.0),
        -1: lambda d: (0.0, point[d]),
        2: lambda d: ((0.2, 0.6), (0.25, 0.7))[d],
    }
    combos = [(a, b) for a in (0, 1, -1, 2) for b in (0, 1, -1, 2)]
    dims = _uspace_dims(combos, {k: point for k in (0, 1, -1, 2)})
    ll = family._pair_loglik(params, dims[0], dims[1])
    joint, to_scale = _density_in_scale(family, params)
    x0, y0 = to_scale(point[0]), to_scale(point[1])
    # the density of the point's own coordinate, divided out of the
    # one-dimensional integrals (1 in the copula scale)
    fx = fy = 1.0
    if family is StudentT:
        fx = quad(lambda y: joint(x0, y), -np.inf, np.inf)[0]
        fy = quad(lambda x: joint(x, y0), -np.inf, np.inf)[0]

    def span(code, d):
        return [to_scale(e) for e in region[code](d)]

    opts = {"epsabs": 1e-12, "epsrel": 1e-10, "limit": 200}
    for (a, b), got in zip(combos, ll):
        if a == 0 and b == 0:
            want = float(family.pdf(*point, *params))
        elif a == 0:
            want = quad(lambda y: joint(x0, y), *span(b, 1), **opts)[0] / fx
        elif b == 0:
            want = quad(lambda x: joint(x, y0), *span(a, 0), **opts)[0] / fy
        else:
            (xa, xb), (ya, yb) = span(a, 0), span(b, 1)
            want = dblquad(
                lambda y, x: joint(x, y), xa, xb, ya, yb, epsabs=1e-11
            )[0]
        assert np.exp(got) == pytest.approx(want, rel=1e-6), (a, b)


@pytest.mark.parametrize(
    "family, params, how",
    [
        (Joe, [2.86], "IFM"),
        (Joe, [2.86], "MLE"),
        (AMH, [0.8], "IFM"),
        (StudentT, [0.7, 4.0], "IFM"),
    ],
)
def test_fit_recovers_the_parameters_under_censoring(family, params, how):
    X = family.from_params(params, MARGINS).random(1000, random_state=3)
    rng = np.random.default_rng(4)
    cens = np.column_stack(
        [rng.uniform(0, 25, 1000), rng.uniform(0, 30, 1000)]
    )
    c = (X > cens).astype(int)
    x = np.minimum(X, cens)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = family.fit(x, c=c, margins=[Weibull, LogNormal], how=how)
    tol = {"Joe": [0.4], "AMH": [0.2], "StudentT": [0.06, 3.0]}[family.name]
    assert np.all(np.abs(model.params - params) <= tol), model.params
    assert model.kendall_tau() == pytest.approx(
        family.kendall_tau(*params), abs=0.05
    )


def test_mle_is_at_least_as_likely_as_ifm():
    X = Joe.from_params([2.86], MARGINS).random(300, random_state=0)
    ifm = Joe.fit(X, margins=[Weibull, LogNormal])
    mle = Joe.fit(X, margins=[Weibull, LogNormal], how="MLE")
    assert mle.log_likelihood >= ifm.log_likelihood - 1e-6


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, caught


def test_t_on_gaussian_data_warns_that_nu_runs_away():
    X = Gaussian.from_params([0.6], MARGINS).random(300, random_state=1)
    model, caught = _caught(StudentT.fit, X, margins=[Weibull, LogNormal])
    assert len(caught) == 1
    message = str(caught[0].message)
    assert message.startswith("No finite maximum: nu grows without bound")
    assert "fit the Gaussian copula instead" in message
    assert caught[0].filename == __file__
    gauss = Gaussian.fit(X, margins=[Weibull, LogNormal])
    assert model.params[0] == pytest.approx(gauss.params[0], abs=1e-3)
    assert model.log_likelihood == pytest.approx(
        gauss.log_likelihood, abs=1e-3
    )


def test_t_on_t_data_is_silent():
    X = StudentT.from_params([0.6, 4.0], MARGINS).random(300, random_state=1)
    model, caught = _caught(StudentT.fit, X, margins=[Weibull, LogNormal])
    assert caught == []
    gauss = Gaussian.fit(X, margins=[Weibull, LogNormal])
    assert model.log_likelihood > gauss.log_likelihood + 1


@pytest.mark.parametrize("family", [Joe, StudentT])
def test_perfectly_dependent_data_warn_once(family):
    x1 = 10.0 * (-np.log1p(-(np.arange(1, 21) - 0.3) / 20.4)) ** 0.5
    X = np.column_stack([x1, x1 / 2])
    model, caught = _caught(family.fit, X, margins=[Weibull, Weibull])
    assert len(caught) == 1
    assert "perfectly concordant" in str(caught[0].message)
    assert np.isfinite(model.params).all()


def test_amh_beyond_its_range_returns_the_bound_silently():
    # tau = 0.5 data: more dependent than the AMH reaches (tau <= 1/3)
    X = Clayton.from_params([2.0], MARGINS).random(500, random_state=0)
    model, caught = _caught(AMH.fit, X, margins=[Weibull, LogNormal])
    assert caught == []
    assert model.params[0] > 0.99


@pytest.mark.parametrize("family, params", FAMILIES, ids=IDS)
def test_serialisation_round_trip(family, params):
    model = family.from_params(params, MARGINS)
    back = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    assert back.copula is family
    pts = [[3.0, 8.0], [12.0, 20.0]]
    np.testing.assert_array_equal(back.sf(pts), model.sf(pts))
    np.testing.assert_array_equal(back.params, model.params)


def test_fitted_t_round_trips_with_its_likelihood():
    X = StudentT.from_params([0.6, 4.0], MARGINS).random(200, random_state=2)
    model = StudentT.fit(X, margins=[Weibull, LogNormal])
    back = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    assert back.parameter_names == ["rho", "nu"]
    assert back.k == model.k == 6
    assert back.aic() == pytest.approx(model.aic(), rel=1e-12)
    np.testing.assert_allclose(back.params, model.params)


@pytest.mark.parametrize(
    "family, bad",
    [
        (Joe, [0.5]),
        (AMH, [1.5]),
        (AMH, [-1.01]),
        (StudentT, [0.5]),
        (StudentT, [1.0, 4.0]),
        (StudentT, [0.5, 0.0]),
    ],
)
def test_from_params_validates(family, bad):
    with pytest.raises(ValueError):
        family.from_params(bad, MARGINS)


def test_bounds_that_are_copulas_are_accepted():
    for family, params in ((Joe, [1.0]), (AMH, [1.0]), (AMH, [-1.0])):
        model = family.from_params(params, MARGINS)
        assert np.isfinite(model.sf([[5.0, 10.0]])).all()
    np.testing.assert_allclose(
        Joe.cdf([0.2, 0.7], [0.4, 0.9], 1.0), [0.08, 0.63], rtol=1e-14
    )


def test_data_with_counts_and_truncation():
    X = Joe.from_params([2.0], MARGINS).random(200, random_state=5)
    n = np.r_[np.full(100, 2), np.ones(100, int)]
    t = np.zeros((200, 2, 2))
    t[..., 1] = np.inf
    t[:, 0, 0] = np.minimum(1.0, X[:, 0] / 2)
    model = Joe.fit(X, n=n, t=t, margins=[Weibull, LogNormal])
    repeated = Joe.fit(
        np.repeat(X, n, axis=0),
        t=np.repeat(t, n, axis=0),
        margins=[Weibull, LogNormal],
    )
    assert model.params == pytest.approx(repeated.params, rel=1e-4)
    data = MultivariateSurpyvalData(X, n=n, t=t)
    assert data.N == 200
