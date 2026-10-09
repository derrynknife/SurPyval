"""LogNormal and Gamma hazards stay finite and accurate deep in the tail,
where 1 - F(x) (or the incomplete gamma itself) underflows."""

import warnings

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad
from scipy.special import gammaln

import surpyval as surv


@pytest.mark.parametrize("x", [5.0, 60.0, 300.0, 1000.0])
def test_lognormal_cumulative_hazard_in_the_tail(x):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Hf = surv.LogNormal.Hf(np.array([x]), 1.0, 0.5)[0]
        hf = surv.LogNormal.hf(np.array([x]), 1.0, 0.5)[0]
    assert Hf == pytest.approx(
        -stats.lognorm.logsf(x, 0.5, scale=np.e), rel=1e-12
    )
    assert np.isfinite(hf) and hf > 0


def _gamma_log_q(a, x):
    # log Q(a, x) by direct integration, with e^{-x} factored out
    v, _ = quad(lambda u: (x + u) ** (a - 1) * np.exp(-u), 0, np.inf)
    return -x + np.log(v) - gammaln(a)


@pytest.mark.parametrize(
    "a, x", [(2.0, 300.0), (3.5, 720.0), (2.0, 1000.0), (0.5, 900.0)]
)
def test_gamma_cumulative_hazard_in_the_tail(a, x):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Hf = surv.Gamma.Hf(np.array([x]), a, 1.0)[0]
        hf = surv.Gamma.hf(np.array([x]), a, 1.0)[0]
    assert Hf == pytest.approx(-_gamma_log_q(a, x), rel=1e-12)
    # the hazard tends to the rate: 1 - (a - 1)/x to first order
    assert hf == pytest.approx(1 - (a - 1) / x, rel=1e-4)


@pytest.mark.parametrize(
    "a, beta, x, hf, d_x, d_a",
    [
        # (mpmath, 60 digits) f / S lost y eps to rounding: 4e-9 at
        # y = beta x = 1e8, 12% at 1e15 (#760)
        (
            2.5,
            0.25,
            4e4,
            0.24996250374981247,
            9.3731251406531092e-10,
            -2.4997499750131224e-5,
        ),
        (
            2.5,
            0.25,
            4e8,
            0.24999999625000004,
            9.3749998125000014e-18,
            -2.4999999749999998e-9,
        ),
        (
            10.0,
            1.0,
            1e15,
            0.999999999999991,
            8.999999999999982e-30,
            -9.99999999999999e-16,
        ),
        (
            0.7,
            1.0,
            1500.0,
            1.0001998668706449,
            -1.3315596320225885e-7,
            -0.00066622299064922648,
        ),
        (
            40.0,
            1.0,
            1200.0,
            0.96752794340260667,
            2.7036022855697686e-5,
            -0.00083259285276073516,
        ),
        (
            1000.0,
            1.0,
            3000.0,
            0.66716616809812125,
            0.00011086173328370967,
            -0.00033308420413947986,
        ),
    ],
)
def test_gamma_hazard_far_in_the_tail(a, beta, x, hf, d_x, d_a):
    # Past y = 1000 (30 since #777; and a + 2 sqrt(a) + 1) the hazard is
    # the continued fraction of the upper incomplete gamma, whose e**-y
    # cancels exactly (#760): its value and its autograd derivatives are
    # mpmath's.
    from autograd import elementwise_grad, grad

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value = surv.Gamma.hf(np.array([x]), a, beta)[0]
        slope_x = elementwise_grad(lambda t: surv.Gamma.hf(t, a, beta))(
            np.array([x])
        )[0]
        slope_a = grad(lambda s: surv.Gamma.hf(np.array([x]), s, beta)[0])(a)
    assert value == pytest.approx(hf, rel=1e-14)
    assert slope_x == pytest.approx(d_x, rel=1e-12)
    assert slope_a == pytest.approx(d_a, rel=1e-12)


@pytest.mark.parametrize(
    "a, beta, x, want",
    [
        # (mpmath, 60 digits) hf, d/dx, d/da, d2/da2, d2/dadx, d2/dx2
        (
            0.7,
            1.0,
            120.0,
            [
                1.0024795549645153,
                -2.0495730073969042e-5,
                -0.0082653477508986985,
                1.094299267067353e-6,
                6.8323141751136921e-5,
                3.3885940113073551e-7,
            ],
        ),
        (
            2.5,
            0.5,
            200.0,
            [
                0.49257461962091225,
                3.6755731387106147e-5,
                -0.004949525731123334,
                9.843966214468137e-7,
                2.4493012058815598e-5,
                -3.6386513014951545e-7,
            ],
        ),
        (
            0.3,
            2.0,
            25.0,
            [
                2.0274680616550884,
                -0.0010784880050780106,
                -0.039249834227961605,
                2.7501533443865596e-5,
                0.0015418156356119566,
                8.4737933004350062e-5,
            ],
        ),
        (
            40.0,
            1.0,
            100.0,
            [
                0.61608035429922089,
                0.0037459868309287988,
                -0.0097536617123594265,
                7.2076371422332832e-6,
                9.2458460601177029e-5,
                -7.210756063586866e-5,
            ],
        ),
    ],
)
def test_777_gamma_hazard_derivatives_near_y_100(a, beta, x, want):
    # Below y = 1000 the hazard was f / S, whose derivatives in x and a
    # cancel to a size 1 / y: a's was 2e-7 off at y = 100, and its second
    # derivative in a off by more than itself. The continued fraction
    # now takes over from y = 30, differentiated with it (#777).
    import autograd.numpy as anp
    from autograd import elementwise_grad, grad, hessian

    def hf(p):
        return surv.Gamma.hf(anp.ones(1) * p[1], p[0], beta)[0]

    p = np.array([a, x])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = [hf(p), *grad(hf)(p)[::-1]]
        H = hessian(hf)(p)
        slope_x = elementwise_grad(lambda t: surv.Gamma.hf(t, a, beta))(
            np.array([x])
        )[0]
    got = [got[0], got[1], got[2], H[0, 0], H[0, 1], H[1, 1]]
    np.testing.assert_allclose(got, want, rtol=1e-12)
    assert H[0, 1] == H[1, 0]
    assert slope_x == pytest.approx(want[1], rel=1e-12)


# -- every distribution, far out and at extreme parameters (#561) ---------

FUNCTIONS = ("sf", "ff", "df", "hf", "Hf", "log_sf", "log_ff", "log_df")
# x = 0, tiny, moderate, huge and infinite (and the negative ones for the
# families on the whole line; below a support they take its edge values)
X_CONTINUOUS = np.array(
    [-np.inf, -1e300, 0.0, 5e-324, 1e-300, 0.5, 1e10, 1e300, 1.7e308, np.inf]
)
X_DISCRETE = np.array([0.0, 1.0, 2.0, 1e6, 1e15, 1e300, np.inf])
# A proper distribution's functions at x = inf (the hazard's limit is the
# family's own)
AT_INFINITY = {
    "sf": 0.0,
    "ff": 1.0,
    "df": 0.0,
    "Hf": np.inf,
    "log_sf": -np.inf,
    "log_ff": 0.0,
    "log_df": -np.inf,
}
# Functions a distribution refuses on purpose, or does not have
REFUSED = {
    "Bernoulli": FUNCTIONS,  # defined at x = 0 and 1 only
    "ExactEventTime": ("df", "hf", "log_df"),  # a point mass
    "FixedEventProbability": FUNCTIONS,  # no time axis
}


def _parametric_cases():
    from surpyval.tests.conformance.registry import CASES

    return [c.name for c in CASES if c.model_class == "surpyval.Parametric"]


def _quiet_values(fn, x, *params):
    """``fn(x, *params)``, failing on any RuntimeWarning (numpy's
    overflow, division and invalid-value warnings)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return np.asarray(fn(x, *params), dtype=float)


def _variants(dist, params):
    """The fitted parameters, and each one 1e8 times larger and smaller
    (where its bounds and the family's own check allow it)."""
    params = np.asarray(params, dtype=float)
    out = [params]
    bounds = getattr(dist, "bounds", None)
    for i in range(params.size):
        for factor in (1e-8, 1e8):
            p = params.copy()
            p[i] = p[i] * factor if p[i] != 0 else factor
            lo, hi = bounds[i] if bounds is not None else (None, None)
            if (lo is not None and p[i] <= lo) or (
                hi is not None and p[i] >= hi
            ):
                continue
            try:
                getattr(dist, "_check_params", lambda p: None)(p)
                # (a Hypoexponential refuses rates too close to tell
                # apart, as its functions check)
                dist.sf(np.array([1.0]), *p)
            except ValueError:
                continue
            out.append(p)
    return out


@pytest.mark.parametrize("name", _parametric_cases())
def test_561_functions_far_out_are_quiet_and_right(name):
    # Weibull.sf far in the tail warned "overflow encountered in power"
    # although its 0 was right, and a dozen families gave NaN at x = inf
    # (a Gamma's sf, a Poisson's sf, a Weibull's df at 1e300: inf * 0).
    from surpyval.tests._helpers import fresh_conformance_fit

    model = fresh_conformance_fit(name)
    dist = model.dist
    x = X_DISCRETE if dist.discrete else X_CONTINUOUS
    refused = REFUSED.get(dist.name, ())
    for f in FUNCTIONS:
        if f in refused or not callable(getattr(dist, f, None)):
            continue
        # the model's own, which carry an offset, lfp or zero inflation
        if callable(getattr(model, f, None)):
            values = _quiet_values(getattr(model, f), x)
            assert not np.isnan(values).any(), (f, values)
        for params in _variants(dist, model.params):
            values = _quiet_values(getattr(dist, f), x, *params)
            assert not np.isnan(values).any(), (f, params, values)
            if f in AT_INFINITY:
                assert values[-1] == AT_INFINITY[f], (f, params, values)


@pytest.mark.parametrize(
    "dist, params, limit",
    [
        (surv.Exponential, (0.5,), 0.5),
        (surv.Weibull, (3.0, 0.5), 0.0),
        (surv.Weibull, (3.0, 1.0), 1 / 3),
        (surv.Weibull, (3.0, 2.0), np.inf),
        (surv.ExpoWeibull, (3.0, 1.0, 2.0), 1 / 3),
        (surv.ExpoWeibull, (3.0, 2.0, 0.5), np.inf),
        (surv.Gamma, (2.0, 1.5), 1.5),
        (surv.Poisson, (3.0,), 1.0),
        (surv.Geometric, (0.2,), 0.2),
        (surv.NegativeBinomial, (3.0, 0.3), 0.3),
        (surv.DiscreteWeibull, (0.8, 1.0), 0.2),
        (surv.DiscreteWeibull, (0.8, 2.0), 1.0),
        (surv.DiscreteWeibull, (0.8, 0.5), 0.0),
    ],
)
def test_561_hazard_takes_its_limit_at_infinity(dist, params, limit):
    hf = _quiet_values(dist.hf, np.array([1e300, np.inf]), *params)
    assert hf[-1] == pytest.approx(limit, rel=1e-12)


def test_561_a_discretized_hazard_past_the_survival_underflow():
    # R(k - 1) underflows at k = 1e6 for a Weibull(4.4, 1.6): df / R(k - 1)
    # was 0 / 0. The hazard there is 1 to double precision, and with
    # shape 0.5 it is 1 - R(k) / R(k - 1) = -expm1(H(k - 1) - H(k)).
    disc = surv.Discretize(surv.Weibull)
    k = np.array([1e6, 1e15, np.inf])
    assert (_quiet_values(disc.hf, k, 4.4, 1.6) == 1.0).all()
    hf = _quiet_values(disc.hf, np.array([1e6]), 4.4, 0.5)[0]
    H = surv.Weibull.Hf(np.array([1e6 - 1.0, 1e6]), 4.4, 0.5)
    assert hf == pytest.approx(-np.expm1(H[0] - H[1]), rel=1e-9)


def test_561_weibull_far_tail_is_zero_and_quiet():
    # The issue's case: a Weibull-kind forest's sf at a far-tail time
    sf = _quiet_values(surv.Weibull.sf, np.array([1e6, 1e300]), 10.0, 3.0)
    assert (sf == 0.0).all()
    model = surv.Weibull.from_params([10.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert model.sf(1e300) == 0.0
        assert model.df(1e300) == 0.0
