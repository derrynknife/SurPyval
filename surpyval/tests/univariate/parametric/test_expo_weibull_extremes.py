"""The ExpoWeibull likelihood at extreme parameters (#472).

Its log-density was taken as
:math:`\\ln(\\beta\\mu) + (\\beta - 1)\\ln x - \\beta\\ln\\alpha + (\\mu -
1)\\ln g - t` with :math:`g = 1 - e^{-t}`, :math:`t = (x/\\alpha)^\\beta`.
Where :math:`x < \\alpha` and :math:`\\beta` is huge the three middle terms
are each of the size of :math:`\\beta \\ln(x/\\alpha)` and cancel: at
:math:`\\beta = 7 \\times 10^{19}` the error was about :math:`10^4` per
point, and on the conformance registry's fit (alpha, beta, mu = 10.27,
2.30, 1.01) the likelihood came out 3.7e6 in deviance *above* its maximum.
The likelihood-ratio searches walk into that corner (the profile of ``mu``
follows a valley to ``mu -> 0`` with ``beta -> inf``), and only a guard in
``_lr_neg_ll`` kept them from taking those values for the best point.
"""

import decimal
import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval import ExpoWeibull
from surpyval.tests._helpers import no_warnings
from surpyval.tests.conformance.registry import CASE_BY_NAME

_CTX = decimal.Context(prec=60, Emin=-999999, Emax=999999)


@pytest.fixture(scope="module")
def ew():
    case = CASE_BY_NAME["ExpoWeibull"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return case.fit(case.data())


def _deviance(model, theta):
    nll = model._lr_raw_neg_ll(np.asarray(theta, dtype=float))
    return 2.0 * (nll - model._lr_raw_neg_ll(np.asarray(model.params)))


@pytest.mark.parametrize(
    "theta",
    [
        # found by a random search over the extreme box; deviances of
        # -3.7e6, -3.1e6 and -1.8e6 before #472
        (2.5466608967340324e10, 7.000802066479639e19, 2.0610496097e-175),
        (8.572633737412338e10, 7.088111298029461e19, 1.340816588868e-299),
        (1.2405462860761684e11, 3.507784377363136e19, 3.467613871357e-17),
        # the issue's corner
        (10.27, 1e14, 1.01),
        (17.0, 1e14, 1e-14),
    ],
)
def test_likelihood_is_not_above_the_maximum_at_these_points(ew, theta):
    assert _deviance(ew, theta) >= 0.0


def test_likelihood_is_never_above_the_maximum(ew):
    rng = np.random.default_rng(472)
    n = 3000
    thetas = 10.0 ** np.column_stack(
        [
            rng.uniform(-12, 12, n),
            rng.uniform(-12, 20, n),
            rng.uniform(-300, 300, n),
        ]
    )
    dev = np.array([_deviance(ew, theta) for theta in thetas])
    assert not np.any(dev < 0), thetas[np.argmin(dev)]


@pytest.mark.parametrize(
    "x, alpha, beta, mu",
    [
        (5.0, 1e10, 7e19, 2e-175),
        (5.0, 1e10, 1e19, 1e-17),
        (2.0, 3.0, 1e16, 1e-12),
        (2.0, 3.0, 2.5, 1.5),
    ],
)
def test_log_df_at_extreme_shapes(x, alpha, beta, mu):
    # With t = (x / alpha)^beta far below 1, ln g = ln t - t / 2 to 60
    # digits here (t is below 1e-30 or the case is ordinary), and
    # ln f = ln(beta mu / x) + mu ln g + (ln t - ln g) - t.
    X, A, B, M = (decimal.Decimal(v) for v in (x, alpha, beta, mu))
    log_t = _CTX.multiply(B, _CTX.ln(_CTX.divide(X, A)))
    t = _CTX.exp(log_t)
    if log_t < -70:
        # ln t - ln g = t / 2, to far below 1e-60 of it
        t_less_g = _CTX.divide(t, 2)
    else:
        t_less_g = _CTX.subtract(log_t, _CTX.ln(1 - _CTX.exp(-t)))
    log_g = _CTX.subtract(log_t, t_less_g)
    true = _CTX.subtract(
        _CTX.add(
            _CTX.ln(_CTX.divide(_CTX.multiply(B, M), X)),
            _CTX.add(_CTX.multiply(M, log_g), t_less_g),
        ),
        t,
    )
    got = ExpoWeibull.log_df(np.array([x]), alpha, beta, mu)[0]
    # The rounding of x and alpha alone moves ln t by beta * 1e-16 * |ln|,
    # which reaches ln f through mu ln g only.
    sens = float(M * B * (abs(_CTX.ln(X)) + abs(_CTX.ln(A)))) + 1.0
    assert abs(got - float(true)) <= 64 * 2.2e-16 * max(sens, abs(float(true)))


def test_hf_at_extreme_shapes_is_f_over_r():
    # beta mu ln(x / alpha) = -0.69; the terms of the old form were each
    # 7e16 here, and cancelled to an error of about 10 in ln h
    x, alpha, beta, mu = 5.0, 10.0, 1e17, 1e-17
    log_h = np.log(ExpoWeibull.hf(np.array([x]), alpha, beta, mu))[0]
    log_f = ExpoWeibull.log_df(np.array([x]), alpha, beta, mu)[0]
    log_r = ExpoWeibull.log_sf(np.array([x]), alpha, beta, mu)[0]
    assert log_h == pytest.approx(log_f - log_r, rel=1e-12, abs=1e-9)


def test_lr_searches_never_see_a_likelihood_above_the_maximum(ew):
    # Record every likelihood the searches evaluate: none may be above the
    # fit's (the guard in _lr_neg_ll is then never needed here).
    seen = []
    raw = type(ew)._lr_raw_neg_ll

    def recording(self, theta):
        value = raw(self, theta)
        seen.append(value)
        return value

    nll_hat = ew._lr_raw_neg_ll(np.asarray(ew.params))
    model = ew
    try:
        type(ew)._lr_raw_neg_ll = recording
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name in ("beta", "mu"):
                model.param_cb(name, alpha_ci=0.2, method="lr")
    finally:
        type(ew)._lr_raw_neg_ll = raw
    seen = np.asarray(seen, dtype=float)
    finite = seen[np.isfinite(seen)]
    assert finite.size > 50
    assert 2.0 * (finite.min() - nll_hat) > -1e-6


# ---------------------------------------------------------------------------
# The tail is stable (#257).
# ---------------------------------------------------------------------------


def test_expo_weibull_tail_is_stable():
    # 1 - (1 - e^-t)^mu underflowed to exactly 0 once e^-t < 1e-16.
    s = float(ExpoWeibull.sf(8, 3, 4, 1.2))
    assert 0.0 < s < 1e-18
    assert np.isfinite(float(ExpoWeibull.Hf(8, 3, 4, 1.2)))
    assert np.isfinite(float(ExpoWeibull.log_sf(8, 3, 4, 1.2)))
    # Moderate-x values agree with the naive form.
    naive = 1 - (1 - np.exp(-((2 / 3) ** 4))) ** 1.2
    assert float(ExpoWeibull.sf(2, 3, 4, 1.2)) == pytest.approx(
        naive, abs=1e-12
    )


# ---------------------------------------------------------------------------
# The moments at any scale.
# ---------------------------------------------------------------------------


W, E, G = surv.Weibull, surv.Exponential, surv.Geometric


@pytest.mark.parametrize("alpha", [1e-4, 1.0, 1e4])
def test_expo_weibull_moments_at_any_scale(alpha):
    EW = surv.ExpoWeibull
    mean = no_warnings(EW.mean, alpha, 2.0, 1.5)
    assert mean == pytest.approx(alpha * 1.0394154617791786, rel=1e-9)
    m2 = no_warnings(EW.moment, 2, alpha, 2.0, 1.5)
    assert m2 == pytest.approx(alpha**2 * EW.moment(2, 1.0, 2.0, 1.5))
    entropy = no_warnings(EW.entropy, alpha, 2.0, 1.5)
    assert entropy == pytest.approx(
        EW.entropy(1.0, 2.0, 1.5) + np.log(alpha), abs=1e-9
    )


def test_expo_weibull_moments_match_the_weibull_at_mu_one():
    assert surv.ExpoWeibull.moment(3, 7.0, 1.3, 1.0) == pytest.approx(
        W.moment(3, 7.0, 1.3), rel=1e-10
    )
