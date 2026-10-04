"""Card: reliability growth of three prototypes in test-fix-test.

Persona: a reliability engineer running a development growth programme
on three inverter prototypes, each tested to 2000 h with fixes made as
failures appear. Questions (MIL-HDBK-189C; Crow, 1974 and 1982):

1. Is the failure rate falling at all (trend tests)?
2. What is the growth rate (the Crow-AMSAA beta), with bounds, and does
   the power law fit (Cramer-von Mises)?
3. What MTBF has been demonstrated at the end of the test -- the
   instantaneous MTBF, with its lower confidence bound against the
   requirement?

Truth: each prototype a power-law NHPP with cumulative intensity
(t / 12)^0.55.
"""

import numpy as np
import pytest
from scipy.stats import norm

from surpyval.recurrent import CrowAMSAA, laplace, mil_hdbk_189c
from surpyval.tests.scenarios._oracles import best_of, contains

ALPHA, BETA, T_END = 12.0, 0.55, 2000.0


def _growth_test():
    rng = np.random.default_rng(8)
    x, i, c = [], [], []
    for unit in range(3):
        g = np.cumsum(rng.exponential(1.0, 400))
        t = ALPHA * g ** (1 / BETA)
        t = t[t < T_END]
        x += list(t) + [T_END]
        i += [unit] * (t.size + 1)
        c += [0] * t.size + [1]
    return np.array(x), np.array(i), np.array(c)


X, I, C = _growth_test()


def _power_law_neg_ll(p):
    """Time-terminated power-law NHPP over the three prototypes."""
    alpha, beta = np.exp(p)
    t = X[C == 0]
    log_iif = np.log(beta / alpha) + (beta - 1) * np.log(t / alpha)
    return -(log_iif.sum() - 3 * (T_END / alpha) ** beta)


def _inst_mtbf(alpha, beta, t):
    return 1.0 / (beta / alpha * (t / alpha) ** (beta - 1))


def test_growth_is_detected():
    assert laplace(X, I, c=C).trend == "decreasing"
    assert mil_hdbk_189c(X, I, c=C).trend == "decreasing"


def test_crow_amsaa_is_the_mle_and_recovers_beta():
    model = CrowAMSAA.fit(x=X, i=I, c=C)
    ref = best_of(_power_law_neg_ll, [np.log(model.params), [2.0, 0.0]])
    assert -model.log_likelihood == pytest.approx(ref.fun, abs=1e-5)
    assert contains(model.param_cb("beta", alpha_ci=0.05), BETA)


def test_demonstrated_mtbf_point_estimate():
    model = CrowAMSAA.fit(x=X, i=I, c=C)
    expected = _inst_mtbf(*model.params, T_END)
    assert 1 / float(np.squeeze(model.iif(T_END))) == pytest.approx(expected)
    assert float(np.squeeze(model.mtbf(T_END))) == pytest.approx(expected)


def test_demonstrated_mtbf_lower_bound():
    # #578: the lower bound on the demonstrated MTBF is mtbf_cb at the end
    # of the test. Its default is the delta method on log MTBF, which the
    # analyst used to write by hand (as here); Crow's (1982) exact bounds
    # apply too, as every prototype is time terminated at T_END.
    model = CrowAMSAA.fit(x=X, i=I, c=C)
    lower = float(
        np.squeeze(model.mtbf_cb(T_END, alpha_ci=0.2, bound="lower"))
    )
    alpha, beta = model.params
    h = 1e-6

    def log_mtbf(a, b):
        return np.log(_inst_mtbf(a, b, T_END))

    grad = np.array(
        [
            (log_mtbf(alpha + h, beta) - log_mtbf(alpha - h, beta)) / (2 * h),
            (log_mtbf(alpha, beta + h) - log_mtbf(alpha, beta - h)) / (2 * h),
        ]
    )
    se = np.sqrt(grad @ model.covariance() @ grad)
    by_hand = np.exp(log_mtbf(alpha, beta) - norm.ppf(0.8) * se)
    assert lower == pytest.approx(by_hand, rel=1e-6)

    exact = model.mtbf_cb(T_END, alpha_ci=0.2, bound="lower", method="crow")
    assert float(np.squeeze(exact)) < float(np.squeeze(model.mtbf(T_END)))
    truth = _inst_mtbf(ALPHA, BETA, T_END)
    assert contains(model.mtbf_cb(T_END, alpha_ci=0.1, method="crow"), truth)
