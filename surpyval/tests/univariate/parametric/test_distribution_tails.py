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
