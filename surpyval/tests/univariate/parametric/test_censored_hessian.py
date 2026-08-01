"""
Regression tests for #270: censored/truncated Gamma and Beta fits used to
get a silently corrupted (even asymmetric) Wald covariance because the
autograd shims for the incomplete gamma/beta functions stripped the trace
in their shape-parameter VJPs, zeroing every second-derivative
contribution through a shape parameter.
"""

import numpy as np
import pytest
from scipy.stats import beta as sbeta
from scipy.stats import gamma as sgamma

from surpyval import Beta, Gamma, Weibull


def _numerical_inv_information(nll, params):
    import numdifftools as nd  # type: ignore

    return np.linalg.inv(nd.Hessian(nll)(params))


class TestCensoredGammaCovariance:
    def test_covariance_matches_observed_information(self):
        np.random.seed(7)
        n = 150
        x0 = np.random.gamma(3, 2.0, n)
        cens = 6.0
        c = (x0 > cens).astype(int)
        x = np.minimum(x0, cens)
        m = Gamma.fit(x=x, c=c)
        H = np.array(m.hess_inv)

        def nll(p):
            a, b = p
            return -(
                sgamma.logpdf(x[c == 0], a, scale=1 / b).sum()
                + sgamma.logsf(x[c == 1], a, scale=1 / b).sum()
            )

        ref = _numerical_inv_information(nll, m.params)
        # Was 12.5x off and 29% asymmetric before the fix.
        np.testing.assert_allclose(H, ref, rtol=1e-3)
        assert abs(H[0, 1] - H[1, 0]) <= 1e-4 * abs(H[0, 1])

    def test_offset_censored_gamma_covariance_finite(self):
        # The mixed d2/dadx path flows through the offset parameter.
        np.random.seed(9)
        x0 = 5.0 + np.random.gamma(3, 2.0, 3000)
        c = (x0 > 11.0).astype(int)
        x = np.minimum(x0, 11.0)
        m = Gamma.fit(x=x, c=c, offset=True)
        assert np.all(np.isfinite(m.hess_inv))
        H = np.array(m.hess_inv)
        assert np.allclose(H, H.T, rtol=1e-3, atol=1e-10)


class TestCensoredBetaCovariance:
    def test_covariance_matches_observed_information(self):
        np.random.seed(8)
        x0 = np.random.beta(2.0, 5.0, 200)
        c = (x0 > 0.4).astype(int)
        x = np.minimum(x0, 0.4)
        m = Beta.fit(x=x, c=c)
        H = np.array(m.hess_inv)

        def nll(p):
            a, b = p
            return -(
                sbeta.logpdf(x[c == 0], a, b).sum()
                + sbeta.logsf(x[c == 1], a, b).sum()
            )

        ref = _numerical_inv_information(nll, m.params)
        np.testing.assert_allclose(H, ref, rtol=1e-3)


class TestControls:
    def test_censored_weibull_unaffected(self):
        # Weibull never touches the incomplete-gamma shims; its Hessian
        # must stay exactly symmetric.
        np.random.seed(7)
        x0 = 10 * np.random.weibull(2, 200)
        c = (x0 > 12.0).astype(int)
        x = np.minimum(x0, 12.0)
        m = Weibull.fit(x=x, c=c)
        H = np.array(m.hess_inv)
        assert abs(H[0, 1] - H[1, 0]) < 1e-12

    def test_uncensored_gamma_unchanged(self):
        # The uncensored likelihood is analytic; parameter CIs must be
        # unchanged and tight around the truth for a large sample.
        np.random.seed(10)
        x = np.random.gamma(3, 2.0, 2000)
        m = Gamma.fit(x=x)
        H = np.array(m.hess_inv)
        assert np.allclose(H, H.T, atol=1e-12)
        se_alpha = np.sqrt(H[0, 0])
        assert m.params[0] == pytest.approx(3.0, abs=4 * se_alpha)
