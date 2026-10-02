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


class TestTruncationAtTheSupportEdge:
    # A left truncation time of 0 (or below the support) truncates
    # nothing, but its CDF was evaluated, and a Weibull's or Gamma's
    # second derivative there is nan: the covariance went to the
    # numerical Hessian (45% of a left-truncated Weibull fit at 1e5
    # rows), and with an offset the
    # gradient was nan, leaving Nelder-Mead to stop short of the maximum.

    @staticmethod
    def _data(n=800, seed=4):
        rng = np.random.default_rng(seed)
        x = 10 * rng.weibull(1.7, n)
        cens = 12 * rng.uniform(0.3, 1.3, n)
        c = (x > cens).astype(int)
        x = np.minimum(x, cens)
        tl = np.where(rng.uniform(size=n) < 0.3, 0.4 * x, 0.0)
        return x, c, tl

    def test_a_zero_truncation_is_no_truncation(self):
        x, c, tl = self._data()
        m0 = Weibull.fit(x=x, c=c, tl=tl)
        m1 = Weibull.fit(x=x, c=c, tl=np.where(tl == 0, -np.inf, tl))
        np.testing.assert_allclose(m0.params, m1.params, rtol=1e-12)
        np.testing.assert_allclose(m0.neg_ll(), m1.neg_ll(), rtol=1e-14)
        np.testing.assert_allclose(m0.cov_matrix, m1.cov_matrix, rtol=1e-12)

    def test_the_hessian_is_analytic(self, monkeypatch):
        import surpyval.univariate.parametric.fitters.mle as mle

        calls = []
        numerical = mle.Hessian

        def counting(f):
            calls.append(f)
            return numerical(f)

        monkeypatch.setattr(mle, "Hessian", counting)
        x, c, tl = self._data()
        Weibull.fit(x=x, c=c, tl=tl)
        Weibull.fit(x=x, c=c, tl=np.zeros_like(x))
        assert calls == []

    def test_offset_below_a_zero_truncation(self):
        # The truncation times of 0 lie below the fitted threshold
        x, c, tl = self._data(n=2000)
        tl = np.where(tl > 0, tl + 5, 0.0)
        m0 = Weibull.fit(x=x + 5, c=c, tl=tl, offset=True)
        m1 = Weibull.fit(
            x=x + 5, c=c, tl=np.where(tl == 0, -np.inf, tl), offset=True
        )
        # Nelder-Mead stopped at a log-likelihood 1.23 lower, with a
        # warning that the maximum was not verified
        assert m0.optimizer != "Nelder-Mead"
        np.testing.assert_allclose(m0.neg_ll(), m1.neg_ll(), rtol=1e-9)
        np.testing.assert_allclose(m0.gamma, m1.gamma, rtol=1e-6)
