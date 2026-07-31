"""
Regression tests for #268: the MPS estimator was wrong under truncation,
censoring, ties, and offset — the censored/ties block was scaled by a
different count than the spacings (inconsistent even without truncation),
censored terms were not conditioned on the truncation window, and offset
fits passed unshifted truncation bounds to the shifted distribution.
"""

import numpy as np
import pytest

from surpyval import Weibull


class TestMPSTies:
    def test_tied_data_tracks_mle(self):
        # Was (13.7, 1.52) vs MLE (10.07, 2.01).
        np.random.seed(13)
        x = np.round(10 * np.random.weibull(2, 3000))
        x = x[x > 0]
        mps = Weibull.fit(x=x, how="MPS")
        mle = Weibull.fit(x=x, how="MLE")
        np.testing.assert_allclose(mps.params, mle.params, rtol=0.02)


class TestMPSCensoring:
    def test_censored_data_recovers_parameters(self):
        # The old block weighting converged to (9.46, 2.07) as n grew.
        np.random.seed(14)
        t = 10 * np.random.weibull(2, 20000)
        c = (t > 13).astype(int)
        x = np.minimum(t, 13.0)
        mps = Weibull.fit(x=x, c=c, how="MPS")
        assert mps.params[0] == pytest.approx(10.0, rel=0.02)
        assert mps.params[1] == pytest.approx(2.0, rel=0.02)


class TestMPSTruncation:
    def test_truncated_censored_data_recovers_parameters(self):
        # Was (11.7, 4.36) with optimiser success.
        np.random.seed(14)
        t = 10 * np.random.weibull(2, 90000)
        t = t[t > 8][:15000]
        c = (t > 13).astype(int)
        x = np.minimum(t, 13.0)
        mps = Weibull.fit(x=x, c=c, tl=8.0, how="MPS")
        mle = Weibull.fit(x=x, c=c, tl=8.0, how="MLE")
        np.testing.assert_allclose(mps.params, mle.params, rtol=0.02)

    def test_offset_truncated_recovers_parameters(self):
        # The unshifted bound made the objective infinite at the truth;
        # the fit returned gamma=104, alpha=6.1, beta=1.3.
        np.random.seed(31)
        T = 100 + 10 * np.random.weibull(2, 60000)
        x = T[T > 103][:5000]
        m = Weibull.fit(x=x, tl=103.0, how="MPS", offset=True)
        assert m.gamma == pytest.approx(100.0, abs=1.0)
        assert m.params[0] == pytest.approx(10.0, rel=0.1)
        assert m.params[1] == pytest.approx(2.0, rel=0.1)


class TestMPSValidation:
    def test_interval_censored_raises_clearly(self):
        with pytest.raises(ValueError, match="interval-censored"):
            Weibull.fit(x=[[1, 2], [2, 4], [3, 5]], how="MPS")

    def test_plain_fit_unchanged(self):
        np.random.seed(15)
        x = 10 * np.random.weibull(2, 2000)
        m = Weibull.fit(x=x, how="MPS")
        assert m.params[0] == pytest.approx(10.0, rel=0.05)
        assert m.params[1] == pytest.approx(2.0, rel=0.05)
