"""
Regression tests for the round-3 review fixes:

- #284: alpha series/parallel composition removed (Repyability covers RBDs)
- #285: mcf_cb selects by query position before masking; [lower, upper]
- #286: CoxLewis log-intercept unbounded below
- #288: NHPP x_prev typo; PI fitters honour init
- #289: MPS tie densities, AH bandwidth on coincident events, Beta4 hf
"""

import numpy as np
import pytest

from surpyval import AdditiveHazards, Beta4, Weibull
from surpyval.recurrent import (
    CoxLewis,
    CrowAMSAA,
    NonParametricCounting,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)


class TestAlphaCompositionRemoved:
    def test_models_no_longer_importable(self):
        with pytest.raises(ImportError):
            from surpyval.alpha import SeriesModel  # noqa: F401
        with pytest.raises(ImportError):
            from surpyval.alpha import ParallelModel  # noqa: F401


class TestMCFConfidenceBounds:
    @staticmethod
    def _model():
        return NonParametricCounting.fit(
            x=[4, 5, 6, 8, 10, 12], i=[1, 2, 1, 2, 1, 2], c=[0] * 6
        )

    def test_two_sided_off_grid_queries(self):
        # 285: masks used to zero the whole upper-bound column and crash
        # when queries outnumbered the two bound rows.
        m = self._model()
        out = np.asarray(m.mcf_cb(np.array([1.0, 5.5, 6.5, 20.0])))
        assert out.shape == (4, 2)
        # below the first time: [0, 0]; above the last: NaN
        assert np.all(out[0] == 0.0)
        assert np.all(np.isnan(out[3]))
        # in-range rows are finite with lower < upper
        for row in out[1:3]:
            assert np.all(np.isfinite(row))
            assert row[0] < row[1]

    def test_more_queries_than_bound_rows_no_crash(self):
        m = self._model()
        out = np.asarray(m.mcf_cb(np.array([1.0, 5.5, 20.0])))
        assert out.shape == (3, 2)

    def test_one_sided_out_of_range(self):
        m = self._model()
        low = np.asarray(
            m.mcf_cb(np.array([1.0, 6.5, 20.0]), bound="lower"), dtype=float
        )
        assert low[0] == 0.0
        assert np.isfinite(low[1])
        assert np.isnan(low[2])

    def test_column_order_lower_upper(self):
        # 285: two-sided order used to be [upper, lower], inconsistent
        # with the parametric cif_cb.
        m = self._model()
        out = np.asarray(m.mcf_cb(np.array([8.5])))
        assert out[0, 0] < out[0, 1]


class TestCoxLewisBounds:
    def test_negative_log_intercept_recovered(self):
        # 286: the (0, None) bound pinned alpha at 0 for baseline rates
        # below one event per time unit.
        np.random.seed(8)
        xs, iis, cs = [], [], []
        for it in range(100):
            t = 0.0
            while True:
                u = np.random.uniform()
                inc = (
                    np.log(1 - 0.05 * np.log(u) / np.exp(-1.0 + 0.05 * t))
                    / 0.05
                )
                t += inc
                if t > 30.0:
                    break
                xs.append(t)
                iis.append(it)
                cs.append(0)
            xs.append(30.0)
            iis.append(it)
            cs.append(1)
        m = CoxLewis.fit(x=xs, i=iis, c=cs)
        assert m.params[0] == pytest.approx(-1.0, abs=0.15)
        assert m.params[1] == pytest.approx(0.05, abs=0.01)


class TestNHPPDegenerateIntervals:
    def test_degenerate_pairs_match_1d(self):
        # 288: x_prev typo cancelled the exposure term for 2-D input
        # without interval rows.
        np.random.seed(5)
        u = np.sort(np.random.uniform(size=40))
        t_ev = 100.0 * u ** (1 / 3)
        m1 = CrowAMSAA.fit(
            x=t_ev, i=np.ones(40, dtype=int), c=np.zeros(40, dtype=int)
        )
        m2 = CrowAMSAA.fit(
            x=np.column_stack([t_ev, t_ev]), i=np.ones(40, dtype=int)
        )
        np.testing.assert_allclose(m1.params, m2.params, rtol=1e-8)


class TestPIFittersHonourInit:
    @staticmethod
    def _data():
        np.random.seed(9)
        xs, iis, cs, Zs = [], [], [], []
        for it in range(30):
            z = np.random.binomial(1, 0.5)
            t = 0.0
            rate = 0.3 * np.exp(0.5 * z)
            while True:
                t += np.random.exponential(1 / rate)
                if t > 20:
                    break
                xs.append(t)
                iis.append(it)
                cs.append(0)
                Zs.append([z])
            xs.append(20.0)
            iis.append(it)
            cs.append(1)
            Zs.append([z])
        return xs, iis, cs, Zs

    def test_hpp_accepts_and_validates_init(self):
        xs, iis, cs, Zs = self._data()
        m = ProportionalIntensityHPP.fit(
            x=xs, Z=Zs, i=iis, c=cs, init=[0.3, 0.5]
        )
        assert np.all(np.isfinite(np.atleast_1d(m.params)))
        with pytest.raises(ValueError, match="init must have"):
            ProportionalIntensityHPP.fit(
                x=xs, Z=Zs, i=iis, c=cs, init=[0.3, 0.5, 0.9]
            )

    def test_nhpp_accepts_and_validates_init(self):
        xs, iis, cs, Zs = self._data()
        m = ProportionalIntensityNHPP.fit(
            x=xs, Z=Zs, i=iis, c=cs, dist=CrowAMSAA, init=[10.0, 1.0, 0.5]
        )
        assert np.all(np.isfinite(np.atleast_1d(m.params)))
        with pytest.raises(ValueError, match="init must have"):
            ProportionalIntensityNHPP.fit(
                x=xs, Z=Zs, i=iis, c=cs, dist=CrowAMSAA, init=[1.0]
            )


class TestRound2FollowUps:
    def test_mps_tie_objective_inf_not_nan(self):
        # 289: untied points no longer contribute 0*log(0) to the tie
        # density block.
        v = Weibull.neg_mean_D(
            np.array([1.0, 2, 3, 300]),
            np.zeros(4),
            np.array([1, 2, 1, 1]),
            -np.inf,
            np.inf,
            10.0,
            2.0,
        )
        assert np.isinf(v)

    def test_ah_bandwidth_coincident_events(self):
        # 289: nearly-tied event times used to collapse the bandwidth to
        # the floor and return Dirac spikes (~1.6e9).
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = AdditiveHazards.fit(x=[5.0, 5.0 + 1e-9], Z=[[0.0], [0.1]])
        hf = float(np.ravel(m.hf([5.0], np.array([0.05])))[0])
        assert np.isfinite(hf)
        assert hf < 100.0

    def test_beta4_hf_defined_outside_support(self):
        out = Beta4.hf(np.array([-1.0, 3.0, 10.0]), 2, 3, 0, 5)
        assert out[0] == 0.0
        assert np.isfinite(out[1]) and out[1] > 0
        assert np.isposinf(out[2])
