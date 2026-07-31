"""
Regression tests for the round-2 review fix batch A:

- #271: parametric PH ``random()`` samples the model's own distribution and
  supports multi-covariate models.
- #269: LFP + left truncation uses the mixture survival in the truncation
  normaliser (the old form was unbounded above).
- #274: Royston-Parmar keeps the best finite optimiser result and validates
  knot placement.
- #275: the numeric MOM path optimises to convergence.
"""

import numpy as np
import pytest

from surpyval import RoystonParmar, Weibull, WeibullPH


class TestPHRandom:
    def test_random_matches_model_sf(self):
        # 271: draws must come from S(x|Z) = S0(x)^phi.
        np.random.seed(1)
        u = np.random.uniform(size=3000)
        Z = np.random.binomial(1, 0.5, 3000).reshape(-1, 1)
        phi = np.exp(1.0 * Z[:, 0])
        t = 10 * (-np.log(u) / phi) ** 0.5
        m = WeibullPH.fit(x=t, Z=Z)

        np.random.seed(2)
        xs, zs = m.random(100_000, np.array([[1.0]]))
        for tt in (2.0, 4.0, 6.0):
            emp = (np.asarray(xs) > tt).mean()
            mod = float(np.ravel(m.sf(tt, np.array([[1.0]])))[0])
            assert emp == pytest.approx(mod, abs=0.01)

    def test_random_two_covariates(self):
        # 271: used to raise a broadcast ValueError for >= 2 covariates.
        np.random.seed(3)
        t = 10 * np.random.weibull(2, 500)
        Z = np.hstack(
            [
                np.random.binomial(1, 0.5, 500).reshape(-1, 1),
                np.random.normal(size=(500, 1)),
            ]
        )
        m = WeibullPH.fit(x=t, Z=Z)
        x, z_out = m.random(7, np.array([[1.0, 0.5]]))
        assert np.shape(x) == (7,)
        assert np.shape(z_out) == (7, 2)
        assert np.all(np.isfinite(x))


class TestLFPTruncation:
    def test_lfp_left_truncated_recovers_parameters(self):
        # 269: the old normaliser (p - f0) * (1 - F0(tl)) made the
        # likelihood unbounded; the fit returned alpha ~ 1e-42.
        np.random.seed(11)
        N = 30000
        is_mortal = np.random.uniform(size=N) < 0.6
        t = np.where(is_mortal, 10 * np.random.weibull(2, N), np.inf)
        entry = 3.0
        t_seen = t[t > entry]
        observed = t_seen < 20
        x = np.where(observed, t_seen, 20.0)
        c = (~observed).astype(int)

        model = Weibull.fit(x=x, c=c, tl=np.full(len(x), entry), lfp=True)
        alpha, beta = model.params
        assert alpha == pytest.approx(10.0, rel=0.05)
        assert beta == pytest.approx(2.0, rel=0.05)
        assert model.p == pytest.approx(0.6, abs=0.03)

    def test_plain_interval_likelihood_unchanged(self):
        # The f0 terms cancel for finite bounds: a plain interval-censored
        # fit must be unaffected by the 269 change.
        np.random.seed(12)
        t = 10 * np.random.weibull(2, 2000)
        xl = np.floor(t)
        xr = xl + 1.0
        model = Weibull.fit(x=np.column_stack([xl, xr]), c=np.full(2000, 2))
        assert model.params[0] == pytest.approx(10.0, rel=0.05)
        assert model.params[1] == pytest.approx(2.0, rel=0.1)


class TestRoystonParmarGuards:
    def test_double_truncation_returns_finite_model(self):
        # 274: the unguarded BFGS polish used to replace the finite
        # Nelder-Mead result with NaN on doubly-truncated data.
        np.random.seed(21)
        t = 10 * np.random.weibull(2, 20000)
        x = t[(t > 3) & (t < 15)][:2000]
        m = RoystonParmar.fit(x=x, tl=3, tr=15, df=1)
        assert np.isfinite(m.neg_ll())
        sf10 = float(np.ravel(m.sf(10))[0])
        # True Weibull(10, 2) survival at 10 is exp(-1).
        assert sf10 == pytest.approx(np.exp(-1), abs=0.05)

    def test_too_few_distinct_events_raises(self):
        # 274: one event + censored rows at the default df=3 used to
        # return a silent NaN model.
        with pytest.raises(ValueError, match="distinct event times"):
            RoystonParmar.fit(x=[5.0, 6.0, 7.0, 8.0], c=[0, 1, 1, 1])

    def test_tied_events_raise_instead_of_nan(self):
        with pytest.raises(ValueError, match="distinct event times|distinct"):
            RoystonParmar.fit(x=[1.0] * 10 + [2.0] * 10, df=3)


class TestMOMNumericPath:
    def test_mom_offset_recovers_parameters(self):
        # 275: tol=1e-1 used to stop the optimiser at beta ~ 3-4.
        np.random.seed(12)
        T = 50 + 10 * np.random.weibull(2, 5000)
        m = Weibull.fit(x=T, how="MOM", offset=True)
        assert m.gamma == pytest.approx(50.0, abs=1.0)
        assert m.params[0] == pytest.approx(10.0, rel=0.1)
        assert m.params[1] == pytest.approx(2.0, rel=0.1)
