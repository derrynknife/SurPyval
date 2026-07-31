"""
Regression tests for the round-2 medium/low batch (#276-#282):

- #276 concordance-index tie handling (see test_score.py for unit cases)
- #277 Lin-Ying hf/df baseline rate; phi() on additive models
- #278 Aalen-Johansen CIF uses product-limit weights (CIF <= 1)
- #279 Efron tie corrections in Cox residuals / check_ph / robust SEs
- #280 distribution edge cases (see also test_distribution_fixes.py)
- #281 xrd_to_xcnt late-entry guard
- #282 container/robustness batch
"""

import numpy as np
import pytest

import surpyval
from surpyval import (
    AdditiveHazards,
    Beta4,
    CoxPH,
    ExponentialAH,
    KaplanMeier,
    LogLogistic,
    NelsonAalen,
    Rayleigh,
    Uniform,
)
from surpyval.univariate.competing_risks.nonparametric.competing_risks import (
    CompetingRisks,
)
from surpyval.univariate.regression.proportional_hazards.diagnostics import (
    check_ph,
    compute_residuals,
)
from surpyval.utils import xrd_to_xcnt
from surpyval.utils.surpyval_data import SurpyvalData


class TestLinYingHazardRate:
    def test_hf_df_on_hazard_scale(self):
        # 277: hf used to return ~beta'Z (the baseline jump vanishes as
        # n grows); it must estimate h0(t) + beta'Z.
        np.random.seed(2)
        n = 20000
        Z = np.random.uniform(size=(n, 1))
        t = np.random.exponential(1 / (0.5 + 0.7 * Z[:, 0]))
        m = AdditiveHazards.fit(x=t, Z=Z)
        z = np.array([0.5])
        hf = np.ravel(m.hf([0.5, 1.0, 1.5], z))
        assert np.all(np.abs(hf - 0.85) < 0.12)
        df = float(np.ravel(m.df([1.0], z))[0])
        assert df == pytest.approx(0.85 * np.exp(-0.85), rel=0.15)

    def test_parametric_ah_phi_raises_not_implemented(self):
        np.random.seed(4)
        Z = np.random.uniform(size=(300, 1))
        t = np.random.exponential(1 / (0.5 + 0.7 * Z[:, 0]))
        m = ExponentialAH.fit(x=t, Z=Z)
        with pytest.raises(NotImplementedError, match="additive"):
            m.phi(np.array([0.5]))


class TestCIFProductLimit:
    def test_na_method_cif_sums_to_one(self):
        # 278: default Nelson-Aalen method paired d/r with exp(-H) and
        # the total incidence exceeded 1 (1.216 here).
        cr = CompetingRisks.fit(
            x=[1, 1, 1, 2, 2, 3], e=["a", "a", "b", "a", "b", "a"]
        )
        total = float(np.ravel(cr.cif(3, "a"))[0]) + float(
            np.ravel(cr.cif(3, "b"))[0]
        )
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_extreme_case_capped_at_one(self):
        cr = CompetingRisks.fit(x=[1] * 9 + [2], e=["a"] * 10)
        assert float(np.ravel(cr.cif(2, "a"))[0]) == pytest.approx(1.0)


class TestEfronDiagnostics:
    @staticmethod
    def _tied_fit():
        np.random.seed(11)
        n = 120
        Z = np.column_stack(
            [np.random.binomial(1, 0.5, n), np.random.normal(size=n)]
        )
        u = np.random.uniform(size=n)
        t = -np.log(u) / (0.3 * np.exp(0.7 * Z[:, 0] - 0.4 * Z[:, 1]))
        x = np.ceil(np.clip(t, 0.5, 6)).astype(float)
        c = (np.random.uniform(size=n) < 0.2).astype(int)
        return x, c, Z

    def test_residual_sums_vanish_at_mle(self):
        # 279: these identities only hold when the residuals use the
        # same tie handling as the fitted likelihood.
        x, c, Z = self._tied_fit()
        for method in ("efron", "breslow"):
            m = CoxPH.fit(x=x, Z=Z, c=c, method=method)
            assert compute_residuals(m, "martingale").sum() == pytest.approx(
                0.0, abs=1e-8
            )
            assert np.abs(
                compute_residuals(m, "score").sum(axis=0)
            ).max() == pytest.approx(0.0, abs=1e-8)

    def test_check_ph_matches_lifelines_under_ties(self):
        # Reference values from lifelines 0.30.3 on this exact dataset
        # (km transform; identity and log also agree — lifelines is not
        # a CI dependency, so the values are pinned).
        x, c, Z = self._tied_fit()
        m = CoxPH.fit(x=x, Z=Z, c=c, method="efron")
        res = check_ph(m, transform="km")
        stats = [e["statistic"] for e in res["per_covariate"]]
        assert stats[0] == pytest.approx(1.3936, abs=2e-3)
        assert stats[1] == pytest.approx(0.0013, abs=2e-3)

    def test_dfbeta_tracks_exact_leave_one_out(self):
        x, c, Z = self._tied_fit()
        m = CoxPH.fit(x=x, Z=Z, c=c, method="efron")
        dfb = compute_residuals(m, "dfbeta")
        # Spot-check 15 rows of exact leave-one-out influence.
        rows = np.arange(0, 120, 8)
        loo = np.zeros((rows.size, 2))
        for r, i in enumerate(rows):
            keep = np.ones(120, dtype=bool)
            keep[i] = False
            mi = CoxPH.fit(x=x[keep], Z=Z[keep], c=c[keep], method="efron")
            loo[r] = m.beta - mi.beta
        corr = np.corrcoef(dfb[rows, 0], loo[:, 0])[0, 1]
        assert corr > 0.99


class TestDistributionEdges:
    def test_loglogistic_at_zero(self):
        assert LogLogistic.sf(0.0, 3, 4) == pytest.approx(1.0)
        assert LogLogistic.ff(0.0, 3, 4) == pytest.approx(0.0)
        vals = LogLogistic.sf(np.array([0.0, 1.0]), 3, 4)
        assert np.all(np.isfinite(vals))

    def test_loglogistic_log_sf_no_overflow(self):
        from scipy.stats import fisk

        got = float(
            np.ravel(LogLogistic.log_sf(np.array([1e41]), 1e40, 10))[0]
        )
        want = float(fisk.logsf(1e41, 10, scale=1e40))
        assert got == pytest.approx(want, rel=1e-6)

    def test_beta4_density_zero_outside_support(self):
        vals = Beta4.df(np.array([1.0, 2.5, 6.0]), 3, 4, 2, 5)
        assert vals[0] == 0.0
        assert vals[2] == 0.0
        assert vals[1] > 0
        assert Beta4.df(1.0, 2.5, 4, 2, 5) == 0.0  # fractional alpha

    def test_rayleigh_mpp_uses_truncation(self):
        np.random.seed(3)
        x = 3 * np.sqrt(-2 * np.log(np.random.uniform(size=6000)))
        tl = np.random.uniform(0, 2, 6000)
        keep = x > tl
        x_obs, tl_obs = x[keep][:3000], tl[keep][:3000]
        with_tl = Rayleigh.fit(x_obs, tl=tl_obs, how="MPP").params
        without = Rayleigh.fit(x_obs, how="MPP").params
        assert not np.allclose(with_tl, without)
        assert with_tl[0] == pytest.approx(3.0, rel=0.05)

    def test_rayleigh_mpp_ecdf_finite(self):
        np.random.seed(3)
        x = 3 * np.sqrt(-2 * np.log(np.random.uniform(size=1000)))
        params = Rayleigh.fit(x, how="MPP", heuristic="ECDF").params
        assert np.all(np.isfinite(params))

    def test_uniform_interval_clear_error(self):
        with pytest.raises(ValueError, match="interval-censored"):
            Uniform.fit(x=[1.0, 2.0, [2, 4], 3.5, [3, 5]], c=[0, 0, 2, 0, 2])


class TestDataHandling:
    def test_xrd_to_xcnt_rejects_late_entry(self):
        # 281: np.abs silently converted late-entry data into a
        # different study.
        with pytest.raises(ValueError, match="risk set increases"):
            xrd_to_xcnt(
                np.array([1.0, 2, 3, 4]),
                np.array([2, 3, 2, 1]),
                np.array([1, 1, 1, 1]),
            )

    def test_xrd_to_xcnt_monotone_unchanged(self):
        x, c, n, t = xrd_to_xcnt(
            np.array([1.0, 2, 3, 4]),
            np.array([4, 3, 2, 1]),
            np.array([1, 1, 1, 1]),
        )
        assert np.asarray(x).tolist() == [1.0, 2.0, 3.0, 4.0]
        assert np.asarray(c).tolist() == [0, 0, 0, 0]

    def test_surpyval_data_scalar_interval_index(self):
        d = SurpyvalData([1, 2, [2, 5], 3, 6], c=[0, 1, 2, 0, 0])
        row = d[2]
        assert row.x.ndim == 2  # keeps the interval row structure

    def test_surpyval_data_slice_keeps_covariates(self):
        d = SurpyvalData([3, 1, 2], Z=[[1], [2], [3]])
        assert d[0:2].Z is not None
        assert d[0:2].Z.shape == (2, 1)

    def test_to_xrd_cache_respects_estimator(self):
        d = SurpyvalData([1, 2, 3, 4], c=[1, 0, 0, 0])
        a = d.to_xrd(estimator="Nelson-Aalen")
        b = d.to_xrd(estimator="Kaplan-Meier")
        assert a is not b


class TestNonParametricScalars:
    def test_scalar_hf_df_finite(self):
        na = NelsonAalen.fit([1.0, 2, 3, 4, 5])
        assert float(np.ravel(na.hf(2.5))[0]) == pytest.approx(0.25)
        assert np.isfinite(float(np.ravel(na.df(2.5))[0]))
        # Matches what the array path returns for the same point.
        grid = np.ravel(na.hf([1.5, 2.5]))
        assert float(np.ravel(na.hf(2.5))[0]) == pytest.approx(grid[1])

    def test_single_observation_cb_drawable(self):
        km = KaplanMeier.fit([5.0])
        cb = np.asarray(km.cb([5.0], on="sf"), dtype=float)
        assert np.all(np.isfinite(cb))

    def test_check_ph_no_spurious_truncation_warning(self):
        import warnings as _w

        np.random.seed(7)
        x = np.random.exponential(2.0, 60)
        Z = np.random.normal(size=(60, 1))
        m = CoxPH.fit(x=x, Z=Z, tl=np.zeros(60))
        with _w.catch_warnings():
            _w.simplefilter("error")
            check_ph(m)


def test_concordance_discordant_tied_pair_scores_zero():
    # 276 (unit cases in tests/utils/test_score.py; pinned here too).
    from surpyval.utils.score import score

    assert score([5.0, 5.0], [0, 1], [1.0, 2.0]) == 0.0
    assert score([5.0, 5.0], [0, 1], [2.0, 1.0]) == 1.0


def test_surpyval_namespace_unchanged():
    # Guard: the fixes must not have removed public names.
    for name in ("AdditiveHazards", "CoxPH", "Rayleigh", "Uniform"):
        assert hasattr(surpyval, name)
