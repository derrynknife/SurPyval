"""The max-combo, trend, permutation and maximally selected log-rank
tests against stored R results.

The values are in ``r_logrank_extensions.json``, written by
``scripts/reference/reference_r_logrank_extensions.R`` (R 4.6.2,
survival 3.8.12, nph 2.1, coin 1.4.6, maxstat 0.7.26, mvtnorm 1.4.2), on
survival's lung, veteran and aml data.

Tolerances:

- Closed forms (the components' ``z`` and correlation of the max-combo
  test, the trend statistic, the permutation statistic, the maximally
  selected statistic and its cut-point, the Lausen bounds) agree to
  rounding, ``rtol=1e-9``.
- Exact enumeration (the AML permutation p-values) agrees to rounding;
  coin computes the same distribution by its shift algorithm.
- The max-combo p-value is a multivariate normal probability that both
  programs estimate numerically: SurPyval's has a relative error of about
  0.1%, mvtnorm's an absolute error of about 1e-5, so ``rtol=5e-3``.
- The "exact" maximally selected p-value is a deterministic quadrature in
  SurPyval and mvtnorm's integration in R: ``rtol=1e-3`` at p = 0.29.
  At p = 2e-5 R warned that it did not reach ``abseps`` and is 5.5%
  lower; there SurPyval is checked against an independent estimate (see
  the test).
- coin's stratified ``logrank_test(y ~ x | block)`` forms the log-rank
  scores from the pooled data, so the stratified reference passes scores
  formed within each stratum to ``independence_test``.
- Monte Carlo p-values (coin's 1e5 resamples, maxstat's ``condMC``) are
  compared with SurPyval's exact or deterministic value, or its own Monte
  Carlo value, within four Monte Carlo standard errors of the two.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, values

EXACT = dict(rtol=1e-9, atol=1e-12)
SRC = "r_logrank_extensions"


def _lung():
    d = fixture("lung")
    return d["time"], d["c"], d


def _lung_ecog():
    t, c, d = _lung()
    keep = ~np.isnan(d["ph_ecog"]) & (d["ph_ecog"] < 3)
    return t[keep], c[keep], d["ph_ecog"][keep], d["sex"][keep]


# --- max-combo ------------------------------------------------------------


@pytest.mark.parametrize(
    "alt, ref_alt",
    [("two-sided", "two_sided"), ("greater", "greater"), ("less", "less")],
)
def test_maxcombo_matches_nph(alt, ref_alt):
    t, c, d = _lung()
    ref = values(SRC, f"maxcombo_lung_sex_{ref_alt}")
    res = sp.maxcombo_logrank(t, d["sex"], c=c, alternative=alt)
    assert_allclose(res.z, ref["z"], **EXACT)
    assert_allclose(res.correlation, ref["correlation"], **EXACT)
    assert_allclose(res.p_values, ref["p_each"], rtol=1e-9)
    assert_allclose(res.p_bonferroni, ref["p_bonferroni"], rtol=1e-9)
    assert_allclose(res.p_value, ref["p_value"], rtol=5e-3)


def test_maxcombo_three_components_matches_nph():
    d = fixture("veteran")
    ref = values(SRC, "maxcombo_veteran_trt_nph_default")
    res = sp.maxcombo_logrank(
        d["time"], d["trt"], c=1 - d["status"], rho=(0, 0, 1), gamma=(0, 1, 0)
    )
    assert_allclose(res.z, ref["z"], **EXACT)
    assert_allclose(res.p_value, ref["p_value"], rtol=5e-3)


# --- trend ----------------------------------------------------------------


@pytest.mark.parametrize(
    "ref_id, kwargs",
    [
        ("trend_lung_ecog", {}),
        (
            "trend_lung_ecog_rho1",
            {"weighting": "fleming-harrington", "rho": 1},
        ),
        ("trend_lung_ecog_strata_sex", {"strata": "sex"}),
    ],
)
def test_trend_matches_survdiff_by_hand(ref_id, kwargs):
    t, c, ecog, sex = _lung_ecog()
    if kwargs.get("strata") == "sex":
        kwargs = {"strata": sex}
    ref = values(SRC, ref_id)
    res = sp.trend_logrank(t, ecog, c=c, scores=[0, 1, 2], **kwargs)
    assert_allclose(res.z, ref["z"], **EXACT)
    assert_allclose(res.statistic, ref["chisq"], **EXACT)
    assert_allclose(res.p_value, ref["p_value"], rtol=1e-9)


def test_trend_unequal_scores_matches_survdiff():
    d = fixture("veteran")
    ref = values(SRC, "trend_veteran_celltype_unequal")
    res = sp.trend_logrank(
        d["time"], d["celltype"], c=1 - d["status"], scores=[1, 2, 4, 8]
    )
    assert_allclose(res.z, ref["z"], **EXACT)
    # The same scores as a dict.
    res_dict = sp.trend_logrank(
        d["time"],
        d["celltype"],
        c=1 - d["status"],
        scores={1.0: 1, 2.0: 2, 3.0: 4, 4.0: 8},
    )
    assert_allclose(res_dict.z, ref["z"], **EXACT)


def test_trend_is_the_cox_score_test_with_breslow_ties():
    # With tied times the hypergeometric variance of survdiff and the
    # Breslow information of coxph differ a little; without ties they
    # are equal. lung has ties, so the two agree only to 0.3%.
    t, c, ecog, _ = _lung_ecog()
    ref = values(SRC, "trend_lung_ecog_cox_score")
    res = sp.trend_logrank(t, ecog, c=c, scores=[0, 1, 2])
    assert_allclose(res.statistic, ref["score_test"], rtol=5e-3)


# --- permutation ----------------------------------------------------------


@pytest.mark.parametrize(
    "ref_id, kwargs",
    [
        ("permutation_aml_exact", {}),
        (
            "permutation_aml_exact_fh10",
            {"weighting": "fleming-harrington", "rho": 1},
        ),
    ],
)
def test_permutation_exact_matches_coin(ref_id, kwargs):
    d = fixture("aml")
    ref = values(SRC, ref_id)
    # coin's first group is maintained = 0, as SurPyval's sorted labels.
    res = sp.permutation_logrank(
        d["time"],
        d["maintained"],
        c=1 - d["status"],
        method="exact",
        max_exact=2_000_000,
        **kwargs,
    )
    assert res.n_permutations == 1352078
    # coin's scores are minus SurPyval's.
    assert_allclose(res.z, -ref["z"], **EXACT)
    assert_allclose(res.p_value_asymptotic, ref["p_value_asymptotic"], **EXACT)
    assert_allclose(res.p_value, ref["p_value"], rtol=1e-9)


def test_permutation_k_sample_matches_coin_approximate():
    t, c, ecog, _ = _lung_ecog()
    ref = values(SRC, "permutation_lung_ecog_mc")
    res = sp.permutation_logrank(
        t, ecog, c=c, method="monte-carlo", n_permutations=99999,
        random_state=1,
    )  # fmt: skip
    assert_allclose(res.statistic, ref["statistic"], **EXACT)
    assert_allclose(res.p_value_asymptotic, ref["p_value_asymptotic"], **EXACT)
    p = ref["p_value"]
    se = np.hypot(res.p_value_se, np.sqrt(p * (1 - p) / 1e5))
    assert abs(res.p_value - p) <= 4 * se, (res.p_value, p, se)


def test_permutation_stratified_matches_coin_approximate():
    d = fixture("veteran")
    ref = values(SRC, "permutation_veteran_trt_strata_mc")
    res = sp.permutation_logrank(
        d["time"],
        d["trt"],
        c=1 - d["status"],
        strata=d["celltype"],
        n_permutations=99999,
        random_state=2,
    )
    assert res.method == "monte-carlo"
    assert_allclose(res.z, -ref["z"], **EXACT)
    assert_allclose(res.p_value_asymptotic, ref["p_value_asymptotic"], **EXACT)
    p = ref["p_value"]
    se = np.hypot(res.p_value_se, np.sqrt(p * (1 - p) / 1e5))
    assert abs(res.p_value - p) <= 4 * se, (res.p_value, p, se)


# --- maximally selected statistic -----------------------------------------


@pytest.mark.parametrize(
    "pmethod, method, rtol",
    [
        ("exactGauss", "exact", 1e-3),
        ("Lau94", "lausen94", 1e-9),
        ("Lau92", "lausen92", 1e-9),
    ],
)
def test_maxstat_matches_maxstat(pmethod, method, rtol):
    t, c, d = _lung()
    ref = values(SRC, f"maxstat_lung_age_{pmethod}")
    res = sp.maxstat_logrank(t, d["age"], c=c, method=method)
    assert_allclose(res.statistic, ref["statistic"], **EXACT)
    assert res.cutpoint == ref["cutpoint"]
    assert_allclose(res.cuts, ref["cuts"], **EXACT)
    assert_allclose(res.statistics, ref["stats"], **EXACT)
    assert_allclose(res.p_value, ref["p_value"], rtol=rtol)


def test_maxstat_permutation_matches_cond_mc():
    t, c, d = _lung()
    ref = values(SRC, "maxstat_lung_age_condMC")
    res = sp.maxstat_logrank(
        t, d["age"], c=c, method="permutation", n_permutations=99999,
        random_state=3,
    )  # fmt: skip
    p = ref["p_value"]
    se = np.hypot(res.p_value_se, np.sqrt(p * (1 - p) / 1e5))
    assert abs(res.p_value - p) <= 4 * se, (res.p_value, p, se)


def test_maxstat_veteran_karnofsky():
    # A small p-value (2e-5), where mvtnorm's integration in R warned that
    # it did not reach its error bound and is 5.5% below SurPyval's.
    # SurPyval's quadrature of the chain is checked against an independent
    # estimate of the same normal probability with the full correlation
    # matrix (the union estimator behind maxcombo_logrank, relative error
    # about 0.1%): they agree to 0.02% (seen), so R is the one that is off.
    from surpyval.univariate.nonparametric.logrank_maxcombo import (
        _max_normal_sf,
    )

    d = fixture("veteran")
    ref = values(SRC, "maxstat_veteran_karno_exactGauss")
    res = sp.maxstat_logrank(d["time"], d["karno"], c=1 - d["status"])
    assert_allclose(res.statistic, ref["statistic"], **EXACT)
    assert res.cutpoint == ref["cutpoint"]
    N = d["time"].size
    m = np.array([np.sum(d["karno"] <= cut) for cut in res.cuts], float)
    lo, hi = np.minimum.outer(m, m), np.maximum.outer(m, m)
    R = np.sqrt(lo * (N - hi) / (hi * (N - lo)))
    independent = _max_normal_sf(res.statistic, R, "two-sided")
    assert_allclose(res.p_value, independent, rtol=3e-3)
    assert_allclose(res.p_value, ref["p_value"], rtol=0.07)


def test_maxstat_cscores_differ_only_by_ties():
    # maxstat's own smethod = "LogRank" scores treat tied times as
    # ordered; recorded to show the difference is small and in the
    # statistic only (the same cut-point).
    t, c, d = _lung()
    ref = values(SRC, "maxstat_lung_age_exactGauss")
    res = sp.maxstat_logrank(t, d["age"], c=c)
    assert res.cutpoint == ref["cscores_cutpoint"]
    assert_allclose(res.statistic, ref["cscores_statistic"], rtol=1e-3)
