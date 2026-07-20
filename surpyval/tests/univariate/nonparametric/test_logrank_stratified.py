"""Stratified log-rank test (#214).

The stratified log-rank accumulates the observed-minus-expected numerators and
their variances *within* each stratum before forming the statistic, so groups
are only ever compared against others in the same stratum. The tests pin:

* reduction to the ordinary log-rank with a single stratum;
* calibration under the null;
* the correction it provides -- when a stratum confounds the comparison, the
  pooled (unstratified) test rejects almost always while the stratified test
  stays near nominal; and
* power against a genuine within-stratum group effect.
"""

import numpy as np

from surpyval.univariate.nonparametric.logrank import logrank


def _rc(rng, scale, n):
    t = rng.exponential(scale)
    cens = rng.exponential(scale * 2.0)
    return np.minimum(t, cens), (cens < t).astype(int)


def test_single_stratum_equals_ordinary():
    x = np.array(
        [
            9,
            13,
            13,
            18,
            23,
            28,
            31,
            34,
            45,
            48,
            161,
            5,
            5,
            8,
            8,
            12,
            16,
            23,
            27,
            30,
            33,
            43,
            45.0,
        ]
    )
    c = np.array(
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    )
    Z = np.array([1] * 11 + [2] * 12)
    plain = logrank(x, Z, c=c)
    strat = logrank(x, Z, c=c, strata=np.ones(len(x)))
    assert strat.statistic == plain.statistic
    assert strat.strata == 1


def test_calibrated_under_null():
    def one(seed):
        rng = np.random.default_rng(seed)
        n = 300
        strata = rng.integers(0, 3, n)
        grp = rng.integers(0, 2, n)
        scale = np.array([3.0, 8.0, 15.0])[strata]  # baseline by stratum only
        x, c = _rc(rng, scale, n)
        return logrank(x, grp, c=c, strata=strata).p_value

    pvals = np.array([one(s) for s in range(200)])
    assert (pvals < 0.05).mean() < 0.12
    assert 0.4 < pvals.mean() < 0.6


def test_stratification_removes_confounding():
    # Stratum drives both the baseline hazard and the group allocation, but
    # there is no true within-stratum group effect. The pooled test is fooled;
    # the stratified test is not.
    def one(seed, stratified):
        rng = np.random.default_rng(seed)
        n = 400
        strata = rng.integers(0, 2, n)
        p_grp = np.where(strata == 0, 0.8, 0.2)
        grp = (rng.random(n) < p_grp).astype(int)
        scale = np.where(strata == 0, 3.0, 20.0)  # NO group effect
        x, c = _rc(rng, scale, n)
        if stratified:
            return logrank(x, grp, c=c, strata=strata).p_value
        return logrank(x, grp, c=c).p_value

    pooled = np.array([one(s, False) for s in range(200)])
    strat = np.array([one(s, True) for s in range(200)])
    assert (pooled < 0.05).mean() > 0.5  # badly over-rejects
    assert (strat < 0.05).mean() < 0.15  # near nominal


def test_power_against_within_stratum_effect():
    def one(seed):
        rng = np.random.default_rng(seed)
        n = 300
        strata = rng.integers(0, 3, n)
        grp = rng.integers(0, 2, n)
        scale = np.array([3.0, 8.0, 15.0])[strata] * np.where(
            grp == 1, 0.5, 1.0
        )
        x, c = _rc(rng, scale, n)
        return logrank(x, grp, c=c, strata=strata).p_value

    pvals = np.array([one(s) for s in range(60)])
    assert (pvals < 0.05).mean() > 0.8


def test_strata_length_mismatch_raises():
    import pytest

    x = [1.0, 2.0, 3.0, 4.0]
    Z = [0, 0, 1, 1]
    with pytest.raises(ValueError, match="label for each observation"):
        logrank(x, Z, strata=[0, 1])
