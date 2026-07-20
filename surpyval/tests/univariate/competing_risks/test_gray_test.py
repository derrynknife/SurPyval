"""Gray's test for comparing cumulative incidence functions (#216).

Gray's test is validated by simulation: under the null (identical cause-``k``
cumulative incidence across groups) its p-values are ~Uniform -- including
under independent censoring, which exercises the IPCW subdistribution
weighting -- and it is powerful against a genuine CIF difference.
"""

import numpy as np
import pytest

import surpyval as sp


def _sim_cr(seed, n, lam1_by_group, lam2=0.5, cens_rate=0.0, groups=2):
    """Two-cause competing-risks sample; cause-1 rate varies by group."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, groups, n)
    lam1 = np.asarray(lam1_by_group)[g]
    t1 = rng.exponential(1 / lam1)
    t2 = rng.exponential(1 / lam2, n)
    x = np.minimum(t1, t2)
    e = np.where(t1 < t2, 1, 2).astype(object)
    if cens_rate > 0:
        cens = rng.exponential(1 / cens_rate, n)
        obs = x <= cens
        x = np.where(obs, x, cens)
        e = np.where(obs, e, None)
    return x, e, g


def _null_pvalues(cens_rate=0.0, groups=2, reps=120):
    out = []
    for s in range(reps):
        x, e, g = _sim_cr(
            s, 200, [1.0] * groups, cens_rate=cens_rate, groups=groups
        )
        out.append(sp.gray_test(x, e, g, cause=1).p_value)
    return np.array(out)


def test_gray_null_calibration_no_censoring():
    p = _null_pvalues(cens_rate=0.0)
    assert (p < 0.05).mean() < 0.12
    assert 0.4 < p.mean() < 0.6


def test_gray_null_calibration_with_censoring():
    # The IPCW subdistribution weighting must keep the test calibrated when
    # observations are independently censored.
    p = _null_pvalues(cens_rate=0.4)
    assert (p < 0.05).mean() < 0.12
    assert 0.4 < p.mean() < 0.6


def test_gray_null_calibration_three_groups():
    p = _null_pvalues(cens_rate=0.0, groups=3)
    assert (p < 0.05).mean() < 0.12
    result = sp.gray_test(*_sim_cr(0, 200, [1.0, 1.0, 1.0], groups=3), cause=1)
    assert result.df == 2


def test_gray_detects_cif_difference():
    x, e, g = _sim_cr(1, 400, [2.0, 0.5], cens_rate=0.0)
    result = sp.gray_test(x, e, g, cause=1)
    assert result.p_value < 0.01
    assert result.df == 1
    assert result.cause == 1


def test_gray_censoring_inferred_from_none():
    # e is None for censored; passing c explicitly must give the same result.
    x, e, g = _sim_cr(2, 200, [1.5, 0.8], cens_rate=0.3)
    c = np.array([1 if ei is None else 0 for ei in e])
    r_inferred = sp.gray_test(x, e, g, cause=1)
    r_explicit = sp.gray_test(x, e, g, cause=1, c=c)
    assert r_inferred.p_value == pytest.approx(r_explicit.p_value)


def test_gray_requires_two_groups():
    x, e, g = _sim_cr(3, 100, [1.0], groups=1)
    with pytest.raises(ValueError, match="two groups"):
        sp.gray_test(x, e, g, cause=1)


def test_gray_unknown_cause_raises():
    x, e, g = _sim_cr(4, 100, [1.0, 1.0])
    with pytest.raises(ValueError, match="No events of cause"):
        sp.gray_test(x, e, g, cause=99)
