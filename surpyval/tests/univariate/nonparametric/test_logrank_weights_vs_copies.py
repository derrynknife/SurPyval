"""Counts ``n`` and sampling weights ``w`` in the log-rank test, against
the same rows written out ``w`` times (see "Weights and copies in the
log-rank test" in ``docs/Weighted Data.rst``).

The counts ``n`` are frequency weights: they give the answer of the copies
to rounding. The weights ``w`` give the same observed-minus-expected
numerator but a different variance (the robust one, which counts the
units, not the copies), so a different statistic and p-value."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.univariate.nonparametric.logrank import (
    WEIGHTINGS,
    _logrank_z_v,
    _weighted_u_v,
)

W = np.array([1, 3, 2, 4, 1, 2, 3, 1, 4, 2, 1, 3, 2, 1, 4, 2])
TIES_X = np.array([2, 3, 3, 5, 5, 5, 7, 7, 8, 9, 9, 10, 12, 12, 14, 15.0])
TIES_Z = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
TIES_TL = np.array([0, 0, 1, 0, 2, 0, 3, 0, 1, 0, 2, 4, 0, 5, 0, 6.0])
CENSORED = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])

DATASETS = {
    "no_ties": dict(
        x=np.array(
            [15.8, 8.26, 11.92, 6.54, 0.97, 5.73, 7.44, 2.37]
            + [1.5, 10.0, 10.59, 11.08, 3.73, 16.82, 12.7, 15.95]
        ),
        Z=np.arange(16) % 2,
        c=CENSORED,
        tl=None,
    ),
    "heavy_ties_three_groups": dict(x=TIES_X, Z=TIES_Z, c=CENSORED, tl=None),
    "ties_and_late_entry": dict(x=TIES_X, Z=TIES_Z, c=CENSORED, tl=TIES_TL),
}


def _kw(weighting):
    return {"rho": 1, "gamma": 1} if weighting == "fleming-harrington" else {}


def _copies(d):
    idx = np.repeat(np.arange(W.size), W)
    tl = None if d["tl"] is None else d["tl"][idx]
    return d["x"][idx], d["Z"][idx], d["c"][idx], tl


@pytest.mark.parametrize("name", list(DATASETS))
@pytest.mark.parametrize("weighting", WEIGHTINGS)
def test_counts_n_equal_physical_copies(name, weighting):
    d = DATASETS[name]
    x, Z, c, tl = _copies(d)
    kw = _kw(weighting)
    by_n = sp.logrank(
        d["x"], d["Z"], c=d["c"], n=W, tl=d["tl"], weighting=weighting, **kw
    )
    by_copies = sp.logrank(x, Z, c=c, tl=tl, weighting=weighting, **kw)
    assert by_n.dof == by_copies.dof
    assert_allclose(by_n.statistic, by_copies.statistic, rtol=1e-10)
    assert_allclose(by_n.p_value, by_copies.p_value, rtol=1e-10)


def test_pairwise_counts_n_equal_physical_copies():
    d = DATASETS["heavy_ties_three_groups"]
    x, Z, c, _ = _copies(d)
    by_n = sp.pairwise_logrank(d["x"], d["Z"], c=d["c"], n=W)
    by_copies = sp.pairwise_logrank(x, Z, c=c)
    assert_allclose(
        by_n.table["statistic"], by_copies.table["statistic"], rtol=1e-10
    )


@pytest.mark.parametrize("name", list(DATASETS))
@pytest.mark.parametrize(
    "weighting", ["log-rank", "fleming-harrington", "gehan"]
)
def test_weights_w_share_the_numerator_but_not_the_variance(name, weighting):
    d = DATASETS[name]
    x, Z, c, tl = _copies(d)
    groups = np.unique(d["Z"])
    kw = _kw(weighting)
    u_w, M, _ = _weighted_u_v(
        d["x"],
        d["Z"],
        d["c"],
        None,
        W.astype(float),
        groups,
        weighting,
        kw.get("rho", 0),
        kw.get("gamma", 0),
        tl=d["tl"],
    )
    u_c, V, _ = _logrank_z_v(
        x,
        Z,
        c,
        None,
        groups,
        weighting,
        kw.get("rho", 0),
        kw.get("gamma", 0),
        tl,
    )
    # Same observed minus expected events, to rounding ...
    assert_allclose(u_w, u_c, rtol=1e-10, atol=1e-10)
    # ... against a variance about sum(w^2) / sum(w) times larger (here
    # 7.5 / 2.5 = 3 on average): the robust one counts 16 units, not 40.
    ratio = M[0, 0] / V[0, 0]
    assert 2.0 < ratio < 4.0


@pytest.mark.parametrize("name", list(DATASETS))
@pytest.mark.parametrize("weighting", ["log-rank", "fleming-harrington"])
def test_weights_w_give_a_smaller_statistic_than_copies(name, weighting):
    d = DATASETS[name]
    x, Z, c, tl = _copies(d)
    kw = _kw(weighting)
    by_w = sp.logrank(
        d["x"], d["Z"], c=d["c"], w=W, tl=d["tl"], weighting=weighting, **kw
    )
    by_copies = sp.logrank(x, Z, c=c, tl=tl, weighting=weighting, **kw)
    # The copies claim 40 units' worth of information from 16 units.
    assert by_w.statistic < by_copies.statistic
    assert by_w.p_value > by_copies.p_value


def test_weights_w_scale_does_not_matter_but_copies_do():
    d = DATASETS["no_ties"]
    base = sp.logrank(d["x"], d["Z"], c=d["c"], w=W)
    scaled = sp.logrank(d["x"], d["Z"], c=d["c"], w=10 * W)
    assert_allclose(scaled.statistic, base.statistic, rtol=1e-10)
    # Ten copies of each row is ten times the data: ten times the statistic.
    one = sp.logrank(d["x"], d["Z"], c=d["c"], n=W)
    ten = sp.logrank(d["x"], d["Z"], c=d["c"], n=10 * W)
    assert ten.statistic > 5 * one.statistic


def test_unit_weights_are_the_robust_test_not_the_classical_one():
    d = DATASETS["heavy_ties_three_groups"]
    plain = sp.logrank(d["x"], d["Z"], c=d["c"])
    unit = sp.logrank(d["x"], d["Z"], c=d["c"], w=np.ones(d["x"].size))
    assert abs(plain.statistic - unit.statistic) > 1e-3
