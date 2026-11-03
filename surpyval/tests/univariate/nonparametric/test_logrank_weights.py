"""The log-rank tests of data with sampling weights ``w``: the properties
(R's values are in ``reference/test_weighted_logrank.py``)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.tests._helpers import weighted_reference_data
from surpyval.univariate.nonparametric.logrank import WEIGHTINGS


@pytest.fixture(scope="module")
def data():
    d = weighted_reference_data()
    d["g3"] = np.digitize(d["Z"][:, 1], [-0.5, 0.5])
    d["s"] = d["Z"][:, 0].astype(int)
    return d


def _kw(weighting):
    return {"rho": 1, "gamma": 1} if weighting == "fleming-harrington" else {}


@pytest.mark.parametrize("weighting", WEIGHTINGS)
@pytest.mark.parametrize("scale", [0.1, 7.3])
def test_weight_scale_changes_nothing(data, weighting, scale):
    x, c, w, g = data["x"], data["c"], data["w"], data["g3"]
    kw = dict(c=c, tl=data["tl"], strata=data["s"], weighting=weighting)
    kw.update(_kw(weighting))
    base = sp.logrank(x, g, w=w, **kw)
    scaled = sp.logrank(x, g, w=w * scale, **kw)
    assert_allclose(scaled.statistic, base.statistic, rtol=1e-10)
    assert scaled.dof == base.dof == 2


@pytest.mark.parametrize("weighting", WEIGHTINGS)
def test_row_order_and_counts(data, weighting):
    x, c, w, g = data["x"], data["c"], data["w"], data["g3"]
    kw = dict(weighting=weighting, **_kw(weighting))
    base = sp.logrank(x, g, c=c, w=w, **kw).statistic
    order = np.random.default_rng(3).permutation(x.size)
    permuted = sp.logrank(x[order], g[order], c=c[order], w=w[order], **kw)
    assert_allclose(permuted.statistic, base, rtol=1e-10)
    # A row counted twice is the row written twice.
    n = np.ones(x.size, dtype=int)
    n[:5] = 2
    rows = np.concatenate([np.arange(x.size), np.arange(5)])
    counted = sp.logrank(x, g, c=c, n=n, w=w, **kw).statistic
    written = sp.logrank(x[rows], g[rows], c=c[rows], w=w[rows], **kw)
    assert_allclose(counted, written.statistic, rtol=1e-10)


def test_numerator_with_whole_weights_is_the_counted_one(data):
    # The numerator (the score) of whole-number weights is that of the
    # rows counted that many times; only the variance differs, robust
    # against hypergeometric. Checked through the unit-weight identity:
    # with w = 1 and with w = 2 the statistic is the same.
    x, c, g = data["x"], data["c"], data["g3"]
    one = sp.logrank(x, g, c=c, w=np.ones(x.size)).statistic
    two = sp.logrank(x, g, c=c, w=np.full(x.size, 2.0)).statistic
    assert_allclose(one, two, rtol=1e-12)


def test_unweighted_test_is_unchanged(data):
    x, c, g = data["x"], data["c"], data["g3"]
    res = sp.logrank(x, g, c=c)
    assert not res.weighted
    assert "robust" not in repr(res)
    assert "robust" in repr(sp.logrank(x, g, c=c, w=data["w"]))


def test_pairwise_takes_w(data):
    x, c, w, g = data["x"], data["c"], data["w"], data["g3"]
    res = sp.pairwise_logrank(x, g, c=c, w=w, tl=data["tl"])
    for _, row in res.table.iterrows():
        mask = (g == row["group_a"]) | (g == row["group_b"])
        alone = sp.logrank(
            x[mask], g[mask], c=c[mask], w=w[mask], tl=data["tl"][mask]
        )
        assert_allclose(row["statistic"], alone.statistic, rtol=1e-12)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_bad_weights_are_refused(data, bad):
    w = data["w"].copy()
    w[3] = bad
    with pytest.raises(ValueError, match="'w'"):
        sp.logrank(data["x"], data["g3"], c=data["c"], w=w)


def test_weights_need_one_per_row(data):
    with pytest.raises(ValueError, match="one weight per row"):
        sp.logrank(data["x"], data["g3"], c=data["c"], w=data["w"][:-1])


def test_interval_flags_still_refused(data):
    with pytest.raises(ValueError, match="right censored"):
        sp.logrank(
            data["x"], data["g3"], c=np.full(data["x"].size, 2), w=data["w"]
        )
