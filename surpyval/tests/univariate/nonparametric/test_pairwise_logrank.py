"""Pairwise log-rank tests with a multiplicity adjustment."""

import json
import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.nonparametric.logrank import adjust_p_values


def _lung():
    df = sp.datasets.load_lung().dropna(subset=["ph.ecog"])
    df = df[df["ph.ecog"] < 3]
    return (
        df["time"].to_numpy(),
        df["ph.ecog"].astype(int).to_numpy(),
        1 - df["status"].to_numpy(),
    )


# lifelines 0.30.3, pairwise_logrank_test(time, ph.ecog, event, ...) on
# the lung data by ECOG score 0, 1 and 2: the unadjusted statistics.
LIFELINES_LUNG = {
    "log-rank": [3.456831, 16.682328, 8.433446],
    "peto-peto": [3.600063, 17.264611, 10.672991],
    "fleming-harrington": [2.956593, 13.584734, 4.493224],  # p = q = 1
}


@pytest.mark.parametrize("weighting", sorted(LIFELINES_LUNG))
def test_pairwise_matches_lifelines_on_lung(weighting):
    x, Z, c = _lung()
    kwargs = (
        {"rho": 1, "gamma": 1} if weighting == "fleming-harrington" else {}
    )
    res = sp.pairwise_logrank(x, Z, c=c, weighting=weighting, **kwargs)
    assert list(zip(res.table.group_a, res.table.group_b)) == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    np.testing.assert_allclose(
        res.table.statistic, LIFELINES_LUNG[weighting], atol=5e-7
    )


def test_pairwise_t0_matches_lifelines():
    # lifelines 0.30.3, pairwise_logrank_test(..., t_0=365).
    x, Z, c = _lung()
    res = sp.pairwise_logrank(x, Z, c=c, t0=365)
    np.testing.assert_allclose(
        res.table.statistic, [3.151803, 17.170780, 10.011764], atol=5e-7
    )


def test_each_row_is_logrank_on_that_pair():
    x, Z, c = _lung()
    strata = (np.arange(x.size) % 3).astype(int)
    res = sp.pairwise_logrank(x, Z, c=c, strata=strata, weighting="gehan")
    for row in res.table.itertuples():
        mask = (Z == row.group_a) | (Z == row.group_b)
        alone = sp.logrank(
            x[mask], Z[mask], c=c[mask], strata=strata[mask], weighting="gehan"
        )
        assert row.statistic == alone.statistic
        assert row.p_value == alone.p_value
    assert res.strata == 3


def test_holm_is_the_default_adjustment():
    x, Z, c = _lung()
    res = sp.pairwise_logrank(x, Z, c=c)
    assert res.adjust == "holm"
    np.testing.assert_array_equal(
        res.table.p_adjusted, adjust_p_values(res.table.p_value, "holm")
    )
    raw = sp.pairwise_logrank(x, Z, c=c, adjust="none")
    np.testing.assert_array_equal(raw.table.p_adjusted, raw.table.p_value)


# The adjusted p-values worked by hand from the definitions in Holm (1979)
# and Benjamini and Hochberg (1995), the methods of R's p.adjust, for
# p = (0.01, 0.02, 0.03, 0.04, 0.05) and (0.04, 0.001, 0.03, 0.2, 0.012)
# (out of order on purpose).
@pytest.mark.parametrize(
    "p, method, expected",
    [
        (
            [0.01, 0.02, 0.03, 0.04, 0.05],
            "holm",
            [0.05, 0.08, 0.09, 0.09, 0.09],
        ),
        ([0.01, 0.02, 0.03, 0.04, 0.05], "bh", [0.05] * 5),
        (
            [0.01, 0.02, 0.03, 0.04, 0.05],
            "bonferroni",
            [0.05, 0.10, 0.15, 0.20, 0.25],
        ),
        (
            [0.04, 0.001, 0.03, 0.2, 0.012],
            "holm",
            [0.09, 0.005, 0.09, 0.2, 0.048],
        ),
        (
            [0.04, 0.001, 0.03, 0.2, 0.012],
            "bh",
            [0.05, 0.005, 0.05, 0.2, 0.03],
        ),
        ([0.5, 0.6], "bonferroni", [1.0, 1.0]),
    ],
)
def test_adjust_p_values_against_hand_calculation(p, method, expected):
    np.testing.assert_allclose(
        adjust_p_values(p, method), expected, rtol=1e-12
    )


def test_adjust_p_values_skips_nan_and_refuses_unknown():
    out = adjust_p_values([0.01, np.nan, 0.04], "holm")
    np.testing.assert_allclose(out, [0.02, np.nan, 0.04])
    with pytest.raises(ValueError, match="'adjust'"):
        adjust_p_values([0.1], "sidak")
    with pytest.raises(ValueError, match="'adjust'"):
        sp.pairwise_logrank([1, 2, 3, 4], [0, 0, 1, 1], adjust="sidak")


def test_a_pair_with_nothing_to_compare_is_not_counted():
    # Group 2 has no event and is entered after every event of 0 and 1:
    # the pair (0, 2) has an expected count only in group 0.
    x = np.array([1, 2, 3, 4, 5, 6, 10, 11])
    c = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    Z = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    tl = np.array([0, 0, 0, 0, 0, 0, 7, 7])
    res = sp.pairwise_logrank(x, Z, c=c, tl=tl, adjust="bonferroni")
    table = res.table.set_index(["group_a", "group_b"])
    assert table.loc[(0, 2), "dof"] == 0
    assert table.loc[(0, 2), "p_adjusted"] == 1.0
    # Only the one informative pair is a test, so nothing is multiplied.
    assert table.loc[(0, 1), "p_adjusted"] == table.loc[(0, 1), "p_value"]


def test_pairwise_warns_once_for_singleton_groups():
    x = np.arange(1.0, 9.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.pairwise_logrank(x, x * 1.3)
    assert sum("single member" in str(w.message) for w in caught) == 1


def test_pairwise_refusals_name_the_argument():
    with pytest.raises(ValueError, match="at least two groups"):
        sp.pairwise_logrank([1, 2, 3], [0, 0, 0])
    with pytest.raises(ValueError, match="'weighting'"):
        sp.pairwise_logrank([1, 2, 3, 4], [0, 0, 1, 1], weighting="x")
    with pytest.raises(ValueError, match="'Z'"):
        sp.pairwise_logrank([1, 2, 3, 4], [0, None, 1, 1])
    with pytest.raises(ValueError, match="'c'"):
        sp.pairwise_logrank([1, 2, 3, 4], [0, 0, 1, 1], c=[0, 0])


def test_result_repr_and_strict_json():
    x, Z, c = _lung()
    res = sp.pairwise_logrank(x, Z, c=c, weighting="peto-peto")
    text = repr(res)
    assert "Pairwise Log-Rank Tests" in text and "holm" in text
    out = res.to_dict()
    json.dumps(out, allow_nan=False)
    assert out["weighting"] == "peto-peto"
    assert [(r["group_a"], r["group_b"]) for r in out["pairs"]] == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
