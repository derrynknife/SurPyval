"""Edge cases and refusals of the concordance index and Brier scores.

The values are closed forms where one exists (a perfect, reversed or
fully tied ranking; a Brier score where every unit failed before the
time) and otherwise counts of concordant pairs worked by hand, which
agree with the values observed at commit e3412770. The refusal tests
pin inputs that are refused, naming the argument (Design Principle 2).
"""

import numpy as np
import pytest

from surpyval import Weibull
from surpyval.metrics import (
    brier_score,
    concordance_index,
    integrated_brier_score,
)

X4 = [1.0, 2.0, 3.0, 4.0]
C4 = [0, 0, 0, 0]


def _brier_data():
    xx = np.random.default_rng(1).weibull(2, 30) * 10
    m = Weibull.fit(xx)
    times = [2.0, 5.0, 8.0]
    S = np.tile(m.sf(times), (30, 1))
    return xx, np.zeros(30), S, times


def test_concordance_refuses_nan_risk():
    """``concordance_index([1, 2, 3, 4], [0] * 4, [4, nan, 2, 1])``
    returned nan at e3412770; it is refused, naming ``risk``."""
    with pytest.raises(ValueError, match="risk"):
        concordance_index(X4, C4, [4.0, np.nan, 2.0, 1.0])


def test_brier_refuses_probabilities_outside_unit_interval():
    """``brier_score(x, c, S * 2, times)`` returned scores at e3412770;
    it is refused, naming ``survival``."""
    xx, c, S, times = _brier_data()
    with pytest.raises(ValueError, match="survival"):
        brier_score(xx, c, S * 2, times)


@pytest.mark.parametrize(
    "x, c, risk, expected",
    [
        (X4, C4, [4.0, 3.0, 2.0, 1.0], 1.0),  # perfect ranking
        (X4, C4, [1.0, 2.0, 3.0, 4.0], 0.0),  # reversed
        (X4, C4, [1.0, 1.0, 1.0, 1.0], 0.5),  # all tied
        ([1.0, 2.0], [0, 0], [2.0, 1.0], 1.0),  # one pair
        # A huge finite risk ranks top: one discordant pair, 5 / 6.
        (X4, C4, [4.0, 1e300, 2.0, 1.0], 0.8333333333333334),
        # Within tie_tol: risks 2 and 2+1e-9 are tied, 5.5 / 6.
        (X4, C4, [4.0, 3.0, 2.0, 2.0 + 1e-9], 0.9166666666666666),
        # Outside tie_tol: one discordant pair, 5 / 6.
        (X4, C4, [4.0, 3.0, 2.0, 2.0 + 1e-7], 0.8333333333333334),
        # Equal infinities are tied: the first pair counts 0.5, 5.5 / 6.
        (X4, C4, [np.inf, np.inf, 2.0, 1.0], 0.9166666666666666),
        (X4, C4, [4.0, 3.0, -np.inf, -np.inf], 0.9166666666666666),
        # An infinity is not tied with the largest finite score: 6 / 6.
        (X4, C4, [np.inf, 1.7e308, 2.0, 1.0], 1.0),
    ],
)
def test_concordance_values(x, c, risk, expected):
    assert concordance_index(x, c, risk) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize(
    "risk, expected",
    [
        # inf ranks top, like 1e300: (t=1, risk 4) vs (t=2, inf) is the
        # one discordant pair, 5 / 6.
        ([4.0, np.inf, 2.0, 1.0], 5 / 6),
        # -inf ranks bottom: two discordant pairs, 4 / 6.
        ([4.0, -np.inf, 2.0, 1.0], 4 / 6),
    ],
    ids=["inf", "-inf"],
)
def test_concordance_infinite_risk(risk, expected):
    """At e3412770: 0.9166666666666666 for ``[4, inf, 2, 1]`` and
    0.8333333333333334 for ``[4, -inf, 2, 1]``; the O(n^2) definition
    (``pairwise`` in ``test_concordance.py``) and a finite 1e300 in place
    of inf give 5 / 6 and 4 / 6. In ``surpyval/metrics/concordance.py``,
    ``_close`` bounds ``|a - b|`` by ``1e-9 * max(|a|, |b|)``, which is
    inf when either score is infinite, so an infinite score counted as
    tied with finite scores; ``math.isclose``, which it mirrors, does not,
    and now neither does ``_close``.
    """
    assert concordance_index(X4, C4, risk) == pytest.approx(
        expected, abs=1e-15
    )


def test_concordance_order_invariant():
    x, c, risk = [1, 2, 3, 4], [0, 1, 0, 0], [4, 1, 3, 2]
    assert concordance_index(x, c, risk) == 1.0
    assert concordance_index(x[::-1], c[::-1], risk[::-1]) == 1.0


def test_concordance_tie_conventions():
    x, c, risk = [1, 1, 2, 3], C4, [3, 2, 2, 1]
    assert concordance_index(x, c, risk, ties="harrell") == pytest.approx(
        0.8333333333333334, abs=1e-15
    )
    assert concordance_index(x, c, risk) == pytest.approx(0.9, abs=1e-15)


@pytest.mark.parametrize(
    "x, c, risk",
    [
        ([1.0, 1.0, 1.0], [0, 0, 0], [1.0, 2.0, 3.0]),  # all times tied
        ([1.0], [0], [1.0]),  # one subject
        (X4, [1, 1, 1, 1], [4.0, 3.0, 2.0, 1.0]),  # all censored
    ],
    ids=["tied-times", "n=1", "all-censored"],
)
def test_concordance_refuses_no_pairs(x, c, risk):
    with pytest.raises(ValueError, match="No usable pairs"):
        concordance_index(x, c, risk)


@pytest.mark.parametrize("s, expected", [(1.0, 1.0), (0.0, 0.0)])
def test_brier_after_every_failure(s, expected):
    # Every unit failed before t = 1e6, so the score is mean(S(t)**2).
    xx, c, *_ = _brier_data()
    _, score = brier_score(xx, c, np.full((30, 1), s), [1e6])
    assert score[0] == expected


def test_integrated_brier_unsorted_times():
    xx, c, S, times = _brier_data()
    sorted_call = integrated_brier_score(xx, c, S, times)
    reversed_call = integrated_brier_score(xx, c, S[:, ::-1], times[::-1])
    assert reversed_call == pytest.approx(sorted_call, abs=1e-12)
    # Observed at e3412770 on this data, pinned as a regression value.
    assert sorted_call == pytest.approx(0.15903805225743808, rel=1e-9)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_brier_refusal_counts_the_values(bad):
    xx, c, S, times = _brier_data()
    S = S.copy()
    S[4, 1] = bad
    with pytest.raises(ValueError, match="1 of its 90 values are outside"):
        brier_score(xx, c, S, times)
    with pytest.raises(ValueError, match="survival"):
        integrated_brier_score(xx, c, S, times)


def test_brier_refuses_wrong_shape():
    xx, c, S, times = _brier_data()
    with pytest.raises(ValueError, match="survival"):
        brier_score(xx, c, S[:, :2], times)


def test_model_concordance_with_a_missing_covariate_is_nan():
    # A model's concordance on new data with a missing covariate keeps the
    # prediction rule (NaN in, NaN out) rather than the refusal of a NaN
    # score passed to concordance_index directly.
    from surpyval import CoxPH

    rng = np.random.default_rng(3)
    Z = rng.normal(size=(20, 1))
    x = rng.weibull(1.5, 20) * np.exp(-0.5 * Z[:, 0])
    model = CoxPH.fit(x, Z)
    Z_new = Z.copy()
    Z_new[2, 0] = np.nan
    assert np.isnan(model.concordance(x, np.zeros(20), Z_new))
