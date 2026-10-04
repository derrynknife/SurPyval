"""Tests for the concordance index, ``surpyval.metrics.concordance_index``
(formerly ``surpyval.utils.score.score``).

It computes Harrell's c-index (with Therneau's tied-event convention by
default) for mortality-like risk scores: a higher score predicts an
earlier event. Pairs are compared with the earlier time first, which must
not depend on the order the samples are passed in.
"""

import numpy as np
import pytest

from surpyval.metrics import concordance_index as score


def test_perfect_concordance_is_one():
    # Shuffled input: the pair ordering must come from the times, not
    # from the input order.
    x = [3.0, 1.0, 4.0, 2.0]
    c = [0, 0, 0, 0]
    mortality = [20.0, 40.0, 10.0, 30.0]  # strictly decreasing in x
    assert score(x, c, mortality) == 1.0


def test_perfect_anticoncordance_is_zero():
    x = [3.0, 1.0, 4.0, 2.0]
    c = [0, 0, 0, 0]
    mortality = [30.0, 10.0, 40.0, 20.0]  # strictly increasing in x
    assert score(x, c, mortality) == 0.0


def test_constant_scores_give_half():
    x = [1.0, 2.0, 3.0, 4.0]
    c = [0, 0, 0, 0]
    assert score(x, c, [7.0] * 4) == 0.5


def test_pairs_with_earlier_censored_are_omitted():
    # (x=1, censored) is incomparable with both later samples, so only
    # the (2, 3) pair counts -- and it is concordant.
    x = [1.0, 2.0, 3.0]
    c = [1, 0, 0]
    mortality = [0.0, 5.0, 1.0]
    assert score(x, c, mortality) == 1.0


def test_censored_after_event_still_comparable():
    # The event at x=1 precedes the censoring at x=2: the pair counts,
    # and the event carrying the higher mortality is concordant.
    x = [1.0, 2.0]
    c = [0, 1]
    assert score(x, c, [5.0, 1.0]) == 1.0
    assert score(x, c, [1.0, 5.0]) == 0.0


def test_tied_time_event_vs_censored():
    # At a tied time the censored sample outlived the event, so the pair
    # is fully comparable (Harrell): the event having the higher
    # mortality scores 1, a genuine score tie 0.5, and the event having
    # the *lower* mortality is discordant and scores 0 (it used to be
    # credited 0.5, biasing tie-heavy data toward 0.5, #276).
    x = [5.0, 5.0]
    c = [0, 1]
    assert score(x, c, [2.0, 1.0]) == 1.0
    assert score(x, c, [1.5, 1.5]) == 0.5
    assert score(x, c, [1.0, 2.0]) == 0.0


def test_tied_time_both_events():
    # Harrell's original convention: a usable pair, 1 for tied scores,
    # else 0.5.
    x = [5.0, 5.0]
    c = [0, 0]
    assert score(x, c, [3.0, 3.0], ties="harrell") == 1.0
    assert score(x, c, [1.0, 2.0], ties="harrell") == 0.5
    # Therneau's (the default, as R and lifelines): not a pair.
    with pytest.raises(ValueError, match="No usable pairs"):
        score(x, c, [1.0, 2.0])
    assert score(x + [6.0], c + [1], [1.0, 2.0, 0.0]) == 1.0


def test_harrell_convention_of_the_removed_score():
    # surpyval.utils.score.score (removed in v0.23) counted tied events as
    # a pair, Harrell's convention, which ties="harrell" gives.
    x, c, s = [1.0, 1, 2, 3], [0, 0, 0, 1], [0.9, 0.5, 0.7, 0.2]
    assert score(x, c, s, ties="harrell") == 0.75
    assert score(x, c, s) == 0.8


def test_input_order_invariance():
    rng = np.random.default_rng(1)
    x = rng.exponential(10.0, 25)
    c = (rng.random(25) < 0.3).astype(int)
    mortality = rng.normal(0.0, 1.0, 25)
    baseline = score(x, c, mortality)
    perm = rng.permutation(25)
    assert score(x[perm], c[perm], mortality[perm]) == pytest.approx(baseline)


# ---------------------------------------------------------------------------
# #276: a discordant tied pair scores zero.
# ---------------------------------------------------------------------------


def test_concordance_discordant_tied_pair_scores_zero():
    # 276 (unit cases in tests/utils/test_score.py; pinned here too).
    from surpyval.metrics import concordance_index as score

    assert score([5.0, 5.0], [0, 1], [1.0, 2.0]) == 0.0
    assert score([5.0, 5.0], [0, 1], [2.0, 1.0]) == 1.0
