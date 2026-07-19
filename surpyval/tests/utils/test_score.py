"""Tests for the concordance index in ``surpyval.utils.score``.

``score`` computes Harrell's c-index for mortality-like risk scores: a
higher score predicts an earlier event. Pairs are compared with the
earlier time first, which must not depend on the order the samples are
passed in.
"""

import numpy as np
import pytest

from surpyval.utils.score import score


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
    # At a tied time the censored sample outlived the event, so the
    # event having the higher mortality scores 1, otherwise 0.5.
    x = [5.0, 5.0]
    c = [0, 1]
    assert score(x, c, [2.0, 1.0]) == 1.0
    assert score(x, c, [1.0, 2.0]) == 0.5


def test_tied_time_both_events():
    x = [5.0, 5.0]
    c = [0, 0]
    assert score(x, c, [3.0, 3.0]) == 1.0  # tied scores for a tied pair
    assert score(x, c, [1.0, 2.0]) == 0.5


def test_input_order_invariance():
    rng = np.random.default_rng(1)
    x = rng.exponential(10.0, 25)
    c = (rng.random(25) < 0.3).astype(int)
    mortality = rng.normal(0.0, 1.0, 25)
    baseline = score(x, c, mortality)
    perm = rng.permutation(25)
    assert score(x[perm], c[perm], mortality[perm]) == pytest.approx(baseline)
