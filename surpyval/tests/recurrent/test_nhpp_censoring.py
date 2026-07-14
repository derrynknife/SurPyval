import numpy as np
import pytest

from surpyval.recurrent import HPP, CrowAMSAA
from surpyval.utils.recurrent_utils import handle_xicn


def test_hpp_interval_censoring_matches_analytic_rate():
    # Interval-censored counts: n_k events in [a_k, b_k]. The HPP rate MLE is
    # total events / total exposure. Here 6 events over a width-15 window.
    x = [[0, 5], [5, 10], [10, 15]]
    c = [2, 2, 2]
    n = [2, 3, 1]
    i = [1, 1, 1]
    model = HPP.fit(x, i, c=c, n=n)
    assert np.isclose(model.params[0], 6.0 / 15.0)


def test_crow_amsaa_interval_censoring_fits():
    x = [[0, 5], [5, 10], [10, 20]]
    c = [2, 2, 2]
    n = [1, 2, 2]
    i = [1, 1, 1]
    model = CrowAMSAA.fit(x, i, c=c, n=n)
    assert np.all(np.isfinite(model.params))


def test_interval_left_must_not_exceed_right():
    # A reversed interval [5, 1] is invalid regardless of input type. The
    # check is shared with the univariate handler via coerce_xcnt_x.
    with pytest.raises(ValueError, match="less than or equal to right"):
        handle_xicn(np.array([[5.0, 1.0]]), np.array([1]), c=np.array([2]))
    with pytest.raises(ValueError, match="less than or equal to right"):
        handle_xicn([[5.0, 1.0]], [1], c=[2])


def test_single_interval_row_does_not_crash():
    # A single interval row must not trip the old row-0-vs-row-1 check.
    data = handle_xicn([[1.0, 5.0]], [1], c=[2], n=[2])
    assert data.x.shape == (1, 2)


def test_interval_censoring_respects_truncation_window():
    # An interval beginning below the left-truncation bound is rejected.
    x = np.array([[1.0, 5.0], [6.0, 9.0]])
    with pytest.raises(ValueError, match="outside its observation window"):
        HPP.fit(
            x, np.array([1, 1]), c=np.array([2, 2]), n=np.array([1, 1]), tl=3.0
        )
