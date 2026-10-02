"""
Tests Standby Nodes.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

import numpy as np
import pytest

import surpyval as surv
from surpyval import Uniform


def test_impermitted_truncation():
    x = Uniform.random(100, 0, 10)
    x = np.sort(x)
    tr = np.ones_like(x) * np.inf
    tr[-1] = 100
    # Truncation beyond the data does not change the exact-data MLE (it
    # used to be refused whenever the largest value was truncated).
    assert (
        pytest.approx(np.array([x.min(), x.max()]))
        == Uniform.fit(x, tr=tr).params
    )

    tr = np.ones_like(x) * np.inf
    tr[-2] = 100
    assert (
        pytest.approx(np.array([x.min(), x.max()]))
        == Uniform.fit(x, tr=tr).params
    )

    x = Uniform.random(100, 0, 10)
    x = np.sort(x)
    tl = np.ones_like(x) * np.inf
    tl[0] = -1
    # Every other value sits below its left truncation point: invalid data
    with pytest.raises(ValueError):
        Uniform.fit(x, tl=tl)


def test_fitted_support_is_data_dependent():
    # A fitted uniform's support is its [a, b] interval, not the whole real
    # line. The distribution declares NaN support and resolves it from the
    # fitted a/b parameters (support_param_index defaults to (0, 1)).
    assert np.all(np.isnan(Uniform.support))

    x = Uniform.random(1_000, 2.0, 7.0)
    model = Uniform.fit(x)
    assert pytest.approx(model.params) == np.array(model.support)
    assert np.isfinite(model.support).all()


def test_from_params_support_matches_params():
    model = Uniform.from_params([3.0, 9.0])
    assert np.array_equal(model.support, [3.0, 9.0])


def test_impermitted_censoring():
    # MLE takes exactly observed values only (#460): censored or not at the
    # extremes, and with or without any exact value.
    x = np.sort(Uniform.random(100, 0, 10))
    for idx, flag in ((-1, 1), (-2, 1), (0, -1), (50, 1)):
        c = np.zeros_like(x)
        c[idx] = flag
        with pytest.raises(ValueError, match="does not support censored"):
            Uniform.fit(x, c=c)
    with pytest.raises(ValueError, match="does not support censored"):
        Uniform.fit([2.0, 5.0, 6.0], c=[1, -1, -1])


# ---------------------------------------------------------------------------
# The Uniform MLE with censoring.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x, c",
    [
        ([0, 9.9, 9.9, 9.9, 10], [0, 1, 1, 1, 0]),  # right, below the max
        ([0, 10, 10, 5], [0, 0, 1, 0]),  # right, tied with the max
        ([-10, -10, -5, 0], [0, -1, 0, 0]),  # left
        ([1.0, 2, 3, 5], [0, 0, 0, 1]),  # right, at the max
    ],
)
def test_uniform_mle_refuses_censored_data(x, c):
    # The censored MLE exists but sits on a wall of the likelihood, where
    # its covariance was not positive definite and the Wald bounds were
    # nan or several times too wide (#460).
    with pytest.raises(ValueError, match="does not support censored"):
        surv.Uniform.fit(x, c=c)


def test_uniform_other_methods_take_censored_data():
    x, c = [0, 2, 4, 6, 9.9, 9.9, 10], [0, 0, 0, 0, 1, 1, 0]
    model = surv.Uniform.fit(x, c=c, how="MPS")
    assert model.params[0] <= 0 and model.params[1] >= 10


def test_uniform_complete_data_still_closed_form():
    x = np.array([2.0, 3.5, 7.0, 4.2])
    model = surv.Uniform.fit(x)
    np.testing.assert_array_equal(model.params, [2.0, 7.0])


def test_uniform_reports_its_closed_form_as_the_optimizer():
    assert surv.Uniform.fit([1.0, 2.0, 3.0, 5.0]).optimizer == "closed-form"


# ---------------------------------------------------------------------------
# The Uniform MLE with truncation.
# ---------------------------------------------------------------------------


def test_uniform_accepts_truncation_beyond_the_data():
    model = surv.Uniform.fit([1.0, 2, 3, 4], tr=100)
    assert np.allclose(model.params, [1, 4])


def test_uniform_refuses_censored_data_even_when_truncated():
    # (#460: no censored Uniform MLE; the truncated window used to add its
    # own "no unique MLE" refusal.)
    with pytest.raises(ValueError, match="does not support censored"):
        surv.Uniform.fit([1.0, 2, 3, 5], [0, 0, 0, 1], tr=100)
