"""The MCF's two-sided bounds name their columns (ConfidenceBounds)."""

import pickle

import numpy as np
import pytest

from surpyval import CauseSpecificMCF, NonParametricCounting
from surpyval.utils.bounds import ConfidenceBounds

X = [5, 10, 30, 8, 25, 70, 40, 60, 90]
I = [1, 1, 1, 2, 2, 2, 3, 3, 3]
C = [0, 0, 0, 0, 0, 1, 0, 0, 1]


@pytest.fixture(scope="module")
def model():
    return NonParametricCounting.fit(X, i=I, c=C)


def test_two_sided_is_labelled_and_unchanged(model):
    b = model.mcf_cb([10, 30, 60], alpha_ci=0.1)
    assert isinstance(b, ConfidenceBounds)
    assert b.shape == (3, 2)
    np.testing.assert_array_equal(b.lower, np.asarray(b)[:, 0])
    np.testing.assert_array_equal(b.upper, np.asarray(b)[:, 1])
    assert np.all(b.lower <= model.mcf([10, 30, 60]))
    assert np.all(model.mcf([10, 30, 60]) <= b.upper)
    # The one-sided bounds at twice the level are the same numbers.
    np.testing.assert_array_equal(
        b.lower, model.mcf_cb([10, 30, 60], "lower", alpha_ci=0.05)
    )
    np.testing.assert_array_equal(
        b.upper, model.mcf_cb([10, 30, 60], "upper", alpha_ci=0.05)
    )
    assert b.alpha_ci == 0.1
    assert repr(b).startswith("ConfidenceBounds(lower=[")
    assert "level=90%" in repr(b)


def test_scalar_query(model):
    b = model.mcf_cb(30)
    assert b.shape == (2,)
    assert np.ndim(b.lower) == 0 and np.ndim(b.upper) == 0
    assert (b[0], b[1]) == (b.lower, b.upper)
    assert repr(b) == (
        f"ConfidenceBounds(lower={np.array2string(np.asarray(b.lower))}, "
        f"upper={np.array2string(np.asarray(b.upper))}, level=95%)"
    )


def test_one_sided_is_a_plain_array(model):
    one = model.mcf_cb([10, 30], bound="upper")
    assert type(one) is np.ndarray
    assert type(model.mcf_cb(30, bound="lower")) is np.float64


def test_computed_values_are_plain_arrays(model):
    b = model.mcf_cb([10, 30, 60])
    for derived in (b * 2, b.round(2), b[:, 0], b.mean(), np.asarray(b)):
        assert not isinstance(derived, ConfidenceBounds)
    # A subset of the queries is still a [lower, upper] pair.
    assert isinstance(b[1:], ConfidenceBounds)
    assert isinstance(b[0], ConfidenceBounds)
    assert b.tolist() == np.asarray(b).tolist()


def test_pickles(model):
    b = model.mcf_cb([10, 30])
    back = pickle.loads(pickle.dumps(b))
    assert isinstance(back, ConfidenceBounds)
    assert back.alpha_ci == 0.05
    np.testing.assert_array_equal(back, b)


def test_positional_bound_and_cause_specific():
    m = NonParametricCounting.fit(X, i=I, c=C)
    assert isinstance(m.mcf_cb([10], "two-sided"), ConfidenceBounds)
    cs = CauseSpecificMCF.fit(
        [5, 10, 30, 8, 25, 40, 60],
        [1, 1, 1, 2, 2, 3, 3],
        [0, 0, 1, 0, 1, 0, 1],
        e=["a", "b", None, "a", None, "b", None],
    )
    assert isinstance(cs.mcf_cb([10], "a"), ConfidenceBounds)
    assert not isinstance(
        cs.mcf_cb([10], "a", bound="lower"), ConfidenceBounds
    )


def test_wrap_refuses_a_non_pair():
    with pytest.raises(ValueError, match=r"\[lower, upper\]"):
        ConfidenceBounds.wrap([1.0, 2.0, 3.0])
