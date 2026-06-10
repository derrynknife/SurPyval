"""
Tests Standby Nodes.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import numpy as np
import pytest

from surpyval import Uniform


def test_impermitted_truncation():
    x = Uniform.random(100, 0, 10)
    x = np.sort(x)
    tr = np.ones_like(x) * np.inf
    tr[-1] = 100
    with pytest.raises(ValueError):
        Uniform.fit(x, tr=tr)

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
    with pytest.raises(ValueError):
        Uniform.fit(x, tl=tl)


def test_impermitted_censoring():
    x = Uniform.random(100, 0, 10)
    x = np.sort(x)
    c = np.zeros_like(x)
    c[-1] = 1
    with pytest.raises(ValueError):
        Uniform.fit(x, c)

    c = np.zeros_like(x)
    c[-2] = 1
    assert (
        pytest.approx(np.array([x.min(), x.max()]))
        == Uniform.fit(x, c=c).params
    )

    x = Uniform.random(100, 0, 10)
    x = np.sort(x)
    c = np.zeros_like(x)
    c[0] = -1
    with pytest.raises(ValueError):
        Uniform.fit(x, c=c)
