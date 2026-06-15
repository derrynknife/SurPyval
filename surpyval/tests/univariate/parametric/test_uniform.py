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
