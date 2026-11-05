"""Checks that kill mutants of the sampling weights' shared code
(``utils/weights.py``) that the suite let survive (#396). The rest of
that module's survivors change only the words of a refusal."""

import warnings

import numpy as np
from numpy.testing import assert_allclose

import surpyval as sp


def _left_censored():
    rng = np.random.default_rng(306)
    x = np.round(rng.weibull(2.0, 40) * 10, 1) + 0.1
    c = np.zeros(40, dtype=int)
    c[::5] = -1
    c[1::7] = 1
    w = np.round(rng.uniform(0.3, 3.0, 40), 2)
    return x, c, w


def test_column_of_weights_is_one_weight_per_row():
    """Kills, in ``check_weights``, the flattening of a column ``(k, 1)``
    of weights edited (``ndim == 3``, ``arr[:, 1]``, ``None``): a column
    is the same weights as a flat array."""
    x, c, w = _left_censored()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat = sp.Weibull.fit(x, c, w=w)
        column = sp.Weibull.fit(x, c, w=w[:, None])
    assert_allclose(column.params, flat.params, rtol=1e-12)
    assert_allclose(column.covariance(), flat.covariance(), rtol=1e-12)


def test_robust_covariance_with_left_censored_rows():
    """Kills, in ``with_counts``, the left-censored rows' counts set to
    ``None``: the rows' scores, of which the robust covariance is built,
    then lose (or fail on) the left-censored rows. No weighted test had
    left-censored rows. The covariance must be finite, positive and not
    change with the scale of the weights."""
    x, c, w = _left_censored()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = sp.Weibull.fit(x, c, w=w).covariance()
        scaled = sp.Weibull.fit(x, c, w=10 * w).covariance()
    assert np.all(np.isfinite(cov)) and np.all(np.diag(cov) > 0)
    assert_allclose(scaled, cov, rtol=1e-6)
