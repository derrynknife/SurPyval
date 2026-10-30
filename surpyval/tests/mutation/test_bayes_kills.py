"""Checks that kill mutants of the Bayesian likelihood that the suite let
survive (#396).

``log_likelihood_fn`` builds its data from the arguments it is given; a
dropped argument (``n``, ``xl`` or ``t`` passed on as nothing) left every
test passing, since the PyMC tests that use it are skipped without PyMC
and ``fit_bayes`` builds its data another way. Each form is checked
against the same data given another way.
"""

import numpy as np
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.bayes import log_likelihood_fn

POINTS = np.array([[4.0, 2.0], [5.0, 1.5]])
X = [2.1, 3.4, 4.05, 5.0]
C = [0, 0, 0, 1]


def test_counts_are_repeated_rows():
    """Kills ``n=n`` -> ``n=None`` in ``log_likelihood_fn``."""
    counted = log_likelihood_fn(sp.Weibull, X, c=C, n=[1, 2, 1, 3])
    repeated = log_likelihood_fn(
        sp.Weibull,
        [2.1, 3.4, 3.4, 4.05, 5.0, 5.0, 5.0],
        c=[0, 0, 0, 0, 1, 1, 1],
    )
    assert_allclose(counted(POINTS), repeated(POINTS), rtol=1e-12)


def test_xl_xr_are_interval_rows():
    """Kills ``xl=xl`` -> ``xl=None`` in ``log_likelihood_fn``."""
    xl, xr = [1.0, 2.0, 3.0], [2.0, 4.0, 5.0]
    given = log_likelihood_fn(sp.Weibull, xl=xl, xr=xr)
    pairs = log_likelihood_fn(sp.Weibull, np.c_[xl, xr], c=[2, 2, 2])
    assert_allclose(given(POINTS), pairs(POINTS), rtol=1e-12)


def test_t_is_the_truncation_windows():
    """Kills ``t=t`` dropped in ``log_likelihood_fn``: the windows given
    as ``t`` are those given as ``tl`` and ``tr``, and they change the
    likelihood."""
    t = np.c_[[0.5, 1.0, -np.inf, 2.0], [np.inf, 10.0, 6.0, np.inf]]
    as_t = log_likelihood_fn(sp.Weibull, X, c=C, t=t)(POINTS)
    as_tl_tr = log_likelihood_fn(sp.Weibull, X, c=C, tl=t[:, 0], tr=t[:, 1])
    untruncated = log_likelihood_fn(sp.Weibull, X, c=C)(POINTS)
    assert_allclose(as_t, as_tl_tr(POINTS), rtol=1e-12)
    assert np.all(np.abs(as_t - untruncated) > 0.1)
