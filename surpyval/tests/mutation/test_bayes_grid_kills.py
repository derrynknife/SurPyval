"""Checks that kill mutants of the Bayesian grid engine
(``bayes/engines/grid.py``) that the suite let survive (#396).

The suite checked that ``method="laplace"`` is labelled approximate, not
what it approximates with. It is the normal approximation at the
posterior mode in the search space (here ``mu`` and ``log sigma``), so
its intervals are closed forms of the mode and the curvature there.
"""

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm

import surpyval as sp

X = np.array([4.1, 5.3, 6.0, 4.8, 5.5, 6.7, 5.1, 4.4, 5.9, 5.0])


def test_laplace_intervals_are_the_normal_at_the_mode():
    """Kills, in ``laplace_grid``, the normal's log density edited
    (``+0.5``, ``-1.5``, ``z * 2``, ``z**3``, ``/``), the lattice made
    one-sided (``lo = +6``) and, in ``hessian``, edits of its finite
    differences: the curvature at the mode of the normal likelihood,
    with the default priors, is ``n / sigma**2`` for ``mu`` and ``2 n``
    for ``log sigma``, with no cross term, so the 95% intervals are ``mu
    +- 1.96 sigma / sqrt(n)`` and ``sigma exp(+-1.96 / sqrt(2 n))``,
    ``mu`` and ``sigma`` the mode. The grid's quadrature holds them to
    about 1e-4."""
    model = sp.Normal.fit_bayes(X, method="laplace")
    mu, sigma = model.diagnostics["mode"]
    n, z = X.size, norm.ppf(0.975)
    assert_allclose(mu, X.mean(), rtol=1e-6)
    assert_allclose(
        np.ravel(model.param_cri("mu")),
        mu + np.array([-1, 1]) * z * sigma / np.sqrt(n),
        rtol=2e-4,
    )
    assert_allclose(
        np.ravel(model.param_cri("sigma")),
        sigma * np.exp(np.array([-1, 1]) * z / np.sqrt(2 * n)),
        rtol=2e-4,
    )
