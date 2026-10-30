"""Checks that kill mutants of the Cox model that the suite let survive
(#396).

Each test names the mutant it kills: the function, then the edit. The
data are small and the tests fast: they run in the normal suite.
"""

import warnings

import numpy as np
import pytest

from surpyval import CoxPH

# Untied event times, three of the entries exactly at an earlier event
# time: rows 1, 2 and 4 enter at 3, 5 and 8, when rows 0, 1 and 2 fail.
X = np.array([3.0, 5.0, 8.0, 9.0, 12.0])
C = np.array([0, 0, 0, 1, 0])
Z = np.array([[1.0], [0.0], [2.0], [0.5], [1.5]])
TL = np.array([-np.inf, 3.0, 5.0, -np.inf, 8.0])


@pytest.mark.parametrize("tie_method", ["efron", "breslow", "exact"])
def test_entry_at_an_event_time(tie_method):
    """Kills, in ``cox_likelihood.py::cox_at_risk_mask``, entry made
    inclusive (``tl <= tau``): a row entering at an event time would be
    at risk for it, against the (entry, exit] convention. Only the exact
    tie methods' risk sets use the mask, so that mutant gave beta 0.2574
    for ``tie_method="exact"`` alone.

    With no ties the three methods are the same partial likelihood. The
    risk sets at 3, 5, 8 and 12 are rows {0, 3}, {1, 3}, {2, 3} and {4};
    with ``b = exp(beta)`` it is ``b / (b + b**0.5) * 1 / (1 + b**0.5) *
    b**2 / (b**2 + b**0.5)``, largest at ``b = 4``, ``beta = log 4``. The
    baseline increments at Z = 0 are then ``1 / (4 + 2)``, ``1 / (1 +
    2)``, ``1 / (16 + 2)``, 0 at the censoring and ``1 / 4**1.5``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = CoxPH.fit(X, Z, C, tl=TL, tie_method=tie_method)
    assert m.maximum == "verified"
    assert m.params[0] == pytest.approx(np.log(4), rel=1e-10)
    np.testing.assert_allclose(m.x, X)
    np.testing.assert_allclose(
        m.h0, [1 / 6, 1 / 3, 1 / 18, 0.0, 1 / 8], rtol=1e-9, atol=1e-15
    )


def _weighted_cox():
    from surpyval.univariate.regression.proportional_hazards._weighted_cox import (  # noqa: E501
        WeightedCox,
    )

    rng = np.random.default_rng(5)
    Z = rng.normal(size=(40, 2))
    x = rng.weibull(1.5, 40) * np.exp(-0.5 * Z[:, 0])
    c = (rng.random(40) < 0.2).astype(int)
    w = rng.uniform(0.5, 2.0, 40)
    return WeightedCox(x, c, np.ones(40), w, np.full(40, -np.inf), Z, "efron")


def test_weighted_newton_says_when_it_stops_short():
    """Kills, in ``_weighted_cox.py::newton``, ``return beta, False,
    max_iter`` made ``True``: a search stopped by its iteration limit
    would be reported as converged, and the fit as a verified maximum,
    with no warning. One Newton step from 0 does not reach a step shorter
    than 1e-12 standard errors; the full search does."""
    from surpyval.univariate.regression.proportional_hazards._weighted_cox import (  # noqa: E501
        newton,
    )

    lik = _weighted_cox()
    beta, converged, iters = newton(lik, 2, 1e-12, max_iter=1)
    assert (converged, iters) == (False, 1)
    beta, converged, iters = newton(lik, 2, 1e-9)
    assert converged and 1 < iters < 100
