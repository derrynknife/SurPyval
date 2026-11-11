"""Parametric competing-risks regression against two other routes to the
same cumulative incidence.

- The multi-state route: the competing-risks model is a multi-state model
  with one "event-free" state and one absorbing state per cause. Its
  Aalen-Johansen product integral, built from the fitted causes'
  cumulative-hazard increments on a fine time grid (the step the
  cause-specific Cox model takes on its event times), converges to the
  model's integral ``int h_k S du`` as the grid is refined. The step is
  exact for hazards proportional within each step, so the error is
  second order in the step.
- The semi-parametric route: on data simulated from Weibull proportional
  hazards per cause, the cause-specific Cox model estimates the same
  coefficients and the same incidences without assuming the baseline
  shape. Over the range of the data the two agree to sampling error.
"""

import warnings

import numpy as np
import pytest

from surpyval import ParametricCompetingRisksRegression
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
)
from surpyval.univariate.competing_risks._assembly import (
    aalen_johansen_steps,
)


def _weibull_ph_causes(N, seed, censor):
    """Two causes with Weibull proportional hazards in two covariates:
    cause 1 ``alpha = 10, beta = 2``, coefficients ``(0.6, -0.3)``; cause
    2 ``alpha = 15, beta = 1.2``, coefficients ``(-0.4, 0.5)``; uniform
    censoring on ``(0, censor)``."""
    rng = np.random.default_rng(seed)
    Z = np.column_stack([rng.binomial(1, 0.5, N), rng.normal(0, 1, N)])
    truth = {
        1: (10.0, 2.0, np.array([0.6, -0.3])),
        2: (15.0, 1.2, np.array([-0.4, 0.5])),
    }
    latent = np.column_stack(
        [
            alpha * (rng.exponential(size=N) / np.exp(Z @ coef)) ** (1 / beta)
            for alpha, beta, coef in truth.values()
        ]
    )
    x = latent.min(axis=1)
    e = np.array([1, 2], dtype=object)[latent.argmin(axis=1)]
    cens = rng.uniform(0, censor, N)
    e[cens < x] = None
    return np.minimum(x, cens), Z, e


# -- the multi-state (Aalen-Johansen) route ----------------------------------


@pytest.mark.parametrize("row", [[0.0, 0.0], [1.0, 1.5]])
def test_cif_is_the_multi_state_product_integral(row):
    x, Z, e = _weibull_ph_causes(400, 3, 30.0)
    model = ParametricCompetingRisksRegression.fit(x, Z, e)
    # A geometric grid of 20,000 steps from 1e-6 to 60, twice the end of
    # follow-up: the product integral's error is second order in the
    # step, about 2e-8 here; the tolerance is 1e-6.
    grid = np.concatenate([[0.0], np.geomspace(1e-6, 60.0, 20_000)])
    increments = np.array(
        [np.diff(model.models[k].Hf(grid, row)) for k in model.causes]
    )
    product_integral = aalen_johansen_steps(increments).cumsum(axis=1)
    at = np.searchsorted(grid, [1.0, 5.0, 10.0, 20.0, 40.0, 60.0])
    for i, k in enumerate(model.causes):
        np.testing.assert_allclose(
            product_integral[i, at - 1],
            model.cif(grid[at], row, k),
            rtol=0,
            atol=1e-6,
        )
    # Every step's increments sum to the step in the all-cause incidence.
    np.testing.assert_allclose(
        product_integral.sum(axis=0)[at - 1],
        model.ff(grid[at], row),
        rtol=1e-12,
    )


def test_cox_incidence_is_the_shared_multi_state_step():
    # The cause-specific Cox model's incidence is the same Aalen-Johansen
    # step, on its Breslow increments.
    x, Z, e = _weibull_ph_causes(200, 4, 30.0)
    cox = CompetingRisksProportionalHazards.fit(x, Z, e)
    row = np.array([1.0, 0.5])
    steps = cox._incidence_steps(row)
    np.testing.assert_allclose(
        steps[0].cumsum(), cox.cif(cox.x, row, 1), rtol=1e-12
    )
    np.testing.assert_allclose(
        steps.sum(axis=0).cumsum(),
        -np.expm1(-np.asarray(cox.Hf(cox.x, row), dtype=float)),
        rtol=1e-12,
    )


# -- the semi-parametric (cause-specific Cox) route --------------------------


@pytest.fixture(scope="module")
def both():
    x, Z, e = _weibull_ph_causes(2000, 11, 30.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parametric = ParametricCompetingRisksRegression.fit(x, Z, e)
        cox = CompetingRisksProportionalHazards.fit(x, Z, e, model="Cox")
    return parametric, cox, x, e


def test_cox_and_parametric_coefficients_agree(both):
    # Both models are proportional hazards per cause: the same log hazard
    # ratios, estimated with and without the Weibull baseline. They agree
    # to within half a standard error (seen: at most 0.35 of one).
    parametric, cox, _, _ = both
    for k in parametric.causes:
        own = parametric.models[k]
        coef = np.ravel(own.params)[-2:]
        se = np.sqrt(np.diag(own.covariance()))[-2:]
        np.testing.assert_array_less(
            np.abs(coef - cox.betas[cox.event_idx_map[k]]), 0.5 * se
        )


# slow: the band's gradient takes 17 incidence integrals over 48 points,
# each inverting the regressions' quantile functions numerically.
@pytest.mark.slow
def test_cox_incidence_within_the_parametric_bands(both):
    # Over the data (to the 90th percentile of the event times) the
    # cause-specific Cox model's step incidence lies inside the parametric
    # model's pointwise 95% Wald band, and within 0.03 of the parametric
    # curve (seen: at most 0.015), at four covariate rows. The rows and
    # times are paired, so each cause's band is one call.
    parametric, cox, x, e = both
    times = np.linspace(0.5, np.quantile(x[e != None], 0.9), 12)  # noqa: E711
    rows = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    t = np.tile(times, len(rows))
    Z = np.repeat(rows, times.size, axis=0)
    for k in parametric.causes:
        smooth = parametric.cif(t, Z, k)
        steps = np.concatenate([cox.cif(times, row, k) for row in rows])
        band = parametric.cb(t, Z, k)
        inside = (band[:, 0] <= steps) & (steps <= band[:, 1])
        assert inside.all(), (k, t[~inside], band[~inside], steps[~inside])
        np.testing.assert_allclose(steps, smooth, rtol=0, atol=0.03)
