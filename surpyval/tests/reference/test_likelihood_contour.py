"""The joint likelihood-ratio contour of a Weibull fit
(``Parametric.likelihood_contour``) against R.

``data/weibullr_contour.json`` is written by
``scripts/reference/reference_r_likelihood_contour.R`` (R 4.6.2,
WeibullR 1.2.4) on 20 right-censored Weibull lifetimes, at the 90% and
95% joint regions (chi-squared with two degrees of freedom). It holds two
references:

* the exact extremes of each parameter over the region, from the profile
  likelihood solved in base R (``optimize`` and ``uniroot``), which
  SurPyval's ``bounds`` and the contour's extreme points must match to
  1e-6 relative;
* the points of ``WeibullR::MLEcontour(..., dof = 2)``, which must lie on
  SurPyval's contour. WeibullR finds them to its own radial tolerance
  (``RadLimit = 1e-5``) about its own estimate, whose log-likelihood is
  5e-5 below the maximum, so its points are on the contour to about 4e-4
  in log-likelihood; the test holds them to 1e-3, and checks SurPyval's
  own points to 1e-6.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import surpyval as sp

DATA = Path(__file__).parent / "data" / "weibullr_contour.json"


@pytest.fixture(scope="module")
def reference():
    with open(DATA) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model(reference):
    data = reference["data"]
    return sp.Weibull.fit(data["x"], c=data["c"])


def _loglik(model, points):
    return np.array([-model._lr_neg_ll(np.asarray(p)) for p in points])


LEVELS = ["0.1", "0.05"]


def test_estimate_matches_r(reference, model):
    mle = reference["mle"]
    np.testing.assert_allclose(
        model.params, [mle["alpha"], mle["beta"]], rtol=1e-6
    )
    assert -model._lr_neg_ll(model.params) == pytest.approx(
        mle["log_likelihood"], abs=1e-8
    )


@pytest.mark.parametrize("level", LEVELS)
def test_cut_and_extremes_match_r(reference, model, level):
    ref = reference["levels"][level]
    contour = model.likelihood_contour(alpha_ci=ref["alpha_ci"])
    assert contour.critical_value == pytest.approx(
        ref["critical_value"], rel=1e-12
    )
    assert contour.log_likelihood_cut == pytest.approx(
        ref["log_likelihood_cut"], abs=1e-8
    )
    expected = np.array([ref["bounds"]["alpha"], ref["bounds"]["beta"]])
    # The exact extremes (profile likelihood, base R): 1e-6 relative.
    np.testing.assert_allclose(contour.bounds, expected, rtol=1e-6)
    # They are points of the contour, so its extremes too.
    np.testing.assert_allclose(contour.points.min(axis=0), expected[:, 0])
    np.testing.assert_allclose(contour.points.max(axis=0), expected[:, 1])


@pytest.mark.parametrize("level", LEVELS)
def test_points_lie_on_the_cut(reference, model, level):
    ref = reference["levels"][level]
    contour = model.likelihood_contour(alpha_ci=ref["alpha_ci"])
    assert contour.closed
    ll = _loglik(model, contour.points)
    np.testing.assert_allclose(ll, ref["log_likelihood_cut"], atol=1e-6)


@pytest.mark.parametrize("level", LEVELS)
def test_weibullr_points_lie_on_surpyval_contour(reference, model, level):
    ref = reference["levels"][level]
    r_points = np.column_stack([ref["points"]["alpha"], ref["points"]["beta"]])
    # WeibullR's points, to its radial tolerance (see the module note).
    ll = _loglik(model, r_points)
    np.testing.assert_allclose(ll, ref["log_likelihood_cut"], atol=1e-3)
    # Its 121 points miss the tangent points a little: the ranges agree
    # to 2e-3 relative, and never reach beyond SurPyval's exact extremes
    # by more than that.
    contour = model.likelihood_contour(alpha_ci=ref["alpha_ci"])
    np.testing.assert_allclose(
        r_points.min(axis=0), contour.bounds[:, 0], rtol=2e-3
    )
    np.testing.assert_allclose(
        r_points.max(axis=0), contour.bounds[:, 1], rtol=2e-3
    )


def test_reference_file_records_its_source(reference):
    assert reference["generator"] == (
        "scripts/reference/reference_r_likelihood_contour.R"
    )
    assert "WeibullR" in reference["version"]
    for level in LEVELS:
        assert "MLEcontour" in reference["levels"][level]["call"]
