"""The rectangle probabilities of three coordinates (the quadrature of
``copula/_rectangle.py``) against R's mvtnorm.

``data/r_mvtnorm_rectangle3.json`` is written by
``scripts/reference/reference_r_mvtnorm_rectangle3.R`` (R 4.6.2, mvtnorm
1.5.0): random correlation matrices of three coordinates, half of them
nearly singular (smallest eigenvalue 7e-4, 1e-3, 1e-2 or 0.05), with
limits mixing right-, left- and interval-censored coordinates; the normal
by Miwa's algorithm (confirmed by Genz-Bretz), the t (3 and 8 degrees of
freedom) by Genz-Bretz. Each case records mvtnorm's error estimate.

The tolerance is 1e-9 plus three times the reference's own error. Before
the quadrature split its panels where the other two coordinates'
standardised limits meet, it missed by up to 1e-5 (3.7e-4 of the
probability) for the nearly singular t cases, which this test catches.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from surpyval.multivariate.parametric.copula._rectangle import rectangle

DATA = Path(__file__).parent / "data" / "r_mvtnorm_rectangle3.json"
CASES = json.loads(DATA.read_text())["cases"]


def _limits(values, infinite):
    return np.array([infinite if v is None else v for v in values], float)


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda c: f"nu{c['nu']}-eig{c['smallest_eigenvalue']:.1g}",
)
def test_three_coordinates_agree_with_mvtnorm(case):
    lo = _limits(case["lo"], -np.inf)[None, :]
    hi = _limits(case["hi"], np.inf)[None, :]
    ours = rectangle(lo, hi, np.asarray(case["corr"]), case["nu"])[0]
    tolerance = 1e-9 + 3.0 * case["error"]
    assert abs(ours - case["value"]) < tolerance, (ours, case["value"])
