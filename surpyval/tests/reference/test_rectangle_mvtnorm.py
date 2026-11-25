"""The rectangle probabilities of four or more coordinates
(``copula/_rectangle.py``) against R's mvtnorm.

``data/mvtnorm_rectangle_r.json`` is written by
``scripts/reference/reference_r_mvtnorm_rectangle.R`` (R 4.6.2, mvtnorm
1.5.0): random correlation matrices of four to eight coordinates with
limits mixing right-, left- and interval-censored coordinates, the normal
by Miwa's algorithm up to six coordinates (confirmed by Genz-Bretz to
1e-7) and Genz-Bretz at 1e-10 for eight, the t (3 and 8 degrees of
freedom) by Genz-Bretz at an absolute tolerance of 1e-6. Each case
records mvtnorm's error estimate.

The tolerance, fixed before the comparison was run, is the lattice rule's
stated accuracy (3e-6) plus three times the reference's own error.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from surpyval.multivariate.parametric.copula._rectangle import rectangle

DATA = Path(__file__).parent / "data" / "mvtnorm_rectangle_r.json"
CASES = json.loads(DATA.read_text())["cases"]


def _limits(values, infinite):
    return np.array([infinite if v is None else v for v in values], float)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"k{c['k']}-nu{c['nu']}")
def test_rectangle_agrees_with_mvtnorm(case):
    lo = _limits(case["lo"], -np.inf)[None, :]
    hi = _limits(case["hi"], np.inf)[None, :]
    ours = rectangle(lo, hi, np.asarray(case["corr"]), case["nu"])[0]
    tolerance = 3e-6 + 3.0 * case["error"]
    assert abs(ours - case["value"]) < tolerance, (ours, case["value"])
