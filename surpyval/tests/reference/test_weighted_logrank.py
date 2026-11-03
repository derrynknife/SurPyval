"""The log-rank tests of data with sampling weights ``w`` against R.

The data are ``_helpers.weighted_reference_data()``. The values are R's
robust score test of the group indicators at ``beta = 0``,
``coxph(..., weights = w, robust = TRUE, ties = "breslow")$rscore``
(R 4.6.2, ``survival`` 3.8.12), printed by
``scripts/reference/reference_r_weighted_logrank.R``. They are closed
forms, so they agree to rounding.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.tests._helpers import weighted_reference_data

R_RSCORE = {
    "two_groups": 5.46611305064827,
    "unit_weights": 6.69882512785284,
    "three_groups_tl": 11.8627823946261,
    "strata_tl": 2.37894639219675,
    "t0_5": 8.88248769668906,
}


@pytest.fixture(scope="module")
def data():
    d = weighted_reference_data()
    z1, z2 = d["Z"][:, 0], d["Z"][:, 1]
    d["g2"] = z1.astype(int)
    d["g3"] = np.digitize(z2, [-0.5, 0.5])
    d["s"] = (z2 > 0).astype(int)
    return d


def _test(data, case):
    x, c, w = data["x"], data["c"], data["w"]
    if case == "two_groups":
        return sp.logrank(x, data["g2"], c=c, w=w)
    if case == "unit_weights":
        return sp.logrank(x, data["g2"], c=c, w=np.ones(x.size))
    if case == "three_groups_tl":
        return sp.logrank(x, data["g3"], c=c, w=w, tl=data["tl"])
    if case == "strata_tl":
        return sp.logrank(
            x, data["g2"], c=c, w=w, tl=data["tl"], strata=data["s"]
        )
    return sp.logrank(x, data["g3"], c=c, w=w, t0=5.0)


@pytest.mark.parametrize("case", sorted(R_RSCORE))
def test_matches_the_robust_score_test(data, case):
    res = _test(data, case)
    assert res.weighted
    assert_allclose(res.statistic, R_RSCORE[case], rtol=1e-10)
    assert res.dof == (2 if "three" in case or case == "t0_5" else 1)
