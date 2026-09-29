"""SurPyval's Lin-Ying additive hazards model against stored R ``timereg``
results (#379).

``timereg::aalen`` with every covariate in ``const()`` is the
semi-parametric additive model with only the baseline time-varying, whose
estimator is Lin and Ying's; its ``var.gamma`` is the Lin-Ying sandwich.
Both are closed-form (no iteration), so they must agree to rounding
(``rtol=1e-9``).
"""

import numpy as np
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, values

EXACT = dict(rtol=1e-9, atol=1e-12)


def test_lin_ying_matches_timereg_aalen_const():
    d = fixture("additive")
    Z = np.column_stack([d["z1"], d["z2"]])
    ref = values("r_timereg", "aalen_additive")
    model = sp.AdditiveHazards.fit(d["x"], Z, c=d["c"])
    assert_allclose(model.beta, ref["gamma"], **EXACT)
    assert_allclose(model.covariance(), ref["var_gamma"], **EXACT)
    # The cumulative baseline B0(t) (timereg's cum): -log S(t | Z = 0).
    t = ref["cum_time"][1:]
    H0 = -np.log(model.sf(t, np.zeros((t.size, 2))))
    assert_allclose(H0, ref["cum_baseline"][1:], **EXACT)
