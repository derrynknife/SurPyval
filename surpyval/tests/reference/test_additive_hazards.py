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
    # The cumulative baseline B0(t) (timereg's cum), the estimate at
    # Z = 0 at the event times: the fitted H0 on its grid.
    t = ref["cum_time"][1:]
    at = np.searchsorted(model.x, t)
    assert_allclose(model.x[at], t, rtol=0, atol=0)
    assert_allclose(model.H0[at], ref["cum_baseline"][1:], **EXACT)
    # B0 is negative early on (the drift of a positive beta'Zbar), where
    # the model predicts with its running maximum from 0 (#376): -log S is
    # max(0, max_{s <= t} B0(s)) there.
    assert np.min(ref["cum_baseline"]) < 0
    H = -np.log(model.sf(t, np.zeros((t.size, 2))))
    envelope = np.maximum(np.maximum.accumulate(model.H0[at]), 0.0)
    assert_allclose(H, envelope, **EXACT)
