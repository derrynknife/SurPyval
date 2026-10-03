"""SurPyval's semi-parametric proportional odds model against stored R
``survival`` results (#341).

The proportional odds model is a Cox model in which each subject carries
its own unit-exponential (gamma, variance 1) frailty, and for a fixed
frailty variance ``coxph``'s penalised partial likelihood is maximised by
the coefficients of the marginal likelihood (Therneau, Grambsch and
Pankratz 2003). With ``ties = 'breslow'`` that is the likelihood of Murphy,
Rossini and van der Vaart (1997), which ``ProportionalOdds`` maximises, so
the two NPMLEs agree to the tolerance of R's iteration (``eps = 1e-12``).
R's coefficients are on the odds of failure, SurPyval's on the odds of
survival: they are each other's negatives.
"""

import numpy as np
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, values


def test_po_lung_matches_survival_gamma_frailty():
    d = fixture("lung")
    keep = ~np.isnan(d["ph_ecog"])
    Z = np.column_stack([d["age"], d["sex"], d["ph_ecog"]])[keep]
    model = sp.ProportionalOdds.fit(d["time"][keep], Z, c=d["c"][keep])
    ref = values("r_survival", "po_frailty_lung")
    assert_allclose(model.beta, -ref["coef"], rtol=1e-6)


def test_po_ties_matches_survival_gamma_frailty():
    d = fixture("ties")
    Z = np.column_stack([d["z1"], d["z2"]])
    model = sp.ProportionalOdds.fit(d["x"], Z, c=d["c"])
    ref = values("r_survival", "po_frailty_ties")
    assert_allclose(model.beta, -ref["coef"], rtol=1e-6)
