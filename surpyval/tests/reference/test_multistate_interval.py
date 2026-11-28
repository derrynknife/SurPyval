"""The parametric multi-state model with interval-censored moves
(``ParametricMultiState`` with ``x`` as ``[xl, xr]``) against R.

The values are in ``data/r_multistate_interval.json``, written by
``scripts/reference/reference_r_multistate_interval.R`` (R 4.6.2, msm 1.9,
flexsurv 2.3.2), on the fixture ``multistate_inspected``: 300 units of a
progressive illness-death model seen at visits, deaths timed exactly.

What is compared:

- with constant hazards (``Exponential``), R ``msm`` with each death an
  exactly timed move from the state seen before it (``obstype = 2``):
  with no way back to healthy, a unit seen healthy and then ill moved
  once, between the two visits, so msm's likelihood is that of the long
  form with the onset of illness interval censored. The log-likelihoods
  agree to 1e-10 relative, the intensities to 1e-6 relative and their
  standard errors to 1e-4 (both from numerical Hessians);
- a two-state model (the time to the onset of illness alone, a Weibull)
  is R ``flexsurvreg``'s univariate interval-censored fit: the
  log-likelihood to 1e-9 relative, the parameters to 1e-5.
"""

import numpy as np
from numpy.testing import assert_allclose

from surpyval import Exponential, Weibull
from surpyval.multistate import ParametricMultiState, TransitionMatrix
from surpyval.tests.multistate._inspected import TRANSITIONS, spells

from ._data import fixture, reference

SOURCE = "r_multistate_interval"


def _values(ref_id):
    return reference(SOURCE, ref_id)["values"]


def _data():
    d = fixture("multistate_inspected")
    return spells(d["id"], d["time"], d["state"].astype(int))


def test_exponential_is_msms_likelihood():
    x, e, state, tl, i = _data()
    model = ParametricMultiState.fit(
        x,
        e,
        state,
        tl=tl,
        i=i,
        transitions=TransitionMatrix(TRANSITIONS),
        dist=Exponential,
    )
    v = _values("exponential")
    assert_allclose(-model.neg_ll(), v["loglik"], rtol=1e-10)
    assert_allclose(model.params, v["q"], rtol=1e-6)
    assert_allclose(model.standard_errors(), v["q_se"], rtol=1e-4)
    assert model.maximum == "verified"


def test_two_state_is_flexsurvs_interval_censored_fit():
    d = fixture("multistate_inspected")
    # the time to the onset of illness: (last healthy visit, first ill
    # visit] for a unit seen ill, else right censored at its last healthy
    # visit (a unit seen healthy only at 0 says nothing about it)
    X, E = [], []
    for u in np.unique(d["id"]):
        sel = d["id"] == u
        t, s = d["time"][sel], d["state"][sel]
        healthy, ill = t[s == 1], t[s == 2]
        if ill.size:
            X.append([healthy[healthy < ill.min()].max(), ill.min()])
            E.append("ill")
        elif healthy.max() > 0:
            X.append([healthy.max()] * 2)
            E.append(None)
    model = ParametricMultiState.fit(
        np.array(X),
        E,
        ["healthy"] * len(E),
        transitions=[("healthy", "ill")],
        dist=Weibull,
    )
    v = _values("two_state")
    assert sum(e is not None for e in E) == v["n_interval"]
    assert_allclose(-model.neg_ll(), v["loglik"], rtol=1e-9)
    assert_allclose(model.params, [v["scale"], v["shape"]], rtol=1e-5)
