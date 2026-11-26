"""``ParametricPanelMultiState`` against R ``msm`` and ``nhm``.

The values are in ``data/r_multistate_panel_parametric.json``, written by
``scripts/reference/reference_r_multistate_panel_parametric.R`` (R 4.6.2,
msm 1.9, nhm 0.1.2) on the fixture ``multistate_inspected``: 300 units of
a progressive illness-death model with Weibull hazards on the time since
the start, seen at visits, each death an exactly timed entry from a
state not seen.

What is compared:

- exponential hazards against msm (``deathexact``): the log-likelihood to
  1e-9 relative, the intensities and their standard errors to 1e-5 and
  1e-4 (msm's optimiser at ``reltol = 1e-14``; SurPyval's search on
  numerical derivatives); on both clocks (the model is progressive, so a
  constant hazard does not care which);
- Weibull hazards on the forward clock against nhm, which solves the
  same forward equations with lsoda (tolerances 1e-11): the minimised
  negative log-likelihood to 1e-8, the scales and shapes to 1e-4, and
  the standard errors of their logs (nhm's parameters) to 1e-3.
"""

import pytest
from numpy.testing import assert_allclose

from surpyval import Exponential, Weibull
from surpyval.multistate import ParametricPanelMultiState, TransitionMatrix

from ._data import fixture, values

SOURCE = "r_multistate_panel_parametric"
TM = TransitionMatrix([(1, 2), (1, 3), (2, 3)], states=[1, 2, 3])


@pytest.fixture(scope="module")
def data():
    d = fixture("multistate_inspected")
    return dict(
        x=d["time"],
        s=d["state"].astype(int),
        i=d["id"].astype(int),
        transitions=TM,
        exact_entry=3,
    )


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_exponential_is_msm(data, clock):
    ref = values(SOURCE, "exponential")
    model = ParametricPanelMultiState.fit(
        **data, dist=Exponential, clock=clock
    )
    assert model.maximum == "verified"
    assert_allclose(model.log_likelihood, ref["loglik"], rtol=1e-9)
    assert_allclose(model.params, ref["q"], rtol=1e-5)
    assert_allclose(model.standard_errors(), ref["q_se"], rtol=1e-4)


def test_weibull_is_nhm(data):
    ref = values(SOURCE, "weibull")
    model = ParametricPanelMultiState.fit(**data, dist=Weibull)
    assert model.maximum == "verified"
    assert_allclose(model.neg_ll(), ref["value"], rtol=1e-8)
    params = model.params.reshape(-1, 2)
    assert_allclose(params[:, 0], ref["scale"], rtol=1e-4)
    assert_allclose(params[:, 1], ref["shape"], rtol=1e-4)
    se = (model.standard_errors() / model.params).reshape(-1, 2)
    assert_allclose(se[:, 0], ref["par_se"][:3], rtol=1e-3)
    assert_allclose(se[:, 1], ref["par_se"][3:], rtol=1e-3)
