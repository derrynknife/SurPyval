"""``ParametricPanelMultiStateRegression`` against R ``msm`` and ``nhm``.

The values are in ``data/r_multistate_panel_regression.json``, written by
``scripts/reference/reference_r_multistate_panel_regression.R`` (R 4.6.2,
msm 1.9, nhm 0.1.2) on the fixture ``multistate_inspected_covariates``:
300 units of a progressive illness-death model with Weibull proportional
hazards on the time since the start, seen at visits, each death an
exactly timed entry from a state not seen; ``male`` is fixed for a unit,
``w`` is drawn at each visit and holds until the next.

What is compared:

- exponential proportional hazards with ``male`` and ``w`` against msm
  (``covariates = ~ male + w``, the covariates of a visit holding over
  the gap that starts at it), on both clocks (the model is progressive,
  so a constant hazard does not care which): the log-likelihood, the
  baseline intensities, the coefficients and their standard errors;
- Weibull proportional hazards with ``male`` on the forward clock against
  nhm, which solves the same forward equations with lsoda (tolerances
  1e-11): the likelihood itself at nhm's estimates (a check of the
  likelihood that does not depend on either search), the minimised
  negative log-likelihood, the scales, shapes and coefficients, and the
  standard errors of nhm's parameters.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval import ExponentialPH, WeibullPH
from surpyval.multistate import (
    ParametricPanelMultiStateRegression,
    TransitionMatrix,
)
from surpyval.multistate.parametric import _fitters

from ._data import fixture, values

SOURCE = "r_multistate_panel_regression"
TM = TransitionMatrix([(1, 2), (1, 3), (2, 3)], states=[1, 2, 3])


@pytest.fixture(scope="module")
def data():
    d = fixture("multistate_inspected_covariates")
    return dict(
        x=d["time"],
        s=d["state"].astype(int),
        i=d["id"].astype(int),
        transitions=TM,
        exact_entry=3,
    ), np.column_stack([d["male"], d["w"]]).astype(float)


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_exponential_ph_is_msm(data, clock):
    kw, Z = data
    ref = values(SOURCE, "exponential")
    model = ParametricPanelMultiStateRegression.fit(
        Z=Z, **kw, model=ExponentialPH, clock=clock
    )
    assert model.maximum == "verified"
    assert_allclose(model.log_likelihood, ref["loglik"], rtol=1e-9)
    rates = [float(m.params[0]) for m in model._models()]
    assert_allclose(rates, ref["q"], rtol=1e-5)
    assert_allclose(model.betas[:, 0], ref["b_male"], rtol=1e-4, atol=1e-6)
    assert_allclose(model.betas[:, 1], ref["b_w"], rtol=1e-4, atol=1e-6)
    se = model.standard_errors().reshape(3, 3)
    assert_allclose(se[:, 0], ref["q_se"], rtol=1e-3)
    assert_allclose(se[:, 1], ref["b_male_se"], rtol=1e-3)
    assert_allclose(se[:, 2], ref["b_w_se"], rtol=1e-3)


def _nhm_models(ref):
    """WeibullPH models at nhm's estimates."""
    import pandas as pd

    base = WeibullPH.fit(
        x=[1.0, 2, 3, 4], Z=pd.DataFrame({"z0": [0.0, 1, 0, 1]})
    )
    return [
        base.with_params([a, b, c])
        for a, b, c in zip(ref["scale"], ref["shape"], ref["b_male"])
    ]


def test_weibull_ph_likelihood_at_nhms_estimates(data):
    from surpyval.multistate.panel import _check_panel

    kw, Z = data
    ref = values(SOURCE, "weibull")
    pdata = _check_panel(
        kw["x"], kw["s"], kw["i"], Z[:, :1], None, TM, None, 3, None
    )
    lik = ParametricPanelMultiStateRegression._likelihood(
        pdata, _fitters(WeibullPH, TM, True), "forward"
    )
    models = _nhm_models(ref)
    lik.plan(models)
    assert_allclose(lik.neg_ll(models), ref["value"], rtol=1e-9)


def test_weibull_ph_is_nhm(data):
    kw, Z = data
    ref = values(SOURCE, "weibull")
    model = ParametricPanelMultiStateRegression.fit(Z=Z[:, :1], **kw)
    assert model.maximum == "verified"
    assert_allclose(model.neg_ll(), ref["value"], rtol=1e-8)
    params = model.params.reshape(-1, 3)
    assert_allclose(params[:, 0], ref["scale"], rtol=1e-4)
    assert_allclose(params[:, 1], ref["shape"], rtol=1e-4)
    assert_allclose(params[:, 2], ref["b_male"], rtol=1e-3, atol=1e-5)
    # nhm's parameters: log(1 / scale), log(shape), the coefficient
    se = model.standard_errors().reshape(-1, 3)
    par_se = np.asarray(ref["par_se"])
    assert_allclose(se[:, 0] / params[:, 0], par_se[:3], rtol=1e-3)
    assert_allclose(se[:, 1] / params[:, 1], par_se[3:6], rtol=1e-3)
    assert_allclose(se[:, 2], par_se[6:], rtol=1e-3)
