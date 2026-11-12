"""Landmark analysis (``landmark``, ``LandmarkCox``) against R ``dynpred``
and ``survival``.

The values are in ``data/r_landmark.json``, written by
``scripts/reference/reference_r_landmark.R`` (R 4.6.2, survival 3.8.12,
dynpred 0.1.2), on the fixture ``landmark``: 300 units with a fixed
covariate ``z1`` and a covariate ``z2`` that switches from 0 to 1 at
``v`` (one unit switches exactly at the landmark 2, one enters late).

What is compared:

- the landmark data sets of ``dynpred::cutLM`` at the landmarks 0 to 4
  with the window 3 (its wide format, whose time-varying covariate is the
  value at or before the landmark): the units kept, their times, status
  and covariates, exactly;
- a Cox model at the landmark 2 (``coxph`` on the landmark data): the
  coefficients and standard errors to 1e-8 relative, the window
  probabilities ``1 - S(5)`` from ``survfit`` to 1e-10;
- the stratified and pooled supermodels over the five landmarks, effects
  quadratic in ``u = L / 4``, by ``coxph(... + cluster(id))`` with
  Efron's and Breslow's ties: coefficients to 1e-8, robust and
  model-based standard errors to 1e-8 relative, and the window
  probabilities ``1 - S(L + 3) / S(L)`` from ``survfit(fit, newdata)`` at
  each landmark (and at 1.5, between landmarks, for the pooled model) to
  1e-9.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval import (
    ItemLifetime,
    LandmarkCox,
    StepSchedule,
    SurpyvalData,
    landmark,
)

from ._data import fixture, values

SOURCE = "r_landmark"
AT = [0, 1, 2, 3, 4]


def _life():
    d = fixture("landmark")
    items = []
    for u, e, t, st, z1, v in zip(
        d["id"], d["entry"], d["time"], d["status"], d["z1"], d["v"]
    ):
        if np.isnan(v):
            schedule = StepSchedule.from_changepoints([0], [0])
        else:
            schedule = StepSchedule.from_changepoints([0, v], [0, 1])
        outcome = {"observed": t} if st == 1 else {"right_censored": t}
        items.append(
            ItemLifetime(
                int(u),
                **outcome,
                left_truncation=e if e > 0 else None,
                covariates={"z1": z1},
                time_varying_covariates=schedule,
            )
        )
    return SurpyvalData.from_items(items)


LIFE = _life()


def _profiles(v):
    return np.column_stack([v["profiles"]["z1"], v["profiles"]["z2"]])


@pytest.mark.parametrize("L", AT)
def test_landmark_data_is_cutLM(L):
    v = values(SOURCE, f"cutLM_{L}")
    data = landmark(LIFE, at=L, horizon=3)
    assert_allclose(np.array(data.ids, dtype=float), v["id"])
    assert_allclose(data.x, v["time"])
    assert_allclose(1 - data.c, v["status"])
    assert_allclose(data.Z[:, 1], v["z2"])
    assert_allclose(data.tl, L)


def test_cox_at_one_landmark():
    v = values(SOURCE, "cox_landmark_2")
    model = LandmarkCox.fit(LIFE, at=2, horizon=3)
    assert model.parameter_names == ["z1", "time_varying_0"]
    assert_allclose(model.params, v["coef"], rtol=1e-8)
    # One row per unit: the robust and model-based errors differ, and R's
    # vcov here is the model-based one
    assert_allclose(model.standard_errors(robust=False), v["se"], rtol=1e-8)
    assert_allclose(
        model.predict(2, _profiles(v)), v["probability"], atol=1e-10
    )


@pytest.mark.parametrize("ties", ["efron", "breslow"])
@pytest.mark.parametrize("baseline", ["stratified", "pooled"])
def test_supermodel(baseline, ties):
    v = values(SOURCE, f"supermodel_{baseline}_{ties}")
    model = LandmarkCox.fit(
        LIFE, at=AT, horizon=3, baseline=baseline, tie_method=ties
    )
    assert model.n_rows == v["n_rows"]
    assert_allclose(model.params, v["coef"], rtol=1e-8, atol=1e-9)
    assert_allclose(model.standard_errors(), v["robust_se"], rtol=1e-8)
    assert_allclose(
        model.standard_errors(robust=False), v["naive_se"], rtol=1e-8
    )
    got = np.array([model.predict(s, _profiles(v)) for s in v["landmarks"]])
    assert_allclose(got, v["probability"], atol=1e-9)
