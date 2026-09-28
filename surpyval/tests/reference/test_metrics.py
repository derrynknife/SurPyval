"""SurPyval's prediction metrics against stored results (#379): the Brier
score from R ``pec`` and ``riskRegression::Score``, Uno's time-dependent
AUC from ``riskRegression::Score``, and all three metrics from
scikit-survival.

Each fixture carries a fixed matrix of predicted survival, so only the
metric is compared, not a model. The metrics are closed-form weighted
sums, so where the conventions agree the values must agree to rounding
(``rtol=1e-9``).

The convention that differs (#365): with an event and a censoring at the
same time, SurPyval (like pec and riskRegression) weights the event by
``1 / G(x_i-)``, the censoring survival just before it; scikit-survival
uses ``1 / G(x_i)``, which also discounts the censorings at ``x_i``. On
the tied fixture SurPyval must therefore agree with R and differ from
scikit-survival; on the untied one it agrees with all three.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, fixture_extra, values

EXACT = dict(rtol=1e-9, atol=1e-12)


def _scores(name):
    d = fixture("prediction_" + name)
    S = np.array(fixture_extra("prediction_" + name, "survival"))
    times = np.array(fixture_extra("prediction_" + name, "times"))
    _, bs = sp.brier_score(d["x"], d["c"], S, times)
    _, auc = sp.auc_td(d["x"], d["c"], 1 - S, times)
    ibs = sp.integrated_brier_score(d["x"], d["c"], S, times)
    return bs, auc, ibs


@pytest.mark.parametrize("name", ["ties", "continuous"])
def test_brier_score_matches_pec(name):
    bs, _, _ = _scores(name)
    ref = values("r_pec", "brier_" + name)
    assert_allclose(bs, ref["brier"], **EXACT)


@pytest.mark.parametrize("name", ["ties", "continuous"])
def test_brier_and_auc_match_riskregression_score(name):
    bs, auc, _ = _scores(name)
    ref = values("r_riskregression", "score_" + name)
    assert_allclose(bs, ref["brier"], **EXACT)
    assert_allclose(auc, ref["auc"], **EXACT)


def test_metrics_match_scikit_survival_without_ties():
    bs, auc, ibs = _scores("continuous")
    ref = values("py_sksurv", "metrics_continuous")
    assert_allclose(bs, ref["brier"], **EXACT)
    assert_allclose(auc, ref["auc"], **EXACT)
    assert_allclose(ibs, ref["integrated_brier"], **EXACT)


def test_metrics_follow_pec_not_scikit_survival_at_ties():
    # The documented convention (#365): at event-censoring ties the IPCW
    # weight of an event is 1 / G(x_i-), as pec; scikit-survival's
    # 1 / G(x_i) over-weights the event. The Brier scores differ by up to
    # 0.013 here and the AUCs by 9e-4.
    bs, auc, _ = _scores("ties")
    ref = values("py_sksurv", "metrics_ties")
    assert np.max(np.abs(bs - ref["brier"])) > 1e-3
    assert np.max(np.abs(auc - ref["auc"])) > 1e-4
