"""Fits with sampling weights ``w`` against R ``survival`` and lifelines.

The data are ``_helpers.weighted_reference_data()`` (forty rows with
weights from 0.34 to 2.96, ties among the events, late entry on a third of
the rows). The values were computed once and are stored here:

- R 4.5.3 and ``survival`` 3.8.12:
  ``survfit(Surv(x, d) ~ 1, weights = w, robust = TRUE, stype, ctype)``
  (``stype = 1, ctype = 1`` Kaplan-Meier, ``2, 1`` Nelson-Aalen, ``2, 2``
  Fleming-Harrington) at ``times = c(1, 2.2, 3.7, 6.1, 8.3)``, with
  ``std.err / surv`` (the standard error of ``log S``) and, for
  Nelson-Aalen, ``std.chaz`` (of the cumulative hazard; R's ``std.err``
  for ``stype = 2, ctype = 1`` is not on that scale);
  ``survfit(Surv(tl, x, d) ~ 1, ..., id = id)`` for the late entries.
- lifelines 0.30.3 for the point estimate (``KaplanMeierFitter`` with
  weights).

R's ``print(survfit(...))`` gives ``se(rmean)`` from the classical
formula even with ``robust = TRUE``, so the robust variance of the
restricted mean is checked against an independent numerical
infinitesimal jackknife instead.

Tolerances: the estimates and their variances are closed forms, equal to
rounding (``rtol=1e-10``).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.tests._helpers import weighted_reference_data

TIMES = np.array([1, 2.2, 3.7, 6.1, 8.3])
CLOSED = dict(rtol=1e-10, atol=1e-12)

R_SURVFIT = {
    "KaplanMeier": (
        [0.923161189358372, 0.776525821596244, 0.538054872467689,
         0.441984942284838, 0.267143908812717],
        [0.0425276576222045, 0.0916447161673752, 0.1635459209094220,
         0.2038634103678010, 0.3471761338708973],
    ),
    "NelsonAalen": (
        [0.924324874876702, 0.780088308368704, 0.547374712686876,
         0.455424805354952, 0.286873256063502],
        # std.chaz
        [0.0411306096341895, 0.0880588734666975, 0.1537705223373366,
         0.1864578438579789, 0.2981738039526620],
    ),
    "FlemingHarrington": (
        [0.923904286727219, 0.779733351029604, 0.545463478547742,
         0.451090426213569, 0.284143019496056],
        [0.0416622759000732, 0.0883041754537210, 0.1557894238812997,
         0.1924831942107194, 0.3017495188139375],
    ),
}  # fmt: skip
R_SURVFIT_TRUNCATED = (
    [0.912915846230892, 0.767907853703625, 0.532083481064552,
     0.437079745399302, 0.264179116703005],
    [0.0483340686320875, 0.0957344234784129, 0.1650857868245073,
     0.2059623153393602, 0.3476031445164834],
)  # fmt: skip

LIFELINES_KM = [
    0.9231611893583727,
    0.7765258215962443,
    0.5380548724676895,
    0.44198494228483776,
    0.26714390881271705,
]


@pytest.fixture(scope="module")
def data():
    return weighted_reference_data()


def _log_sf_se(model, x):
    idx = np.searchsorted(model.x, x, side="right") - 1
    return np.sqrt(model.greenwood[idx])


@pytest.mark.parametrize("name", list(R_SURVFIT))
def test_survfit_robust(data, name):
    model = getattr(sp, name).fit(data["x"], data["c"], w=data["w"])
    surv, se = R_SURVFIT[name]
    assert_allclose(model.sf(TIMES), surv, **CLOSED)
    assert_allclose(_log_sf_se(model, TIMES), se, **CLOSED)
    # The bounds are the usual log(-log) ones on this variance
    z = 1.959963984540054
    s = np.asarray(surv)
    v = np.asarray(se) ** 2 / np.log(s) ** 2
    lower = np.exp(-np.exp(np.log(-np.log(s)) + z * np.sqrt(v)))
    upper = np.exp(-np.exp(np.log(-np.log(s)) - z * np.sqrt(v)))
    assert_allclose(model.cb(TIMES), np.column_stack([lower, upper]), **CLOSED)


def test_survfit_robust_late_entry(data):
    model = sp.KaplanMeier.fit(
        data["x"], data["c"], tl=data["tl"], w=data["w"]
    )
    surv, se = R_SURVFIT_TRUNCATED
    assert_allclose(model.sf(TIMES), surv, **CLOSED)
    assert_allclose(_log_sf_se(model, TIMES), se, **CLOSED)


def test_kaplan_meier_matches_lifelines(data):
    model = sp.KaplanMeier.fit(data["x"], data["c"], w=data["w"])
    assert_allclose(model.sf(TIMES), LIFELINES_KM, **CLOSED)


def test_rmst_robust_variance_is_the_jackknife(data):
    # The derivative of the restricted mean in each unit's weight, taken
    # numerically, and the infinitesimal jackknife sum_i w_i^2 U_i^2.
    x, c, w = data["x"], data["c"], data["w"]
    model = sp.KaplanMeier.fit(x, c, w=w)
    res = model.rmst(tau=6.0)

    def area(weights):
        return sp.KaplanMeier.fit(x, c, w=weights).mean(tau=6.0)

    h = 1e-6
    U = np.array(
        [
            (area(w + h * np.eye(len(w))[i]) - res["rmst"]) / h
            for i in range(len(w))
        ]
    )
    assert_allclose(res["rmst"], 4.104576926253483, rtol=1e-12)
    assert_allclose(res["se"], np.sqrt(np.sum(w**2 * U**2)), rtol=1e-5)
