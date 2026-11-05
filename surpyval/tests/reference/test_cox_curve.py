"""The confidence bounds on a Cox model's predicted survival curve
(``SemiParametricRegressionModel.cb``) against stored R ``survival``
results (``scripts/reference/reference_r_cox_curve.R``):
``survfit(fit, newdata = nd)`` with ``conf.type`` "log" (R's default),
"log-log" and "plain", at 95% and (log) 90%.

The fits: ``lung`` (``~ age + sex``) under Efron and Breslow ties,
unweighted and with case weights (R's robust variance); ``lung`` with
``strata(sex)``; ``ovarian``; and the start-stop ``heart`` data. R's
``std.err`` (the standard error of the cumulative hazard) is read back
from the bounds. The coefficients agree with R's to about 3e-9, and the
bounds, smooth functions of them, to about 1e-9; they are tested to
``rtol=1e-6`` (``atol=1e-8`` where a value is near 0).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

import surpyval as sp

from ._data import fixture, values

SOURCE = "r_cox_curve"
TOL = dict(rtol=1e-6, atol=1e-8)
BOUND_TYPES = {"log": "log", "log_log": "log-log", "plain": "plain"}


def _lung(ties, weighted=False):
    d = fixture("lung")
    Z = np.column_stack([d["age"], d["sex"]])
    w = 0.5 + (np.arange(1, d["time"].size + 1) % 5) / 4 if weighted else None
    model = sp.CoxPH.fit(d["time"], Z, c=d["c"], w=w, tie_method=ties)
    return model, [[60, 1], [72, 2], [45, 2]], None


def _lung_strata():
    d = fixture("lung")
    keep = ~np.isnan(d["ph_ecog"])
    Z = np.column_stack([d["age"], d["ph_ecog"]])[keep]
    model = sp.CoxPH.fit(
        d["time"][keep], Z, c=d["c"][keep], strata=d["sex"][keep]
    )
    return model, [[60, 1], [70, 2]], [1.0, 2.0]


def _ovarian():
    d = fixture("ovarian")
    Z = np.column_stack([d["age"], d["ecog_ps"]])
    model = sp.CoxPH.fit(d["futime"], Z, c=1 - d["fustat"])
    return model, [[50, 1], [65, 2]], None


def _heart():
    d = fixture("heart")
    Z = np.column_stack([d["age"], d["year"], d["transplant"]])
    model = sp.CoxPH.fit_tvc(d["id"], d["start"], d["stop"], d["c"], Z)
    return model, [[-5, 2, 0], [5, 4, 1]], None


CASES = {
    "lung_efron": lambda: _lung("efron"),
    "lung_breslow": lambda: _lung("breslow"),
    "lung_weighted_efron": lambda: _lung("efron", True),
    "lung_weighted_breslow": lambda: _lung("breslow", True),
    "lung_strata_sex_efron": _lung_strata,
    "ovarian_efron": _ovarian,
    "heart_efron": _heart,
}


def _curves(ref_id):
    """Each newdata row's model, row, stratum, times and R values."""
    model, rows, strata = CASES[ref_id]()
    ref = values(SOURCE, ref_id)
    assert_allclose(model.params, ref["coef"], rtol=1e-7)
    for k, row in enumerate(rows):
        stratum = None if strata is None else strata[k]
        t = ref["time"][k] if strata is not None else ref["time"]
        yield model, row, stratum, t, {
            key: v[k] for key, v in ref.items() if key not in ("time", "coef")
        }


@pytest.mark.parametrize("ref_id", list(CASES))
def test_survival_and_standard_error_match_r(ref_id):
    for model, row, stratum, t, r in _curves(ref_id):
        assert_allclose(model.sf(t, row, stratum), r["surv"], **TOL)
        assert_allclose(model.Hf(t, row, stratum), r["cumhaz"], **TOL)
        # R's std.err, the standard error of H, from the log bound:
        # its upper bound on H is H + z se
        z = norm.isf(0.025)
        H = model.Hf(t, row, stratum)
        H_hi = model.cb(t, row, stratum, on="Hf")[:, 1]
        assert_allclose((H_hi - H) / z, r["std_err"], **TOL)


@pytest.mark.parametrize("ref_id", list(CASES))
@pytest.mark.parametrize("conf_type", list(BOUND_TYPES))
def test_bounds_match_r(ref_id, conf_type):
    for model, row, stratum, t, r in _curves(ref_id):
        got = model.cb(t, row, stratum, bound_type=BOUND_TYPES[conf_type])
        for j, side in enumerate(("lower", "upper")):
            want = r[f"{side}_{conf_type}"]
            # R gives NA for a log-log bound where the survival is 1
            # with a standard error (none here) or 0
            ok = ~np.isnan(want)
            assert_allclose(got[ok, j], want[ok], **TOL)


@pytest.mark.parametrize("ref_id", list(CASES))
def test_90_percent_bounds_match_r(ref_id):
    for model, row, stratum, t, r in _curves(ref_id):
        got = model.cb(t, row, stratum, alpha_ci=0.1)
        assert_allclose(got[:, 0], r["lower_log_90"], **TOL)
        assert_allclose(got[:, 1], r["upper_log_90"], **TOL)
