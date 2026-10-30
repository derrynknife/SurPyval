"""The Cox proportional-hazards test and ``diagnose()`` against stored R
``survival`` results (#868).

The proportional-hazards test (``diagnose()``'s table and rows) is
R's ``cox.zph`` (survival 3.0 and later: the score test of adding
``Z * g(t)``, per covariate and GLOBAL) on the ``km`` and ``rank``
scales of time, under Efron and Breslow ties;
the functional-form rows are the score test of adding a covariate's
square, which R reports as ``coxph(..., init = c(coef(fit), 0),
iter.max = 0)$score``. Three data sets: R's ``lung`` (a textbook fit
that meets the assumption), R's ``veteran`` (the textbook violation:
the Karnofsky score's effect fades) and a seeded synthetic set with a
deliberate violation (a binary covariate whose hazard ratio reverses, a
covariate acting through its square, delayed entry and ties).

Tolerances: the statistics are smooth functions of the coefficients,
which agree with R's to about 1e-9 (see ``test_cox.py``), so they agree
to ``rtol=1e-6``; so do the scaled Schoenfeld residuals.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.univariate.regression.proportional_hazards.diagnostics import (
    _zph_times,
    added_covariate_score_test,
    compute_residuals,
    zph_score_test,
)

from ._data import fixture, values

DERIVED = dict(rtol=1e-6, atol=1e-10)


def _lung(ties):
    d = fixture("lung")
    keep = ~np.isnan(d["ph_ecog"])
    Z = np.column_stack([d["age"], d["sex"], d["ph_ecog"]])[keep]
    model = sp.CoxPH.fit(d["time"][keep], Z, c=d["c"][keep], tie_method=ties)
    return model, Z, ["age", "sex", "ph_ecog"]


def _violation(ties):
    d = fixture("ph_violation")
    Z = np.column_stack([d["z1"], d["z2"], d["z3"]])
    model = sp.CoxPH.fit(d["x"], Z, c=d["c"], tl=d["tl"], tie_method=ties)
    return model, Z, ["z1", "z2", "z3"]


def _veteran():
    d = fixture("veteran")
    cell = d["celltype"]
    Z = np.column_stack(
        [
            d["trt"],
            d["karno"],
            d["diagtime"],
            d["age"],
            d["prior"],
            cell == 2,
            cell == 3,
            cell == 4,
        ]
    ).astype(float)
    model = sp.CoxPH.fit(d["time"], Z, c=1 - d["status"], tie_method="efron")
    names = ["trt", "karno", "diagtime", "age", "prior"]
    return model, Z, names + ["smallcell", "adeno", "large"]


CASES = {
    "zph_lung_efron": lambda: _lung("efron"),
    "zph_lung_breslow": lambda: _lung("breslow"),
    "zph_ph_violation_efron": lambda: _violation("efron"),
    "zph_ph_violation_breslow": lambda: _violation("breslow"),
    "zph_veteran_efron": _veteran,
}


@pytest.mark.parametrize("ref_id", sorted(CASES))
@pytest.mark.parametrize("transform", ["km", "rank"])
def test_ph_test_matches_cox_zph(ref_id, transform):
    model, _, names = CASES[ref_id]()
    ref = values("r_survival", ref_id)
    assert ref["rows"] == names + ["GLOBAL"]
    res = zph_score_test(model, transform)
    rows = res["per_covariate"] + [res["global"]]
    assert_allclose(
        [r["statistic"] for r in rows], ref[transform + "_chisq"], **DERIVED
    )
    assert_allclose(
        [r["p_value"] for r in rows], ref[transform + "_p"], **DERIVED
    )
    assert [r["df"] for r in rows] == ref[transform + "_df"].tolist()


@pytest.mark.parametrize("ref_id", sorted(CASES))
@pytest.mark.parametrize("transform", ["km", "rank"])
def test_the_table_and_diagnose_report_the_same_test(ref_id, transform):
    # One implementation: diagnose()'s table and rows, on the transform
    # given, are the same numbers, R's.
    model, _, names = CASES[ref_id]()
    model.feature_names = names
    ref = values("r_survival", ref_id)
    d = model.diagnose(transform=transform)
    table = d.tables["proportional_hazards"]
    assert list(table.index) == names + ["GLOBAL"]
    assert table["df"].tolist() == ref[transform + "_df"].tolist()
    assert_allclose(table["statistic"], ref[transform + "_chisq"], **DERIVED)
    assert_allclose(table["p"], ref[transform + "_p"], **DERIVED)
    d = d.table
    ph = d[
        (d["check"] == "proportional_hazards") & (d["transform"] == transform)
    ]
    np.testing.assert_array_equal(ph["statistic"], table["statistic"])
    np.testing.assert_array_equal(ph["p"], table["p"])


@pytest.mark.parametrize("ref_id", sorted(CASES))
def test_square_score_test_matches_coxph(ref_id):
    model, Z, names = CASES[ref_id]()
    ref = values("r_survival", ref_id)
    squares = [k for k in ref if k.startswith("square_")]
    assert squares
    for key in squares:
        j = names.index(key[len("square_") :])
        res = added_covariate_score_test(model, Z[:, j] ** 2)
        assert_allclose(res["statistic"], ref[key], **DERIVED)


@pytest.mark.parametrize("ref_id", sorted(CASES))
def test_diagnose_reports_the_r_values(ref_id):
    model, Z, names = CASES[ref_id]()
    model.feature_names = names
    ref = values("r_survival", ref_id)
    table = model.diagnose().table
    for transform in ("km", "rank"):
        ph = table[
            (table["check"] == "proportional_hazards")
            & (table["transform"] == transform)
        ]
        assert [
            c[0] if len(c) == 1 else "GLOBAL" for c in ph["covariates"]
        ] == (names + ["GLOBAL"])
        assert_allclose(ph["p"], ref[transform + "_p"], **DERIVED)
    form = table[table["check"] == "functional_form"].set_index(
        table.loc[table["check"] == "functional_form", "covariates"].str[0]
    )
    for key in (k for k in ref if k.startswith("square_")):
        name = key[len("square_") :]
        assert_allclose(form.loc[name, "statistic"], ref[key], **DERIVED)


def test_scaled_schoenfeld_and_km_times_match_cox_zph():
    # cox.zph's $y (the scaled Schoenfeld residuals, beta + d V s) and $x
    # (the km-scale times), event by event; R orders the events by time,
    # SurPyval keeps them in the data's order, so both are sorted by time
    # and, within a tie, by value.
    model, _, _ = _lung("efron")
    ref = values("r_survival", "zph_lung_efron")
    data = model._fit_data
    events = np.flatnonzero(data["c"] == 0)
    t = data["x"][events]
    assert_allclose(np.sort(t), np.sort(ref["km_time"]), rtol=0, atol=0)
    g = _zph_times(data, "km")[events]
    assert_allclose(np.sort(g), np.sort(ref["km_x"]), **DERIVED)
    scaled = compute_residuals(model, "scaled_schoenfeld")
    for j in range(scaled.shape[1]):
        mine = scaled[np.lexsort((scaled[:, j], t)), j]
        theirs_t = np.asarray(ref["km_time"])
        theirs = ref["km_y"][:, j]
        theirs = theirs[np.lexsort((theirs, theirs_t))]
        assert_allclose(mine, theirs, **DERIVED)


def test_the_violations_are_found():
    # What the fixtures were built to show: veteran's Karnofsky score and
    # the synthetic z1 fail proportional hazards, and z3 its functional
    # form; the lung fit passes both.
    model, _, names = _veteran()
    model.feature_names = names
    failed = model.diagnose().table.query("verdict == 'fail'")
    assert ("karno",) in set(failed["covariates"])
    model, _, names = _violation("efron")
    model.feature_names = names
    d = model.diagnose()
    failed = d.table.query("verdict == 'fail'")
    ph = failed[failed["check"] == "proportional_hazards"]["covariates"]
    assert ("z1",) in set(ph) and ("z2",) not in set(ph)
    form = failed[failed["check"] == "functional_form"]["covariates"]
    assert set(form) == {("z3",)}
    model, _, names = _lung("efron")
    model.feature_names = names
    table = model.diagnose().table
    tests = table[table["check"].isin(["proportional_hazards"])]
    assert (tests["verdict"] != "fail").all()
