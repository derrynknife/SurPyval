"""``diagnose()`` is the one assumptions-checking API: it gives the
proportional-hazards test ``check_ph`` gave, as the same table, on the
same ``transform`` (``d.tables["proportional_hazards"]``), and
``check_ph``, deprecated in v0.26, warns until it is removed in v0.28.

Each test here that calls ``check_ph`` goes when it is removed; the
``diagnose()`` side stays.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.diagnostics import Diagnosis
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
)
from surpyval.univariate.regression.proportional_hazards import diagnostics
from surpyval.utils.deprecation import SCHEDULED_REMOVALS

TRANSFORMS = ("km", "rank", "identity", "log")
TIES = ("efron", "breslow", "exact", "kalbfleisch-prentice")
MESSAGE = (
    "check_ph is deprecated and will be removed in v0.28; use "
    "model.diagnose(), which runs the same proportional-hazards test "
    "(transform=...)"
)


def _quiet(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args, **kwargs)


def _tied_data(seed=3, n=150):
    """Two covariates (a binary and a continuous one), times rounded so
    that events tie, some right censored."""
    rng = np.random.default_rng(seed)
    Z = np.column_stack([rng.binomial(1, 0.5, n), rng.normal(size=n)])
    t = rng.exponential(size=n) / np.exp(0.6 * Z[:, 0] - 0.4 * Z[:, 1])
    x = np.ceil(4 * np.clip(t, 0.1, 3)) / 4
    c = (rng.random(n) < 0.2).astype(int)
    return x, Z.astype(float), c


def _old_table(model, transform):
    return _quiet(model.check_ph, transform)


# -- the table is check_ph's, number for number ------------------------------


@pytest.mark.parametrize("tie_method", TIES)
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_diagnose_table_is_check_ph(tie_method, transform):
    x, Z, c = _tied_data()
    model = _quiet(sp.CoxPH.fit, x, Z, c=c, tie_method=tie_method)
    d = _quiet(model.diagnose, transform=transform)
    table = d.tables["proportional_hazards"]
    old = _old_table(model, transform)
    pd.testing.assert_frame_equal(table, old, check_exact=True)
    assert table.attrs == old.attrs == {"transform": transform}


def test_default_transform_is_check_ph_default():
    x, Z, c = _tied_data()
    model = _quiet(sp.CoxPH.fit, x, Z, c=c)
    table = _quiet(model.diagnose).tables["proportional_hazards"]
    pd.testing.assert_frame_equal(table, _quiet(model.check_ph))
    assert table.attrs["transform"] == "km"


def test_names_from_a_dataframe():
    df = sp.datasets.load_rossi_static()
    df["censored"] = 1 - df["arrest"]
    model = sp.CoxPH.fit_from_df(
        df, x_col="week", c_col="censored", Z_cols=["fin", "age", "prio"]
    )
    table = model.diagnose(transform="rank").tables["proportional_hazards"]
    assert list(table.index) == ["fin", "age", "prio", "GLOBAL"]
    pd.testing.assert_frame_equal(table, _old_table(model, "rank"))


def test_the_table_holds_every_covariate_whatever_columns():
    x, Z, c = _tied_data()
    model = _quiet(sp.CoxPH.fit, x, Z, c=c)
    d = _quiet(model.diagnose, columns=["coef_1"])
    pd.testing.assert_frame_equal(
        d.tables["proportional_hazards"], _old_table(model, "km")
    )


def test_the_rows_are_the_table():
    # The table's numbers are the rows' on its transform: one test.
    x, Z, c = _tied_data()
    model = _quiet(sp.CoxPH.fit, x, Z, c=c)
    for transform in TRANSFORMS:
        d = _quiet(model.diagnose, transform=transform)
        rows = d.table[
            (d.table["check"] == "proportional_hazards")
            & (d.table["transform"] == transform)
        ]
        table = d.tables["proportional_hazards"]
        np.testing.assert_array_equal(rows["statistic"], table["statistic"])
        np.testing.assert_array_equal(rows["p"], table["p"])


def test_a_transform_beyond_km_and_rank_adds_its_rows():
    x, Z, c = _tied_data()
    model = _quiet(sp.CoxPH.fit, x, Z, c=c)
    by_default = _quiet(model.diagnose)
    ph = by_default.table["check"] == "proportional_hazards"
    assert set(by_default.table.loc[ph, "transform"]) == {"km", "rank"}
    logged = _quiet(model.diagnose, transform="log").table
    ph = logged["check"] == "proportional_hazards"
    assert list(dict.fromkeys(logged.loc[ph, "transform"])) == [
        "km",
        "rank",
        "log",
    ]


def test_an_unknown_transform_is_refused():
    x, Z, c = _tied_data()
    model = _quiet(sp.CoxPH.fit, x, Z, c=c)
    with pytest.raises(ValueError, match="'transform' must be one of"):
        model.diagnose(transform="bogus")


def test_stratified_both_refuse():
    x, Z, c = _tied_data()
    strata = np.arange(x.size) % 2
    model = _quiet(sp.CoxPH.fit, x, Z, c=c, strata=strata)
    with pytest.raises(NotImplementedError, match="stratified"):
        model.diagnose()
    with pytest.raises(NotImplementedError, match="stratified"):
        _old_table(model, "km")


# -- the competing-risks (cause-specific Cox) model --------------------------


def _competing():
    rng = np.random.default_rng(5)
    n = 240
    Z = rng.normal(size=(n, 2))
    t = rng.exponential(size=n) / np.exp(0.5 * Z[:, 0])
    e = rng.integers(1, 3, size=n)
    return _quiet(CompetingRisksProportionalHazards.fit, t, Z, e=e)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_competing_risks_table_is_each_cause_check_ph(transform):
    model = _competing()
    d = _quiet(model.diagnose, transform=transform)
    table = d.tables["proportional_hazards"]
    assert table.index.names == ["cause", "covariate"]
    assert table.attrs == {"transform": transform}
    for cause, cause_model in model.models.items():
        pd.testing.assert_frame_equal(
            table.loc[str(cause)],
            _old_table(cause_model, transform),
            check_exact=True,
        )


# -- the report keeps the table ----------------------------------------------


@pytest.mark.parametrize("competing", [False, True])
def test_the_table_survives_to_dict(competing):
    if competing:
        model = _competing()
    else:
        x, Z, c = _tied_data()
        model = _quiet(sp.CoxPH.fit, x, Z, c=c)
    d = _quiet(model.diagnose, transform="rank")
    text = json.dumps(d.to_dict(), allow_nan=False)
    back = Diagnosis.from_dict(json.loads(text))
    pd.testing.assert_frame_equal(
        back.tables["proportional_hazards"], d.tables["proportional_hazards"]
    )
    assert back.tables["proportional_hazards"].attrs == {"transform": "rank"}
    assert back.to_dict() == d.to_dict()


def test_a_report_without_tables_reads_back():
    # A report written before the tables were kept has none.
    x, Z, c = _tied_data()
    data = _quiet(sp.CoxPH.fit, x, Z, c=c).diagnose().to_dict()
    del data["tables"]
    assert Diagnosis.from_dict(data).tables == {}


def test_a_family_without_a_table_has_none():
    x = sp.Weibull.random(60, 10, 2, random_state=1)
    d = _quiet(sp.Weibull.fit, x).diagnose(n_boot=20, random_state=1)
    assert d.tables == {}


# -- check_ph is deprecated --------------------------------------------------


def test_check_ph_is_scheduled_for_removal_in_028():
    assert SCHEDULED_REMOVALS["check_ph"] == "0.28"


def test_the_method_warns_and_returns_the_same_table():
    x, Z, c = _tied_data()
    model = _quiet(sp.CoxPH.fit, x, Z, c=c)
    with pytest.warns(DeprecationWarning) as caught:
        old = model.check_ph("rank")
    assert [str(w.message) for w in caught] == [MESSAGE]
    # The warning points at the line that called check_ph.
    assert caught[0].filename == __file__
    pd.testing.assert_frame_equal(
        old, model.diagnose(transform="rank").tables["proportional_hazards"]
    )


def test_the_function_warns_and_returns_the_same_dict():
    x, Z, c = _tied_data()
    model = _quiet(sp.CoxPH.fit, x, Z, c=c)
    with pytest.warns(DeprecationWarning) as caught:
        old = diagnostics.check_ph(model, "log")
    assert [str(w.message) for w in caught] == [MESSAGE]
    assert caught[0].filename == __file__
    assert old == diagnostics.zph_score_test(model, "log")


def test_diagnose_raises_no_deprecation_warning():
    # diagnose() and the shared functions use the test directly: nothing
    # inside SurPyval calls check_ph.
    x, Z, c = _tied_data()
    model = _quiet(sp.CoxPH.fit, x, Z, c=c)
    competing = _competing()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        model.diagnose(transform="log")
        competing.diagnose()
        diagnostics.zph_score_test(model)
        diagnostics.zph_table(model, "rank")


# -- a penalised fit: check_ph refuses as diagnose() does --------------------


def test_a_penalised_fit_is_refused_alike():
    x, Z, c = _tied_data()
    model = _quiet(sp.CoxPH.fit_penalised, x, Z, c=c, penalizer=0.1)
    with pytest.raises(ValueError) as by_diagnose:
        model.diagnose()
    with pytest.raises(ValueError) as by_method:
        _old_table(model, "km")
    with pytest.raises(ValueError) as by_function:
        _quiet(diagnostics.check_ph, model)
    assert "penalty" in str(by_diagnose.value)
    assert str(by_method.value) == str(by_diagnose.value)
    assert str(by_function.value) == str(by_diagnose.value)
