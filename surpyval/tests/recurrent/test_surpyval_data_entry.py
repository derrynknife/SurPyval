"""Recurrent fits from a SurpyvalData.

Every recurrent model's ``fit`` takes a ``SurpyvalData`` with units ``i``
in place of the arrays (``CrowAMSAA.fit(data)``). The data's rows are
turned into the arrays a user would pass, so each fit must equal the fit
of those arrays exactly: the parameters, the covariance where there is
one, a prediction and the whole ``to_dict`` (none of its fields differ:
the data is the same). What the fit made of the terminal flags ``a`` is
in the model's ``notes``, shown by its ``repr``.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import SurpyvalData
from surpyval import recurrent as rc
from surpyval.tests.conformance.registry_fixtures import recurrent_data

X_QUERY = np.array([1.0, 5.0, 12.0, 25.0, 40.0, 55.0])


def _arrays(with_Z=False, with_e=False):
    return recurrent_data(with_Z, with_e)


def _data(d, **extra):
    kw = {"x": d["x"], "c": d["c"], "n": d["n"], "i": d["i"].tolist()}
    if "Z" in d:
        kw["Z"] = d["Z"]
    if "e" in d:
        kw["e"] = d["e"]
    return SurpyvalData(**kw, **extra)


def _same(a, b):
    """Two fitted models are the same fit: the whole to_dict."""
    da = json.dumps(a.to_dict(), sort_keys=True, default=str)
    db = json.dumps(b.to_dict(), sort_keys=True, default=str)
    assert da == db


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --- every model: fit(data) is fit(arrays) ----------------------------------

INTENSITY = ("HPP", "CrowAMSAA", "Duane", "CoxLewis")


@pytest.mark.parametrize("name", INTENSITY)
def test_intensity_models(name):
    d = _arrays()
    fitter = getattr(rc, name)
    got = fitter.fit(_data(d))
    ref = fitter.fit(d["x"], i=d["i"], c=d["c"], n=d["n"])
    _same(got, ref)
    np.testing.assert_array_equal(got.params, ref.params)
    np.testing.assert_array_equal(got.covariance(), ref.covariance())
    np.testing.assert_array_equal(got.cif(X_QUERY), ref.cif(X_QUERY))
    np.testing.assert_array_equal(got.cif_cb(X_QUERY), ref.cif_cb(X_QUERY))


def test_mcf_and_its_bounds():
    d = _arrays()
    got = rc.NonParametricCounting.fit(_data(d))
    ref = rc.NonParametricCounting.fit(d["x"], i=d["i"], c=d["c"])
    _same(got, ref)
    np.testing.assert_array_equal(got.mcf(X_QUERY), ref.mcf(X_QUERY))
    np.testing.assert_array_equal(got.mcf_cb(X_QUERY), ref.mcf_cb(X_QUERY))
    diff = rc.mcf_diff(got, ref)
    np.testing.assert_array_equal(diff.difference, 0.0)


RENEWALS: tuple[tuple[str, dict], ...] = (
    ("GeneralizedRenewal", {}),
    ("GeneralizedRenewal", {"kijima": "ii"}),
    ("GeneralizedOneRenewal", {}),
    ("ARA", {"m": 1}),
    ("ARI", {"m": 1}),
)


@pytest.mark.parametrize(
    "name, kw", RENEWALS, ids=[f"{n}{k}" for n, k in RENEWALS]
)
def test_renewal_models(name, kw):
    d = _arrays()
    fitter = getattr(rc, name)
    got = fitter.fit(_data(d), **kw)
    ref = fitter.fit(d["x"], i=d["i"], c=d["c"], **kw)
    _same(got, ref)
    np.testing.assert_array_equal(got.model.params, ref.model.params)
    np.testing.assert_array_equal(got.covariance(), ref.covariance())
    np.testing.assert_array_equal(
        got.mcf(X_QUERY, items=20, random_state=1),
        ref.mcf(X_QUERY, items=20, random_state=1),
    )


@pytest.mark.parametrize(
    "name", ("ProportionalIntensityHPP", "ProportionalIntensityNHPP")
)
def test_proportional_intensity(name):
    d = _arrays(with_Z=True)
    fitter = getattr(rc, name)
    got = fitter.fit(_data(d))
    ref = fitter.fit(d["x"], d["Z"], i=d["i"], c=d["c"])
    _same(got, ref)
    np.testing.assert_array_equal(got.coeffs, ref.coeffs)
    np.testing.assert_array_equal(got.covariance(), ref.covariance())
    np.testing.assert_array_equal(
        got.cif(X_QUERY, [0.5]), ref.cif(X_QUERY, [0.5])
    )


@pytest.mark.parametrize(
    "name, kw",
    (
        ("AndersenGill", {}),
        ("PrenticeWilliamsPeterson", {}),
        ("PrenticeWilliamsPeterson", {"time_scale": "gap"}),
        ("WeiLinWeissfeld", {}),
    ),
)
def test_recurrent_cox(name, kw):
    d = _arrays(with_Z=True)
    fitter = getattr(rc, name)
    got = fitter.fit(_data(d), **kw)
    ref = fitter.fit(d["x"], d["Z"], i=d["i"], c=d["c"], **kw)
    _same(got, ref)
    np.testing.assert_array_equal(got.params, ref.params)
    np.testing.assert_array_equal(got.standard_errors(), ref.standard_errors())


def test_named_covariates():
    d = _arrays(with_Z=True)
    data = _data(d, covariate_names=["load"])
    for fitter in (rc.ProportionalIntensityHPP, rc.AndersenGill):
        got = fitter.fit(data)
        ref = fitter.fit(
            d["x"], pd.DataFrame(d["Z"], columns=["load"]), i=d["i"], c=d["c"]
        )
        _same(got, ref)
        assert got.feature_names == ["load"]


@pytest.mark.parametrize("name", ("CauseSpecificMCF", "CauseSpecificNHPP"))
def test_cause_specific(name):
    d = _arrays(with_e=True)
    fitter = getattr(rc, name)
    got = fitter.fit(_data(d))
    ref = fitter.fit(d["x"], i=d["i"], c=d["c"], e=d["e"])
    _same(got, ref)
    assert got.event_types == ref.event_types
    for cause in got.event_types:
        f = "mcf" if name == "CauseSpecificMCF" else "cif"
        np.testing.assert_array_equal(
            getattr(got, f)(X_QUERY, cause), getattr(ref, f)(X_QUERY, cause)
        )


def test_cause_specific_proportional_intensity():
    d = _arrays(with_Z=True, with_e=True)
    fitter = rc.CauseSpecificProportionalIntensity
    got = fitter.fit(_data(d))
    ref = fitter.fit(d["x"], d["Z"], i=d["i"], c=d["c"], e=d["e"])
    _same(got, ref)
    np.testing.assert_array_equal(
        got.total_cif(X_QUERY, [0.5]), ref.total_cif(X_QUERY, [0.5])
    )


def test_every_recurrent_fit_takes_a_surpyval_data():
    # The checklist: every public recurrent fitter with a fit has the
    # entry. CountingProcess is the abstract base of the intensity
    # fitters (no fit of its own).
    fitters = {
        name: getattr(rc, name)
        for name in dir(rc)
        if callable(getattr(getattr(rc, name), "fit", None))
    }
    assert set(fitters) == {
        "ARA",
        "ARI",
        "AndersenGill",
        "CauseSpecificMCF",
        "CauseSpecificNHPP",
        "CauseSpecificProportionalIntensity",
        "CoxLewis",
        "CrowAMSAA",
        "Duane",
        "GeneralizedOneRenewal",
        "GeneralizedRenewal",
        "HPP",
        "NonParametricCounting",
        "PrenticeWilliamsPeterson",
        "ProportionalIntensityHPP",
        "ProportionalIntensityNHPP",
        "WeiLinWeissfeld",
    }

    def entry(f):
        while f is not None:
            if getattr(f, "_accepts_recurrent_surpyval_data", False):
                return True
            f = getattr(f, "__wrapped__", None)
        return False

    for name, fitter in fitters.items():
        assert entry(fitter.fit), name


# --- several windows, counts, delayed entry ---------------------------------


def _gapped():
    # Unit 1 watched to 20, not watched from 20 to 30, then to 60; unit 2
    # watched throughout, to 60.
    return SurpyvalData(
        x=[3, 9, 20, 35, 60, 4, 11, 25, 60],
        c=[0, 0, 1, 0, 1, 0, 0, 0, 1],
        tl=[0, -np.inf, -np.inf, 30, -np.inf, 0, -np.inf, -np.inf, -np.inf],
        i=[1, 1, 1, 1, 1, 2, 2, 2, 2],
    )


GAP_ARRAYS = {
    "x": [3, 9, 35, 4, 11, 25],
    "i": [1, 1, 1, 2, 2, 2],
    "windows": {1: [(0, 20), (30, 60)], 2: [(0, 60)]},
}


@pytest.mark.parametrize("name", INTENSITY + ("NonParametricCounting",))
def test_gaps_are_windows(name):
    fitter = getattr(rc, name)
    got = fitter.fit(_gapped())
    ref = fitter.fit(**GAP_ARRAYS)
    _same(got, ref)


@pytest.mark.parametrize("name", ("GeneralizedRenewal", "ARA"))
def test_gaps_refused_by_models_without_windows(name):
    with pytest.raises(ValueError, match="gaps in observation") as info:
        getattr(rc, name).fit(_gapped())
    assert "unit 1" in str(info.value)
    assert "CrowAMSAA" in str(info.value)


def test_gaps_with_covariates_go_to_fit_tvc():
    data = SurpyvalData(
        x=[3, 9, 20, 35, 60, 4, 11, 25, 60],
        c=[0, 0, 1, 0, 1, 0, 0, 0, 1],
        tl=[0, -np.inf, -np.inf, 30, -np.inf, 0, -np.inf, -np.inf, -np.inf],
        i=[1, 1, 1, 1, 1, 2, 2, 2, 2],
        Z=[[0.0]] * 5 + [[1.0]] * 4,
    )
    got = rc.ProportionalIntensityHPP.fit(data)
    ref = rc.ProportionalIntensityHPP.fit_tvc(
        i=[1, 1, 1, 1, 1, 2, 2, 2, 2],
        xl=[0, 3, 9, 30, 35, 0, 4, 11, 25],
        xr=[3, 9, 20, 35, 60, 4, 11, 25, 60],
        c=[0, 0, 1, 0, 1, 0, 0, 0, 1],
        Z=[[0.0]] * 5 + [[1.0]] * 4,
    )
    _same(got, ref)


def test_time_varying_covariates_go_to_fit_tvc():
    # Unit 1 ran under load 0 to 20, then under load 1.
    rows = dict(
        x=[3, 9, 20, 35, 56, 60, 4, 11, 25, 60],
        c=[0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        Z=[[0.0]] * 3 + [[1.0]] * 3 + [[0.5]] * 4,
    )
    i = [1] * 6 + [2] * 4
    data = SurpyvalData(**rows, i=i, covariate_names=["load"])
    xl = [0, 3, 9, 20, 35, 56, 0, 4, 11, 25]
    for fitter in (rc.ProportionalIntensityNHPP, rc.AndersenGill):
        got = fitter.fit(data)
        ref = fitter.fit_tvc(
            i=i,
            xl=xl,
            xr=rows["x"],
            c=rows["c"],
            Z=pd.DataFrame(rows["Z"], columns=["load"]),
        )
        _same(got, ref)
    with pytest.raises(ValueError, match="change within a unit"):
        rc.WeiLinWeissfeld.fit(data)


def test_stretch_rows_without_a_change_are_dropped():
    # A c = 1 row followed by the unit's next row from its x (the unit
    # watched throughout) is the arrays without it.
    data = SurpyvalData(
        x=[3, 9, 20, 35, 60, 4, 11, 60],
        c=[0, 0, 1, 0, 1, 0, 0, 1],
        i=[1, 1, 1, 1, 1, 2, 2, 2],
    )
    got = rc.CrowAMSAA.fit(data)
    ref = rc.CrowAMSAA.fit(
        [3, 9, 35, 60, 4, 11, 60],
        i=[1, 1, 1, 1, 2, 2, 2],
        c=[0, 0, 0, 1, 0, 0, 1],
    )
    _same(got, ref)


def test_counts_and_delayed_entry():
    # Unit 1 counted 3 failures between 10 and 20 and entered at 5.
    data = SurpyvalData(
        x=[[10, 20], [25, 25], [40, 40], [8, 8], [30, 30]],
        c=[2, 0, 1, 0, 1],
        n=[3, 1, 1, 1, 1],
        tl=[5, -np.inf, -np.inf, 2, -np.inf],
        i=[1, 1, 1, 2, 2],
    )
    got = rc.CrowAMSAA.fit(data)
    ref = rc.CrowAMSAA.fit(
        [[10, 20], [25, 25], [40, 40], [8, 8], [30, 30]],
        i=[1, 1, 1, 2, 2],
        c=[2, 0, 1, 0, 1],
        n=[3, 1, 1, 1, 1],
        tl=[5, 5, 5, 2, 2],
    )
    _same(got, ref)


def test_right_truncation_is_carried():
    d = _arrays()
    tr = np.where(d["i"] == 1, 70.0, np.inf)
    got = rc.CrowAMSAA.fit(_data(d, tr=tr))
    ref = rc.CrowAMSAA.fit(d["x"], i=d["i"], c=d["c"], tr=tr)
    _same(got, ref)


# --- the terminal flag a and the notes --------------------------------------


def _terminal(a=None):
    # Unit 1 scrapped at its event at 56; unit 2 still running at 60.
    return SurpyvalData(
        x=[3, 9, 20, 35, 56, 4, 11, 25, 44, 60],
        c=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        i=[1] * 5 + [2] * 5,
        a=a,
    )


TERMINAL_ARRAYS = dict(
    x=[3, 9, 20, 35, 56, 4, 11, 25, 44, 60],
    i=[1] * 5 + [2] * 5,
    c=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
)


@pytest.mark.parametrize("name", INTENSITY + ("NonParametricCounting",))
def test_terminal_event_ends_observation(name):
    fitter = getattr(rc, name)
    got = fitter.fit(_terminal(a=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
    ref = fitter.fit(**TERMINAL_ARRAYS)
    _same(got, ref)
    assert got.notes == (
        "1 unit ended in a terminal event (a = 1: 1). This model does not "
        "model terminal events, so it is fitted as an event followed by "
        "the end of that unit's observation.",
    )
    assert "Notes" in repr(got) and "terminal event" in repr(got)
    assert ref.notes == () and "Notes" not in repr(ref)


def test_inferred_terminal_flag_is_noted():
    got = rc.CrowAMSAA.fit(_terminal())
    _same(got, rc.CrowAMSAA.fit(**TERMINAL_ARRAYS))
    assert len(got.notes) == 2
    assert got.notes[1] == (
        "a was not given for 1 of them (1): an event on a unit's last row "
        "with no a is read as terminal (a = 1)."
    )


def test_event_carried_on_without_end_of_record():
    # a = 0 given on the last event and no c = 1 row: as for the arrays,
    # the record ends at that event, and the notes say so.
    got = rc.HPP.fit(_terminal(a=[0] * 10))
    _same(got, rc.HPP.fit(**TERMINAL_ARRAYS))
    assert got.notes == (
        "1 unit ends at an event with a = 0 and no end-of-record (c = 1) "
        "row (1): its observation is taken to end at that event, as for "
        "the arrays.",
    )


def test_single_event_data():
    # Each unit observed to its first event: what the arrays give.
    data = SurpyvalData(
        x=[5.0, 8.0, 12.0, 20.0], c=[0, 0, 0, 1], i=[1, 2, 3, 4]
    )
    got = rc.CrowAMSAA.fit(data)
    _same(
        got, rc.CrowAMSAA.fit([5, 8, 12, 20], i=[1, 2, 3, 4], c=[0, 0, 0, 1])
    )
    assert got.notes[0].startswith("3 units ended in a terminal event")
    assert "a was not given for 3 of them" in got.notes[1]


@pytest.mark.parametrize(
    "name, parts",
    (
        ("ProportionalIntensityHPP", ("event_types",)),
        ("WeiLinWeissfeld", ("event_types",)),
        ("GeneralizedRenewal", ("covariates", "event_types")),
        ("CauseSpecificMCF", ("covariates",)),
        ("CauseSpecificProportionalIntensity", ()),
    ),
)
def test_notes_on_every_family(name, parts):
    data = SurpyvalData(
        x=[3, 9, 20, 35, 56, 4, 11, 25, 44, 60, 5, 18, 30, 50, 60],
        c=[0] * 9 + [1] + [0] * 4 + [1],
        i=[1] * 5 + [2] * 5 + [3] * 5,
        a=[0, 0, 0, 0, 1] + [0] * 10,
        Z=[[0.0]] * 5 + [[1.0]] * 5 + [[0.5]] * 5,
        e=["a", "b"] * 4 + ["a", None, "a", "b", "a", "b", None],
    )
    model = getattr(rc, name).fit(data.without(*parts) if parts else data)
    assert model.notes and "terminal event" in model.notes[0]
    assert "Notes" in repr(model)


# --- refusals ---------------------------------------------------------------


def test_refuses_data_without_units():
    # Refused as before, now naming the unit: i.
    for fitter in (rc.CrowAMSAA, rc.NonParametricCounting, rc.ARA):
        with pytest.raises(TypeError, match=r"each row's unit \(Surpy"):
            fitter.fit(SurpyvalData(x=[1, 2, 3]))


def test_refuses_data_arguments_beside_data():
    d = _arrays()
    with pytest.raises(TypeError, match="leave out 'i', 'c'"):
        rc.CrowAMSAA.fit(_data(d), i=d["i"], c=d["c"])


def test_refuses_states():
    data = SurpyvalData(
        x=[3, 5, 7, 4],
        tl=[0, 3, 0, 0],
        s=["well", "ill", "well", "well"],
        e=["ill", "dead", None, "dead"],
        i=["A", "A", "B", "C"],
    )
    with pytest.raises(ValueError, match="multi-state models"):
        rc.CrowAMSAA.fit(data)


def test_refuses_what_the_fit_has_no_argument_for():
    d = _arrays(with_Z=True, with_e=True)
    with pytest.raises(ValueError, match="takes no covariates or event types"):
        rc.CrowAMSAA.fit(_data(d))
    with pytest.raises(ValueError, match="cause-specific models"):
        rc.CrowAMSAA.fit(_data(d).without("covariates"))
    with pytest.raises(TypeError, match="needs covariates, Z"):
        rc.ProportionalIntensityHPP.fit(_data(_arrays()))


def test_refuses_weights():
    d = _arrays()
    with pytest.raises(ValueError, match="no recurrent model takes weights"):
        rc.CrowAMSAA.fit(_data(d, w=np.full(16, 2.0)))


def test_rows_after_a_terminal_row_are_refused_by_the_data():
    with pytest.raises(ValueError, match="after a row with a = 1"):
        SurpyvalData(x=[2, 5, 9], i=[1, 1, 1], a=[0, 1, 0])


def test_handle_xicn_refusals_surface():
    # Simultaneous events: an exact row stands for one event (n = 1).
    data = SurpyvalData(x=[2, 5, 9], c=[0, 0, 1], n=[1, 2, 1], i=[1, 1, 1])
    with pytest.raises(ValueError, match="stands for one event"):
        rc.CrowAMSAA.fit(data)


# --- fit_from_df with a_col -------------------------------------------------


def _log():
    return pd.DataFrame(
        {
            "hours": [56, 3, 9, 20, 35, 4, 11, 25, 44, 60],
            "unit": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "c": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "a": [1, 0, 0, 0, 0, 0, 0, 0, 0, None],
        }
    )


def test_fit_from_df_with_a_col():
    log = _log()
    got = rc.CrowAMSAA.fit_from_df(
        log, x_col="hours", i_col="unit", c_col="c", a_col="a"
    )
    ref = rc.CrowAMSAA.fit_from_df(log, x_col="hours", i_col="unit", c_col="c")
    _same(got, ref)
    assert (
        got.notes == rc.CrowAMSAA.fit(_terminal([0] * 4 + [1] + [0] * 5)).notes
    )
    assert ref.notes == ()


def test_fit_from_df_a_col_refusal_names_the_frame_row():
    log = _log()
    log.loc[1, "a"] = 1  # the event at 3 cannot end unit 1: more follow
    with pytest.raises(ValueError, match=r"row 2 after row 1 of unit 1"):
        rc.CrowAMSAA.fit_from_df(
            log, x_col="hours", i_col="unit", c_col="c", a_col="a"
        )


def test_fit_from_df_a_col_on_every_entry():
    log = _log().assign(load=[0.0] * 5 + [1.0] * 5)
    log["e"] = ["a", "a", "b", "a", "b", "a", "b", "a", "b", None]
    cols = dict(x_col="hours", i_col="unit", c_col="c", a_col="a")
    models = [
        rc.HPP.fit_from_df(log, **cols),
        rc.ARI.fit_from_df(log, **cols),
        rc.ProportionalIntensityHPP.fit_from_df(log, Z_cols="load", **cols),
        rc.AndersenGill.fit_from_df(log, Z_cols="load", **cols),
        rc.CauseSpecificMCF.fit_from_df(log, e_col="e", **cols),
        rc.CauseSpecificNHPP.fit_from_df(log, e_col="e", **cols),
        rc.CauseSpecificProportionalIntensity.fit_from_df(
            log, Z_cols="load", e_col="e", **cols
        ),
    ]
    for model in models:
        assert model.notes and "terminal event" in model.notes[0]


# --- the existing entry points are unchanged --------------------------------


def test_event_history_data_entry_unchanged():
    d = _arrays()
    history = sp.handle_xicn(d["x"], d["i"], d["c"], d["n"])
    _same(
        rc.CrowAMSAA.fit(history),
        rc.CrowAMSAA.fit(d["x"], i=d["i"], c=d["c"]),
    )
    model = rc.CrowAMSAA.fit_from_recurrent_data(history)
    _same(model, rc.CrowAMSAA.fit(d["x"], i=d["i"], c=d["c"]))
    assert model.notes == ()
