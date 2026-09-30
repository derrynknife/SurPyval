"""``formula=`` for the parametric regressions' time-varying ``_from_df``
fits, as for Cox's (#485).

``fit_tvc_from_df`` (PH, AH, PO and AFT) and ``fit_tvc_timeline_from_df``
(PH, AH and PO) took only ``Z_cols``, so a categorical column (a
``"yes"`` / ``"no"`` column, say) had to be coded by hand. They now take a
``formulaic`` formula instead, with the design of ``fit_from_df``: the
model keeps ``feature_names``, ``formula`` and its encoding, predicts from
a DataFrame with the same design, round-trips through ``to_dict``, and
names the formula's columns in an aliasing warning (#476).
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp

START_STOP = ["WeibullPH", "WeibullAH", "WeibullPO", "WeibullAFT"]
TIMELINE = ["WeibullPH", "WeibullAH", "WeibullPO"]


def _start_stop(n=150):
    """Start-stop rows: a stress switches on at a random time (``on``,
    and as text in ``stress``), and a fixed covariate ``z`` acts
    throughout."""
    rng = np.random.default_rng(0)
    switch = rng.uniform(0.3, 1.5, n)
    z = rng.normal(size=n)
    t_low = rng.exponential(2.0, n) * np.exp(-0.3 * z)
    t_high = switch + rng.exponential(2.0 / np.e, n) * np.exp(-0.3 * z)
    T = np.where(t_low > switch, t_high, t_low)
    one = T <= switch
    on = np.r_[np.zeros(n), np.ones((~one).sum())]
    return pd.DataFrame(
        {
            "id": np.r_[np.arange(n), np.flatnonzero(~one)],
            "start": np.r_[np.zeros(n), switch[~one]],
            "stop": np.r_[np.where(one, T, switch), T[~one]],
            "c": np.r_[np.where(one, 0, 1), np.zeros((~one).sum(), int)],
            "on": on,
            "stress": np.where(on == 1, "on", "off"),
            "z": np.r_[z, z[~one]],
        }
    )


def _timeline(df):
    """The same subjects as a covariate timeline: each start-stop row's
    start, and a terminal row per subject with its exit and status."""
    rows = df[["id", "start", "on", "stress", "z"]].rename(
        columns={"start": "time"}
    )
    rows = rows.assign(c=1)
    last = df.groupby("id").tail(1)
    end = pd.DataFrame(
        {
            "id": last["id"],
            "time": last["stop"],
            "on": last["on"],
            "stress": last["stress"],
            "z": last["z"],
            "c": last["c"],
        }
    )
    out = pd.concat([rows, end]).sort_values(["id", "time"], kind="stable")
    return out.reset_index(drop=True)


@pytest.fixture(scope="module")
def data():
    return _start_stop()


def _quiet(fit):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model = fit()
    return model, rec


@pytest.mark.parametrize("name", START_STOP)
def test_start_stop_formula_matches_z_cols(name, data):
    F = getattr(sp, name)
    by_cols = F.fit_tvc_from_df(data, "id", "start", "stop", "c", ["on", "z"])
    model = F.fit_tvc_from_df(
        data, "id", "start", "stop", "c", formula="on + z"
    )
    np.testing.assert_allclose(model.params, by_cols.params, rtol=1e-8)
    assert model.feature_names == ["on", "z"]
    assert model.formula == "on + z"
    assert model._model_spec is not None
    assert by_cols.formula is None and by_cols._model_spec is None


@pytest.mark.parametrize("name", TIMELINE)
def test_timeline_formula_matches_z_cols(name, data):
    F = getattr(sp, name)
    tl = _timeline(data)
    by_cols = F.fit_tvc_timeline_from_df(tl, "id", "time", ["on", "z"], "c")
    model = F.fit_tvc_timeline_from_df(
        tl, "id", "time", None, "c", formula="on + z"
    )
    np.testing.assert_allclose(model.params, by_cols.params, rtol=1e-8)
    # the timeline is the same data as the start-stop rows
    ss = F.fit_tvc_from_df(data, "id", "start", "stop", "c", ["on", "z"])
    np.testing.assert_allclose(model.params, ss.params, rtol=1e-6)


@pytest.mark.parametrize("name", START_STOP)
def test_a_categorical_column_through_a_formula(name, data):
    # Before: Z_cols=["stress"] raised; the column had to be coded by hand
    F = getattr(sp, name)
    model = F.fit_tvc_from_df(
        data, "id", "start", "stop", "c", formula="stress + z"
    )
    assert model.feature_names == ["stress[T.on]", "z"]
    coded = F.fit_tvc_from_df(data, "id", "start", "stop", "c", ["on", "z"])
    np.testing.assert_allclose(model.params, coded.params, rtol=1e-8)


def test_a_categorical_timeline_through_a_formula(data):
    tl = _timeline(data)
    model = sp.WeibullPH.fit_tvc_timeline_from_df(
        tl, "id", "time", None, "c", formula="stress + z"
    )
    coded = sp.WeibullPH.fit_tvc_timeline_from_df(
        tl, "id", "time", ["on", "z"], "c"
    )
    np.testing.assert_allclose(model.params, coded.params, rtol=1e-8)


@pytest.mark.parametrize("name", START_STOP)
def test_a_string_column_suggests_a_formula(name, data):
    F = getattr(sp, name)
    with pytest.raises(ValueError, match="not numeric.*formula="):
        F.fit_tvc_from_df(data, "id", "start", "stop", "c", ["stress"])


def test_a_string_timeline_column_suggests_a_formula(data):
    with pytest.raises(ValueError, match="not numeric.*formula="):
        sp.WeibullPO.fit_tvc_timeline_from_df(
            _timeline(data), "id", "time", "stress", "c"
        )


@pytest.mark.parametrize("name", START_STOP)
def test_exactly_one_of_z_cols_and_formula(name, data):
    F = getattr(sp, name)
    with pytest.raises(ValueError, match="not both"):
        F.fit_tvc_from_df(
            data, "id", "start", "stop", "c", ["on"], formula="on"
        )
    with pytest.raises(ValueError, match="must be provided"):
        F.fit_tvc_from_df(data, "id", "start", "stop", "c")


@pytest.mark.parametrize("name", START_STOP)
def test_predicts_from_a_dataframe_with_the_formula(name, data):
    F = getattr(sp, name)
    model = F.fit_tvc_from_df(
        data, "id", "start", "stop", "c", formula="stress + z"
    )
    rows = pd.DataFrame({"stress": ["off", "on"], "z": [0.5, -1.0]})
    Z = np.array([[0.0, 0.5], [1.0, -1.0]])
    np.testing.assert_allclose(
        model.sf([1.0, 2.0], rows), model.sf([1.0, 2.0], Z), rtol=1e-12
    )
    # a level the fit never saw is refused, not coded as the reference
    with pytest.raises(ValueError, match="stress"):
        model.sf([1.0], pd.DataFrame({"stress": ["maybe"], "z": [0.0]}))


@pytest.mark.parametrize("name", START_STOP)
def test_round_trips_with_its_formula(name, data):
    F = getattr(sp, name)
    model = F.fit_tvc_from_df(
        data, "id", "start", "stop", "c", formula="stress + z"
    )
    stored = json.loads(json.dumps(model.to_dict()))
    assert stored["formula"] == "stress + z"
    restored = sp.from_dict(stored)
    assert restored.feature_names == model.feature_names
    rows = pd.DataFrame({"stress": ["on", "off"], "z": [0.2, 1.1]})
    np.testing.assert_allclose(
        restored.sf([0.5, 3.0], rows), model.sf([0.5, 3.0], rows), rtol=1e-12
    )


@pytest.mark.parametrize("name", START_STOP)
def test_the_aliasing_warning_names_the_formulas_columns(name, data):
    F = getattr(sp, name)
    twice = data.assign(stress2=data["stress"])
    model, rec = _quiet(
        lambda: F.fit_tvc_from_df(
            twice, "id", "start", "stop", "c", formula="stress + z + stress2"
        )
    )
    # (The additive hazards fit also warns that it ends on its positivity
    # boundary on this data, as it does without the repeated column.)
    aliased = [w for w in rec if "cannot be estimated" in str(w.message)]
    assert len(aliased) == 1, [str(w.message) for w in rec]
    assert "2 ('stress2[T.on]')" in str(aliased[0].message)
    assert aliased[0].filename == __file__
    assert np.isnan(model.params[-1])


def test_the_timeline_aliasing_warning_names_the_formulas_columns(data):
    tl = _timeline(data).assign(double_z=lambda d: 2 * d["z"])
    _, rec = _quiet(
        lambda: sp.WeibullPH.fit_tvc_timeline_from_df(
            tl, "id", "time", None, "c", formula="on + z + double_z"
        )
    )
    messages = [str(w.message) for w in rec]
    assert len(messages) == 1 and "double_z" in messages[0], messages
