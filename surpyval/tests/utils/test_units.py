"""The ``unit=`` label on models (#845): kept as typed on the model, shown
in its repr and on its plots, and saved by ``to_dict``; read with pint only
where labels are compared (decided 13 Nov 2026)."""

import inspect
import subprocess
import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import recurrent
from surpyval.utils.units import (
    axis_label,
    normalise_unit,
    same_unit,
    takes_unit,
    unit_symbol,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

X = sp.Weibull.random(30, 1000, 2, random_state=1)
Z = np.r_[np.zeros(15), np.ones(15)]


@pytest.mark.parametrize(
    "given, stored",
    [
        ("hrs", "hrs"),
        ("hours", "hours"),
        ("Hours", "Hours"),
        ("h", "h"),
        ("km", "km"),
        ("kilometres", "kilometres"),
        ("cycles", "cycles"),
        ("Cycles", "Cycles"),
        ("starts", "starts"),
        ("years", "years"),
        ("kWh", "kWh"),
        ("mm/s", "mm/s"),
        ("°C", "°C"),
        ("bar", "bar"),
        ("  pump   starts ", "pump   starts"),
        ("landing gear swaps", "landing gear swaps"),
        (None, None),
    ],
)
def test_normalise_unit(given, stored):
    assert normalise_unit(given) == stored


def test_a_pint_unit_is_taken_by_its_name():
    import pint

    assert normalise_unit(pint.UnitRegistry().hour) == "hour"


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_units_refused(bad):
    with pytest.raises(ValueError, match="leave it out"):
        sp.Weibull.fit(X, unit=bad)


def test_non_text_units_refused_before_fitting():
    with pytest.raises(TypeError, match="unit must be text"):
        sp.Weibull.fit(X, unit=5)


def test_symbols_and_axis_labels():
    assert unit_symbol("hour") == "h"
    assert unit_symbol("year") == "yr"
    assert unit_symbol("cycle") == "cycles"
    assert axis_label(None) == "Time"
    assert axis_label("h") == "Time (h)"
    assert axis_label("hour") == "Time (hour)"
    assert axis_label("km") == "Distance (km)"
    assert axis_label("cycle") == "Usage (cycle)"
    assert axis_label("pump starts") == "Usage (pump starts)"


def _fits():
    df = pd.DataFrame({"x": X, "z": Z})
    return {
        "Weibull.fit": lambda u: sp.Weibull.fit(X, unit=u),
        "Weibull.fit_from_df": lambda u: sp.Weibull.fit_from_df(
            df, x_col="x", unit=u
        ),
        "Weibull.from_params": lambda u: sp.Weibull.from_params(
            [1000, 2], unit=u
        ),
        "KaplanMeier.fit": lambda u: sp.KaplanMeier.fit(X, unit=u),
        "WeibullPH.fit": lambda u: sp.WeibullPH.fit(X, Z=Z, unit=u),
        "CoxPH.fit_from_df": lambda u: sp.CoxPH.fit_from_df(
            df, x_col="x", Z_cols=["z"], unit=u
        ),
        "MixtureModel.from_params": lambda u: sp.MixtureModel.from_params(
            sp.Weibull, [[100, 2], [1000, 3]], w=[0.5, 0.5], unit=u
        ),
        "CrowAMSAA.fit": lambda u: recurrent.CrowAMSAA.fit(
            np.cumsum(X), unit=u
        ),
        "NonParametricCounting.fit": (
            lambda u: recurrent.NonParametricCounting.fit(np.cumsum(X), unit=u)
        ),
        "HPP.from_params": lambda u: recurrent.HPP.from_params([0.01], unit=u),
    }


@pytest.mark.parametrize("name", list(_fits()))
def test_units_kept_shown_and_saved(name):
    make = _fits()[name]
    model = make("hrs")
    assert model.unit == "hrs"
    assert "\nUnit" in repr(model) and ": hrs" in repr(model)
    saved = model.to_dict()
    assert saved["unit"] == "hrs"
    assert sp.from_dict(saved).unit == "hrs"
    assert type(model).from_dict(saved).unit == "hrs"
    # None is unknown, and leaves everything as it was
    plain = make(None)
    assert plain.unit is None
    assert "unit" not in plain.to_dict()
    assert "Unit" not in repr(plain)


def test_every_fit_shows_units_in_its_signature():
    for fn in (
        sp.Weibull.fit,
        sp.Weibull.fit_from_df,
        sp.Weibull.from_params,
        sp.KaplanMeier.fit,
        sp.CoxPH.fit,
        sp.WeibullAFT.fit,
        recurrent.CrowAMSAA.fit,
    ):
        assert "unit" in inspect.signature(fn).parameters


def test_repr_units_line_is_aligned():
    lines = repr(sp.Weibull.from_params([10, 2], unit="h")).split("\n")
    units = next(line for line in lines if line.startswith("Unit"))
    first = lines[2]
    assert units.index(":") == first.index(":")


def test_with_params_keeps_the_units():
    model = sp.Weibull.from_params([10, 2], unit="km")
    assert model.with_params([5, 2]).unit == "km"
    mixture = sp.MixtureModel.from_params(
        sp.Weibull, [[10, 2], [50, 3]], w=[0.5, 0.5], unit="km"
    )
    assert mixture.with_params([[10, 2], [50, 3]]).unit == "km"


def test_a_bad_serialised_unit_is_refused():
    saved = sp.Weibull.from_params([10, 2]).to_dict()
    saved["unit"] = 3
    with pytest.raises(ValueError, match="'unit' must be text"):
        sp.from_dict(saved)


@pytest.mark.parametrize(
    "make, label",
    [
        (lambda: sp.Weibull.fit(X, unit="hours"), "Time (hours)"),
        (lambda: sp.KaplanMeier.fit(X, unit="km"), "Distance (km)"),
        (lambda: sp.WeibullPH.fit(X, Z=Z, unit="years"), "Time (years)"),
        (
            lambda: recurrent.CrowAMSAA.fit(np.cumsum(X), unit="cycles"),
            "Usage (cycles)",
        ),
        (
            lambda: recurrent.NonParametricCounting.fit(
                np.cumsum(X), unit="pump starts"
            ),
            "Usage (pump starts)",
        ),
    ],
)
def test_plots_label_the_time_axis_with_the_unit(make, label):
    fig, ax = plt.subplots()
    try:
        assert make().plot(ax=ax).get_xlabel() == label
        assert ax.get_ylabel()
    finally:
        plt.close(fig)


def test_a_label_already_on_the_axes_is_kept():
    fig, ax = plt.subplots()
    try:
        ax.set_xlabel("Operating time")
        recurrent.CrowAMSAA.fit(np.cumsum(X), unit="hours").plot(ax=ax)
        assert ax.get_xlabel() == "Operating time"
    finally:
        plt.close(fig)


def test_single_probability_plot_ticks_do_not_crowd():
    fig, ax = plt.subplots()
    try:
        sp.Weibull.fit(X).plot(ax=ax)
        ticks = np.log10([t for t in ax.get_xticks() if t > 0])
        lo, hi = np.log10(ax.get_xlim())
        assert np.all(np.diff(ticks) >= (hi - lo) / 14 - 1e-12)
    finally:
        plt.close(fig)


def test_takes_unit_leaves_a_method_with_its_own_units_alone():
    def fit(self, x, unit="seconds"):
        return unit

    assert takes_unit(fit) is fit


def test_import_surpyval_does_not_load_pint():
    code = (
        "import sys, surpyval, surpyval.recurrent, surpyval.degradation; "
        "print('pint' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.stdout.strip() == "False"


def test_degradation_keeps_its_units_ids_beside_the_unit():
    # A degradation model's ``units`` are the units measured (their ids);
    # the label is ``unit``, so the two do not collide.
    from surpyval.degradation import DegradationAnalysis

    x = np.tile(np.arange(100, 1100, 100), 4)
    i = np.repeat([1, 2, 3, 4], 10)
    y = 10 + np.repeat([0.31, 0.28, 0.44, 0.37], 10) * x
    model = DegradationAnalysis.fit(x, y, i, threshold=150, unit="hours")
    assert list(model.units) == [1, 2, 3, 4]
    assert model.unit == "hours"
    restored = sp.from_dict(model.to_dict())
    assert list(restored.units) == [1, 2, 3, 4]
    assert restored.unit == "hours"
    assert "Unit " in repr(model)


# -- #945: units on the data classes, carried to fits -------------------


def _hours():
    return [
        sp.ItemLifetime(f"P{k}", observed=v, unit="hrs")
        for k, v in enumerate([3, 5, 8, 11])
    ] + [sp.ItemLifetime("B", right_censored=9, unit="hours")]


def test_issue_945_records_and_datasets_carry_a_unit():
    item = sp.ItemLifetime("A", observed=5, unit="hrs")
    assert item.unit == "hrs" and "unit='hrs'" in repr(item)
    assert item.clone("B").unit == "hrs" and item.extend  # kept by clone
    assert sp.ItemLifetime.from_dict(item.to_dict()).unit == "hrs"
    # "hrs" and "hours" are one unit (pint): the first record's label
    data = sp.SurpyvalData.from_items(_hours())
    assert data.unit == "hrs"
    assert "ages in hrs" in repr(data) and "ages in hrs" in data.summary()
    for other in (
        data.censor_at(6),
        data.enter_at(1),
        data.without("truncation"),
        data.select(ids="B"),
        data[0:2],
        sp.SurpyvalData.from_json(data.to_json()),
        sp.SurpyvalData.from_dict(data.to_dict()),
    ):
        assert other.unit == "hrs"
    assert all(i.unit == "hrs" for i in data.to_items())
    assert sp.SurpyvalData([1, 2, 3], unit="km").unit == "km"
    assert sp.SurpyvalData.from_observations(observed=[1], unit="cycles").unit
    assert sp.SurpyvalData.from_items(_hours()[:2], unit="hour").unit == "hour"
    with pytest.raises(ValueError, match="in different units: item 'P0' in"):
        sp.SurpyvalData.from_items(
            [_hours()[0], sp.ItemLifetime("B", right_censored=9, unit="days")]
        )
    with pytest.raises(ValueError, match="unit='days' was given, but the"):
        sp.SurpyvalData.from_items(_hours(), unit="days")
    with pytest.raises(ValueError, match="'unit' is the name of the unit"):
        sp.ItemLifetime("A", observed=5, unit=3)


def test_issue_945_fit_data_takes_the_units_unit():
    data = sp.SurpyvalData.from_items(_hours())
    assert sp.Weibull.fit(data).unit == "hrs"
    assert sp.KaplanMeier.fit(data).unit == "hrs"
    # The same unit typed another way at the fit: the fit's label
    assert sp.Weibull.fit(data, unit="hours").unit == "hours"
    assert sp.Weibull.fit_from_surpyval_data(data).unit == "hrs"
    assert sp.fit_best(data).unit == "hrs"
    with pytest.raises(ValueError, match="ages are in 'hrs', but unit='days'"):
        sp.Weibull.fit(data, unit="days")
    with_Z = sp.SurpyvalData(
        [1, 2, 3, 4, 5, 6], Z=[[1], [0], [1], [0], [1], [0]], unit="km"
    )
    assert sp.WeibullPH.fit(with_Z).unit == "km"
    assert sp.CoxPH.fit(with_Z).unit == "km"
    assert sp.Weibull.fit(sp.SurpyvalData([1, 2, 3, 4])).unit is None


def test_issue_945_histories_carry_a_unit():
    data = sp.EventHistoryData.from_items(
        [
            sp.ItemHistory("M1", [1, 2], right_censored=3, unit="days"),
            sp.ItemHistory("M2", [1.5], right_censored=3, unit="day"),
        ]
    )
    assert data.unit == "days" and "ages in days" in repr(data)
    for other in (
        data.censor_at(2.5),
        data.select(ids="M1"),
        data[0:2],
        data.to_first_events(),
        data.to_gaps(),
        sp.EventHistoryData.from_json(data.to_json()),
    ):
        assert other.unit == "days"
    assert data.item("M1").unit == "days"
    assert sp.recurrent.CrowAMSAA.fit(data).unit == "days"
    assert sp.recurrent.NonParametricCounting.fit(data).unit == "days"
    assert sp.handle_xicn([1, 2, 3], unit="h").unit == "h"
    assert (
        sp.EventHistoryData.from_histories(
            observed={"A": [1]}, right_censored={"A": 2}, unit="km"
        ).unit
        == "km"
    )
    with pytest.raises(ValueError, match="in different units"):
        sp.EventHistoryData.from_items(
            [
                sp.ItemHistory("M1", [1], right_censored=3, unit="days"),
                sp.ItemHistory("M2", [1], right_censored=3, unit="hours"),
            ]
        )
    with pytest.raises(ValueError, match="but unit='hours' was given"):
        sp.recurrent.CrowAMSAA.fit(data, unit="hours")


# -- labels kept as typed (decided 13 Nov 2026) ------------------------


@pytest.mark.parametrize("typed", ["mm/s", "°C", "bar", "h"])
def test_labels_print_and_round_trip_as_typed(typed):
    model = sp.Weibull.from_params([10, 2], unit=typed)
    assert model.unit == typed
    assert f": {typed}\n" in repr(model) + "\n"
    assert sp.from_dict(model.to_dict()).unit == typed
    data = sp.SurpyvalData([1.0, 2.0, 3.0], unit=typed)
    assert f"ages in {typed}" in repr(data)
    assert sp.SurpyvalData.from_json(data.to_json()).unit == typed
    assert sp.SurpyvalData.from_dict(data.to_dict()).unit == typed


def test_axis_shows_the_label_as_typed():
    fig, ax = plt.subplots()
    try:
        sp.Weibull.fit(X, unit="hrs").plot(ax=ax)
        assert ax.get_xlabel() == "Time (hrs)"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "a, b, same",
    [
        ("h", "hours", True),
        ("hrs", "Hour", True),
        ("mm/s", "millimeter / second", True),
        ("°C", "degree_Celsius", True),
        ("bar", "bar", True),
        ("cycles", "cycle", True),
        ("pump starts", "pump  starts", True),
        ("h", "km", False),
        ("mm/s", "m/s", False),
        ("pump starts", "pump trips", False),
        (None, None, True),
        (None, "h", False),
    ],
)
def test_same_unit_reads_labels_with_pint(a, b, same):
    assert same_unit(a, b) is same
    assert same_unit(b, a) is same


def test_unit_helpers_are_public_in_surpyval_utils():
    # Tools that combine models (RePyability) check units with these.
    import surpyval.utils as su

    assert {"normalise_unit", "same_unit"} <= set(su.__all__)
    assert su.same_unit is same_unit
    assert su.normalise_unit is normalise_unit
    assert su.same_unit("hrs", "hours") and not su.same_unit("h", "days")
    assert su.same_unit("cycles", "cycle")
    assert not su.same_unit("cycles", "starts")
    assert su.normalise_unit(" hrs ") == "hrs"


def test_an_unknown_unit_is_kept_as_before():
    # pint does not know it: kept as a label, compared as text
    model = sp.Weibull.from_params([10, 2], unit="landing gear swaps")
    assert model.unit == "landing gear swaps"
    data = sp.SurpyvalData([1.0, 2.0], unit="landing gear swaps")
    assert sp.Weibull.fit(data, unit="landing gear swaps").unit == (
        "landing gear swaps"
    )
    with pytest.raises(ValueError, match="but unit='gear swaps'"):
        sp.Weibull.fit(data, unit="gear swaps")


def test_a_file_saved_with_pints_name_still_loads_and_agrees():
    saved = sp.SurpyvalData([1.0, 2.0, 3.0], unit="hours").to_dict()
    saved["unit"] = "hour"  # as files saved before 13 Nov 2026 hold it
    data = sp.SurpyvalData.from_dict(saved)
    assert data.unit == "hour"
    # Compared with pint: the fit's "hrs" agrees, and is the model's label
    assert sp.Weibull.fit(data, unit="hrs").unit == "hrs"


def test_records_in_one_unit_typed_differently_make_one_dataset():
    data = sp.SurpyvalData.from_items(
        [
            sp.ItemLifetime("A", observed=3, unit="h"),
            sp.ItemLifetime("B", observed=5, unit="hours"),
        ]
    )
    assert data.unit == "h"
    given = sp.SurpyvalData.from_items(
        [sp.ItemLifetime("A", observed=3, unit="h")], unit="hrs"
    )
    assert given.unit == "hrs"


def test_combined_bayes_data_agree_on_one_unit_typed_two_ways():
    from surpyval.bayes.model import _combine

    a = sp.SurpyvalData(x=[1.0, 2.0], unit="h")
    b = sp.SurpyvalData(x=[3.0], unit="hours")
    assert _combine(a, b).unit == "h"
