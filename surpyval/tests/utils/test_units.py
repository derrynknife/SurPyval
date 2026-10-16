"""The ``unit=`` label on models (#845): normalised by pint, kept on the
model, shown in its repr and on its plots, and saved by ``to_dict``."""

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
        ("hrs", "hour"),
        ("hours", "hour"),
        ("Hours", "hour"),
        ("h", "hour"),
        ("km", "kilometer"),
        ("kilometres", "kilometer"),
        ("cycles", "cycle"),
        ("Cycles", "cycle"),
        ("starts", "start"),
        ("years", "year"),
        ("kWh", "kilowatt_hour"),
        ("  pump   starts ", "pump starts"),
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
    assert axis_label("hour") == "Time (h)"
    assert axis_label("kilometer") == "Distance (km)"
    assert axis_label("cycle") == "Usage (cycles)"
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
    assert model.unit == "hour"
    assert "\nUnit" in repr(model) and ": hour" in repr(model)
    saved = model.to_dict()
    assert saved["unit"] == "hour"
    assert sp.from_dict(saved).unit == "hour"
    assert type(model).from_dict(saved).unit == "hour"
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
    assert model.with_params([5, 2]).unit == "kilometer"
    mixture = sp.MixtureModel.from_params(
        sp.Weibull, [[10, 2], [50, 3]], w=[0.5, 0.5], unit="km"
    )
    assert mixture.with_params([[10, 2], [50, 3]]).unit == "kilometer"


def test_a_bad_serialised_unit_is_refused():
    saved = sp.Weibull.from_params([10, 2]).to_dict()
    saved["unit"] = 3
    with pytest.raises(ValueError, match="'unit' must be text"):
        sp.from_dict(saved)


@pytest.mark.parametrize(
    "make, label",
    [
        (lambda: sp.Weibull.fit(X, unit="hours"), "Time (h)"),
        (lambda: sp.KaplanMeier.fit(X, unit="km"), "Distance (km)"),
        (lambda: sp.WeibullPH.fit(X, Z=Z, unit="years"), "Time (yr)"),
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
    assert model.unit == "hour"
    restored = sp.from_dict(model.to_dict())
    assert list(restored.units) == [1, 2, 3, 4]
    assert restored.unit == "hour"
    assert "Unit " in repr(model)


# -- #945: units on the data classes, carried to fits -------------------


def _hours():
    return [
        sp.ItemLifetime(f"P{k}", observed=v, unit="hrs")
        for k, v in enumerate([3, 5, 8, 11])
    ] + [sp.ItemLifetime("B", right_censored=9, unit="hours")]


def test_issue_945_records_and_datasets_carry_a_unit():
    item = sp.ItemLifetime("A", observed=5, unit="hrs")
    assert item.unit == "hour" and "unit='hour'" in repr(item)
    assert item.clone("B").unit == "hour" and item.extend  # kept by clone
    assert sp.ItemLifetime.from_dict(item.to_dict()).unit == "hour"
    data = sp.SurpyvalData.from_items(_hours())
    assert data.unit == "hour"
    assert "ages in hour" in repr(data) and "ages in hour" in data.summary()
    for other in (
        data.censor_at(6),
        data.enter_at(1),
        data.without("truncation"),
        data.select(ids="B"),
        data[0:2],
        sp.SurpyvalData.from_json(data.to_json()),
        sp.SurpyvalData.from_dict(data.to_dict()),
    ):
        assert other.unit == "hour"
    assert all(i.unit == "hour" for i in data.to_items())
    assert sp.SurpyvalData([1, 2, 3], unit="km").unit == "kilometer"
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
    assert sp.Weibull.fit(data).unit == "hour"
    assert sp.KaplanMeier.fit(data).unit == "hour"
    assert sp.Weibull.fit(data, unit="hours").unit == "hour"
    assert sp.Weibull.fit_from_surpyval_data(data).unit == "hour"
    assert sp.fit_best(data).unit == "hour"
    with pytest.raises(
        ValueError, match="ages are in 'hour', but unit='days'"
    ):
        sp.Weibull.fit(data, unit="days")
    with_Z = sp.SurpyvalData(
        [1, 2, 3, 4, 5, 6], Z=[[1], [0], [1], [0], [1], [0]], unit="km"
    )
    assert sp.WeibullPH.fit(with_Z).unit == "kilometer"
    assert sp.CoxPH.fit(with_Z).unit == "kilometer"
    assert sp.Weibull.fit(sp.SurpyvalData([1, 2, 3, 4])).unit is None


def test_issue_945_histories_carry_a_unit():
    data = sp.EventHistoryData.from_items(
        [
            sp.ItemHistory("M1", [1, 2], right_censored=3, unit="days"),
            sp.ItemHistory("M2", [1.5], right_censored=3, unit="day"),
        ]
    )
    assert data.unit == "day" and "ages in day" in repr(data)
    for other in (
        data.censor_at(2.5),
        data.select(ids="M1"),
        data[0:2],
        data.to_first_events(),
        data.to_gaps(),
        sp.EventHistoryData.from_json(data.to_json()),
    ):
        assert other.unit == "day"
    assert data.item("M1").unit == "day"
    assert sp.recurrent.CrowAMSAA.fit(data).unit == "day"
    assert sp.recurrent.NonParametricCounting.fit(data).unit == "day"
    assert sp.handle_xicn([1, 2, 3], unit="h").unit == "hour"
    assert (
        sp.EventHistoryData.from_histories(
            observed={"A": [1]}, right_censored={"A": 2}, unit="km"
        ).unit
        == "kilometer"
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
