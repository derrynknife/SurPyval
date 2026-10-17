"""surpyval.at_risk_table: the number-at-risk table of plot_nonparametric
as data, counted without loading a plotting library."""

import subprocess
import sys

import numpy as np
import pytest

import surpyval as sp

X = [2, 3, 3, 5, 7, 8, 8, 11, 13, 15]
C = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]


def test_it_loads_no_plotting_library():
    code = (
        "import sys, surpyval\n"
        "km = surpyval.KaplanMeier.fit([1, 2, 3], c=[0, 1, 0])\n"
        "surpyval.at_risk_table(km, times=[0, 2], rows='all')\n"
        "print('matplotlib' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


def test_hand_count():
    km = sp.KaplanMeier.fit(X, c=C)
    table = sp.at_risk_table(km, times=[0, 5, 10, 15], rows="all")
    assert table["rows"] == ["At risk", "Censored", "Events"]
    np.testing.assert_array_equal(table["times"], [0, 5, 10, 15])
    counts = table["counts"]["Kaplan-Meier"]
    np.testing.assert_array_equal(counts["At risk"], [10, 7, 3, 1])
    np.testing.assert_array_equal(counts["Censored"], [0, 1, 2, 4])
    np.testing.assert_array_equal(counts["Events"], [0, 3, 5, 6])
    interval = sp.at_risk_table(
        km, times=[0, 5, 10, 15], rows="all", counts="interval"
    )["counts"]["Kaplan-Meier"]
    np.testing.assert_array_equal(interval["Censored"], [0, 1, 1, 2])
    np.testing.assert_array_equal(interval["Events"], [0, 3, 2, 1])
    np.testing.assert_array_equal(interval["At risk"], counts["At risk"])


def test_the_counts_are_the_plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = sp.KaplanMeier.fit(X, c=C)
    b = sp.NelsonAalen.fit([4, 6, 9, 10, 12, 14], c=[0, 1, 0, 0, 1, 0])
    times = [0, 4, 8, 12]
    out = sp.plot_nonparametric(
        {"A": a, "B": b}, at_risk_times=times, at_risk_rows="all"
    )
    table = sp.at_risk_table({"A": a, "B": b}, times=times, rows="all")
    np.testing.assert_array_equal(table["times"], out.times)
    for label in ("A", "B"):
        for row in table["rows"]:
            np.testing.assert_array_equal(
                table["counts"][label][row], out.counts[label][row]
            )
    plt.close(out.fig)


def test_default_times_and_labels():
    a = sp.KaplanMeier.fit(X, c=C)
    b = sp.KaplanMeier.fit([4, 6, 9], c=[0, 1, 0])
    table = sp.at_risk_table([a, b])
    # the start of the axis and every time in the estimates
    expected = np.unique(np.concatenate([[0.0], a.x, b.x]))
    np.testing.assert_array_equal(table["times"], expected)
    assert list(table["counts"]) == ["Group 1", "Group 2"]
    named = sp.at_risk_table([a, b], times=[5], labels=["x", "y"])
    assert list(named["counts"]) == ["x", "y"]
    assert named["counts"]["x"]["At risk"][0] == 7


def test_interval_censored_rows():
    x = [[1, 3], [2, 5], 4, 6, [0, 2]]
    c = [2, 2, 0, 1, 2]
    tb = sp.Turnbull.fit(x, c=c)
    table = sp.at_risk_table(tb, times=[0, 2.5, 7], rows="all")
    assert table["rows"][-1] == "In interval"
    counts = table["counts"]["Turnbull"]
    # at 2.5 the intervals (1, 3] and (2, 5] span the time
    assert counts["In interval"][1] == 2
    assert counts["At risk"][2] == 0


def test_a_model_without_its_data_counts_from_its_ladder():
    km = sp.KaplanMeier.fit(X, c=C)
    ladder = sp.KaplanMeier.from_xrd(km.x, km.r, km.d)
    table = sp.at_risk_table(ladder, times=[0, 5, 10], rows=["At risk"])
    np.testing.assert_array_equal(
        table["counts"]["Kaplan-Meier"]["At risk"], [10, 7, 3]
    )
    with pytest.raises(ValueError, match="does not hold"):
        sp.at_risk_table(ladder, times=[0, 5], rows=["Censored"])


def test_refusals_say_what_to_do():
    km = sp.KaplanMeier.fit(X, c=C)
    with pytest.raises(ValueError, match="'counts'"):
        sp.at_risk_table(km, counts="daily")
    with pytest.raises(ValueError, match="'rows'"):
        sp.at_risk_table(km, rows="Survivors")
    with pytest.raises(ValueError, match="'rows' is empty"):
        sp.at_risk_table(km, rows=[])
    with pytest.raises(ValueError, match="finite times"):
        sp.at_risk_table(km, times=[0, np.inf])
    with pytest.raises(TypeError, match="fit it first"):
        sp.at_risk_table(sp.KaplanMeier)
    with pytest.raises(ValueError, match="'models' is empty"):
        sp.at_risk_table([])
