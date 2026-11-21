"""SurpyvalData.plot_lifetimes: the data drawn one line per unit."""

import matplotlib
import numpy as np
import pytest

from surpyval import SurpyvalData

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _lines(ax):
    # the unit lines are the two-point grey lines drawn first
    return [
        ln
        for ln in ax.lines
        if len(ln.get_xdata()) == 2
        and ln.get_ydata()[0] == ln.get_ydata()[1]
        and ln.get_marker() in ("None", "", None)
    ]


def _markers(ax):
    return [ln.get_marker() for ln in ax.lines if ln.get_marker() != "None"]


def test_every_kind_of_observation_has_its_mark():
    data = SurpyvalData.from_observations(
        observed=[1, 2],
        right_censored=[5],
        left_censored=[3],
        interval_censored=[[1, 4]],
    )
    ax = data.plot_lifetimes()
    marks = _markers(ax)
    assert marks.count("o") == 2
    assert ">" in marks and "<" in marks and "|" in marks
    assert len(ax.get_yticks()) == 5
    assert ax.get_legend() is not None


def test_late_entry_starts_the_line_there():
    ax = SurpyvalData([3, 7, 9], c=[0, 1, 0], tl=[0, 2, 1]).plot_lifetimes(
        sort="data"
    )
    starts = sorted(ln.get_xdata()[0] for ln in _lines(ax))
    assert starts == [0, 1, 2]


def test_counts_are_labelled():
    ax = SurpyvalData([3, 7], c=[0, 1], n=[1, 40]).plot_lifetimes()
    assert [t.get_text() for t in ax.texts] == ["40 x"]


def test_right_truncation_is_marked_where_finite():
    ax = SurpyvalData([1, 2], tr=[5, np.inf]).plot_lifetimes()
    assert [m for m in _markers(ax)].count("|") == 1


def test_rows_of_one_item_share_a_line_with_each_event_marked():
    data = SurpyvalData(x=[2, 5, 9, 4], i=["P1", "P1", "P1", "P2"])
    ax = data.plot_lifetimes()
    assert len(_lines(ax)) == 2
    assert _markers(ax).count("o") == 4
    assert sorted(t.get_text() for t in ax.get_yticklabels()) == ["P1", "P2"]


def test_event_types_are_coloured_with_a_legend():
    data = SurpyvalData([1, 2, 3], e=["wear", "shock", "wear"])
    ax = data.plot_lifetimes()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "wear" in labels and "shock" in labels
    colours = {ln.get_color() for ln in ax.lines if ln.get_marker() == "o"}
    assert len(colours) == 2


def test_max_rows_says_how_many_were_left_out():
    data = SurpyvalData(np.arange(1, 31))
    ax = data.plot_lifetimes(max_rows=10)
    assert len(_lines(ax)) == 10
    assert any("20 left out" in t.get_text() for t in ax.texts)
    assert not any(
        "left out" in t.get_text()
        for t in data.plot_lifetimes(max_rows=None).texts
    )


def test_draws_on_a_given_axes_and_orders_by_end():
    fig, ax = plt.subplots()
    out = SurpyvalData([9, 3, 6]).plot_lifetimes(ax=ax, sort="end")
    assert out is ax
    ends = [
        ln.get_xdata()[1]
        for ln in sorted(_lines(ax), key=lambda ln: ln.get_ydata()[0])
    ]
    assert ends == [3, 6, 9]


def test_refusals_say_what_to_do():
    data = SurpyvalData([1, 2])
    with pytest.raises(ValueError, match="Pass one of those"):
        data.plot_lifetimes(sort="up")
    with pytest.raises(ValueError, match="None for every line"):
        data.plot_lifetimes(max_rows=0)
