"""surpyval.plot_nonparametric: the curves, bounds, ticks and quantile
lines it draws are the models' own, and its number-at-risk table counts
what a hand count gives. Asserts on the artists' data, not on images."""

import matplotlib
import numpy as np
import pytest

import surpyval as sp
from surpyval import plotting

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

X = [2, 3, 3, 5, 7, 8, 8, 11, 13, 15]
C = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _curve(out, label):
    line = out.lines[label]
    return np.asarray(line.get_xdata()), np.asarray(line.get_ydata())


def _table(out):
    """The table's counts, row by row from the top: [(x, text), ...] of
    the texts SurPyval tags as cells."""
    rows = {}
    for t in out.ax.texts:
        if t.get_gid() != "at-risk-cell":
            continue
        x = t.get_position()[0]
        y = round(t.get_transform().transform((x, 0))[1], 3)
        rows.setdefault(y, []).append((x, t.get_text()))
    return [sorted(cells) for _, cells in sorted(rows.items(), reverse=True)]


def _row_texts(out):
    return [
        t.get_text() for t in out.ax.texts if t.get_gid() == "at-risk-label"
    ]


# -- the curve -----------------------------------------------------------


@pytest.mark.parametrize("on", ["sf", "ff", "Hf"])
def test_km_steps_are_the_models_own(on):
    km = sp.KaplanMeier.fit(X, c=C)
    out = sp.plot_nonparametric(km, on=on, at_risk=False)
    x, y = _curve(out, "Kaplan-Meier")
    # every step of the ladder is drawn, from 0 to the last value
    assert set(km.x) <= set(x)
    assert x[0] == 0 and x[-1] == 15
    assert out.lines["Kaplan-Meier"].get_drawstyle() == "steps-post"
    np.testing.assert_allclose(y, getattr(km, on)(x))


def test_step_values_match_a_hand_computed_km():
    km = sp.KaplanMeier.fit(X, c=C)
    out = sp.plot_nonparametric(km, at_risk=False, ci=False)
    x, y = _curve(out, "Kaplan-Meier")
    # 10 at risk; deaths at 2 (1/10), 3 (1/9), 5 (1/7), 8 (2/5), 13 (1/2)
    expected = {
        0: 1.0,
        2: 0.9,
        3: 0.9 * 8 / 9,
        5: 0.9 * 8 / 9 * 6 / 7,
        8: 0.9 * 8 / 9 * 6 / 7 * 3 / 5,
        13: 0.9 * 8 / 9 * 6 / 7 * 3 / 5 / 2,
    }
    for t, v in expected.items():
        assert y[list(x).index(t)] == pytest.approx(v)


@pytest.mark.parametrize("on", ["sf", "ff", "Hf"])
def test_pointwise_bounds_are_the_models_cb(on):
    km = sp.KaplanMeier.fit(X, c=C)
    out = sp.plot_nonparametric(
        km, on=on, ci="pointwise", ci_style="lines", at_risk=False
    )
    dashed = [ln for ln in out.ax.lines if ln.get_linestyle() == "--"]
    assert len(dashed) == 2
    q = np.asarray(dashed[0].get_xdata())
    cb = km.cb(q, on)
    np.testing.assert_allclose(dashed[0].get_ydata(), cb[:, 0])
    np.testing.assert_allclose(dashed[1].get_ydata(), cb[:, 1])
    # not past the last observed value, where the bounds are NaN
    assert q.max() == km.x[-1]


def test_shaded_bounds_span_the_models_cb():
    km = sp.KaplanMeier.fit(X, c=C)
    out = sp.plot_nonparametric(km, alpha_ci=0.1, at_risk=False)
    (fill,) = out.ax.collections
    ys = np.unique(np.round(fill.get_paths()[0].vertices[:, 1], 10))
    cb = km.cb(km.x, alpha_ci=0.1)
    expected = np.unique(np.round(np.concatenate([cb.ravel(), [1.0]]), 10))
    assert set(ys) == set(expected)


def test_bound_type_and_one_sided_bound_are_passed_to_cb():
    km = sp.KaplanMeier.fit(X, c=C)
    out = sp.plot_nonparametric(
        km, bound="lower", bound_type="normal", at_risk=False
    )
    (dashed,) = (ln for ln in out.ax.lines if ln.get_linestyle() == "--")
    q = np.asarray(dashed.get_xdata())
    np.testing.assert_allclose(
        dashed.get_ydata(), km.cb(q, bound="lower", bound_type="normal")
    )


@pytest.mark.parametrize("on", ["sf", "ff", "Hf"])
def test_band_is_the_models_band(on):
    x = sp.Weibull.random(60, 10, 1.5, random_state=3)
    km = sp.KaplanMeier.fit(x)
    out = sp.plot_nonparametric(
        km,
        on=on,
        ci="band",
        band_method="nair",
        ci_style="lines",
        at_risk=False,
    )
    lower, upper = (ln for ln in out.ax.lines if ln.get_linestyle() == "--")
    q = np.asarray(lower.get_xdata())
    band = km.band(q, method="nair")  # on sf, arcsine
    if on == "sf":
        exp_lo, exp_hi = band[:, 0], band[:, 1]
    elif on == "ff":
        exp_lo, exp_hi = 1 - band[:, 1], 1 - band[:, 0]
    else:
        exp_lo, exp_hi = -np.log(band[:, 1]), -np.log(band[:, 0])
    np.testing.assert_allclose(lower.get_ydata(), exp_lo)
    np.testing.assert_allclose(upper.get_ydata(), exp_hi)


def test_censor_ticks_sit_on_the_curve():
    km = sp.KaplanMeier.fit(X, c=C)
    out = sp.plot_nonparametric(km, at_risk=False)
    (ticks,) = (ln for ln in out.ax.lines if ln.get_marker() == "|")
    np.testing.assert_array_equal(ticks.get_xdata(), [3, 7, 11, 15])
    np.testing.assert_allclose(ticks.get_ydata(), km.sf([3, 7, 11, 15]))
    out = sp.plot_nonparametric(km, at_risk=False, show_censors=False)
    assert not [ln for ln in out.ax.lines if ln.get_marker() == "|"]


@pytest.mark.parametrize("fitter", ["NelsonAalen", "FlemingHarrington"])
def test_every_estimator_draws_its_own_curve(fitter):
    model = getattr(sp, fitter).fit(X, c=C)
    out = sp.plot_nonparametric(model, at_risk=False)
    x, y = _curve(out, model.model)
    np.testing.assert_allclose(y, model.sf(x))
    assert out.ax.get_title() == f"{model.model} estimate"


def test_smoothed_hazard():
    km = sp.KaplanMeier.fit(X, c=C)
    out = sp.plot_nonparametric(km, on="hf", hf_bandwidth=3, at_risk=False)
    x, y = _curve(out, "Kaplan-Meier")
    np.testing.assert_allclose(y, km.smoothed_hf(x, 3))
    assert out.ax.get_ylabel() == "Hazard rate (smoothed)"
    assert not out.ax.collections  # no bounds on the rate


# -- quantiles -----------------------------------------------------------


def test_median_lines_and_values():
    a = sp.KaplanMeier.fit(X, c=C)
    b = sp.KaplanMeier.fit([1, 2, 3, 4], c=[0, 1, 1, 1])  # never reaches 0.5
    out = sp.plot_nonparametric(
        {"a": a, "b": b}, quantiles=[0.5, 0.25], at_risk=False
    )
    assert out.quantiles["a"][0.5] == a.qf(0.5) == 8
    assert out.quantiles["a"][0.25] == a.qf(0.25)
    assert np.isnan(out.quantiles["b"][0.5])
    dotted = [ln for ln in out.ax.lines if ln.get_linestyle() == ":"]
    # a horizontal and a vertical line for each quantile reached: a's two,
    # and b's 0.25 (its survival drops to 0.75 at 1)
    assert len(dotted) == 6
    assert out.quantiles["b"][0.25] == 1
    verticals = [ln for ln in dotted if ln.get_xdata()[0] == 8]
    (vert,) = (ln for ln in verticals if ln.get_xdata()[1] == 8)
    np.testing.assert_allclose(vert.get_ydata(), [0, 0.5])
    assert "8" in [t.get_text() for t in out.ax.texts]


def test_quantile_level_on_ff_and_Hf():
    km = sp.KaplanMeier.fit(X, c=C)
    for on, level in (("ff", 0.5), ("Hf", np.log(2))):
        out = sp.plot_nonparametric(km, on=on, quantiles=0.5, at_risk=False)
        horiz = [
            ln
            for ln in out.ax.lines
            if ln.get_linestyle() == ":" and ln.get_ydata()[0] > 0
        ]
        assert horiz[0].get_ydata()[0] == pytest.approx(level)


# -- the number-at-risk table ---------------------------------------------


def test_at_risk_matches_a_hand_count():
    km = sp.KaplanMeier.fit(X, c=C)
    out = sp.plot_nonparametric(
        km, at_risk_times=[0, 3, 8, 10, 15, 20], at_risk_rows="all"
    )
    counts = out.counts["Kaplan-Meier"]
    # at risk at t: time >= t
    np.testing.assert_array_equal(counts["At risk"], [10, 9, 5, 3, 1, 0])
    # censored at or before t: 3, 7, 11, 15
    np.testing.assert_array_equal(counts["Censored"], [0, 1, 2, 2, 4, 4])
    # events at or before t: 2, 3, 5, 8, 8, 13
    np.testing.assert_array_equal(counts["Events"], [0, 2, 5, 5, 6, 6])
    assert "In interval" not in counts


def test_interval_counts():
    km = sp.KaplanMeier.fit(X, c=C)
    out = sp.plot_nonparametric(
        km,
        at_risk_times=[0, 3, 8, 10, 15],
        at_risk_rows=["Censored", "Events"],
        counts="interval",
    )
    counts = out.counts["Kaplan-Meier"]
    np.testing.assert_array_equal(counts["Censored"], [0, 1, 1, 0, 2])
    np.testing.assert_array_equal(counts["Events"], [0, 2, 3, 0, 1])


def test_at_risk_equals_the_estimators_risk_sets():
    x = sp.Weibull.random(80, 10, 1.3, random_state=7)
    c = (np.random.default_rng(7).uniform(size=80) < 0.3).astype(int)
    tl = np.where(np.arange(80) % 3 == 0, x * 0.4, 0.0)
    km = sp.KaplanMeier.fit(x, c=c, tl=tl)
    out = sp.plot_nonparametric(km, at_risk_times=km.x)
    np.testing.assert_array_equal(out.counts["Kaplan-Meier"]["At risk"], km.r)


def test_at_risk_rises_with_left_truncation():
    x = [5, 7, 9, 10, 12]
    tl = [0, 0, 4, 6, 8]
    km = sp.KaplanMeier.fit(x, tl=tl)
    out = sp.plot_nonparametric(km, at_risk_times=[1, 5, 6, 7, 9, 12])
    # entered before t and time >= t
    np.testing.assert_array_equal(
        out.counts["Kaplan-Meier"]["At risk"], [2, 3, 2, 3, 3, 1]
    )


def test_counts_with_n():
    km = sp.KaplanMeier.fit([1, 2, 3], c=[0, 1, 0], n=[4, 3, 2])
    out = sp.plot_nonparametric(
        km, at_risk_times=[0, 2, 3], at_risk_rows="all"
    )
    counts = out.counts["Kaplan-Meier"]
    np.testing.assert_array_equal(counts["At risk"], [9, 5, 2])
    np.testing.assert_array_equal(counts["Censored"], [0, 3, 3])
    np.testing.assert_array_equal(counts["Events"], [0, 4, 6])


def test_table_text_is_drawn_under_its_times():
    a = sp.KaplanMeier.fit(X, c=C)
    b = sp.KaplanMeier.fit([4, 6, 9, 10, 12, 14])
    out = sp.plot_nonparametric({"A": a, "B": b}, at_risk_times=[0, 5, 10])
    texts = _row_texts(out)
    assert "At risk" in texts and "A" in texts and "B" in texts
    assert _table(out) == [
        [(0, "10"), (5, "7"), (10, "3")],
        [(0, "6"), (5, "5"), (10, "3")],
    ]
    # the axis ticks are fixed at the table's times
    np.testing.assert_array_equal(out.ax.get_xticks(), [0, 5, 10])
    # each group's name takes its curve's colour
    (name_a,) = (t for t in out.ax.texts if t.get_text() == "A")
    assert name_a.get_color() == out.lines["A"].get_color()


def test_default_times_are_the_major_ticks():
    km = sp.KaplanMeier.fit(X, c=C)
    out = sp.plot_nonparametric(km)
    lo, hi = out.ax.get_xlim()
    assert lo == 0 and hi == 15
    np.testing.assert_array_equal(out.times, out.ax.get_xticks())
    assert out.times.min() >= lo and out.times.max() <= hi
    assert len(out.times) >= 3


def test_log_axis_and_xlim():
    km = sp.KaplanMeier.fit(X, c=C)
    out = sp.plot_nonparametric(km, log_x=True, at_risk_times=[2, 5, 10])
    assert out.ax.get_xscale() == "log"
    x, _ = _curve(out, "Kaplan-Meier")
    assert x.min() == 2  # the smallest positive time, not 0
    out = sp.plot_nonparametric(km, xlim=(0, 30), at_risk_times=[20, 30])
    assert out.ax.get_xlim() == (0, 30)
    np.testing.assert_array_equal(
        out.counts["Kaplan-Meier"]["At risk"], [0, 0]
    )


def test_turnbull_curve_and_counts_on_right_censored_data():
    tb = sp.Turnbull.fit(X, c=C)
    km = sp.KaplanMeier.fit(X, c=C)
    out = sp.plot_nonparametric(
        {"T": tb, "K": km}, at_risk_times=tb.x, at_risk_rows="all"
    )
    x, y = _curve(out, "T")
    np.testing.assert_allclose(y, tb.sf(x))
    for row in ("At risk", "Censored", "Events"):
        np.testing.assert_array_equal(
            out.counts["T"][row], out.counts["K"][row]
        )
    # the observed risk sets, those of the Kaplan-Meier (the Turnbull
    # ladder's r and d are the EM's expected counts, not these)
    at_km = sp.plot_nonparametric(tb, at_risk_times=km.x).counts["Turnbull"]
    np.testing.assert_array_equal(at_km["At risk"], km.r)
    assert "In interval" not in out.counts["T"]


def test_turnbull_counts_on_interval_censored_data():
    xl = [0, 1, 2, 3, 3, 5, 6, 8]
    xr = [2, 3, 4, np.inf, 3, 8, np.inf, 10]
    tb = sp.Turnbull.fit(xl=xl, xr=xr)
    times = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10])
    out = sp.plot_nonparametric(tb, at_risk_times=times, at_risk_rows="all")
    counts = out.counts["Turnbull"]
    lower = np.array(xl, dtype=float)
    upper = np.array(xr, dtype=float)
    right = np.isinf(upper)
    exact = lower == upper
    interval = ~right & ~exact
    for i, t in enumerate(times):
        assert counts["At risk"][i] == (t <= lower).sum()
        assert counts["Events"][i] == (~right & (upper <= t)).sum()
        assert counts["Censored"][i] == (right & (lower <= t)).sum()
        assert (
            counts["In interval"][i]
            == (interval & (lower < t) & (t < upper)).sum()
        )
    # without exact values the rows split the units: each is at risk,
    # in its interval, known to have failed or censored
    xl2 = [0, 1, 2, 3, 5, 6, 8]
    xr2 = [2, 3, 4, np.inf, 8, np.inf, 10]
    tb2 = sp.Turnbull.fit(xl=xl2, xr=xr2)
    t2 = np.array([0.5, 1.5, 2.5, 3.5, 5.5, 7, 9, 11])
    c2 = sp.plot_nonparametric(
        tb2, at_risk_times=t2, at_risk_rows="all"
    ).counts["Turnbull"]
    total = c2["At risk"] + c2["In interval"] + c2["Events"]
    total += c2["Censored"]
    np.testing.assert_array_equal(total, 7)
    # the curve is the model's
    x, y = _curve(out, "Turnbull")
    np.testing.assert_allclose(y, tb.sf(x))


def test_turnbull_left_censored_and_truncated():
    x = [3, 4, 6, 7, 9]
    c = [-1, 0, 1, 0, -1]
    tb = sp.Turnbull.fit(x, c=c, tl=[0, 1, 2, 0, 5])
    out = sp.plot_nonparametric(
        tb, at_risk_times=[1, 2, 4, 6, 9], at_risk_rows="all"
    )
    counts = out.counts["Turnbull"]
    # known at risk: entered before t, exact or right censored at >= t
    np.testing.assert_array_equal(counts["At risk"], [1, 2, 3, 2, 0])
    # left censored rows are in their interval (entry, x) until x
    np.testing.assert_array_equal(counts["In interval"], [1, 1, 0, 1, 0])
    np.testing.assert_array_equal(counts["Events"], [0, 0, 2, 2, 4])


def test_ladder_counts_for_a_model_without_data():
    na = sp.NelsonAalen.from_xrd([1, 2, 3, 4], [10, 8, 5, 2], [2, 1, 1, 1])
    out = sp.plot_nonparametric(
        na, at_risk_times=[0, 1, 1.5, 4, 5], at_risk_rows=["At risk", "Events"]
    )
    counts = out.counts["Nelson-Aalen"]
    np.testing.assert_array_equal(counts["At risk"], [10, 10, 8, 2, 0])
    np.testing.assert_array_equal(counts["Events"], [0, 2, 2, 5, 5])
    assert not [ln for ln in out.ax.lines if ln.get_marker() == "|"]
    with pytest.raises(ValueError, match="does not hold"):
        sp.plot_nonparametric(na, at_risk_rows="all")


def test_restored_model_counts_as_the_original():
    km = sp.KaplanMeier.fit(X, c=C)
    back = sp.from_dict(km.to_dict(with_data=True))
    a = sp.plot_nonparametric(km, at_risk_rows="all")
    b = sp.plot_nonparametric(back, at_risk_rows="all")
    for row in ("At risk", "Censored", "Events"):
        np.testing.assert_array_equal(
            a.counts["Kaplan-Meier"][row], b.counts["Kaplan-Meier"][row]
        )


# -- labels, colours, theme ---------------------------------------------


def test_labels_and_colours():
    a = sp.KaplanMeier.fit(X, c=C)
    b = sp.KaplanMeier.fit([4, 6, 9, 10, 12, 14])
    out = sp.plot_nonparametric([a, b], at_risk=False)
    assert list(out.lines) == ["Group 1", "Group 2"]
    assert out.lines["Group 1"].get_color() == plotting.PALETTE[0]
    assert out.lines["Group 2"].get_color() == plotting.PALETTE[1]
    assert out.ax.get_legend() is not None
    out = sp.plot_nonparametric(
        [a, b], labels=["x", "y"], colors=["k", "r"], at_risk=False
    )
    assert out.lines["x"].get_color() == "k"
    out = sp.plot_nonparametric({"p": a, "q": b}, colors={"q": "g"})
    assert out.lines["q"].get_color() == "g"
    # one model: no legend unless asked
    out = sp.plot_nonparametric(a, at_risk=False)
    assert out.ax.get_legend() is None
    assert out.ax.get_ylabel() == "Survival probability"
    assert out.ax.get_xlabel() == "Time"


def test_labels_titles_and_given_axes():
    fig, ax = plt.subplots()
    km = sp.KaplanMeier.fit(X, c=C)
    out = sp.plot_nonparametric(
        km,
        ax=ax,
        xlabel="Days",
        ylabel="S(t)",
        title="Trial",
        legend=True,
    )
    assert out.ax is ax and out.fig is fig
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == (
        "Days",
        "S(t)",
        "Trial",
    )
    assert ax.get_legend() is not None


def test_theme_off():
    plotting.set_theme(False)
    try:
        km = sp.KaplanMeier.fit(X, c=C)
        out = sp.plot_nonparametric(km, quantiles=0.5)
        x, y = _curve(out, "Kaplan-Meier")
        np.testing.assert_allclose(y, km.sf(x))
    finally:
        plotting.set_theme(True)


def test_unit_labels_the_time_axis():
    km = sp.KaplanMeier.fit(X, c=C, unit="hour")
    out = sp.plot_nonparametric(km, at_risk=False)
    assert out.ax.get_xlabel() == "Time (h)"


# -- refusals -------------------------------------------------------------


def test_refusals_say_what_to_do():
    km = sp.KaplanMeier.fit(X, c=C)
    with pytest.raises(TypeError, match="fit it first"):
        sp.plot_nonparametric(sp.KaplanMeier)
    with pytest.raises(TypeError, match="its own plot"):
        sp.plot_nonparametric(sp.Weibull.fit(X, c=C))
    with pytest.raises(ValueError, match="empty"):
        sp.plot_nonparametric([])
    with pytest.raises(ValueError, match="one label per model"):
        sp.plot_nonparametric([km, km], labels=["a"])
    with pytest.raises(ValueError, match="repeats a label"):
        sp.plot_nonparametric([km, km], labels=["a", "a"])
    with pytest.raises(ValueError, match="ci=False"):
        sp.plot_nonparametric(km, on="hf", ci=True)
    with pytest.raises(ValueError, match="drop quantiles"):
        sp.plot_nonparametric(km, on="hf", quantiles=0.5)
    with pytest.raises(ValueError, match="two-sided"):
        sp.plot_nonparametric(km, ci="band", bound="lower")
    with pytest.raises(ValueError, match="on"):
        sp.plot_nonparametric(km, on="df")
    with pytest.raises(ValueError, match="at_risk_rows"):
        sp.plot_nonparametric(km, at_risk_rows=["At risk", "Deaths"])
    with pytest.raises(ValueError, match="log_x=False|positive"):
        sp.plot_nonparametric(km, log_x=True, xlim=(0, 10))
    ecdf = sp.NonParametric.fit_from_ecdf([1, 2, 3], [0.8, 0.5, 0.1])
    with pytest.raises(ValueError, match="at_risk=False"):
        sp.plot_nonparametric(ecdf)
    # without bounds or a table it draws
    out = sp.plot_nonparametric(ecdf, at_risk=False)
    assert out.ax.get_title() == "Survival estimate"
    assert not out.ax.collections
