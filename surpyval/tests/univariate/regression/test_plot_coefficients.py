"""surpyval.plot_coefficients: the points and intervals it draws are the
models' own summary() values, on the scale asked for, and it refuses what
it cannot draw. Asserts on the artists' data, not on images."""

import matplotlib
import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.datasets import load_rossi_static

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLS = ["fin", "age", "prio"]


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def rossi():
    return load_rossi_static().assign(censored=lambda d: 1 - d.arrest)


def _fit(fitter, df):
    return fitter.fit_from_df(df, x_col="week", c_col="censored", Z_cols=COLS)


@pytest.fixture(scope="module")
def cox(rossi):
    return _fit(sp.CoxPH, rossi)


@pytest.fixture(scope="module")
def aft(rossi):
    return _fit(sp.WeibullAFT, rossi)


@pytest.fixture(scope="module")
def ah(rossi):
    return _fit(sp.AdditiveHazards, rossi)


def _drawn(out, label):
    """(estimates, lowers, uppers, y) as drawn for ``label``."""
    point = out.points[label]
    segments = out.intervals[label].get_segments()
    lower = np.array([s[0][0] for s in segments])
    upper = np.array([s[1][0] for s in segments])
    ys = np.array([s[0][1] for s in segments])
    assert_allclose(point.get_ydata(), ys)
    return np.asarray(point.get_xdata()), lower, upper, ys


def _summary(model, alpha_ci=0.05, exp=False):
    table = model.summary(alpha_ci=alpha_ci)
    if table.index.nlevels > 1:
        table = table.loc["coefficients"]
    level = f"{100 * (1 - alpha_ci):g}%"
    pre = "exp(coef)" if exp else "coef"
    return (
        table[pre].to_numpy(),
        table[f"{pre} lower {level}"].to_numpy(),
        table[f"{pre} upper {level}"].to_numpy(),
    )


@pytest.mark.parametrize("name", ["cox", "aft"])
@pytest.mark.parametrize("on", ["coef", "exp"])
def test_points_and_intervals_are_the_summary(request, name, on):
    model = request.getfixturevalue(name)
    out = sp.plot_coefficients({"m": model}, on=on, alpha_ci=0.1)
    est, lo, hi, ys = _drawn(out, "m")
    ref = _summary(model, 0.1, exp=on == "exp")
    assert_allclose(est, ref[0], rtol=1e-12)
    assert_allclose(lo, ref[1], rtol=1e-12)
    assert_allclose(hi, ref[2], rtol=1e-12)
    assert out.covariates == COLS
    assert [t.get_text() for t in out.ax.get_yticklabels()] == COLS
    # the first covariate at the top
    assert_allclose(ys, [0, 1, 2])
    assert out.ax.get_ylim()[0] > out.ax.get_ylim()[1]


def test_reference_line_and_scale(cox):
    exp = sp.plot_coefficients(cox, on="exp")
    coef = sp.plot_coefficients(cox, on="coef")

    def ref(out):
        (line,) = (v for v in out.ax.lines if v.get_gid() == "reference")
        return line.get_xdata()[0]

    assert ref(exp) == 1 and exp.ax.get_xscale() == "log"
    assert ref(coef) == 0 and coef.ax.get_xscale() == "linear"
    assert exp.ax.get_xlabel() == "Hazard ratio, exp(coef)"
    assert coef.ax.get_xlabel() == "Coefficient"


def test_default_scale_is_the_ratio_where_every_model_has_one(cox, aft, ah):
    assert sp.plot_coefficients(cox).on == "exp"
    assert sp.plot_coefficients([cox, aft]).on == "exp"
    assert sp.plot_coefficients(ah).on == "coef"
    assert sp.plot_coefficients([cox, ah]).on == "coef"
    label = sp.plot_coefficients(aft).ax.get_xlabel()
    assert label == "Acceleration factor, exp(coef)"
    assert sp.plot_coefficients([cox, aft]).ax.get_xlabel() == "exp(coef)"


def test_additive_model_on_its_coefficient_scale(ah):
    out = sp.plot_coefficients(ah)
    est, lo, hi, _ = _drawn(out, "AdditiveHazardsModel")
    ref = _summary(ah)
    assert_allclose(est, ref[0], rtol=1e-12)
    assert_allclose(lo, ref[1], rtol=1e-12)
    assert_allclose(hi, ref[2], rtol=1e-12)


def test_parametric_additive_hazards_coefficients(rossi):
    # The Rossi fit reaches the model's support boundary, which it says.
    with pytest.warns(UserWarning, match="boundary"):
        model = _fit(sp.WeibullAH, rossi)
    out = sp.plot_coefficients(model)
    assert out.on == "coef"
    est, _, _, _ = _drawn(out, "WeibullAH")
    assert_allclose(est, _summary(model)[0], rtol=1e-12)


def test_exp_scale_refused_for_an_additive_model(ah):
    with pytest.raises(ValueError, match="on='exp'.*use on='coef'"):
        sp.plot_coefficients(ah, on="exp")


def test_models_are_dodged_and_take_the_palette(cox, aft):
    from surpyval.plotting import PALETTE

    out = sp.plot_coefficients({"Cox": cox, "AFT": aft}, on="coef")
    _, _, _, y_cox = _drawn(out, "Cox")
    _, _, _, y_aft = _drawn(out, "AFT")
    assert np.all(y_cox < y_aft)
    assert_allclose((y_cox + y_aft) / 2, [0, 1, 2])
    assert out.points["Cox"].get_color() == PALETTE[0]
    assert out.points["AFT"].get_color() == PALETTE[1]
    labels = [t.get_text() for t in out.ax.get_legend().get_texts()]
    assert labels == ["Cox", "AFT"]


def test_labels_colors_title(cox, aft):
    out = sp.plot_coefficients(
        [cox, aft],
        labels=["a", "b"],
        colors={"a": "black", "b": "red"},
        title="T",
        xlabel="X",
    )
    assert list(out.values) == ["a", "b"]
    assert out.points["b"].get_color() == "red"
    assert out.ax.get_title() == "T" and out.ax.get_xlabel() == "X"


def test_default_labels_from_a_list(cox, aft):
    out = sp.plot_coefficients([cox, aft])
    assert list(out.values) == ["CoxPH", "WeibullAFT"]
    assert list(sp.plot_coefficients([cox, cox]).values) == [
        "Model 1",
        "Model 2",
    ]


def test_covariates_selects_rows_and_union_keeps_order(rossi, cox):
    small = sp.CoxPH.fit_from_df(
        rossi, x_col="week", c_col="censored", Z_cols=["prio", "fin"]
    )
    out = sp.plot_coefficients({"all": cox, "two": small})
    assert out.covariates == COLS
    assert np.isnan(out.values["two"].loc["age", "estimate"])
    assert_allclose(
        out.values["two"].loc["prio", "estimate"],
        small.summary().loc["prio", "exp(coef)"],
    )
    only = sp.plot_coefficients(cox, covariates=["prio"])
    assert only.covariates == ["prio"]


def test_sorting(cox):
    by_est = sp.plot_coefficients(cox, on="coef", sort="estimate")
    est = by_est.values["CoxPH"]["estimate"].to_numpy()
    assert np.all(np.diff(est) >= 0)
    by_name = sp.plot_coefficients(cox, sort="name")
    assert by_name.covariates == sorted(COLS)


def test_frailty_and_baseline_rows_left_out(rossi):
    groups = np.arange(len(rossi)) % 20
    model = sp.WeibullFrailty.fit(
        rossi["week"].values,
        rossi[COLS].values.astype(float),
        c=rossi["censored"].values,
        groups=groups,
    )
    out = sp.plot_coefficients(model)
    assert out.covariates == ["coef_0", "coef_1", "coef_2"]


def test_accelerated_life_needs_named_rows(rossi):
    model = sp.AcceleratedLife(sp.Weibull, sp.life_models.Power).fit(
        rossi["week"].values,
        rossi["age"].values.astype(float),
        c=rossi["censored"].values,
    )
    with pytest.raises(ValueError, match="covariates="):
        sp.plot_coefficients(model)
    out = sp.plot_coefficients(model, covariates=["n"])
    assert out.covariates == ["n"] and out.on == "coef"
    with pytest.raises(ValueError, match="'covariates'"):
        sp.plot_coefficients(model, covariates=["nope"])


def test_fine_gray(rossi):
    x = rossi["week"].values
    c = rossi["censored"].values
    e = np.where(np.arange(len(x)) % 2 == 0, 1, 2).astype(object)
    e[c == 1] = None
    Z = rossi[COLS].values.astype(float)
    model = sp.FineGray.fit(x, Z, e=e, c=c, event=1)
    out = sp.plot_coefficients(model)
    assert out.on == "exp"
    assert out.ax.get_xlabel() == "Subdistribution hazard ratio, exp(coef)"


def test_refusals(cox):
    km = sp.KaplanMeier.fit([1, 2, 3], c=[0, 0, 1])
    with pytest.raises(TypeError, match="'models'.*no coefficient table"):
        sp.plot_coefficients(km)
    with pytest.raises(ValueError, match="'models' is empty"):
        sp.plot_coefficients([])
    with pytest.raises(ValueError, match="'labels'"):
        sp.plot_coefficients([cox], labels=["a", "b"])
    with pytest.raises(ValueError, match="on"):
        sp.plot_coefficients(cox, on="hr")
    with pytest.raises(ValueError, match="sort"):
        sp.plot_coefficients(cox, sort="size")
    with pytest.raises(ValueError, match="alpha_ci"):
        sp.plot_coefficients(cox, alpha_ci=1.5)


def test_theme_off_uses_matplotlibs_colours(cox, aft):
    from surpyval import plotting

    plotting.set_theme(False)
    try:
        out = sp.plot_coefficients([cox, aft])
    finally:
        plotting.set_theme(True)
    assert out.points["CoxPH"].get_color() == "C0"
    assert out.points["WeibullAFT"].get_color() == "C1"


def test_on_given_axes(cox):
    fig, ax = plt.subplots()
    out = sp.plot_coefficients(cox, ax=ax)
    assert out.ax is ax and out.fig is fig
