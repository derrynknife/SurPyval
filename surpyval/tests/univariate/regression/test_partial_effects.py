"""Partial-effects plots (#876): ``model.plot_partial_effects(covariate,
values)`` draws the model's own curve at each value of one covariate, the
others held at reference values.

Every check reads the line data off the returned axes and compares them
with the model's function at the rows the plot is documented to build.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import plotting
from surpyval.univariate.regression.parametric_regression_model import (
    ParametricRegressionModel,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _no_figure_leaks():
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for num in leaked:
        plt.close(num)
    assert not leaked, "a test left a figure open"


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "load": rng.uniform(0.5, 1.0, n),
            "temp": rng.normal(50, 5, n),
            "site": rng.choice(["a", "b", "c"], n),
            "n": rng.integers(1, 4, n),
        }
    )
    scale = np.exp(
        -df["load"] + 0.02 * (df["temp"] - 50) + 0.3 * (df["site"] == "b")
    )
    df["x"] = sp.Weibull.random(n, 10, 2, random_state=seed + 1) * scale
    df["c"] = (df["x"] > 15).astype(int)
    df.loc[df["c"] == 1, "x"] = 15.0
    return df


DF = _frame()
# The n-weighted mean of a column, the documented default reference.
TEMP_MEAN = float(np.average(DF["temp"], weights=DF["n"]))
LOAD_MEAN = float(np.average(DF["load"], weights=DF["n"]))


@pytest.fixture(scope="module")
def ph():
    return sp.WeibullPH.fit_from_df(
        DF,
        x_col="x",
        c_col="c",
        n_col="n",
        formula="load + temp + C(site)",
        unit="hour",
    )


@pytest.fixture(scope="module")
def cox():
    return sp.CoxPH.fit_from_df(
        DF, x_col="x", c_col="c", n_col="n", Z_cols=["load", "temp"]
    )


def _band(collection, size):
    """The ``[lower, upper]`` rows of a fill_between of ``size`` points."""
    v = collection.get_paths()[0].vertices
    lower = v[1 : size + 1, 1]
    upper = v[size + 2 : 2 * size + 2, 1][::-1]
    return np.column_stack([lower, upper])


# -- the curves are the model's ---------------------------------------------


def test_curves_are_the_models_survival_at_the_rows(ph, ax):
    values = [0.5, 0.75, 1.0]
    ph.plot_partial_effects("load", values, ax=ax)
    lines = ax.get_lines()
    assert [ln.get_label() for ln in lines] == [
        "load = 0.5",
        "load = 0.75",
        "load = 1",
    ]
    for line, value in zip(lines, values):
        x = line.get_xdata()
        row = pd.DataFrame({"load": [value], "temp": [TEMP_MEAN], "site": "a"})
        np.testing.assert_allclose(line.get_ydata(), ph.sf(x, row), rtol=1e-12)
    # In the palette's order, as every plot takes it
    assert [ln.get_color() for ln in lines] == list(plotting.PALETTE[:3])


@pytest.mark.parametrize("on", ["ff", "Hf"])
def test_on_draws_that_function(ph, ax, on):
    ph.plot_partial_effects("temp", [45, 55], on=on, ax=ax)
    for line, value in zip(ax.get_lines(), [45, 55]):
        row = pd.DataFrame({"load": [LOAD_MEAN], "temp": [value], "site": "a"})
        expected = getattr(ph, on)(line.get_xdata(), row)
        np.testing.assert_allclose(line.get_ydata(), expected, rtol=1e-12)
    label = {"ff": "Failure probability", "Hf": "Cumulative hazard"}[on]
    assert ax.get_ylabel() == label


def test_bands_are_the_models_cb(ph, ax):
    x = np.linspace(1, 14, 30)
    ph.plot_partial_effects(
        "load", [0.5, 1.0], x=x, plot_bounds=True, alpha_ci=0.1, ax=ax
    )
    assert len(ax.collections) == 2
    for band, value in zip(ax.collections, [0.5, 1.0]):
        row = pd.DataFrame({"load": [value], "temp": [TEMP_MEAN], "site": "a"})
        expected = ph.cb(x, row, on="sf", alpha_ci=0.1)
        np.testing.assert_allclose(_band(band, x.size), expected, rtol=1e-12)
    # A wash of its curve's colour, labelled once
    labels = [c.get_label() for c in ax.collections]
    assert labels[0] == "90% confidence bands"
    assert labels[1].startswith("_")


def test_parametric_families_on_arrays_with_bands(ax):
    Z = DF[["load", "temp"]].values
    x = np.linspace(1, 14, 20)
    for fitter in (sp.WeibullAFT, sp.WeibullPO, sp.LogNormalAFT):
        model = fitter.fit(DF["x"].values, Z, c=DF["c"].values)
        ax.clear()
        model.plot_partial_effects(
            1, [45, 55], x=x, on="ff", plot_bounds=True, ax=ax
        )
        assert [ln.get_label() for ln in ax.get_lines()] == [
            "Z[:, 1] = 45",
            "Z[:, 1] = 55",
        ]
        load = DF["load"].mean()
        for line, band, value in zip(ax.get_lines(), ax.collections, [45, 55]):
            np.testing.assert_allclose(
                line.get_ydata(), model.ff(x, [load, value]), rtol=1e-12
            )
            np.testing.assert_allclose(
                _band(band, x.size),
                model.cb(x, [load, value], on="ff"),
                rtol=1e-12,
            )


def test_accelerated_life_by_stress(ax):
    model = sp.AcceleratedLife(sp.Weibull, sp.life_models.Power).fit(
        DF["x"].values, 100 * DF["load"].values, c=DF["c"].values
    )
    model.plot_partial_effects(0, [50, 75, 100], plot_bounds=True, ax=ax)
    for line, stress in zip(ax.get_lines(), [50, 75, 100]):
        np.testing.assert_allclose(
            line.get_ydata(), model.sf(line.get_xdata(), [stress]), rtol=1e-12
        )


def test_cox_curves_are_steps_of_its_sf(cox, ax):
    cox.plot_partial_effects("load", [0.5, 1.0], ax=ax)
    for line, value in zip(ax.get_lines(), [0.5, 1.0]):
        assert line.get_drawstyle() == "steps-post"
        x = line.get_xdata()
        # Every jump of the baseline, from 0 to the last fitted time
        assert x[0] == 0 and x[-1] == DF["x"].max()
        assert np.isin(cox.x, x).all()
        np.testing.assert_allclose(
            line.get_ydata(), cox.sf(x, [value, TEMP_MEAN]), rtol=1e-12
        )


# -- reference values -------------------------------------------------------


def test_default_reference_is_the_weighted_mean_the_center(ax):
    # A fit with center=True keeps the same n-weighted means as center
    model = sp.WeibullPH.fit_from_df(
        DF,
        x_col="x",
        c_col="c",
        n_col="n",
        Z_cols=["load", "temp"],
        center=True,
    )
    np.testing.assert_allclose(model.center, [LOAD_MEAN, TEMP_MEAN])
    model.plot_partial_effects("load", [0.6], ax=ax)
    (line,) = ax.get_lines()
    np.testing.assert_allclose(
        line.get_ydata(),
        model.sf(line.get_xdata(), [0.6, model.center[1]]),
        rtol=1e-12,
    )


def test_reference_overrides_by_name(ph, ax):
    ph.plot_partial_effects(
        "load", [0.5], reference={"temp": 40, "site": "c"}, ax=ax
    )
    (line,) = ax.get_lines()
    row = pd.DataFrame({"load": [0.5], "temp": [40], "site": ["c"]})
    np.testing.assert_allclose(
        line.get_ydata(), ph.sf(line.get_xdata(), row), rtol=1e-12
    )


def test_reference_of_the_plotted_covariate_is_ignored(ph, ax):
    ph.plot_partial_effects(
        "load", [0.5], reference={"load": 0.9, "temp": 40}, ax=ax
    )
    (line,) = ax.get_lines()
    row = pd.DataFrame({"load": [0.5], "temp": [40], "site": ["a"]})
    np.testing.assert_allclose(
        line.get_ydata(), ph.sf(line.get_xdata(), row), rtol=1e-12
    )


def test_formula_categorical_levels(ph, ax):
    ph.plot_partial_effects("site", ["a", "b", "c"], ax=ax)
    lines = ax.get_lines()
    assert [ln.get_label() for ln in lines] == [
        "site = a",
        "site = b",
        "site = c",
    ]
    for line, level in zip(lines, "abc"):
        row = pd.DataFrame(
            {"load": [LOAD_MEAN], "temp": [TEMP_MEAN], "site": [level]}
        )
        np.testing.assert_allclose(
            line.get_ydata(), ph.sf(line.get_xdata(), row), rtol=1e-12
        )
    # Level "b" lives longer (its times were scaled by e^0.3)
    assert lines[1].get_ydata()[-1] > lines[0].get_ydata()[-1]


def test_formula_categorical_reference_is_the_base_level(ax):
    model = sp.CoxPH.fit_from_df(
        DF,
        x_col="x",
        c_col="c",
        formula='load + C(site, contr.treatment("b"))',
    )
    model.plot_partial_effects("load", [0.5], ax=ax)
    (line,) = ax.get_lines()
    row = pd.DataFrame({"load": [0.5], "site": ["b"]})
    np.testing.assert_allclose(
        line.get_ydata(), model.sf(line.get_xdata(), row), rtol=1e-12
    )


def test_unseen_level_refused(ph):
    with pytest.raises(ValueError, match="Unknown categorical level"):
        ph.plot_partial_effects("site", ["a", "z"])


def test_categorical_without_a_reference_level_refused(ax):
    model = sp.CoxPH.fit_from_df(
        DF, x_col="x", c_col="c", formula="load + C(site, contr.sum)"
    )
    with pytest.raises(ValueError, match=r"reference=\{'site': \.\.\.\}"):
        model.plot_partial_effects("load", [0.5], ax=ax)
    assert not ax.get_lines()
    model.plot_partial_effects("load", [0.5], reference={"site": "a"}, ax=ax)
    (line,) = ax.get_lines()
    row = pd.DataFrame({"load": [0.5], "site": ["a"]})
    np.testing.assert_allclose(
        line.get_ydata(), model.sf(line.get_xdata(), row), rtol=1e-12
    )


# -- transformed terms and interactions, through the formula ----------------

FORMULA = "np.log(load) + temp + load:temp"


@pytest.fixture(scope="module")
def transformed():
    return sp.WeibullPH.fit_from_df(DF, x_col="x", c_col="c", formula=FORMULA)


def test_transformed_variable_goes_through_the_formula(transformed, ax):
    transformed.plot_partial_effects("load", [0.5, 1.0], ax=ax)
    temp = DF["temp"].mean()
    for line, value in zip(ax.get_lines(), [0.5, 1.0]):
        row = pd.DataFrame({"load": [value], "temp": [temp]})
        np.testing.assert_allclose(
            line.get_ydata(), transformed.sf(line.get_xdata(), row), rtol=1e-12
        )


def test_transformed_variable_needs_a_reference(transformed, ax):
    # The model keeps np.log(load) and load:temp, not load: no mean of it
    with pytest.raises(ValueError) as info:
        transformed.plot_partial_effects("temp", [45, 55], ax=ax)
    message = str(info.value)
    assert "'np.log(load)', 'load:temp'" in message
    assert "reference={'load': ...}" in message
    assert not ax.get_lines()
    transformed.plot_partial_effects(
        "temp", [45, 55], reference={"load": 0.75}, ax=ax
    )
    for line, value in zip(ax.get_lines(), [45, 55]):
        row = pd.DataFrame({"load": [0.75], "temp": [value]})
        np.testing.assert_allclose(
            line.get_ydata(), transformed.sf(line.get_xdata(), row), rtol=1e-12
        )


@pytest.mark.parametrize(
    "term, variables",
    [
        ("np.log(load)", "the variable 'load'"),
        ("load:temp", "the variables 'load' and 'temp'"),
    ],
)
def test_a_term_is_refused_naming_its_variables(transformed, term, variables):
    with pytest.raises(ValueError, match="not one of its variables") as info:
        transformed.plot_partial_effects(term, [0.0, 1.0])
    assert "give " + variables in str(info.value)


def test_design_column_of_a_level_is_refused(ph):
    with pytest.raises(ValueError, match="give the variable 'site'"):
        ph.plot_partial_effects("C(site)[T.b]", [0, 1])


# -- names -------------------------------------------------------------------


def test_unknown_covariate_refused_naming_the_covariates(ph, cox):
    with pytest.raises(ValueError, match=r"\['load', 'temp', 'site'\]"):
        ph.plot_partial_effects("pressure", [1, 2])
    with pytest.raises(ValueError, match=r"its covariates are \['load', 'te"):
        cox.plot_partial_effects("pressure", [1, 2])
    with pytest.raises(ValueError, match="reference covariate 'pres' is not"):
        cox.plot_partial_effects("load", [1], reference={"pres": 2})
    with pytest.raises(ValueError, match="reference must be a dict"):
        cox.plot_partial_effects("load", [1], reference=[("temp", 2)])


def test_arrays_take_a_position():
    model = sp.WeibullAFT.fit(DF["x"].values, DF[["load", "temp"]].values)
    with pytest.raises(ValueError, match="give the column's position in Z"):
        model.plot_partial_effects("load", [0.5])
    with pytest.raises(ValueError, match="positions 0 to 1"):
        model.plot_partial_effects(2, [0.5])
    with pytest.raises(ValueError, match="must be numbers"):
        model.plot_partial_effects(0, ["high"])


# -- the time grid ----------------------------------------------------------


def test_default_grid_spans_the_data_with_units(ph, ax):
    ph.plot_partial_effects("load", [0.5], ax=ax)
    (line,) = ax.get_lines()
    np.testing.assert_array_equal(
        line.get_xdata(), np.linspace(0, DF["x"].max(), 200)
    )
    assert ax.get_xlabel() == "Time (h)"
    assert ax.get_ylabel() == "Survival probability"


def test_given_grid_is_drawn_as_given(ph, ax):
    x = np.array([0.5, 2.0, 7.5, 30.0])
    ph.plot_partial_effects("load", [0.5, 1.0], x=x, ax=ax)
    for line in ax.get_lines():
        np.testing.assert_array_equal(line.get_xdata(), x)
        assert line.get_ydata().shape == x.shape


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"x": [[1.0, 2.0]]}, "x must be a 1-D array"),
        ({"x": []}, "x must be a 1-D array"),
        ({"values": []}, "values is empty"),
        ({"values": [[0.5, 1.0]]}, "values must be a 1-D list"),
        ({"on": "hf"}, r"use one of \['sf', 'ff', 'Hf'\]"),
        ({"plot_bounds": True, "alpha_ci": 1.5}, "alpha_ci"),
    ],
)
def test_bad_arguments_refused(ph, ax, kwargs, match):
    call = {"values": [0.5], **kwargs}
    with pytest.raises(ValueError, match=match):
        ph.plot_partial_effects("load", ax=ax, **call)
    assert not ax.get_lines()


def test_draws_on_the_current_axes_by_default(ph):
    fig, ax = plt.subplots()
    try:
        assert ph.plot_partial_effects("load", [0.5]) is ax
    finally:
        plt.close(fig)


# -- what a model cannot do is refused --------------------------------------


def test_cox_has_no_bands(cox):
    with pytest.raises(ValueError, match=r"cb\), which Semi"):
        cox.plot_partial_effects("load", [0.5], plot_bounds=True)


def test_stratified_cox_refused():
    model = sp.CoxPH.fit_from_df(
        DF, x_col="x", c_col="c", Z_cols=["load"], strata_col="site"
    )
    with pytest.raises(ValueError, match="stratum=..."):
        model.plot_partial_effects("load", [0.5])


def test_restored_model_needs_references_and_times(ax):
    fitted = sp.WeibullAFT.fit(DF["x"].values, DF[["load", "temp"]].values)
    model = ParametricRegressionModel.from_dict(fitted.to_dict())
    with pytest.raises(ValueError, match=r"Z\[:, 1\] \(the model does not"):
        model.plot_partial_effects(0, [0.5], ax=ax)
    with pytest.raises(ValueError, match="pass the times to draw at"):
        model.plot_partial_effects(0, [0.5], reference={1: 50}, ax=ax)
    x = np.linspace(1, 10, 5)
    model.plot_partial_effects(0, [0.5], reference={1: 50}, x=x, ax=ax)
    (line,) = ax.get_lines()
    np.testing.assert_allclose(line.get_ydata(), fitted.sf(x, [0.5, 50]))


def test_time_varying_fit_needs_references():
    rng = np.random.default_rng(1)
    n = 100
    Z = rng.normal(size=(n, 2))
    T = rng.exponential(1 / np.exp(0.7 * Z[:, 0]))
    s = 0.5 * T
    ident = np.repeat(np.arange(n), 2)
    xl = np.column_stack([np.zeros(n), s]).ravel()
    xr = np.column_stack([s, T]).ravel()
    c = np.tile([1, 0], n)
    model = sp.CoxPH.fit_tvc(ident, xl, xr, c, np.repeat(Z, 2, axis=0))
    with pytest.raises(ValueError, match="intervals, not units"):
        model.plot_partial_effects(0, [0.0, 1.0])
    fig, ax = plt.subplots()
    try:
        model.plot_partial_effects(0, [0.0, 1.0], reference={1: 0.0}, ax=ax)
        line = ax.get_lines()[1]
        np.testing.assert_allclose(
            line.get_ydata(), model.sf(line.get_xdata(), [1.0, 0.0])
        )
    finally:
        plt.close(fig)


# -- competing risks: cumulative incidence ----------------------------------


@pytest.fixture(scope="module")
def causes():
    rng = np.random.default_rng(3)
    e = rng.choice(["wear", "shock"], len(DF)).astype(object)
    e[DF["c"].values == 1] = None
    return e


def test_competing_risks_cif(causes, ax):
    Z = DF[["load", "temp"]].values
    model = sp.CompetingRisksProportionalHazards.fit(DF["x"].values, Z, causes)
    with pytest.raises(ValueError, match=r"reference=\{1: \.\.\.\}"):
        model.plot_partial_effects(0, [0.5, 1.0], event="wear", ax=ax)
    with pytest.raises(ValueError, match="pass `event`"):
        model.plot_partial_effects(0, [0.5], reference={1: 50}, ax=ax)
    assert not ax.get_lines()
    model.plot_partial_effects(
        0, [0.5, 1.0], event="wear", reference={1: 50}, ax=ax
    )
    for line, value in zip(ax.get_lines(), [0.5, 1.0]):
        assert line.get_drawstyle() == "steps-post"
        np.testing.assert_allclose(
            line.get_ydata(),
            model.cif(line.get_xdata(), [value, 50], "wear"),
            rtol=1e-12,
        )
    assert ax.get_ylabel() == "Cumulative incidence of wear"


def test_fine_gray_cif(causes, ax):
    df = DF.assign(e=causes)
    model = sp.FineGray.fit(
        df["x"].values, df[["load", "temp"]].values, causes, event="wear"
    )
    with pytest.raises(ValueError, match="Unknown cause 'shock'"):
        model.plot_partial_effects(0, [0.5], event="shock", reference={1: 50})
    model.plot_partial_effects(0, [0.5, 1.0], reference={1: 50}, ax=ax)
    for line, value in zip(ax.get_lines(), [0.5, 1.0]):
        np.testing.assert_allclose(
            line.get_ydata(),
            model.cif(line.get_xdata(), [value, 50]),
            rtol=1e-12,
        )
