import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from surpyval import LogNormal, Weibull  # noqa: E402
from surpyval.degradation import (  # noqa: E402
    DegradationAnalysis,
    ExponentialPath,
)


def linear_data(slopes, intercept=10.0, n_obs=10):
    """Noiseless linear degradation data, one row of times per unit."""
    x = np.tile(np.arange(100, 100 * (n_obs + 1), 100), len(slopes))
    i = np.repeat(np.arange(1, len(slopes) + 1), n_obs)
    y = intercept + np.repeat(slopes, n_obs) * x
    return x, y, i


def test_linear_pseudo_failure_times_exact():
    slopes = np.array([0.31, 0.28, 0.44, 0.37])
    x, y, i = linear_data(slopes)
    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    expected = (150 - 10.0) / slopes
    assert np.allclose(model.pseudo_failure_times, expected)
    assert (model.c == 0).all()
    assert model.path_model.name == "Linear"
    assert np.allclose(model.path_params[:, 0], 10.0)
    assert np.allclose(model.path_params[:, 1], slopes)


def test_life_model_matches_direct_fit_of_pseudo_times():
    slopes = np.array([0.31, 0.28, 0.44, 0.37, 0.52])
    x, y, i = linear_data(slopes)
    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    direct = Weibull.fit((150 - 10.0) / slopes)
    assert np.allclose(model.life_model.params, direct.params, rtol=1e-4)


def test_weibull_parameters_recovered():
    # slopes chosen so the exact threshold-crossing times are Weibull
    rng = np.random.default_rng(123)
    alpha, beta = 345.0, 4.5
    times = alpha * rng.weibull(beta, 50)
    threshold, intercept = 150.0, 10.0
    slopes = (threshold - intercept) / times
    x, y, i = linear_data(slopes)
    model = DegradationAnalysis.fit(x, y, i, threshold=threshold)
    direct = Weibull.fit(times)
    assert np.allclose(model.pseudo_failure_times, times)
    assert np.allclose(model.life_model.params, direct.params, rtol=1e-4)
    assert np.allclose(model.life_model.params, [alpha, beta], rtol=0.15)


def test_exponential_path_end_to_end():
    a = np.array([2.0, 2.5, 1.8, 2.2])
    b = np.array([0.004, 0.005, 0.006, 0.0045])
    n_obs = 10
    x = np.tile(np.arange(50, 50 * (n_obs + 1), 50), len(a))
    i = np.repeat(np.arange(1, len(a) + 1), n_obs)
    y = np.repeat(a, n_obs) * np.exp(np.repeat(b, n_obs) * x)
    model = DegradationAnalysis.fit(x, y, i, threshold=20, path="exponential")
    expected = np.log(20 / a) / b
    assert np.allclose(model.pseudo_failure_times, expected, rtol=1e-5)
    assert (model.c == 0).all()


def test_non_degrading_unit_is_right_censored():
    slopes = np.array([0.31, 0.28, 0.44, -0.05])
    x, y, i = linear_data(slopes)
    with pytest.warns(UserWarning, match="right censored"):
        model = DegradationAnalysis.fit(x, y, i, threshold=150)
    assert model.c.tolist() == [0, 0, 0, 1]
    # censored at the unit's last observed time
    assert model.pseudo_failure_times[-1] == 1000
    assert int((model.c == 1).sum()) == 1


def test_all_censored_raises():
    slopes = np.array([-0.31, -0.28])
    x, y, i = linear_data(slopes)
    with pytest.raises(ValueError, match="reaches the threshold"):
        DegradationAnalysis.fit(x, y, i, threshold=150)


def test_alternative_distribution_and_method():
    slopes = np.array([0.31, 0.28, 0.44, 0.37, 0.52])
    x, y, i = linear_data(slopes)
    model = DegradationAnalysis.fit(
        x, y, i, threshold=150, distribution=LogNormal, how="MPP"
    )
    assert model.life_model.dist.name == "LogNormal"
    assert model.life_model.method == "MPP"


def test_fit_from_df_matches_fit():
    slopes = np.array([0.31, 0.28, 0.44, 0.37])
    x, y, i = linear_data(slopes)
    df = pd.DataFrame({"time": x, "measurement": y, "unit": i})
    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    df_model = DegradationAnalysis.fit_from_df(
        df, x="time", y="measurement", i="unit", threshold=150
    )
    assert np.allclose(
        model.pseudo_failure_times, df_model.pseudo_failure_times
    )
    assert np.allclose(model.life_model.params, df_model.life_model.params)


def test_life_model_delegation():
    slopes = np.array([0.31, 0.28, 0.44, 0.37])
    x, y, i = linear_data(slopes)
    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    t = np.array([300.0, 400.0, 500.0])
    assert np.allclose(model.sf(t), model.life_model.sf(t))
    assert np.allclose(model.ff(t), model.life_model.ff(t))
    assert np.allclose(model.df(t), model.life_model.df(t))
    assert np.allclose(model.hf(t), model.life_model.hf(t))
    assert np.allclose(model.Hf(t), model.life_model.Hf(t))
    assert np.allclose(model.qf(0.5), model.life_model.qf(0.5))
    assert np.allclose(model.mean(), model.life_model.mean())
    assert len(model.random(5)) == 5


def test_unit_path_evaluation():
    slopes = np.array([0.31, 0.28, 0.44, 0.37])
    x, y, i = linear_data(slopes)
    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    assert np.allclose(
        model.path([100, 200], 3), 10 + 0.44 * np.array([100, 200])
    )


def test_repr():
    slopes = np.array([0.31, 0.28, 0.44, 0.37])
    x, y, i = linear_data(slopes)
    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    out = repr(model)
    assert "Degradation Analysis SurPyval Model" in out
    assert "Linear" in out
    assert "Weibull" in out


def test_plot():
    slopes = np.array([0.31, 0.28, 0.44, -0.05])
    x, y, i = linear_data(slopes)
    with pytest.warns(UserWarning):
        model = DegradationAnalysis.fit(x, y, i, threshold=150)
    ax = model.plot()
    # one path line per unit plus the threshold line
    assert len(ax.get_lines()) == len(slopes) + 1
    matplotlib.pyplot.close("all")


def test_custom_path_model_instance():
    a = np.array([2.0, 2.5, 1.8, 2.2])
    b = np.array([0.004, 0.005, 0.006, 0.0045])
    n_obs = 10
    x = np.tile(np.arange(50, 50 * (n_obs + 1), 50), len(a))
    i = np.repeat(np.arange(1, len(a) + 1), n_obs)
    y = np.repeat(a, n_obs) * np.exp(np.repeat(b, n_obs) * x)
    model = DegradationAnalysis.fit(
        x, y, i, threshold=20, path=ExponentialPath
    )
    assert model.path_model is ExponentialPath


def test_predict_failure_time_new_trajectory():
    slopes = np.array([0.31, 0.28, 0.44, 0.37])
    x, y, i = linear_data(slopes)
    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    # a new unit observed for only 300 hours, degrading at 0.5/hour
    x_new = np.array([100.0, 200.0, 300.0])
    y_new = 10 + 0.5 * x_new
    predicted = model.predict_failure_time(x_new, y_new)
    assert np.isclose(predicted, (150 - 10) / 0.5)
    remaining = model.predict_remaining_life(x_new, y_new)
    assert np.isclose(remaining, (150 - 10) / 0.5 - 300)


def test_predict_failure_time_already_crossed():
    slopes = np.array([0.31, 0.28, 0.44, 0.37])
    x, y, i = linear_data(slopes)
    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    # a unit whose trajectory crossed the threshold before its last
    # observation: predicted failure is in the past, remaining life < 0
    x_new = np.array([100.0, 200.0, 300.0])
    y_new = 10 + 1.0 * x_new
    assert np.isclose(model.predict_failure_time(x_new, y_new), 140)
    assert model.predict_remaining_life(x_new, y_new) < 0


def test_predict_failure_time_non_degrading_is_nan():
    slopes = np.array([0.31, 0.28, 0.44, 0.37])
    x, y, i = linear_data(slopes)
    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    x_new = np.array([100.0, 200.0, 300.0])
    y_new = 10 - 0.1 * x_new
    with pytest.warns(UserWarning, match="never\\s+reaches"):
        predicted = model.predict_failure_time(x_new, y_new)
    assert np.isnan(predicted)
    with pytest.warns(UserWarning, match="never\\s+reaches"):
        remaining = model.predict_remaining_life(x_new, y_new)
    assert np.isnan(remaining)


@pytest.mark.parametrize(
    "x_new,y_new",
    [
        ([1, 2, 3], [1, 2]),  # mismatched lengths
        ([100], [20]),  # too few points
        ([100, 100], [20, 21]),  # single distinct time
        ([100, 200], [20, np.nan]),  # non-finite measurement
    ],
)
def test_predict_failure_time_validation(x_new, y_new):
    slopes = np.array([0.31, 0.28, 0.44, 0.37])
    x, y, i = linear_data(slopes)
    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    with pytest.raises(ValueError):
        model.predict_failure_time(x_new, y_new)


@pytest.mark.parametrize(
    "kwargs",
    [
        # mismatched lengths
        dict(x=[1, 2, 3], y=[1, 2], i=[1, 1, 1], threshold=10),
        # empty
        dict(x=[], y=[], i=[], threshold=10),
        # non-finite measurement
        dict(
            x=[1, 2, 1, 2], y=[1, np.nan, 1, 2], i=[1, 1, 2, 2], threshold=10
        ),
        # non-finite threshold
        dict(x=[1, 2, 1, 2], y=[1, 2, 1, 2], i=[1, 1, 2, 2], threshold=np.inf),
        # only one unit
        dict(x=[1, 2, 3], y=[1, 2, 3], i=[1, 1, 1], threshold=10),
        # unknown path model
        dict(
            x=[1, 2, 1, 2],
            y=[1, 2, 1, 2],
            i=[1, 1, 2, 2],
            threshold=10,
            path="cubic",
        ),
        # unit 2 has only one measurement
        dict(x=[1, 2, 1], y=[1, 2, 1], i=[1, 1, 2], threshold=10),
        # unit measured at a single distinct time
        dict(x=[1, 2, 3, 3], y=[1, 2, 1, 2], i=[1, 1, 2, 2], threshold=10),
    ],
)
def test_input_validation(kwargs):
    with pytest.raises(ValueError):
        DegradationAnalysis.fit(**kwargs)
