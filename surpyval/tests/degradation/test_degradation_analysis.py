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


def test_noiseless_population_estimates():
    slopes = np.array([0.31, 0.28, 0.44, 0.37])
    x, y, i = linear_data(slopes)
    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    # exact paths: no measurement noise, so no correction is applied
    assert model.measurement_var == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(model.path_param_mean, [10.0, slopes.mean()])
    assert np.allclose(model.path_param_cov, model.path_param_sample_cov)
    true_cov = np.cov(
        np.column_stack([np.full(4, 10.0), slopes]), rowvar=False, ddof=1
    )
    assert np.allclose(model.path_param_sample_cov, true_cov)


def test_noise_correction_recovers_between_unit_covariance():
    # Small design (4 points close together) so the least-squares
    # estimation noise is comparable to the between-unit variance and
    # the correction is material.
    rng = np.random.default_rng(7)
    n_units = 200
    x_times = np.array([10.0, 20.0, 30.0, 40.0])
    a_true = rng.normal(10.0, 1.0, n_units)  # var(a) = 1.0
    b_true = rng.normal(0.4, 0.05, n_units)  # var(b) = 0.0025
    sigma = 1.0  # measurement noise
    x = np.tile(x_times, n_units)
    i = np.repeat(np.arange(n_units), len(x_times))
    y = (
        np.repeat(a_true, len(x_times))
        + np.repeat(b_true, len(x_times)) * x
        + rng.normal(0, sigma, len(x))
    )
    model = DegradationAnalysis.fit(x, y, i, threshold=100)

    assert np.isclose(model.measurement_var, sigma**2, rtol=0.25)

    # for this design: V_a = sigma^2 * 3000 / (4 * 500) = 1.5 and
    # V_b = sigma^2 / 500 = 0.002, so the naive sample covariance is
    # inflated to ~2.5 and ~0.0045 respectively
    naive_a = model.path_param_sample_cov[0, 0]
    naive_b = model.path_param_sample_cov[1, 1]
    corrected_a = model.path_param_cov[0, 0]
    corrected_b = model.path_param_cov[1, 1]
    assert naive_a > 1.7
    assert naive_b > 0.0035
    assert abs(corrected_a - 1.0) < 0.6
    assert abs(corrected_b - 0.0025) < 0.0012
    assert abs(corrected_a - 1.0) < abs(naive_a - 1.0)
    assert abs(corrected_b - 0.0025) < abs(naive_b - 0.0025)


def test_clip_psd():
    from surpyval.degradation.degradation_analysis import _clip_psd

    # eigenvalues 3 and -1: must be clipped
    clipped_matrix, clipped = _clip_psd(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert clipped
    assert np.allclose(clipped_matrix, clipped_matrix.T)
    assert (np.linalg.eigvalsh(clipped_matrix) >= 0).all()
    # already PSD: unchanged and not flagged
    psd = np.array([[2.0, 0.5], [0.5, 1.0]])
    same_matrix, clipped = _clip_psd(psd)
    assert not clipped
    assert np.allclose(same_matrix, psd)


def _noisy_population_data(n_units=100, seed=7):
    """Balanced noisy linear degradation data: a ~ N(10, 1),
    b ~ N(0.4, 0.05^2), measurement noise sd 1."""
    rng = np.random.default_rng(seed)
    x_times = np.array([10.0, 20.0, 30.0, 40.0])
    a = rng.normal(10.0, 1.0, n_units)
    b = rng.normal(0.4, 0.05, n_units)
    x = np.tile(x_times, n_units)
    i = np.repeat(np.arange(n_units), len(x_times))
    y = (
        np.repeat(a, len(x_times))
        + np.repeat(b, len(x_times)) * x
        + rng.normal(0, 1.0, len(x))
    )
    return x, y, i


def _unbalanced_population_data(n_units=100, seed=3):
    """Unbalanced noisy linear degradation data (different measurement
    schedules per unit), same population as _noisy_population_data."""
    rng = np.random.default_rng(seed)
    xs, ys, iis = [], [], []
    for k in range(n_units):
        n_k = int(rng.integers(3, 9))
        x_k = np.sort(rng.uniform(5, 60, n_k))
        a_k, b_k = rng.normal(10.0, 1.0), rng.normal(0.4, 0.05)
        y_k = a_k + b_k * x_k + rng.normal(0, 1.0, n_k)
        xs.append(x_k)
        ys.append(y_k)
        iis.append(np.full(n_k, k))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(iis)


def _noisy_model(**kwargs):
    x, y, i = _noisy_population_data()
    return DegradationAnalysis.fit(x, y, i, threshold=100, **kwargs)


def test_predict_rul_agrees_with_extrapolation_given_much_data():
    model = _noisy_model()
    rng = np.random.default_rng(11)
    x_new = np.arange(10.0, 210.0, 10.0)
    y_new = 10 + 0.4 * x_new + rng.normal(0, 1.0, len(x_new))
    pred = model.predict_rul(x_new, y_new, random_state=1)
    plain = model.predict_failure_time(x_new, y_new)
    assert abs(pred.failure_time - plain) / plain < 0.1
    lower, upper = pred.failure_time_interval
    assert lower < pred.failure_time < upper
    assert pred.rul == pytest.approx(pred.failure_time - 200.0, rel=1e-9)
    assert pred.prob_failed == 0.0
    assert pred.prob_never_fails == 0.0


def test_predict_rul_single_measurement():
    model = _noisy_model()
    pred_one = model.predict_rul([20.0], [18.0], random_state=2)
    assert np.isfinite(pred_one.failure_time)
    assert pred_one.failure_time > 0
    # a single point must give a wider interval than a long trajectory
    rng = np.random.default_rng(11)
    x_new = np.arange(10.0, 210.0, 10.0)
    y_new = 10 + 0.4 * x_new + rng.normal(0, 1.0, len(x_new))
    pred_long = model.predict_rul(x_new, y_new, random_state=2)
    width_one = np.diff(pred_one.failure_time_interval)[0]
    width_long = np.diff(pred_long.failure_time_interval)[0]
    assert width_one > width_long


def test_predict_rul_shrinks_toward_population():
    model = _noisy_model()
    # two points implying a slope of 1.2, way above the population
    x_new = np.array([10.0, 20.0])
    y_new = np.array([22.0, 34.0])
    pred = model.predict_rul(x_new, y_new, random_state=3)
    ls_slope = 1.2
    population_slope = model.path_param_mean[1]
    assert population_slope < pred.posterior_mean[1] < ls_slope


def test_predict_rul_already_failed_unit():
    model = _noisy_model()
    # a plausible fast unit (slope ~0.5, intercept ~10) observed well
    # past its threshold crossing at ~185
    x_new = np.array([150.0, 200.0, 250.0])
    y_new = np.array([85.0, 110.0, 135.0])
    pred = model.predict_rul(x_new, y_new, random_state=4)
    assert pred.prob_failed > 0.9
    assert pred.rul < 0


def test_predict_rul_reproducible():
    model = _noisy_model()
    pred_a = model.predict_rul([20.0, 40.0], [18.0, 26.0], random_state=5)
    pred_b = model.predict_rul([20.0, 40.0], [18.0, 26.0], random_state=5)
    assert pred_a.failure_time == pred_b.failure_time
    assert pred_a.rul_interval == pred_b.rul_interval
    assert np.array_equal(pred_a.samples, pred_b.samples)


def test_predict_rul_raises_without_measurement_noise():
    slopes = np.array([0.31, 0.28, 0.44, 0.37])
    x, y, i = linear_data(slopes)
    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    with pytest.raises(ValueError, match="measurement"):
        model.predict_rul([100.0, 200.0], [40.0, 70.0])


@pytest.mark.parametrize(
    "x_new,y_new",
    [
        ([], []),
        ([1, 2, 3], [1, 2]),
        ([100, 200], [20, np.nan]),
    ],
)
def test_predict_rul_validation(x_new, y_new):
    model = _noisy_model()
    with pytest.raises(ValueError):
        model.predict_rul(x_new, y_new)


def test_predict_rul_nonlinear_path():
    rng = np.random.default_rng(21)
    n_units, x_times = 40, np.arange(50.0, 550.0, 50.0)
    a = rng.normal(2.0, 0.05, n_units)
    b = rng.normal(0.005, 0.0003, n_units)
    x = np.tile(x_times, n_units)
    i = np.repeat(np.arange(n_units), len(x_times))
    y = np.repeat(a, len(x_times)) * np.exp(
        np.repeat(b, len(x_times)) * x
    ) + rng.normal(0, 0.1, len(x))
    model = DegradationAnalysis.fit(x, y, i, threshold=20, path="exponential")
    # new unit: a = 2, b = 0.0055, observed to 200 hours
    x_new = np.arange(25.0, 225.0, 25.0)
    y_new = 2.0 * np.exp(0.0055 * x_new)
    pred = model.predict_rul(x_new, y_new, random_state=6)
    truth = np.log(20 / 2.0) / 0.0055
    assert 0.5 * truth < pred.failure_time < 2.0 * truth
    assert pred.rul == pytest.approx(pred.failure_time - 200.0, rel=1e-9)


def test_reml_matches_moments_on_balanced_data():
    # classical result: for balanced designs (every unit measured at
    # the same times) REML coincides with the two-stage moments
    # estimator
    x, y, i = _noisy_population_data()
    moments = DegradationAnalysis.fit(x, y, i, threshold=100)
    reml = DegradationAnalysis.fit(
        x, y, i, threshold=100, population_method="reml"
    )
    assert moments.population_method == "moments"
    assert reml.population_method == "reml"
    assert np.allclose(
        reml.path_param_mean, moments.path_param_mean, rtol=1e-4
    )
    assert np.allclose(reml.path_param_cov, moments.path_param_cov, rtol=1e-3)
    assert np.isclose(reml.measurement_var, moments.measurement_var, rtol=1e-4)


def test_reml_on_unbalanced_data():
    x, y, i = _unbalanced_population_data()
    reml = DegradationAnalysis.fit(
        x, y, i, threshold=100, population_method="reml"
    )
    moments = DegradationAnalysis.fit(x, y, i, threshold=100)
    # recovers the generating population
    assert np.allclose(reml.path_param_mean, [10.0, 0.4], atol=[0.5, 0.03])
    assert np.isclose(reml.measurement_var, 1.0, rtol=0.3)
    assert np.isclose(reml.path_param_cov[0, 0], 1.0, rtol=0.6)
    assert np.isclose(reml.path_param_cov[1, 1], 0.0025, rtol=0.6)
    # a genuine optimisation: positive definite and not identical to
    # the moments starting point on unbalanced data
    assert (np.linalg.eigvalsh(reml.path_param_cov) > 0).all()
    assert not np.allclose(
        reml.path_param_cov, moments.path_param_cov, rtol=1e-6
    )
    # the fitted model supports Bayesian prediction
    pred = reml.predict_rul([20.0, 40.0], [18.0, 26.0], random_state=8)
    assert np.isfinite(pred.failure_time)


def test_reml_rejects_nonlinear_path():
    x, y, i = _noisy_population_data()
    # rejected before any per-unit fitting happens
    with pytest.raises(ValueError, match="linear in its parameters"):
        DegradationAnalysis.fit(
            x,
            y,
            i,
            threshold=100,
            path="exponential",
            population_method="reml",
        )


def test_reml_requires_noise():
    slopes = np.array([0.31, 0.28, 0.44, 0.37])
    x, y, i = linear_data(slopes)
    with pytest.raises(ValueError, match="measurement"):
        DegradationAnalysis.fit(
            x, y, i, threshold=150, population_method="reml"
        )


def test_invalid_population_method():
    slopes = np.array([0.31, 0.28, 0.44, 0.37])
    x, y, i = linear_data(slopes)
    with pytest.raises(ValueError, match="population_method"):
        DegradationAnalysis.fit(
            x, y, i, threshold=150, population_method="bayes"
        )


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
