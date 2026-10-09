import matplotlib

matplotlib.use("Agg")

import json  # noqa: E402
import warnings  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import surpyval  # noqa: E402
import surpyval as sp  # noqa: E402
from surpyval import (  # noqa: E402
    CoxPH,
    KaplanMeier,
    NelsonAalen,
    NonParametric,
)
from surpyval.tests._helpers import (  # noqa: E402
    TURNBULL_MIXED_CENSORING,
    fit_turnbull_quietly,
    no_warnings,
    sharp_drop_long_tail_data,
    small_kaplan_meier,
)
from surpyval.univariate import nonparametric as nonp  # noqa: E402
from surpyval.univariate.regression.proportional_hazards.diagnostics import (  # noqa: E402, E501
    check_ph,
)


def test_qf_uncensored():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.KaplanMeier.fit(x)
    assert np.allclose(model.qf([0.1, 0.5, 0.9]), [1.0, 3.0, 5.0])
    assert model.median == 3.0


def test_qf_not_reached_is_nan():
    # With heavy right censoring the CDF never reaches 0.5
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = np.array([0, 0, 1, 1, 1])
    model = surpyval.KaplanMeier.fit(x, c=c)
    assert np.isnan(model.qf(0.5)).all()
    assert np.isnan(model.median)


def test_611_qf_outside_unit_interval_warns_nan():
    # As every model's qf (#576): a probability outside [0, 1] is NaN
    # with one warning, not a ValueError; 0 is the first step.
    model = surpyval.KaplanMeier.fit(np.array([1.0, 2.0, 3.0]))
    assert model.qf(0.0) == 1.0
    with pytest.warns(UserWarning, match=r"outside \[0, 1\]") as caught:
        q = model.qf([-0.5, 0.5, 1.5])
    assert len(caught) == 1
    assert caught[0].filename == __file__
    np.testing.assert_array_equal(q, [np.nan, 2.0, np.nan])


def test_626_quantile_cb_outside_unit_interval_warns_nan():
    # The same rule as qf (it raised): NaN with one warning outside
    # [0, 1], the other probabilities bounded as before.
    model = surpyval.KaplanMeier.fit(np.arange(1.0, 11.0))
    with pytest.warns(UserWarning, match=r"quantile_cb: 2 of the 4") as rec:
        got = model.quantile_cb([-0.5, 0.5, 1.0, 1.5])
    assert len(rec) == 1 and rec[0].filename == __file__
    np.testing.assert_array_equal(got[1], model.quantile_cb([0.5])[0])
    assert got[2, 0] == 10.0
    assert np.isnan(got[[0, 3]]).all()


def test_quantile_cb_brookmeyer_crowley_inversion():
    # The quantile interval limits must be the first observed times at
    # which the survival bounds cross 1 - p.
    x = np.array([4.0, 7.0, 9.0, 13.0, 16.0, 21.0, 28.0, 33.0, 41.0, 50.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    model = surpyval.KaplanMeier.fit(x, c=c)
    p = 0.5
    bounds = model.cb(model.x)
    level = 1 - p
    expected_lower = model.x[np.argmax(bounds[:, 0] <= level)]
    if (bounds[:, 1] < level).any():
        expected_upper = model.x[np.argmax(bounds[:, 1] < level)]
    else:
        expected_upper = np.nan
    cb = model.quantile_cb(p)
    assert cb.shape == (2,)
    assert cb[0] == expected_lower
    assert np.isnan(cb[1]) == np.isnan(expected_upper)
    # The interval contains the point estimate when the median is reached
    assert cb[0] <= model.median


def test_quantile_cb_brackets_estimate_large_sample():
    rng = np.random.default_rng(5)
    x = rng.exponential(1.0, 200)
    model = surpyval.KaplanMeier.fit(x)
    cb = model.quantile_cb(0.5)
    assert cb[0] <= model.median <= cb[1]
    # With n=200 the median CI should be reasonably tight around ln(2)
    assert cb[0] > 0.4
    assert cb[1] < 1.1


def test_mean_uncensored_equals_sample_mean():
    # With no censoring the KM restricted mean to the largest
    # observation is the sample mean.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.KaplanMeier.fit(x)
    assert np.isclose(model.mean(), 3.0, atol=1e-12)


def test_mean_restricted_to_tau():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.KaplanMeier.fit(x)
    # integral of S over [0, 3): 1*1 + 1*0.8 + 1*0.6
    assert np.isclose(model.mean(tau=3.0), 2.4, atol=1e-12)


def test_mean_cb_matches_manual_variance():
    # Klein & Moeschberger RMST variance: sum(A_i^2 * v_i) with A_i the
    # area under S from x_i to tau and v_i the Greenwood increments.
    # For uncensored x = 1..5: A = [2.0, 1.2, 0.6, 0.2, 0] and
    # v = [1/20, 1/12, 1/6, 1/2, nan] giving var = 0.4 (the final term
    # is zero since A is zero there).
    from scipy.stats import norm

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.KaplanMeier.fit(x)
    z = norm.ppf(0.975)
    se = np.sqrt(0.4)
    cb = model.mean_cb()
    assert np.allclose(cb, [3.0 - z * se, 3.0 + z * se], atol=1e-12)


def test_mean_with_censoring_below_uncensored_mean():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = np.array([0, 0, 0, 0, 1])
    model = surpyval.KaplanMeier.fit(x, c=c)
    # S now plateaus at 0.2 after x=4; RMST to 5 is
    # 1 + .8 + .6 + .4 + .2 = 3.0
    assert np.isclose(model.mean(), 3.0, atol=1e-12)
    lower, upper = model.mean_cb()
    assert lower < model.mean() < upper


def test_mean_negative_support_raises():
    model = surpyval.KaplanMeier.fit(np.array([-1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        model.mean()


def test_bootstrap_cb_kaplan_meier():
    x = np.array([4.0, 7.0, 9.0, 13.0, 16.0, 21.0, 28.0, 33.0, 41.0, 50.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    model = surpyval.KaplanMeier.fit(x, c=c)
    x_test = [10.0, 30.0]
    cb = model.bootstrap_cb(x_test, n_boot=100, random_state=42)
    assert cb.shape == (2, 2)
    assert (cb[:, 0] <= cb[:, 1]).all()
    # Bootstrap interval should contain the point estimate
    sf = model.sf(x_test)
    assert (cb[:, 0] <= sf).all()
    assert (sf <= cb[:, 1]).all()
    # Reproducible with the same seed
    cb2 = model.bootstrap_cb(x_test, n_boot=100, random_state=42)
    assert np.allclose(cb, cb2)


def test_bootstrap_cb_turnbull():
    left = np.array([1, 8, 8, 7, 7, 17, 37, 46, 46, 45.0])
    right = np.array([7, 8, 10, 16, 14, np.inf, 44, np.inf, np.inf, np.inf])
    model = surpyval.Turnbull.fit(xl=left, xr=right)
    cb = model.bootstrap_cb([10.0, 20.0], n_boot=30, random_state=7)
    assert cb.shape == (2, 2)
    assert (cb[:, 0] <= cb[:, 1]).all()
    sf = model.sf([10.0, 20.0])
    assert (cb[:, 0] <= sf + 1e-12).all()
    assert (sf <= cb[:, 1] + 1e-12).all()


def test_bootstrap_cb_one_sided():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.KaplanMeier.fit(x)
    lower = model.bootstrap_cb([2.5], bound="lower", n_boot=50, random_state=1)
    upper = model.bootstrap_cb([2.5], bound="upper", n_boot=50, random_state=1)
    assert lower.shape == (1,)
    assert (lower <= upper).all()
    with pytest.raises(ValueError):
        model.bootstrap_cb([2.5], bound="middle")


def test_bootstrap_cb_requires_data():
    model = surpyval.KaplanMeier.from_xrd([1, 2, 3], [10, 8, 6], [2, 1, 1])
    with pytest.raises(ValueError, match="data"):
        model.bootstrap_cb([2.0])


def test_plot_with_bounds_and_censors():
    x = np.array([4.0, 7.0, 9.0, 13.0, 16.0, 21.0, 28.0, 33.0, 41.0, 50.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    model = surpyval.KaplanMeier.fit(x, c=c)
    fig, ax = plt.subplots()
    out = model.plot(ax=ax, color="C2", label="KM")
    assert out is ax
    # survival curve + censor markers, and the shaded bound band
    assert len(ax.lines) == 2
    assert len(ax.collections) == 1
    assert ax.lines[0].get_color() == "C2"
    plt.close(fig)


def test_plot_without_bounds_or_censors():
    model = surpyval.KaplanMeier.fit(np.array([1.0, 2.0, 3.0]))
    fig, ax = plt.subplots()
    model.plot(ax=ax, plot_bounds=False, show_censors=False)
    assert len(ax.lines) == 1
    assert len(ax.collections) == 0
    plt.close(fig)


# ---------------------------------------------------------------------------
# ``random`` is seedable and draws from the observed values.
# ---------------------------------------------------------------------------


def test_random_is_reproducible_with_seed():
    model = surpyval.KaplanMeier.fit(sharp_drop_long_tail_data())
    a = model.random(500, random_state=42)
    b = model.random(500, random_state=42)
    c = model.random(500, random_state=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_draws_from_observed_values():
    model = surpyval.KaplanMeier.fit(sharp_drop_long_tail_data())
    draws = model.random(1000, random_state=1)
    assert set(np.unique(draws)).issubset(set(model.x))
    assert draws.size == 1000


# ---------------------------------------------------------------------------
# #408: ``df`` where the estimate reaches zero.
# ---------------------------------------------------------------------------


def test_df_is_the_step_probability_where_the_estimate_reaches_zero():
    model = sp.KaplanMeier.fit([1.0, 2.0, 3.0])
    df = no_warnings(model.df, [1.5, 2.5, 3.5])
    # Each step of 1 takes 1/3; past 3 the last jump is carried, as hf
    # carries the infinite jump to zero.
    np.testing.assert_allclose(df, [1 / 3, 1 / 3, 1 / 3], rtol=1e-12)
    np.testing.assert_allclose(no_warnings(model.df, 3.5), 1 / 3)
    assert np.all(np.isposinf(model.hf([2.5, 3.5])))


@pytest.mark.parametrize("interp", ["step", "linear", "cubic"])
def test_df_is_finite_and_a_probability(interp):
    model = sp.KaplanMeier.fit([1, 2, 3, 4, 5])
    df = no_warnings(model.df, [1.0, 2.0, 3.0, 4.0, 5.0], interp=interp)
    assert np.all(np.isfinite(df)) and np.all((df >= 0) & (df <= 1))


def test_df_is_the_drop_in_sf_over_the_step_hf_differences():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    c = np.array([0, 1, 0, 0, 1, 0, 0, 1])
    model = sp.KaplanMeier.fit(x, c=c)
    q = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 6.5])
    sf, hf, df = model.sf(q), model.hf(q), model.df(q)
    # 0.5 repeats 1.5; 2.5 has no failure since 1.5 and takes 1.5's step.
    before = np.array([1.0, 1.0, 1.0, sf[2], sf[3], sf[4]])
    after = np.array([sf[1], sf[1], sf[1], sf[3], sf[4], sf[5]])
    np.testing.assert_allclose(df, before - after, rtol=1e-12)
    # The same step as hf: df = S (1 - exp(-hf)).
    np.testing.assert_allclose(df, before * -np.expm1(-hf), rtol=1e-12)


def test_df_with_a_support_is_zero_before_the_first_value():
    model = sp.KaplanMeier.fit([1.0, 2.0, 3.0]).set_support(0, 10)
    df = no_warnings(model.df, [-1.0, 0.5, 2.5, 5.0])
    # From 0.5 (sf 1) to 2.5 (sf 1/3), then to 5 (sf 0).
    np.testing.assert_allclose(df, [np.nan, 0.0, 2 / 3, 1 / 3])


# ---------------------------------------------------------------------------
# ``bootstrap_cb`` refits Turnbull resamples with the fit's
# ``tol`` and ``max_iter``.
# ---------------------------------------------------------------------------


class TestBootstrapSettings:
    def _record(self, monkeypatch):
        calls = []
        real = nonp.turnbull

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(nonp, "turnbull", spy)
        return calls

    def test_resamples_use_fit_settings(self, monkeypatch):
        model = fit_turnbull_quietly(
            **TURNBULL_MIXED_CENSORING,
            turnbull_estimator="Nelson-Aalen",
            tol=1e-6,
            max_iter=7,
        )
        calls = self._record(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.bootstrap_cb([6.0], n_boot=3, random_state=1)
        assert len(calls) == 3
        for kwargs in calls:
            assert kwargs["estimator"] == "Nelson-Aalen"
            assert kwargs["tol"] == 1e-6
            assert kwargs["max_iter"] == 7

    def test_settings_survive_serialisation(self, monkeypatch):
        model = fit_turnbull_quietly(
            **TURNBULL_MIXED_CENSORING, tol=1e-8, max_iter=50
        )
        restored = NonParametric.from_dict(model.to_dict(with_data=True))
        assert restored.data["tol"] == 1e-8
        assert restored.data["max_iter"] == 50
        calls = self._record(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            restored.bootstrap_cb([6.0], n_boot=2, random_state=1)
        assert all(k["tol"] == 1e-8 and k["max_iter"] == 50 for k in calls)


# ---------------------------------------------------------------------------
# ``qf`` and the median with round-off; ``bootstrap_cb``
# argument checks; a restored Turnbull model plots; ``random``
# on an all-censored model; ``mean`` refuses a negative
# ``tau``.
# ---------------------------------------------------------------------------


def test_median_of_1_to_30_is_15():
    # F at 15 is 0.4999999999999999, which used to push the median to 16.
    assert sp.KaplanMeier.fit(np.arange(1, 31)).median == 15.0


def test_qf_inverts_ff_at_the_steps():
    assert sp.KaplanMeier.fit([1, 2, 3, 4, 5]).qf(0.2) == 1.0
    for N in range(2, 60):
        model = sp.KaplanMeier.fit(np.arange(1, N + 1))
        x = model.x[:-1]
        np.testing.assert_array_equal(model.qf(model.ff(x)), x)


def test_qf_agrees_between_kaplan_meier_and_turnbull():
    x, tl = [2, 3, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2]
    km = sp.KaplanMeier.fit(x, tl=tl)
    tb = sp.Turnbull.fit(x, tl=tl, turnbull_estimator="Kaplan-Meier")
    p = [0.25, 0.5, 0.7]
    np.testing.assert_array_equal(km.qf(p), tb.qf(p))


def test_qf_tiny_p_does_not_match_a_zero_cdf():
    # A Turnbull ladder starts with F exactly 0; the tolerance must not
    # make a p below it match there.
    tb = sp.Turnbull.fit([2, 3, 4], turnbull_estimator="Kaplan-Meier")
    assert tb.F[0] == 0
    assert tb.qf(1e-12) == tb.x[np.argmax(tb.F > 0)]


@pytest.mark.parametrize("n_boot", [0, -3, 2.5])
def test_bootstrap_cb_rejects_bad_n_boot(n_boot):
    with pytest.raises(ValueError, match="'n_boot'"):
        small_kaplan_meier().bootstrap_cb(2, n_boot=n_boot)


def test_bootstrap_cb_error_names_with_data():
    restored = sp.from_dict(small_kaplan_meier().to_dict())
    with pytest.raises(ValueError, match="with_data=True"):
        restored.bootstrap_cb(2, n_boot=5)


def test_restored_turnbull_model_plots():
    tb = sp.Turnbull.fit([1, 2, 3, 4, 5, 6], c=[0, 1, 0, 0, 1, 0])
    restored = sp.from_dict(json.loads(json.dumps(tb.to_dict())))
    fig, ax = plt.subplots()
    restored.plot(ax=ax)
    with pytest.raises(ValueError, match="with_data=True"):
        restored.plot(ax=ax, show_censors=True)
    with_data = sp.from_dict(
        json.loads(json.dumps(tb.to_dict(with_data=True)))
    )
    n_lines = len(ax.lines)
    with_data.plot(ax=ax)
    # The curve and the censoring marks.
    assert len(ax.lines) == n_lines + 2
    plt.close(fig)


def test_random_on_an_all_censored_model():
    # No failure within the data: the estimate stays at 1, so every
    # lifetime drawn from it lies beyond the data (inf).
    draws = sp.KaplanMeier.fit([1, 2], c=[1, 1]).random(3)
    assert np.all(np.isposinf(draws))


def test_mean_rejects_negative_tau():
    with pytest.raises(ValueError, match="tau"):
        small_kaplan_meier().mean(tau=-1)
    assert small_kaplan_meier().mean(tau=0) == 0.0


# ---------------------------------------------------------------------------
# #282: scalar inputs and a single observation.
# ---------------------------------------------------------------------------


class TestNonParametricScalars:
    def test_scalar_hf_df_finite(self):
        na = NelsonAalen.fit([1.0, 2, 3, 4, 5])
        assert float(np.ravel(na.hf(2.5))[0]) == pytest.approx(0.25)
        assert np.isfinite(float(np.ravel(na.df(2.5))[0]))
        # Matches what the array path returns for the same point.
        grid = np.ravel(na.hf([1.5, 2.5]))
        assert float(np.ravel(na.hf(2.5))[0]) == pytest.approx(grid[1])

    def test_single_observation_cb_drawable(self):
        km = KaplanMeier.fit([5.0])
        cb = np.asarray(km.cb([5.0], on="sf"), dtype=float)
        assert np.all(np.isfinite(cb))

    def test_check_ph_no_spurious_truncation_warning(self):
        import warnings as _w

        np.random.seed(7)
        x = np.random.exponential(2.0, 60)
        Z = np.random.normal(size=(60, 1))
        m = CoxPH.fit(x=x, Z=Z, tl=np.zeros(60))
        with _w.catch_warnings():
            _w.simplefilter("error")
            check_ph(m)


@pytest.mark.parametrize(
    "name", ["KaplanMeier", "NelsonAalen", "FlemingHarrington", "Turnbull"]
)
def test_728_Hf_before_the_first_time_is_plus_zero(name):
    # -log(1) is -0.0; the cumulative hazard there is 0.0 (#728). The
    # first value is censored, so the stored H starts at sf = 1 too.
    model = getattr(sp, name).fit([1.0, 2, 3, 4, 5], c=[1, 0, 0, 1, 0])
    H = model.Hf([0.0, 0.5, 1.0, 3.0])
    assert np.all(H[:3] == 0) and not np.any(np.signbit(H))
    assert not np.signbit(model.Hf(0.5))
    assert H[3] > 0
    assert not np.any(np.signbit(model.H))


@pytest.mark.parametrize("bound", ["two-sided", "lower", "upper"])
def test_746_Hf_bounds_before_the_first_failure_are_plus_zero(bound):
    # The survival bounds are 1 at the censored first value; -log(1) of
    # them was -0.0 (#746).
    model = sp.KaplanMeier.fit([1.0, 2, 3, 4, 5], c=[1, 0, 0, 0, 0])
    b = model.cb([0.5, 1.0, 1.5], on="Hf", bound=bound)
    assert np.all(b == 0) and not np.any(np.signbit(b))
