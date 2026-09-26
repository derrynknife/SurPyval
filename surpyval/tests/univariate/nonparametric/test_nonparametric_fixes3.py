"""Regression tests for the third round of non-parametric fixes."""

import json
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.stats import kstwobign, norm  # noqa: E402

import surpyval as sp  # noqa: E402
from surpyval import NonParametric  # noqa: E402
from surpyval.univariate.nonparametric import plotting_positions  # noqa: E402


def _no_warnings(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return func(*args, **kwargs)


# -- qf / median: round-off tolerance -----------------------------------------


def test_median_of_1_to_30_is_15():
    # F at 15 is 0.4999999999999999, which used to push the median to 16.
    assert sp.KaplanMeier.fit(np.arange(1, 31)).median == 15.0


def test_qf_inverts_ff_at_the_steps():
    assert sp.KaplanMeier.fit([1, 2, 3, 4, 5]).qf(0.2)[0] == 1.0
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
    assert tb.qf(1e-12)[0] == tb.x[np.argmax(tb.F > 0)]


# -- set_lower_limit and fit_from_ecdf validation -----------------------------


@pytest.mark.parametrize("limit", [1, 2, np.nan])
def test_set_lower_limit_must_be_below_the_data(limit):
    with pytest.raises(ValueError, match="set_lower_limit"):
        sp.KaplanMeier.fit([1, 2, 3], set_lower_limit=limit)


def test_set_lower_limit_below_the_data_still_works():
    model = sp.KaplanMeier.fit([1, 2, 3], set_lower_limit=0)
    np.testing.assert_array_equal(model.x, [0, 1, 2, 3])


@pytest.mark.parametrize(
    "x, R",
    [
        ([2, 1, 3], [0.8, 0.5, 0.1]),
        ([1, 2, 3], [0.5, 0.8, 0.1]),
        ([1, 2, 3], [1.2, 0.5, 0.1]),
        ([1, 2, 3], [0.8, 0.5]),
        ([1, 2, 3], [0.8, np.nan, 0.1]),
    ],
)
def test_fit_from_ecdf_rejects_invalid_curves(x, R):
    with pytest.raises(ValueError):
        NonParametric.fit_from_ecdf(x, R)


def test_fit_from_ecdf_can_plot_without_bounds():
    model = NonParametric.fit_from_ecdf([1, 2, 3], [0.8, 0.5, 0.1])
    fig, ax = plt.subplots()
    model.plot(ax=ax, plot_bounds=False)
    plt.close(fig)
    assert model.get_plot_data(plot_bounds=False)["cbs"] is None
    with pytest.raises(ValueError, match="variance"):
        model.plot(plot_bounds=True)


# -- logrank ------------------------------------------------------------------


def test_logrank_ignores_a_group_never_at_risk():
    base = sp.logrank([1, 2, 3, 4, 5, 6], list("bbbccc"))
    extra = sp.logrank(
        [1, 2, 3, 4, 5, 6, 0.5], list("bbbccca"), c=[0] * 6 + [1]
    )
    assert extra.dof == 1
    assert extra.statistic == pytest.approx(base.statistic)
    assert extra.p_value == pytest.approx(base.p_value)
    assert extra.p_value == pytest.approx(0.0246, abs=1e-4)


def test_logrank_without_informative_groups():
    # Group 1 is censored before the first event, so every risk set holds
    # group 0 only and there is nothing to compare.
    res = sp.logrank([2, 3, 1, 1.5], [0, 0, 1, 1], c=[0, 0, 1, 1])
    assert (res.statistic, res.dof, res.p_value) == (0.0, 0, 1.0)


@pytest.mark.parametrize("label", [np.nan, None])
def test_logrank_refuses_missing_group_labels(label):
    with pytest.raises(ValueError, match="missing"):
        sp.logrank([1, 2, 3, 4], np.array([0, 0, 1, label], dtype=object))


def test_logrank_refuses_missing_strata_labels():
    with pytest.raises(ValueError, match="strata"):
        sp.logrank([1, 2, 3, 4], [0, 0, 1, 1], strata=[0, np.nan, 0, np.nan])


@pytest.mark.parametrize("arg", ["c", "n"])
def test_logrank_checks_c_and_n_lengths(arg):
    with pytest.raises(ValueError, match=f"'{arg}'"):
        sp.logrank([1, 2, 3, 4], [0, 0, 1, 1], **{arg: [0, 0, 1]})


# -- cb / plot / bootstrap argument checks ------------------------------------


def _km():
    return sp.KaplanMeier.fit([1, 2, 3, 4, 5], c=[0, 1, 0, 0, 1])


def test_cb_rejects_unknown_on():
    with pytest.raises(ValueError, match="'on'"):
        _km().cb(2, on="hf")


def test_cb_and_friends_reject_unknown_bound():
    model = _km()
    with pytest.raises(ValueError, match="'bound'"):
        model.cb(2, bound="both")
    with pytest.raises(ValueError, match="'bound'"):
        model.R_cb(2, bound="both")
    with pytest.raises(ValueError, match="'bound'"):
        model.plot(bound="both")
    with pytest.raises(ValueError, match="'bound'"):
        model.bootstrap_cb(2, bound="both", B=5)


@pytest.mark.parametrize("B", [0, -3, 2.5])
def test_bootstrap_cb_rejects_bad_B(B):
    with pytest.raises(ValueError, match="'B'"):
        _km().bootstrap_cb(2, B=B)


def test_bootstrap_cb_error_names_with_data():
    restored = sp.from_dict(_km().to_dict())
    with pytest.raises(ValueError, match="with_data=True"):
        restored.bootstrap_cb(2, B=5)


# -- Filliben with censoring --------------------------------------------------


def test_filliben_end_points_go_to_extreme_ranks_only():
    x = [1, 2, 3, 4, 5]
    N = 5
    complete = plotting_positions(x, heuristic="Filliben")[3]
    assert complete[0] == pytest.approx(1 - 0.5 ** (1 / N))
    assert complete[-1] == pytest.approx(0.5 ** (1 / N))

    # Last item censored: the failure at rank 4 keeps the interior value
    # and the censored row carries it forward (it used to get 0.8706).
    last = plotting_positions(x, c=[0, 0, 0, 0, 1], heuristic="Filliben")[3]
    interior = (4 - 0.3175) / (N + 0.365)
    np.testing.assert_allclose(last[3:], [interior, interior])

    # First item censored: 0 before the first failure, not NaN.
    first = plotting_positions(x, c=[1, 0, 0, 0, 0], heuristic="Filliben")[3]
    assert first[0] == 0
    assert np.isfinite(first).all()


def test_modal_needs_two_items():
    with pytest.raises(ValueError, match="at least two"):
        _no_warnings(plotting_positions, [5.0], heuristic="Modal")


# -- band critical values -----------------------------------------------------


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2])
def test_hall_wellner_critical_value_is_kolmogorov_over_whole_range(alpha):
    cv = NonParametric._band_critical_value(0.0, 1.0, alpha, False)
    assert cv == pytest.approx(kstwobign.ppf(1 - alpha), abs=1e-6)


def test_band_critical_values_at_a_single_point():
    z = norm.ppf(0.975)
    hw = NonParametric._band_critical_value(0.3, 0.3, 0.05, False)
    assert hw == pytest.approx(z * np.sqrt(0.3 * 0.7), rel=1e-4)
    ep = NonParametric._band_critical_value(0.3, 0.3, 0.05, True)
    assert ep == pytest.approx(z, rel=1e-4)


def test_band_critical_values_are_symmetric_in_a():
    # B(a) and B(1 - a) have the same law.
    for standardized in (False, True):
        lo = NonParametric._band_critical_value(0.05, 0.4, 0.05, standardized)
        hi = NonParametric._band_critical_value(0.6, 0.95, 0.05, standardized)
        assert lo == pytest.approx(hi, rel=1e-8)


def test_equal_precision_critical_value():
    # Klein & Moeschberger table C.4 range; Miller-Siegmund's
    # approximation gives a tail of about 0.05 here too.
    cv = NonParametric._band_critical_value(0.1, 0.9, 0.05, True)
    assert cv == pytest.approx(3.052, abs=2e-3)


def test_band_on_a_narrow_range_does_not_crash():
    one_event = sp.KaplanMeier.fit([1, 2, 3], c=[0, 1, 1]).band()
    assert np.isfinite(one_event).all()
    x = np.r_[[1, 2, 3], np.full(10000, 10)]
    c = np.r_[[0, 0, 0], np.ones(10000, int)]
    band = sp.KaplanMeier.fit(x, c=c).band()
    assert np.isfinite(band[:3]).all()


def test_band_sim_arguments_are_deprecated():
    with pytest.warns(DeprecationWarning, match="n_sims"):
        _km().band(n_sims=100)


# -- plotting restored models -------------------------------------------------


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


# -- other clear errors -------------------------------------------------------


def test_random_on_an_all_censored_model():
    with pytest.raises(ValueError, match="no failures"):
        sp.KaplanMeier.fit([1, 2], c=[1, 1]).random(3)


def test_success_run_needs_a_positive_run():
    with pytest.raises(ValueError, match="'n'"):
        sp.success_run(0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        sp.success_run(5, alpha=1.5)


def test_smoothed_hf_default_bandwidth_on_one_value():
    model = sp.KaplanMeier.fit([2, 2, 2])
    with pytest.raises(ValueError, match="single distinct value"):
        model.smoothed_hf([2])


def test_mean_rejects_negative_tau():
    with pytest.raises(ValueError, match="tau"):
        _km().mean(tau=-1)
    assert _km().mean(tau=0) == 0.0


def test_exact_infinite_value_is_refused():
    with pytest.raises(ValueError, match="finite"):
        sp.KaplanMeier.fit([1, 2, np.inf])
    with pytest.raises(ValueError, match="finite"):
        sp.Turnbull.fit([1, 2, np.inf])
    # Right censored at infinity is still accepted.
    sp.KaplanMeier.fit([1, 2, np.inf], c=[0, 0, 1])


# -- Turnbull -----------------------------------------------------------------


def test_turnbull_linear_interp_matches_kaplan_meier():
    x, c = [1, 2, 3, 4, 5, 6], [0, 1, 0, 0, 1, 0]
    tb = sp.Turnbull.fit(x, c=c, turnbull_estimator="Kaplan-Meier")
    km = sp.KaplanMeier.fit(x, c=c)
    grid = [1.5, 2.5, 3.5, 4.5, 5.5]
    np.testing.assert_allclose(
        tb.sf(grid, interp="linear"), km.sf(grid, interp="linear"), atol=1e-8
    )


def test_turnbull_healthy_truncated_fits_do_not_warn():
    # Right-truncated exact data (the Lynden-Bell estimator) and
    # left-truncated exact data (the Kaplan-Meier) are identifiable.
    rt = _no_warnings(
        sp.Turnbull.fit,
        [7, 3, 5, 2, 7, 6],
        tr=[11, 3, 7, 5, 7, 6],
        turnbull_estimator="Kaplan-Meier",
    )
    assert rt.exploitable_mass == 0.0
    x, tl = [8, 1, 4, 7, 8, 2], [5, 0, 1, 0, 7, 1]
    lt = _no_warnings(
        sp.Turnbull.fit, x, tl=tl, turnbull_estimator="Kaplan-Meier"
    )
    assert lt.exploitable_mass == 0.0
    np.testing.assert_allclose(
        lt.sf([1, 2, 4, 7]),
        sp.KaplanMeier.fit(x, tl=tl).sf([1, 2, 4, 7]),
        atol=1e-8,
    )


def test_turnbull_non_identifiable_warning_wording():
    with pytest.warns(UserWarning, match="not identifiable") as record:
        sp.Turnbull.fit(
            x=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            c=[-1, 0, 0, -1, 0, 0],
            tl=np.linspace(0.1, 1.0, 6),
            turnbull_estimator="Kaplan-Meier",
        )
    message = str(record[0].message)
    assert "needs left-censored" not in message
    assert "left- or interval-censored" in message


def test_turnbull_nelson_aalen_em_converges_on_complete_data():
    # The last risk count flipped between 9e-16 and 0 every iteration.
    model = _no_warnings(
        sp.Turnbull.fit,
        [1, 2, 3, 4],
        n=[2, 1, 2, 1],
        turnbull_estimator="Nelson-Aalen",
    )
    assert model.converged
    assert model.iters < 50
