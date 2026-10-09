"""
Tests for the nonparametric competing-risks (Aalen-Johansen) estimator.

Covers the #253 fixes: the incidence increment must weight the
cause-specific hazard by the survival just *before* each event time
(S(t-)), queries before the first observed time must return the
zero/one boundary values instead of wrapping to the last step, and
``fit_from_df`` must not shadow the ``df`` (density) method.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval.univariate.competing_risks.nonparametric.competing_risks import (  # noqa: E501
    CompetingRisks,
)


def test_single_cause_km_cif_reaches_one():
    # With one cause and no censoring the KM-weighted Aalen-Johansen CIF
    # is 1 - KM, which reaches exactly 1 at the last event time.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    e = np.array(["a"] * 5)
    cr = CompetingRisks.fit(x=x, e=e, how="Kaplan-Meier")
    assert cr.cif(np.array([5.0]), "a")[0] == pytest.approx(1.0, abs=1e-12)


def test_two_cause_cifs_sum_to_one_minus_km():
    rng = np.random.default_rng(1)
    n = 400
    t1 = rng.weibull(2, n) * 10
    t2 = rng.weibull(1.5, n) * 12
    t = np.minimum(t1, t2)
    ev = np.where(t1 < t2, "a", "b")
    cens = t > 15
    tt = np.minimum(t, 15.0)
    c = cens.astype(int)
    e = np.array(
        [ev[i] if not cens[i] else None for i in range(n)], dtype=object
    )
    cr = CompetingRisks.fit(x=tt, e=e, c=c, how="Kaplan-Meier")

    q = np.array([2.0, 5.0, 10.0, 14.0])
    total = cr.cif(q, "a") + cr.cif(q, "b")
    idx = np.searchsorted(cr.x, q, side="right") - 1
    assert np.allclose(total, 1.0 - cr.S[idx], atol=1e-12)


def test_queries_before_first_event_are_boundary_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    e = np.array(["a"] * 5)
    cr = CompetingRisks.fit(x=x, e=e)
    assert cr.sf(np.array([0.1]))[0] == 1.0
    assert cr.Hf(np.array([0.1]))[0] == 0.0
    assert cr.hf(np.array([0.1]))[0] == 0.0
    assert cr.cif(np.array([0.1]), "a")[0] == 0.0
    assert cr.iif(np.array([0.1]), "a")[0] == 0.0


def test_fit_from_df_does_not_shadow_density_method():
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": ["a", "b", "a", "b"],
        }
    )
    model = CompetingRisks.fit_from_df(frame, x_col="time", e_col="event")
    assert model.source_df is frame
    # ``df`` must still be the density method.
    vals = model.df(np.array([2.5]))
    assert np.isfinite(vals).all()


def test_the_requested_survival_estimator_is_reported():
    # method only changed the stored S array; sf/ff/Hf always used exp(-H)
    import json

    from surpyval import KaplanMeier, NelsonAalen
    from surpyval.univariate.competing_risks import CompetingRisks

    rng = np.random.default_rng(0)
    t1 = rng.exponential(2.0, 60)
    t2 = rng.exponential(3.0, 60)
    x = np.minimum(t1, t2).round(1)
    e = np.where(t1 < t2, 1, 2)
    q = np.array([0.5, 1.0, 2.0])

    km = CompetingRisks.fit(x, e, how="Kaplan-Meier")
    na = CompetingRisks.fit(x, e)
    assert np.allclose(km.sf(q), KaplanMeier.fit(x).sf(q))
    assert np.allclose(na.sf(q), NelsonAalen.fit(x).sf(q))
    assert np.allclose(km.ff(q), 1 - km.sf(q))
    assert np.allclose(km.sf(q), np.exp(-km.Hf(q)))
    # one cause's (net) survival: product limit of its own increments
    d1 = np.array([(x[e == 1] == t).sum() for t in km.x])
    net = np.cumprod(1 - d1 / km.r)
    assert np.allclose(km.sf(km.x, 1), net)

    restored = CompetingRisks.from_dict(json.loads(json.dumps(km.to_dict())))
    assert restored.how == "Kaplan-Meier"
    assert np.allclose(restored.sf(q, 2), km.sf(q, 2))


# ---------------------------------------------------------------------------
# The query shape is kept; empty data are refused.
# ---------------------------------------------------------------------------


def test_nonparametric_sf_keeps_the_query_shape():
    model = CompetingRisks.fit([1, 2, 3], ["a", "b", "a"])
    query = np.array([[1, 2], [3, 4]])
    out = model.sf(query)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out.ravel(), model.sf(query.ravel()))
    assert model.cif(query, "a").shape == (2, 2)


def test_empty_data_is_refused():
    with pytest.raises(ValueError):
        CompetingRisks.fit([], [])


# ---------------------------------------------------------------------------
# #278: the Aalen-Johansen CIF uses product-limit weights
# (CIF <= 1).
# ---------------------------------------------------------------------------


class TestCIFProductLimit:
    def test_na_method_cif_sums_to_one(self):
        # 278: default Nelson-Aalen method paired d/r with exp(-H) and
        # the total incidence exceeded 1 (1.216 here).
        cr = CompetingRisks.fit(
            x=[1, 1, 1, 2, 2, 3], e=["a", "a", "b", "a", "b", "a"]
        )
        total = float(np.ravel(cr.cif(3, "a"))[0]) + float(
            np.ravel(cr.cif(3, "b"))[0]
        )
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_extreme_case_capped_at_one(self):
        cr = CompetingRisks.fit(x=[1] * 9 + [2], e=["a"] * 10)
        assert float(np.ravel(cr.cif(2, "a"))[0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# #728: confidence bounds (``cb``) on the cumulative incidence, and on the
# all-cause and net sf / ff / Hf.
# ---------------------------------------------------------------------------
X10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
E10 = ["a", "b", "a", None, "a", "b", "a", None, "b", "a"]
Z95 = 1.959963984540054


class TestCIFBounds:
    def test_normal_bounds_use_aalens_variance(self):
        # Three units failing from a, b, a: the variance of F_a, by hand
        # (cmprsk's formula), is 1/9, 1/9 and 1/4 at times 1, 2, 3.
        model = CompetingRisks.fit([1, 2, 3], ["a", "b", "a"])
        b = model.cb([1, 2, 3], "a", bound_type="normal")
        F = np.array([1, 1, 2]) / 3
        sd = np.sqrt([1 / 9, 1 / 9, 1 / 4])
        np.testing.assert_allclose(b[:, 0], F - Z95 * sd, rtol=1e-12)
        np.testing.assert_allclose(b[:, 1], F + Z95 * sd, rtol=1e-12)

    def test_exp_bounds_are_on_the_log_minus_log_scale(self):
        model = CompetingRisks.fit(X10, E10)
        F = model.cif(5, "a")
        # (One side at 0.025 is the two-sided 95% end.)
        upper = model.cb(
            5, "a", alpha_ci=0.025, bound="upper", bound_type="normal"
        )
        sd = (upper - F) / Z95
        s = Z95 * sd / (F * abs(np.log(F)))
        np.testing.assert_allclose(
            model.cb(5, "a"), [F ** np.exp(s), F ** np.exp(-s)], rtol=1e-12
        )

    def test_bounds_contain_the_estimate_and_stay_in_the_unit_interval(self):
        model = CompetingRisks.fit(X10, E10)
        q = np.linspace(0, 10, 41)
        for cause in ("a", "b"):
            F = model.cif(q, cause)
            b = model.cb(q, cause)
            assert np.all((b[:, 0] <= F) & (F <= b[:, 1]))
            assert np.all((b >= 0) & (b <= 1))

    def test_one_sided_bound_is_an_end_of_the_two_sided(self):
        model = CompetingRisks.fit(X10, E10)
        q = [2, 5, 9]
        two = model.cb(q, "b", alpha_ci=0.2)
        np.testing.assert_allclose(
            model.cb(q, "b", alpha_ci=0.1, bound="lower"), two[:, 0]
        )
        np.testing.assert_allclose(
            model.cb(q, "b", alpha_ci=0.1, bound="upper"), two[:, 1]
        )

    def test_zero_before_the_first_time_and_the_cause_first_event(self):
        # The first time is a's event; b's first is at 2. Both bounds are
        # exactly 0 (not -0.0) where the incidence is.
        model = CompetingRisks.fit(X10, E10)
        b = model.cb([-5, 0.5, 1, 1.5], "b")
        assert np.all(b == 0) and not np.signbit(b).any()
        assert np.all(model.cb(0.5, "a") == 0)
        np.testing.assert_array_equal(model.cb(0.5, on="sf"), [1.0, 1.0])

    def test_nan_past_the_last_time_with_one_warning(self):
        model = CompetingRisks.fit(X10, E10)
        with pytest.warns(UserWarning, match="past the last observed") as w:
            b = model.cb([5, 11, 12], "a")
        assert len(w) == 1
        assert "11, 12" in str(w[0].message)
        assert "cif there only holds" in str(w[0].message)
        assert np.isnan(b[1:]).all() and np.isfinite(b[0]).all()
        # Both point at the caller's line.
        assert w[0].filename == __file__
        with pytest.warns(UserWarning, match="past the last observed") as w:
            assert np.isnan(model.cb(11, on="sf")).all()
        assert len(w) == 1 and w[0].filename == __file__

    def test_support_carries_the_last_bounds(self):
        model = CompetingRisks.fit(X10, E10).set_support(0, 20)
        b = model.cb([-1, 0, 10, 15, 25], "a")
        assert np.isnan(b[[0, 4]]).all()
        np.testing.assert_array_equal(b[1], [0.0, 0.0])
        np.testing.assert_array_equal(b[3], b[2])
        b = model.cb([-1, 0, 15], on="sf")
        assert np.isnan(b[0]).all()
        np.testing.assert_array_equal(b[1], [1.0, 1.0])

    def test_the_same_whichever_survival_estimator(self):
        # The incidence is Aalen-Johansen either way, and so its bounds.
        q = np.linspace(0, 10, 21)
        na = CompetingRisks.fit(X10, E10)
        km = CompetingRisks.fit(X10, E10, how="Kaplan-Meier")
        np.testing.assert_array_equal(na.cb(q, "a"), km.cb(q, "a"))

    def test_a_cif_that_reaches_one(self):
        # One cause, no censoring: F reaches 1 at the last time, where
        # the log(-log) scale has no interval. The upper bound is 1 and
        # the lower the largest before.
        model = CompetingRisks.fit([1, 2, 3, 4, 5], ["a"] * 5)
        b = model.cb([1, 2, 3, 4, 5], "a")
        assert np.isfinite(b).all()
        assert b[-1, 1] == 1.0 and b[-1, 0] == b[:-1, 0].max()
        assert np.all(b[:-1, 0] <= b[:-1, 1])

    def test_shape_and_serialisation(self):
        import json

        model = CompetingRisks.fit(X10, E10)
        assert model.cb(5, "a").shape == (2,)
        assert model.cb(5, "a", bound="lower").shape == ()
        assert model.cb([[1, 2], [3, 4]], "a").shape == (2, 2, 2)
        restored = CompetingRisks.from_dict(
            json.loads(json.dumps(model.to_dict()))
        )
        np.testing.assert_array_equal(
            restored.cb([2, 5, 9], "b"), model.cb([2, 5, 9], "b")
        )


@pytest.mark.parametrize("how", ["Nelson-Aalen", "Kaplan-Meier"])
@pytest.mark.parametrize("on", ["sf", "ff", "Hf"])
def test_survival_bounds_are_the_single_event_estimates(how, on):
    # All causes: every failure is an event; one cause's net function:
    # the other causes are censored.
    import surpyval as sp

    fitter = sp.KaplanMeier if how == "Kaplan-Meier" else sp.NelsonAalen
    model = CompetingRisks.fit(X10, E10, how=how)
    e = np.array(E10, dtype=object)
    q = [0.5, 2, 5, 9.5]
    for event in (None, "a"):
        failed = np.array([v is not None for v in E10])
        c = ~failed if event is None else e != event
        single = fitter.fit(X10, c=c.astype(int))
        b = model.cb(q, event, on=on)
        np.testing.assert_allclose(b, single.cb(q, on=on), rtol=1e-12)
        # And the estimate inside them is the model's own.
        f = getattr(model, on)(q, event=event)
        assert np.all((b[:, 0] <= f + 1e-12) & (f <= b[:, 1] + 1e-12))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"event": "a", "on": "hf"}, "'on' must be one of"),
        ({"event": None}, "pass `event`"),
        ({"event": "c"}, "c"),
        ({"event": "a", "bound": "both"}, "'bound' must be one of"),
        ({"event": "a", "bound_type": "log"}, "'bound_type' must be one of"),
        ({"event": "a", "alpha_ci": 1.5}, "'alpha_ci' must be strictly"),
    ],
)
def test_cb_refuses_bad_arguments(kwargs, match):
    model = CompetingRisks.fit(X10, E10)
    with pytest.raises(ValueError, match=match):
        model.cb([1, 2], **kwargs)


def test_cb_warns_of_a_confidence_given_for_alpha_ci():
    model = CompetingRisks.fit(X10, E10)
    for on in ("cif", "sf"):
        with pytest.warns(UserWarning, match="alpha_ci") as w:
            model.cb([1, 2], "a", on=on, alpha_ci=0.95)
        assert len(w) == 1


@pytest.mark.parametrize("how", ["Kaplan-Meier", "Nelson-Aalen"])
def test_728_Hf_before_the_first_time_is_plus_zero(how):
    # -log(1) was -0.0 with Kaplan-Meier, before the first time and
    # before a cause's own first event (#728).
    x = [1.0, 2, 3, 4, 5, 6]
    e = [1, 2, None, 1, 2, 1]
    c = [0, 0, 1, 0, 0, 0]
    model = CompetingRisks.fit(x, e, c, how=how)
    q = np.array([0.0, 0.5, 1.0])
    for event in (None, 1, 2):
        H = model.Hf(q, event=event)
        assert not np.any(np.signbit(H))
    assert np.all(model.Hf(q[:2]) == 0)
    assert model.Hf(1.0, event=2) == 0 and not np.signbit(model.Hf(0.0))


def _band_values(collection):
    # The y values of a fill_between band's outline.
    return np.unique(collection.get_paths()[0].vertices[:, 1])


def test_746_plot_draws_the_cif_bounds():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from surpyval import KaplanMeier

    model = CompetingRisks.fit(X10, E10)
    # Unstacked: a band per cause, the cause's cb, in its curve's colour.
    _, ax = plt.subplots()
    model.plot(stacked=False, ax=ax, alpha_ci=0.1)
    assert len(ax.collections) == 2
    for line, band, cause in zip(ax.lines, ax.collections, ["a", "b"]):
        cb = model.cb(model.x, cause, alpha_ci=0.1)
        np.testing.assert_allclose(
            _band_values(band), np.unique(np.concatenate([[0.0], *cb.T]))
        )
        np.testing.assert_allclose(
            band.get_facecolor()[0][:3],
            matplotlib.colors.to_rgb(line.get_color()),
        )
    # Stacked (the default): one band, on the all-cause failure
    # probability at the top of the stack, as KaplanMeier's.
    _, ax = plt.subplots()
    model.plot(ax=ax)
    assert len(ax.collections) == 3
    km = KaplanMeier.from_xrd(model.x, model.r, model.d)
    cb = km.cb(model.x, on="ff")
    np.testing.assert_allclose(
        _band_values(ax.collections[-1]),
        np.unique(np.concatenate([[0.0], *cb.T])),
    )
    # A one-sided bound is a dashed step line.
    _, ax = plt.subplots()
    model.plot(stacked=False, ax=ax, bound="upper")
    assert len(ax.lines) == 4 and not ax.collections
    np.testing.assert_allclose(
        ax.lines[1].get_ydata()[1:], model.cb(model.x, "a", bound="upper")
    )
    # plot_bounds=False draws the curves alone.
    _, ax = plt.subplots()
    model.plot(stacked=False, ax=ax, plot_bounds=False)
    assert len(ax.lines) == 2 and not ax.collections
    _, ax = plt.subplots()
    model.plot(ax=ax, plot_bounds=False)
    assert len(ax.collections) == 2
    with pytest.raises(ValueError, match="bound"):
        model.plot(ax=ax, bound="both")
    plt.close("all")


@pytest.mark.parametrize("on", ["hf", "df", "iif"])
def test_746_no_bounds_on_the_jumps_as_for_a_single_event(on):
    # By design: the single-event estimates refuse 'hf' and 'df' too.
    from surpyval import KaplanMeier

    model = CompetingRisks.fit(X10, E10)
    with pytest.raises(ValueError, match="jumps of the step estimate"):
        model.cb([1, 2], "a", on=on)
    if on != "iif":
        with pytest.raises(ValueError, match="'on' must be one of"):
            KaplanMeier.fit(X10).cb([1, 2], on=on)
