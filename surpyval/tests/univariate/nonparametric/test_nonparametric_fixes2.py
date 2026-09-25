"""
Regression tests for the second round of non-parametric fixes:

1. The Turnbull variance ladder is aligned with the survival estimate, so
   ``cb()`` no longer steps a piece early on interval-censored data.
2. ``smoothed_hf`` works on Turnbull models (they now carry ``H``).
3. ``bootstrap_cb`` refits Turnbull resamples with the fit's ``tol`` and
   ``max_iter``.
4. Greenwood's variance treats round-off in the proportion failing as
   exact, so the last value is undefined rather than ~1e14.
5. ``'Benard'`` plotting positions use Benard's (i - 0.3) / (N + 0.4).
6. The ``turnbull_estimator`` docstring says where the option acts.
7. An unknown ``turnbull_estimator`` raises a clear ``ValueError``.
"""

import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval import KaplanMeier, NonParametric, Turnbull
from surpyval.univariate import nonparametric as nonp
from surpyval.univariate.nonparametric import (
    greenwood_variance,
    nelson_aalen_variance,
    plotting_positions,
)

# The 11-row mixed-censoring example from the docs review.
MIXED = dict(
    x=[1, 2, [3, 6], 7, 8, 9, [5, 9], [4, 10], [7, 10], 11, 12],
    c=[1, 1, 2, 0, 0, 0, 2, 2, 2, -1, 0],
    n=[1, 2, 1, 3, 2, 2, 1, 1, 2, 1, 1],
)


def _fit(**kwargs):
    with warnings.catch_warnings():
        # Some of these fits are deliberately slow to converge; the
        # warning is not what is under test.
        warnings.simplefilter("ignore", UserWarning)
        return Turnbull.fit(**kwargs)


class TestVarianceAlignment:
    def test_interval_censored_bounds_where_estimate_is_one(self):
        # Before the fix, cb(1) was [0, 1] while sf(1) was 1: the variance
        # at 1 already held the expected failures in (1, 2].
        model = _fit(
            xl=[0, 1, 2, 3],
            xr=[2, 3, 4, 5],
            turnbull_estimator="Kaplan-Meier",
        )
        assert model.sf(1)[0] == 1.0
        np.testing.assert_allclose(model.cb(1), [[1.0, 1.0]])

    def test_mixed_example_bounds_where_estimate_is_one(self):
        model = _fit(**MIXED, turnbull_estimator="Kaplan-Meier")
        assert model.sf(5.5)[0] == 1.0
        np.testing.assert_allclose(model.cb(5.5), [[1.0, 1.0]])

    @pytest.mark.parametrize(
        "estimator", ["Kaplan-Meier", "Nelson-Aalen", "Fleming-Harrington"]
    )
    def test_reported_ladder_generates_the_estimate(self, estimator):
        # The r and d reported with the model are the ones behind R (and,
        # without truncation, behind the variance) at the same x.
        model = _fit(**MIXED, turnbull_estimator=estimator)
        np.testing.assert_allclose(
            nonp.FIT_FUNCS[estimator](model.r, model.d), model.R, atol=1e-12
        )
        np.testing.assert_allclose(
            nonp.VAR_FUNCS[estimator](model.r, model.d),
            model.greenwood,
            equal_nan=True,
        )

    def test_variance_zero_exactly_where_estimate_is_one(self):
        model = _fit(**MIXED, turnbull_estimator="Kaplan-Meier")
        at_one = model.R == 1.0
        assert at_one.any() and (~at_one).any()
        assert (model.greenwood[at_one] == 0).all()
        assert (
            model.greenwood[~at_one & np.isfinite(model.greenwood)] > 0
        ).all()

    def test_truncated_interval_censored_bounds(self):
        # Same slicing on the truncated (observed-count) variance ladder.
        model = _fit(
            xl=[0, 1, 2, 3, 1],
            xr=[2, 3, 4, 5, 4],
            tl=[-1, -1, 0.5, 0.5, 0],
            turnbull_estimator="Kaplan-Meier",
            max_iter=20_000,
        )
        # The EM leaves the estimate 1 up to round-off before 2.
        at_one = model.R > 1 - 1e-12
        assert at_one.sum() == 3
        np.testing.assert_allclose(model.cb(model.x[at_one]), 1.0)
        assert (model.greenwood[at_one] == 0).all()
        # And the bounds bracket the estimate everywhere they are defined.
        cb = model.cb(model.x)
        assert (cb[:, 0] <= model.R + 1e-12).all()
        assert (cb[:, 1] >= model.R - 1e-12).all()

    def test_mass_only_on_innermost_intervals(self):
        # Without truncation the NPMLE puts no mass outside Turnbull's
        # innermost intervals; restricting the EM to them removes the
        # slowly-decaying residual mass (d ~ 2e-8 on (4, 5] here) that made
        # the estimate 1 - 1e-9 and its log(-log) bounds [0, 1].
        model = _fit(**MIXED, turnbull_estimator="Kaplan-Meier")
        assert model.converged
        np.testing.assert_array_equal(model.d[model.x <= 5], 0.0)
        np.testing.assert_array_equal(model.R[model.x <= 5], 1.0)

    def test_right_censored_still_matches_kaplan_meier(self):
        x = [1, 2, 2, 3, 4, 5, 5, 6, 7, 9]
        c = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
        tb = _fit(x=x, c=c, turnbull_estimator="Kaplan-Meier")
        km = KaplanMeier.fit(x=x, c=c)
        grid = np.linspace(1, 9, 33)
        np.testing.assert_allclose(tb.sf(grid), km.sf(grid), atol=1e-10)
        np.testing.assert_allclose(tb.cb(grid), km.cb(grid), atol=1e-10)

    def test_left_truncated_still_matches_kaplan_meier(self):
        x = [2, 3, 3, 4, 5, 6, 7, 8]
        tl = [0, 0, 1, 1, 2, 2, 3, 0]
        tb = _fit(x=x, tl=tl, turnbull_estimator="Kaplan-Meier")
        km = KaplanMeier.fit(x=x, tl=tl)
        grid = np.linspace(2, 8, 25)
        np.testing.assert_allclose(tb.sf(grid), km.sf(grid), atol=1e-9)
        np.testing.assert_allclose(tb.cb(grid), km.cb(grid), atol=1e-9)


class TestSmoothedHazard:
    @pytest.mark.parametrize(
        "estimator", ["Kaplan-Meier", "Nelson-Aalen", "Fleming-Harrington"]
    )
    def test_smoothed_hf_on_turnbull(self, estimator):
        model = _fit(**MIXED, turnbull_estimator=estimator)
        with np.errstate(divide="ignore"):
            np.testing.assert_allclose(model.H, -np.log(model.R))
        h = model.smoothed_hf([4.0, 7.0, 10.0], bandwidth=3)
        assert np.isfinite(h).all() and (h >= 0).all()

    def test_matches_kaplan_meier_on_right_censored_data(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        c = [0, 1, 0, 0, 1, 0, 0, 1]
        tb = _fit(x=x, c=c, turnbull_estimator="Kaplan-Meier")
        km = KaplanMeier.fit(x=x, c=c)
        t = [2.0, 4.0, 6.0]
        np.testing.assert_allclose(
            tb.smoothed_hf(t, bandwidth=2),
            km.smoothed_hf(t, bandwidth=2),
            atol=1e-9,
        )

    def test_restored_legacy_turnbull_dict(self):
        # Turnbull models serialised before they carried H stored it as
        # None; the restored model still supports smoothed_hf.
        model = _fit(**MIXED, turnbull_estimator="Kaplan-Meier")
        payload = model.to_dict()
        payload["H"] = None
        restored = NonParametric.from_dict(payload)
        np.testing.assert_allclose(
            restored.smoothed_hf([4.0, 7.0], bandwidth=3),
            model.smoothed_hf([4.0, 7.0], bandwidth=3),
        )


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
        model = _fit(
            **MIXED,
            turnbull_estimator="Nelson-Aalen",
            tol=1e-6,
            max_iter=7,
        )
        calls = self._record(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.bootstrap_cb([6.0], B=3, random_state=1)
        assert len(calls) == 3
        for kwargs in calls:
            assert kwargs["estimator"] == "Nelson-Aalen"
            assert kwargs["tol"] == 1e-6
            assert kwargs["max_iter"] == 7

    def test_settings_survive_serialisation(self, monkeypatch):
        model = _fit(**MIXED, tol=1e-8, max_iter=50)
        restored = NonParametric.from_dict(model.to_dict(with_data=True))
        assert restored.data["tol"] == 1e-8
        assert restored.data["max_iter"] == 50
        calls = self._record(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            restored.bootstrap_cb([6.0], B=2, random_state=1)
        assert all(k["tol"] == 1e-8 and k["max_iter"] == 50 for k in calls)


class TestGreenwoodRoundOff:
    def test_last_value_with_round_off_is_undefined(self):
        # The EM's last r and d, equal but for round-off (and fractional).
        r = np.array([3.0, 1.2142857142857106])
        d = np.array([1.0, 1.2142857142857142])
        var = greenwood_variance(r, d)
        assert var[0] == pytest.approx(1 / 6)
        assert np.isnan(var[1])
        # Exactly as for exactly equal counts.
        exact = greenwood_variance(np.array([3.0, 2.0]), np.array([1.0, 2.0]))
        np.testing.assert_array_equal(np.isnan(var), np.isnan(exact))

    def test_zero_count_with_round_off_is_no_event(self):
        var = greenwood_variance(
            np.array([17.0, 17.0]), np.array([2e-16, 1.0])
        )
        assert var[0] == 0.0
        assert var[1] == pytest.approx(1 / (17 * 16))
        var = nelson_aalen_variance(np.array([17.0]), np.array([2e-16]))
        assert var[0] == 0.0

    def test_turnbull_last_value_bounds(self):
        model = _fit(**MIXED, turnbull_estimator="Kaplan-Meier")
        assert np.isnan(model.greenwood[-1])
        assert np.nanmax(np.abs(model.greenwood)) < 1e3
        # Undefined at the last value: lower 0, upper the last finite one.
        lower, upper = model.cb(12)[0]
        assert lower == 0.0
        assert upper == pytest.approx(model.cb(11)[0, 1])
        # The estimate there is 0, not round-off below it.
        assert model.sf(12)[0] == 0.0

    def test_integer_counts_unchanged(self):
        r = np.array([10.0, 8.0, 5.0, 2.0])
        d = np.array([2.0, 1.0, 3.0, 2.0])
        with np.errstate(all="ignore"):
            raw = np.cumsum(
                np.where(
                    np.isfinite(d / (r * (r - d))), d / (r * (r - d)), np.nan
                )
            )
        np.testing.assert_array_equal(greenwood_variance(r, d), raw)


class TestBenard:
    def test_benard_is_the_median_rank_approximation(self):
        x = np.arange(1.0, 11.0)
        N = x.size
        _, _, _, F = plotting_positions(x, heuristic="Benard")
        i = np.arange(1, N + 1)
        np.testing.assert_allclose(F, (i - 0.3) / (N + 0.4))
        _, _, _, F_median = plotting_positions(x, heuristic="Median")
        np.testing.assert_allclose(F, F_median)


class TestTurnbullEstimatorOption:
    def test_docstring_says_truncation_iterates_with_kaplan_meier(self):
        doc = surv.KaplanMeier.fit.__doc__
        assert (
            "With truncation the EM always iterates with the Kaplan-Meier"
            in doc
        )

    def test_truncated_ladder_does_not_depend_on_option(self):
        # What the docstring now says: under truncation the EM itself is
        # the same whatever the option.
        kw = dict(x=[2, 3, 3, 4, 5, 6], tl=[0, 0, 1, 1, 2, 2])
        km = _fit(**kw, turnbull_estimator="Kaplan-Meier")
        na = _fit(**kw, turnbull_estimator="Nelson-Aalen")
        np.testing.assert_allclose(km.r, na.r)
        np.testing.assert_allclose(km.d, na.d)
        assert km.sf(2)[0] == pytest.approx(0.75)
        assert na.sf(2)[0] == pytest.approx(0.779, abs=5e-4)

    @pytest.mark.parametrize("call", ["fit", "function", "plotting"])
    def test_unknown_estimator_raises(self, call):
        with pytest.raises(ValueError, match="turnbull_estimator.*'foo'"):
            if call == "fit":
                Turnbull.fit(xl=[0, 1], xr=[2, 3], turnbull_estimator="foo")
            elif call == "function":
                nonp.turnbull(
                    np.array([1.0, 2.0]),
                    np.array([0, 0]),
                    np.array([1, 1]),
                    np.array([[-np.inf, np.inf]] * 2),
                    estimator="foo",
                )
            else:
                plotting_positions(
                    [1.0, 2.0, 3.0],
                    heuristic="Turnbull",
                    turnbull_estimator="foo",
                )

    def test_error_lists_the_options(self):
        with pytest.raises(ValueError) as info:
            Turnbull.fit([1.0, 2.0], turnbull_estimator="Kaplan Meier")
        for option in ("Fleming-Harrington", "Nelson-Aalen", "Kaplan-Meier"):
            assert option in str(info.value)
