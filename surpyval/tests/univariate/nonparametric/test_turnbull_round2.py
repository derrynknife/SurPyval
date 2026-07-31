"""
Regression tests for the round-2 Turnbull fixes:

- #272: interval/left-censored observations follow the (l, r] convention —
  the exact-time atom at the right endpoint is inside the support.
- #273: under truncation the variance ladder uses observed counts (no
  negative variances, reduces to delayed-entry KM Greenwood), supports are
  intersected with truncation windows, and degenerate-interval inputs use
  the KM-reducible variance.
"""

import numpy as np
import pytest

from surpyval import KaplanMeier, Turnbull


class TestIntervalEndpointConvention:
    def test_right_endpoint_tie_matches_npmle(self):
        # 272: exact {2} x1, {3} x5, interval (1, 3] x5. The (l, r] NPMLE
        # puts mass 1/6 at 2 and 5/6 at 3 (lifelines NPMLE agrees), so
        # S(2.5) = 5/6. The old support excluded the atom at 3 and gave
        # S(2.5) = 5/11.
        tb = Turnbull.fit(
            x=[2] * 1 + [3] * 5 + [[1, 3]] * 5,
            c=[0] * 6 + [2] * 5,
            turnbull_estimator="Kaplan-Meier",
        )
        sf25 = float(np.ravel(tb.sf(2.5))[0])
        assert sf25 == pytest.approx(5 / 6, abs=1e-6)

    def test_left_censored_at_event_time(self):
        # 272: left-censored at 3 means "failed at or before 3": with
        # exact {3} x5 + left-censored-at-3 x5 all mass sits at 3.
        tb = Turnbull.fit(
            x=[3.0] * 5 + [3.0] * 5,
            c=[0] * 5 + [-1] * 5,
            turnbull_estimator="Kaplan-Meier",
        )
        R = np.asarray(tb.R)
        # Survival is 1 before 3 and 0 at/after 3.
        assert R[-1] == pytest.approx(0.0, abs=1e-9)

    def test_generic_intervals_unchanged(self):
        # Non-coinciding endpoints were already correct; sanity-pin one.
        x = np.array([[1, 5], [2, 3], [3, 6], [1, 8], [9, 10]])
        model = Turnbull.fit(x)
        R = np.asarray(model.R)
        assert np.all(np.isfinite(R))
        assert np.all(np.diff(R) <= 1e-12)


class TestTruncatedVariance:
    def test_variance_matches_delayed_entry_km(self):
        # 273: exact + right-censored + left-truncated reduces to
        # delayed-entry KM — including the Greenwood ladder, which used
        # to produce a -1.5e15 increment at the last event.
        x = [2, 3, 4, 5, 6, 7, 8]
        c = [0, 0, 1, 0, 0, 0, 0]
        tl = [0, 1, 1, 2, 3, 0, 5]
        tb = Turnbull.fit(x=x, c=c, tl=tl, turnbull_estimator="Kaplan-Meier")
        km = KaplanMeier.fit(x=x, c=c, tl=tl)

        tb_x = np.asarray(tb.x)
        tb_gw = np.asarray(tb.greenwood)
        km_x = np.asarray(km.x)
        km_gw = np.asarray(km.greenwood)
        # Compare at each KM event time (TB's ladder has doubled bounds;
        # take the last TB position at or before each KM time).
        for xv, gv in zip(km_x, km_gw):
            idx = np.searchsorted(tb_x, xv, side="right") - 1
            if np.isnan(gv):
                assert np.isnan(tb_gw[idx])
            else:
                assert tb_gw[idx] == pytest.approx(gv, abs=1e-10)
        # No negative variance anywhere on the ladder.
        finite = tb_gw[np.isfinite(tb_gw)]
        assert np.all(finite >= 0)

    def test_survival_still_matches_delayed_entry_km(self):
        x = [2, 3, 4, 5, 6, 7, 8]
        c = [0, 0, 1, 0, 0, 0, 0]
        tl = [0, 1, 1, 2, 3, 0, 5]
        tb = Turnbull.fit(x=x, c=c, tl=tl, turnbull_estimator="Kaplan-Meier")
        km = KaplanMeier.fit(x=x, c=c, tl=tl)
        t_eval = [2.5, 3.5, 5.5, 6.5, 7.5]
        np.testing.assert_allclose(
            np.ravel(tb.sf(t_eval)), np.ravel(km.sf(t_eval)), atol=1e-9
        )


class TestSupportWindowIntersection:
    def test_left_censored_with_entry_converges(self):
        # 273: valid left-censored + delayed-entry data used to hit the
        # degenerate all-zero fixed point because the (-inf, x] support
        # was not intersected with the (tl, inf) window.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tb = Turnbull.fit(
                x=[3, 4, 5, 6, 7],
                c=[0, -1, 0, 0, 0],
                tl=[1, 2, 2, 3, 1],
                turnbull_estimator="Kaplan-Meier",
            )
        R = np.asarray(tb.R)
        assert np.all(np.isfinite(R))
        assert np.nanmax(R) == pytest.approx(1.0)


class TestDegenerateIntervalReducibility:
    def test_degenerate_intervals_match_1d_form(self):
        # 273: the same exact + right-censored data in degenerate-interval
        # form must give identical survival and Greenwood variance to the
        # 1-D form (both on the KM-reducible observed-count ladder).
        x1 = [1, 2, 3, 4, 5, 6]
        c1 = [0, 0, 1, 0, 1, 0]
        x2 = [[v, v] if cc == 0 else [v, np.inf] for v, cc in zip(x1, c1)]
        t1 = Turnbull.fit(x=x1, c=c1, turnbull_estimator="Kaplan-Meier")
        t2 = Turnbull.fit(x=x2, turnbull_estimator="Kaplan-Meier")
        np.testing.assert_allclose(
            np.asarray(t1.greenwood),
            np.asarray(t2.greenwood),
            atol=1e-12,
        )
        km = KaplanMeier.fit(x=x1, c=c1)
        t_eval = [1.5, 2.5, 3.5, 4.5, 5.5]
        np.testing.assert_allclose(
            np.ravel(t2.sf(t_eval)), np.ravel(km.sf(t_eval)), atol=1e-8
        )
