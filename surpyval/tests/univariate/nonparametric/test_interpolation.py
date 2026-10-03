"""Smooth interpolation of the non-parametric estimates: ``interp='cubic'``
is monotone (PCHIP), stays a valid survival function, and its bounds close
onto it.
"""

import numpy as np
import pytest

import surpyval
from surpyval.tests._helpers import no_warnings, sharp_drop_long_tail_data


def test_cubic_interpolation_is_monotone_and_bounded():
    model = surpyval.KaplanMeier.fit(sharp_drop_long_tail_data())
    grid = np.linspace(model.x.min(), model.x.max(), 500)
    sf = model.sf(grid, interp="cubic")
    finite = sf[np.isfinite(sf)]
    # Non-increasing and inside [0, 1] -- a plain cubic spline breaks both.
    assert np.all(np.diff(finite) <= 1e-12)
    assert finite.min() >= 0.0 and finite.max() <= 1.0


def test_cubic_interpolation_passes_through_estimates():
    # A shape-preserving interpolant must still interpolate: at the
    # observed times it returns the fitted survival exactly.
    model = surpyval.KaplanMeier.fit(sharp_drop_long_tail_data())
    assert np.allclose(model.sf(model.x, interp="cubic"), model.R, atol=1e-9)


def test_cubic_interpolation_does_not_poison_hazard():
    # A negative interpolated survival used to make Hf = -log(sf) NaN in
    # the interior. With a monotone interpolant Hf is instead a valid,
    # non-decreasing cumulative hazard (with +inf only where the survival
    # estimate legitimately reaches 0 at the last uncensored point).
    model = surpyval.KaplanMeier.fit(sharp_drop_long_tail_data())
    grid = np.linspace(model.x.min(), model.x.max(), 200)
    Hf = model.Hf(grid, interp="cubic")
    assert not np.any(np.isnan(Hf))
    finite = Hf[np.isfinite(Hf)]
    assert np.all(np.diff(finite) >= -1e-9)


def test_cubic_interpolation_with_turnbull_zero_width_bounds():
    # Turnbull's ``x`` carries duplicated (zero-width) bounds at exact
    # times; PCHIP requires strictly increasing abscissae, so the fix must
    # collapse them rather than raise.
    x = np.array([[1, 5], [2, 3], [3, 6], [1, 8], [9, 10.0]])
    model = surpyval.Turnbull.fit(x, turnbull_estimator="Kaplan-Meier")
    grid = np.linspace(1.0, 10.0, 200)
    sf = model.sf(grid, interp="cubic")
    finite = sf[np.isfinite(sf)]
    assert np.all(np.diff(finite) <= 1e-9)
    assert finite.min() >= -1e-9 and finite.max() <= 1 + 1e-9


# ---------------------------------------------------------------------------
# #417: the cubic bounds close onto ``sf``.
# ---------------------------------------------------------------------------


def _km_ending_at_zero():
    # The estimate reaches 0 at the last time, where the variance is
    # undefined (the registry's Kaplan-Meier fixture).
    from surpyval.tests.conformance.registry import CASE_BY_NAME, fitted

    return fitted(CASE_BY_NAME["KaplanMeier"])


@pytest.mark.parametrize("bound_type", ["exp", "normal"])
def test_cubic_bounds_close_onto_sf(bound_type):
    model = _km_ending_at_zero()
    x = np.array([2.0, 6.0, 10.0, 13.0])
    sf = model.sf(x, interp="cubic")
    cb = no_warnings(
        model.cb, x, interp="cubic", alpha_ci=1 - 1e-6, bound_type=bound_type
    )
    # 13 is between the last two times with a variance: the upper bound
    # there was 0.1873 against sf 0.1948.
    np.testing.assert_allclose(cb[:, 0], sf, rtol=1e-5)
    np.testing.assert_allclose(cb[:, 1], sf, rtol=1e-5)


@pytest.mark.parametrize("bound_type", ["exp", "normal"])
def test_cubic_bounds_hold_sf_between_them(bound_type):
    model = _km_ending_at_zero()
    x = np.linspace(model.x[0], model.x[-1], 201)
    sf = model.sf(x, interp="cubic")
    cb = model.cb(x, interp="cubic", bound_type=bound_type)
    assert np.all(cb[:, 0] <= sf + 1e-12) and np.all(sf <= cb[:, 1] + 1e-12)
    if bound_type == "exp":
        assert np.all((cb >= 0) & (cb <= 1))


def test_linear_bounds_are_unchanged():
    # For the linear kind the estimate plus the interpolated distance is
    # the interpolated bound itself.
    model = _km_ending_at_zero()
    x = np.linspace(model.x[0], model.x[-2], 50)
    got = model.R_cb(x, interp="linear")
    knots = model.R_cb(model.x)
    np.testing.assert_allclose(got[:, 0], np.interp(x, model.x, knots[:, 0]))
    np.testing.assert_allclose(got[:, 1], np.interp(x, model.x, knots[:, 1]))
