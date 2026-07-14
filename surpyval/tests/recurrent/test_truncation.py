import numpy as np
import pytest

from surpyval.recurrent import (
    ARA,
    ARI,
    HPP,
    CrowAMSAA,
    GeneralizedRenewal,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)
from surpyval.utils.recurrent_utils import handle_xicn


def test_handle_xicn_default_truncation():
    # Without truncation the window defaults to the whole real line on the
    # right, with the fallback origin 0 on the left.
    data = handle_xicn(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))
    assert np.all(data.tl == -np.inf)
    assert np.all(data.tr == np.inf)


def test_handle_xicn_rejects_negative_x_when_untruncated():
    # Untruncated event times are integrated from the fallback origin 0, so a
    # negative time would give a negative interarrival. It is rejected rather
    # than silently corrupting the likelihood. (Genuinely negative event
    # times are admitted only with an explicit negative left-truncation
    # window; see test_explicit_negative_left_truncation_is_used_as_origin.)
    with pytest.raises(ValueError, match="outside its observation window"):
        handle_xicn(np.array([-3.0, -1.0, 2.0]), np.array([1, 1, 1]))


def test_explicit_negative_left_truncation_is_used_as_origin():
    # An explicit (possibly negative) tl is the integration origin; the first
    # interval starts exactly there rather than at 0.
    data = handle_xicn(
        np.array([-1.0, 2.0]), np.array([1, 1]), tl=-4.0, tr=5.0
    )
    x_prev = data.get_previous_x()
    assert np.isclose(x_prev[0], -4.0)


def test_handle_xicn_scalar_truncation_broadcasts():
    data = handle_xicn(np.array([3.0, 7.0]), np.array([1, 1]), tl=2.0, tr=10.0)
    assert np.all(data.tl == 2.0)
    assert np.all(data.tr == 10.0)


def test_handle_xicn_rejects_events_outside_window():
    with pytest.raises(ValueError, match="outside its observation window"):
        handle_xicn(np.array([1.0, 5.0]), np.array([1, 1]), tl=2.0)


def test_handle_xicn_rejects_inconsistent_window_within_item():
    # Different truncation bounds for the same item are not a single window.
    with pytest.raises(ValueError, match="inconsistent truncation"):
        handle_xicn(
            np.array([3.0, 7.0]),
            np.array([1, 1]),
            tl=np.array([2.0, 4.0]),
        )


def test_handle_xicn_t_and_tl_conflict():
    with pytest.raises(ValueError, match="Cannot use"):
        handle_xicn(np.array([3.0]), np.array([1]), t=[[0.0, 10.0]], tl=1.0)


def test_hpp_left_truncation_matches_analytic_mle():
    # HPP observed over [tl, tr]: the rate MLE is events / exposure. With two
    # events in a window of width 8 the rate is exactly 0.25.
    x = np.array([3.0, 7.0, 10.0])
    c = np.array([0, 0, 1])
    i = np.array([1, 1, 1])
    model = HPP.fit(x, i, c=c, tl=2.0)
    assert np.isclose(model.params[0], 2.0 / 8.0)

    # Default (no truncation) integrates from 0, giving 2 / 10.
    model0 = HPP.fit(x, i, c=c)
    assert np.isclose(model0.params[0], 2.0 / 10.0)


def test_nhpp_left_truncation_changes_fit():
    # Left truncation must move the estimate (the integral starts at tl).
    np.random.seed(0)
    x = np.cumsum(np.random.exponential(2.0, 25))
    i = np.ones_like(x)
    untruncated = CrowAMSAA.fit(x, i)
    truncated = CrowAMSAA.fit(x, i, tl=float(x[0]) - 0.5)
    assert not np.allclose(untruncated.params, truncated.params)


def test_hpp_right_truncation_closes_window_like_censoring_row():
    # A finite tr closes the observation window in the NHPP integral, so the
    # rate is events / exposure even with no explicit right-censoring row:
    # two events observed over [2, 10] gives exactly 0.25.
    x = np.array([3.0, 7.0])
    i = np.array([1, 1])
    model = HPP.fit(x, i, c=np.array([0, 0]), tl=2.0, tr=10.0)
    assert np.isclose(model.params[0], 0.25)

    # The same window expressed with an explicit c=1 row at the close time
    # must give an identical fit.
    model_c1 = HPP.fit(
        np.array([3.0, 7.0, 10.0]),
        np.array([1, 1, 1]),
        c=np.array([0, 0, 1]),
        tl=2.0,
    )
    assert np.isclose(model.params[0], model_c1.params[0])

    # Without tr (or a c=1 row) the window closes at the last event (t=7), so
    # the integral is shorter and the rate is larger: 2 / 5.
    model_open = HPP.fit(x, i, c=np.array([0, 0]), tl=2.0)
    assert np.isclose(model_open.params[0], 0.4)


def test_nhpp_right_truncation_matches_censoring_row():
    # For a genuine NHPP the tr window-close must agree with supplying the
    # close time as an explicit right-censoring row.
    x = [3.0, 7.0, 12.0]
    i = [1, 1, 1]
    via_tr = CrowAMSAA.fit(x, i, c=[0, 0, 0], tr=15.0)
    via_row = CrowAMSAA.fit(
        [3.0, 7.0, 12.0, 15.0], [1, 1, 1, 1], c=[0, 0, 0, 1]
    )
    assert np.allclose(via_tr.params, via_row.params, atol=1e-4)


@pytest.mark.parametrize(
    "fitter", [ProportionalIntensityNHPP, ProportionalIntensityHPP]
)
def test_proportional_intensity_right_truncation(fitter):
    # The proportional-intensity regression fitters now accept truncation and
    # close each item's window at tr; this must match the equivalent fit with
    # explicit c=1 rows at the window close.
    x = [9, 14, 18, 7, 12, 16, 19, 5, 9, 13, 16, 18, 6, 10, 13, 15, 17, 19]
    i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
    Z = np.array(
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ).reshape(-1, 1)
    via_tr = fitter.fit(x, Z, i=i, tr=20.0)

    x_c1 = x + [20, 20, 20, 20]
    i_c1 = i + [1, 2, 3, 4]
    Z_c1 = np.vstack([Z, np.array([[0], [0], [1], [1]])])
    c_c1 = [0] * len(x) + [1, 1, 1, 1]
    via_row = fitter.fit(x_c1, Z_c1, i=i_c1, c=c_c1)

    assert np.allclose(via_tr.params, via_row.params, atol=1e-3)
    assert np.allclose(via_tr.coeffs, via_row.coeffs, atol=1e-3)


@pytest.mark.parametrize(
    "model, kwargs",
    [
        (GeneralizedRenewal, {}),
        (ARA, {"m": 1}),
        (ARI, {"m": 1}),
    ],
)
def test_virtual_age_models_reject_left_truncation(model, kwargs):
    data = handle_xicn(
        np.array([1.0, 3.0, 6.0, 9.0]),
        np.array([1, 1, 1, 1]),
        tl=0.5,
    )
    with pytest.raises(ValueError, match="does not support left truncation"):
        model.fit_from_recurrent_data(data, **kwargs)
