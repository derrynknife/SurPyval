import numpy as np
import pytest

from surpyval.recurrent import ARA, ARI, Crow, GeneralizedRenewal, HPP
from surpyval.utils.recurrent_utils import handle_xicn


def test_handle_xicn_default_truncation():
    # Without truncation the window defaults to [0, inf) for every row.
    data = handle_xicn(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))
    assert np.all(data.tl == 0.0)
    assert np.all(np.isinf(data.tr))


def test_handle_xicn_scalar_truncation_broadcasts():
    data = handle_xicn(
        np.array([3.0, 7.0]), np.array([1, 1]), tl=2.0, tr=10.0
    )
    assert np.all(data.tl == 2.0)
    assert np.all(data.tr == 10.0)


def test_handle_xicn_rejects_events_outside_window():
    with pytest.raises(ValueError, match="outside its truncation window"):
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
        handle_xicn(
            np.array([3.0]), np.array([1]), t=[[0.0, 10.0]], tl=1.0
        )


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
    untruncated = Crow.fit(x, i)
    truncated = Crow.fit(x, i, tl=float(x[0]) - 0.5)
    assert not np.allclose(untruncated.params, truncated.params)


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
