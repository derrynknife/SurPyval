"""Input validation for ``handle_xicn``.

Recurrent-event data enters the fitters through ``handle_xicn``. Malformed
input (empty arrays, NaN/inf, nonsensical counts or censoring codes, event
times below the integration origin) previously flowed silently into the
optimiser and produced meaningless fits. These tests pin the informative
``ValueError``s that now guard those cases -- and, importantly, confirm the
valid edge cases that must *not* be rejected.
"""

import numpy as np
import pytest

from surpyval.utils.recurrent_utils import handle_xicn

# --- empty / degenerate shapes -------------------------------------------


def test_empty_x_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        handle_xicn(np.array([]))


# --- non-finite event times ----------------------------------------------


def test_nan_x_rejected():
    # NaN is caught upstream by coerce_xcnt_x, but the contract is asserted
    # here so the guarantee is visible from the recurrent entry point.
    with pytest.raises(ValueError):
        handle_xicn(np.array([1.0, np.nan, 3.0]), np.array([1, 1, 1]))


def test_inf_x_rejected():
    with pytest.raises(ValueError, match="must be finite"):
        handle_xicn(np.array([1.0, np.inf, 3.0]), np.array([1, 1, 1]))


# --- censoring codes ------------------------------------------------------


@pytest.mark.parametrize("bad_code", [3, -2, 0.5, 10])
def test_invalid_censoring_code_rejected(bad_code):
    with pytest.raises(ValueError, match="Censoring 'c' must be one of"):
        handle_xicn(
            np.array([1.0, 2.0, 3.0]),
            np.array([1, 1, 1]),
            c=np.array([0, bad_code, 1]),
        )


def test_nan_censoring_code_rejected():
    with pytest.raises(ValueError, match="Censoring 'c' must be one of"):
        handle_xicn(
            np.array([1.0, 2.0, 3.0]),
            np.array([1, 1, 1]),
            c=np.array([0.0, np.nan, 1.0]),
        )


def test_all_valid_censoring_codes_accepted():
    # -1 (left), 0 (observed), 1 (right) on a single item, ordered so the
    # left-censoring is first and right-censoring last (the handler's other
    # rules). Interval (2) is exercised with 2D x below.
    data = handle_xicn(
        np.array([1.0, 2.0, 3.0]),
        np.array([1, 1, 1]),
        c=np.array([-1, 0, 1]),
    )
    assert np.array_equal(np.sort(np.unique(data.c)), [-1, 0, 1])


def test_interval_censoring_code_accepted():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    data = handle_xicn(x, np.array([1, 1]), c=np.array([2, 2]))
    assert np.all(data.c == 2)


# --- counts ---------------------------------------------------------------


def test_zero_count_rejected():
    with pytest.raises(ValueError, match="strictly positive"):
        handle_xicn(np.array([1.0, 2.0]), np.array([1, 1]), n=np.array([1, 0]))


def test_negative_count_rejected():
    with pytest.raises(ValueError, match="strictly positive"):
        handle_xicn(
            np.array([1.0, 2.0]), np.array([1, 1]), n=np.array([1, -2])
        )


def test_nan_count_rejected():
    with pytest.raises(ValueError, match="Counts 'n' must be finite"):
        handle_xicn(
            np.array([1.0, 2.0]), np.array([1, 1]), n=np.array([1.0, np.nan])
        )


def test_inf_count_rejected():
    with pytest.raises(ValueError, match="Counts 'n' must be finite"):
        handle_xicn(
            np.array([1.0, 2.0]), np.array([1, 1]), n=np.array([1.0, np.inf])
        )


# --- item identifiers -----------------------------------------------------


def test_nan_item_id_rejected():
    with pytest.raises(ValueError, match="Item identifiers 'i' must"):
        handle_xicn(np.array([1.0, 2.0]), np.array([1.0, np.nan]))


def test_string_item_ids_accepted():
    # Item labels need not be numeric; the finiteness check must skip them.
    data = handle_xicn(
        np.array([1.0, 2.0, 1.0]),
        np.array(["a", "a", "b"]),
    )
    assert set(np.unique(data.i)) == {"a", "b"}


# --- truncation bounds ----------------------------------------------------


def test_default_infinite_truncation_accepted():
    # +/-inf is the default open window and must not be rejected.
    data = handle_xicn(np.array([1.0, 2.0]), np.array([1, 1]))
    assert np.all(data.tl == -np.inf) and np.all(data.tr == np.inf)


def test_nan_truncation_rejected():
    with pytest.raises(ValueError, match="Truncation bounds must not"):
        handle_xicn(
            np.array([1.0, 2.0]),
            np.array([1, 1]),
            tl=np.array([np.nan, np.nan]),
            tr=np.array([5.0, 5.0]),
        )


# --- event times vs the integration origin --------------------------------


def test_negative_time_untruncated_rejected():
    with pytest.raises(ValueError, match="outside its observation window"):
        handle_xicn(np.array([-1.0, 2.0]), np.array([1, 1]))


def test_negative_time_with_negative_left_truncation_accepted():
    # An explicit negative left-truncation window legitimately admits
    # negative event times (delayed/early entry on a shifted scale).
    data = handle_xicn(
        np.array([-1.0, 2.0]), np.array([1, 1]), tl=-4.0, tr=5.0
    )
    assert np.allclose(np.sort(data.x), [-1.0, 2.0])


def test_event_after_right_truncation_rejected():
    with pytest.raises(ValueError, match="outside its observation window"):
        handle_xicn(np.array([1.0, 9.0]), np.array([1, 1]), tl=0.0, tr=5.0)


# --- covariates -----------------------------------------------------------


def test_non_finite_covariates_rejected():
    with pytest.raises(ValueError, match="Covariates 'Z' must be finite"):
        handle_xicn(
            np.array([1.0, 2.0]),
            np.array([1, 1]),
            Z=np.array([[0.5], [np.inf]]),
        )


# --- a fully valid call still works ---------------------------------------


def test_valid_data_passes():
    data = handle_xicn(
        np.array([1.0, 2.0, 3.0, 1.5]),
        i=np.array([1, 1, 1, 2]),
        c=np.array([0, 0, 1, 0]),
        n=np.array([1, 1, 1, 1]),
    )
    assert data.x.shape[0] == 4
