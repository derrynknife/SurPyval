import numpy as np
import pytest

from surpyval.recurrent import (
    ARA,
    ARI,
    HPP,
    CrowAMSAA,
    GeneralizedOneRenewal,
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
        (GeneralizedRenewal, {"kijima": "ii"}),
        (GeneralizedOneRenewal, {}),
        (ARA, {"m": 2}),
        (ARI, {"m": 1}),
    ],
)
def test_615_renewal_fits_take_delayed_entry_as_new(model, kwargs):
    # Delayed entry in the virtual-age models: each item as new at its
    # entry, so the fit is that of the times since entry. They used to
    # refuse tl > 0, and to ignore a negative tl (counting the first gap
    # from 0).
    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2, 4.5, 5])
    c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3])
    tl = np.where(i == 3, 0.5, np.where(i == 2, -1.0, 0.0))
    # (A negative entry age warns as a likely data error, #664.)
    with pytest.warns(UserWarning, match="negative age"):
        entered = model.fit(x, i, c, tl=tl, **kwargs)
    shifted = model.fit(x - tl, i, c, **kwargs)
    assert np.allclose(entered.params, shifted.params, rtol=1e-8)
    assert entered.log_likelihood == pytest.approx(
        shifted.log_likelihood, rel=1e-10
    )
    # The model's data are on its clock: the times since entry.
    assert np.allclose(entered.data.x, x - tl)
    # fit_from_df reads the entry column the same way.
    import pandas as pd

    log = pd.DataFrame({"t": x, "unit": i, "c": c, "entry": tl})
    with pytest.warns(UserWarning, match="negative age"):
        via_df = model.fit_from_df(
            log, x_col="t", i_col="unit", c_col="c", tl_col="entry", **kwargs
        )
    assert np.allclose(via_df.params, entered.params, rtol=1e-8)


@pytest.mark.parametrize(
    "model, kwargs",
    [
        (GeneralizedRenewal, {}),
        (GeneralizedOneRenewal, {}),
        (ARA, {"m": 1}),
        (ARI, {"m": 1}),
    ],
)
def test_624_renewal_fits_close_the_window_at_tr(model, kwargs):
    # A finite tr is each item's end of observation, as the NHPP fitters
    # take it: the same fit as a c=1 row at tr. It used to be ignored.
    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
    via_tr = model.fit_from_recurrent_data(
        handle_xicn(x, i, tl=np.where(i == 3, 0.5, 0.0), tr=20.0), **kwargs
    )
    rows = np.r_[x, 20, 20, 20]
    items = np.r_[i, 1, 2, 3]
    ends = np.r_[np.zeros_like(x), 1, 1, 1]
    via_row = model.fit_from_recurrent_data(
        handle_xicn(rows, items, ends, tl=np.where(items == 3, 0.5, 0.0)),
        **kwargs,
    )
    assert np.allclose(via_tr.params, via_row.params, rtol=1e-8)
    assert via_tr.log_likelihood == pytest.approx(via_row.log_likelihood)
    # A c=1 row already at tr is not doubled.
    closed = model.fit_from_recurrent_data(
        handle_xicn(rows, items, ends, tr=20.0), **kwargs
    )
    assert len(closed.data.x) == len(rows)


def test_624_renewal_refuses_rows_past_tr():
    # Hand-built data (handle_xicn refuses it too): rows after the end of
    # observation.
    from surpyval.utils.recurrent_event_data import RecurrentEventData

    data = RecurrentEventData(
        np.array([1.0, 3.0, 6.0, 2.0, 4.0]),
        np.array([1, 1, 1, 2, 2]),
        np.zeros(5),
        np.ones(5),
        tr=np.array([5.0, 5.0, 5.0, 9.0, 9.0]),
    )
    with pytest.raises(ValueError, match=r"Item 1 has a row at 6\.0 after"):
        GeneralizedRenewal.fit_from_recurrent_data(data)


def _per_unit_case(labels):
    # Three units whose rows are interleaved in x (not contiguous), labelled
    # by ``labels`` in the order the rows name them: per-unit bounds are in
    # the sorted order of the labels (np.unique(i)), per-row ones follow i.
    x = np.array([12.0, 30, 5, 44, 61, 18, 90, 75, 9, 130, 52, 110])
    k = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 1])
    i = np.array(labels)[k]
    entry, end = {0: 0.0, 1: 2.0, 2: 4.0}, {0: 150.0, 1: 120.0, 2: 140.0}
    units = np.unique(i).tolist()
    tl_unit = np.array([entry[labels.index(u)] for u in units])
    tr_unit = np.array([end[labels.index(u)] for u in units])
    tl_row = np.array([entry[j] for j in k])
    tr_row = np.array([end[j] for j in k])
    return x, i, tl_unit, tr_unit, tl_row, tr_row


@pytest.mark.parametrize(
    "labels",
    [["pump-b", "pump-c", "pump-a"], [40, 7, 1000], [3, 1, 2]],
)
def test_per_unit_tl_tr_equal_per_row(labels):
    # tl / tr may be given one value per unit (in np.unique(i) order) in
    # place of one per row: handle_xicn expands them, so the data and every
    # fit are exactly those of the per-row form.
    from surpyval.recurrent import NonParametricCounting

    x, i, tl_unit, tr_unit, tl_row, tr_row = _per_unit_case(labels)
    for kwargs_unit, kwargs_row in [
        ({"tl": tl_unit, "tr": tr_unit}, {"tl": tl_row, "tr": tr_row}),
        ({"tr": tr_unit}, {"tr": tr_row}),
        ({"tl": tl_row, "tr": tr_unit}, {"tl": tl_row, "tr": tr_row}),
    ]:
        by_unit = handle_xicn(x, i, **kwargs_unit)
        by_row = handle_xicn(x, i, **kwargs_row)
        for name in ("x", "i", "c", "n", "tl", "tr"):
            assert np.array_equal(
                getattr(by_unit, name), getattr(by_row, name)
            ), name
        for fitter in (CrowAMSAA, HPP):
            assert np.array_equal(
                fitter.fit(x, i=i, **kwargs_unit).params,
                fitter.fit(x, i=i, **kwargs_row).params,
            )
    # The stored per-row bound is each row's own unit's value.
    data = handle_xicn(x, i, tr=tr_unit)
    by_label = dict(zip(np.unique(i).tolist(), tr_unit.tolist()))
    assert data.tr.tolist() == [by_label[u] for u in data.i.tolist()]

    unit = {"tl": tl_unit, "tr": tr_unit}
    row = {"tl": tl_row, "tr": tr_row}
    mcf_unit = NonParametricCounting.fit(x, i=i, **unit)
    mcf_row = NonParametricCounting.fit(x, i=i, **row)
    assert np.array_equal(mcf_unit.mcf_hat, mcf_row.mcf_hat)
    Z = {u: [float(n % 2)] for n, u in enumerate(np.unique(i).tolist())}
    assert np.array_equal(
        ProportionalIntensityNHPP.fit(x, i=i, Z=Z, **unit).params,
        ProportionalIntensityNHPP.fit(x, i=i, Z=Z, **row).params,
    )
    # A renewal fit takes per-unit entry ages too.
    units = np.unique(i)
    rows = np.r_[x, tr_unit]
    items = np.r_[i, units]
    ends = np.r_[np.zeros_like(x), np.ones(len(units))]
    tl_rows = np.r_[tl_row, tl_unit]
    assert np.array_equal(
        GeneralizedRenewal.fit(rows, items, ends, tl=tl_unit).params,
        GeneralizedRenewal.fit(rows, items, ends, tl=tl_rows).params,
    )


def test_per_unit_tr_wrong_length_names_both_lengths():
    # Neither one per row nor one per unit: refused, naming both lengths.
    x, i, _, tr_unit, _, _ = _per_unit_case([1, 2, 3])
    with pytest.raises(
        ValueError,
        match=r"'tr' must have one entry per row of 'x' \(12\) or one per "
        r"item in 'i' \(3",
    ):
        CrowAMSAA.fit(x, i=i, tr=np.r_[tr_unit, 200.0])
    with pytest.raises(ValueError, match=r"'tl' must have one entry per row"):
        handle_xicn(x, i, tl=[0.0, 1.0])


def test_per_unit_bound_with_as_many_rows_as_items_is_per_row():
    # When the rows and units are equally many the bound is read per row.
    data = handle_xicn([1.0, 2.0], i=[2, 1], tr=[10.0, 20.0])
    assert data.i.tolist() == [1, 2]
    assert data.tr.tolist() == [20.0, 10.0]
