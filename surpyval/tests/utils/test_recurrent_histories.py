"""RecurrentEventData.from_histories / to_histories, and the recurrent
fitters taking a RecurrentEventData as ``x``."""

import numpy as np
import pytest

from surpyval import RecurrentEventData, SurpyvalData, handle_xicn
from surpyval import recurrent as rc

RNG = np.random.default_rng(4)
OBSERVED: dict = {}
ENDS: dict = {}
COVARIATES: dict = {}
for _k in range(8):
    _ages = np.cumsum(RNG.weibull(1.5, size=RNG.integers(2, 6)) * 300)
    OBSERVED[f"U{_k}"] = _ages.round(1).tolist()
    ENDS[f"U{_k}"] = float(round(_ages[-1] + RNG.uniform(10, 400), 1))
    COVARIATES[f"U{_k}"] = [float(_k % 2)]

DATA = RecurrentEventData.from_histories(
    observed=OBSERVED, right_censored=ENDS
)
DZ = RecurrentEventData.from_histories(
    observed=OBSERVED, right_censored=ENDS, covariates=COVARIATES
)


def _arrays(observed, ends):
    x, i, c = [], [], []
    for item, ages in observed.items():
        x += list(ages)
        i += [item] * len(ages)
        c += [0] * len(ages)
        if item in ends:
            x.append(ends[item])
            i.append(item)
            c.append(1)
    return np.array(x), np.array(i), np.array(c)


# -- from_histories ---------------------------------------------------------


def test_from_histories_is_the_arrays():
    x, i, c = _arrays(OBSERVED, ENDS)
    by_hand = handle_xicn(x, i, c)
    for name in ("x", "i", "c", "n"):
        np.testing.assert_array_equal(
            getattr(DATA, name), getattr(by_hand, name)
        )


def test_an_item_without_an_end_ends_at_its_last_failure():
    data = RecurrentEventData.from_histories(
        observed={"A": [5, 9], "B": [3]}, right_censored={"A": 12}
    )
    np.testing.assert_array_equal(data.x[data.i == "B"], [3])
    np.testing.assert_array_equal(data.c[data.i == "B"], [0])


def test_an_item_with_no_failures_has_only_its_end():
    data = RecurrentEventData.from_histories(
        observed={"A": [5]}, right_censored={"A": 8, "B": 20}
    )
    assert data.items == ["A", "B"]
    np.testing.assert_array_equal(data.c[data.i == "B"], [1])


def test_gaps_are_running_totals_from_the_start_of_observation():
    data = RecurrentEventData.from_histories(
        observed={"A": [10, 5], "B": [4]},
        right_censored={"A": 2, "B": 1},
        left_truncation={"B": 100},
        gaps=True,
    )
    np.testing.assert_array_equal(data.x[data.i == "A"], [10, 15, 17])
    np.testing.assert_array_equal(data.x[data.i == "B"], [104, 105])
    np.testing.assert_array_equal(data.tl[data.i == "B"], [100, 100])


def test_truncation_per_item_or_shared():
    data = RecurrentEventData.from_histories(
        observed={"A": [5, 9], "B": [30]},
        left_truncation={"B": 20},
        right_truncation=50,
    )
    np.testing.assert_array_equal(data.tl, [-np.inf, -np.inf, 20])
    np.testing.assert_array_equal(data.tr, [50, 50, 50])


def test_covariates_are_one_row_per_item_on_every_row():
    np.testing.assert_array_equal(
        DZ.Z[DZ.i == "U1"], np.ones((int((DZ.i == "U1").sum()), 1))
    )


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({}, ValueError, "Give each item"),
        ({"observed": [1, 2]}, TypeError, "dict keyed"),
        ({"observed": {"A": [1], 2: [3]}}, ValueError, "one type"),
        ({"observed": {"A": []}}, ValueError, "no failures and no end"),
        ({"observed": {"A": [1, np.inf]}}, ValueError, "finite"),
        (
            {"observed": {"A": [5, 9]}, "right_censored": {"A": 7}},
            ValueError,
            "before its last failure",
        ),
        (
            {"observed": {"A": [5]}, "left_truncation": {"B": 1}},
            ValueError,
            "names items",
        ),
        (
            {"observed": {"A": [5], "B": [6]}, "covariates": {"A": 1}},
            ValueError,
            "missing \\['B'\\]",
        ),
        (
            {
                "observed": {"A": [5], "B": [6]},
                "covariates": {"A": 1, "B": [1, 2]},
            },
            ValueError,
            "same length",
        ),
        (
            {"observed": {"A": [5, -1]}, "gaps": True},
            ValueError,
            "cannot be negative",
        ),
    ],
)
def test_from_histories_refuses(kwargs, error, match):
    with pytest.raises(error, match=match):
        RecurrentEventData.from_histories(**kwargs)


# -- to_histories -------------------------------------------------------------


def test_to_histories_round_trips():
    data = RecurrentEventData.from_histories(
        observed={"A": [5, 9], "B": [30]},
        right_censored={"A": 12, "C": 40},
        left_truncation={"B": 20},
        covariates={"A": [1, 0], "B": [0, 1], "C": [1, 1]},
    )
    histories = data.to_histories()
    assert histories["right_censored"] == {"A": 12.0, "C": 40.0}
    assert "right_truncation" not in histories
    again = RecurrentEventData.from_histories(**histories)
    for name in ("x", "i", "c", "n", "tl", "tr", "Z"):
        np.testing.assert_array_equal(
            getattr(again, name), getattr(data, name)
        )


def test_to_histories_refuses_what_a_history_cannot_hold():
    with_types = handle_xicn(
        [1, 3, 5], i=[1, 1, 1], c=[0, 0, 1], e=["a", "b", None]
    )
    with pytest.raises(ValueError, match="event types"):
        with_types.to_histories()
    counts = handle_xicn([[1, 3]], i=[1], c=[2], n=[3])
    with pytest.raises(ValueError, match="interval censored"):
        counts.to_histories()


# -- fit(data) ---------------------------------------------------------------

X, I, C = DATA.x, DATA.i, DATA.c


@pytest.mark.parametrize(
    "fitter",
    [
        rc.CrowAMSAA,
        rc.Duane,
        rc.HPP,
        rc.GeneralizedRenewal,
        rc.GeneralizedOneRenewal,
        rc.ARA,
        rc.ARI,
    ],
)
def test_a_fit_of_the_data_is_the_fit_of_its_arrays(fitter):
    np.testing.assert_allclose(
        fitter.fit(DATA).params, fitter.fit(X, I, C).params
    )


def test_the_mcf_of_the_data_is_the_mcf_of_its_arrays():
    np.testing.assert_allclose(
        rc.NonParametricCounting.fit(DATA).mcf([500, 1000]),
        rc.NonParametricCounting.fit(X, I, C).mcf([500, 1000]),
    )


@pytest.mark.parametrize(
    "fitter", [rc.ProportionalIntensityNHPP, rc.ProportionalIntensityHPP]
)
def test_proportional_intensity_takes_the_covariates_from_the_data(fitter):
    np.testing.assert_allclose(
        fitter.fit(DZ).params, fitter.fit(DZ.x, DZ.Z, DZ.i, DZ.c).params
    )


def test_cause_specific_fits_take_the_event_types_from_the_data():
    data = handle_xicn(
        [1, 3, 5, 2, 6, 7],
        i=[1, 1, 1, 2, 2, 2],
        c=[0, 0, 1, 0, 0, 1],
        e=["a", "b", None, "a", "a", None],
    )
    model = rc.CauseSpecificMCF.fit(data)
    by_hand = rc.CauseSpecificMCF.fit(data.x, data.i, data.c, e=data.e)
    np.testing.assert_allclose(
        model.mcf([4, 6], "a"), by_hand.mcf([4, 6], "a")
    )


def test_left_truncated_data_fits_as_its_arrays():
    data = RecurrentEventData.from_histories(
        observed=OBSERVED, right_censored=ENDS, left_truncation={"U0": 1.0}
    )
    np.testing.assert_allclose(
        rc.CrowAMSAA.fit(data).params,
        rc.CrowAMSAA.fit(data.x, data.i, data.c, tl=data.tl).params,
    )


@pytest.mark.parametrize(
    "call, error, match",
    [
        (lambda: rc.CrowAMSAA.fit(DATA, c=C), TypeError, "leave out 'c'"),
        (lambda: rc.CrowAMSAA.fit(DZ), ValueError, "no covariates"),
        (
            lambda: rc.ProportionalIntensityHPP.fit(DATA),
            TypeError,
            "holds none",
        ),
        (
            lambda: rc.GeneralizedRenewal.fit(
                RecurrentEventData.from_histories(
                    observed=OBSERVED, right_truncation=1e6
                )
            ),
            ValueError,
            "no right truncation",
        ),
        (
            lambda: rc.CrowAMSAA.fit(
                handle_xicn(
                    [1, 3, 6], i=[1, 1, 1], windows={1: [(0, 4), (5, 8)]}
                )
            ),
            ValueError,
            "observation windows",
        ),
        (
            lambda: rc.CrowAMSAA.fit(SurpyvalData([1.0, 2.0])),
            TypeError,
            "does not take",
        ),
    ],
)
def test_fit_data_refusals(call, error, match):
    with pytest.raises(error, match=match):
        call()


def test_unit_still_applies():
    assert rc.CrowAMSAA.fit(DATA, unit="hours").unit == "hour"
