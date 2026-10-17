"""ItemLifetime: one item's lifetime, and SurpyvalData built from (and
back to) a list of them, with how many identical items each stands for."""

import numpy as np
import pytest

import surpyval as sp
from surpyval import ItemLifetime, StepSchedule, SurpyvalData

X = sp.Weibull.random(30, 10, 2, random_state=7)
C = (X > 12).astype(int)


def _items(x=X, c=C):
    return [
        ItemLifetime(k, **{"right_censored" if ck else "observed": xk})
        for k, (xk, ck) in enumerate(zip(x, c))
    ]


def test_items_fit_as_their_arrays():
    data = SurpyvalData.from_items(_items())
    np.testing.assert_allclose(
        sp.Weibull.fit(data).params, sp.Weibull.fit(X, C).params
    )
    np.testing.assert_allclose(
        sp.KaplanMeier.fit(data).sf([5, 10]),
        sp.KaplanMeier.fit(X, C).sf([5, 10]),
    )


def test_counts_are_identical_items():
    items = [
        ItemLifetime("A", observed=5),
        ItemLifetime("B", right_censored=9),
        ItemLifetime("C", interval_censored=[2, 4]),
        ItemLifetime("D", observed=7),
    ]
    data = SurpyvalData.from_items(items, counts=[3, 10, 2, 0])
    assert data.ids == ["A", "B", "C"]
    np.testing.assert_array_equal(data.n, [3, 10, 2])
    repeated = SurpyvalData(
        [5, 5, 5] + [9] * 10 + [[2, 4]] * 2, c=[0] * 3 + [1] * 10 + [2] * 2
    )
    np.testing.assert_allclose(
        sp.Weibull.fit(data).params, sp.Weibull.fit(repeated).params
    )


def test_every_outcome_and_truncation_round_trips():
    items = [
        ItemLifetime("A", observed=5, left_truncation=1),
        ItemLifetime("B", right_censored=9, right_truncation=20),
        ItemLifetime("C", left_censored=3),
        ItemLifetime("D", interval_censored=[2, 4], left_truncation=2),
    ]
    data = SurpyvalData.from_items(items, counts=[1, 2, 1, 3])
    assert data.to_items() == items
    again = SurpyvalData.from_items(data.to_items(), counts=data.n)
    for a, b in zip(again.to_xcnt(), data.to_xcnt()):
        np.testing.assert_array_equal(a, b)


def test_covariates_fit_a_regression():
    Z = (np.arange(30) % 2).astype(float)
    items = [
        ItemLifetime(
            k, **{"right_censored" if ck else "observed": xk}, covariates=zk
        )
        for k, (xk, ck, zk) in enumerate(zip(X, C, Z))
    ]
    data = SurpyvalData.from_items(items)
    np.testing.assert_allclose(
        sp.WeibullPH.fit(data).params, sp.WeibullPH.fit(X, Z, C).params
    )


def test_event_types_and_time_varying_covariates_are_kept_but_not_fitted_yet():
    schedule = StepSchedule.from_changepoints([0, 3], [[0.0], [1.0]])
    items = [
        ItemLifetime(
            "A",
            observed=5,
            event_type="spall",
            time_varying_covariates=schedule,
        ),
        ItemLifetime(
            "B", right_censored=9, time_varying_covariates={0: [0.0]}
        ),
    ]
    data = SurpyvalData.from_items(items)
    assert list(data.e) == ["spall", None]
    held = data.time_varying_covariates["A"]
    assert held.edges.tolist() == schedule.edges.tolist()
    assert held.Z.tolist() == schedule.Z.tolist()
    assert data.to_items() == items
    with pytest.raises(ValueError, match="no event types"):
        sp.Weibull.fit(data)
    with pytest.raises(ValueError, match="event types"):
        sp.Weibull.fit_from_surpyval_data(data)


def test_to_items_of_plain_data_numbers_the_rows():
    items = SurpyvalData([3, 7], c=[0, 1]).to_items()
    assert items == [
        ItemLifetime(0, observed=3.0),
        ItemLifetime(1, right_censored=7.0),
    ]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "exactly one outcome"),
        ({"observed": 5, "right_censored": 9}, "exactly one outcome"),
        ({"right_censored": 9, "event_type": "a"}, "has not failed"),
        ({"interval_censored": [4, 2]}, "lower < upper"),
        ({"interval_censored": [2, np.inf]}, "open at"),
        ({"interval_censored": 5}, "one \\[lower, upper\\] pair"),
        ({"observed": 5, "left_truncation": 5}, "not after it came under"),
        ({"observed": 50, "right_truncation": 20}, "after right_truncation"),
        (
            {"observed": 5, "time_varying_covariates": {2: [1.0]}},
            "start at age 2",
        ),
        ({"observed": np.nan}, "finite"),
    ],
)
def test_an_item_refuses(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ItemLifetime("A", **kwargs)


def test_the_dataset_needs_consistent_items():
    a = ItemLifetime("A", observed=5, covariates=1)
    with pytest.raises(ValueError, match="more than once"):
        SurpyvalData.from_items(
            [a, ItemLifetime("A", observed=6, covariates=1)]
        )
    with pytest.raises(ValueError, match="missing \\['B'\\]"):
        SurpyvalData.from_items([a, ItemLifetime("B", observed=6)])
    with pytest.warns(UserWarning, match="masked causes\\): items 'B'"):
        SurpyvalData.from_items(
            [
                ItemLifetime("A", observed=5, event_type="a"),
                ItemLifetime("B", observed=6),
            ]
        )
    with pytest.raises(ValueError, match="Every count is 0"):
        SurpyvalData.from_items([ItemLifetime("A", observed=5)], counts=0)
    with pytest.raises(ValueError, match="one per item"):
        SurpyvalData.from_items([ItemLifetime("A", observed=5)], counts=[1, 2])
    with pytest.raises(TypeError, match="ItemLifetime records"):
        SurpyvalData.from_items([sp.ItemHistory("A", observed=[5])])


# -- #931: every record that builds goes into a dataset ----------------------


def test_issue_931_right_censored_at_its_right_truncation_is_refused():
    with pytest.raises(ValueError, match="Item 'a': right censored at 12"):
        ItemLifetime("a", right_censored=12, right_truncation=12)
    with pytest.raises(ValueError, match="Item 'a': right censored at 10"):
        ItemLifetime("a", right_censored=5, right_truncation=10).extend(10)
    # Observed or found failed by the truncation age is fine: it failed
    ok = ItemLifetime("a", observed=12, right_truncation=12)
    assert SurpyvalData.from_items([ok]).tr.tolist() == [12.0]
    ok = ItemLifetime("a", interval_censored=[8, 12], right_truncation=12)
    assert SurpyvalData.from_items([ok]).c.tolist() == [2]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"observed": 5, "tl": 2}, "'tl': a record's left truncation"),
        ({"observed": 5, "Z": 2}, "'Z': a record's covariates are"),
        ({"observed": 5, "e": "x"}, "'e': what failed is event_type="),
        ({"x": 5}, "'x': the outcome is one of observed="),
        ({"observed": 5, "n": 3}, "'n': how many identical items"),
        (
            {"observed": 5, "left_truncaton": 2},
            "did you mean 'left_truncation'",
        ),
    ],
)
def test_issue_931_unknown_keywords_name_the_argument_meant(kwargs, match):
    with pytest.raises(TypeError, match=match):
        ItemLifetime("a", **kwargs)
    with pytest.raises(TypeError, match=match):
        ItemLifetime("a", observed=1).clone(**kwargs)


def test_issue_931_from_items_and_from_observations_check_keywords():
    items = [ItemLifetime("a", observed=5)]
    with pytest.raises(TypeError, match="'n': how many identical items"):
        SurpyvalData.from_items(items, n=[2])
    with pytest.raises(TypeError, match="'count'.*counts="):
        SurpyvalData.from_items(items, count=[2])
    with pytest.raises(TypeError, match="'tl': left truncation is"):
        SurpyvalData.from_observations(observed=[1, 2], tl=0.5)
    with pytest.raises(TypeError, match="'x': the times go in the list"):
        SurpyvalData.from_observations(x=[1, 2])
    with pytest.raises(TypeError, match="did you mean 'observed_counts'"):
        SurpyvalData.from_observations(observed=[1, 2], observed_count=[1, 2])


@pytest.mark.parametrize(
    "bad, match",
    [
        (None, "needs an id.*None, which is missing"),
        (np.nan, "needs an id.*which is missing"),
        ("", "needs an id.*which is missing"),
        (["a"], r"got \['a'\], a list"),
        ({"a": 1}, "a dict"),
    ],
)
def test_issue_931_bad_ids_are_refused_when_the_record_is_made(bad, match):
    with pytest.raises(ValueError, match=match):
        ItemLifetime(bad, observed=1)
    with pytest.raises(ValueError, match=match):
        sp.ItemHistory(bad, observed=[1])
    with pytest.raises(ValueError, match=match):
        SurpyvalData.from_observations(observed=[1], observed_ids=[bad])


def test_issue_931_mixed_id_kinds_name_one_of_each():
    with pytest.raises(
        ValueError, match=r"ids 1 \(a number\) and 'b' \(text\) are mixed"
    ):
        SurpyvalData.from_items(
            [ItemLifetime(1, observed=1), ItemLifetime("b", observed=2)]
        )
    with pytest.raises(
        ValueError, match=r"ids 'b' \(text\) and 1 \(a number\)"
    ):
        SurpyvalData.from_observations(observed=[1, 2], observed_ids=["b", 1])
    # Tuple ids (a gap's id) are one kind
    SurpyvalData.from_items(
        [
            ItemLifetime(("a", 1), observed=1),
            ItemLifetime(("a", 2), observed=2),
        ]
    )
