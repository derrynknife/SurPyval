"""A model fitted on item records predicts from an item (its fixed and
time-varying covariates lined up as fitted), and datasets combine."""

import numpy as np
import pytest

import surpyval as sp
from surpyval import (
    CovariatePath,
    EventHistoryData,
    ItemHistory,
    ItemLifetime,
    StepSchedule,
    SurpyvalData,
)

RNG = np.random.default_rng(2)
LIFE = RNG.weibull(2, 80) * 900
SITE = (np.arange(80) % 2).astype(float)
STRESS = StepSchedule.from_changepoints([0, 400], [[0.0], [1.0]])
ITEMS = [
    ItemLifetime(k, observed=t, covariates=[s], time_varying_covariates=STRESS)
    for k, (t, s) in enumerate(zip(LIFE, SITE))
]
TIMES = [200.0, 500.0, 800.0]


@pytest.mark.parametrize("fitter", [sp.WeibullPH, sp.CoxPH])
def test_predicting_from_an_item_lines_its_covariates_up_as_fitted(fitter):
    model = fitter.fit(SurpyvalData.from_items(ITEMS))
    by_hand = StepSchedule.from_changepoints(
        [0, 400], [[1.0, 0.0], [1.0, 1.0]]
    )
    item = ITEMS[1]  # site 1
    np.testing.assert_allclose(
        model.sf_tvc(TIMES, item), model.sf_tvc(TIMES, by_hand)
    )


def test_an_item_with_fixed_covariates_only_is_a_constant_schedule():
    items = [
        ItemLifetime(k, observed=t, covariates=[s])
        for k, (t, s) in enumerate(zip(LIFE, SITE))
    ]
    model = sp.WeibullPH.fit(SurpyvalData.from_items(items))
    np.testing.assert_allclose(
        model.sf_tvc(TIMES, items[1]), model.sf(np.array(TIMES), [1.0])
    )


def test_an_item_on_a_continuous_path_with_fixed_covariates():
    model = sp.WeibullPH.fit(SurpyvalData.from_items(ITEMS))
    ramp = CovariatePath.from_points([0, 600], [0.0, 1.0])
    item = ITEMS[1].clone(time_varying_covariates=ramp)
    by_hand = CovariatePath.from_points([0, 600], [[1.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        model.sf_tvc(TIMES, item), model.sf_tvc(TIMES, by_hand), rtol=1e-8
    )


def test_predicting_from_an_item_without_covariates_or_with_xl():
    model = sp.WeibullPH.fit(SurpyvalData.from_items(ITEMS))
    with pytest.raises(ValueError, match="no covariates to predict with"):
        model.sf_tvc(TIMES, ItemLifetime("Z", observed=5))
    with pytest.raises(ValueError, match="xl must not be given"):
        model.sf_tvc(TIMES, ITEMS[0], xl=[0.0])


# -- combine -------------------------------------------------------------


def test_combine_stacks_data_without_ids():
    a = SurpyvalData([1, 2], c=[0, 1])
    b = SurpyvalData([[3, 4]], c=[2])
    both = SurpyvalData.combine(a, b)
    assert both.ids is None
    assert both.x.tolist() == [[1, 1], [2, 2], [3, 4]]
    assert SurpyvalData.combine([a, b]).x.tolist() == both.x.tolist()


def test_combine_joins_data_with_ids_and_refuses_a_clash():
    a = SurpyvalData.from_items([ItemLifetime("A", observed=1)], counts=3)
    b = SurpyvalData.from_items([ItemLifetime("B", right_censored=2)])
    both = SurpyvalData.combine(a, b)
    assert both.ids == ["A", "B"] and both.n.tolist() == [3, 1]
    with pytest.raises(ValueError, match="more than once: 'A'"):
        SurpyvalData.combine(a, a)


def test_combine_needs_datasets_that_agree():
    with_ids = SurpyvalData.from_items([ItemLifetime("A", observed=1)])
    plain = SurpyvalData([2.0])
    with pytest.raises(ValueError, match="some do not"):
        SurpyvalData.combine(with_ids, plain)
    with_z = SurpyvalData([1, 2], Z=[0.0, 1.0])
    with pytest.raises(ValueError, match="have covariates and some do not"):
        SurpyvalData.combine(with_z, plain)
    with pytest.raises(TypeError, match="takes SurpyvalData"):
        SurpyvalData.combine(plain, [1, 2])


def test_combine_event_histories():
    a = EventHistoryData.from_items(
        [ItemHistory("M1", observed=[5], right_censored=9)]
    )
    b = EventHistoryData.from_items(
        [ItemHistory("M2", observed=[3], right_censored=8)]
    )
    assert EventHistoryData.combine(a, b).ids == ["M1", "M2"]
    with pytest.raises(ValueError, match="more than once: 'M1'"):
        EventHistoryData.combine(a, a)
