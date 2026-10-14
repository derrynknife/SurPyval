"""SurpyvalData carrying event types (the competing-risks fitters take
them from the data) and step time-varying covariates (fitted by the
model's fit_tvc)."""

import numpy as np
import pytest

import surpyval as sp
from surpyval import ItemLifetime, StepSchedule, SurpyvalData
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    CompetingRisksProportionalHazards,
    FineGray,
    ParametricCompetingRisks,
)

RNG = np.random.default_rng(3)
X = RNG.weibull(2, 80) * 10
C = (X > 13).astype(int)
E = RNG.choice(["a", "b"], 80).astype(object)
E[C == 1] = None
Z = RNG.normal(size=(80, 1))
TL = X.min() / 2  # a delayed entry before every failure

# the same rows in from_observations order: failures, then suspensions
ORDER = np.r_[np.flatnonzero(C == 0), np.flatnonzero(C == 1)]
XO, CO, EO, ZO = X[ORDER], C[ORDER], E[ORDER], Z[ORDER]
DATA = SurpyvalData.from_observations(
    observed=X[C == 0],
    observed_event_types=list(E[C == 0]),
    right_censored=X[C == 1],
)
DZ = SurpyvalData.from_observations(
    observed=X[C == 0],
    observed_event_types=list(E[C == 0]),
    observed_covariates=Z[C == 0],
    right_censored=X[C == 1],
    right_censored_covariates=Z[C == 1],
)


# -- event types -----------------------------------------------------------


def test_from_observations_keeps_event_types_in_row_order():
    assert list(DATA.e) == list(EO)
    lists = DATA.to_observations()
    assert lists["observed_event_types"] == list(E[C == 0])
    again = SurpyvalData.from_observations(**lists)
    assert list(again.e) == list(DATA.e)


def test_nonparametric_and_parametric_competing_risks_take_the_data():
    for fitter in (CompetingRisks, ParametricCompetingRisks):
        np.testing.assert_allclose(
            fitter.fit(DATA).cif([5, 10], "a"),
            fitter.fit(XO, EO, CO).cif([5, 10], "a"),
        )


def test_competing_risks_regressions_take_the_data():
    np.testing.assert_allclose(
        CompetingRisksProportionalHazards.fit(DZ).cif([5, 10], ZO[:1], "a"),
        CompetingRisksProportionalHazards.fit(XO, ZO, EO, CO).cif(
            [5, 10], ZO[:1], "a"
        ),
    )
    np.testing.assert_allclose(
        FineGray.fit(DZ, event="a").params,
        FineGray.fit(XO, ZO, EO, CO, event="a").params,
    )


def test_the_constructor_takes_event_types():
    data = SurpyvalData([1, 2, 3], c=[0, 0, 1], e=["a", "b", None])
    assert list(data.e) == ["a", "b", None]
    assert list(data[0:2].e) == ["a", "b"]
    # A failure without a type is a masked cause, warned of (#928)
    with pytest.warns(sp.utils.warnings.MaskedCauseWarning, match="rows 1"):
        masked = SurpyvalData([1, 2], c=[0, 0], e=["a", None])
    assert list(masked.e) == ["a", None]
    with pytest.raises(ValueError, match="one event type per row"):
        SurpyvalData([1, 2], e=["a"])


def test_competing_risks_need_event_types_and_others_refuse_them():
    with pytest.raises(TypeError, match="needs event types"):
        CompetingRisks.fit(SurpyvalData(X, C))
    with pytest.raises(ValueError, match="event types"):
        sp.Weibull.fit(DATA)
    with pytest.raises(TypeError, match="leave out 'e'"):
        CompetingRisks.fit(DATA, e=E)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {"observed": [1, 2], "observed_event_types": ["a"]},
            "one per item",
        ),
        (
            {"right_censored": [1], "observed_event_types": "a"},
            "without 'observed'",
        ),
    ],
)
def test_from_observations_checks_event_types(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SurpyvalData.from_observations(**kwargs)


def test_from_observations_takes_masked_causes():
    # A failure without a type, or a kind of failure given without types,
    # is a masked cause (#928)
    with pytest.warns(sp.utils.warnings.MaskedCauseWarning):
        data = SurpyvalData.from_observations(
            observed=[1, 2],
            observed_event_types=["a", None],
            left_censored=[0.5],
        )
    assert list(data.e) == ["a", None, None]


def test_left_truncation_reaches_a_fit_taking_only_tl():
    data = SurpyvalData.from_observations(
        observed=X[C == 0],
        observed_event_types=list(E[C == 0]),
        right_censored=X[C == 1],
        left_truncation=TL,
    )
    np.testing.assert_allclose(
        ParametricCompetingRisks.fit(data).cif([5], "a"),
        ParametricCompetingRisks.fit(XO, EO, CO, tl=TL).cif([5], "a"),
    )


# -- step time-varying covariates ---------------------------------------------

SWITCH = 4.0
STEPS = StepSchedule.from_changepoints([0, SWITCH], [[0.0], [1.0]])
ITEMS = [
    ItemLifetime(
        k,
        **{"right_censored" if ck else "observed": xk},
        time_varying_covariates=STEPS,
    )
    for k, (xk, ck) in enumerate(zip(X, C))
]
TVC = SurpyvalData.from_items(ITEMS)


def _start_stop(x=X, c=C, starts=None, static=None):
    i, xl, xr, cc, z = [], [], [], [], []
    for k, (xk, ck) in enumerate(zip(x, c)):
        s = 0.0 if starts is None else starts[k]
        cuts = [s] + [SWITCH] * int(s < SWITCH < xk) + [xk]
        for j, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            i.append(k)
            xl.append(a)
            xr.append(b)
            cc.append(ck if j == len(cuts) - 2 else 1)
            row = [float(a >= SWITCH)]
            z.append(row if static is None else [*static[k], *row])
    return np.array(i), np.array(xl), np.array(xr), np.array(cc), np.array(z)


@pytest.mark.parametrize("fitter", [sp.WeibullPH, sp.CoxPH, sp.WeibullAFT])
def test_step_covariates_fit_by_fit_tvc(fitter):
    np.testing.assert_allclose(
        fitter.fit(TVC).params, fitter.fit_tvc(*_start_stop()).params
    )


def test_static_covariates_come_first():
    items = [it.clone(covariates=Z[k]) for k, it in enumerate(ITEMS)]
    data = SurpyvalData.from_items(items)
    np.testing.assert_allclose(
        sp.WeibullPH.fit(data).params,
        sp.WeibullPH.fit_tvc(*_start_stop(static=Z)).params,
    )


def test_counts_and_delayed_entry_reach_fit_tvc():
    items = [it.clone(left_truncation=TL) for it in ITEMS]
    data = SurpyvalData.from_items(items, counts=2)
    i, xl, xr, c, z = _start_stop(starts=np.full(80, TL))
    np.testing.assert_allclose(
        sp.WeibullPH.fit(data).params,
        sp.WeibullPH.fit_tvc(i, xl, xr, c, z, n=np.full(len(i), 2)).params,
    )


def test_what_fit_tvc_cannot_take_is_refused():
    path = sp.CovariatePath.from_points([0, 10], [0.0, 1.0])
    with pytest.raises(ValueError, match="#908"):
        sp.WeibullPH.fit(
            SurpyvalData.from_items(
                [it.clone(time_varying_covariates=path) for it in ITEMS]
            )
        )
    found = [ITEMS[0].clone(interval_censored=[1, 2])] + ITEMS[1:]
    with pytest.raises(ValueError, match="left or interval censored"):
        sp.WeibullPH.fit(SurpyvalData.from_items(found))
    with pytest.raises(ValueError, match="time-varying covariates"):
        sp.Weibull.fit(TVC)
    with pytest.raises(ValueError, match="time-varying covariates"):
        sp.Weibull.fit_from_surpyval_data(TVC)
