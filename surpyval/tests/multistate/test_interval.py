"""Interval-censored moves in the parametric multi-state models:
``x`` as ``[xl, xr]``, a move known only to lie between two times."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import dblquad, quad

from surpyval import Exponential, Weibull, WeibullPH
from surpyval.multistate import (
    MultiState,
    ParametricMultiState,
    ParametricMultiStateRegression,
    TransitionMatrix,
)
from surpyval.multistate._data import check_spells
from surpyval.multistate._interval import IntervalLikelihood, tanh_sinh
from surpyval.tests.multistate._inspected import TRANSITIONS, simulate

TM = TransitionMatrix(TRANSITIONS)
PARS = {
    ("healthy", "ill"): (6.0, 1.5),
    ("healthy", "dead"): (25.0, 1.2),
    ("ill", "dead"): (8.0, 2.0),
}


def _H(pair, t):
    a, b = PARS[pair]
    return (t / a) ** b


def _h(pair, t):
    a, b = PARS[pair]
    return b / a * (t / a) ** (b - 1)


def _models():
    return [Weibull.from_params(list(PARS[p])) for p in TM.transitions]


def _hazard(model, t, rows, rate):
    return model.hf(t) if rate else model.Hf(t)


def _simulate(n, seed, clock="forward", gap=1.0):
    return simulate(n, seed, PARS, clock, gap)


def _next(k, x, e, state, tl, i):
    """The spell begun by row ``k``'s interval-censored move."""
    return (i == i[k]) & (state == e[k]) & (tl >= x[k, 0]) & (tl <= x[k, 1])


def test_tanh_sinh_rule():
    f, fr, w = tanh_sinh()
    assert_allclose(f + fr, 1.0, rtol=1e-15)
    assert_allclose(w.sum(), 1.0, rtol=1e-12)
    # an end where the integrand is infinite
    assert_allclose(np.sum(w * f**-0.5), 2.0, rtol=1e-9)
    assert_allclose(np.sum(w * np.exp(f)), np.e - 1, rtol=1e-13)


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_likelihood_of_two_moves_in_a_row(clock):
    """A unit seen healthy at 2, ill at 4 and 5, dead by 7: the double
    integral over both moves' times, by scipy's adaptive quadrature."""
    data = check_spells(
        [[2.0, 4.0], [5.0, 7.0]],
        ["ill", "dead"],
        ["healthy", "ill"],
        tl=[0.0, 4.0],
        i=[1, 1],
        transitions=TM,
        intervals=True,
    )
    lik = IntervalLikelihood(data, clock, _hazard, [0.0] * 3)
    value = np.exp(-lik.neg_ll(_models()))
    hi, hd, idd = TM.transitions

    def stay_ill(u, v):
        if clock == "reset":
            return np.exp(-_H(idd, v - u)) * _h(idd, v - u)
        return np.exp(-(_H(idd, v) - _H(idd, u))) * _h(idd, v)

    def f(v, u):
        return np.exp(-_H(hi, u) - _H(hd, u)) * _h(hi, u) * stay_ill(u, v)

    exact, _ = dblquad(f, 2.0, 4.0, 5.0, 7.0, epsabs=0, epsrel=1e-12)
    assert_allclose(value, exact, rtol=1e-9)


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_move_with_no_spell_after_it(clock):
    """Seen healthy at 1 and ill at 3, nothing after: the unit stayed ill
    from the move until 3."""
    data = check_spells(
        [[1.0, 3.0]], ["ill"], ["healthy"], transitions=TM, intervals=True
    )
    lik = IntervalLikelihood(data, clock, _hazard, [0.0] * 3)
    value = np.exp(-lik.neg_ll(_models()))
    hi, hd, idd = TM.transitions

    def f(u):
        stay = (
            _H(idd, 3.0 - u) if clock == "reset" else _H(idd, 3.0) - _H(idd, u)
        )
        return np.exp(-_H(hi, u) - _H(hd, u) - stay) * _h(hi, u)

    exact, _ = quad(f, 1.0, 3.0, epsabs=0, epsrel=1e-13)
    assert_allclose(value, exact, rtol=1e-10)


def test_two_states_is_the_univariate_interval_censored_fit():
    rng = np.random.default_rng(7)
    t = 10 * rng.weibull(0.8, 200)
    visits = np.arange(0.0, 60.0, 4.0)
    xl = np.array([visits[visits < v].max() for v in t])
    xr = np.array([visits[visits >= v].min() if v < 56 else 56 for v in t])
    cens = t > 56
    xl[cens] = xr[cens] = 56.0
    model = ParametricMultiState.fit(
        np.column_stack([xl, xr]),
        [None if c else "dead" for c in cens],
        ["alive"] * t.size,
        transitions=[("alive", "dead")],
    )
    uni = Weibull.fit(
        x=np.column_stack([xl, xr]),
        c=np.where(cens, 1, 2),
    )
    assert_allclose(model.params, uni.params, rtol=1e-6)
    assert_allclose(model.neg_ll(), uni.neg_ll(), rtol=1e-10)
    assert_allclose(
        model.standard_errors(),
        np.sqrt(np.diag(uni.covariance())),
        rtol=1e-4,
    )


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_an_interval_shrunk_to_a_point_is_the_exact_fit(clock):
    x, e, state, tl, i = _simulate(150, 3, clock)
    # the moves' times, known: the middle of each interval
    mid = x.mean(axis=1)
    succ_tl = tl.copy()
    for k in np.flatnonzero(x[:, 0] < x[:, 1]):
        succ_tl[_next(k, x, e, state, tl, i)] = mid[k]
    exact = ParametricMultiState.fit(
        mid, e, state, tl=succ_tl, i=i, transitions=TM, clock=clock
    )
    half = 5e-7
    X = np.column_stack([mid - half, mid + half])
    X[x[:, 0] == x[:, 1]] = mid[x[:, 0] == x[:, 1], None]
    narrow = ParametricMultiState.fit(
        X, e, state, tl=succ_tl, i=i, transitions=TM, clock=clock
    )
    assert narrow._joint is not None
    # (the same maximum, to the optimisers' tolerances)
    se = exact.standard_errors()
    assert np.all(np.abs(narrow.params - exact.params) < 1e-3 * se)
    assert_allclose(
        narrow.standard_errors(), exact.standard_errors(), rtol=1e-3
    )


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_recovers_the_weibull_hazards(clock):
    """One seeded data set of 400 units: every parameter within three
    standard errors of the truth."""
    x, e, state, tl, i = _simulate(400, 11, clock)
    assert np.sum(x[:, 0] < x[:, 1]) > 100
    model = ParametricMultiState.fit(
        x, e, state, tl=tl, i=i, transitions=TM, clock=clock
    )
    truth = np.concatenate([PARS[p] for p in TM.transitions])
    z = (model.params - truth) / model.standard_errors()
    assert np.all(np.abs(z) < 3), z
    assert model.maximum == "verified"


def test_joint_fit_covariance_and_criteria():
    x, e, state, tl, i = _simulate(200, 5)
    model = ParametricMultiState.fit(
        x, e, state, tl=tl, i=i, transitions=TM, dist=Exponential
    )
    cov = model.covariance()
    assert cov.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    # the moves out of healthy share the interval-censored factors
    assert abs(cov[0, 1]) > 0
    n = float(np.sum([v is not None for v in e]))
    assert_allclose(model.aic(), 2 * 3 + 2 * model.neg_ll())
    assert_allclose(model.bic(), 2 * model.neg_ll() + 3 * np.log(n))
    band = model.cb([2.0, 5.0], "ill")
    p = model.occupancy([2.0, 5.0], "ill")
    assert np.all((band[:, 0] < p) & (p < band[:, 1]))


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_round_trip(clock):
    x, e, state, tl, i = _simulate(80, 2, clock)
    model = ParametricMultiState.fit(
        x, e, state, tl=tl, i=i, transitions=TM, dist=Exponential, clock=clock
    )
    back = ParametricMultiState.from_dict(model.to_dict())
    assert_allclose(back.params, model.params)
    assert_allclose(back.covariance(), model.covariance())
    assert back.neg_ll() == model.neg_ll()
    assert back.maximum == model.maximum
    assert_allclose(back.cb(4.0, "ill"), model.cb(4.0, "ill"))


def test_regression_with_interval_censored_moves():
    x, e, state, tl, i = _simulate(200, 9)
    rng = np.random.default_rng(1)
    z = rng.integers(0, 2, size=np.unique(i).size)[i]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ParametricMultiStateRegression.fit(
            x, z[:, None], e, state, tl=tl, i=i, transitions=TM
        )
    assert model._joint is not None
    assert model.covariance().shape == (9, 9)
    assert model.maximum == "verified"
    table = model.summary()
    assert list(table.index.get_level_values(0).unique()) == list(
        TM.transition_names()
    )
    assert_allclose(table["se(coef)"], model.standard_errors())
    back = ParametricMultiStateRegression.from_dict(model.to_dict())
    assert_allclose(back.params, model.params)
    # the shrunk-interval limit is the exact fit
    mid = x.mean(axis=1)
    succ_tl = tl.copy()
    for k in np.flatnonzero(x[:, 0] < x[:, 1]):
        succ_tl[_next(k, x, e, state, tl, i)] = mid[k]
    X = np.where(x[:, :1] < x[:, 1:], mid[:, None] + [-5e-7, 5e-7], x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        narrow = ParametricMultiStateRegression.fit(
            X,
            z[:, None],
            e,
            state,
            tl=succ_tl,
            i=i,
            transitions=TM,
            model=WeibullPH,
        )
        exact = ParametricMultiStateRegression.fit(
            mid,
            z[:, None],
            e,
            state,
            tl=succ_tl,
            i=i,
            transitions=TM,
            model=WeibullPH,
        )
    assert_allclose(narrow.params, exact.params, rtol=1e-4)


def test_fit_from_df_takes_two_columns():
    import pandas as pd

    x, e, state, tl, i = _simulate(60, 4)
    df = pd.DataFrame(
        {"xl": x[:, 0], "xr": x[:, 1], "e": e, "s": state, "tl": tl, "i": i}
    )
    a = ParametricMultiState.fit_from_df(
        df,
        ["xl", "xr"],
        "e",
        "s",
        tl_col="tl",
        i_col="i",
        transitions=TM,
        dist=Exponential,
    )
    b = ParametricMultiState.fit(
        x, e, state, tl=tl, i=i, transitions=TM, dist=Exponential
    )
    assert_allclose(a.params, b.params)


def test_refusals():
    kw = dict(transitions=TM)
    with pytest.raises(ValueError, match="censored spell ends"):
        ParametricMultiState.fit([[1.0, 2.0]], [None], ["healthy"], **kw)
    with pytest.raises(ValueError, match="xl <= xr"):
        ParametricMultiState.fit([[3.0, 2.0]], ["ill"], ["healthy"], **kw)
    with pytest.raises(ValueError, match="tl <= xl"):
        ParametricMultiState.fit(
            [[1.0, 3.0]], ["ill"], ["healthy"], tl=[2.0], **kw
        )
    with pytest.raises(ValueError, match="cannot be before"):
        ParametricMultiState.fit(
            [[1.0, 3.0], [2.0, 4.0]],
            ["ill", "dead"],
            ["healthy", "ill"],
            tl=[0.0, 1.0],
            i=[1, 1],
            **kw,
        )
    with pytest.raises(ValueError, match="ParametricMultiState"):
        MultiState.fit([[1.0, 3.0]], ["ill"], ["healthy"], **kw)


# --- a SurpyvalData with interval-censored rows (c = 2) ---------------------


def _flags(x, e):
    """The SurpyvalData flags of the long form: 2 where the move is known
    only to lie in (xl, xr], 1 where the unit was still in its state, 0
    where the move's time is known."""
    censored = np.array([v is None for v in e])
    return np.where(x[:, 0] < x[:, 1], 2, np.where(censored, 1, 0))


@pytest.mark.parametrize("clock", ["forward", "reset"])
@pytest.mark.parametrize("regression", [False, True], ids=["dist", "reg"])
def test_surpyval_data_is_the_fit_from_arrays(clock, regression):
    import surpyval as sp

    # (80 units were too few for the reset clock's array fit itself: the
    # midpoint start of one transition had a single distinct time)
    x, e, state, tl, i = _simulate(150, 13, clock=clock)
    assert (x[:, 0] < x[:, 1]).sum() > 10
    c = _flags(x, e)
    z = (np.arange(np.unique(i).size) % 2).astype(float)[i][:, None]
    cls = (
        ParametricMultiStateRegression if regression else ParametricMultiState
    )
    zkw = {"Z": z} if regression else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = cls.fit(
            x, *zkw.values(), e, state, tl=tl, i=i, transitions=TM,
            clock=clock,
        )  # fmt: skip
        # the two columns of x, and the lower and upper ends in xl, xr
        for data in (
            sp.SurpyvalData(x=x, c=c, e=e, s=state, tl=tl, i=i, **zkw),
            sp.SurpyvalData(
                x=np.where(c == 2, np.nan, x[:, 1]),
                xl=np.where(c == 2, x[:, 0], np.nan),
                xr=np.where(c == 2, x[:, 1], np.nan),
                c=c, e=e, s=state, tl=tl, i=i, **zkw,
            ),  # fmt: skip
        ):
            got = cls.fit(data, transitions=TM, clock=clock)
            assert got._joint is not None
            np.testing.assert_array_equal(got.params, ref.params)
            np.testing.assert_array_equal(got.covariance(), ref.covariance())
            assert got.to_dict() == ref.to_dict()
            q = (6.0, z[:1]) if regression else (6.0,)
            np.testing.assert_array_equal(
                got.occupancy(*q, "ill"), ref.occupancy(*q, "ill")
            )


def test_surpyval_data_interval_refusals():
    import surpyval as sp

    s = ["healthy", "ill", "healthy"]
    # a left-censored move: taken by the fit, refused by the Bayesian one
    data = sp.SurpyvalData(
        x=[2.0, 5, 3], c=[-1, 0, 1], e=["ill", "dead", None], s=s
    )
    with pytest.raises(ValueError, match="ParametricMultiState.fit and"):
        ParametricMultiState.fit_bayes(data, transitions=TM)
    # the Bayesian fit takes no intervals
    data = sp.SurpyvalData(
        x=[[1.0, 2.0], [5, 5], [3, 3]], c=[2, 0, 1],
        e=["ill", "dead", None], s=s,
    )  # fmt: skip
    with pytest.raises(ValueError, match="Rows 0 are interval censored"):
        ParametricMultiState.fit_bayes(data, transitions=TM)
    with pytest.raises(ValueError, match="Rows 0 are interval censored"):
        MultiState.fit(data, transitions=TM)
