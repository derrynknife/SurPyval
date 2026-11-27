"""Left-censored moves and right truncation in the parametric
multi-state models: each reduces to the univariate fit with two states,
and right truncation is recovered in a seeded simulation (with the bias
of ignoring it shown)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval import Weibull, WeibullPH
from surpyval.multistate import (
    ParametricMultiState,
    ParametricMultiStateRegression,
)

TWO = [("alive", "dead")]


def _inspected(seed=1, n=200, delayed=True):
    """Weibull lifetimes: a third seen when they fail, the rest looked at
    once (failed by then: left censored; or not: right censored); with
    ``delayed`` some enter at 1."""
    rng = np.random.default_rng(seed)
    T = 10 * rng.weibull(1.7, n)
    L = rng.uniform(3, 15, n)
    exact = rng.integers(0, 3, n) == 0
    x = np.where(exact & (T <= L), T, L)
    c = np.where(exact & (T <= L), 0, np.where(T <= L, -1, 1))
    tl = np.where(rng.uniform(size=n) < 0.3, 1.0, 0.0) if delayed else 0 * x
    keep = x > tl
    return x[keep], c[keep], tl[keep]


def _two_state(x, c, tl, **kw):
    e = np.where(c == 1, None, "dead").astype(object)
    X = np.column_stack([np.where(c == -1, -np.inf, x), x])
    return ParametricMultiState.fit(
        X, e, np.array(["alive"] * x.size, dtype=object),
        tl=np.where(tl > 0, tl, -np.inf), i=np.arange(x.size),
        transitions=TWO, **kw,
    )  # fmt: skip


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_left_censored_two_states_is_the_univariate_fit(clock):
    """With two states a left-censored move (x = [-inf, xr]) is the
    univariate left-censored observation, F(xr) (with delayed entry on
    the forward clock: (F(xr) - F(tl)) / S(tl); on the reset clock tl is
    the entry into the state, so none is used there)."""
    x, c, tl = _inspected(delayed=clock == "forward")
    uni = Weibull.fit(x=x, c=c, tl=tl if clock == "forward" else None)
    ms = _two_state(x, c, tl, clock=clock)
    assert_allclose(ms.log_likelihood, -uni.neg_ll(), rtol=0, atol=1e-8)
    assert_allclose(ms.params, uni.params, rtol=1e-5)


def test_left_censored_rows_of_a_surpyval_data():
    x, c, tl = _inspected()
    data = sp.SurpyvalData(
        x=x, c=c, tl=np.where(tl > 0, tl, -np.inf),
        s=np.array(["alive"] * x.size, dtype=object),
        e=np.where(c == 1, None, "dead").astype(object), i=np.arange(x.size),
    )  # fmt: skip
    a = ParametricMultiState.fit(data, transitions=TWO)
    b = _two_state(x, c, tl)
    assert_allclose(a.log_likelihood, b.log_likelihood, rtol=1e-12)
    with pytest.raises(ValueError, match="left censored"):
        sp.MultiState.fit(data, transitions=TWO)


def _truncated(seed=2, n=2000):
    """Weibull lifetimes seen only if they failed by a cut-off, some
    entering (left truncated) at 1."""
    rng = np.random.default_rng(seed)
    T = 10 * rng.weibull(1.7, n)
    tau = rng.uniform(2, 12, n)
    tl = np.where(rng.uniform(size=n) < 0.3, 1.0, 0.0)
    seen = (T <= tau) & (T > tl)
    return T[seen], tau[seen], tl[seen]


def test_right_truncated_two_states_is_the_univariate_fit():
    """f(x) / (F(tau) - F(tl)) for each unit, as Weibull.fit(x, tl, tr)."""
    x, tau, tl = _truncated()
    uni = Weibull.fit(x=x, tl=tl, tr=tau)
    ms = ParametricMultiState.fit(
        x, np.array(["dead"] * x.size, dtype=object),
        np.array(["alive"] * x.size, dtype=object),
        tl=np.where(tl > 0, tl, -np.inf), i=np.arange(x.size),
        transitions=TWO, tr=tau,
    )  # fmt: skip
    assert ms.maximum == "verified"
    assert_allclose(ms.log_likelihood, -uni.neg_ll(), rtol=0, atol=1e-8)
    assert_allclose(ms.params, uni.params, rtol=1e-5)
    back = sp.from_dict(ms.to_dict())
    assert_allclose(back.params, ms.params)
    assert_allclose(back._interval_likelihood().neg_ll(back._models()),
                    ms.neg_ll(), rtol=1e-12)  # fmt: skip


def test_right_truncated_regression_two_states_is_the_univariate_fit():
    x, tau, tl = _truncated(seed=3, n=600)
    z = (np.arange(x.size) % 2).astype(float)
    Z = pd.DataFrame({"z": z})
    uni = WeibullPH.fit(x=x, Z=Z, tl=tl, tr=tau)
    ms = ParametricMultiStateRegression.fit(
        x, Z, np.array(["dead"] * x.size, dtype=object),
        np.array(["alive"] * x.size, dtype=object),
        tl=np.where(tl > 0, tl, -np.inf), i=np.arange(x.size),
        transitions=TWO, tr=tau,
    )  # fmt: skip
    assert_allclose(ms.log_likelihood, -uni.neg_ll(), rtol=0, atol=1e-8)
    assert_allclose(ms.params, uni.params, rtol=1e-4)


PARS = {
    ("healthy", "ill"): (6.0, 1.5),
    ("healthy", "dead"): (25.0, 1.2),
    ("ill", "dead"): (8.0, 2.0),
}


def _register(n, seed):
    """An illness-death model on the forward clock (Weibull hazards of
    the time since the start), every unit healthy at 0, and only those
    dead by their own cut-off (uniform on (2, 12)) seen: a register of
    deaths, each history known exactly."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n):
        tau = rng.uniform(2, 12)
        (a1, b1), (a2, b2), (a3, b3) = PARS.values()
        t1, t2 = a1 * rng.weibull(b1), a2 * rng.weibull(b2)
        if t1 < t2:
            # dead from ill, given alive at t1, on the time since the start
            td = a3 * ((t1 / a3) ** b3 - np.log(rng.uniform())) ** (1 / b3)
            path = [("healthy", -np.inf, t1, "ill"), ("ill", t1, td, "dead")]
            death = td
        else:
            path = [("healthy", -np.inf, t2, "dead")]
            death = t2
        if death <= tau:
            rows += [(u, *p, tau) for p in path]
    return pd.DataFrame(rows, columns=["i", "state", "tl", "x", "e", "tr"])


def test_truncation_is_recovered_and_ignoring_it_is_biased():
    """One seeded register (600 units drawn, those dead by their cut-off
    kept): with the truncation every estimate is within 3 standard
    errors of the truth; ignoring it, some are far off (the units seen
    are the ones that died early)."""
    df = _register(600, 7100)
    kw = dict(tl=df["tl"], i=df["i"], transitions=list(PARS))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ParametricMultiState.fit(
            df["x"], df["e"], df["state"], tr=df["tr"], **kw
        )
        naive = ParametricMultiState.fit(df["x"], df["e"], df["state"], **kw)
    truth = np.concatenate([PARS[p] for p in PARS])
    z = (fit.params - truth) / fit.standard_errors()
    z_naive = (naive.params - truth) / naive.standard_errors()
    assert fit.maximum == "verified"
    assert np.all(np.abs(z) < 3), z
    assert np.max(np.abs(z_naive)) > 3, z_naive


def test_refusals():
    df = _register(150, 7101)
    kw = dict(tl=df["tl"], i=df["i"], transitions=list(PARS))
    args = (df["x"], df["e"], df["state"])
    with pytest.raises(ValueError, match="forward clock only"):
        ParametricMultiState.fit(*args, tr=df["tr"], clock="reset", **kw)
    with pytest.raises(ValueError, match="'i'"):
        ParametricMultiState.fit(
            *args, tl=df["tl"], transitions=list(PARS), tr=df["tr"]
        )
    with pytest.raises(ValueError, match="give the cut-offs in 'tr'"):
        ParametricMultiState.fit(*args, reached="dead", **kw)
    with pytest.raises(ValueError, match="not a state"):
        ParametricMultiState.fit(*args, tr=df["tr"], reached="gone", **kw)
    with pytest.raises(ValueError, match="do not reach that state"):
        ParametricMultiState.fit(*args, tr=df["tr"], reached="ill", **kw)
    tau = df["tr"].to_numpy().copy()
    first = np.flatnonzero(df["i"].duplicated(keep=False))[0]
    tau[first] += 1.0
    with pytest.raises(ValueError, match="different cut-offs"):
        ParametricMultiState.fit(*args, tr=tau, **kw)
    cr = [("alive", "a"), ("alive", "b")]
    with pytest.raises(ValueError, match="2 absorbing states"):
        ParametricMultiState.fit(
            [1.0, 2.0, 3.0], ["a", "b", "a"], ["alive"] * 3, i=[1, 2, 3],
            transitions=cr, tr=[5.0, 5.0, 5.0],
        )  # fmt: skip
    Z = (df["i"] % 2).to_numpy(dtype=float)
    Z[first] = 5.0
    with pytest.raises(ValueError, match="covariates differ"):
        ParametricMultiStateRegression.fit(
            df["x"], Z, df["e"], df["state"], tr=df["tr"], **kw
        )


def test_a_hazard_run_to_zero_under_truncation_is_flagged():
    """Every unit that died truncated (a register closing at 15): the
    death hazards run to 0, a maximum of the truncated likelihood only in
    the limit. The fit warns that it has no finite maximum, says so in
    ``maximum`` and gives nan standard errors for those transitions."""
    from surpyval.tests.conformance.registry_fixtures import _MS_UNITS, ms_data

    d = ms_data()
    i, tr = [], []
    for u, (spells, _, _) in enumerate(_MS_UNITS):
        dead = spells[-1][3] == "dead"
        i += [u] * len(spells)
        tr += [15.0 if dead else np.inf] * len(spells)
    with pytest.warns(UserWarning, match="No finite maximum: under the right"):
        model = ParametricMultiState.fit(
            **d, i=np.array(i), tr=np.array(tr), dist=sp.Exponential
        )
    assert model.maximum == "no finite maximum"
    se = model.standard_errors().reshape(4, 1)
    into_dead = [
        k for k, (a, b) in enumerate(model.transitions.transitions)
        if b == "dead"
    ]  # fmt: skip
    assert np.all(np.isnan(se[into_dead]))
    assert np.all(np.isfinite(np.delete(se, into_dead, axis=0)))


def test_patients_found_through_a_diagnosis():
    """The guide's example: patients seen only because they fell ill by
    a cut-off show no deaths before illness, by design. The fit refuses,
    saying why (leaving the transition out would set its hazard to 0 in
    each patient's chance of being found); with patients followed
    untruncated as well it fits."""
    df = _register(400, 7102)
    found = df[
        df.groupby("i")["state"].transform(lambda s: (s == "ill").any())
    ]
    ill_at = found[found["state"] == "ill"].groupby("i")["tl"].first()
    tr = found["i"].map(ill_at + 1.0).to_numpy()
    kw = dict(tl=found["tl"], i=found["i"], transitions=list(PARS))
    rows = found.copy()
    # only the healthy spell and the illness (the register closes at it)
    with pytest.raises(ValueError, match="by design"):
        ParametricMultiState.fit(
            rows["x"], rows["e"], rows["state"], tr=tr, reached="ill", **kw
        )
    others = _register(200, 7103)
    others["i"] += 10_000
    both = pd.concat([found, others], ignore_index=True)
    tr_both = np.r_[tr, np.full(others.shape[0], np.inf)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ParametricMultiState.fit(
            both["x"], both["e"], both["state"], tl=both["tl"], i=both["i"],
            transitions=list(PARS), tr=tr_both, reached="ill",
        )  # fmt: skip
    assert np.all(np.isfinite(model.params))


def test_left_censored_from_minus_infinity_on_a_real_line_baseline():
    with pytest.raises(ValueError, match="defined below 0"):
        ParametricMultiState.fit(
            [[-np.inf, 2.0], [3.0, 3.0]], ["dead", "dead"], ["alive"] * 2,
            transitions=TWO, dist=sp.Normal,
        )  # fmt: skip
    # with the spell's start given it is read from there
    model = ParametricMultiState.fit(
        [[-np.inf, 2.0], [3.0, 3.0], [4.0, 4.0]], ["dead"] * 3,
        ["alive"] * 3, tl=[0.0, 0.0, 0.0], transitions=TWO, dist=sp.Normal,
    )  # fmt: skip
    assert np.isfinite(model.log_likelihood)


def test_truncated_regression_round_trip():
    x, tau, tl = _truncated(seed=4, n=400)
    Z = pd.DataFrame({"z": (np.arange(x.size) % 2).astype(float)})
    model = ParametricMultiStateRegression.fit(
        x, Z, np.array(["dead"] * x.size, dtype=object),
        np.array(["alive"] * x.size, dtype=object),
        tl=np.where(tl > 0, tl, -np.inf), i=np.arange(x.size),
        transitions=TWO, tr=tau,
    )  # fmt: skip
    back = sp.from_dict(model.to_dict())
    assert isinstance(back, ParametricMultiStateRegression)
    assert_allclose(back.params, model.params)
    assert_allclose(back.covariance(), model.covariance())
    assert_allclose(back._interval_likelihood().neg_ll(back._models()),
                    model.neg_ll(), rtol=1e-12)  # fmt: skip
    assert_allclose(back.occupancy(5.0, [1.0], "dead"),
                    model.occupancy(5.0, [1.0], "dead"))  # fmt: skip
