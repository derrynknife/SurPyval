"""
Tests for time-varying-covariate (start-stop) Cox proportional hazards:
``CoxPH.fit_tvc`` / ``fit_tvc_from_df``, the ``handle_tvc`` validation, the
delayed-entry baseline, and ``predict_tvc``.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval import CoxPH
from surpyval.univariate.regression.proportional_hazards.tvc import handle_tvc


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def _split_dataset(seed=1, n=400):
    """Constant-covariate data, plus a start-stop split of each subject at an
    interior time. The split must give the identical fit."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 1))
    T = rng.exponential(1 / np.exp(Z[:, 0] * 0.7))
    s = T * rng.uniform(0.3, 0.7, size=n)

    ident = np.repeat(np.arange(n), 2)
    xl = np.empty(2 * n)
    xr = np.empty(2 * n)
    c = np.ones(2 * n, dtype=int)  # right-censored (1) unless the event row
    Ztvc = np.empty((2 * n, 1))
    xl[0::2], xr[0::2], c[0::2], Ztvc[0::2] = 0.0, s, 1, Z  # split, censored
    xl[1::2], xr[1::2], c[1::2], Ztvc[1::2] = s, T, 0, Z  # terminal, event
    return (ident, xl, xr, c, Ztvc), (Z, T)


def test_fit_tvc_matches_episode_split_beta_and_baseline():
    (ident, xl, xr, c, Ztvc), (Z, T) = _split_dataset()
    tvc = CoxPH.fit_tvc(ident, xl, xr, c, Ztvc)
    c0 = np.zeros(T.size, dtype=int)
    ref = CoxPH.fit(x=T, Z=Z, c=c0, tl=np.zeros(T.size))

    assert tvc.is_tvc
    # Splitting a subject with a constant covariate is an exact identity of
    # both the partial likelihood and the (delayed-entry) baseline hazard.
    assert np.isclose(tvc.beta[0], ref.beta[0], atol=1e-4)
    tq = np.quantile(T, [0.25, 0.5, 0.75])
    assert np.allclose(
        np.interp(tq, tvc.x, tvc.H0),
        np.interp(tq, ref.x, ref.H0),
        atol=1e-3,
    )


def test_fit_tvc_recovers_time_varying_effect():
    # Genuine time-varying covariate: 0 until a per-subject switch time, then
    # 1. The event time is simulated under the corresponding time-varying
    # hazard, and fit_tvc must recover the coefficient.
    rng = np.random.default_rng(2)
    n, lam, beta = 8000, 0.5, 1.2
    tau = rng.uniform(0.3, 1.5, size=n)
    t1 = rng.exponential(1 / lam, size=n)
    t2 = tau + rng.exponential(1 / (lam * np.exp(beta)), size=n)
    T = np.where(t1 > tau, t2, t1)

    ids, s0, s1, cc, z = [], [], [], [], []
    for k in range(n):
        if T[k] <= tau[k]:
            ids += [k]
            s0 += [0.0]
            s1 += [T[k]]
            cc += [0]  # event
            z += [0.0]
        else:
            ids += [k, k]
            s0 += [0.0, tau[k]]
            s1 += [tau[k], T[k]]
            cc += [1, 0]  # censored change, then event
            z += [0.0, 1.0]
    model = CoxPH.fit_tvc(
        np.array(ids),
        np.array(s0),
        np.array(s1),
        np.array(cc),
        np.array(z).reshape(-1, 1),
    )
    assert abs(model.beta[0] - beta) < 0.15


def test_fit_tvc_from_df_matches_arrays():
    (ident, xl, xr, c, Ztvc), _ = _split_dataset()
    arrays = CoxPH.fit_tvc(ident, xl, xr, c, Ztvc)
    df = pd.DataFrame(
        {
            "id": ident,
            "t0": xl,
            "t1": xr,
            "cc": c,
            "z": Ztvc[:, 0],
        }
    )
    from_df = CoxPH.fit_tvc_from_df(
        df,
        id_col="id",
        xl_col="t0",
        xr_col="t1",
        c_col="cc",
        Z_cols="z",
    )
    assert np.isclose(arrays.beta[0], from_df.beta[0])
    assert from_df.feature_names == ["z"]


def test_predict_tvc_constant_reduces_to_sf():
    (ident, xl, xr, c, Ztvc), _ = _split_dataset()
    model = CoxPH.fit_tvc(ident, xl, xr, c, Ztvc)
    z = np.array([0.5])
    t = np.array([0.3, 0.7, 1.2])
    # a single constant interval spanning the times must equal sf(t, Z)
    _, sf_tvc, Hf_tvc = model.predict_tvc([0.0], [10.0], [z], times=t)
    assert np.allclose(sf_tvc, model.sf(t, z))
    assert np.allclose(Hf_tvc, model.Hf(t, z))


def test_predict_tvc_changing_covariate_is_monotone_and_responds():
    (ident, xl, xr, c, Ztvc), _ = _split_dataset()
    model = CoxPH.fit_tvc(ident, xl, xr, c, Ztvc)

    times, sf, _ = model.predict_tvc(
        xl=[0.0, 0.5], xr=[0.5, 5.0], Z=[[-1.0], [1.5]]
    )
    # survival is a proper (non-increasing) curve
    assert np.all(np.diff(sf) <= 1e-12)
    assert np.all((sf >= 0) & (sf <= 1))
    # switching to a higher-hazard covariate late must leave survival below
    # the all-low-covariate path at the final time
    _, sf_low, _ = model.predict_tvc(
        xl=[0.0], xr=[5.0], Z=[[-1.0]], times=times[-1:]
    )
    assert sf[-1] < sf_low[-1]


@pytest.mark.parametrize(
    "kwargs",
    [
        # overlapping intervals
        dict(
            i=[1, 1],
            xl=[0, 1],
            xr=[3, 4],
            c=[1, 0],
            Z=[[0.1], [0.2]],
        ),
        # more than one event
        dict(
            i=[1, 1],
            xl=[0, 2],
            xr=[2, 4],
            c=[0, 0],
            Z=[[0.1], [0.2]],
        ),
        # event on a non-final interval
        dict(
            i=[1, 1],
            xl=[0, 2],
            xr=[2, 4],
            c=[0, 1],
            Z=[[0.1], [0.2]],
        ),
        # xl >= xr
        dict(
            i=[1, 1],
            xl=[0, 4],
            xr=[2, 4],
            c=[1, 0],
            Z=[[0.1], [0.2]],
        ),
        # length mismatch
        dict(
            i=[1, 1],
            xl=[0, 2],
            xr=[2, 4],
            c=[1],
            Z=[[0.1], [0.2]],
        ),
        # bad c value
        dict(
            i=[1],
            xl=[0],
            xr=[2],
            c=[2],
            Z=[[0.1]],
        ),
    ],
)
def test_handle_tvc_validation(kwargs):
    with pytest.raises(ValueError):
        handle_tvc(**kwargs)


def test_handle_tvc_maps_columns():
    x, c, n, tl, Z, ident = handle_tvc(
        i=[1, 1, 2],
        xl=[0, 2, 0],
        xr=[2, 4, 3],
        c=[1, 0, 1],
        Z=[[0.1], [0.2], [0.3]],
    )
    assert np.array_equal(x, [2.0, 4.0, 3.0])  # x = xr
    assert np.array_equal(c, [1, 0, 1])  # 0 event, 1 censored
    assert np.array_equal(tl, [0.0, 2.0, 0.0])  # tl = xl


def test_predict_tvc_input_validation():
    (ident, xl, xr, c, Ztvc), _ = _split_dataset()
    model = CoxPH.fit_tvc(ident, xl, xr, c, Ztvc)
    with pytest.raises(ValueError):
        model.predict_tvc(xl=[0.0, 1.0], xr=[1.0], Z=[[0.1], [0.2]])
    with pytest.raises(ValueError):
        model.predict_tvc(xl=[2.0], xr=[1.0], Z=[[0.1]])


# ---------------------------------------------------------------------------
# Timeline / xicnt-style TVC input (fit_tvc_timeline)
# ---------------------------------------------------------------------------

from surpyval.univariate.regression.proportional_hazards.tvc import (  # noqa: E402,E501
    handle_tvc_timeline,
)


def _timeline_and_startstop(seed=3, nsub=150):
    """Build the same TVC data in both the timeline and start-stop layouts:
    each subject has a covariate that steps once, then an event/censor."""
    rng = np.random.default_rng(seed)
    ss = {"ident": [], "xl": [], "xr": [], "c": [], "Z": []}
    tl = {"i": [], "x": [], "Z": [], "c": []}
    for s in range(nsub):
        z0 = rng.normal()
        z1 = z0 + 0.5 * rng.normal()
        change = rng.uniform(1.0, 4.0)
        exit_t = change + rng.uniform(0.5, 5.0)
        died = int(rng.uniform() < 0.7)

        ss["ident"] += [s, s]
        ss["xl"] += [0.0, change]
        ss["xr"] += [change, exit_t]
        ss["c"] += [1, 0 if died else 1]
        ss["Z"] += [[z0], [z1]]

        tl["i"] += [s, s, s]
        tl["x"] += [0.0, change, exit_t]
        tl["Z"] += [[z0], [z1], [z1]]  # terminal Z ignored
        tl["c"] += [-1, -1, 0 if died else 1]
    return ss, tl


def test_fit_tvc_timeline_matches_start_stop():
    ss, tl = _timeline_and_startstop()
    m_ss = CoxPH.fit_tvc(
        i=ss["ident"],
        xl=ss["xl"],
        xr=ss["xr"],
        c=ss["c"],
        Z=np.array(ss["Z"]),
    )
    m_tl = CoxPH.fit_tvc_timeline(
        i=tl["i"], x=tl["x"], Z=np.array(tl["Z"]), c=tl["c"]
    )
    assert np.allclose(m_ss.beta, m_tl.beta, atol=1e-8)
    assert m_tl.is_tvc


def test_handle_tvc_timeline_carries_covariate_forward():
    # Two rows -> one interval; a change -> two intervals; entry = first x.
    i_out, xl, xr, c, Z, n = handle_tvc_timeline(
        i=[1, 1, 1],
        x=[2.0, 5.0, 8.0],
        Z=[[0.0], [1.0], [9.9]],  # terminal 9.9 is dropped
        c=[-1, -1, 0],
    )
    assert list(xl) == [2.0, 5.0]  # entry is the first x (delayed entry)
    assert list(xr) == [5.0, 8.0]
    assert list(c) == [1, 0]  # event (c=0) on the last interval
    assert Z.ravel().tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(i=[1], x=[3.0], Z=[[0.5]], c=[0]), "single timeline row"),
        (
            dict(i=[1, 1], x=[5.0, 5.0], Z=[[0.0], [0.0]], c=[-1, 0]),
            "non-increasing",
        ),
        (
            dict(i=[1, 1], x=[0.0, 5.0], Z=[[0.0], [0.0]], c=[-1, 2]),
            "terminal status",
        ),
    ],
)
def test_handle_tvc_timeline_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        handle_tvc_timeline(**kwargs)


def test_fit_tvc_timeline_from_df_matches_arrays():
    ss, tl = _timeline_and_startstop(seed=5, nsub=120)
    df = pd.DataFrame(
        {
            "subj": tl["i"],
            "t": tl["x"],
            "z": [row[0] for row in tl["Z"]],
            "cc": tl["c"],
        }
    )
    m_df = CoxPH.fit_tvc_timeline_from_df(
        df, id_col="subj", time_col="t", Z_cols="z", c_col="cc"
    )
    m_arr = CoxPH.fit_tvc_timeline(
        i=tl["i"], x=tl["x"], Z=np.array(tl["Z"]), c=tl["c"]
    )
    assert np.allclose(m_df.beta, m_arr.beta)
    assert m_df.feature_names == ["z"]
