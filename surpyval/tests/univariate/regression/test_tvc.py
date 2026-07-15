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
    start = np.empty(2 * n)
    stop = np.empty(2 * n)
    event = np.zeros(2 * n, dtype=int)
    Ztvc = np.empty((2 * n, 1))
    start[0::2], stop[0::2], event[0::2], Ztvc[0::2] = 0.0, s, 0, Z
    start[1::2], stop[1::2], event[1::2], Ztvc[1::2] = s, T, 1, Z
    return (ident, start, stop, event, Ztvc), (Z, T)


def test_fit_tvc_matches_episode_split_beta_and_baseline():
    (ident, start, stop, event, Ztvc), (Z, T) = _split_dataset()
    tvc = CoxPH.fit_tvc(ident, start, stop, event, Ztvc)
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

    ids, s0, s1, ev, z = [], [], [], [], []
    for i in range(n):
        if T[i] <= tau[i]:
            ids += [i]
            s0 += [0.0]
            s1 += [T[i]]
            ev += [1]
            z += [0.0]
        else:
            ids += [i, i]
            s0 += [0.0, tau[i]]
            s1 += [tau[i], T[i]]
            ev += [0, 1]
            z += [0.0, 1.0]
    model = CoxPH.fit_tvc(
        np.array(ids),
        np.array(s0),
        np.array(s1),
        np.array(ev),
        np.array(z).reshape(-1, 1),
    )
    assert abs(model.beta[0] - beta) < 0.15


def test_fit_tvc_from_df_matches_arrays():
    (ident, start, stop, event, Ztvc), _ = _split_dataset()
    arrays = CoxPH.fit_tvc(ident, start, stop, event, Ztvc)
    df = pd.DataFrame(
        {
            "id": ident,
            "t0": start,
            "t1": stop,
            "ev": event,
            "z": Ztvc[:, 0],
        }
    )
    from_df = CoxPH.fit_tvc_from_df(
        df,
        id_col="id",
        start_col="t0",
        stop_col="t1",
        event_col="ev",
        Z_cols="z",
    )
    assert np.isclose(arrays.beta[0], from_df.beta[0])
    assert from_df.feature_names == ["z"]


def test_predict_tvc_constant_reduces_to_sf():
    (ident, start, stop, event, Ztvc), _ = _split_dataset()
    model = CoxPH.fit_tvc(ident, start, stop, event, Ztvc)
    z = np.array([0.5])
    t = np.array([0.3, 0.7, 1.2])
    # a single constant interval spanning the times must equal sf(t, Z)
    _, sf_tvc, Hf_tvc = model.predict_tvc([0.0], [10.0], [z], times=t)
    assert np.allclose(sf_tvc, model.sf(t, z))
    assert np.allclose(Hf_tvc, model.Hf(t, z))


def test_predict_tvc_changing_covariate_is_monotone_and_responds():
    (ident, start, stop, event, Ztvc), _ = _split_dataset()
    model = CoxPH.fit_tvc(ident, start, stop, event, Ztvc)

    times, sf, _ = model.predict_tvc(
        start=[0.0, 0.5], stop=[0.5, 5.0], Z=[[-1.0], [1.5]]
    )
    # survival is a proper (non-increasing) curve
    assert np.all(np.diff(sf) <= 1e-12)
    assert np.all((sf >= 0) & (sf <= 1))
    # switching to a higher-hazard covariate late must leave survival below
    # the all-low-covariate path at the final time
    _, sf_low, _ = model.predict_tvc(
        start=[0.0], stop=[5.0], Z=[[-1.0]], times=times[-1:]
    )
    assert sf[-1] < sf_low[-1]


@pytest.mark.parametrize(
    "kwargs",
    [
        # overlapping intervals
        dict(
            ident=[1, 1],
            start=[0, 1],
            stop=[3, 4],
            event=[0, 1],
            Z=[[0.1], [0.2]],
        ),
        # more than one event
        dict(
            ident=[1, 1],
            start=[0, 2],
            stop=[2, 4],
            event=[1, 1],
            Z=[[0.1], [0.2]],
        ),
        # event on a non-final interval
        dict(
            ident=[1, 1],
            start=[0, 2],
            stop=[2, 4],
            event=[1, 0],
            Z=[[0.1], [0.2]],
        ),
        # start >= stop
        dict(
            ident=[1, 1],
            start=[0, 4],
            stop=[2, 4],
            event=[0, 1],
            Z=[[0.1], [0.2]],
        ),
        # length mismatch
        dict(
            ident=[1, 1],
            start=[0, 2],
            stop=[2, 4],
            event=[0],
            Z=[[0.1], [0.2]],
        ),
        # bad event value
        dict(
            ident=[1],
            start=[0],
            stop=[2],
            event=[2],
            Z=[[0.1]],
        ),
    ],
)
def test_handle_tvc_validation(kwargs):
    with pytest.raises(ValueError):
        handle_tvc(**kwargs)


def test_handle_tvc_maps_columns():
    x, c, n, tl, Z, ident = handle_tvc(
        ident=[1, 1, 2],
        start=[0, 2, 0],
        stop=[2, 4, 3],
        event=[0, 1, 0],
        Z=[[0.1], [0.2], [0.3]],
    )
    assert np.array_equal(x, [2.0, 4.0, 3.0])
    assert np.array_equal(c, [1, 0, 1])  # event -> c=0, else c=1
    assert np.array_equal(tl, [0.0, 2.0, 0.0])


def test_predict_tvc_input_validation():
    (ident, start, stop, event, Ztvc), _ = _split_dataset()
    model = CoxPH.fit_tvc(ident, start, stop, event, Ztvc)
    with pytest.raises(ValueError):
        model.predict_tvc(start=[0.0, 1.0], stop=[1.0], Z=[[0.1], [0.2]])
    with pytest.raises(ValueError):
        model.predict_tvc(start=[2.0], stop=[1.0], Z=[[0.1]])
