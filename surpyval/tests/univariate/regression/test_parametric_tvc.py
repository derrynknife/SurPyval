"""Time-varying-covariate fitting for the parametric regression families
(issue #150).

For proportional-hazards and additive-hazards models the cumulative hazard is
additive over disjoint time intervals, so a time-varying-covariate subject
factorises exactly into one left-truncated observation per constant-covariate
interval. ``fit_tvc`` (and the timeline / DataFrame variants) reshape the data
and reuse the ordinary parametric MLE ``fit``; these tests lock in that the
reshape is an exact identity, that a genuine time-varying effect is recovered,
and that the families where the trick is invalid (AFT, PO) do not expose it.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval import (
    AFT,
    PO,
    ExponentialPH,
    Weibull,
    WeibullAH,
    WeibullPH,
)

TVC_FITTERS = {"WeibullPH": WeibullPH, "WeibullAH": WeibullAH}


def _constant_covariate_split(seed=0, n=300):
    """Constant-covariate survival data, plus a start-stop split of each
    subject at an interior time (same covariate on both halves)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 1))
    T = 10.0 * np.exp(-Z[:, 0] * 0.8 / 1.5) * rng.weibull(1.5, n) + 0.2
    c0 = np.zeros(n, dtype=int)

    s = T * rng.uniform(0.3, 0.7, n)
    ident = np.concatenate([np.arange(n), np.arange(n)])
    xl = np.concatenate([np.zeros(n), s])
    xr = np.concatenate([s, T])
    c = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    Zs = np.concatenate([Z, Z])
    return (Z, T, c0), (ident, xl, xr, c, Zs)


@pytest.mark.parametrize("name", list(TVC_FITTERS))
def test_episode_split_is_an_identity(name):
    fitter = TVC_FITTERS[name]
    (Z, T, c0), (ident, xl, xr, c, Zs) = _constant_covariate_split()

    plain = fitter.fit(x=T, Z=Z, c=c0)
    tvc = fitter.fit_tvc(i=ident, xl=xl, xr=xr, c=c, Z=Zs)

    # Splitting a subject with a constant covariate into left-truncated
    # intervals must reproduce the un-split parametric fit exactly.
    assert np.allclose(plain.params, tvc.params, atol=1e-3)
    assert tvc.is_tvc


def test_parametric_ph_recovers_time_varying_effect():
    # Covariate is 0 until a per-subject switch time tau, then 1. The event
    # time is simulated under the corresponding time-varying hazard; the PH
    # coefficient must come back near its true value.
    rng = np.random.default_rng(7)
    n, lam, beta = 4000, 0.5, 1.0
    tau = rng.uniform(0.3, 1.5, n)
    t1 = rng.exponential(1 / lam, n)
    t2 = tau + rng.exponential(1 / (lam * np.exp(beta)), n)
    T = np.where(t1 > tau, t2, t1)

    ids, xl, xr, c, z = [], [], [], [], []
    for k in range(n):
        if T[k] <= tau[k]:
            ids += [k]
            xl += [0.0]
            xr += [T[k]]
            c += [0]
            z += [0.0]
        else:
            ids += [k, k]
            xl += [0.0, tau[k]]
            xr += [tau[k], T[k]]
            c += [1, 0]
            z += [0.0, 1.0]

    model = WeibullPH.fit_tvc(
        i=np.array(ids),
        xl=np.array(xl),
        xr=np.array(xr),
        c=np.array(c),
        Z=np.array(z).reshape(-1, 1),
    )
    assert abs(float(model.params[-1]) - beta) < 0.15


def _timeline_pair(seed=3, nsub=150):
    rng = np.random.default_rng(seed)
    ss = {"i": [], "xl": [], "xr": [], "c": [], "Z": []}
    tl = {"i": [], "x": [], "Z": [], "c": []}
    for s in range(nsub):
        z0 = rng.normal()
        z1 = z0 + 0.4 * rng.normal()
        change = rng.uniform(1.0, 4.0)
        exit_t = change + rng.uniform(0.5, 5.0)
        died = int(rng.uniform() < 0.7)
        ss["i"] += [s, s]
        ss["xl"] += [0.0, change]
        ss["xr"] += [change, exit_t]
        ss["c"] += [1, 0 if died else 1]
        ss["Z"] += [[z0], [z1]]
        tl["i"] += [s, s, s]
        tl["x"] += [0.0, change, exit_t]
        tl["Z"] += [[z0], [z1], [z1]]
        tl["c"] += [-1, -1, 0 if died else 1]
    return ss, tl


def test_timeline_matches_start_stop():
    ss, tl = _timeline_pair()
    m_ss = WeibullPH.fit_tvc(
        i=ss["i"], xl=ss["xl"], xr=ss["xr"], c=ss["c"], Z=np.array(ss["Z"])
    )
    m_tl = WeibullPH.fit_tvc_timeline(
        i=tl["i"], x=tl["x"], Z=np.array(tl["Z"]), c=tl["c"]
    )
    assert np.allclose(m_ss.params, m_tl.params, atol=1e-6)


def test_fit_tvc_from_df_matches_arrays():
    ss, _ = _timeline_pair(seed=5, nsub=120)
    arrays = WeibullPH.fit_tvc(
        i=ss["i"], xl=ss["xl"], xr=ss["xr"], c=ss["c"], Z=np.array(ss["Z"])
    )
    df = pd.DataFrame(
        {
            "subj": ss["i"],
            "start": ss["xl"],
            "stop": ss["xr"],
            "status": ss["c"],
            "z": [row[0] for row in ss["Z"]],
        }
    )
    from_df = WeibullPH.fit_tvc_from_df(
        df,
        id_col="subj",
        xl_col="start",
        xr_col="stop",
        c_col="status",
        Z_cols="z",
    )
    assert np.allclose(arrays.params, from_df.params)
    assert from_df.feature_names == ["z"]


def test_exponential_ph_tvc_also_supported():
    # A one-parameter baseline still works through the same path.
    (_, T, c0), (ident, xl, xr, c, Zs) = _constant_covariate_split(seed=2)
    plain = ExponentialPH.fit(x=T, Z=Zs[: len(T)], c=c0)  # noqa: F841
    tvc = ExponentialPH.fit_tvc(i=ident, xl=xl, xr=xr, c=c, Z=Zs)
    assert tvc.is_tvc
    assert np.all(np.isfinite(tvc.params))


def test_po_does_not_expose_tvc():
    # Proportional odds has no additive/accumulated structure, so it must not
    # offer fit_tvc. (Accelerated failure time now does, via its own
    # accumulated-age likelihood -- see test_aft_tvc_fit.py.)
    assert not hasattr(PO(Weibull), "fit_tvc")
    assert hasattr(AFT(Weibull), "fit_tvc")
