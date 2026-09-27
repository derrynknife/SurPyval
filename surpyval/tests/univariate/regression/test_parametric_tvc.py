"""Time-varying-covariate fitting for the parametric regression families
(issues #150, #372).

For proportional-hazards, additive-hazards and proportional-odds models the
hazard at time t depends only on t and the current covariate, so the
cumulative hazard is additive over disjoint time intervals and a
time-varying-covariate subject factorises exactly into one left-truncated
observation per constant-covariate interval. ``fit_tvc`` (and the timeline /
DataFrame variants) reshape the data and reuse the ordinary parametric MLE
``fit``; these tests lock in that the reshape is an exact identity, that a
genuine time-varying effect is recovered, and that the fitted likelihood is
the one ``sf_tvc`` / ``hf`` give along each subject's path. (Accelerated
failure time, where the reshape is invalid, has its own likelihood -- see
test_aft_tvc_fit.py.)
"""

import numpy as np
import pandas as pd
import pytest

from surpyval import (
    AFT,
    PO,
    ExponentialPH,
    LogisticPO,
    NormalPH,
    ParametricRegressionModel,
    Weibull,
    WeibullAH,
    WeibullPH,
    WeibullPO,
)

TVC_FITTERS = {
    "WeibullPH": WeibullPH,
    "WeibullAH": WeibullAH,
    "WeibullPO": WeibullPO,
}


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


def test_po_and_aft_expose_tvc():
    # Proportional odds fits through the episode-split mixin (#372);
    # accelerated failure time through its own accumulated-age likelihood
    # (see test_aft_tvc_fit.py).
    for method in (
        "fit_tvc",
        "fit_tvc_timeline",
        "fit_tvc_from_df",
        "fit_tvc_timeline_from_df",
    ):
        assert hasattr(PO(Weibull), method)
    assert hasattr(AFT(Weibull), "fit_tvc")


def test_left_truncated_likelihood_does_not_reward_a_vanishing_scale():
    """A region the data rules out must not report a better likelihood.

    The truncation correction used to be a difference of CDFs floored at
    the smallest positive float. Under left truncation that difference is
    ``1 - F(tl)``, which underflows to zero as the fitted scale shrinks;
    the floor then capped the correction at ``log(tiny) = -708`` rather
    than letting it grow without bound. Since the correction is
    *subtracted*, every truncated row appeared to contribute +708, and
    the optimiser walked into a region the data excludes -- reporting a
    log-likelihood some 38,000 higher than its parameters earn (#326).

    The check is that the reported objective is the real one: a scale far
    below the truncation bounds must score worse than the fitted answer,
    not better.
    """
    (Z, T, c0), (ident, xl, xr, c, Zs) = _constant_covariate_split()
    fitter = TVC_FITTERS["WeibullPH"]

    model = fitter.fit_tvc(i=ident, xl=xl, xr=xr, c=c, Z=Zs)
    fitted = np.asarray(model.params, dtype=float)

    # Same objective the fit minimised, evaluated directly.
    def neg_ll(params):
        return float(fitter.neg_ll(model.data, *params))

    assert neg_ll(fitted) == pytest.approx(model._neg_ll, rel=1e-9)

    # Shrinking the scale far below the truncation bounds is where the
    # floor used to manufacture likelihood. Every such point must be
    # worse than the fit.
    for shrink in (10.0, 100.0, 1000.0):
        degenerate = fitted.copy()
        degenerate[0] = fitted[0] / shrink
        assert neg_ll(degenerate) > neg_ll(fitted), (
            f"scale/{shrink:g} scores better than the fit, so the "
            f"truncation correction is still being floored"
        )


# -- Proportional odds (#372) -----------------------------------------------

# A WeibullPO model in the survival-odds convention: S0 is Weibull(10, 2) and
# phi = exp(b1 z1 + b2 z2) multiplies the survival odds.
PO_TRUTH = np.array([10.0, 2.0, 1.0, -0.5])


def _po_sf(t, z1, z2):
    alpha, beta, b1, b2 = PO_TRUTH
    phi = np.exp(b1 * z1 + b2 * z2)
    S0 = np.exp(-((t / alpha) ** beta))
    return phi * S0 / (1 - S0 + phi * S0)


def _po_isf(S, z1, z2):
    # Invert _po_sf: the survival odds S / (1 - S) are phi times the
    # baseline odds, which give S0 and then the Weibull time.
    alpha, beta, b1, b2 = PO_TRUTH
    phi = np.exp(b1 * z1 + b2 * z2)
    odds0 = S / (1 - S) / phi
    S0 = odds0 / (1 + odds0)
    return alpha * (-np.log(S0)) ** (1 / beta)


def _po_step_path_data(seed, n):
    """Start-stop data from the PO model with a covariate z1 that switches
    from 0 to 1 at a random time and a constant z2, right-censored at a
    random time. Each failure time is drawn by inverting the path survival:
    S(t | z1=0) before the switch, and after it S(tau | 0) times the
    exposed hazard's survival from tau, S(t | 1) / S(tau | 1)."""
    rng = np.random.default_rng(seed)
    tau = rng.uniform(2, 12, n)
    z2 = rng.normal(size=n)
    U = rng.uniform(size=n)
    T = _po_isf(U, 0.0, z2)
    late = T > tau
    target = U * _po_sf(tau, 1.0, z2) / _po_sf(tau, 0.0, z2)
    T[late] = _po_isf(target[late], 1.0, z2[late])
    C = rng.uniform(5, 30, n)
    X = np.minimum(T, C)
    c_end = np.where(T <= C, 0, 1)
    split = X > tau
    i = np.r_[np.arange(n), np.flatnonzero(split)]
    xl = np.r_[np.zeros(n), tau[split]]
    xr = np.r_[np.where(split, tau, X), X[split]]
    c = np.r_[np.where(split, 1, c_end), c_end[split]]
    Z = np.r_[
        np.column_stack([np.zeros(n), z2]),
        np.column_stack([np.ones(split.sum()), z2[split]]),
    ]
    order = np.lexsort((xl, i))  # each subject's rows together, in order
    return i[order], xl[order], xr[order], c[order], Z[order]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_po_fit_tvc_recovers_step_path_parameters(seed):
    # Over 8 seeds of 2,000 subjects the estimates average
    # [10.034, 1.978, 0.983, -0.498] (sd 0.14, 0.034, 0.082, 0.033); each
    # fit is required to be within 4 standard errors of the truth.
    i, xl, xr, c, Z = _po_step_path_data(seed, 2000)
    model = WeibullPO.fit_tvc(i, xl, xr, c, Z)
    se = np.sqrt(np.diag(model.covariance()))
    assert model.is_tvc
    assert np.all(np.abs(model.params - PO_TRUTH) < 4 * se)


def _path_neg_ll(model, i, xl, xr, c, Z):
    """-sum of log sf_tvc (and log hf at a failure) along each subject's
    covariate path, conditioned on survival to its entry time."""
    ll = 0.0
    for k in np.unique(i):
        r = i == k
        starts = np.r_[0.0, xl[r][1:]]
        ll += np.log(model.sf_tvc([xr[r][-1]], Z[r], xl=starts)[0])
        if xl[r][0] > 0:
            ll -= np.log(model.sf_tvc([xl[r][0]], Z[r], xl=starts)[0])
        if c[r][-1] == 0:
            ll += np.log(model.hf([xr[r][-1]], Z[r][-1:])[0])
    return -ll


@pytest.mark.filterwarnings("ignore:The additive hazards fit ended")
@pytest.mark.parametrize(
    "fitter",
    [WeibullPO, LogisticPO, WeibullPH, NormalPH, WeibullAH],
    ids=["WeibullPO", "LogisticPO", "WeibullPH", "NormalPH", "WeibullAH"],
)
def test_tvc_neg_ll_is_the_path_likelihood(fitter):
    # The episode-split likelihood equals the path likelihood built from
    # sf_tvc and hf, with delayed entry for a quarter of the subjects. A
    # baseline defined below zero (Logistic, Normal) used to fail this:
    # every subject's first interval was truncated at 0, conditioning the
    # fit on survival to 0, which sf_tvc does not do.
    i, xl, xr, c, Z = _po_step_path_data(0, 400)
    first = np.r_[True, i[1:] != i[:-1]]
    xl = np.where(first & (i % 4 == 0) & (xr > 0.5), 0.5, xl)
    model = fitter.fit_tvc(i, xl, xr, c, Z)
    assert model._neg_ll == pytest.approx(
        _path_neg_ll(model, i, xl, xr, c, Z), rel=1e-10
    )


@pytest.mark.parametrize(
    "fitter",
    [WeibullPO, LogisticPO, NormalPH],
    ids=["WeibullPO", "LogisticPO", "NormalPH"],
)
def test_constant_covariate_tvc_reproduces_fit(fitter):
    # Splitting constant-covariate subjects into intervals must give the
    # ordinary fit. For LogisticPO the tvc fit used to return alpha 5.215
    # (vs 9.330) and neg-ll 952.57 (vs 1010.69); NormalPH 5.557 vs 9.650.
    (Z, T, c0), (ident, xl, xr, c, Zs) = _constant_covariate_split()
    plain = fitter.fit(x=T, Z=Z, c=c0)
    tvc = fitter.fit_tvc(i=ident, xl=xl, xr=xr, c=c, Z=Zs)
    assert np.allclose(plain.params, tvc.params, rtol=1e-4, atol=1e-4)
    assert tvc._neg_ll == pytest.approx(plain._neg_ll, rel=1e-8)


def test_po_tvc_timeline_and_df_match_start_stop():
    ss, tl = _timeline_pair()
    m_ss = WeibullPO.fit_tvc(
        i=ss["i"], xl=ss["xl"], xr=ss["xr"], c=ss["c"], Z=np.array(ss["Z"])
    )
    m_tl = WeibullPO.fit_tvc_timeline(
        i=tl["i"], x=tl["x"], Z=np.array(tl["Z"]), c=tl["c"]
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
    m_df = WeibullPO.fit_tvc_from_df(
        df,
        id_col="subj",
        xl_col="start",
        xr_col="stop",
        c_col="status",
        Z_cols="z",
    )
    assert np.allclose(m_ss.params, m_tl.params, atol=1e-6)
    assert np.allclose(m_ss.params, m_df.params)
    assert m_df.feature_names == ["z"]


def test_po_tvc_serialises_and_counts_subjects():
    import json

    i, xl, xr, c, Z = _po_step_path_data(1, 300)
    model = WeibullPO.fit_tvc(i, xl, xr, c, Z)
    restored = ParametricRegressionModel.from_dict(
        json.loads(json.dumps(model.to_dict()))
    )
    x = [2.0, 6.0, 12.0]
    path = [[0.0, 0.3], [1.0, 0.3]]
    np.testing.assert_allclose(
        restored.sf_tvc(x, path, xl=[0, 5]), model.sf_tvc(x, path, xl=[0, 5])
    )
    np.testing.assert_allclose(
        restored.Hf_tvc(x, path, xl=[0, 5]), model.Hf_tvc(x, path, xl=[0, 5])
    )
    assert restored.bic() == pytest.approx(model.bic())
    assert restored.aic_c() == pytest.approx(model.aic_c())
    np.testing.assert_allclose(
        restored.cb(x, [[1.0, 0.3]]), model.cb(x, [[1.0, 0.3]])
    )
    assert model.n_subjects == 300


def test_po_tvc_aic_c_is_invariant_to_episode_splitting():
    rng = np.random.default_rng(5)
    n = 80
    x = 10 * rng.weibull(1.5, n)
    z = rng.binomial(1, 0.5, n).astype(float)
    whole = WeibullPO.fit_tvc(np.arange(n), np.zeros(n), x, np.zeros(n), z)
    split = WeibullPO.fit_tvc(
        np.r_[np.arange(n), np.arange(n)],
        np.r_[np.zeros(n), x / 2],
        np.r_[x / 2, x],
        np.r_[np.ones(n), np.zeros(n)],
        np.r_[z, z],
    )
    assert split.aic_c() == pytest.approx(whole.aic_c(), rel=1e-6)
    assert split.params == pytest.approx(whole.params, rel=1e-4)
