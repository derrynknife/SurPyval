r"""
Time-varying-covariate *fitting* for accelerated failure time.

Unlike the proportional/additive-hazards families (whose ``fit_tvc`` reshapes
the episodes into independent left-truncated rows and reuses the ordinary MLE),
AFT rescales the time axis, so a subject's likelihood depends on its
*accumulated* accelerated age across intervals and needs a bespoke likelihood.
These tests pin the properties that make that likelihood correct:

* a single-interval fit is identical to the ordinary ``AFT.fit``;
* splitting a subject into equal-covariate contiguous episodes leaves the fit
  unchanged (accelerated age accrues the same either way);
* a genuine time-varying effect is recovered;
* the confidence-bound path (a numerical Hessian of the *custom* likelihood)
  produces a valid covariance;
* the ordinary AFT/PH fits -- the shared MLE code -- are untouched.
"""

import numpy as np

from surpyval import WeibullAFT, WeibullPH
from surpyval.univariate.regression.accelerated_failure_time import (
    aft_tvc_fit,
)


def _plain_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, n)
    x = np.abs(rng.weibull(1.5, n) * 10 * np.exp(-0.3 * Z)) + 0.5
    c = np.zeros(n, dtype=int)
    return Z, x, c


def test_single_interval_equals_ordinary_aft():
    Z, x, c = _plain_data()
    n = x.shape[0]
    plain = WeibullAFT.fit(x=x, Z=Z.reshape(-1, 1), c=c)
    tvc = WeibullAFT.fit_tvc(
        np.arange(n), np.zeros(n), x, np.zeros(n, dtype=int), Z.reshape(-1, 1)
    )
    assert np.allclose(plain.params, tvc.params, atol=1e-2)
    assert tvc.is_tvc
    assert tvc.n_subjects == n


def test_equal_covariate_split_is_identity():
    Z, x, c = _plain_data(seed=1, n=300)
    n = x.shape[0]
    one = WeibullAFT.fit_tvc(
        np.arange(n), np.zeros(n), x, np.zeros(n, dtype=int), Z.reshape(-1, 1)
    )
    i, xl, xr, cc, Zr = [], [], [], [], []
    for s in range(n):
        mid = x[s] / 2
        i += [s, s]
        xl += [0.0, mid]
        xr += [mid, x[s]]
        cc += [1, 0]
        Zr += [[Z[s]], [Z[s]]]  # SAME covariate on both halves
    split = WeibullAFT.fit_tvc(
        np.array(i), np.array(xl), np.array(xr), np.array(cc), np.array(Zr)
    )
    assert np.allclose(one.params, split.params, atol=1e-3)


def test_recovers_time_varying_effect():
    rng = np.random.default_rng(2)
    N = 4000
    true_beta = 0.8
    lam, k = 8.0, 1.4
    switch = rng.uniform(1.0, 4.0, size=N)
    E = -np.log(rng.uniform(size=N))
    psi_target = lam * E ** (1.0 / k)  # baseline age at the event
    phi_after = np.exp(true_beta)
    T = np.where(
        psi_target <= switch,
        psi_target,
        switch + (psi_target - switch) / phi_after,
    )
    i, xl, xr, c, Z = [], [], [], [], []
    for s in range(N):
        if T[s] <= switch[s]:
            i += [s]
            xl += [0.0]
            xr += [T[s]]
            c += [0]
            Z += [[0.0]]
        else:
            i += [s, s]
            xl += [0.0, switch[s]]
            xr += [switch[s], T[s]]
            c += [1, 0]
            Z += [[0.0], [1.0]]
    m = WeibullAFT.fit_tvc(
        np.array(i), np.array(xl), np.array(xr), np.array(c), np.array(Z)
    )
    assert abs(float(m.phi_params[0]) - true_beta) < 0.1


def test_confidence_bounds_use_the_custom_likelihood():
    # The covariance is a numerical Hessian of model.model.neg_ll(model.data,
    # ...). Because the fit binds the accumulated-age likelihood onto the
    # result's fitter, the diagonal must be finite and positive -- i.e. the
    # bounds reflect the TVC likelihood, not the independent-rows one.
    rng = np.random.default_rng(3)
    n = 500
    Z = rng.normal(0, 1, n)
    x = np.abs(rng.weibull(1.5, n) * 10 * np.exp(-0.4 * Z)) + 0.5
    i, xl, xr, c, Zr = [], [], [], [], []
    for s in range(n):
        mid = x[s] * 0.5
        i += [s, s]
        xl += [0.0, mid]
        xr += [mid, x[s]]
        c += [1, 0]
        Zr += [[Z[s]], [Z[s] + 0.1]]
    m = WeibullAFT.fit_tvc(
        np.array(i), np.array(xl), np.array(xr), np.array(c), np.array(Zr)
    )
    cov = m.covariance()
    assert cov.shape == (3, 3)
    assert np.all(np.isfinite(np.diag(cov))) and np.all(np.diag(cov) > 0)


def test_information_criteria_use_subject_counts():
    # aic_c's finite-sample term must use the subject count, not the (larger)
    # episode-row count; with two episodes per subject the two would differ.
    Z, x, _ = _plain_data(seed=4, n=200)
    n = x.shape[0]
    i, xl, xr, c, Zr = [], [], [], [], []
    for s in range(n):
        mid = x[s] * 0.5
        i += [s, s]
        xl += [0.0, mid]
        xr += [mid, x[s]]
        c += [1, 0]
        Zr += [[Z[s]], [Z[s]]]
    m = WeibullAFT.fit_tvc(
        np.array(i), np.array(xl), np.array(xr), np.array(c), np.array(Zr)
    )
    assert m.n_subjects == n
    k = m.k
    expected_aic_c = m.aic() + (2 * k**2 + 2 * k) / (n - k - 1)
    assert np.isclose(m.aic_c(), expected_aic_c)


def test_from_df_and_timeline_match_arrays():
    import pandas as pd

    Z, x, _ = _plain_data(seed=5, n=150)
    n = x.shape[0]
    rows = []
    tl_i, tl_x, tl_Z, tl_c = [], [], [], []
    for s in range(n):
        mid = x[s] * 0.5
        rows.append((s, 0.0, mid, 1, Z[s]))
        rows.append((s, mid, x[s], 0, Z[s] + 0.2))
        # timeline: entry, change, exit
        tl_i += [s, s, s]
        tl_x += [0.0, mid, x[s]]
        tl_Z += [[Z[s]], [Z[s] + 0.2], [Z[s] + 0.2]]
        tl_c += [1, 1, 0]
    df = pd.DataFrame(rows, columns=["id", "xl", "xr", "c", "z"])
    m_df = WeibullAFT.fit_tvc_from_df(
        df, id_col="id", xl_col="xl", xr_col="xr", c_col="c", Z_cols="z"
    )
    m_arr = WeibullAFT.fit_tvc(
        df["id"].values,
        df["xl"].values,
        df["xr"].values,
        df["c"].values,
        df["z"].values.reshape(-1, 1),
    )
    assert np.allclose(m_df.params, m_arr.params)
    assert m_df.feature_names == ["z"]
    m_tl = WeibullAFT.fit_tvc_timeline(
        np.array(tl_i), np.array(tl_x), np.array(tl_Z), np.array(tl_c)
    )
    assert np.allclose(m_tl.params, m_arr.params, atol=1e-4)


def test_core_mle_unchanged():
    # The shared MLE path must be untouched: ordinary AFT and PH still fit and
    # give sensible parameters after the TVC mixin was added.
    Z, x, c = _plain_data(seed=6, n=300)
    aft = WeibullAFT.fit(x=x, Z=Z.reshape(-1, 1), c=c)
    ph = WeibullPH.fit(x=x, Z=Z.reshape(-1, 1), c=c)
    assert np.all(np.isfinite(aft.params)) and aft.params[0] > 0
    assert np.all(np.isfinite(ph.params))


def test_ph_has_no_custom_aft_likelihood():
    # AFT's fit_tvc must not leak onto PH/AH (they use the reshape mixin).
    assert aft_tvc_fit.AFTTVCFitMixin not in type(WeibullPH).__mro__
