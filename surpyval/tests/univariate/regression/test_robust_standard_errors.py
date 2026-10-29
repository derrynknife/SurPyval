"""Robust (sandwich) standard errors of the parametric regressions:
``robust_covariance``, ``robust_standard_errors`` and
``summary(robust=True, cluster=...)``.

The unclustered values are checked against lifelines 0.30.3, which gives
the HC0 sandwich for its parametric regressions (``fit(..., robust=True)``)
and has no clustered form for them. The values below were computed with

.. code-block:: python

    from lifelines import WeibullAFTFitter, LogNormalAFTFitter
    from lifelines.datasets import load_rossi

    df = load_rossi()[["week", "arrest", "fin", "age", "prio"]]
    f = WeibullAFTFitter().fit(df, "week", "arrest", robust=True)
    f.standard_errors_

lifelines' AFT coefficients are SurPyval's with the sign changed (its
covariates multiply the scale, SurPyval's the time), so the standard
errors are the same numbers; its ``rho_`` (Weibull) and ``sigma_``
(LogNormal) intercepts are the log of SurPyval's ``beta`` and ``sigma``,
whose standard errors follow by the delta method.

The clustered values are checked against R's survival 3.8.12 (R 4.4.3),
whose ``survreg`` gives the same sandwich summed by cluster, on the Rossi
data in groups of four consecutive rows:

.. code-block:: r

    rossi$grp <- (seq_len(nrow(rossi)) - 1) %/% 4
    f <- survreg(Surv(week, arrest) ~ age + fin + prio + cluster(grp),
                 data = rossi, dist = "weibull")
    sqrt(diag(vcov(f)))  # (Intercept), age, fin, prio, Log(scale)

``survreg``'s coefficients are lifelines' (the negative of SurPyval's),
and its ``Log(scale)`` is ``-log(beta)`` (Weibull) or ``log(sigma)``
(LogNormal). They are also checked against a likelihood written out here
with numpy, its scores by central differences, summed by cluster.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval import LogNormalAFT, Weibull, WeibullAFT, WeibullPH
from surpyval.tests._helpers import rossi_with_censoring

# lifelines 0.30.3, robust=True, the parameters in lifelines' order:
# lambda_ (age, fin, prio, Intercept), then rho_ / sigma_ (Intercept).
LIFELINES_ROBUST_SE = {
    "WeibullAFT": [
        0.0161806503,
        0.138957929,
        0.0198604435,
        0.3804782952,
        0.0875566231,
    ],
    "LogNormalAFT": [
        0.0186603045,
        0.1708261651,
        0.0226099233,
        0.4706177133,
        0.0941829925,
    ],
}
# The same fits' model-based standard errors and their log-likelihood.
LIFELINES_MODEL_SE = {
    "WeibullAFT": [
        0.0153714917,
        0.137214824,
        0.0200859463,
        0.3580530151,
        0.0892487093,
    ],
    "LogNormalAFT": [
        0.015379434,
        0.1667577712,
        0.0265597905,
        0.3932682139,
        0.0766401188,
    ],
}
# R survival 3.8.12, survreg(... + cluster(grp)): (Intercept), age, fin,
# prio, Log(scale).
R_CLUSTERED_SE = {
    "WeibullAFT": [
        0.3890451396,
        0.01651700205,
        0.1453462213,
        0.01881216435,
        0.08541789624,
    ],
    "LogNormalAFT": [
        0.4861304107,
        0.01957669113,
        0.183505672,
        0.02107696516,
        0.08668892292,
    ],
}
LIFELINES_LL = {
    "WeibullAFT": -682.0412779772754,
    "LogNormalAFT": -687.4410348943401,
}


def _rossi(fitter):
    df = rossi_with_censoring()
    return fitter.fit_from_df(
        df, x_col="week", c_col="censored", Z_cols=["age", "fin", "prio"]
    )


@pytest.mark.parametrize("name", ["WeibullAFT", "LogNormalAFT"])
def test_unclustered_matches_lifelines(name):
    model = _rossi(getattr(sp, name))
    np.testing.assert_allclose(
        -model.neg_ll(), LIFELINES_LL[name], rtol=0, atol=1e-6
    )
    se = model.robust_standard_errors()
    model_se = model.standard_errors()
    for got, ref in (
        (se, LIFELINES_ROBUST_SE),
        (model_se, LIFELINES_MODEL_SE),
    ):
        coef_se = got[2:]
        np.testing.assert_allclose(coef_se, ref[name][:3], rtol=2e-4)
    # The location of the log time: lifelines' lambda_ intercept is
    # log(alpha) for the Weibull, mu itself for the LogNormal.
    scale = model.params[0] if name == "WeibullAFT" else 1.0
    np.testing.assert_allclose(
        se[0] / scale, LIFELINES_ROBUST_SE[name][3], rtol=2e-4
    )
    # rho_ is log(beta), sigma_ log(sigma): se(log p) = se(p) / p.
    np.testing.assert_allclose(
        se[1] / model.params[1], LIFELINES_ROBUST_SE[name][4], rtol=2e-4
    )


@pytest.mark.parametrize("name", ["WeibullAFT", "LogNormalAFT"])
def test_clustered_matches_survreg(name):
    model = _rossi(getattr(sp, name))
    groups = np.arange(432) // 4
    se = model.robust_standard_errors(cluster=groups)
    ref = R_CLUSTERED_SE[name]
    np.testing.assert_allclose(se[2:], ref[1:4], rtol=2e-4)
    scale = model.params[0] if name == "WeibullAFT" else 1.0
    np.testing.assert_allclose(se[0] / scale, ref[0], rtol=2e-4)
    np.testing.assert_allclose(se[1] / model.params[1], ref[4], rtol=2e-4)


def _weibull_aft_row_ll(theta, x, c, Z, tl):
    """One unit's Weibull AFT log-likelihood per row, written out:
    S(x | Z) = exp(-(e^{b'Z} x / alpha)^beta), right censoring and left
    truncation."""
    alpha, beta, b = theta[0], theta[1], theta[2:]
    phi = np.exp(Z @ b)
    u = phi * x / alpha
    log_df = np.log(beta / alpha * phi) + (beta - 1) * np.log(u) - u**beta
    log_sf = -(u**beta)
    ll = np.where(c == 0, log_df, log_sf)
    return ll + ((phi * tl / alpha) ** beta)


def _oracle_meat(theta, x, c, Z, tl, n, cluster):
    """The sum over clusters of the outer products of the clusters'
    scores, by central differences of the likelihood written out."""
    k = theta.size
    h = 6e-6 * np.maximum(np.abs(theta), 1e-2)
    scores = np.zeros((x.size, k))
    for j in range(k):
        e = np.zeros(k)
        e[j] = h[j]
        scores[:, j] = (
            _weibull_aft_row_ll(theta + e, x, c, Z, tl)
            - _weibull_aft_row_ll(theta - e, x, c, Z, tl)
        ) / (2 * h[j])
    labels = np.unique(cluster)
    sums = np.array(
        [(n[cluster == g, None] * scores[cluster == g]).sum(0) for g in labels]
    )
    return sums.T @ sums


def _close_on_the_se_scale(a, b, tol):
    """``a`` and ``b`` agree to ``tol`` relative to the standard errors:
    each entry's difference over the product of the two standard errors
    it pairs. The model-based covariance (the bread of the sandwich) of
    an AFT fit is a numerical Hessian, good to about 1e-3 in its small
    off-diagonal entries, so entries near 0 are compared on this scale."""
    d = np.sqrt(np.outer(np.diag(b), np.diag(b)))
    assert np.max(np.abs(a - b) / d) < tol, np.max(np.abs(a - b) / d)


def _clustered_data(seed=3):
    """Twenty batches of six items, a batch effect shared by its items
    (a gamma frailty on the time scale), right censoring at 25 and late
    entry for a third of the items."""
    rng = np.random.default_rng(seed)
    batch = np.repeat(np.arange(20), 6)
    frailty = rng.gamma(4.0, 0.25, 20)[batch]
    Z = np.c_[rng.normal(size=120), rng.binomial(1, 0.5, 120)]
    x = 12 * rng.weibull(1.8, 120) * np.exp(-0.4 * Z[:, 0]) / frailty
    tl = np.where(np.arange(120) % 3 == 0, 1.0, 0.0)
    keep = x > tl
    x, Z, tl, batch = x[keep], Z[keep], tl[keep], batch[keep]
    c = (x > 25).astype(int)
    x = np.minimum(x, 25)
    return x, c, Z, tl, batch


def test_clustered_matches_a_likelihood_written_out():
    x, c, Z, tl, batch = _clustered_data()
    model = WeibullAFT.fit(x, Z, c=c, tl=tl)
    ours = model.robust_covariance(cluster=batch)
    # The sandwich is the model-based covariance around the clusters'
    # scores: its meat, recovered, is the likelihood's written out here.
    bread = np.linalg.inv(model.covariance())
    oracle = _oracle_meat(
        np.asarray(model.params), x, c, Z, tl, np.ones(x.size), batch
    )
    np.testing.assert_allclose(bread @ ours @ bread, oracle, rtol=1e-6)
    # Clustering by batch changes the standard errors (the batch effect is
    # shared), and every item its own cluster is the unclustered form.
    assert not np.allclose(ours, model.robust_covariance())
    np.testing.assert_allclose(
        model.robust_covariance(cluster=np.arange(x.size)),
        model.robust_covariance(),
        rtol=1e-12,
    )


def test_counts_equal_repetition():
    # A row with n = 3 is three units: without clusters three groups,
    # with clusters three members of the row's cluster (principle 5).
    x, c, Z, tl, batch = _clustered_data()
    n = 1 + np.arange(x.size) % 3
    counted = WeibullAFT.fit(x, Z, c=c, n=n, tl=tl)
    rep = np.repeat(np.arange(x.size), n)
    expanded = WeibullAFT.fit(x[rep], Z[rep], c=c[rep], tl=tl[rep])
    np.testing.assert_allclose(counted.params, expanded.params, rtol=1e-6)
    # The two fits' model-based covariances are numerical Hessians of
    # differently summed likelihoods; the meat each sandwich wraps is what
    # the counts change, and it is the same.
    for cl_counted, cl_expanded in ((None, None), (batch, batch[rep])):
        meats = []
        for model, cl in ((counted, cl_counted), (expanded, cl_expanded)):
            bread = np.linalg.inv(model.covariance())
            meats.append(bread @ model.robust_covariance(cl) @ bread)
        np.testing.assert_allclose(meats[0], meats[1], rtol=1e-5)


def test_row_order_and_dropped_rows():
    # A label belongs to its row, whatever the order the rows are given
    # in, and a row dropped for a missing covariate takes its label away.
    x, c, Z, tl, batch = _clustered_data()
    ref = WeibullAFT.fit(x, Z, c=c, tl=tl).robust_covariance(batch)
    perm = np.random.default_rng(0).permutation(x.size)
    permuted = WeibullAFT.fit(x[perm], Z[perm], c=c[perm], tl=tl[perm])
    _close_on_the_se_scale(permuted.robust_covariance(batch[perm]), ref, 1e-4)
    Z_missing = np.r_[Z, [[np.nan, 1.0]]]
    with pytest.warns(UserWarning):
        dropped = WeibullAFT.fit(
            np.r_[x, 5.0], Z_missing, c=np.r_[c, 0], tl=np.r_[tl, 0.0]
        )
    _close_on_the_se_scale(
        dropped.robust_covariance(np.r_[batch, 99]), ref, 1e-4
    )


def test_invariant_to_the_parameterisation():
    # At the maximum the sandwich transforms by the jacobian of a change
    # of parameters: WeibullPH's coefficients are beta times WeibullAFT's.
    x, c, Z, tl, batch = _clustered_data()
    aft = WeibullAFT.fit(x, Z, c=c, tl=tl)
    ph = WeibullPH.fit(x, Z, c=c, tl=tl)
    beta, b = aft.params[1], aft.params[2:]
    np.testing.assert_allclose(ph.params[2:], beta * b, rtol=1e-4)
    J = np.eye(4)
    J[2:, 1] = b
    J[2, 2] = J[3, 3] = beta
    _close_on_the_se_scale(
        ph.robust_covariance(batch),
        J @ aft.robust_covariance(batch) @ J.T,
        1e-3,
    )
    # A fit on centred covariates (center=True) reports the baseline at
    # the means, so only its coefficients compare.
    centred = WeibullAFT.fit(x, Z, c=c, tl=tl, center=True)
    np.testing.assert_allclose(
        centred.robust_standard_errors(batch)[2:],
        aft.robust_standard_errors(batch)[2:],
        rtol=1e-4,
    )


def test_fixed_and_interval_censored():
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(80, 1))
    t = 10 * rng.weibull(2.0, 80) * np.exp(-0.3 * Z[:, 0])
    lo, hi = np.floor(t), np.floor(t) + 1
    x = np.c_[lo, hi]
    c = np.full(80, 2)
    model = WeibullAFT.fit(x, Z, c=c, fixed={"beta": 2.0})
    cov = model.robust_covariance()
    assert np.all(cov[1] == 0) and np.all(cov[:, 1] == 0)
    assert np.all(np.isfinite(cov)) and cov[0, 0] > 0 and cov[2, 2] > 0
    np.testing.assert_allclose(
        model.summary(robust=True)["se(coef)"].to_numpy()[[0, 2]],
        np.sqrt(np.diag(cov))[[0, 2]],
    )
    assert np.isnan(model.summary(robust=True)["se(coef)"].iloc[1])


def test_summary_robust_table():
    model = _rossi(WeibullAFT)
    df = rossi_with_censoring()
    groups = df.index.to_numpy() // 4
    table = model.summary(robust=True, cluster=groups)
    se = model.robust_standard_errors(groups)
    np.testing.assert_allclose(table["se(coef)"].to_numpy(), se)
    coef = table.loc["coefficients"]
    np.testing.assert_allclose(coef["z"], model.params[2:] / se[2:])
    # The baseline's positive scale has its interval on the log scale, as
    # param_cb gives it.
    lo, hi = table.loc[
        ("baseline", "alpha"), ["coef lower 95%", "coef upper 95%"]
    ]
    a = model.params[0]
    np.testing.assert_allclose(
        [lo, hi],
        a * np.exp(np.array([-1, 1]) * 1.959964 * se[0] / a),
        rtol=1e-6,
    )
    # The model-based table is unchanged.
    np.testing.assert_allclose(
        model.summary()["se(coef)"].to_numpy(), model.standard_errors()
    )


def test_accelerated_life_model():
    np.random.seed(0)
    Z = np.repeat([1.0, 2.0, 3.0], 30)
    x = Weibull.random(90, 10, 2) * Z**-1.0
    model = sp.AcceleratedLife(Weibull, sp.life_models.Power).fit(x, Z)
    se = model.robust_standard_errors()
    assert se[model.parameter_names.index("alpha")] == 0
    assert np.all(np.isfinite(se))
    table = model.summary(robust=True)
    assert np.all(np.isfinite(table["se(coef)"]))


def test_refusals():
    x, c, Z, tl, batch = _clustered_data()
    model = WeibullAFT.fit(x, Z, c=c, tl=tl)
    with pytest.raises(ValueError, match="one label per row given"):
        model.robust_covariance(batch[:-1])
    with pytest.raises(ValueError, match="missing label"):
        model.robust_covariance(np.where(batch == 0, np.nan, batch))
    with pytest.raises(ValueError, match="only with robust=True"):
        model.summary(cluster=batch)
    restored = type(model).from_dict(model.to_dict())
    with pytest.raises(ValueError, match="need the data"):
        restored.robust_covariance()
    i = np.array([1, 1, 2, 2, 3, 4, 5, 6])
    xl = np.array([0, 4, 0, 3, 0, 0, 0, 0.0])
    xr = np.array([4, 9, 3, 7, 6, 8, 5, 10.0])
    ce = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    Ze = np.array([0, 1, 0, 1, 1, 0, 0, 1.0]).reshape(-1, 1)
    tvc = WeibullAFT.fit_tvc(i, xl, xr, ce, Ze)
    with pytest.raises(ValueError, match="fit_tvc"):
        tvc.robust_covariance()


def test_lognormal_with_counts_and_left_censoring():
    # Every censoring kind goes through the per-row likelihood: its total
    # is the fit's log-likelihood.
    from surpyval.univariate.regression._robust import row_log_likelihood

    rng = np.random.default_rng(11)
    Z = rng.normal(size=(60, 1))
    t = np.exp(2 + 0.5 * Z[:, 0] + 0.6 * rng.normal(size=60))
    c = np.where(t > 15, 1, np.where(t < 3, -1, 0))
    x = np.clip(t, 3, 15)
    n = rng.integers(1, 3, 60)
    model = LogNormalAFT.fit(x, Z, c=c, n=n, tl=0.5)
    ll = row_log_likelihood(model.model, model.data, *model.params)
    np.testing.assert_allclose(model.data.n @ ll, -model.neg_ll(), rtol=1e-12)
    assert np.all(np.isfinite(model.robust_standard_errors()))


def _sandwich_by_hand(row_ll, theta):
    """HC0 by numdifftools: ``H^-1 (sum_i s_i s_i') H^-1`` with ``H`` the
    Hessian of the summed log-likelihood and ``s_i`` each row's score."""
    import numdifftools as nd  # type: ignore[import-untyped]

    H = nd.Hessian(lambda th: np.sum(row_ll(th)))(theta)
    S = nd.Jacobian(row_ll)(theta)
    bread = np.linalg.inv(-H)
    return bread @ (S.T @ S) @ bread


def _small():
    np.random.seed(7)
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(120, 2))
    x = sp.Weibull.random(120, 10, 2) * np.exp(-0.5 * Z[:, 0])
    c = (rng.uniform(size=120) < 0.2).astype(int)
    return x, Z, c


def test_spline_baseline_sandwich_by_hand():
    """audit R-N: a Royston-Parmar fit's robust covariance is the HC0
    sandwich of its rows' log-likelihoods (the family's log density and
    log survival at each row), by numdifftools."""
    x, Z, c = _small()
    model = sp.RoystonParmarPH.fit(x, Z, c=c)
    family = model.model

    def row_ll(th):
        return np.where(
            c == 0, family.log_df(x, Z, *th), family.log_sf(x, Z, *th)
        )

    theta = np.asarray(model.params, dtype=float)
    assert np.sum(row_ll(theta)) == pytest.approx(-model.neg_ll(), rel=1e-12)
    np.testing.assert_allclose(
        model.robust_covariance(), _sandwich_by_hand(row_ll, theta), rtol=1e-3
    )


def test_distributional_sandwich_by_hand():
    """audit R-N: a Weibull distributional fit's robust covariance is the
    HC0 sandwich of the Weibull log density and log survival at each
    row's own ``alpha`` and ``beta`` (log links on both), by
    numdifftools."""
    x, Z, c = _small()
    model = sp.WeibullDistributional.fit(x, Z, c=c)
    assert model.parameter_names == [
        "alpha",
        "beta",
        "alpha_coef_0",
        "alpha_coef_1",
        "beta_coef_0",
        "beta_coef_1",
    ]

    def row_ll(th):
        alpha = th[0] * np.exp(Z @ th[2:4])
        beta = th[1] * np.exp(Z @ th[4:6])
        return np.where(
            c == 0,
            Weibull.log_df(x, alpha, beta),
            Weibull.log_sf(x, alpha, beta),
        )

    theta = np.asarray(model.params, dtype=float)
    assert np.sum(row_ll(theta)) == pytest.approx(-model.neg_ll(), rel=1e-12)
    np.testing.assert_allclose(
        model.robust_covariance(), _sandwich_by_hand(row_ll, theta), rtol=1e-3
    )


def _left_censored_data():
    """test_lognormal_with_counts_and_left_censoring's data: exact rows,
    left-censored below 3, right-censored above 15, counts 1 or 2."""
    rng = np.random.default_rng(11)
    Z = rng.normal(size=(60, 1))
    t = np.exp(2 + 0.5 * Z[:, 0] + 0.6 * rng.normal(size=60))
    c = np.where(t > 15, 1, np.where(t < 3, -1, 0))
    x = np.clip(t, 3, 15)
    n = rng.integers(1, 3, 60)
    return x, Z, c, n


@pytest.mark.parametrize("fitter", [LogNormalAFT, WeibullPH])
@pytest.mark.parametrize(
    "window", [{"tl": 0.5, "tr": 40.0}, {"tr": 40.0}], ids=["both", "right"]
)
def test_row_likelihood_with_right_truncation(fitter, window):
    """audit R-W: the per-row likelihood behind the sandwich with a finite
    upper truncation, a window with both ends or right truncation alone,
    sums to the fit's log-likelihood, and the robust standard errors are
    finite. With right truncation alone the left-censored rows stay
    left-censored (the ``mask_l`` term runs); with a lower end too they
    become intervals from it, so the test asserts which it is."""
    from surpyval.univariate.regression._robust import row_log_likelihood

    x, Z, c, n = _left_censored_data()
    assert np.sum(c == -1) > 0
    model = fitter.fit(x, Z, c=c, n=n, **window)
    assert model.maximum == "verified"
    data = model.data
    assert data.truncated_mask.any()
    assert np.all(np.isfinite(data.x_tr))
    if "tl" in window:
        assert not data.mask_l.any() and data.mask_i.any()
        assert np.all(np.isfinite(data.x_tl))
    else:
        assert data.mask_l.any()
        assert not np.isfinite(data.x_tl).any()
    ll = row_log_likelihood(model.model, data, *model.params)
    np.testing.assert_allclose(data.n @ ll, -model.neg_ll(), rtol=1e-12)
    assert np.all(np.isfinite(model.robust_standard_errors()))


def test_truncated_lognormal_sandwich_by_hand():
    """audit R-W: the robust covariance of a LogNormal AFT fit truncated
    to (0.5, 40) with left and right censoring and counts is the HC0
    sandwich of a likelihood written out here (counts as frequency
    weights), by numdifftools. Under the window a left-censored row is
    ``F(x) - F(0.5)`` and a right-censored one ``F(40) - F(x)``, each over
    ``F(40) - F(0.5)``; AFT multiplies time by ``exp(b z)``."""
    import numdifftools as nd

    x, Z, c, n = _left_censored_data()
    lo, hi = 0.5, 40.0
    model = LogNormalAFT.fit(x, Z, c=c, n=n, tl=lo, tr=hi)
    assert model.parameter_names == ["mu", "sigma", "coef_0"]

    def row_ll(th):
        mu, sigma, b = th[0], th[1], th[2:]
        phi = np.exp(Z @ b)
        F = sp.LogNormal.ff
        exact = np.log(phi) + sp.LogNormal.log_df(x * phi, mu, sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = np.select(
                [c == 0, c == 1, c == -1],
                [
                    exact,
                    np.log(F(hi * phi, mu, sigma) - F(x * phi, mu, sigma)),
                    np.log(F(x * phi, mu, sigma) - F(lo * phi, mu, sigma)),
                ],
            )
        window = F(hi * phi, mu, sigma) - F(lo * phi, mu, sigma)
        return ll - np.log(window)

    theta = np.asarray(model.params, dtype=float)
    assert n @ row_ll(theta) == pytest.approx(-model.neg_ll(), rel=1e-12)
    H = nd.Hessian(lambda th: n @ row_ll(th))(theta)
    S = nd.Jacobian(row_ll)(theta)
    bread = np.linalg.inv(-H)
    by_hand = bread @ (S.T @ (n[:, None] * S)) @ bread
    np.testing.assert_allclose(model.robust_covariance(), by_hand, rtol=1e-3)
