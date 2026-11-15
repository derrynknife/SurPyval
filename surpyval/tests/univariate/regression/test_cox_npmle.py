"""``CoxPH``'s full-likelihood route (``cox_npmle``): the proportional
hazards model with an NPMLE baseline per cause, for any censoring and
truncation.

The checks:

1. no covariates and no truncation: Hudgens, Satten and Longini's NPMLE
   of the cumulative incidences (``CompetingRisks``' NPMLE), CIFs to
   1e-6 and the same log-likelihood;
2. one cause: ``IntervalCoxPH`` (coefficients and baseline), with left
   truncation too;
3. one cause, no covariates: ``Turnbull``, with left and right
   truncation;
4. exact data: close to the cause-specific partial-likelihood Cox fit
   (``CoxPH`` per cause), within a tenth of a standard error;
5. an independent general-purpose optimiser over the full likelihood on a
   small mixed dataset gives the same maximum to 1e-6;

with the routing (what goes where, and that the partial-likelihood route
is unchanged), the derivatives, right truncation and serialisation.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize

import surpyval as sp
from surpyval import CoxPH, IntervalCoxPH, Turnbull
from surpyval.univariate.regression import CoxNPMLEModel
from surpyval.univariate.regression.proportional_hazards import (
    _cs_npmle as engine,
)
from surpyval.univariate.regression.proportional_hazards.cox_npmle import (
    NPMLE,
)


def _mixed(rng, N, K, rc=True):
    """``N`` rows of ``K`` causes: exact (to 0.1), interval (whole
    units), left censored and, with ``rc``, right censored."""
    t = rng.weibull(1.5, N) * 5
    cause = rng.integers(0, K, N)
    kind = rng.integers(0, 4 if rc else 3, N)
    xl = np.where(
        kind == 0,
        np.round(t, 1),
        np.where(
            kind == 1,
            np.floor(t),
            np.where(kind == 2, -np.inf, np.round(0.6 * t, 1)),
        ),
    )
    xr = np.where(
        kind == 0,
        np.round(t, 1),
        np.where(
            kind == 1,
            np.floor(t) + 1,
            np.where(kind == 2, np.ceil(t), np.inf),
        ),
    )
    cause = np.where(kind == 3, -1, cause)
    c = np.where(
        cause < 0, 1, np.where(xl == xr, 0, np.where(np.isinf(xl), -1, 2))
    )
    x = np.column_stack([np.where(c == -1, xr, xl), np.where(c == 1, xl, xr)])
    labels = np.array(
        [None if k < 0 else "abc"[k] for k in cause], dtype=object
    )
    return x, c, labels


# ---------------------------------------------------------------------------
# Check 1: no covariates, no truncation = IntervalCompetingRisks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K, seed", [(2, 5), (3, 6), (2, 7)])
def test_no_covariates_equals_the_competing_risks_npmle(K, seed):
    rng = np.random.default_rng(seed)
    x, c, e = _mixed(rng, 60, K)
    model = CoxPH.fit(x, np.zeros((60, 0)), c=c, e=e)
    npmle = sp.CompetingRisks.fit(x, e, c=c)
    assert npmle.estimator == "NPMLE"
    assert model.estimator == NPMLE
    assert model.log_likelihood == pytest.approx(
        npmle.log_likelihood, abs=1e-6
    )
    finite = x[np.isfinite(x)]
    q = np.unique(np.concatenate([finite, finite + 0.05, [0.0, 100.0]]))
    for k in "abc"[:K]:
        np.testing.assert_allclose(
            model.cif(q, np.zeros(0), k), npmle.cif(q, k), atol=1e-6
        )
    np.testing.assert_allclose(
        model.ff(q, np.zeros(0)), npmle.ff(q), atol=1e-6
    )


# ---------------------------------------------------------------------------
# Check 2: one cause = IntervalCoxPH
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("truncate", [False, True])
def test_one_cause_equals_interval_cox(truncate):
    rng = np.random.default_rng(11)
    x, c, _ = _mixed(rng, 80, 1)
    Z = rng.normal(size=(80, 2))
    tl = None
    if truncate:
        start = np.where(c == -1, x[:, 1], x[:, 0])
        tl = np.where((rng.random(80) < 0.4) & (start > 0.3), 0.3, 0.0)
    model = CoxPH.fit(x, Z, c=c, tl=tl)
    ref = IntervalCoxPH.fit(x, Z, c=c, tl=tl)
    np.testing.assert_allclose(model.beta, ref.beta, atol=1e-6)
    assert model.log_likelihood == pytest.approx(ref.log_likelihood, abs=1e-8)
    q = np.linspace(0, 12, 97)
    for row in ([0.0, 0.0], [1.0, -0.5]):
        np.testing.assert_allclose(model.sf(q, row), ref.sf(q, row), atol=1e-6)
    np.testing.assert_allclose(
        model.standard_errors(), ref.standard_errors(), rtol=1e-4
    )


# ---------------------------------------------------------------------------
# Check 3: one cause, no covariates = Turnbull
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["none", "tl", "tr", "both"])
def test_one_cause_without_covariates_equals_turnbull(which):
    """Truncated samples as they arise: each unit has a window (entry at
    0.3 or none; the end of its study at 7 or 10), and only the units
    whose event fell inside it are in the data. The windows overlap, so
    the NPMLE exists."""
    rng = np.random.default_rng(3)
    N = 90
    x, c, _ = _mixed(rng, N, 1, rc=which in ("none", "tl"))
    start = np.where(c == -1, 0.0, x[:, 0])
    end = np.where(c == 1, x[:, 0], x[:, 1])
    tl = np.where(rng.random(N) < 0.5, 0.3, -np.inf)
    tr = np.where(rng.random(N) < 0.5, 7.0, 10.0)
    keep = np.ones(N, dtype=bool)
    kw = {}
    if which in ("tl", "both"):
        keep &= (start > tl) & ~((c == -1) & np.isfinite(tl))
        kw["tl"] = tl
    if which in ("tr", "both"):
        keep &= (end <= tr) & (c != 1)
        kw["tr"] = tr
    x, c = x[keep], c[keep]
    kw = {k: v[keep] for k, v in kw.items()}
    n = int(keep.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = CoxPH.fit(x, np.zeros((n, 0)), c=c, **kw)
        ref = Turnbull.fit(
            x,
            c=c,
            turnbull_estimator="Kaplan-Meier",
            tol=1e-14,
            max_iter=200000,
            **kw,
        )
    assert ref.npmle == "exists"
    q = np.unique(np.concatenate([ref.x, ref.x - 0.01]))
    q = q[q <= model.tau]
    np.testing.assert_allclose(model.sf(q, np.zeros(0)), ref.sf(q), atol=1e-6)


# ---------------------------------------------------------------------------
# Check 4: exact data = cause-specific Cox, as closely as the baselines
# allow
# ---------------------------------------------------------------------------
def test_exact_data_is_close_to_cause_specific_cox():
    """With exact, right-censored data the full likelihood treats each
    event time as a drop of a discrete survival curve (the grouped-data
    model of Prentice and Gloeckler 1978), where Cox's partial
    likelihood treats the baseline as a continuous hazard (Breslow).
    Both are consistent, and they differ by O(1/n): the tolerance is a
    tenth of each coefficient's standard error."""
    rng = np.random.default_rng(2026)
    N = 400
    Z = rng.normal(size=(N, 2))
    beta = np.array([[0.5, -0.3], [-0.4, 0.2]])
    t1 = rng.exponential(1 / (0.1 * np.exp(Z @ beta[0])))
    t2 = rng.exponential(1 / (0.08 * np.exp(Z @ beta[1])))
    cens = rng.uniform(0, 15, N)
    t = np.minimum(np.minimum(t1, t2), cens)
    e = np.where(t == cens, None, np.where(t1 < t2, "a", "b")).astype(object)
    # The full-likelihood route, called directly (exact data would take
    # the partial likelihood through CoxPH.fit).
    from surpyval.univariate.regression.proportional_hazards.cox_npmle import (
        fit_npmle,
        parse,
    )

    rows, causes, single = parse(t, Z, None, None, None, None, e, "refuse")
    full = fit_npmle(rows, causes, single)
    se = full.standard_errors().reshape(2, 2)
    for k, cause in enumerate(causes):
        c_k = np.where(e == cause, 0, 1)
        cox = CoxPH.fit(t, Z, c=c_k, tie_method="breslow")
        assert np.all(np.abs(full.betas[k] - cox.beta) < 0.1 * se[k])


# ---------------------------------------------------------------------------
# Check 5: an independent optimiser over the full likelihood
# ---------------------------------------------------------------------------
def _independent_loglik(model, x, c, e, Z, tl, tr):
    """The log-likelihood of the rows at ``model``'s parameters, written
    out row by row from the model's definition (each cause's probability
    on each atom, the window's probability), not from the engine."""
    left, right = model.left, model.right
    causes = list(model.causes)
    total = 0.0
    for i in range(len(c)):
        z = np.atleast_1d(Z[i]) - model.center
        r = np.exp(np.nan_to_num(model.betas) @ z)
        S = 1.0
        mass = np.zeros((len(causes), right.size))
        for j in range(right.size):
            u = r * model.d[:, j]
            lam = u.sum()
            if j == right.size - 1:
                drop = S
            else:
                drop = S * (1 - np.exp(-lam))
            if lam > 0:
                mass[:, j] = drop * u / lam
            S -= drop
        lo, hi = (x[i, 0], x[i, 1])
        if c[i] == -1:
            lo = -np.inf
        if c[i] == 1:
            hi = np.inf
        inside = (right > lo) & (right <= hi) if lo < hi else right == lo
        inside |= (left == lo) & (right == hi) & (lo == hi)
        if c[i] == 1:
            inside = (right > lo) & (right <= tr[i])
            P = mass[:, inside].sum()
        elif e[i] is None:
            P = mass[:, inside].sum()
        else:
            P = mass[causes.index(e[i]), inside].sum()
        window = (left >= tl[i]) & (right <= tr[i])
        Q = mass[:, window].sum()
        total += np.log(P) - np.log(Q)
    return total


def _small_mixed():
    """24 rows of two causes with every censoring kind, delayed entry on
    some rows and right truncation on others."""
    rng = np.random.default_rng(20261115)
    N = 24
    x, c, e = _mixed(rng, N, 2)
    Z = np.tile([0.0, 1.0], N // 2)
    start = np.where(c == -1, x[:, 1], x[:, 0])
    tl = np.where((np.arange(N) % 4 == 1) & (start > 0.4), 0.4, -np.inf)
    end = np.where(c == 1, x[:, 0] + 1.0, x[:, 1])
    tr = np.where(np.arange(N) % 3 == 2, np.maximum(end, 8.0), np.inf)
    return x, c, e, Z, tl, tr


def test_independent_optimiser_finds_the_same_maximum():
    x, c, e, Z, tl, tr = _small_mixed()
    assert set(c) == {-1, 0, 1, 2}
    model = CoxPH.fit(x, Z, c=c, e=e, tl=tl, tr=tr)
    assert model.maximum == "verified"
    ll_model = _independent_loglik(model, x, c, e, Z, tl, tr)
    assert ll_model == pytest.approx(model.log_likelihood, abs=1e-9)

    # Search over (betas, d >= 0) of the same support from a neutral
    # start, with the likelihood written out above.
    free = model.allowed.copy()
    L = free.shape[1] - 1
    on_last = np.flatnonzero(free[:, L])
    if on_last.size:
        free[on_last[0], L] = False
    K = len(model.causes)
    template = CoxNPMLEModel.from_dict(model.to_dict())

    def nll(theta):
        template.betas = theta[:K].reshape(K, 1)
        d = np.zeros_like(model.d)
        d[free] = theta[K:]
        if on_last.size:
            d[on_last[0], L] = 1.0
        template.d = d
        with np.errstate(all="ignore"):
            v = -_independent_loglik(template, x, c, e, Z, tl, tr)
        return v if np.isfinite(v) else 1e10

    n_free = int(free.sum())
    theta0 = np.concatenate([np.zeros(K), np.full(n_free, 0.05)])
    bounds = [(None, None)] * K + [(0.0, None)] * n_free
    best = None
    for _ in range(4):
        res = minimize(
            nll,
            theta0 if best is None else best.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": 1e-16,
                "gtol": 1e-11,
                "maxiter": 20000,
                "maxfun": 10**6,
            },
        )
        if best is not None and res.fun >= best.fun - 1e-13:
            best = res if res.fun < best.fun else best
            break
        best = res
    assert -best.fun == pytest.approx(model.log_likelihood, abs=1e-6)


# ---------------------------------------------------------------------------
# The engine's derivatives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "seed, truncate, rc",
    [(0, False, True), (1, True, True), (2, False, False)],
)
def test_derivatives_agree_with_finite_differences(seed, truncate, rc):
    rng = np.random.default_rng(seed)
    N, K = 30, 2
    x, c, e = _mixed(rng, N, K, rc=rc)
    if not rc:
        x[-2:] = 20.0
        c[-2:] = 0
        e[-2:] = ["a", "b"]
    xl = np.where(c == -1, -np.inf, x[:, 0])
    xr = np.where(c == 1, np.inf, x[:, 1])
    cause = np.array([-1 if v is None else "ab".index(v) for v in e])
    tl = np.where(truncate & (rng.random(N) < 0.3), 0.1, -np.inf)
    tl = np.where(np.isfinite(xl) & (xl > 0.1), tl, -np.inf)
    tr = np.where(truncate & (rng.random(N) < 0.3), 25.0, np.inf)
    s = engine.build(xl, xr, cause, np.ones(N), tl, tr, K)
    lik = engine.Likelihood(s, rng.normal(size=(N, 2)))
    theta = lik.start(rng.normal(size=(K, 2)) * 0.3)
    theta[lik.n_beta :] *= rng.uniform(0.5, 1.5, theta.size - lik.n_beta)
    der = lik.derivatives(theta)
    assert der.value == pytest.approx(lik.value(theta), abs=1e-10)
    h = 1e-6
    g = np.zeros(theta.size)
    H = np.zeros((theta.size, theta.size))
    for i in range(theta.size):
        step = np.zeros(theta.size)
        step[i] = h
        g[i] = (lik.value(theta + step) - lik.value(theta - step)) / (2 * h)
        H[i] = (
            lik.derivatives(theta + step).grad
            - lik.derivatives(theta - step).grad
        ) / (2 * h)
    scale_g = np.max(np.abs(g))
    scale_H = np.max(np.abs(H))
    assert np.max(np.abs(g - der.grad)) < 1e-7 * scale_g
    assert np.max(np.abs(H - der.hess)) < 1e-6 * scale_H


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def test_exact_right_censored_left_truncated_keep_partial_likelihood():
    rng = np.random.default_rng(1)
    x = rng.exponential(5, 50)
    c = (rng.random(50) < 0.3).astype(int)
    Z = rng.normal(size=(50, 1))
    tl = np.where(rng.random(50) < 0.3, x / 3, 0.0)
    model = CoxPH.fit(x, Z, c=c, tl=tl)
    assert type(model).__name__ == "SemiParametricRegressionModel"
    assert model.estimator == "partial likelihood"


def test_causes_with_exact_data_are_the_cause_specific_partial_likelihood():
    from surpyval.univariate.competing_risks import (
        CompetingRisksProportionalHazards,
    )

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Z = [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]]
    e = ["a", "b", "a", None, "b", "a", "a", None, "b", "a"]
    model = CoxPH.fit(x, Z, e=e)
    ref = CompetingRisksProportionalHazards.fit(x, Z, e)
    assert model.estimator == "partial likelihood"
    np.testing.assert_array_equal(model.betas, ref.betas)


@pytest.mark.parametrize(
    "kw",
    [
        {"c": [0, 0, 2, 0, 1, 0, 0, 1]},
        {"c": [0, 0, -1, 0, 1, 0, 0, 1]},
        {"tr": 20.0},
    ],
)
def test_interval_left_censored_right_truncated_take_full_likelihood(
    kw,
):
    x = np.array(
        [[1, 1], [2, 2], [2, 4], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]],
        dtype=float,
    )
    if kw.get("c", [0, 0, 0])[2] == -1:
        x[2] = [4, 4]
    Z = [0, 1, 0, 1, 0, 1, 0, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = CoxPH.fit(x, Z, **kw)
    assert isinstance(model, CoxNPMLEModel)
    assert model.estimator == NPMLE
    assert "NPMLE" in repr(model)


def test_full_likelihood_route_refuses_strata_and_weights():
    x = [[0, 2], [1, 3], [2, 5], [5, 5]]
    with pytest.raises(ValueError, match="strata"):
        CoxPH.fit(x, [0, 1, 0, 1], c=[2, 2, 2, 0], strata=[0, 0, 1, 1])
    with pytest.raises(ValueError, match="'w'"):
        CoxPH.fit(x, [0, 1, 0, 1], c=[2, 2, 2, 0], w=[1, 2, 1, 1])


def test_masked_causes_are_refused_by_default_and_summed_on_request():
    x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [6, 6], [7, 7], [1, 4]]
    c = [2, 2, 2, 2, 2, 0, 1, 2]
    e = ["a", "b", None, "a", "b", "a", None, "b"]
    Z = [0, 1, 0, 1, 0, 1, 0, 1]
    with pytest.raises(ValueError, match="masked"):
        CoxPH.fit(x, Z, c=c, e=e)
    model = CoxPH.fit(x, Z, c=c, e=e, masked="include")
    assert np.isfinite(model.log_likelihood)


# ---------------------------------------------------------------------------
# Right truncation: no curve past the window
# ---------------------------------------------------------------------------
def test_right_truncation_warns_and_gives_nan_past_the_window():
    rng = np.random.default_rng(4)
    t = rng.weibull(1.5, 40) * 4
    keep = t <= 6
    t = np.round(t[keep], 2)
    Z = rng.normal(size=t.size)
    with pytest.warns(UserWarning, match="cannot be told from these data"):
        model = CoxPH.fit(np.column_stack([t, t]), Z, c=0 * t, tr=6.0)
    assert model.tau == 6.0
    out = model.sf([1.0, 5.0, 6.5], [0.0])
    assert np.isfinite(out[:2]).all() and np.isnan(out[2])
    assert np.isnan(model.cif(7.0, [0.0], None)) if model.single else True


# ---------------------------------------------------------------------------
# The functions agree; saving and loading
# ---------------------------------------------------------------------------
def test_cifs_sum_to_ff_and_identities_hold():
    rng = np.random.default_rng(8)
    x, c, e = _mixed(rng, 50, 3)
    Z = rng.normal(size=(50, 1))
    model = CoxPH.fit(x, Z, c=c, e=e)
    q = np.linspace(0, 12, 50)
    total = sum(model.cif(q, [0.3], k) for k in model.causes)
    np.testing.assert_allclose(total, model.ff(q, [0.3]), atol=1e-12)
    np.testing.assert_allclose(
        model.Hf(q, [0.3]), -np.log(model.sf(q, [0.3])), atol=1e-12
    )
    assert np.all(np.diff(model.sf(q, [0.3])) <= 1e-15)


def test_round_trips_through_json_and_pickle():
    import pickle

    rng = np.random.default_rng(9)
    x, c, e = _mixed(rng, 40, 2)
    Z = rng.normal(size=(40, 2))
    model = CoxPH.fit(x, Z, c=c, e=e)
    for back in (
        sp.from_json(model.to_json()),
        pickle.loads(pickle.dumps(model)),
    ):
        q = np.linspace(0, 10, 21)
        for k in model.causes:
            np.testing.assert_array_equal(
                back.cif(q, [0.1, -0.2], k), model.cif(q, [0.1, -0.2], k)
            )
        np.testing.assert_array_equal(back.params, model.params)
        assert back.parameter_names == model.parameter_names


def test_bounds():
    rng = np.random.default_rng(10)
    x, c, e = _mixed(rng, 60, 2)
    Z = rng.normal(size=(60, 1))
    model = CoxPH.fit(x, Z, c=c, e=e)
    wald = model.param_cb("a: coef_0")
    lr = model.param_cb("a: coef_0", method="lr")
    assert wald[0] < model.params[0] < wald[1]
    assert lr[0] < model.params[0] < lr[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        band = model.cb(
            [2.0, 4.0], [0.0], "a", on="cif", n_boot=30, random_state=1
        )
    point = model.cif([2.0, 4.0], [0.0], "a")
    assert band.shape == (2, 2)
    assert np.all(band[:, 0] <= point + 1e-12)
    assert np.all(point <= band[:, 1] + 1e-12)
