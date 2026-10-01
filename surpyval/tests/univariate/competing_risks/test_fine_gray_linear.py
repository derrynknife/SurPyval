"""Fine-Gray in linear time (#517).

The subdistribution risk set at an event time ``t`` is every row with
``x_i >= t`` (weight 1) plus every competing failure before ``t`` (weight
``G(t-)/G(x_i-)``). ``_fit_cause`` used to build those weights as a dense
(events x N) matrix ``W`` and multiply by it in every likelihood, gradient
and Hessian evaluation: 1.7 s and 476 MiB at 1e4 rows, about 30 GB at 1e5.
It now takes the two parts as a suffix and a prefix cumulative sum over
the rows in time order. The censoring Kaplan-Meier the weights come from
(``censoring_survival``) summed the rows at and after each distinct time
in a loop, O(N x times), a minute at 1e5 rows; it is now a suffix sum.

The old implementation is kept below (``_old_fit_cause``,
``_old_censoring_survival``) and the new one must match it to the last
digits on tied and untied, censored, counted, multi-cause data.
"""

import time
import tracemalloc
import warnings
from typing import Any

import numpy as np
import pytest
from autograd import grad, hessian
from autograd import numpy as anp
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import norm

from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
    FineGray,
)
from surpyval.univariate.competing_risks.labels import label_mask
from surpyval.univariate.competing_risks.regression import fine_gray
from surpyval.univariate.competing_risks.regression.fine_gray import (
    FineGrayModel,
    _cumhaz_at_origin,
    _fit_cause,
)
from surpyval.univariate.regression._aliasing import (
    aliased_columns,
    constant_columns,
    expand,
)
from surpyval.univariate.regression._fit_skeleton import (
    runaway_coefficients,
    search_derivatives,
)
from surpyval.utils import validate_fine_gray_inputs
from surpyval.utils.ipcw import censoring_survival, step_left_limit
from surpyval.utils.linalg import safe_inv

# -- the implementation before #517 -----------------------------------------


def _old_censoring_survival(x, censored, n=None, ties="censoring_first"):
    x = np.asarray(x, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    if n is None:
        n = np.ones(x.size)
    times = np.unique(x)
    G = np.ones(times.size)
    surv = 1.0
    for i, t in enumerate(times):
        cens_here = n[(x == t) & censored].sum()
        if ties == "events_first":
            at_risk = n[x > t].sum() + cens_here
        else:
            at_risk = n[x >= t].sum()
        if at_risk > 0:
            surv *= 1.0 - cens_here / at_risk
        G[i] = surv
    return times, G


def _old_information_at_zero(Z, W, n, is_event):
    Wn = W * n
    S0 = Wn.sum(axis=1)
    d_over = n[is_event] / S0
    M = (Wn @ Z) / S0[:, None]
    a = d_over @ Wn
    return Z.T @ (a[:, None] * Z) - M.T @ (n[is_event][:, None] * M)


def _old_fit_cause(x, Z, e, c, n, cause, center=False):
    """``fine_gray._fit_cause`` before #517 (the dense weight matrix);
    the aliasing warning, which the new code raises the same way, is
    left out."""
    is_cause = label_mask(e, cause)
    is_event = (c == 0) & is_cause
    is_competing = (c == 0) & ~is_cause
    g_times, g_vals = _old_censoring_survival(x, c == 1, n)
    G_x = step_left_limit(g_times, g_vals, x, before=1.0)
    event_times = x[is_event]
    G_t = step_left_limit(g_times, g_vals, event_times, before=1.0)
    at_risk = x[None, :] >= event_times[:, None]
    already_competing = is_competing[None, :] & (
        x[None, :] < event_times[:, None]
    )
    W = at_risk.astype(float) + already_competing * (
        G_t[:, None] / G_x[None, :]
    )
    Z_raw = Z
    mean = (n @ Z) / n.sum()
    Z = Z - mean
    n_event = n[is_event]

    def partial_neg_ll(Zk):
        Zk_event = Zk[is_event]

        def neg_ll(beta):
            weighted_exp = n * anp.exp(anp.dot(Zk, beta))
            denom = anp.dot(W, weighted_exp)
            eta_event = anp.dot(Zk_event, beta)
            ll = anp.sum(n_event * eta_event) - anp.sum(
                n_event * anp.log(denom)
            )
            return -ll

        return neg_ll

    p = Z.shape[1]
    neg_ll = partial_neg_ll(Z)
    aliased = aliased_columns(
        _old_information_at_zero(Z, W, n, is_event),
        Z.shape[0],
        constant_columns(Z_raw),
        float(n_event.sum()) * (n @ Z**2) / n.sum(),
    )
    kept = np.setdiff1d(np.arange(p), aliased)
    if aliased.size:
        neg_ll = partial_neg_ll(Z[:, kept])
    beta0 = np.zeros(kept.size)
    if kept.size:
        res = minimize(neg_ll, beta0, jac=grad(neg_ll), method="BFGS")
    else:
        res = OptimizeResult(
            x=beta0, fun=float(neg_ll(beta0)), success=True, nit=0
        )
    beta = res.x
    derivatives = search_derivatives(neg_ll, beta)
    runaway = runaway_coefficients(
        neg_ll, beta, list(range(beta.size)), beta0, derivatives
    )
    runaway = [int(kept[k]) for k in runaway]
    H = hessian(neg_ll)(beta) if derivatives is None else derivatives[0]
    cov = safe_inv(H)
    var = np.diag(cov)
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.where(var > 0, var, np.nan))
        z_score = beta / se
    p_values = 2.0 * (1.0 - norm.cdf(np.abs(z_score)))
    if aliased.size:
        beta = expand(beta, kept, p)
        se = expand(se, kept, p)
        p_values = expand(p_values, kept, p)
        full = np.full((p, p), np.nan)
        full[np.ix_(kept, kept)] = cov
        cov = full
        beta = np.where(np.isnan(beta), 0.0, beta)
    denom = W @ (n * np.exp(Z @ beta))
    order = np.argsort(event_times, kind="mergesort")
    t_sorted = event_times[order]
    d_over_r = (n_event / denom)[order]
    uniq_t, inv = np.unique(t_sorted, return_inverse=True)
    dL = np.zeros(uniq_t.shape[0])
    np.add.at(dL, inv, d_over_r)
    baseline_cumhaz = np.cumsum(dL)
    if not center:
        baseline_cumhaz = _cumhaz_at_origin(beta, mean, Z_raw, baseline_cumhaz)
        mean = np.zeros_like(mean)
    if aliased.size:
        beta = np.array(beta, dtype=float)
        beta[aliased] = np.nan
    return {
        "cause": cause,
        "beta": beta,
        "center": mean,
        "se": se,
        "p_values": p_values,
        "cov": cov,
        "baseline_times": uniq_t,
        "baseline_cumhaz": baseline_cumhaz,
        "neg_ll": float(res.fun),
        "res": res,
        "runaway": runaway,
    }


# -- data --------------------------------------------------------------------


def _three_causes(N: int, seed: int) -> tuple:
    """Three causes and about 30% independent censoring."""
    rng = np.random.default_rng(seed)
    Z = np.column_stack(
        [rng.normal(size=N), rng.binomial(1, 0.4, N), rng.uniform(-1, 1, N)]
    )
    rates = [
        0.10 * np.exp(Z @ [0.5, -0.4, 0.3]),
        0.07 * np.exp(Z @ [-0.3, 0.6, 0.0]),
        0.05 * np.exp(Z @ [0.0, 0.2, -0.5]),
    ]
    T = np.column_stack([rng.exponential(1 / r) for r in rates])
    t = T.min(axis=1)
    cens = rng.exponential(1 / 0.095, N)
    x = np.minimum(t, cens)
    e = np.array(np.where(cens < t, None, T.argmin(axis=1) + 1), object)
    return x, Z, e, rng


def _cases() -> Any:
    for N, seed in [(150, 1), (600, 2)]:
        x, Z, e, rng = _three_causes(N, seed)
        n = rng.integers(1, 4, N).astype(float)
        yield f"untied-{N}", x, Z, e, None
        # Heavy ties among events, competing failures and censorings.
        yield f"tied-{N}", np.ceil(x), Z, e, None
        yield f"tied-counts-{N}", np.round(x, 1), Z, e, n
    # Two causes, a binary covariate, a censoring tied with every time.
    x, Z, e, rng = _three_causes(200, 4)
    e = np.where(e == 3, 2, e).astype(object)
    yield "two-causes", np.ceil(x / 2), Z[:, 1:2], e, None
    # An aliased (duplicated) column and a constant one (#476).
    x, Z, e, rng = _three_causes(300, 5)
    Z = np.column_stack([Z[:, 0], 2 * Z[:, 0], np.full(300, 3.0), Z[:, 1]])
    yield "aliased", np.round(x, 1), Z, e, None
    # A covariate far from 0 (#463): the baseline is moved to Z = 0.
    x, Z, e, rng = _three_causes(300, 6)
    yield "far-covariate", x, Z + [0.0, 0.0, 30.0], e, None
    # A level with no cause-1 events: no finite maximum (#392).
    x, Z, e, rng = _three_causes(300, 7)
    sep = (rng.random(300) < 0.3).astype(float)
    e = np.where((sep == 1) & (e == 1), 2, e).astype(object)
    yield "runaway", x, np.column_stack([Z[:, 0], sep]), e, None
    # The smallest data: one event and one competing failure.
    yield (
        "two-rows",
        np.array([1.0, 2.0]),
        np.array([[0.0], [1.0]]),
        np.array([2, 1], dtype=object),
        None,
    )


CASES = list(_cases())


def _assert_close(new: Any, old: Any, rtol: float = 1e-12) -> None:
    new, old = np.asarray(new, float), np.asarray(old, float)
    scale = np.nanmax(np.abs(old)) if np.any(np.isfinite(old)) else 1.0
    np.testing.assert_allclose(new, old, rtol=rtol, atol=1e-14 * scale)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("cause", [1, 2])
@pytest.mark.parametrize("center", [False, True])
def test_fit_matches_the_dense_weight_matrix(case, cause, center):
    _, x, Z, e, n = case
    args = validate_fine_gray_inputs(x, Z, e, None, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = _fit_cause(*args, cause, center)
        old = _old_fit_cause(*args, cause, center)
    np.testing.assert_array_equal(new["baseline_times"], old["baseline_times"])
    assert new["runaway"] == old["runaway"]
    np.testing.assert_array_equal(np.isnan(new["beta"]), np.isnan(old["beta"]))
    keys = ["beta", "se", "p_values", "cov", "center", "baseline_cumhaz"]
    rtol = 1e-12
    if new["runaway"]:
        # No finite maximum: BFGS stops where the rise along the runaway
        # direction drops below its tolerance, a point the last digits of
        # the likelihood move by 1e-8 (relatively), and the standard
        # errors there are meaningless (and warned about, #392).
        keys, rtol = ["beta", "center", "baseline_cumhaz"], 1e-7
    for key in keys:
        _assert_close(new[key], old[key], rtol)
    _assert_close(new["neg_ll"], old["neg_ll"], rtol)
    # The predictions.
    t = np.quantile(args[0], [0.1, 0.5, 0.9, 1.0])
    z = args[1][:4] if args[1].shape[0] >= 4 else args[1][:1]
    _assert_close(
        FineGrayModel(new).cif(t, z), FineGrayModel(old).cif(t, z), rtol
    )


def test_risk_set_sums_are_the_weight_matrix_products():
    # Every case and cause: the sums equal W @ v, for v one value per row
    # and several.
    rng = np.random.default_rng(0)
    for _, x, Z, e, n in CASES:
        x, Z, e, c, n = validate_fine_gray_inputs(x, Z, e, None, n)
        is_cause = label_mask(e, 1)
        is_event = (c == 0) & is_cause
        is_competing = (c == 0) & ~is_cause
        g_times, g_vals = censoring_survival(x, c == 1, n)
        G_x = step_left_limit(g_times, g_vals, x, before=1.0)
        sets = fine_gray._risk_sets(
            x, n, is_event, is_competing, G_x, g_times, g_vals
        )

        def weights(t):
            G_t = step_left_limit(g_times, g_vals, t, before=1.0)
            return (x[None, :] >= t[:, None]) + (
                is_competing[None, :] & (x[None, :] < t[:, None])
            ) * (G_t[:, None] / G_x[None, :])

        W = weights(sets.times)
        v = rng.uniform(0.1, 3.0, (x.size, 2))
        _assert_close(fine_gray._risk_set_sums(v[sets.order], sets), W @ v)
        _assert_close(
            fine_gray._risk_set_sums(v[sets.order, 0], sets), W @ v[:, 0]
        )
        # The old information took one row of W per event row.
        Zc = Z - Z.mean(axis=0)
        _assert_close(
            fine_gray._information_at_zero(
                Zc[sets.order], n[sets.order], sets
            ),
            _old_information_at_zero(Zc, weights(x[is_event]), n, is_event),
        )


def test_crph_fine_gray_matches_the_dense_weight_matrix():
    x, Z, e, _ = _three_causes(400, 8)
    x = np.round(x, 1)
    model = CompetingRisksProportionalHazards.fit(x, Z, e, model="Fine-Gray")
    args = validate_fine_gray_inputs(x, Z, e, None, None)
    for cause in (1, 2, 3):
        old = _old_fit_cause(*args, cause)
        _assert_close(model.betas[model.event_idx_map[cause]], old["beta"])
        _assert_close(model._fg_models[cause].se, old["se"])


@pytest.mark.parametrize("ties", ["censoring_first", "events_first"])
def test_censoring_survival_is_bit_identical(ties):
    rng = np.random.default_rng(1)
    for trial in range(300):
        N = int(rng.integers(0, 40))
        if trial % 2:
            x = rng.integers(0, 8, N).astype(float)
        else:
            x = rng.exponential(size=N)
        if trial % 7 == 0 and N:
            x[rng.integers(0, N)] = np.nan
        if trial % 11 == 0 and N:
            x[rng.integers(0, N)] = np.inf
        censored = rng.random(N) < 0.4
        n = None if trial % 3 == 0 else rng.integers(0, 5, N)
        new = censoring_survival(x, censored, n, ties)
        old = _old_censoring_survival(x, censored, n, ties)
        np.testing.assert_array_equal(new[0], old[0])
        np.testing.assert_array_equal(new[1], old[1])


def test_censoring_survival_is_not_quadratic():
    # 1e5 distinct times took about a minute (a sum over the rows at and
    # after each time); it is a suffix sum now.
    rng = np.random.default_rng(2)
    x = rng.exponential(size=100_000)
    censored = rng.random(x.size) < 0.3
    start = time.perf_counter()
    censoring_survival(x, censored)
    assert time.perf_counter() - start < 5.0


def test_fit_memory_is_linear_in_the_rows():
    # The dense weight matrix took 476 MiB at 1e4 rows (and 1.9 GB at
    # 2e4); the cumulative sums a few MiB.
    x, Z, e, _ = _three_causes(20_000, 3)
    tracemalloc.start()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            FineGray.fit(x, Z, e, event=1)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 64 * 2**20, peak / 2**20
