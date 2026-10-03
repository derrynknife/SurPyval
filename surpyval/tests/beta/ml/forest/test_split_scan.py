"""The split searches give the same splits as the per-candidate
reference implementations they replaced (#193, #190, #518).

The deviance split finds the maximised log-likelihood of each child in
closed form (exponential) or by a profile likelihood (Weibull) on
observed / right-censored data, where the bounded optimisers it replaced
stopped at their tolerances: the same splits, the log-likelihoods to the
last digits. On any other data it runs the same optimiser on the node's
likelihood terms sliced to the child, bit for bit what it did on a new
subset.

``log_rank_split`` scores every threshold of a feature from cumulative
at-risk and death counts in one pass; on a small node it must reproduce,
bit for bit, the statistic :func:`log_rank` computes for each candidate
subset, and so choose exactly the same split, ties included. A large node
is scored in O(N log N) from the same sums taken in another order (#549):
the statistics to rounding, the undefined ones exactly.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.beta.ml.forest import deviance_split as dev
from surpyval.beta.ml.forest.log_rank_split import (
    _LogRankScan,
    log_rank,
    log_rank_split,
)
from surpyval.utils.surpyval_data import SurpyvalData


def reference_log_rank_split(
    data, Z, min_leaf_samples, min_leaf_failures, feature_indices_in
):
    """The per-candidate search the scan replaced: a new subset and a
    full log-rank per threshold (with #193's n-weighted failures)."""
    event_weight = data.n * (data.c != 1)
    best, best_u, best_v = -np.inf, -1, -np.inf
    for u in feature_indices_in:
        Z_u = Z[:, u]
        for v in np.unique(Z_u):
            mask = Z_u <= v
            if (
                mask.sum() < min_leaf_samples
                or (~mask).sum() < min_leaf_samples
                or event_weight[mask].sum() < min_leaf_failures
                or event_weight[~mask].sum() < min_leaf_failures
            ):
                continue
            value = log_rank(u, v, data, Z)
            if value > best:
                best, best_u, best_v = value, u, v
    return best_u, best_v


def _case(seed):
    rng = np.random.default_rng(seed)
    N = int(rng.integers(5, 120))
    Z = rng.uniform(0, 1, (N, int(rng.integers(1, 4))))
    if seed % 3 == 0:
        Z = np.round(Z * 4) / 4  # tied feature values
    if seed % 5 == 0:
        Z = np.column_stack([Z, Z[:, 0]])  # a repeated column: exact ties
    x = rng.exponential(10, N) * np.exp(Z[:, 0])
    if seed % 2 == 0:
        x = np.round(x) + 1  # tied times
    c = (rng.uniform(size=N) < 0.3).astype(int)
    n = rng.integers(1, 4, N) if seed % 7 == 0 else None
    t = None
    if seed % 4 == 1:  # delayed entry
        tl = rng.uniform(0, 0.8, N) * x
        tl[rng.uniform(size=N) < 0.5] = 0
        t = np.column_stack([tl, np.full(N, np.inf)])
    data = SurpyvalData(x, c, n, t, group_and_sort=False)
    return data, Z, int(rng.integers(1, 6)), int(rng.integers(0, 4))


@pytest.mark.parametrize("seed", range(60))
def test_scan_matches_per_candidate_search(seed):
    data, Z, mls, mlf = _case(seed)
    features = list(range(Z.shape[1]))
    new = log_rank_split(data, Z, mls, mlf, features)
    old = reference_log_rank_split(data, Z, mls, mlf, features)
    assert (int(new[0]), float(new[1])) == (int(old[0]), float(old[1]))


@pytest.mark.parametrize("seed", range(0, 60, 7))
@pytest.mark.parametrize("block", [None, 5])
def test_scan_statistics_bit_identical(seed, block, monkeypatch):
    import surpyval.beta.ml.forest.log_rank_split as module

    if block is not None:
        # Blocks of a few rows: the carried running totals
        monkeypatch.setattr(module, "_SCAN_BLOCK", block)
    data, Z, _, _ = _case(seed)
    Z_u = Z[:, 0]
    values = np.unique(Z_u)[:-1]  # both children non-empty
    if values.size == 0:
        pytest.skip("one feature value")
    order = np.argsort(Z_u, kind="stable")
    n_left = np.searchsorted(Z_u[order], values, side="right")
    scan = _LogRankScan(data).statistics(order, n_left)
    direct = np.array([log_rank(0, v, data, Z) for v in values])
    np.testing.assert_array_equal(
        scan, np.where(np.isnan(direct), -np.inf, direct)
    )


def _large_case(seed, N=600):
    # Big enough for the sorted scan; ties, counts and delayed entry as
    # in _case.
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (N, 2))
    if seed % 3 == 0:
        Z = np.round(Z * 8) / 8
    x = rng.exponential(10, N) * np.exp(Z[:, 0])
    if seed % 2 == 0:
        x = np.round(x) + 1
    c = (rng.uniform(size=N) < 0.3).astype(int)
    if seed % 5 == 0:
        # Every early row censored: left children wholly at risk at the
        # first deaths (some variances exactly zero)
        c[x < np.quantile(x, 0.3)] = 1
    n = rng.integers(1, 4, N) if seed % 7 == 0 else None
    t = None
    if seed % 4 == 1:
        tl = rng.uniform(0, 0.8, N) * x
        tl[rng.uniform(size=N) < 0.5] = 0
        t = np.column_stack([tl, np.full(N, np.inf)])
    return SurpyvalData(x, c, n, t, group_and_sort=False), Z


@pytest.mark.parametrize("seed", range(12))
def test_549_sorted_scan_matches_log_rank(seed):
    # The O(N log N) scan of a large node gives log_rank's statistic for
    # every threshold to rounding, and exactly the same undefined ones.
    import surpyval.beta.ml.forest.log_rank_split as module

    data, Z = _large_case(seed)
    Z_u = Z[:, 0]
    values = np.unique(Z_u)[:-1]
    order = np.argsort(Z_u, kind="stable")
    n_left = np.searchsorted(Z_u[order], values, side="right")
    scan = _LogRankScan(data)
    assert scan.sortable
    got = module._sorted_statistics(scan, order, n_left)
    direct = np.array([log_rank(0, v, data, Z) for v in values])
    defined = np.isfinite(direct)
    np.testing.assert_array_equal(np.isfinite(got), defined)
    scale = np.abs(direct[defined]).max()
    np.testing.assert_allclose(
        got[defined], direct[defined], rtol=0, atol=1e-11 * scale
    )


@pytest.mark.parametrize("seed", range(12))
def test_549_sorted_scan_same_split(seed):
    data, Z = _large_case(seed, N=300)
    new = log_rank_split(data, Z, 5, 2, [0, 1])
    old = reference_log_rank_split(data, Z, 5, 2, [0, 1])
    assert (int(new[0]), float(new[1])) == (int(old[0]), float(old[1]))


def test_549_large_node_is_not_scanned_densely(monkeypatch):
    # The dense scan holds a (rows x event times) matrix per threshold
    # block: quadratic in the node's size, 485 s for a 20-tree forest of
    # 10,000 rows. A large node is scored without it.
    import surpyval.beta.ml.forest.log_rank_split as module

    sizes = []
    dense = module._LogRankScan._left_counts

    def recording(self, order, n_left):
        sizes.append(int(n_left.max()) * int(self.keep.sum()))
        return dense(self, order, n_left)

    monkeypatch.setattr(module._LogRankScan, "_left_counts", recording)
    data, Z = _large_case(3, N=2000)
    log_rank_split(data, Z, 5, 2, [0, 1])
    assert max(sizes, default=0) <= module._DENSE_LIMIT


# ---------------------------------------------------------------------------
# The deviance split (#190, #518)


def reference_deviance_split(
    data, Z, min_leaf_samples, min_leaf_failures, features, model
):
    """The per-candidate search the closed forms replaced: a new
    SurpyvalData per child and a bounded optimiser per child."""
    event_weight = data.n * (data.c != 1)
    theta0 = dev._exp_theta0(data)
    if theta0 is None:
        return -1, -np.inf
    if model == "weibull":
        la = -theta0
        box = ((la - 15.0, la + 15.0), dev._LOG_BETA_BOUNDS)
        parent_ll, start = dev._wei_max_ll(data, box, np.array([la, 0.0]))

        def child_ll(child):
            return dev._wei_max_ll(child, box, start)[0]

    else:
        bounds = (theta0 - 15.0, theta0 + 15.0)
        parent_ll = dev._exp_max_ll(data, bounds)

        def child_ll(child):
            return dev._exp_max_ll(child, bounds)

    best, best_u, best_v = -np.inf, -1, -np.inf
    for u in features:
        for v in dev._candidate_values(Z[:, u]):
            mask = Z[:, u] <= v
            if (
                mask.sum() < min_leaf_samples
                or (~mask).sum() < min_leaf_samples
                or event_weight[mask].sum() < min_leaf_failures
                or event_weight[~mask].sum() < min_leaf_failures
            ):
                continue
            score = child_ll(data[mask]) + child_ll(data[~mask])
            if score > best:
                best, best_u, best_v = score, int(u), float(v)
    if best_u != -1 and best <= parent_ll + 1e-6:
        return -1, -np.inf
    return best_u, best_v


def _deviance_case(seed, kind="rc"):
    rng = np.random.default_rng(seed)
    N = int(rng.integers(15, 80))
    Z = rng.uniform(0, 1, (N, 2))
    if seed % 3 == 0:
        Z = np.round(Z * 5) / 5
    x = 10 * rng.weibull(rng.uniform(0.6, 3), N) * np.exp(-0.8 * Z[:, 0])
    c = (rng.uniform(size=N) < 0.3).astype(int)
    n = rng.integers(1, 4, N) if seed % 4 == 0 else None
    if kind == "interval":
        lo = np.floor(x)
        xx = np.column_stack([lo, lo + 1.0])
        c = np.where(rng.uniform(size=N) < 0.6, 2, c)
        xx[c != 2] = x[c != 2, None]
        return SurpyvalData(xx, c, n, group_and_sort=False), Z
    if kind == "truncated":
        t = np.column_stack([x * rng.uniform(0, 0.5, N), np.full(N, np.inf)])
        return SurpyvalData(x, c, n, t, group_and_sort=False), Z
    return SurpyvalData(x, c, n, group_and_sort=False), Z


@pytest.mark.parametrize("model", ["exponential", "weibull"])
@pytest.mark.parametrize("seed", range(8))
def test_closed_form_deviance_split_matches_the_optimiser(model, seed):
    data, Z = _deviance_case(seed)
    new = dev.deviance_split(data, Z, 5, 2, [0, 1], model=model)
    old = reference_deviance_split(data, Z, 5, 2, [0, 1], model)
    assert new == old


@pytest.mark.parametrize("model", ["exponential", "weibull"])
@pytest.mark.parametrize("seed", range(6))
def test_closed_form_is_the_maximum_likelihood(model, seed):
    # The node's maximised log-likelihood is the MLE's, to the last
    # digits, and never below the bounded optimiser's
    data, _ = _deviance_case(seed)
    theta0 = dev._exp_theta0(data)
    search = dev._ChildLikelihoods(data, model, theta0)
    assert search.closed
    dist = sp.Weibull if model == "weibull" else sp.Exponential
    mle = -dist.fit(data.x, data.c, data.n).neg_ll()
    assert search.parent_ll == pytest.approx(mle, rel=1e-9, abs=1e-9)
    if model == "weibull":
        la = -theta0
        box = ((la - 15.0, la + 15.0), dev._LOG_BETA_BOUNDS)
        optimiser = dev._wei_max_ll(data, box, np.array([la, 0.0]))[0]
    else:
        optimiser = dev._exp_max_ll(data, (theta0 - 15.0, theta0 + 15.0))
    assert search.parent_ll >= optimiser - 1e-12 * abs(optimiser)


@pytest.mark.parametrize("model", ["exponential", "weibull"])
@pytest.mark.parametrize("kind", ["interval", "truncated"])
def test_other_data_slices_the_node_terms_bit_for_bit(model, kind):
    # Off the closed forms the optimiser runs on the node's likelihood
    # terms sliced to the child: the very arrays a new SurpyvalData of
    # the child would give, so the very same numbers.
    data, Z = _deviance_case(11, kind)
    theta0 = dev._exp_theta0(data)
    search = dev._ChildLikelihoods(data, model, theta0)
    assert not search.closed
    masks = np.array([Z[:, 0] <= v for v in np.quantile(Z[:, 0], [0.3, 0.6])])
    rows = np.concatenate([masks, ~masks])
    got = search.child_lls(rows)
    if model == "weibull":
        want = [
            dev._wei_max_ll(data[m], search.box, search.start)[0] for m in rows
        ]
    else:
        want = [dev._exp_max_ll(data[m], search.bounds) for m in rows]
    np.testing.assert_array_equal(got, want)
    assert dev.deviance_split(
        data, Z, 5, 2, [0, 1], model=model
    ) == reference_deviance_split(data, Z, 5, 2, [0, 1], model)


@pytest.mark.parametrize("model", ["exponential", "weibull"])
def test_549_one_pass_per_node(model, monkeypatch):
    # The node's own maximum and every feature's children are found in
    # one pass, where the node and each feature took one of their own.
    data, Z = _deviance_case(3)
    Z = np.column_stack([Z, Z[:, ::-1]])
    calls = []
    closed = dev._ChildLikelihoods._closed_lls

    def spy(self, rows):
        calls.append(rows.shape[0])
        return closed(self, rows)

    monkeypatch.setattr(dev._ChildLikelihoods, "_closed_lls", spy)
    got = dev.deviance_split(data, Z, 5, 2, [0, 1, 2, 3], model=model)
    assert len(calls) == 1
    monkeypatch.undo()
    assert got == reference_deviance_split(data, Z, 5, 2, [0, 1, 2, 3], model)


@pytest.mark.parametrize("seed", range(6))
def test_549_weibull_children_from_their_own_rows(seed):
    # Each child's profile likelihood sums over its own rows only: the
    # same maximum as the bounded optimiser on that child, and as the
    # profile on a (children x rows) matrix gave, to the last digits.
    data, Z = _deviance_case(seed)
    theta0 = dev._exp_theta0(data)
    search = dev._ChildLikelihoods(data, "weibull", theta0)
    masks = np.array([Z[:, 0] <= v for v in np.quantile(Z[:, 0], [0.3, 0.6])])
    rows = np.concatenate([masks, ~masks])
    got, theta = search._weibull(rows)
    for j, mask in enumerate(rows):
        child = data[mask]
        mle = sp.Weibull.fit(child.x, child.c, child.n)
        np.testing.assert_allclose(np.exp(theta[j]), mle.params, rtol=1e-5)
        assert got[j] >= -mle.neg_ll() - 1e-9 * abs(mle.neg_ll())
