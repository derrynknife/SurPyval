"""The Prentice-Cai and Lin-Ying bivariate survival estimators, and
Kendall's tau from a non-parametric joint survival estimate.

References:

- with no censoring, each estimator is the empirical joint survival
  function, exactly, and Kendall's tau is the sample's tau-a;
- independent brute-force implementations written from the papers here
  (Prentice and Cai 1992, the Volterra equation summed term by term with
  its divisions; Lin and Ying 1993, the counts by loops), on tied and
  untied censored data;
- under independence, every estimator agrees with the product of the
  Kaplan-Meier margins in a large sample;
- a small seeded simulation: bias near zero for each estimator under its
  assumptions, and Lin-Ying refused (and biased, if forced) when the
  censoring is not common;
- R's ``mhazard`` when its reference file is present
  (``scripts/reference/reference_r_mhazard_bivariate.R``).
"""

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import kendalltau

import surpyval as sp
from surpyval import multivariate as mv
from surpyval.multivariate import Dabrowska
from surpyval.multivariate.nonparametric.lin_ying import _LinYingGrid

ESTIMATORS = ("dabrowska", "prentice-cai", "lin-ying")
NEW = ("prentice-cai", "lin-ying")

REFERENCE = (
    Path(__file__).parents[1]
    / "reference"
    / "data"
    / "mhazard_bivariate_r.json"
)


def _fit(x, c=None, n=None, estimator="dabrowska"):
    return Dabrowska.fit(
        x,
        c=c,
        n=n,
        estimator=estimator,
        common_censoring=estimator == "lin-ying",
    )


def _independent(seed=3, N=60, ties=True):
    """Each lifetime censored by its own time."""
    rng = np.random.default_rng(seed)
    T = rng.weibull(1.3, (N, 2)) * 6
    T[:, 1] = 0.6 * T[:, 0] + 0.4 * T[:, 1]
    C = rng.exponential(10.0, (N, 2))
    x = np.minimum(T, C)
    if ties:
        x = np.ceil(x)
    c = (T > C).astype(int)
    n = rng.integers(1, 3, N)
    return x, c, n


def _common(seed=4, N=60, ties=True):
    """Both lifetimes of a unit censored by one time."""
    rng = np.random.default_rng(seed)
    T = rng.weibull(1.3, (N, 2)) * 6
    T[:, 1] = 0.6 * T[:, 0] + 0.4 * T[:, 1]
    C = rng.exponential(8.0, N)
    if ties:
        T, C = np.ceil(T), np.ceil(C)
    x = np.minimum(T, C[:, None])
    c = (T > C[:, None]).astype(int)
    n = rng.integers(1, 3, N)
    return x, c, n


Q = np.array([[1.5, 2.5], [3, 3], [5, 1], [0.5, 4], [7, 7], [2, 0.2], [4, 6]])


# ----------------------------------------------------------------------
# Brute force, from the papers


def _brute_prentice_cai(x, c, n, U, V):
    """Prentice and Cai (1992): S(s, t) = S1(s) S2(t) [1 + sum over failure
    pairs u <= s, v <= t of S(u-, v-) / (S1(u) S2(v)) (L11 - L10 L2 - L1
    L01 + L1 L2)], every count by a loop over the units, the divisions as
    the paper writes them (so only for data whose margins stay above 0)."""
    x1, x2 = x[:, 0], x[:, 1]
    d1, d2 = c[:, 0] == 0, c[:, 1] == 0

    def km(xs, ds, grid):
        out, s = [], 1.0
        for u in grid:
            at = sum(w for xi, w in zip(xs, n) if xi >= u)
            dd = sum(w for xi, di, w in zip(xs, ds, n) if xi == u and di)
            s *= 1 - dd / at
            out.append((s, dd / at))
        return out

    m1, m2 = km(x1, d1, U), km(x2, d2, V)
    S = {}

    def get(i, j):  # S at (U[i], V[j]); -1 is before the first time
        if i < 0 and j < 0:
            return 1.0
        if i < 0:
            return m2[j][0]
        if j < 0:
            return m1[i][0]
        return S[i, j]

    terms = {}
    for i, u in enumerate(U):
        for j, v in enumerate(V):
            Y = a = b = cc = 0.0
            for k in range(len(x)):
                if x1[k] >= u and x2[k] >= v:
                    Y += n[k]
                    if x1[k] == u and d1[k]:
                        a += n[k]
                    if x2[k] == v and d2[k]:
                        b += n[k]
                    if x1[k] == u and d1[k] and x2[k] == v and d2[k]:
                        cc += n[k]
            l1, l2 = m1[i][1], m2[j][1]
            if Y > 0:
                core = cc / Y - a / Y * l2 - l1 * b / Y + l1 * l2
            else:
                core = l1 * l2
            terms[i, j] = get(i - 1, j - 1) / (m1[i][0] * m2[j][0]) * core
            total = sum(
                terms[k, m] for k in range(i + 1) for m in range(j + 1)
            )
            S[i, j] = m1[i][0] * m2[j][0] * (1 + total)
    return np.array([[S[i, j] for j in range(len(V))] for i in range(len(U))])


def _brute_lin_ying(x, c, n, pts):
    """Lin and Ying (1993): H(s, t) / G(max(s, t)), G the Kaplan-Meier
    estimate of the follow-up time (each unit's later time, observed when
    either lifetime is censored)."""
    m = x.max(axis=1)
    e = (c == 1).any(axis=1)
    N = n.sum()
    out = []
    for s, t in pts:
        H = sum(w for xi, w in zip(x, n) if xi[0] > s and xi[1] > t) / N
        G = 1.0
        for u in np.unique(m[e]):
            if u > max(s, t):
                break
            at = sum(w for mi, w in zip(m, n) if mi >= u)
            dd = sum(w for mi, ei, w in zip(m, e, n) if mi == u and ei)
            G *= 1 - dd / at
        out.append(H / G if G > 0 else 0.0)
    return np.array(out)


def _brute_tau(model, a, b):
    """Kendall's tau from the masses of the estimate, pair by pair."""
    g = model._grid
    g1, g2 = g.U[g.U <= a], g.V[g.V <= b]
    p1 = np.concatenate([[-np.inf], g1])
    p2 = np.concatenate([[-np.inf], g2])
    S = model.sf_grid(p1, p2)
    pts, mass = [], []
    for i in range(len(g1)):
        for j in range(len(g2)):
            pts.append((g1[i], g2[j]))
            mass.append(S[i, j] - S[i + 1, j] - S[i, j + 1] + S[i + 1, j + 1])
    total = sum(mass)
    num = 0.0
    for (p, mp), (q, mq) in itertools.product(zip(pts, mass), repeat=2):
        num += mp * mq * np.sign(p[0] - q[0]) * np.sign(p[1] - q[1])
    count = total * model.n.sum()
    return num / total**2 * count / (count - 1)


# ----------------------------------------------------------------------
# Exact properties


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("seed", range(3))
def test_uncensored_is_the_empirical_joint_survival(estimator, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 8, (30, 2)).astype(float)  # many ties
    model = _fit(x, estimator=estimator)
    q = rng.uniform(0, 9, (200, 2))
    empirical = np.array(
        [np.mean((x[:, 0] > a) & (x[:, 1] > b)) for a, b in q]
    )
    np.testing.assert_allclose(model.sf(q), empirical, atol=1e-13)
    ff = np.array([np.mean((x[:, 0] <= a) & (x[:, 1] <= b)) for a, b in q])
    np.testing.assert_allclose(model.ff(q), ff, atol=1e-13)


@pytest.mark.parametrize("estimator", ("dabrowska", "prentice-cai"))
def test_margins_are_kaplan_meier(estimator):
    x, c, n = _independent()
    model = _fit(x, c, n, estimator)
    s = np.linspace(0, 20, 81)
    zero = np.zeros_like(s)
    for d, pts in enumerate(
        [np.column_stack([s, zero]), np.column_stack([zero, s])]
    ):
        km = sp.KaplanMeier.fit(x[:, d], c=c[:, d], n=n).sf(s)
        np.testing.assert_allclose(model.sf(pts), km, atol=1e-14)


def test_lin_ying_margin_is_its_weighted_empirical_one():
    x, c, n = _common()
    model = _fit(x, c, n, "lin-ying")
    s = np.linspace(0, 20, 41)
    pts = np.column_stack([s, np.full_like(s, -np.inf)])
    np.testing.assert_allclose(
        model.sf(pts), _brute_lin_ying(x, c, n, pts), atol=1e-14
    )
    # Up to the first censoring it is Kaplan-Meier's
    first = x[c.any(axis=1)].max(axis=1).min()
    early = s[s < first]
    km = sp.KaplanMeier.fit(x[:, 0], c=c[:, 0], n=n).sf(early)
    np.testing.assert_allclose(
        model.sf(np.column_stack([early, np.zeros_like(early)])),
        km,
        atol=1e-14,
    )


@pytest.mark.parametrize("ties", [True, False])
@pytest.mark.parametrize(
    "data", [_independent, _common], ids=["independent", "common"]
)
def test_prentice_cai_against_brute_force(data, ties):
    x, c, n = data(N=25, ties=ties)
    model = _fit(x, c, n, "prentice-cai")
    U, V = model._grid.U, model._grid.V
    # (the brute force divides by the margins: keep to where they are > 0)
    keepU = model.sf(np.column_stack([U, np.zeros_like(U)])) > 0
    keepV = model.sf(np.column_stack([np.zeros_like(V), V])) > 0
    U, V = U[keepU], V[keepV]
    want = _brute_prentice_cai(x, c, n, U, V)
    np.testing.assert_allclose(model.sf_grid(U, V), want, atol=1e-13)


@pytest.mark.parametrize("ties", [True, False])
def test_lin_ying_against_brute_force(ties):
    x, c, n = _common(N=40, ties=ties)
    model = _fit(x, c, n, "lin-ying")
    rng = np.random.default_rng(5)
    pts = np.vstack([rng.uniform(0, x.max(), (100, 2)), x[:20], x[:20] - 1e-9])
    np.testing.assert_allclose(
        model.sf(pts), _brute_lin_ying(x, c, n, pts), atol=1e-14
    )


def test_prentice_cai_where_a_margin_reaches_zero():
    """The recursion divides by nothing, so the last failure of a series
    (its Kaplan-Meier at 0) is no special case: with no censoring the
    estimate there is still the empirical one."""
    x = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    model = _fit(x, estimator="prentice-cai")
    np.testing.assert_allclose(
        model.sf_grid([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
        Dabrowska.fit(x).sf_grid([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
        atol=1e-15,
    )


def test_prentice_cai_and_dabrowska_are_close_but_not_equal():
    x, c, n = _independent(N=200, ties=False)
    a = _fit(x, c, n, "dabrowska").sf(Q)
    b = _fit(x, c, n, "prentice-cai").sf(Q)
    assert np.max(np.abs(a - b)) < 0.02
    assert np.max(np.abs(a - b)) > 0


# ----------------------------------------------------------------------
# Common censoring: declared, and checked


def test_lin_ying_needs_the_declaration():
    x, c, n = _common()
    with pytest.raises(ValueError, match="common_censoring=True"):
        Dabrowska.fit(x, c=c, n=n, estimator="lin-ying")


@pytest.mark.parametrize(
    "x, c, match",
    [
        ([[1.0, 2.0], [3.0, 4.0]], [[1, 1], [0, 0]], "different times"),
        ([[1.0, 2.0], [3.0, 4.0]], [[1, 0], [0, 0]], "before the other"),
    ],
)
def test_data_that_contradict_common_censoring_are_refused(x, c, match):
    x, c = np.array(x), np.array(c)
    for estimator in ESTIMATORS:
        with pytest.raises(ValueError, match=match) as info:
            Dabrowska.fit(x, c=c, estimator=estimator, common_censoring=True)
        assert "estimator='dabrowska'" in str(info.value)
    # Without the declaration Dabrowska's takes them
    Dabrowska.fit(x, c=c)


def test_independently_censored_data_are_refused_by_lin_ying():
    x, c, n = _independent()
    with pytest.raises(ValueError, match="common censoring"):
        Dabrowska.fit(x, c=c, n=n, estimator="lin-ying", common_censoring=True)


def test_unknown_estimator():
    with pytest.raises(ValueError, match="'estimator' must be"):
        Dabrowska.fit(Q, estimator="kaplan-meier")
    assert Dabrowska.fit(Q, estimator="volterra").estimator == "Prentice-Cai"


# ----------------------------------------------------------------------
# Bounds, saving, the rest of the interface


@pytest.mark.parametrize("estimator", NEW)
def test_bootstrap_bounds_and_the_wald_refusal(estimator):
    x, c, n = _common()
    model = _fit(x, c, n, estimator)
    sf = model.sf(Q)
    cb = model.cb(Q, n_boot=100, random_state=1)  # bootstrap by default
    same = model.cb(Q, method="bootstrap", n_boot=100, random_state=1)
    np.testing.assert_array_equal(cb, same)
    assert cb.shape == (len(Q), 2)
    assert np.all(cb[:, 0] <= sf + 1e-12) and np.all(sf <= cb[:, 1] + 1e-12)
    ff = model.cb(Q, on="ff", n_boot=50, random_state=1)
    assert ff.shape == (len(Q), 2)
    lower = model.cb(Q, bound="lower", n_boot=50, random_state=1)
    assert lower.shape == (len(Q),)
    for call in (
        lambda: model.cb(Q, method="wald"),
        lambda: model.standard_errors(Q),
        lambda: model.influence(Q),
    ):
        with pytest.raises(ValueError, match="method='bootstrap'"):
            call()


@pytest.mark.parametrize("estimator", NEW)
def test_bootstrap_bound_is_the_percentile_of_resampled_fits(estimator):
    x, c, n = _common()
    model = _fit(x, c, n, estimator)
    pts = np.array([[1.0, 1.5], [2.0, 0.5]])
    got = model.cb(pts, n_boot=30, random_state=7)
    rng = np.random.default_rng(7)
    total = int(n.sum())
    draws = []
    for _ in range(30):
        w = rng.multinomial(total, n / total)
        draws.append(
            _fit(
                np.repeat(x, w, axis=0),
                np.repeat(c, w, axis=0),
                None,
                estimator,
            ).sf(pts)
        )
    want = np.quantile(np.array(draws), [0.025, 0.975], axis=0).T
    np.testing.assert_allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("estimator", NEW)
def test_serialisation_round_trip(estimator):
    x, c, n = _common()
    model = _fit(x, c, n, estimator)
    again = sp.from_dict(json.loads(json.dumps(model.to_dict())))
    assert again.estimator == model.estimator
    np.testing.assert_array_equal(again.sf(Q), model.sf(Q))


@pytest.mark.parametrize("estimator", NEW)
def test_repr_plot_and_frame(estimator):
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, c, n = _common()
    model = _fit(x, c, n, estimator)
    assert model.estimator in repr(model)
    assert model.plot().get_title() == f"Joint survival ({model.estimator})"
    plt.close("all")
    df = pd.DataFrame(
        {"a": x[:, 0], "b": x[:, 1], "ca": c[:, 0], "cb": c[:, 1], "n": n}
    )
    got = Dabrowska.fit_from_df(
        df,
        ["a", "b"],
        ["ca", "cb"],
        "n",
        estimator=estimator,
        common_censoring=estimator == "lin-ying",
    )
    np.testing.assert_array_equal(got.sf(Q), model.sf(Q))


# ----------------------------------------------------------------------
# Kendall's tau


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("seed", range(3))
def test_uncensored_tau_is_the_sample_tau_a(estimator, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 8, (30, 2)).astype(float)  # ties and duplicates
    x[:, 1] = np.ceil((x[:, 0] + x[:, 1]) / 2)
    i, j = np.triu_indices(len(x), 1)
    sign = np.sign(x[i, 0] - x[j, 0]) * np.sign(x[i, 1] - x[j, 1])
    tau_a = sign.sum() / len(i)
    assert _fit(x, estimator=estimator).kendall_tau() == pytest.approx(
        tau_a, abs=1e-12
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_uncensored_untied_tau_is_scipys(estimator):
    rng = np.random.default_rng(8)
    z = rng.normal(size=(80, 2))
    x = np.exp(np.column_stack([z[:, 0], 0.7 * z[:, 0] + 0.7 * z[:, 1]]))
    want = kendalltau(x[:, 0], x[:, 1]).statistic
    got = _fit(x, estimator=estimator).kendall_tau()
    assert got == pytest.approx(want, abs=1e-12)


def test_uncensored_tau_in_a_region_is_the_tau_of_the_units_in_it():
    rng = np.random.default_rng(9)
    x = rng.exponential(1.0, (60, 2))
    x[:, 1] += x[:, 0]
    inside = (x[:, 0] <= 1.0) & (x[:, 1] <= 2.0)
    want = kendalltau(x[inside, 0], x[inside, 1]).statistic
    got = Dabrowska.fit(x).kendall_tau(1.0, 2.0)
    assert got == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_tau_against_brute_force(estimator):
    x, c, n = _common(N=25)
    model = _fit(x, c, n, estimator)
    a, b = model.t1[-1], model.t2[-1]
    assert model.kendall_tau() == pytest.approx(
        _brute_tau(model, a, b), abs=1e-12
    )


def test_tau_cb():
    x, c, n = _common(N=80)
    model = _fit(x, c, n, "dabrowska")
    tau = model.kendall_tau()
    lo, hi = model.kendall_tau_cb(n_boot=100, random_state=3)
    assert lo < tau < hi
    assert model.kendall_tau_cb(
        bound="lower", n_boot=100, random_state=3
    ) == pytest.approx(
        np.quantile(
            model._bootstrap(
                model._grid.upto(model.t1[-1], model.t2[-1]),
                lambda e: model._tau(
                    model._grid.upto(model.t1[-1], model.t2[-1]),
                    e,
                    model.t1[-1],
                    model.t2[-1],
                ),
                100,
                3,
            ),
            0.05,
        )
    )


def test_tau_with_no_failures_in_a_series_is_refused():
    x = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    c = np.array([[1, 0], [1, 0], [1, 0]])
    with pytest.raises(ValueError, match="no failures"):
        Dabrowska.fit(x, c=c).kendall_tau()


# ----------------------------------------------------------------------
# Large samples and a small simulation (seeds, sizes and tolerances fixed
# before the first run)


def test_independence_agrees_with_the_product_of_margins():
    # 3,000 independent pairs, one follow-up time per unit (so all three
    # estimators apply); the tolerance is four of Dabrowska's standard
    # errors at each point.
    rng = np.random.default_rng(765)
    N = 3000
    T = np.column_stack([rng.weibull(1.5, N) * 10, rng.weibull(2.0, N) * 5])
    C = rng.uniform(0, 25, N)
    x = np.minimum(T, C[:, None])
    c = (T > C[:, None]).astype(int)
    pts = np.array([[a, b] for a in (3.0, 8.0, 14.0) for b in (2.0, 4.0, 7.0)])
    km = [sp.KaplanMeier.fit(x[:, d], c=c[:, d]) for d in range(2)]
    product = km[0].sf(pts[:, 0]) * km[1].sf(pts[:, 1])
    se = Dabrowska.fit(x, c=c).standard_errors(pts)
    for estimator in ESTIMATORS:
        got = _fit(x, c, None, estimator).sf(pts)
        assert np.all(np.abs(got - product) < 4 * se), (estimator, got)
        tau = _fit(x, c, None, estimator).kendall_tau()
        assert abs(tau) < 0.03, (estimator, tau)


MARGINS = [
    sp.Weibull.from_params([10.0, 2.0]),
    sp.Weibull.from_params([20.0, 3.0]),
]
TRUE = mv.Clayton.from_params([2.0], MARGINS)
POINTS = np.array([[8.0, 18.0], [4.0, 10.0]])
# Kendall's tau of the Clayton pairs given both fail by (12, 25): the
# sample tau of 200,000 draws inside the rectangle
REGION = (12.0, 25.0)


def _region_tau(seed=12, N=200_000):
    big = TRUE.random(N, random_state=seed)
    inside = (big[:, 0] <= REGION[0]) & (big[:, 1] <= REGION[1])
    return kendalltau(big[inside, 0], big[inside, 1]).statistic


def _simulate(censoring, N, reps, seed):
    """Mean error (and its Monte Carlo standard error) of each estimator at
    ``POINTS``, and of Lin-Ying's formula forced on the data; and of each
    estimator's Kendall's tau in ``REGION``."""
    rng = np.random.default_rng(seed)
    errs = {e: [] for e in ESTIMATORS + ("lin-ying (forced)",)}
    taus: dict = {e: [] for e in ESTIMATORS}
    truth = TRUE.sf(POINTS)
    refused = 0
    for _ in range(reps):
        T = TRUE.random(N, random_state=rng)
        if censoring == "common":
            C = rng.uniform(0, 45, N)[:, None] * np.ones(2)
        else:
            C = rng.uniform(0, [33.0, 60.0], (N, 2))
        x, c = np.minimum(T, C), (T > C).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for e in ESTIMATORS:
                try:
                    m = _fit(x, c, None, e)
                except ValueError:
                    refused += e == "lin-ying"
                    continue
                errs[e].append(m.sf(POINTS) - truth)
                taus[e].append(m.kendall_tau(*REGION))
            grid = _LinYingGrid(x, c)
            est = grid.counts(np.ones(N))
            from surpyval.multivariate.nonparametric.dabrowska import (
                _evaluate,
            )

            errs["lin-ying (forced)"].append(
                _evaluate(grid, est, POINTS)[0] - truth
            )
    out = {}
    for e, v in list(errs.items()) + [
        (f"tau {k}", t) for k, t in taus.items()
    ]:
        if v:
            a = np.array(v)
            out[e] = (a.mean(axis=0), a.std(axis=0, ddof=1) / np.sqrt(len(a)))
    return out, refused


def test_bias_under_common_censoring():
    # 100 data sets of 200 pairs (the full study, 1,000 of 200 and 400,
    # is tests/calibration/test_bias_bivariate_estimators.py)
    out, refused = _simulate("common", 200, 100, 7650)
    assert refused == 0
    for e in ESTIMATORS:
        bias, mcse = out[e]
        assert np.all(np.abs(bias) < 4 * mcse), (e, bias, mcse)
    # (the truth's own Monte Carlo error is about 0.0015)
    tau = _region_tau()
    for e in ESTIMATORS:
        mean, mcse = out[f"tau {e}"]
        assert abs(mean - tau) < 4 * mcse + 0.005, (e, mean, tau, mcse)


def test_bias_under_independent_censoring():
    out, refused = _simulate("independent", 200, 100, 7651)
    assert refused == 100  # Lin-Ying refuses every data set
    tau = _region_tau()
    for e in ("dabrowska", "prentice-cai"):
        bias, mcse = out[e]
        assert np.all(np.abs(bias) < 4 * mcse), (e, bias, mcse)
        mean, mcse = out[f"tau {e}"]
        assert abs(mean - tau) < 4 * mcse + 0.005, (e, mean, tau, mcse)
    bias, mcse = out["lin-ying (forced)"]
    assert np.any(np.abs(bias) > 4 * mcse), (bias, mcse)


# ----------------------------------------------------------------------
# R mhazard


@pytest.mark.skipif(
    not REFERENCE.exists(),
    reason="the mhazard reference file has not been generated yet",
)
@pytest.mark.parametrize("estimator", NEW)
def test_against_r_mhazard(estimator):  # pragma: no cover - needs the file
    refs = json.loads(REFERENCE.read_text())
    for name, ref in refs.items():
        if estimator not in ref["sf"]:
            continue
        x = np.column_stack([ref["x1"], ref["x2"]])
        c = 1 - np.column_stack([ref["d1"], ref["d2"]])
        model = _fit(x, c, None, estimator)
        got = model.sf_grid(ref["t1"], ref["t2"])
        np.testing.assert_allclose(
            got, np.array(ref["sf"][estimator]), atol=1e-12, err_msg=name
        )
