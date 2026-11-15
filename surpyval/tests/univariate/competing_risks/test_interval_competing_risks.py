"""The non-parametric estimate of interval-censored competing risks
(``IntervalCompetingRisks``) and its Gray's test (``interval_gray_test``).
"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

import surpyval as sp
from surpyval import (
    CompetingRisks,
    IntervalCompetingRisks,
    gray_test,
    interval_gray_test,
)
from surpyval.serialisation import from_dict as load_dict


def _exact_data(seed=1, N=200):
    rng = np.random.default_rng(seed)
    ta = rng.exponential(10, N)
    tb = rng.exponential(15, N)
    tc = rng.uniform(0, 20, N)
    x = np.round(np.minimum(np.minimum(ta, tb), tc), 1)
    first = np.where(ta < tb, "a", "b")
    e = np.where(tc < np.minimum(ta, tb), None, first).astype(object)
    return x, e


def _inspected(
    seed=2, N=150, step=2.0, end=20.0, group_effect=0.0, jitter=False
):
    rng = np.random.default_rng(seed)
    group = rng.binomial(1, 0.5, N)
    ta = rng.exponential(1 / (0.08 * np.exp(group_effect * group)))
    tb = rng.exponential(1 / 0.05, N)
    t = np.minimum(ta, tb)
    cause = np.where(ta < tb, "a", "b")
    # Each unit's inspections every ``step``, from its own offset with
    # ``jitter`` (so the windows overlap).
    offset = rng.uniform(0, step, N).round(1) if jitter else 0.0
    lo = np.maximum(np.floor((t - offset) / step) * step + offset, 0.0)
    lo = np.where(t < offset, 0.0, lo)
    seen = lo < end
    x = np.column_stack(
        [np.where(seen, lo, end), np.where(seen, lo + step, np.inf)]
    )
    e = np.where(seen, cause, None).astype(object)
    return x, e, group


# -- the estimate ------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exact_data_is_aalen_johansen(seed):
    x, e = _exact_data(seed)
    model = IntervalCompetingRisks.fit(x, e)
    aj = CompetingRisks.fit(x, e, how="Kaplan-Meier")
    q = np.linspace(0, 21, 85)
    for cause in ("a", "b"):
        np.testing.assert_allclose(
            model.cif(q, cause), aj.cif(q, cause), atol=1e-8
        )
    np.testing.assert_allclose(model.sf(q), aj.sf(q), atol=1e-8)
    assert model.converged


def test_exact_data_with_counts_and_ties():
    x = [1, 1, 2, 2, 2, 3, 4, 4, 5]
    e = ["a", "b", "a", None, "b", "a", None, "b", "a"]
    n = [2, 1, 3, 2, 1, 1, 2, 1, 1]
    model = IntervalCompetingRisks.fit(x, e, n=n)
    aj = CompetingRisks.fit(x, e, n=n, how="Kaplan-Meier")
    q = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 6])
    for cause in ("a", "b"):
        np.testing.assert_allclose(
            model.cif(q, cause), aj.cif(q, cause), atol=1e-10
        )


def test_one_cause_is_turnbull():
    x, e, _ = _inspected(seed=5)
    e1 = np.where(pd.isna(e), None, "fail").astype(object)
    model = IntervalCompetingRisks.fit(x, e1)
    c = np.where(np.isinf(x[:, 1]), 1, 2)
    xt = np.where(c[:, None] == 1, x[:, [0, 0]], x)
    tb = sp.Turnbull.fit(x=xt, c=c, turnbull_estimator="Kaplan-Meier")
    q = np.arange(0, 22, 2.0)
    np.testing.assert_allclose(model.ff(q), tb.ff(q), atol=1e-7)


def _direct_npmle(x, cause, K):
    """The NPMLE by a general constrained optimiser over every elementary
    piece and cause -- no support reduction, no EM -- for comparison.
    The pieces are the open gaps between the distinct ends, a point at
    each exact time, and the gap after the last end."""
    xl, xr = x[:, 0], x[:, 1]
    ends = np.unique(
        np.concatenate([xl[np.isfinite(xl)], xr[np.isfinite(xr)]])
    )
    points = np.unique(xl[xl == xr])
    gaps_lo = np.concatenate([[-np.inf], ends])
    gaps_hi = np.concatenate([ends, [np.inf]])
    lower = np.concatenate([gaps_lo, points])
    upper = np.concatenate([gaps_hi, points])
    point = np.concatenate(
        [np.zeros(gaps_lo.size, bool), np.ones(points.size, bool)]
    )
    P = lower.size

    def inside(i):
        if xl[i] == xr[i]:
            return point & (lower == xl[i])
        gap = ~point & (lower >= xl[i]) & (upper <= xr[i])
        pt = point & (lower > xl[i]) & (lower <= xr[i])
        return gap | pt

    def covers(i):
        rows = np.zeros((K, P), bool)
        if cause[i] >= 0:
            rows[cause[i]] = inside(i)
        else:
            rows[:] = inside(i)
        return rows.ravel()

    A = np.array([covers(i) for i in range(len(xl))], float)
    # The gap after the last end is one shared mass, not one per cause.
    beyond = np.isinf(upper)
    keep = ~np.tile(beyond, K)
    A = np.column_stack([A[:, keep], np.isinf(xr).astype(float)])
    m = A.shape[1]

    def nll(p):
        return -np.sum(np.log(np.maximum(A @ p, 1e-300)))

    def grad(p):
        return -(A.T @ (1.0 / np.maximum(A @ p, 1e-300)))

    res = minimize(
        nll,
        np.full(m, 1.0 / m),
        jac=grad,
        method="SLSQP",
        bounds=[(0, 1)] * m,
        constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1.0}],
        options={"ftol": 1e-15, "maxiter": 5000},
    )
    finite_upper = upper[~beyond]

    def cif(k, q):
        mass = res.x[k * (P - 1) : (k + 1) * (P - 1)]
        return np.array([mass[finite_upper <= v].sum() for v in q])

    return -res.fun, cif


def test_matches_a_general_optimiser():
    # Overlapping windows, exact times, left and right censoring.
    x = np.array(
        [
            [0, 3],
            [1, 4],
            [2, 2],
            [2, 6],
            [-np.inf, 3],
            [3, 7],
            [5, 5],
            [4, 9],
            [6, np.inf],
            [8, np.inf],
            [1, 5],
            [7, 10],
        ],
        float,
    )
    e = np.array(
        ["a", "b", "a", "a", "b", "b", "a", "b", None, None, "a", "a"],
        dtype=object,
    )
    c = np.where(np.isinf(x[:, 1]), 1, np.where(np.isinf(x[:, 0]), -1, 2))
    c[x[:, 0] == x[:, 1]] = 0
    model = IntervalCompetingRisks.fit(x, e, c=c, tol=1e-12)
    cause = np.array([{"a": 0, "b": 1}.get(v, -1) for v in e])
    ll, cif = _direct_npmle(x, cause, 2)
    assert model.log_likelihood == pytest.approx(ll, abs=1e-6)
    assert model.log_likelihood >= ll - 1e-9
    # The CIFs agree at the ends of each cause's intervals (inside an
    # interval the data do not say where the mass is).
    for k, label in enumerate("ab"):
        q = model.innermost_intervals(label)["upper"]
        np.testing.assert_allclose(model.cif(q, label), cif(k, q), atol=1e-4)


def test_masked_causes():
    x = np.array([[0, 2], [0, 2], [2, 4], [2, 4], [4, 6], [6, np.inf]])
    e = np.array(["a", "b", None, "a", None, None], dtype=object)
    c = np.array([2, 2, 2, 2, 2, 1])
    with pytest.raises(ValueError, match="masked='include'"):
        IntervalCompetingRisks.fit(x, e, c=c)
    model = IntervalCompetingRisks.fit(x, e, c=c, masked="include")
    # Every unit failed by 6 except one; the masked rows split their mass
    # between the causes, and the incidences add to the failures.
    assert model.ff([6]) == pytest.approx([5 / 6], abs=1e-8)
    assert model.beyond == pytest.approx(1 / 6, abs=1e-8)
    cause_known = model.cif([2], "a") + model.cif([2], "b")
    assert cause_known == pytest.approx([2 / 6], abs=1e-8)


def test_masked_exact_failures_split_by_cause():
    # A masked exact failure at a time with only cause-a failures goes to
    # cause a; the likelihood decides.
    x = [1, 1, 2, 3, 3]
    e = ["a", None, "b", "a", None]
    c = [0, 0, 0, 0, 1]
    model = IntervalCompetingRisks.fit(x, e, c=c, masked="include")
    assert model.cif([1], "a") == pytest.approx([0.4], abs=1e-6)


def test_fit_paths_agree():
    x, e, _ = _inspected(seed=7, N=60)
    a = IntervalCompetingRisks.fit(x, e)
    b = IntervalCompetingRisks.fit(xl=x[:, 0], xr=x[:, 1], e=e)
    lower = np.where(np.isinf(x[:, 0]), np.nan, x[:, 0])
    upper = np.where(np.isinf(x[:, 1]), np.nan, x[:, 1])
    df = pd.DataFrame({"lo": lower, "hi": upper, "e": e})
    d = IntervalCompetingRisks.fit_from_df(
        df, None, "e", xl_col="lo", xr_col="hi"
    )
    data = sp.SurpyvalData(x=x, e=e)
    f = IntervalCompetingRisks.fit(data)
    q = np.arange(0, 24, 1.0)
    for m in (b, d, f):
        np.testing.assert_allclose(m.cif(q, "a"), a.cif(q, "a"))
        np.testing.assert_allclose(m.cif(q, "b"), a.cif(q, "b"))
    assert d.source_df is df


def test_cif_steps_at_interval_ends_and_query_shapes():
    model = IntervalCompetingRisks.fit(
        [[0, 2], [2, 4], [2, 4], [4, np.inf]], ["a", "a", "b", None]
    )
    np.testing.assert_allclose(
        model.cif([0, 1.9, 2, 3.9, 4], "a"), [0, 0, 0.25, 0.25, 0.5]
    )
    assert np.ndim(model.cif(3, "a")) == 0
    assert model.cif([[1, 2], [3, 4]], "a").shape == (2, 2)
    assert np.isnan(model.cif([np.nan], "a")[0])
    iv = model.innermost_intervals("a")
    np.testing.assert_allclose(iv["lower"], [0, 2])
    np.testing.assert_allclose(iv["upper"], [2, 4])
    np.testing.assert_allclose(iv["mass"], [0.25, 0.25])
    assert model.beyond == pytest.approx(0.25)


def test_refusals():
    x = [[0, 2], [2, 4]]
    model = IntervalCompetingRisks.fit(x, ["a", "b"])
    with pytest.raises(ValueError, match="all-cause"):
        model.sf([1], event="a")
    with pytest.raises(ValueError, match="Name the cause"):
        model.cif([1], None)
    with pytest.raises(ValueError, match="masked must be"):
        IntervalCompetingRisks.fit(x, ["a", "b"], masked="yes")
    with pytest.raises(ValueError, match="at least one failure"):
        IntervalCompetingRisks.fit([1, 2], [None, None])
    with pytest.raises(ValueError, match="weights"):
        IntervalCompetingRisks.fit(x, ["a", "b"], w=[1, 1])
    with pytest.raises(ValueError, match="n_boot"):
        model.cb([1], "a", n_boot=1)


def test_bootstrap_bounds():
    x, e, _ = _inspected(seed=11, N=120)
    model = IntervalCompetingRisks.fit(x, e)
    q = np.array([4.0, 10.0, 18.0])
    cb = model.cb(q, "a", n_boot=200, random_state=3)
    est = model.cif(q, "a")
    assert cb.shape == (3, 2)
    assert np.all(cb[:, 0] <= est + 1e-12) and np.all(est <= cb[:, 1] + 1e-12)
    # Reused for the same seed; one-sided bounds inside the two-sided.
    np.testing.assert_array_equal(
        cb, model.cb(q, "a", n_boot=200, random_state=3)
    )
    lower = model.cb(q, "a", bound="lower", n_boot=200, random_state=3)
    assert np.all(lower >= cb[:, 0] - 1e-12)
    sf = model.cb(q, on="sf", n_boot=200, random_state=3)
    assert np.all(sf[:, 0] <= model.sf(q) + 1e-12)


def test_serialisation_round_trip():
    x, e, _ = _inspected(seed=13, N=50)
    e = e.copy()
    model = IntervalCompetingRisks.fit(x, e)
    d = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    back = load_dict(d)
    assert isinstance(back, IntervalCompetingRisks)
    q = np.arange(0, 24, 0.5)
    for cause in ("a", "b"):
        np.testing.assert_array_equal(back.cif(q, cause), model.cif(q, cause))
    np.testing.assert_allclose(
        back.cb([6.0], "a", n_boot=50, random_state=1),
        model.cb([6.0], "a", n_boot=50, random_state=1),
    )
    assert "Causes: ['a', 'b']" in repr(model)


def test_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, e, _ = _inspected(seed=17, N=40)
    fig, ax = plt.subplots()
    IntervalCompetingRisks.fit(x, e).plot(ax=ax, n_boot=30)
    assert len(ax.get_lines()) == 2
    plt.close(fig)


def test_competing_risks_refusal_names_the_new_estimate():
    with pytest.raises(ValueError, match="IntervalCompetingRisks"):
        CompetingRisks.fit([[0, 2], [1, 3]], ["a", "b"], c=[2, 2])


# -- Gray's test -------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.0, 1.0])
def test_gray_exact_data_is_grays_test(rho):
    x, e = _exact_data(4, 160)
    group = np.random.default_rng(4).binomial(1, 0.5, x.size)
    ref = gray_test(x, e, group, event="a", rho=rho)
    res = interval_gray_test(x, e, group, event="a", rho=rho)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert res.p_value == pytest.approx(ref.p_value, rel=1e-12)
    assert res.df == ref.df and np.isinf(res.df_denominator)
    assert res.relative_increase == 0.0


def test_gray_finds_a_difference_and_combines():
    x, e, group = _inspected(seed=21, N=300, group_effect=1.0, jitter=True)
    res = interval_gray_test(x, e, group, event="a", random_state=2)
    assert res.p_value < 0.01
    assert res.relative_increase > 0
    assert np.isfinite(res.df_denominator)
    again = interval_gray_test(x, e, group, event="a", random_state=2)
    assert again.statistic == res.statistic
    d = json.loads(json.dumps(res.to_dict(), allow_nan=False))
    assert d["groups"] == [0, 1] and d["cause"] == "a"
    assert "Imputations" in repr(res)


def test_gray_with_masked_and_three_groups():
    x, e, _ = _inspected(seed=23, N=150)
    group = np.arange(150) % 3
    e = e.copy()
    e[::10] = np.where(np.isinf(x[::10, 1]), None, None)
    c = np.where(np.isinf(x[:, 1]), 1, 2)
    with pytest.raises(ValueError, match="masked"):
        interval_gray_test(x, e, group, event="a", c=c)
    res = interval_gray_test(
        x,
        e,
        group,
        event="a",
        c=c,
        masked="include",
        n_imputations=5,
        random_state=1,
    )
    assert res.df == 2 and 0 <= res.p_value <= 1


def test_gray_refusals():
    x, e, group = _inspected(seed=25, N=40)
    with pytest.raises(ValueError, match="No failures of cause"):
        interval_gray_test(x, e, group, event="z")
    with pytest.raises(ValueError, match="two groups"):
        interval_gray_test(x, e, np.zeros(40), event="a")
    with pytest.raises(ValueError, match="integer"):
        interval_gray_test(x, e, group, event="a", n=np.full(40, 1.5))
    with pytest.raises(ValueError, match="n_imputations"):
        interval_gray_test(x, e, group, event="a", n_imputations=1)
    with pytest.raises(ValueError, match="interval_gray_test"):
        gray_test(
            np.where(np.isinf(x[:, 1]), x[:, 0], x[:, 1]),
            e,
            group,
            event="a",
            c=np.where(np.isinf(x[:, 1]), 1, 2),
        )
