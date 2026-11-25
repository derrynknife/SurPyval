"""Dabrowska's bivariate survival estimator.

References:

- with no censoring, the empirical joint survival function, exactly;
- its margins are the Kaplan-Meier estimates, exactly;
- R's ``mhazard::npSurv2(estimator = "dabrowska")`` on three data sets
  (``scripts/reference/reference_r_mhazard_dabrowska.R``);
- a small hand-worked example;
- the influence function against finite differences of the weighted
  estimate, and its standard error against the bootstrap's;
- recovery of a Clayton copula's joint survival in simulation.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import surpyval as sp
from surpyval import multivariate as mv
from surpyval.multivariate import Dabrowska, MultivariateSurpyvalData
from surpyval.multivariate.nonparametric.dabrowska import _evaluate

REFERENCE = (
    Path(__file__).parents[1]
    / "reference"
    / "data"
    / "r_mhazard_dabrowska.json"
)


def _data(seed=3, N=60):
    rng = np.random.default_rng(seed)
    x = np.ceil(rng.weibull(1.3, (N, 2)) * 6)
    c = (rng.random((N, 2)) < 0.3).astype(int)
    n = rng.integers(1, 3, N)
    return x, c, n


Q = np.array([[1.5, 2.5], [3, 3], [5, 1], [0.5, 4], [7, 7], [2, 0.2], [4, 6]])


@pytest.mark.parametrize("seed", range(4))
def test_uncensored_is_the_empirical_joint_survival(seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 8, (30, 2)).astype(float)  # many ties
    model = Dabrowska.fit(x)
    q = rng.uniform(0, 9, (200, 2))
    empirical = np.array(
        [np.mean((x[:, 0] > a) & (x[:, 1] > b)) for a, b in q]
    )
    np.testing.assert_allclose(model.sf(q), empirical, atol=1e-14)
    ff = np.array([np.mean((x[:, 0] <= a) & (x[:, 1] <= b)) for a, b in q])
    np.testing.assert_allclose(model.ff(q), ff, atol=1e-14)


def test_margins_are_kaplan_meier():
    x, c, n = _data()
    model = Dabrowska.fit(x, c=c, n=n)
    s = np.linspace(0, 20, 81)
    zero = np.zeros_like(s)
    for d, pts in enumerate(
        [np.column_stack([s, zero]), np.column_stack([zero, s])]
    ):
        km = sp.KaplanMeier.fit(x[:, d], c=c[:, d], n=n).sf(s)
        np.testing.assert_allclose(model.sf(pts), km, atol=1e-14)
        np.testing.assert_allclose(model.margins[d].sf(s), km, atol=1e-14)


@pytest.mark.parametrize("name", ["continuous", "ties", "small"])
def test_against_r_mhazard(name):
    ref = json.loads(REFERENCE.read_text())[name]
    x = np.column_stack([ref["x1"], ref["x2"]])
    c = 1 - np.column_stack([ref["d1"], ref["d2"]])
    model = Dabrowska.fit(x, c=c)
    np.testing.assert_allclose(model.t1, ref["t1"], rtol=0, atol=0)
    np.testing.assert_allclose(model.t2, ref["t2"], rtol=0, atol=0)
    got = model.sf_grid(model.t1, model.t2)
    np.testing.assert_allclose(got, np.array(ref["sf"]), atol=1e-13)


def test_hand_worked():
    # Four pairs; the second's second lifetime is censored at 1.
    #   (1, 2), (2, 1+), (3, 4), (4, 3)
    # U = {1, 2, 3, 4}, V = {2, 3, 4}. S1(1.5) = 3/4; S2(2.5): at 2, 3 at
    # risk, 1 fails: 2/3. At (u, v) = (1, 2): Y = 3 at risk in both
    # (rows 1, 3, 4), a = 1 (row 1 fails at 1), b = 1 (row 1 fails at 2),
    # c = 1, m = 2: the factor 3 * 2 / (2 * 2) = 3/2. So S(1.5, 2.5) =
    # 3/4 * 2/3 * 3/2 = 3/4; the empirical answer would be 3/4 too (only
    # row 1 has failed in both by then).
    x = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    c = np.array([[0, 0], [0, 1], [0, 0], [0, 0]])
    model = Dabrowska.fit(x, c=c)
    assert model.sf([[1.5, 2.5]])[0] == pytest.approx(0.75, abs=1e-15)
    assert model.sf([[1.5, 0.0]])[0] == pytest.approx(0.75, abs=1e-15)
    assert model.sf([[0.0, 2.5]])[0] == pytest.approx(2 / 3, abs=1e-15)
    # At (2.5, 3.5): S1 = 1/2, S2 = 2/3 * 1/2 = 1/3; the factors at
    # (1, 2) 3/2, at (2, *) 1 (Y = 2 at (2, 2): rows 3, 4; a = 0), at
    # (1, 3): Y = 2 (rows 3, 4), a = 0 -> 1. So 1/2 * 1/3 * 3/2 = 1/4.
    assert model.sf([[2.5, 3.5]])[0] == pytest.approx(0.25, abs=1e-15)


def test_influence_is_the_derivative():
    x, c, n = _data()
    model = Dabrowska.fit(x, c=c, n=n)
    h = 1e-6
    for on in ("sf", "ff"):
        analytic = model.influence(Q, on)
        numeric = np.empty_like(analytic)
        for i in range(len(x)):
            w = n.astype(float)
            w[i] += h
            up = _evaluate(model._grid, model._grid.counts(w), Q)
            w[i] -= 2 * h
            down = _evaluate(model._grid, model._grid.counts(w), Q)

            def value(e):
                S, S1, S2 = e
                return S if on == "sf" else 1 - S1 - S2 + S

            numeric[i] = (value(up) - value(down)) / (2 * h)
        np.testing.assert_allclose(analytic, numeric, atol=1e-8)


def test_standard_error_agrees_with_the_bootstrap():
    x, c, n = _data()
    model = Dabrowska.fit(x, c=c, n=n)
    se = model.standard_errors(Q)
    rng = np.random.default_rng(2)
    total = int(n.sum())
    draws = np.array(
        [
            _evaluate(
                model._grid,
                model._grid.counts(
                    rng.multinomial(total, n / total).astype(float)
                ),
                Q,
            )[0]
            for _ in range(2000)
        ]
    )
    np.testing.assert_allclose(se, draws.std(axis=0), rtol=0.1)


def test_cb():
    x, c, n = _data()
    model = Dabrowska.fit(x, c=c, n=n)
    sf = model.sf(Q)
    for method in ("wald", "bootstrap"):
        cb = model.cb(Q, method=method, random_state=1)
        assert cb.shape == (len(Q), 2)
        assert np.all(cb[:, 0] <= sf) and np.all(sf <= cb[:, 1])
        assert np.all((cb >= 0) & (cb <= 1))
        lower = model.cb(Q, method=method, bound="lower", random_state=1)
        assert lower.shape == (len(Q),)
    one = model.cb([3.0, 3.0])
    assert one.shape == (2,)
    ff = model.cb(Q, on="ff")
    assert np.all(ff[:, 0] <= model.ff(Q)) and np.all(model.ff(Q) <= ff[:, 1])
    with pytest.raises(ValueError, match="likelihood"):
        model.cb(Q, method="lr")


def test_recovers_a_clayton_copula():
    # 1,000 pairs from a Clayton copula (Kendall's tau 0.5) with Weibull
    # margins, each series censored by its own exponential time (about
    # 25% and 30% censored). Seed and tolerances fixed before the run.
    margins = [
        sp.Weibull.from_params([10, 2]),
        sp.Weibull.from_params([5, 1.5]),
    ]
    truth = mv.Clayton.from_params([2.0], margins)
    T = truth.random(1000, random_state=11)
    rng = np.random.default_rng(12)
    C = rng.exponential([30.0, 14.0], (1000, 2))
    x = np.minimum(T, C)
    c = (T > C).astype(int)
    model = Dabrowska.fit(x, c=c)
    pts = np.array([[a, b] for a in (4.0, 8.0, 12.0) for b in (2.0, 4.0, 7.0)])
    err = model.sf(pts) - truth.sf(pts)
    se = model.standard_errors(pts)
    assert np.max(np.abs(err)) < 0.04
    assert np.all(np.abs(err) < 3.5 * se)
    # Dependence is seen: the estimate is far from the product of margins
    km = [m.sf(pts[:, d]) for d, m in enumerate(model.margins)]
    assert np.all(model.sf(pts) > km[0] * km[1])


def test_fit_from_multivariate_data_and_counts():
    x, c, n = _data()
    data = MultivariateSurpyvalData(x, c=c, n=n, names=("eye 1", "eye 2"))
    a = Dabrowska.fit(data)
    b = Dabrowska.fit(np.repeat(x, n, axis=0), c=np.repeat(c, n, axis=0))
    np.testing.assert_allclose(a.sf(Q), b.sf(Q), atol=1e-14)
    assert a.names == ("eye 1", "eye 2")
    with pytest.raises(ValueError, match="not both"):
        Dabrowska.fit(data, c=c)


def test_grid_and_points_agree():
    x, c, n = _data()
    model = Dabrowska.fit(x, c=c, n=n)
    a, b = np.array([0.5, 2.0, 4.0]), np.array([1.0, 3.0])
    grid = model.sf_grid(a, b)
    assert grid.shape == (3, 2)
    for i, s in enumerate(a):
        for j, t in enumerate(b):
            assert grid[i, j] == model.sf([[s, t]])[0]


def test_serialisation_round_trip():
    x, c, n = _data()
    model = Dabrowska.fit(x, c=c, n=n)
    again = sp.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_array_equal(again.sf(Q), model.sf(Q))


def test_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, c, n = _data()
    ax = Dabrowska.fit(x, c=c, n=n).plot()
    assert ax.get_title() == "Joint survival (Dabrowska)"
    Dabrowska.fit(x, c=c, n=n).plot(filled=False)
    plt.close("all")


def test_repr():
    x, c, n = _data()
    text = repr(Dabrowska.fit(x, c=c, n=n))
    assert text.startswith("Non-parametric joint survival of two lifetimes")
    assert "Dabrowska" in text


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            dict(x=np.array([[1.0, 2.0], [2.0, 3.0]]), c=[[0, -1], [0, 0]]),
            "left-censored",
        ),
        (
            dict(
                x=np.array([[1.0, 2.0], [2.0, 3.0]]),
                c=np.array([[0, 2], [0, 0]]),
                xl=np.array([[1.0, 1.0], [2.0, 3.0]]),
                xr=np.array([[1.0, 2.0], [2.0, 3.0]]),
            ),
            "interval-censored",
        ),
        (
            dict(
                x=np.array([[1.0, 2.0], [2.0, 3.0]]),
                t=np.array([[[0.5, np.inf], [-np.inf, np.inf]]] * 2),
            ),
            "truncated",
        ),
        (dict(x=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])), "3 series"),
    ],
)
def test_refusals_name_the_alternative(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Dabrowska.fit(MultivariateSurpyvalData(**kwargs))
    if set(kwargs) <= {"x", "c"}:
        with pytest.raises(ValueError, match=match):
            Dabrowska.fit(**kwargs)


def test_missing_query_is_nan():
    x, c, n = _data()
    model = Dabrowska.fit(x, c=c, n=n)
    out = model.sf([[np.nan, 1.0], [1.0, 1.0]])
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_fit_from_df():
    import pandas as pd

    x, c, n = _data()
    df = pd.DataFrame(
        {"a": x[:, 0], "b": x[:, 1], "ca": c[:, 0], "cb": c[:, 1], "n": n}
    )
    got = Dabrowska.fit_from_df(df, ["a", "b"], ["ca", "cb"], "n")
    np.testing.assert_array_equal(got.sf(Q), Dabrowska.fit(x, c=c, n=n).sf(Q))
