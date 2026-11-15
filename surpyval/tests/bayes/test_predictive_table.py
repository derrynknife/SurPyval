"""The tabulated posterior predictive (``bayes/_predictive_table.py``).

A posterior of more than ``TABULATE_ABOVE`` points has its predictive
cumulative hazard tabulated once and read by a Hermite cubic, checked
against the full sum over the points. These tests hold it to 1e-6
relative in ``sf``, ``ff`` and ``Hf`` (1e-5 in ``hf`` and ``df``, the
slope of the curve) against that full sum on representative posteriors
(a Weibull grid, an exponential of rate near 1e-9, offsets, a limited
failure population, zero inflation),
check that it is fast, that ``qf`` brackets the cases that used to fail,
and that draws still follow a seed.
"""

import pickle
import time
import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import prior
from surpyval.bayes import BayesianParametric
from surpyval.bayes import _predictive_table as pt

#: The accuracy promised against the full sum: sf, ff and Hf; hf and df
#: (the slope of the tabulated curve).
PROMISE = {"sf": 1e-6, "ff": 1e-6, "Hf": 1e-6, "hf": 1e-5, "df": 1e-5}
#: Seed of every set of draws below, fixed before any result was seen.
SEED = 5
N_DRAWS = 6000

X = np.r_[2100.0, 3400.0, 4050.0, np.linspace(1500, 6000, 37)]
C = np.r_[0, 0, 0, np.ones(37)]


def _draws(extra, n=N_DRAWS, seed=SEED):
    g = np.random.default_rng(seed)
    cols = [200 * g.lognormal(0, 0.15, n), 1.8 * g.lognormal(0, 0.15, n)]
    names = ["alpha", "beta"]
    for name, (lo, hi) in extra.items():
        cols.append(g.uniform(lo, hi, n))
        names.append(name)
    return BayesianParametric.from_draws(sp.Weibull, np.c_[tuple(cols)], names)


def _grid():
    return sp.Weibull.fit_bayes(
        X, C, priors={"beta": prior.around(2, between=(1.5, 3))}
    )


def _huge_scale():
    lam = np.random.default_rng(2).lognormal(np.log(1e-9), 0.5, 5000)
    return BayesianParametric.from_draws(sp.Exponential, lam[:, None])


CASES = {
    "weibull grid": _grid,
    "exponential, rate near 1e-9": _huge_scale,
    "offset": lambda: _draws({"gamma": (10.0, 30.0)}),
    "limited failure population": lambda: _draws({"lfp_p": (0.7, 0.9)}),
    "zero inflation": lambda: _draws({"f0": (0.01, 0.05)}),
    "all three": lambda: _draws(
        {"gamma": (10.0, 30.0), "lfp_p": (0.7, 0.9), "f0": (0.01, 0.05)}
    ),
}


@pytest.fixture(scope="module", params=list(CASES))
def model(request):
    return CASES[request.param]()


def _summed(model, fname, t):
    model.predictive_table = False
    try:
        return getattr(model, fname)(t)
    finally:
        model.predictive_table = True


def test_it_is_tabulated(model):
    table = model._table()
    assert table is not None and table.ok
    assert len(model.posterior.weights) > pt.TABULATE_ABOVE
    # The bulk of the distribution is read from the table (near offsets,
    # where each point's own start is a kink, some stretches are summed)
    top = float(model.ff(np.inf))
    lo, hi = model.qf(np.array([0.05, 0.95 * top]))
    where, _, _ = table.read(np.linspace(lo, hi, 2001))
    assert np.mean(where == 1) > 0.9


def test_within_1e6_of_the_full_sum(model):
    table = model._table()
    # Across the tabulated range, a decade or so past each end, and below
    # the support's edge
    v = np.linspace(table.v_lo - 2.0, table.v_hi + 2.0, 1201)
    t = np.r_[table._t(v), -1.0, 0.0]
    if model.offset:
        t = np.r_[t, 5.0, table.edge]
    for fname in ("sf", "ff", "Hf", "hf", "df"):
        got = getattr(model, fname)(t)
        want = _summed(model, fname, t)
        assert np.all(np.isfinite(got) == np.isfinite(want)), fname
        both = np.isfinite(want) & (want > 0)
        rel = np.abs(got[both] - want[both]) / want[both]
        assert rel.max() <= PROMISE[fname], (fname, rel.max())
        assert np.all(got[want == 0] == 0), fname


def test_a_proper_distribution(model):
    table = model._table()
    t = np.sort(table._t(np.linspace(table.v_lo, table.v_hi, 5001)))
    sf = model.sf(t)
    assert np.all(np.diff(sf) <= 0)
    assert np.all((sf >= 0) & (sf <= 1))
    assert np.array_equal(model.ff(t), 1 - sf) or np.allclose(
        model.ff(t) + sf, 1.0, rtol=0, atol=1e-15
    )
    assert np.all(model.hf(t) >= 0)
    # The density is the derivative of ff (here by its integral)
    lo, hi = t[1000], t[4000]
    tt = np.linspace(lo, hi, 20001)
    integral = np.trapezoid(model.df(tt), tt)
    assert integral == pytest.approx(model.ff(hi) - model.ff(lo), rel=1e-5)


def test_qf_inverts_ff(model):
    p = np.array([1e-6, 0.001, 0.04, 0.1, 0.5, 0.7, 0.75])
    top = float(model.ff(np.inf))
    p = p[(p > float(model.ff(0.0))) & (p < top - 1e-9)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q = model.qf(p)
    assert np.all(np.isfinite(q))
    assert model.ff(q) == pytest.approx(p, rel=1e-9)


def test_seeded_draws_repeat(model):
    a = model.random(500, random_state=11)
    b = model.random(500, random_state=11)
    assert np.array_equal(a, b)


def test_pickle_and_dict_give_the_same_curve(model):
    t = np.linspace(1.0, 400.0, 7)
    if model.dist is sp.Exponential:
        t = t * 1e7
    want = model.sf(t)
    assert "_ptable" not in pickle.loads(pickle.dumps(model)).__dict__
    assert np.array_equal(pickle.loads(pickle.dumps(model)).sf(t), want)
    back = sp.from_dict(model.to_dict())
    assert np.array_equal(back.sf(t), want)


def test_a_small_posterior_is_summed_exactly():
    fit = _draws({"gamma": (10.0, 30.0)}, n=400)
    assert fit._table() is None


def test_a_copy_with_other_points_is_not_given_the_table():
    fit = _grid()
    fit.sf(3000.0)
    sub = fit._sub(fit.posterior.points[:3000])
    assert sub._table() is not fit._table()


# -- qf on the cases that failed to bracket (RePyability's workaround) -----


def _one_point(**extras):
    names = ["alpha", "beta", *extras]
    row = [200.0, 1.8, *extras.values()]
    fit = BayesianParametric.from_draws(
        sp.Weibull, np.tile(row, (5, 1)), names
    )
    return fit, sp.Weibull.from_params([200.0, 1.8], **extras)


@pytest.mark.parametrize(
    "extras",
    [
        {"gamma": 20.0, "lfp_p": 0.8},
        {"gamma": 20.0, "lfp_p": 0.8, "f0": 0.05},
        {"f0": 0.05},
        {"gamma": 20.0},
        {"lfp_p": 0.8},
    ],
    ids=["offset+lfp", "all three", "zi", "offset", "lfp"],
)
def test_qf_brackets_offsets_and_shares(extras):
    fit, plain = _one_point(**extras)
    p = np.array([0.001, 0.04, 0.1, 0.5, 0.75, 0.79])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = fit.qf(p)
        one = np.array([fit.qf(float(v)) for v in p])
    want = plain.qf(p)
    assert got == pytest.approx(want, rel=1e-10)
    assert one == pytest.approx(want, rel=1e-10)


def test_qf_offset_lfp_at_004():
    """The case RePyability reported: p 0.04, offset 20, lfp_p 0.8."""
    fit, plain = _one_point(gamma=20.0, lfp_p=0.8)
    assert float(fit.qf(0.04)) == pytest.approx(
        float(plain.qf(0.04)), rel=1e-12
    )


# -- qf at 0 with points that each have their own offset -----------------


def _spread(n, zi=False, lfp=False):
    """Points whose offsets (and shares) differ, drawn from a seed fixed
    before any result was seen."""
    rng = np.random.default_rng(SEED)
    cols = [
        rng.uniform(150.0, 250.0, n),
        rng.uniform(1.5, 2.2, n),
        rng.uniform(10.0, 30.0, n),
    ]
    names = ["alpha", "beta", "gamma"]
    if lfp:
        cols.append(rng.uniform(0.7, 0.9, n))
        names.append("lfp_p")
    if zi:
        cols.append(rng.uniform(0.02, 0.08, n))
        names.append("f0")
    pts = np.column_stack(cols)
    return BayesianParametric.from_draws(sp.Weibull, pts, names), pts


P_LOW = np.array([0.0, 1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.5])


@pytest.mark.parametrize("n", [200, pt.TABULATE_ABOVE + 500])
@pytest.mark.parametrize("lfp", [False, True], ids=["", "lfp"])
def test_qf_at_0_is_the_smallest_offset(n, lfp):
    """qf(0) is where the predictive starts, the smallest point's offset,
    not the posterior median offset (which is above qf of small p)."""
    fit, pts = _spread(n, lfp=lfp)
    q = fit.qf(P_LOW)
    assert q[0] == pts[:, 2].min()
    assert q[0] < fit.gamma
    assert np.all(np.diff(q) >= 0)
    assert fit.ff(q[1:]) == pytest.approx(P_LOW[1:], rel=1e-6)
    assert float(fit.qf(0.0)) == q[0]


@pytest.mark.parametrize("n", [200, pt.TABULATE_ABOVE + 500])
@pytest.mark.parametrize("offset", [False, True], ids=["", "offset"])
def test_zero_inflated_qf_at_0_is_0(n, offset):
    """With zero inflation the predictive starts at 0 (the units dead on
    arrival), so qf is 0 up to the mass there and inverts ff above it."""
    fit, pts = _spread(n, zi=True)
    if not offset:
        keep = [0, 1, 3]
        fit = BayesianParametric.from_draws(
            sp.Weibull, pts[:, keep], ["alpha", "beta", "f0"]
        )
    q = fit.qf(P_LOW)
    assert q[0] == 0.0
    assert np.all(np.diff(q) >= 0)
    mass = float(fit.ff(0.0))
    assert 0.02 < mass < 0.08
    np.testing.assert_array_equal(q[P_LOW <= mass], 0.0)
    above = P_LOW > mass
    assert fit.ff(q[above]) == pytest.approx(P_LOW[above], rel=1e-6)


# -- speed -----------------------------------------------------------------


def test_a_calculator_load_is_fast():
    """A two-parameter grid of about 37,500 points (a vague prior updated
    by four rows): building the table and 2,000 calls of 50 times each,
    the load of a replacement-interval search, well inside a generous
    ceiling (the full sum takes minutes for it)."""
    pr = prior.from_params(sp.Weibull, [5000, 2], within=0.6)
    fit = sp.Weibull.fit_bayes(
        [900.0, 1500, 2000, 2500], [0, 0, 1, 1], priors=pr
    )
    assert len(fit.posterior.weights) > 30_000
    t = np.linspace(10.0, 9000.0, 50)
    start = time.perf_counter()
    for i in range(2000):
        fit.sf(t * (1 + i * 1e-4))
        fit.hf(t)
    fit.qf(np.array([0.1, 0.5, 0.9]))
    elapsed = time.perf_counter() - start
    assert elapsed < 15.0
