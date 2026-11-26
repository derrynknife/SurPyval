"""``ParametricPanelMultiState(clock="reset")`` with a way back to a state
(an illness-death model with recovery): the paths between two visits of
at most ``max_moves`` moves are counted, and what the others would add is
checked gap by gap at the estimate."""

import time
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import dblquad
from scipy.linalg import expm

from surpyval import Exponential, Weibull
from surpyval.multistate import (
    PanelMultiState,
    ParametricPanelMultiState,
    TransitionMatrix,
)
from surpyval.multistate.panel import _check_panel
from surpyval.multistate.parametric import _fitters

STATES = ("healthy", "ill", "dead")
TM = TransitionMatrix(
    [("healthy", "ill"), ("healthy", "dead"), ("ill", "healthy"),
     ("ill", "dead")],
    states=list(STATES),
)  # fmt: skip
RATES = (0.3, 0.05, 0.2, 0.25)


def _markov_visits(n, seed, rates=RATES, gap=1.0, last=6.0):
    """Visits every ``gap`` to ``last`` of a Markov illness-death model
    with recovery (constant intensities ``rates``, in the order of
    ``TM.transitions``), all healthy at 0."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((3, 3))
    for (a, b), q in zip(TM.transitions, rates):
        Q[STATES.index(a), STATES.index(b)] = q
    Q -= np.diag(Q.sum(axis=1))
    P = expm(Q * gap)
    x, s, i = [], [], []
    for unit in range(n):
        state = 0
        for visit in np.arange(0.0, last, gap):
            if visit > 0:
                state = rng.choice(3, p=P[state] / P[state].sum())
            x.append(visit)
            s.append(STATES[state])
            i.append(unit)
            if state == 2:
                break
    return dict(x=np.array(x), s=np.array(s, dtype=object), i=np.array(i))


def _lik(pdata, dist, max_moves):
    return ParametricPanelMultiState._likelihood(
        pdata, _fitters(dist, pdata.tm, False), "reset", max_moves
    )


def _blocks(rates, span, K):
    """``[P_0, ..., P_K]``: ``P_k[a, b]``, the chance of exactly ``k``
    moves in ``span`` ending in ``b`` from ``a``, for constant
    intensities: the top row of blocks of the exponential of the
    block-bidiagonal matrix with ``-D`` on the diagonal and the off-
    diagonal intensities ``R`` above it (a Dyson series, exact)."""
    R = np.zeros((3, 3))
    for (a, b), q in zip(TM.transitions, rates):
        R[STATES.index(a), STATES.index(b)] = q
    D = np.diag(R.sum(axis=1))
    n = K + 1
    A = np.zeros((3 * n, 3 * n))
    for k in range(n):
        A[3 * k : 3 * k + 3, 3 * k : 3 * k + 3] = -D
        if k + 1 < n:
            A[3 * k : 3 * k + 3, 3 * k + 3 : 3 * k + 6] = R
    E = expm(A * span)
    return [E[0:3, 3 * k : 3 * k + 3] for k in range(n)], R


@pytest.mark.parametrize("K", [1, 2, 3])
def test_truncated_likelihood_is_the_dyson_series(K):
    """With constant hazards the chance of exactly k moves in a gap is a
    term of the Dyson series of the transition matrix (exact, by one
    block matrix exponential); the truncated likelihood is the product
    over the gaps of the terms up to K (for a death timed exactly, up to
    K - 1 moves before the move into death), and what it leaves out is
    the rest of the series."""
    rates = (0.6, 0.1, 0.5, 0.3)
    x = [0, 1.0, 2.5, 3.0, 0, 1.5, 2.0]
    s = ["healthy", "healthy", "ill", "dead", "healthy", "ill", "healthy"]
    i = [1, 1, 1, 1, 2, 2, 2]
    obstype = ["panel", "panel", "panel", "exact_entry"] + ["panel"] * 3
    pdata = _check_panel(x, s, i, None, obstype, TM, None, None, None)
    lik = _lik(pdata, Exponential, K)
    models = [Exponential.from_params([q]) for q in rates]
    H, I = STATES.index("healthy"), STATES.index("ill")
    Dd = STATES.index("dead")

    def M(span, kmax):
        blocks, R = _blocks(rates, span, K)
        return sum(blocks[: kmax + 1]), R

    # unit 1: healthy 0 -> healthy 1 -> ill 2.5 -> dead exactly at 3
    m1, _ = M(1.0, K)
    m2, _ = M(1.5, K)
    m3, R = M(0.5, K - 1)
    L1 = m1[H, H] * m2[H, I] * (m3[I, :] @ R[:, Dd]) / 1.0
    # (the forward recursion normalises each gap; the product of its
    # factors is the chance of the whole sequence)
    m4, _ = M(1.5, K)
    m5, _ = M(0.5, K)
    L2 = m4[H, I] * m5[I, H]
    assert_allclose(lik.neg_ll(models), -np.log(L1 * L2), rtol=1e-9)
    # what is left out: 1 - the counted chance of any end state
    omitted = dict(lik.omitted(models))
    full = {1: m1[H].sum(), 2: m2[H].sum(), 5: m4[H].sum(), 6: m5[I].sum()}
    for row, kept in full.items():
        assert_allclose(omitted[row], 1 - kept, rtol=1e-6, atol=1e-12)


def test_two_moves_out_and_back_by_quadrature():
    """Weibull hazards (one shape below 1, infinite at entry): healthy at
    0 and at 1.5 is staying, or falling ill and recovering in between
    (two moves, at unknown times), against scipy's adaptive quadrature;
    with ``max_moves=2`` nothing else is counted."""
    pars = {
        ("healthy", "ill"): (2.0, 0.8),
        ("healthy", "dead"): (10.0, 1.3),
        ("ill", "healthy"): (1.0, 1.6),
        ("ill", "dead"): (4.0, 1.2),
    }

    def H(p, d):
        a, b = pars[p]
        return (max(d, 0.0) / a) ** b

    def h(p, d):
        a, b = pars[p]
        return b / a * (d / a) ** (b - 1)

    def Sh(d):
        return np.exp(-H(("healthy", "ill"), d) - H(("healthy", "dead"), d))

    def Si(d):
        return np.exp(-H(("ill", "healthy"), d) - H(("ill", "dead"), d))

    T = 1.5
    back = dblquad(
        lambda v, u: Sh(u)
        * h(("healthy", "ill"), u)
        * Si(v)
        * h(("ill", "healthy"), v)
        * Sh(T - u - v),
        0,
        T,
        0,
        lambda u: T - u,
        epsabs=0,
        epsrel=1e-11,
    )[0]
    L = Sh(T) + back
    pdata = _check_panel(
        [0, T], ["healthy", "healthy"], [1, 1], None, None, TM, None, None,
        None,
    )  # fmt: skip
    lik = _lik(pdata, Weibull, 2)
    models = [Weibull.from_params(list(pars[p])) for p in TM.transitions]
    assert_allclose(lik.neg_ll(models), -np.log(L), rtol=1e-8)


@pytest.fixture(scope="module")
def recovery():
    return _markov_visits(20, 7)


def test_exponential_agreement_with_panel_multistate_grows_with_the_bound(
    recovery,
):
    """Constant hazards with recovery: the truncated fit on the reset
    clock approaches ``PanelMultiState``'s Markov fit as ``max_moves``
    rises (its log-likelihood, estimates and largest omission all
    shrinking), and the fit warns where the omission is above the
    tolerance. (What each gap's omission is, exactly, is the Dyson-series
    test above.)"""
    markov = PanelMultiState.fit(transitions=TM, **recovery)
    gaps, params, errors = [], [], []
    for K in (1, 2, 3):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = ParametricPanelMultiState.fit(
                transitions=TM,
                dist=Exponential,
                clock="reset",
                max_moves=K,
                **recovery,
            )
        assert any("max_moves" in str(w.message) for w in caught)
        assert model.max_moves == K
        table = model.truncation_errors
        assert len(table) == model.n_obs
        assert model.truncation_error == table["omitted"].max()
        gaps.append(markov.log_likelihood - model.log_likelihood)
        params.append(np.max(np.abs(model.params / markov.params - 1)))
        errors.append(model.truncation_error)
        assert gaps[-1] > 0
    assert gaps[0] > gaps[1] > gaps[2]
    assert params[0] > params[1] > params[2]
    assert errors[0] > errors[1] > errors[2]
    # (bound 3 is within 2e-2 of the
    # Markov fit's intensities on these data, gaps of 1 with rates up to
    # 0.45 a unit of time)
    assert params[2] < 2e-2


def test_a_progressive_model_counts_every_path():
    """Without a way back the paths between visits are finite: every one
    is counted, nothing is left out, whatever ``max_moves``."""
    tm = TransitionMatrix(
        [("healthy", "ill"), ("healthy", "dead"), ("ill", "dead")]
    )
    d = dict(
        x=[0, 1, 2, 0, 1, 3, 0, 2, 3, 0, 1, 2],
        s=["healthy", "ill", "dead", "healthy", "healthy", "ill",
           "healthy", "dead", "dead", "healthy", "healthy", "healthy"],
        i=[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    )  # fmt: skip
    d["x"], d["i"] = d["x"][:8] + d["x"][9:], d["i"][:8] + d["i"][9:]
    d["s"] = d["s"][:8] + d["s"][9:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ParametricPanelMultiState.fit(
            transitions=tm, dist=Exponential, clock="reset", max_moves=1, **d
        )
        b = ParametricPanelMultiState.fit(
            transitions=tm, dist=Exponential, clock="reset", **d
        )
    assert a.neg_ll() == b.neg_ll()
    assert a.truncation_error == 0.0
    assert a.truncation_errors.empty


def test_round_trip_keeps_the_truncation(recovery):
    import surpyval as sp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ParametricPanelMultiState.fit(
            transitions=TM, dist=Exponential, clock="reset", max_moves=2,
            **recovery,
        )  # fmt: skip
    back = sp.from_dict(model.to_dict())
    assert back.max_moves == 2
    assert back.truncation_error == model.truncation_error
    assert back.truncation_errors.equals(model.truncation_errors)
    assert back.neg_ll() == model.neg_ll()
    assert_allclose(back.occupancy(3.0, "ill"), model.occupancy(3.0, "ill"))


def test_timing_at_the_default_bound(recovery):
    """Not a check of speed: a record of it (printed with ``-s``). One
    likelihood of the 20 units at bound 3."""
    pdata = _check_panel(
        recovery["x"], recovery["s"], recovery["i"], None, None, TM, None,
        None, None,
    )  # fmt: skip
    lik = _lik(pdata, Exponential, 3)
    models = [Exponential.from_params([q]) for q in RATES]
    lik.neg_ll(models)
    start = time.perf_counter()
    lik.neg_ll(models)
    print(
        "one likelihood, 20 units, bound 3: "
        f"{time.perf_counter() - start:.2f} s"
    )
