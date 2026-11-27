"""Time-varying covariates in ``ParametricMultiStateRegression``: each
unit's covariate path given once, its spells split at the changes inside
the fit, on either clock (the reset clock carrying the time in the state
on across the splits), and predictions along a path."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

import surpyval as sp
from surpyval import StepSchedule, WeibullAFT, WeibullPH
from surpyval.multistate import (
    ParametricMultiState,
    ParametricMultiStateRegression,
)

TM = [("healthy", "ill"), ("healthy", "dead"), ("ill", "dead")]


def _spells(n=150, seed=708):
    """An illness-death history per unit (fixed-covariate Weibull
    hazards: the spells need not come from the model for the identities
    checked here), a fixed covariate ``male``, a load ``w`` that steps
    from 0 to 1 at a unit's own time, and every fifth unit entering late
    (its first spell starts at 1)."""
    rng = np.random.default_rng(seed)
    rows = []
    paths = {}
    for u in range(n):
        male = float(rng.uniform() < 0.5)
        switch = float(rng.uniform(0.5, 6.0))
        entry = 1.0 if u % 5 == 0 else 0.0
        end = float(rng.uniform(6.0, 10.0))
        t_ill = entry + 6.0 * rng.weibull(1.5)
        t_dead = entry + 20.0 * rng.weibull(1.2)
        u_name = f"u{u}"
        if t_ill < min(t_dead, end):
            rows.append((u_name, entry, t_ill, "healthy", "ill", male))
            t_d2 = t_ill + 5.0 * rng.weibull(2.0)
            if t_d2 < end:
                rows.append((u_name, t_ill, t_d2, "ill", "dead", male))
            else:
                rows.append((u_name, t_ill, end, "ill", None, male))
        elif t_dead < end:
            rows.append((u_name, entry, t_dead, "healthy", "dead", male))
        else:
            rows.append((u_name, entry, end, "healthy", None, male))
        paths[u_name] = StepSchedule.from_changepoints([0.0, switch], [0, 1])
    df = pd.DataFrame(rows, columns=["i", "tl", "x", "state", "e", "male"])
    return df, paths


def _hand_split(df, paths):
    """The spells split by hand at each unit's change of ``w``, the
    second part of each split marked as carrying on its stay."""
    out = []
    for r in df.itertuples():
        switch = float(paths[r.i].edges[1])
        if r.tl < switch < r.x:
            out.append((r.i, r.tl, switch, r.state, None, r.male, 0.0, 0))
            out.append((r.i, switch, r.x, r.state, r.e, r.male, 1.0, 1))
        else:
            w = 1.0 if r.tl >= switch else 0.0
            out.append((r.i, r.tl, r.x, r.state, r.e, r.male, w, 0))
    return pd.DataFrame(
        out,
        columns=["i", "tl", "x", "state", "e", "male", "w", "carry"],
    )


@pytest.fixture(scope="module")
def data():
    df, paths = _spells()
    return df, paths, _hand_split(df, paths)


def _fit(df, Z, clock, i=True, **kw):
    return ParametricMultiStateRegression.fit(
        df["x"].to_numpy(), Z, df["e"].to_numpy(dtype=object),
        df["state"].to_numpy(dtype=object), tl=df["tl"].to_numpy(),
        i=df["i"].to_numpy(dtype=object) if i else None, transitions=TM,
        clock=clock, **kw,
    )  # fmt: skip


def _stay_starts(rows):
    """The start of each row's stay, by hand: its ``tl``, or for a row
    marked as carrying on, the start of the stay of the row before it
    (the rows of a unit in time order; a unit may enter a state again)."""
    stay = np.empty(len(rows))
    for k, r in enumerate(rows.itertuples()):
        if r.carry:
            assert rows["i"].iloc[k - 1] == r.i
            stay[k] = stay[k - 1]
        else:
            stay[k] = r.tl
    return stay


def _by_formula(model, hand, clock, z_of=lambda r: [[r.male, r.w]]):
    r"""The log-likelihood written out from the hand-split rows: each
    row's :math:`d \log h(b \mid z) - [H(b \mid z) - H(a \mid z)]` for
    the transitions out of its state, on the time since the start
    (forward) or since entering the state (reset; the stay's start
    carried over the rows marked as carrying on)."""
    total = 0.0
    stay = _stay_starts(hand)
    for k, r in enumerate(hand.itertuples()):
        z = np.array(z_of(r))
        zero = stay[k] if clock == "reset" else 0.0
        a, b = r.tl - zero, r.x - zero
        for pair, m in zip(model.transitions.transitions, model._models()):
            if pair[0] != r.state:
                continue
            Hb = float(m.Hf(np.array([b]), z)[0])
            Ha = float(m.Hf(np.array([a]), z)[0]) if a > 0 else 0.0
            total -= Hb - Ha
            if r.e == pair[1]:
                total += float(np.log(m.hf(np.array([b]), z)[0]))
    return total


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_internal_split_is_the_hand_split(data, clock):
    """The fit with each unit's path is the fit to the stays split by
    hand (to 1e-8 in the log-likelihood), and both are the likelihood
    written out row by row at the estimate; delayed entry kept."""
    df, paths, hand = data
    tvc = _fit(df, df[["male"]], clock, time_varying_covariates=paths)
    by_hand = _fit(hand, hand[["male", "w"]], clock,
                   carry_on=hand["carry"].to_numpy())  # fmt: skip
    assert tvc.covariate_names == ["male", "time_varying_0"]
    assert tvc._data.x.size == hand.shape[0]
    assert_allclose(tvc.log_likelihood, by_hand.log_likelihood, rtol=0,
                    atol=1e-8)  # fmt: skip
    assert_allclose(tvc.params, by_hand.params, rtol=1e-6, atol=1e-8)
    assert_allclose(
        _by_formula(tvc, hand, clock), tvc.log_likelihood, rtol=0, atol=1e-8
    )


def test_reset_clock_split_rows_carry_on_only_where_marked(data):
    """On the reset clock rows marked ``carry_on`` continue their stay;
    unmarked, each row starts a stay of its own (the reading of a unit
    renewed in its state), a different likelihood for split stays. On
    the forward clock the marks change nothing."""
    df, paths, hand = data
    carry = hand["carry"].to_numpy()
    right = _fit(hand, hand[["male", "w"]], "reset", carry_on=carry)
    own = _fit(hand, hand[["male", "w"]], "reset")
    assert abs(right.log_likelihood - own.log_likelihood) > 1.0
    a = _fit(hand, hand[["male", "w"]], "forward", carry_on=carry)
    b = _fit(hand, hand[["male", "w"]], "forward")
    assert_allclose(a.log_likelihood, b.log_likelihood, rtol=0, atol=0)


@pytest.mark.filterwarnings("ignore:Covariate column")
def test_a_constant_path_is_the_fixed_covariate_fit(data):
    df, _, _ = data
    paths = {u: StepSchedule.constant([0.0]) for u in df["i"].unique()}
    for clock in ("forward", "reset"):
        tvc = _fit(df, df[["male"]], clock, time_varying_covariates=paths)
        fixed = _fit(df.assign(w=0.0), df.assign(w=0.0)[["male", "w"]], clock)
        assert_allclose(tvc.log_likelihood, fixed.log_likelihood, atol=1e-8)


def test_surpyval_data_with_time_varying_covariates(data):
    df, paths, _ = data
    d = sp.SurpyvalData(
        x=df["x"].to_numpy(), s=df["state"].to_numpy(dtype=object),
        e=df["e"].to_numpy(dtype=object), i=df["i"].to_numpy(dtype=object),
        tl=df["tl"].to_numpy(), Z=df[["male"]],
    )  # fmt: skip
    d.time_varying_covariates = paths
    a = ParametricMultiStateRegression.fit(d, transitions=TM)
    b = _fit(df, df[["male"]], "forward", time_varying_covariates=paths)
    assert_allclose(a.log_likelihood, b.log_likelihood, rtol=1e-12)
    with pytest.raises(ValueError, match="ParametricMultiStateRegression"):
        sp.MultiStateProportionalHazards.fit(d, transitions=TM)


def test_round_trip_keeps_the_split(data):
    df, paths, _ = data
    for clock in ("forward", "reset"):
        model = _fit(df, df[["male"]], clock, time_varying_covariates=paths)
        back = sp.from_dict(model.to_dict())
        assert_allclose(back.params, model.params)
        assert_allclose(
            back.occupancy(4.0, [1.0, 1.0], "ill"),
            model.occupancy(4.0, [1.0, 1.0], "ill"),
            rtol=1e-12,
        )


def test_refusals(data):
    df, paths, _ = data
    with pytest.raises(ValueError, match="accelerated failure time"):
        _fit(df, df[["male"]], "forward", model=WeibullAFT,
             time_varying_covariates=paths)  # fmt: skip
    short = dict(list(paths.items())[1:])
    with pytest.raises(ValueError, match="no path for 1 unit"):
        _fit(df, df[["male"]], "forward", time_varying_covariates=short)
    with pytest.raises(ValueError, match="'i'"):
        _fit(df, df[["male"]], "forward", i=False,
             time_varying_covariates=paths)  # fmt: skip
    late = dict(paths)
    late["u1"] = StepSchedule.from_changepoints([0.5, 2.0], [0, 1])
    with pytest.raises(ValueError, match="start at 0.5"):
        _fit(df, df[["male"]], "forward", time_varying_covariates=late)


def _P_along(model, path, z_male, s, t):
    """P(s, t) along a step path by an ODE solver of the forward
    equations (the hazards at each time with the covariates then)."""
    tm = model.transitions
    frm, to = tm.from_index(), tm.to_index()
    k = tm.n_states

    def rhs(u, p):
        P = p.reshape(k, k)
        Q = np.zeros((k, k))
        w = 1.0 if u >= float(path.edges[1]) else 0.0
        for h, m in enumerate(model._models()):
            Q[frm[h], to[h]] = m.hf(np.array([u]), np.array([[z_male, w]]))[0]
        Q -= np.diag(Q.sum(axis=1))
        return (P @ Q).ravel()

    switch = float(path.edges[1])
    p = np.eye(k).ravel()
    pieces = [(s, switch), (switch, t)] if s < switch < t else [(s, t)]
    for a, b in pieces:
        p = solve_ivp(rhs, (a, b), p, rtol=1e-11, atol=1e-13).y[:, -1]
    return p.reshape(k, k)


def test_predictions_along_a_path(data):
    """Occupancy along a step path agrees with an ODE solution of the
    forward equations and its bounds
    contain it; before the path's change it is the prediction for the
    first load, and with a constant path the fixed-covariate one."""
    df, paths, _ = data
    model = _fit(df, df[["male"]], "forward", time_varying_covariates=paths)
    path = StepSchedule(edges=[0.0, 3.0, np.inf], Z=[[1.0, 0.0], [1.0, 1.0]])
    t = np.array([2.0, 5.0, 8.0])
    P = model.transition_probabilities(t, path, s=0.0)
    for j, tj in enumerate(t):
        assert_allclose(P[j], _P_along(model, path, 1.0, 0.0, tj), atol=1e-7)
    occ = model.occupancy(t, path, "dead", from_state="healthy")
    lo = model.occupancy(t, [1.0, 0.0], "dead", from_state="healthy")
    hi = model.occupancy(t, [1.0, 1.0], "dead", from_state="healthy")
    assert_allclose(occ[0], lo[0], rtol=1e-8)
    # (after the switch it is neither: in a model with an intermediate
    # state the occupancy along a path need not lie between the two)
    assert not np.allclose(occ[1:], lo[1:]) and not np.allclose(occ, hi)
    cb = model.cb(t, path, "dead", from_state="healthy")
    assert np.all((cb[:, 0] <= occ) & (occ <= cb[:, 1]))
    const = StepSchedule.constant([1.0, 1.0])
    assert_allclose(
        model.occupancy(t, const, "ill", from_state="healthy"),
        model.occupancy(t, [1.0, 1.0], "ill", from_state="healthy"),
        rtol=1e-7,
    )
    H = model.cumulative_hazard(t, path, "healthy", "ill")
    m = model._models()[0]
    assert_allclose(H, m.Hf_tvc(t, path), rtol=1e-12)


def test_path_predictions_refused_where_they_cannot_be_right(data):
    df, paths, _ = data
    path = StepSchedule(edges=[0.0, 3.0, np.inf], Z=[[1.0, 0.0], [1.0, 1.0]])
    reset = _fit(df, df[["male"]], "reset", time_varying_covariates=paths)
    with pytest.raises(ValueError, match="clock='reset'"):
        reset.occupancy(4.0, path, "ill")
    aft = _fit(df, df[["male"]], "forward", model=WeibullAFT)
    with pytest.raises(ValueError, match="covariate path gives 2"):
        aft.occupancy(4.0, path, "ill")
    one = StepSchedule(edges=[0.0, 3.0, np.inf], Z=[[0.0], [1.0]])
    with pytest.raises(ValueError, match="Accelerated Failure Time"):
        aft.occupancy(4.0, one, "ill")


def test_a_continuous_path_is_fitted_on_steps_until_they_stop_mattering(data):
    """A ramp (a ``CovariatePath``) is cut into steps, halved until no
    estimate moves by more than 1% of its standard error; the fit is the
    fit to the path cut into the final steps by hand."""
    from surpyval import CovariatePath
    from surpyval.multistate import _tvc

    df, paths, _ = data
    ramps = {
        u: CovariatePath.from_points([0.0, float(p.edges[1]), 20.0],
                                     [0.0, 1.0, 1.0])
        for u, p in paths.items()
    }  # fmt: skip
    model = _fit(df, df[["male"]], "forward", time_varying_covariates=ramps)
    report = model.path_steps
    assert report["converged"]
    assert report["change"] <= report["tolerance"] == 0.01
    # by hand: the same steps, through the step-schedule route
    model2 = ParametricMultiStateRegression.fit(
        df["x"].to_numpy(), df[["male"]], df["e"].to_numpy(dtype=object),
        df["state"].to_numpy(dtype=object), tl=df["tl"].to_numpy(),
        i=df["i"].to_numpy(dtype=object), transitions=TM,
    )  # fmt: skip
    data_ = model2._data
    units = _tvc.unit_paths(ramps, df["i"].to_numpy(dtype=object), data_)
    split, Z = _tvc.split(data_, df[["male"]].to_numpy(), units, 0.0,
                          report["steps"])  # fmt: skip
    assert split.x.size == report["rows"]
    again = ParametricMultiStateRegression._fit_named(
        split, Z, ["male", "time_varying_0"],
        [WeibullPH] * 3, "forward",
    )  # fmt: skip
    assert_allclose(again.log_likelihood, model.log_likelihood, atol=1e-10)


def _up_down(n=100, seed=7081, end=30.0):
    """A machine that fails and is repaired, again and again: Weibull
    hazards of the time in each state (up: scale 5, shape 1.8; down:
    scale 1, shape 1.2), a fixed covariate per unit, followed to
    ``end``; every second stay split by hand at its middle, the second
    part marked as carrying on."""
    rng = np.random.default_rng(seed)
    one, split = [], []
    k = 0
    for u in range(n):
        z = float(u % 2)
        t, state = 0.0, "up"
        while t < end:
            a, b = (5.0, 1.8) if state == "up" else (1.0, 1.2)
            d = a * rng.weibull(b)
            nxt = "down" if state == "up" else "up"
            x, e = (t + d, nxt) if t + d < end else (end, None)
            one.append((u, t, x, state, e, z, 0))
            if k % 2 and x - t > 0.2:
                m = 0.5 * (t + x)
                split.append((u, t, m, state, None, z, 0))
                split.append((u, m, x, state, e, z, 1))
            else:
                split.append((u, t, x, state, e, z, 0))
            k += 1
            t, state = x, nxt
    cols = ["i", "tl", "x", "state", "e", "z", "carry"]
    return pd.DataFrame(one, columns=cols), pd.DataFrame(split, columns=cols)


def _fit_ud(df, carry=True, **kw):
    return ParametricMultiStateRegression.fit(
        df["x"].to_numpy(), df[["z"]], df["e"].to_numpy(dtype=object),
        df["state"].to_numpy(dtype=object), tl=df["tl"].to_numpy(),
        i=df["i"].to_numpy(), transitions=[("up", "down"), ("down", "up")],
        clock="reset", carry_on=df["carry"].to_numpy() if carry else None,
        **kw,
    )  # fmt: skip


def test_split_stays_with_re_entry_are_the_unsplit_stays():
    """A unit that enters a state again and again: half its stays split
    by hand and marked fit as the stays given whole, and as the
    likelihood written out row by row (each re-entry a new stay). The
    split rows' likelihood at the whole stays' estimate is the whole
    stays' maximum: the same objective. (The two searches, one with left
    truncation and one without, stop about 3e-6 apart on a coefficient,
    within the optimiser's tolerance.)"""
    whole, split = _up_down()
    a = _fit_ud(whole)
    b = _fit_ud(split)
    assert split.shape[0] > whole.shape[0] + 100
    assert_allclose(b.log_likelihood, a.log_likelihood, rtol=0, atol=1e-8)
    assert_allclose(
        _by_formula(a, split, "reset", z_of=lambda r: [[r.z]]),
        a.log_likelihood,
        rtol=0,
        atol=1e-8,
    )
    assert_allclose(
        _by_formula(b, split, "reset", z_of=lambda r: [[r.z]]),
        b.log_likelihood,
        rtol=0,
        atol=1e-8,
    )


def test_a_renewal_in_the_same_state_starts_a_new_stay():
    """Preventive maintenance: a unit is censored at age 2 of its stay
    and renewed, as good as new, in the same 'up' state at the same time
    under the same i. Unmarked rows read as new stays (the default), and
    recover the up-time Weibull (scale 3, shape 2.5) within 3 standard
    errors; marked as carrying on they would read as one long stay."""
    rng = np.random.default_rng(7082)
    rows = []
    for u in range(150):
        t = 0.0
        while t < 12.0:
            d = 3.0 * rng.weibull(2.5)
            if d < 2.0:
                rows.append((u, t, t + d, "up", "failed"))
                break
            rows.append((u, t, t + 2.0, "up", None))
            t += 2.0
    df = pd.DataFrame(rows, columns=["i", "tl", "x", "state", "e"])
    kw = dict(tl=df["tl"], i=df["i"], transitions=[("up", "failed")],
              clock="reset")  # fmt: skip
    renewal = ParametricMultiState.fit(df["x"], df["e"], df["state"], **kw)
    z = (renewal.params - np.array([3.0, 2.5])) / renewal.standard_errors()
    assert np.all(np.abs(z) < 3), z
    carried = df["tl"].to_numpy() > 0
    one = ParametricMultiState.fit(
        df["x"], df["e"], df["state"], carry_on=carried, **kw
    )
    assert not np.allclose(one.params, renewal.params, rtol=0.05)


def test_carry_on_refusals():
    whole, split = _up_down(n=10)
    bad = split["carry"].to_numpy().copy()
    bad[0] = 1  # a unit's first row
    with pytest.raises(ValueError, match="do not carry on the row before"):
        _fit_ud(split.assign(carry=bad))
    gap = split.copy()
    k = int(np.flatnonzero(gap["carry"].to_numpy())[0])
    gap.loc[k, "tl"] += 0.01
    with pytest.raises(ValueError, match="1 row"):
        _fit_ud(gap)
    other = whole.copy()
    other.loc[1, "carry"] = 1  # its row before is a stay in another state
    with pytest.raises(ValueError, match="same state"):
        _fit_ud(other)
    with pytest.raises(ValueError, match="'i'"):
        ParametricMultiState.fit(
            split["x"], split["e"], split["state"], tl=split["tl"],
            transitions=[("up", "down"), ("down", "up")], clock="reset",
            carry_on=split["carry"].to_numpy(),
        )  # fmt: skip
    with pytest.raises(ValueError, match="one True or False per row"):
        ParametricMultiState.fit(
            split["x"], split["e"], split["state"], tl=split["tl"],
            i=split["i"], transitions=[("up", "down"), ("down", "up")],
            clock="reset", carry_on=[True],
        )  # fmt: skip


def test_bayes_marked_split_stays_are_the_whole_stays():
    """The Bayesian fit on the reset clock: stays split by hand and
    marked give each transition the posterior of the stays given whole
    (the grid's points, weights and evidence)."""
    whole, split = _up_down(n=40)
    kw = dict(transitions=[("up", "down"), ("down", "up")], clock="reset",
              dist=sp.Weibull, method="grid")  # fmt: skip

    def fit(df, carry):
        return ParametricMultiState.fit_bayes(
            df["x"].to_numpy(), df["e"].to_numpy(dtype=object),
            df["state"].to_numpy(dtype=object), tl=df["tl"].to_numpy(),
            i=df["i"].to_numpy(), carry_on=carry, **kw,
        )  # fmt: skip

    a = fit(whole, None)
    b = fit(split, split["carry"].to_numpy())
    for pair in a.models:
        pa, pb = a.models[pair], b.models[pair]
        assert_allclose(pb.log_evidence, pa.log_evidence, rtol=0, atol=1e-8)
        assert_allclose(pb.posterior.points, pa.posterior.points, rtol=1e-8)
        assert_allclose(pb.posterior.weights, pa.posterior.weights,
                        rtol=1e-6, atol=1e-12)  # fmt: skip


def test_predictions_along_a_ramp(data):
    """Along a CovariatePath ramp (w from 0 at time 0 to 1 at 6, then
    held) the transition probabilities agree with an ODE solution of the
    forward equations with the hazards at each time's w."""
    from surpyval import CovariatePath

    df, paths, _ = data
    model = _fit(df, df[["male"]], "forward", time_varying_covariates=paths)
    ramp = CovariatePath.from_points(
        [0.0, 6.0, 20.0], [[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
    )
    tm = model.transitions
    frm, to = tm.from_index(), tm.to_index()
    k = tm.n_states

    def rhs(u, p):
        Q = np.zeros((k, k))
        z = np.array([[1.0, min(u / 6.0, 1.0)]])
        for h, m in enumerate(model._models()):
            Q[frm[h], to[h]] = m.hf(np.array([u]), z)[0]
        Q -= np.diag(Q.sum(axis=1))
        return (p.reshape(k, k) @ Q).ravel()

    t = np.array([2.0, 5.0, 8.0])
    P = model.transition_probabilities(t, ramp, s=0.0)
    for j, tj in enumerate(t):
        pieces = [(0.0, min(tj, 6.0))] + ([(6.0, tj)] if tj > 6 else [])
        p = np.eye(k).ravel()
        for a, b in pieces:
            p = solve_ivp(rhs, (a, b), p, rtol=1e-11, atol=1e-13).y[:, -1]
        assert_allclose(P[j], p.reshape(k, k), atol=1e-6)


def test_twice_the_steps_moves_the_estimates_by_under_one_percent(data):
    """The steps the fit on a continuous path chose: halving them again
    moves no estimate by more than 1% of its standard error."""
    from surpyval import CovariatePath
    from surpyval.multistate import _tvc

    df, paths, _ = data
    ramps = {
        u: CovariatePath.from_points([0.0, float(p.edges[1]), 20.0],
                                     [0.0, 1.0, 1.0])
        for u, p in paths.items()
    }  # fmt: skip
    model = _fit(df, df[["male"]], "forward", time_varying_covariates=ramps)
    steps = model.path_steps["steps"]
    base = _fit(df, df[["male"]], "forward")
    units = _tvc.unit_paths(ramps, df["i"].to_numpy(dtype=object), base._data)
    split, Z = _tvc.split(base._data, df[["male"]].to_numpy(), units, 0.0,
                          2 * steps)  # fmt: skip
    finer = ParametricMultiStateRegression._fit_named(
        split, Z, ["male", "time_varying_0"], [WeibullPH] * 3, "forward"
    )
    move = np.abs(finer.params - model.params) / finer.standard_errors()
    assert np.max(move) <= 0.01, move
