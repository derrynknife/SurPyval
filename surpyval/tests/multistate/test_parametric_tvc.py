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
from surpyval.multistate import ParametricMultiStateRegression

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
    """The spells split by hand at each unit's change of ``w``."""
    out = []
    for r in df.itertuples():
        switch = float(paths[r.i].edges[1])
        if r.tl < switch < r.x:
            out.append((r.i, r.tl, switch, r.state, None, r.male, 0.0))
            out.append((r.i, switch, r.x, r.state, r.e, r.male, 1.0))
        else:
            w = 1.0 if r.tl >= switch else 0.0
            out.append((r.i, r.tl, r.x, r.state, r.e, r.male, w))
    return pd.DataFrame(
        out, columns=["i", "tl", "x", "state", "e", "male", "w"]
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


def _by_formula(model, hand, clock):
    r"""The log-likelihood written out from the hand-split rows: each
    row's :math:`d \log h(b \mid z) - [H(b \mid z) - H(a \mid z)]` for
    the transitions out of its state, on the time since the start
    (forward) or since entering the state (reset; the stay's start is
    the first row's ``tl``, carried over the split)."""
    total = 0.0
    stay = hand.groupby(["i", "state"])["tl"].transform("min").to_numpy()
    for k, r in enumerate(hand.itertuples()):
        z = np.array([[r.male, r.w]])
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
    by_hand = _fit(hand, hand[["male", "w"]], clock)
    assert tvc.covariate_names == ["male", "time_varying_0"]
    assert tvc._data.x.size == hand.shape[0]
    assert_allclose(tvc.log_likelihood, by_hand.log_likelihood, rtol=0,
                    atol=1e-8)  # fmt: skip
    assert_allclose(tvc.params, by_hand.params, rtol=1e-6, atol=1e-8)
    assert_allclose(
        _by_formula(tvc, hand, clock), tvc.log_likelihood, rtol=0, atol=1e-8
    )


def test_reset_clock_split_does_not_restart_the_stay(data):
    """On the reset clock the split rows of a stay are one stay: without
    the units (``i``) each row restarts the clock, a different and wrong
    likelihood."""
    df, paths, hand = data
    right = _fit(hand, hand[["male", "w"]], "reset")
    restarted = _fit(hand, hand[["male", "w"]], "reset", i=False)
    assert abs(right.log_likelihood - restarted.log_likelihood) > 1.0
    # (and on the forward clock the units make no difference)
    a = _fit(hand, hand[["male", "w"]], "forward")
    b = _fit(hand, hand[["male", "w"]], "forward", i=False)
    assert_allclose(a.log_likelihood, b.log_likelihood, rtol=1e-12)


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
