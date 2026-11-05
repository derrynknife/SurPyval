"""``CoxPH.fit(..., time_interaction=)``: time-varying coefficients
beta(t) = beta + gamma g(t). The R references are in
``surpyval/tests/reference/test_cox_time_interaction.py``; these are the
properties: the split equals the start-stop fit written out by hand,
predictions apply beta(t) at each baseline time, the sign convention, and
the refusals."""

import json
import pickle

import matplotlib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from surpyval import CoxPH
from surpyval.univariate.regression.proportional_hazards.time_interaction import (  # noqa: E501
    TimeVaryingCoefficientModel,
)

matplotlib.use("Agg")


def _data(seed=0, size=200, gamma=1.0):
    """A binary covariate whose log hazard ratio is gamma * log(t) (so
    the hazard ratio is t**gamma: below 1 early, above 1 late), and a
    second, constant-effect covariate."""
    rng = np.random.default_rng(seed)
    z = rng.binomial(1, 0.5, size).astype(float)
    u = rng.normal(size=size)
    e = rng.exponential(size=size) / np.exp(0.5 * u)
    # H(t) = t for z = 0 and t**(1 + gamma) / (1 + gamma) for z = 1
    t = np.where(z == 1, ((1 + gamma) * e) ** (1 / (1 + gamma)), e)
    cens = rng.uniform(0.5, 3.0, size)
    x = np.minimum(t, cens)
    c = (t > cens).astype(int)
    return x, c, np.column_stack([z, u])


def _by_hand(x, c, Z, tl, j, g):
    """The rows split at every event time by a loop, as survSplit."""
    events = np.unique(x[c == 0])
    rows = []
    for k in range(len(x)):
        lo = tl[k]
        cuts = [e for e in events if lo < e < x[k]]
        starts = [lo] + cuts
        stops = cuts + [x[k]]
        for m, (s, t) in enumerate(zip(starts, stops)):
            last = m == len(stops) - 1
            rows.append((k, s, t, c[k] if last else 1, *Z[k], Z[k, j] * g(t)))
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(int), a[:, 4:]


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_equals_the_start_stop_fit_written_out_by_hand(ties):
    x, c, Z = _data()
    tl = np.where(np.arange(len(x)) % 4 == 0, 0.2 * x, -np.inf)
    model = CoxPH.fit(
        x, Z, c=c, tl=tl, tie_method=ties, time_interaction={0: "log"}
    )
    i, xl, xr, cc, Zs = _by_hand(x, c, Z, tl, 0, np.log)
    hand = CoxPH.fit_tvc(i, xl, xr, cc, Zs, tie_method=ties)
    assert isinstance(model, TimeVaryingCoefficientModel)
    assert_allclose(model.params, hand.params, rtol=1e-9, atol=1e-12)
    assert_allclose(model.covariance(), hand.covariance(), rtol=1e-8)
    assert_allclose(model.H0, hand.H0, rtol=1e-9)


def test_rows_with_a_zero_covariate_are_not_split_and_nothing_changes():
    x, c, Z = _data(1)
    model = CoxPH.fit(x, Z, c=c, time_interaction={0: "t"})
    i, xl, xr, cc, Zs = _by_hand(x, c, Z, np.full(len(x), -np.inf), 0, float)
    hand = CoxPH.fit_tvc(i, xl, xr, cc, Zs)
    # Only the rows with z = 1 were split
    assert len(model._fit_data["x"]) < len(xr)
    assert_allclose(model.params, hand.params, rtol=1e-9)
    assert_allclose(model.standard_errors(), hand.standard_errors())


def test_an_effect_that_grows_with_time_has_a_positive_gamma():
    # Positive coefficient = shorter life, for gamma too: the hazard
    # ratio of z = 1 rises with time.
    x, c, Z = _data(2, size=600, gamma=1.0)
    model = CoxPH.fit(x, Z, c=c, time_interaction={0: "log"})
    gamma, se = model.params[2], model.standard_errors()[2]
    assert gamma > 0
    # Simulated beta(t) = 0 + 1 log(t), so both within 3 standard errors
    assert abs(gamma - 1.0) < 3 * se
    assert abs(model.params[0]) < 3 * model.standard_errors()[0]
    effect = model.time_varying_effect(0, [0.2, 2.0])
    assert effect.beta[0] < 0 < effect.beta[1]
    assert np.all(effect.lower < effect.beta)
    assert np.all(effect.beta < effect.upper)
    # Its survival crosses that of z = 0: better early, worse late
    early, late = 0.2, 2.5
    assert CoxPH and model.sf(early, [1, 0]) > model.sf(early, [0, 0])
    S1 = model.sf(late, [1, 0])
    S0 = model.sf(late, [0, 0])
    assert S1 < S0


def test_predictions_apply_beta_of_t_at_each_baseline_time():
    x, c, Z = _data(3)
    model = CoxPH.fit(x, Z, c=c, time_interaction={0: ["t", "log"]})
    b = model.params
    z = np.array([1.0, 0.3])
    eta = (
        b[0] * z[0]
        + b[1] * z[1]
        + z[0] * (b[2] * model.x + b[3] * np.log(model.x))
    )
    jumps = model.h0 * np.exp(eta)
    t = np.array([0.1, 0.5, 1.0, 2.0])
    H = np.array([jumps[model.x <= q].sum() for q in t])
    assert_allclose(model.Hf(t, z), H, rtol=1e-12)
    assert_allclose(model.sf(t, z), np.exp(-H), rtol=1e-12)
    # The hazard step at a baseline time
    k = np.flatnonzero(model.d > 0)[5]
    assert_allclose(model.hf(model.x[k], z), jumps[k], rtol=1e-12)
    # grid and paired agree
    rows = np.array([[1.0, 0.3], [0.0, -1.0], [1.0, 1.0]])
    grid = model.sf(t, rows, grid=True)
    assert grid.shape == (3, 4)
    for r in range(3):
        assert_allclose(model.sf(t, rows[r]), grid[r])
    paired = model.sf(t[:3], rows)
    assert_allclose(paired, np.diag(grid[:, :3]))
    assert_allclose(model.sf(1.0, rows), grid[:, 2])
    # A row whose interacting covariate is 0 has proportional hazards
    assert_allclose(
        model.Hf(t, [0.0, 0.5]),
        model.Hf(t, [0.0, 0.0]) * np.exp(b[1] * 0.5),
        rtol=1e-12,
    )


def test_quantiles_and_draws_take_one_value_per_covariate():
    x, c, Z = _data(4)
    model = CoxPH.fit(x, Z, c=c, time_interaction={0: "log"})
    q = model.qf(0.3, [1.0, 0.0])
    assert np.isfinite(q)
    assert model.ff(q, [1.0, 0.0]) >= 0.3 - 1e-9
    draws, rows = model.random(5, [[1.0, 0.0]], random_state=1)
    assert draws.shape == (5,) and rows.shape == (5, 2)
    with pytest.raises(ValueError, match="covariate"):
        model.sf(1.0, [1.0, 0.0, 0.0])


def test_named_columns_name_the_coefficients():
    x, c, Z = _data(5)
    df = pd.DataFrame({"x": x, "c": c, "dose": Z[:, 0], "age": Z[:, 1]})
    model = CoxPH.fit_from_df(
        df,
        x_col="x",
        c_col="c",
        Z_cols=["dose", "age"],
        time_interaction={"dose": "log"},
    )
    assert model.parameter_names == ["dose", "age", "dose:log(t)"]
    assert list(model.summary().index) == model.parameter_names
    on_array = CoxPH.fit(x, Z, c=c, time_interaction={0: "log"})
    assert_allclose(model.params, on_array.params)
    assert on_array.parameter_names == ["coef_0", "coef_1", "coef_0:log(t)"]
    frame = CoxPH.fit(
        x, df[["dose", "age"]], c=c, time_interaction={"dose": "log"}
    )
    assert_allclose(frame.params, on_array.params)
    new = pd.DataFrame({"age": [0.0], "dose": [1.0]})
    assert_allclose(model.sf([1.0], new), on_array.sf([1.0], [1.0, 0.0]))
    assert "dose:log(t)" in repr(model)
    effect = model.time_varying_effect("dose")
    assert effect.covariate == "dose"
    assert json.dumps(effect.to_dict(), allow_nan=False)
    assert list(effect.to_frame().columns)[:2] == ["x", "beta"]


def test_a_callable_and_strata():
    x, c, Z = _data(6)
    s = np.arange(len(x)) % 2

    def sqrt(t):
        return np.sqrt(t)

    model = CoxPH.fit(x, Z, c=c, strata=s, time_interaction={1: sqrt})
    assert model.parameter_names[-1] == "coef_1:sqrt(t)"
    assert model.is_stratified
    assert np.all(np.isfinite(model.sf([0.5, 1.0], [1.0, 0.2], stratum=1)))
    lam = CoxPH.fit(x, Z, c=c, time_interaction={1: lambda t: t**0.5})
    assert lam.parameter_names[-1] == "coef_1:g(t)"


def test_the_robust_covariance_clusters_the_episodes_of_each_row():
    x, c, Z = _data(7)
    model = CoxPH.fit(x, Z, c=c, time_interaction={0: "log"})
    by_row = model.robust_covariance()
    rows = model.tvc_subject_ids
    assert_allclose(by_row, model.robust_covariance(rows))


def test_pickles_with_a_named_function():
    x, c, Z = _data(8)
    model = CoxPH.fit(x, Z, c=c, time_interaction={0: "log"})
    again = pickle.loads(pickle.dumps(model))
    assert_allclose(again.sf([0.5, 1.0], [1, 0]), model.sf([0.5, 1.0], [1, 0]))


def test_partial_effects_plot_takes_the_covariates():
    import matplotlib.pyplot as plt

    x, c, Z = _data(9)
    model = CoxPH.fit(x, Z, c=c, time_interaction={0: "log"})
    _, ax = plt.subplots()
    ax = model.plot_partial_effects(0, [0.0, 1.0], ax=ax)
    assert len(ax.get_lines()) == 2
    _, ax = plt.subplots()
    ax = model.time_varying_effect(0).plot(ax)
    assert ax.get_lines()
    plt.close("all")


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda m: m.phi([1.0, 0.0]), "hazard ratio changes with time"),
        (lambda m: m.concordance(), "order of risk changes"),
        (lambda m: m.sf_tvc([1.0], [[1.0, 0.0]], xl=[0.0]), "sf\\(x, Z\\)"),
        (lambda m: m.to_dict(), "not supported yet"),
        (lambda m: m.cb([1.0], [1.0, 0.0]), "time_varying_effect"),
        (lambda m: m.diagnose(), "time_varying_effect"),
    ],
)
def test_what_has_no_meaning_is_refused_plainly(call, match):
    x, c, Z = _data(10)
    model = CoxPH.fit(x, Z, c=c, time_interaction={0: "log"})
    with pytest.raises(ValueError, match=match):
        call(model)


@pytest.mark.parametrize(
    "spec, error, match",
    [
        ("log", TypeError, "must be a dict"),
        ({5: "log"}, ValueError, "has 2 column"),
        ({"dose": "log"}, ValueError, "no column names"),
        ({0: "sqrt"}, ValueError, "must be one of"),
        ({0: 3}, TypeError, "callable"),
        ({0: ["log", "log"]}, ValueError, "twice"),
        ({True: "log"}, TypeError, "by position"),
    ],
)
def test_a_bad_time_interaction_is_refused(spec, error, match):
    x, c, Z = _data(11)
    with pytest.raises(error, match=match):
        CoxPH.fit(x, Z, c=c, time_interaction=spec)


def test_log_of_an_event_at_zero_is_refused_naming_a_fix():
    x, c, Z = _data(12)
    x[0], c[0], Z[0, 0] = 0.0, 0, 1.0
    with pytest.raises(ValueError, match=r"np\.log\(t \+ 1\)"):
        CoxPH.fit(x, Z, c=c, time_interaction={0: "log"})
    # Shifted, it fits
    model = CoxPH.fit(x, Z, c=c, time_interaction={0: lambda t: np.log(t + 1)})
    assert np.all(np.isfinite(model.params))
