"""Panel-count data: PanelCountMCF, PanelCountRegression, and the parametric
intensity models fitted to counts between inspections."""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

import surpyval as sp
from surpyval.recurrent import (
    CrowAMSAA,
    NonParametricCounting,
    PanelCountMCF,
    PanelCountRegression,
    ProportionalIntensityNHPP,
)
from surpyval.recurrent import panel_count as pc

X = [2, 5, 9, 3, 8, 4, 6, 10, 3, 7]
N = [1, 0, 2, 1, 2, 0, 2, 1, 0, 1]
I = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
Z = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]


def _events(rng, units, end, scale=1.0, shape=1.5, frailty_var=0.0):
    """Event times of power-law processes (mean ``(t / scale) ** shape``),
    with a gamma frailty of variance ``frailty_var`` per unit."""
    out = []
    for _ in range(units):
        f = rng.gamma(1 / frailty_var, frailty_var) if frailty_var else 1.0
        arrivals = np.cumsum(rng.exponential(size=200)) / f
        t = scale * arrivals ** (1 / shape)
        out.append(t[t <= end])
    return out


def _panel(events, inspections):
    """Rows (unit, time, count since the previous inspection)."""
    rows = []
    for k, (ev, ins) in enumerate(zip(events, inspections)):
        cum = np.searchsorted(ev, ins, side="right")
        rows += [(k, t, c) for t, c in zip(ins, np.diff(np.r_[0, cum]))]
    u, t, c = map(np.array, zip(*rows))
    return t.astype(float), c.astype(float), u


# --- the two reductions that are exact --------------------------------------


@pytest.mark.parametrize("how", ["pseudo", "MLE"])
def test_common_inspection_times_give_the_mean_count(how):
    # Every unit inspected at the same times: both estimators are the
    # mean cumulative count at each time.
    rng = np.random.default_rng(1)
    events = _events(rng, 40, 10)
    times = np.array([1.0, 2.5, 4.0, 7.0, 10.0])
    x, n, i = _panel(events, [times] * 40)
    model = PanelCountMCF.fit(x, n, i=i, how=how)
    expected = np.array(
        [np.mean([np.sum(e <= t) for e in events]) for t in times]
    )
    np.testing.assert_allclose(model.mcf(times), expected, rtol=1e-9)


def test_likelihood_estimate_on_a_common_grid_is_the_nelson_aalen_mcf():
    # Event times rounded up onto a grid every unit is inspected on, units
    # leaving at different grid points: the likelihood estimate is the
    # Nelson-Aalen MCF of the rounded times (each cell's increment is its
    # events over the units still observed).
    rng = np.random.default_rng(2)
    grid = np.round(np.arange(0.1, 10.0001, 0.1), 10)
    ends = rng.choice(grid[20:], 60)
    events = [
        np.ceil(e[e <= end] * 10 - 1e-9) / 10
        for e, end in zip(_events(rng, 60, 10), ends)
    ]
    x, n, i = _panel(events, [grid[grid <= end] for end in ends])
    model = PanelCountMCF.fit(x, n, i=i)
    xs, ids, cs = [], [], []
    for k, (e, end) in enumerate(zip(events, ends)):
        xs += list(e) + [end]
        ids += [k] * (len(e) + 1)
        cs += [0] * len(e) + [1]
    mcf = NonParametricCounting.fit(xs, i=ids, c=cs)
    np.testing.assert_allclose(model.mcf(grid), mcf.mcf(grid), rtol=1e-8)


@pytest.mark.parametrize("how", ["pseudo", "MLE"])
def test_fine_own_inspections_approach_the_mcf(how):
    # Each unit inspected at its own times, about every 0.05: the panel
    # estimate is close to the MCF of the exact event times (tolerance
    # fixed before the run: 5% of the MCF at the end of follow-up).
    rng = np.random.default_rng(3)
    events = _events(rng, 150, 10)
    inspections = [
        np.r_[np.cumsum(rng.uniform(0.02, 0.08, 400)), 10.0] for _ in events
    ]
    inspections = [np.unique(np.minimum(t, 10.0)) for t in inspections]
    x, n, i = _panel(events, inspections)
    model = PanelCountMCF.fit(x, n, i=i, how=how)
    xs = np.concatenate([np.r_[e, 10.0] for e in events])
    ids = np.concatenate([[k] * (len(e) + 1) for k, e in enumerate(events)])
    cs = np.concatenate([np.r_[np.zeros(len(e)), 1] for e in events])
    mcf = NonParametricCounting.fit(xs, i=ids, c=cs)
    t = np.array([1.0, 3.0, 5.0, 8.0, 9.9])
    assert np.max(np.abs(model.mcf(t) - mcf.mcf(t))) < 0.05 * mcf.mcf(10.0)


# --- the likelihood estimate is the maximum ---------------------------------


def _small_panel(seed):
    rng = np.random.default_rng(seed)
    events = _events(rng, 12, 6, frailty_var=0.5)
    inspections = [
        np.sort(rng.uniform(0.2, 6, rng.integers(2, 5))) for _ in events
    ]
    return _panel(events, inspections)


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_likelihood_mcf_agrees_with_a_general_optimiser(seed):
    x, n, i = _small_panel(seed)
    model = PanelCountMCF.fit(x, n, i=i)
    panel = pc._panel_data(x, i, n)
    parts = pc._likelihood_parts(panel)
    ones = np.ones(panel.n_units)

    def negll(u):
        return -pc._full_loglik(panel, np.exp(u), ones, parts)

    delta, ok = pc._full_mean(panel, ones)
    assert ok
    np.testing.assert_allclose(model.mcf_hat[panel.rel], np.cumsum(delta))
    ll_best = pc._full_loglik(panel, delta, ones, parts)
    res = minimize(
        negll,
        np.zeros(panel.r_L),
        method="Nelder-Mead",
        options={"maxiter": 200000, "xatol": 1e-10, "fatol": 1e-12},
    )
    # No point the general search finds is higher (to rounding)
    assert -res.fun <= ll_best + 1e-8


def test_likelihood_mcf_does_not_depend_on_the_start():
    x, n, i = _small_panel(7)
    panel = pc._panel_data(x, i, n)
    ones = np.ones(panel.n_units)
    a, ok_a = pc._full_mean(panel, ones)
    rng = np.random.default_rng(0)
    b, ok_b = pc._full_mean(panel, ones, rng.uniform(0.01, 2, a.size))
    assert ok_a and ok_b
    np.testing.assert_allclose(np.cumsum(a), np.cumsum(b), atol=1e-7)


def test_mcf_is_non_decreasing_and_a_step_function():
    model = PanelCountMCF.fit(X, N, i=I)
    assert np.all(np.diff(model.mcf_hat) >= 0)
    assert model.mcf(2.5) == model.mcf(2)
    np.testing.assert_allclose(model.mcf(0.5), 0.0)
    assert np.isnan(model.mcf([-1, 11])).all()
    lin = model.mcf([1.0], interp="linear")
    np.testing.assert_allclose(lin, model.mcf(2) / 2)


# --- regression -------------------------------------------------------------


@pytest.mark.parametrize("how", ["pseudo", "MLE"])
def test_regression_recovers_the_effect_under_a_mixed_poisson(how):
    # A gamma frailty (variance 0.5): the counts are not Poisson, the mean
    # model holds. Effect 0.7; 400 units; tolerance fixed in advance.
    rng = np.random.default_rng(8)
    z = rng.integers(0, 2, 400)
    events = []
    for zk in z:
        events += _events(
            rng, 1, 8, scale=np.exp(-0.7 * zk / 1.5), frailty_var=0.5
        )
    inspections = [np.sort(rng.uniform(0.3, 8, rng.integers(1, 6))) for _ in z]
    x, n, i = _panel(events, inspections)
    model = PanelCountRegression.fit(x, z[i], n, i=i, how=how)
    assert model.converged
    assert abs(model.params[0] - 0.7) < 0.15


def test_regression_mcf_scales_the_baseline():
    model = PanelCountRegression.fit(X, Z, N, i=I)
    np.testing.assert_allclose(
        model.mcf([4, 9], [1]),
        model.baseline_mcf([4, 9]) * np.exp(model.params[0]),
    )


def test_regression_init_does_not_change_the_answer():
    a = PanelCountRegression.fit(X, Z, N, i=I, how="MLE")
    b = PanelCountRegression.fit(X, Z, N, i=I, how="MLE", init=[2.0])
    np.testing.assert_allclose(a.params, b.params, atol=1e-7)


# Units with covariate 0 counted no events: the coefficient runs to +inf.
SEP_N = [1, 0, 2, 0, 0, 0, 2, 1, 0, 0]
# Separation in time: unit 2 (Z = 0) has no events by time 1, when unit 1
# (Z = 1) has its one, and nothing later is compared with unit 1.
TIME_SEP = ([1, 1, 2], [1, 0, 0], [1, 0, 1], [1, 2, 2])


def _no_maximum(caught):
    return [w for w in caught if "No finite maximum" in str(w.message)]


@pytest.mark.parametrize("how", ["pseudo", "MLE"])
@pytest.mark.parametrize(
    "data",
    [(X, [1, 1, 1, 0, 0, 1, 1, 1, 0, 0], SEP_N, I), TIME_SEP],
    ids=["level-with-no-events", "in-time"],
)
def test_regression_with_no_finite_maximum_warns_and_is_not_converged(
    how, data
):
    x, z, n, i = data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PanelCountRegression.fit(x, z, n, i=i, how=how)
    found = _no_maximum(caught)
    assert len(found) == 1
    message = str(found[0].message)
    assert "covariate column(s) 0" in message and "+inf" in message
    # One warning: not also the search's own
    assert not [
        w for w in caught if w.category is pc.PanelCountConvergenceWarning
    ]
    assert not model.converged
    assert model.params[0] > 5


def test_runoff_direction_finds_a_combination():
    # Neither covariate separates alone; z1 - z0 does (units with events
    # have z1 - z0 = 1, the rest 0 or -1).
    z = np.array([[0, 1], [1, 2], [2, 3], [1, 1], [0, 0], [3, 2]], float)
    unit = np.repeat(np.arange(6), 2)
    x = np.tile([2.0, 5.0], 6)
    n = np.where(np.repeat(z[:, 1] - z[:, 0] == 1, 2), 1.0, 0.0)
    panel = pc._panel_data(x, unit, n, z[unit])
    for k in range(2):
        alone = pc._panel_data(x, unit, n, z[unit][:, [k]])
        assert pc._runoff_direction(alone) is None
    d = pc._runoff_direction(panel)
    assert d is not None
    np.testing.assert_allclose(d / d[1], [-1.0, 1.0], atol=1e-6)


def test_a_large_finite_estimate_is_not_a_runoff():
    # The two units with events have nearly equal covariates and the
    # later of them the lower: the mean function cannot fall, so the
    # profile peaks at a large but finite coefficient (about -27.7).
    x = [7, 5, 6, 1, 3, 6, 1, 3, 4]
    i = [0, 1, 1, 2, 2, 2, 3, 3, 3]
    z = [-0.625, -0.6, -0.6, -0.087, -0.087, -0.087, -1.077, -1.077, -1.077]
    n = [2, 0, 1, 0, 0, 0, 0, 0, 0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PanelCountRegression.fit(x, z, n, i=i)
    assert not _no_maximum(caught)
    assert model.converged
    assert -30 < model.params[0] < -25


@pytest.mark.parametrize(
    "how, ordinary, fixture",
    [
        ("pseudo", 0.18232155679395265, -0.30770781375147804),
        ("MLE", 0.3709192553360213, -0.2946531871780949),
    ],
)
def test_the_separation_check_leaves_an_ordinary_fit_unchanged(
    how, ordinary, fixture
):
    # The coefficients the fit gave before the check was added
    from surpyval.tests.conformance.registry_fixtures import panel_count_data

    d = panel_count_data(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = PanelCountRegression.fit(X, Z, N, i=I, how=how)
        b = PanelCountRegression.fit(d["x"], d["Z"], d["n"], i=d["i"], how=how)
    assert a.converged and b.converged
    assert a.params.tolist() == [ordinary]
    assert b.params.tolist() == [fixture]


def test_a_saved_separated_fit_reloads_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PanelCountRegression.fit(
            X, [1, 1, 1, 0, 0, 1, 1, 1, 0, 0], SEP_N, i=I
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = PanelCountRegression.from_dict(model.to_dict())
    assert not again.converged


def test_regression_is_invariant_to_shifting_a_covariate():
    a = PanelCountRegression.fit(X, Z, N, i=I)
    b = PanelCountRegression.fit(X, np.asarray(Z) + 5.0, N, i=I)
    np.testing.assert_allclose(a.params, b.params, rtol=1e-7)
    np.testing.assert_allclose(a.mcf([9], [1]), b.mcf([9], [6]), rtol=1e-7)


# --- bounds -----------------------------------------------------------------


def test_mcf_cb_is_reproducible_and_brackets_the_estimate():
    model = PanelCountMCF.fit(X, N, i=I)
    cb = model.mcf_cb([3, 7], n_boot=200, random_state=1)
    assert cb.shape == (2, 2)
    np.testing.assert_array_equal(
        cb, model.mcf_cb([3, 7], n_boot=200, random_state=1)
    )
    est = model.mcf([3, 7])
    assert np.all(cb[:, 0] <= est) and np.all(est <= cb[:, 1])
    lower = model.mcf_cb([3, 7], bound="lower", n_boot=200, random_state=1)
    assert lower.shape == (2,)


def test_param_cb_and_standard_errors():
    model = PanelCountRegression.fit(X, Z, N, i=I)
    assert model.parameter_names == ["coef_0"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cb = model.param_cb("coef_0", n_boot=200, random_state=2)
        normal = model.param_cb(
            "coef_0", method="normal", n_boot=200, random_state=2
        )
        se = model.standard_errors(n_boot=200, random_state=2)
        betas, _ = model.bootstrap(200, random_state=2)
        mcb = model.mcf_cb([5], [1], n_boot=200, random_state=2)
    assert cb.shape == (2,) and normal.shape == (2,)
    np.testing.assert_allclose(se, np.nanstd(betas, axis=0, ddof=1))
    np.testing.assert_allclose(
        normal, model.params[0] + np.array([-1, 1]) * 1.959963984540054 * se
    )
    assert mcb.shape == (1, 2)


def test_column_names_from_a_data_frame():
    df = pd.DataFrame({"t": X, "n": N, "u": I, "treated": Z})
    model = PanelCountRegression.fit_from_df(
        df, "t", "treated", "n", i_col="u"
    )
    assert model.parameter_names == ["treated"]


def test_bootstrap_refuses_bad_n_boot():
    model = PanelCountMCF.fit(X, N, i=I)
    with pytest.raises(ValueError, match="n_boot"):
        model.mcf_cb([3], n_boot=1)


# --- refusals ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"x": [0, 1], "n": [1, 1]}, "positive"),
        ({"x": [1, 2], "n": [1, -1]}, "not negative"),
        ({"x": [1, 2], "n": None}, "n is required"),
        ({"x": [1, 1], "n": [1, 1]}, "inspected twice"),
        ({"x": [1, np.nan], "n": [1, 1]}, "finite"),
        ({"x": [1, 2], "n": [1]}, "one count per row"),
    ],
)
def test_mcf_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        PanelCountMCF.fit(kw["x"], kw["n"])


def test_options_are_checked():
    with pytest.raises(ValueError, match="how"):
        PanelCountMCF.fit(X, N, i=I, how="ml")
    with pytest.raises(ValueError, match="interp"):
        PanelCountMCF.fit(X, N, i=I).mcf(3, interp="spline")


def test_regression_refusals():
    with pytest.raises(ValueError, match="No covariate in Z"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PanelCountRegression.fit(X, np.ones(10), N, i=I)
    changing = list(Z)
    changing[1] = 1
    with pytest.raises(ValueError, match="change between its rows"):
        PanelCountRegression.fit(X, changing, N, i=I)
    with pytest.raises(ValueError, match="Z is required"):
        PanelCountRegression.fit(X, None, N, i=I)
    with pytest.raises(ValueError, match="No events"):
        PanelCountRegression.fit(X, Z, np.zeros(10), i=I)
    with pytest.raises(ValueError, match="one value per covariate"):
        PanelCountRegression.fit(X, Z, N, i=I).mcf(3, [1, 2])


def test_a_collinear_column_is_aliased():
    both = np.c_[Z, 2 * np.asarray(Z)]
    with pytest.warns(UserWarning, match="cannot be estimated"):
        model = PanelCountRegression.fit(X, both, N, i=I)
    np.testing.assert_array_equal(model.aliased, [1])
    assert np.isnan(model.params[1])
    np.testing.assert_allclose(
        model.params[0], PanelCountRegression.fit(X, Z, N, i=I).params[0]
    )


# --- data frames and serialisation ------------------------------------------


def test_fit_from_df_is_fit():
    df = pd.DataFrame({"t": X, "n": N, "u": I, "z": Z})
    a = PanelCountMCF.fit_from_df(df, "t", "n", i_col="u")
    np.testing.assert_array_equal(
        a.mcf_hat, PanelCountMCF.fit(X, N, i=I).mcf_hat
    )
    b = PanelCountRegression.fit_from_df(df, "t", ["z"], "n", i_col="u")
    np.testing.assert_array_equal(
        b.params, PanelCountRegression.fit(X, Z, N, i=I).params
    )


@pytest.mark.parametrize("how", ["pseudo", "MLE"])
def test_round_trips(how):
    for model in (
        PanelCountMCF.fit(X, N, i=I, how=how, unit="hours"),
        PanelCountRegression.fit(X, Z, N, i=I, how=how),
    ):
        text = model.to_json()
        back = sp.from_json(text)
        assert type(back) is type(model)
        assert back.unit == model.unit
        assert back.how == how
        np.testing.assert_allclose(
            np.asarray(json.loads(text)["x"]), np.asarray(X, float)
        )
        if isinstance(model, PanelCountMCF):
            np.testing.assert_array_equal(back.mcf_hat, model.mcf_hat)
        else:
            np.testing.assert_allclose(back.params, model.params, atol=1e-9)
        again = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(again.x, model.x)


def test_string_unit_ids_round_trip():
    ids = ["a", "a", "a", "b", "b", "c", "c", "c", "d", "d"]
    model = PanelCountMCF.fit(X, N, i=ids)
    back = PanelCountMCF.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_array_equal(back.mcf_hat, model.mcf_hat)


def test_repr():
    assert "4 units" in repr(PanelCountMCF.fit(X, N, i=I))
    assert "Coefficients" in repr(PanelCountRegression.fit(X, Z, N, i=I))


# --- parametric intensity models on panel counts ----------------------------


def _parametric_rows(seed, units=300, effect=0.0):
    """Panel counts of a Crow-AMSAA process (alpha 2, beta 1.5) as interval
    counts: x = [previous inspection, inspection], c = 2, n the count (0
    where none were found)."""
    rng = np.random.default_rng(seed)
    rows, zs = [], []
    for k in range(units):
        z = int(rng.integers(0, 2))
        ins = np.sort(rng.uniform(0.5, 10, rng.integers(2, 6)))
        prev = 0.0
        for t in ins:
            mean = ((t / 2) ** 1.5 - (prev / 2) ** 1.5) * np.exp(effect * z)
            rows.append((k, prev, t, rng.poisson(mean)))
            zs.append(z)
            prev = t
    r = np.array(rows, dtype=float)
    return r[:, 1:3], r[:, 0], r[:, 3], np.array(zs, float)


def test_crow_amsaa_fits_panel_counts_with_zero_counts():
    x, i, n, _ = _parametric_rows(9)
    assert np.any(n == 0)
    model = CrowAMSAA.fit(x, i=i, c=np.full(len(n), 2), n=n)
    # The likelihood is the Poisson likelihood of the increments
    cif = model.cif
    mean = cif(x[:, 1]) - cif(x[:, 0])
    ll = np.sum(n * np.log(mean) - mean) - np.sum(
        [np.sum(np.log(np.arange(1, k + 1))) for k in n]
    )
    np.testing.assert_allclose(-model.neg_ll(), ll, rtol=1e-10)
    np.testing.assert_allclose(model.params, [2.0, 1.5], rtol=0.1)


def test_proportional_intensity_fits_panel_counts():
    x, i, n, z = _parametric_rows(10, effect=0.6)
    model = ProportionalIntensityNHPP.fit(
        x, Z=z, i=i, c=np.full(len(n), 2), n=n, baseline=CrowAMSAA
    )
    assert abs(model.coeffs[0] - 0.6) < 0.15


def test_zero_counts_only_on_counted_rows():
    with pytest.raises(ValueError, match="strictly positive"):
        CrowAMSAA.fit([1.0, 2.0], i=[1, 1], c=[0, 1], n=[0, 1])
    with pytest.raises(ValueError, match="no events"):
        CrowAMSAA.fit([[0.0, 1.0], [1.0, 2.0]], i=[1, 1], c=[2, 2], n=[0, 0])
