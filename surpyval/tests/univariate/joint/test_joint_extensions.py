"""The joint model's separate hazard and path covariates, its interval
censored and truncated items, and its dynamic prediction (#891).

The fleet is ``_fleet.simulate_split``'s: the pumps of ``_fleet.py`` with
two 0/1 covariates, ``site`` in the hazard only and ``load`` in the line's
rate only, optionally inspected every 90 days (failures interval or left
censored). The checks: the fit names and recovers each part's
covariates; the likelihood of every kind of item -- failed, running,
left and interval censored, left and right truncated -- agrees with an
independent dense-grid integral over the random effects; the dynamic
prediction agrees with the model's own array prediction, its hazard with
the derivative of its survival, its quantiles invert it, and its marker
with a brute-force posterior mean; and the old forms (one covariate set,
saved models) are unchanged.
"""

import functools
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval
from surpyval import ItemLifetime, JointModel, SurpyvalData, WeibullJointPH
from surpyval.tests.univariate.joint._fleet import (
    CENTRE,
    TRUTH,
    fleet,
    simulate_split,
)
from surpyval.univariate.joint import DynamicPrediction
from surpyval.univariate.joint.joint_ph import search_space

#: The split fleet's truth: ``site`` only in the hazard, ``load`` only in
#: the line's rate (0.6 mm/s a year more).
GAMMA, DELTA_B = 0.5, 0.6 / 365
SPLIT = dict(hazard_covariates=["site"], path_covariates={"b": ["load"]})


def _truth(model, **extra):
    """The truth in the model's parameterisation (its ``alpha`` at its
    marker reference, the truth's at ``CENTRE``)."""
    out = dict(TRUTH, site=GAMMA, b_load=DELTA_B, **extra)
    shift = model.marker_reference - CENTRE
    out["alpha"] = TRUTH["alpha"] * np.exp(
        -TRUTH["marker_effect"] * shift / TRUTH["beta"]
    )
    return np.array([out[name] for name in model.parameter_names])


@functools.cache
def _split_data(inspect=None):
    items, _, _ = simulate_split(
        300, seed=21 if inspect is None else 22, inspect=inspect
    )
    return SurpyvalData.from_items(items)


@functools.cache
def _split(inspect=None):
    return WeibullJointPH.fit(_split_data(inspect), **SPLIT)


@functools.cache
def _small():
    return WeibullJointPH.fit(fleet(80, seed=4))


# -- separate covariates ----------------------------------------------------


def test_each_part_has_its_own_covariates():
    model = _split()
    assert model.maximum == "verified"
    assert model.parameter_names == [
        "a",
        "b",
        "b_load",
        "sd_a",
        "sd_b",
        "corr_a_b",
        "sd_noise",
        "alpha",
        "beta",
        "site",
        "marker_effect",
    ]
    assert model.hazard_covariates == ["site"]
    assert model.path_covariates == {"a": [], "b": ["load"]}
    z = (model.params - _truth(model)) / model.standard_errors()
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))


@pytest.mark.parametrize(
    "parts, names",
    [
        # hazard only: no shifts of the path
        (
            dict(hazard_covariates=["site", "load"], path_covariates=[]),
            ["a", "b", "site", "load"],
        ),
        # path only: no hazard coefficients
        (
            dict(hazard_covariates=[], path_covariates=["site", "load"]),
            ["a", "a_site", "a_load", "b", "b_site", "b_load"],
        ),
        # both, each its own
        (
            dict(hazard_covariates="site", path_covariates="load"),
            ["a", "a_load", "b", "b_load", "site"],
        ),
    ],
)
def test_hazard_only_path_only_and_both(parts, names):
    model = WeibullJointPH.fit(_split_data(), **parts)
    assert model.maximum == "verified"
    shown = [
        n
        for n in model.parameter_names
        if n not in ("sd_a", "sd_b", "corr_a_b", "sd_noise", "alpha", "beta")
        and n != "marker_effect"
    ]
    assert shown == names
    table = model.summary()
    assert list(table.index.get_level_values("name")) == list(
        model.parameter_names
    )
    # the prediction takes every covariate, by name
    sf = model.sf(400, {"site": 1.0, "load": 0.0}, {0: 3.0, 60: 3.4}, 100)
    assert 0 < sf < 1


def test_the_default_is_every_covariate_in_both_parts():
    data = _split_data()
    plain = WeibullJointPH.fit(data)
    explicit = WeibullJointPH.fit(
        data,
        hazard_covariates=["site", "load"],
        path_covariates={"a": ["site", "load"], "b": ["load", "site"]},
    )
    assert plain.parameter_names == explicit.parameter_names
    np.testing.assert_allclose(plain.params, explicit.params, rtol=1e-6)
    # saved as before: no parts
    assert plain.path_covariates is None and plain.hazard_covariates is None
    assert explicit.path_covariates is None
    assert "path_covariates" in plain.to_dict()


def test_a_model_saved_before_the_parts_loads_and_predicts_the_same():
    model = _small()
    old = model.to_dict()
    old.pop("path_covariates")
    old.pop("hazard_covariates")
    restored = JointModel.from_dict(old)
    item = dict(markers={0: 3.0, 90: 3.9}, given=120)
    assert restored.sf(400, **item) == model.sf(400, **item)


def test_split_models_round_trip():
    model = _split()
    restored = surpyval.from_dict(model.to_dict())
    assert restored.parameter_names == model.parameter_names
    assert restored.path_covariates == model.path_covariates
    Z = {"site": 1.0, "load": 1.0}
    item = dict(markers={0: 3.0, 90: 3.9}, given=120)
    np.testing.assert_allclose(
        restored.cb([200, 500], Z, **item), model.cb([200, 500], Z, **item)
    )


def test_the_parts_work_from_arrays_and_dataframes():
    data = _split_data()
    markers = [data.markers[i] for i in data.ids]
    model = _split()
    arrays = WeibullJointPH.fit(
        data.x,
        c=data.c,
        Z=pd.DataFrame(data.Z, columns=data.covariate_names),
        markers=markers,
        **SPLIT,
    )
    np.testing.assert_allclose(arrays.params, model.params, rtol=1e-8)
    df = pd.DataFrame(
        {
            "age": data.x,
            "c": data.c,
            "site": data.Z[:, 0],
            "load": data.Z[:, 1],
            "readings": markers,
        }
    )
    frame = WeibullJointPH.fit_from_df(
        df,
        x_col="age",
        c_col="c",
        markers_col="readings",
        Z_cols=["site", "load"],
        **SPLIT,
    )
    np.testing.assert_allclose(frame.params, model.params, rtol=1e-8)
    # unnamed covariates are coef_0, coef_1, ...
    unnamed = WeibullJointPH.fit(
        data.x,
        c=data.c,
        Z=np.asarray(data.Z),
        markers=markers,
        hazard_covariates=["coef_0"],
        path_covariates={"b": ["coef_1"]},
    )
    assert "b_coef_1" in unnamed.parameter_names
    assert "coef_0" in unnamed.parameter_names


@pytest.mark.parametrize(
    "parts, match",
    [
        (dict(hazard_covariates=["heat"]), "not covariates of the data"),
        (dict(path_covariates={"c": ["site"]}), "path parameters"),
        (
            dict(hazard_covariates=["site"], path_covariates=[]),
            "enter neither the hazard nor the path",
        ),
        (dict(hazard_covariates=3), "list of covariate names"),
    ],
)
def test_parts_that_cannot_be_fitted_are_refused(parts, match):
    with pytest.raises(ValueError, match=match):
        WeibullJointPH.fit(_split_data(), **parts)


# -- interval censoring and truncation ----------------------------------------


#: Items of every kind: failed, running, interval censored (one from 0),
#: left truncated and failed, running and right truncated, interval
#: censored inside left and right truncation, left censored.
KINDS = dict(
    x=np.array(
        [
            [300.0, 300.0],
            [500.0, 500.0],
            [0.0, 90.0],
            [180.0, 270.0],
            [400.0, 400.0],
            [350.0, 350.0],
            [200.0, 380.0],
            [600.0, 600.0],
        ]
    ),
    c=np.array([0, 1, 2, 2, 0, 0, 2, -1]),
    n=np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0]),
    t=np.array(
        [
            [-np.inf, np.inf],
            [-np.inf, np.inf],
            [-np.inf, np.inf],
            [-np.inf, np.inf],
            [100.0, np.inf],
            [-np.inf, 450.0],
            [60.0, 500.0],
            [-np.inf, np.inf],
        ]
    ),
    Z=np.array(
        [[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0]],
        dtype=float,
    ),
    markers=[
        {0: 2.9, 60: 3.3, 120: 3.1, 180: 3.8, 240: 4.0, 300: 4.4},
        {0: 3.5, 100: 3.9, 300: 4.9, 450: 5.6},
        {},
        {0: 2.0, 90: 2.6, 180: 3.1},
        {100: 3.0, 160: 3.3, 250: 3.9, 400: 4.6},
        {0: 3.0, 200: 3.4},
        {60: 3.0, 120: 3.2, 200: 3.4},
        {0: 2.5, 300: 3.0, 550: 3.9},
    ],
)


def _brute_force_neg_ll(model, data):
    """The negative log-likelihood of ``data`` by a dense grid over the
    random effects (161 points a dimension, 7 sds each way) and Simpson's
    rule in age (2001 points), from the model's parameters alone: no
    quadrature or code of the model's."""
    p = dict(zip(model.parameter_names, model.params))
    v0, v1, r = p["sd_a"] ** 2, p["sd_b"] ** 2, p["corr_a_b"]
    D = np.array([[v0, r * np.sqrt(v0 * v1)], [r * np.sqrt(v0 * v1), v1]])
    g0 = np.linspace(-7, 7, 161) * np.sqrt(v0)
    g1 = np.linspace(-7, 7, 161) * np.sqrt(v1)
    B0, B1 = (a.ravel() for a in np.meshgrid(g0, g1, indexing="ij"))
    b = np.stack([B0, B1], 1)
    log_prior = (
        -0.5 * np.einsum("ni,ij,nj->n", b, np.linalg.inv(D), b)
        - np.log(2 * np.pi)
        - 0.5 * np.log(np.linalg.det(D))
    )
    cell = (g0[1] - g0[0]) * (g1[1] - g1[0])
    names = model.feature_names

    def log_sum(terms):
        top = terms.max()
        return top + np.log(np.sum(np.exp(terms - top)) * cell)

    total = 0.0
    for k, (lo, hi) in enumerate(data["x"]):
        z = dict(zip(names, data["Z"][k]))
        m0 = p["a"] + sum(p.get(f"a_{n}", 0.0) * z[n] for n in names) + B0
        m1 = p["b"] + sum(p.get(f"b_{n}", 0.0) * z[n] for n in names) + B1
        lin = sum(p.get(n, 0.0) * z[n] for n in names)

        def hazard(t, m0=m0, m1=m1, lin=lin):
            """At ages ``t`` ``(A,)``: ``(grid, A)``."""
            t = np.atleast_1d(t)[None, :]
            h0 = (p["beta"] / p["alpha"]) * (t / p["alpha"]) ** (p["beta"] - 1)
            m = m0[:, None] + m1[:, None] * t
            return h0 * np.exp(
                lin + p["marker_effect"] * (m - model.marker_reference)
            )

        def survival(t, hazard=hazard):
            if not np.isfinite(t):
                return np.zeros(B0.size)
            if t <= 0:
                return np.ones(B0.size)
            s = np.linspace(0, t, 2001)
            w = np.ones(s.size)
            w[1:-1:2], w[2:-1:2] = 4, 2
            return np.exp(-hazard(s) @ w * (s[1] - s[0]) / 3)

        readings = data["markers"][k]
        ages = np.array(list(readings), dtype=float)
        ys = np.array(list(readings.values()), dtype=float)
        resid = ys[None, :] - m0[:, None] - m1[:, None] * ages[None, :]
        sd = p["sd_noise"]
        log_y = -0.5 * np.sum(
            resid**2, axis=1
        ) / sd**2 - ages.size * np.log(np.sqrt(2 * np.pi) * sd)
        tl, tr = data["t"][k]
        entry = max(tl, 0.0)
        c = data["c"][k]
        if c == 0:
            life = np.log(hazard(lo)[:, 0] * survival(lo))
        elif c == 1:
            life = np.log(survival(lo))
        elif c == 2:
            life = np.log(survival(lo) - survival(hi))
        else:  # failed by hi, having reached its entry
            life = np.log(survival(entry) - survival(hi))
        ll = log_sum(log_y + log_prior + life)
        if np.isfinite(tl) or np.isfinite(tr):
            ll -= log_sum(np.log(survival(entry) - survival(tr)) + log_prior)
        total += data["n"][k] * ll
    return -total


def _with_params(model, **values):
    d = model.to_dict()
    for name, value in values.items():
        d["params"][model.parameter_names.index(name)] = value
    d.pop("covariance", None)
    return JointModel.from_dict(d)


@pytest.mark.parametrize("moved", [False, True], ids=["fitted", "moved"])
def test_every_kind_of_item_agrees_with_a_dense_grid_integral(moved):
    model = _split(90.0)
    if moved:
        # away from the maximum, where the nodes are placed afresh
        p = dict(zip(model.parameter_names, model.params))
        model = _with_params(
            model,
            beta=1.3,
            marker_effect=1.4,
            site=-0.3,
            b_load=0.0,
            sd_noise=0.4,
            corr_a_b=-0.5,
            alpha=0.7 * p["alpha"],
        )
    f, u, _ = search_space(model, **KINDS)
    # a path far from the readings underflows the survival on the grid
    with np.errstate(divide="ignore"):
        brute = _brute_force_neg_ll(model, KINDS)
    # 8 rows, 11 items: to within 1e-5 an item
    assert float(f(u)) == pytest.approx(brute, abs=1.1e-4)


def test_recovers_the_fleet_from_inspections():
    data = _split_data(90.0)
    counts = {flag: int(np.sum(data.c == flag)) for flag in (-1, 0, 1, 2)}
    assert counts[0] == 0 and counts[2] > 50 and counts[-1] > 0
    model = _split(90.0)
    assert model.maximum == "verified"
    assert model.n_events == counts[2] + counts[-1]
    z = (model.params - _truth(model)) / model.standard_errors()
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))


def test_intervals_from_arrays_and_dataframes():
    data = _split_data(90.0)
    markers = [data.markers[i] for i in data.ids]
    model = _split(90.0)
    # a left censored item is an interval from 0
    left = data.c == -1
    x = np.column_stack([np.where(left, 0.0, data.x[:, 0]), data.x[:, 1]])
    c = np.where(left, 2, data.c)
    arrays = WeibullJointPH.fit(
        x,
        c=c,
        Z=pd.DataFrame(data.Z, columns=["site", "load"]),
        markers=markers,
        **SPLIT,
    )
    np.testing.assert_allclose(arrays.params, model.params, rtol=1e-5)
    df = pd.DataFrame(
        {
            "lo": x[:, 0],
            "hi": x[:, 1],
            "c": c,
            "site": data.Z[:, 0],
            "load": data.Z[:, 1],
            "readings": markers,
        }
    )
    frame = WeibullJointPH.fit_from_df(
        df,
        xl_col="lo",
        xr_col="hi",
        c_col="c",
        markers_col="readings",
        Z_cols=["site", "load"],
        **SPLIT,
    )
    np.testing.assert_allclose(frame.params, model.params, rtol=1e-5)


def test_rows_with_two_ends_are_intervals_unless_c_says():
    from surpyval.univariate.joint.joint_ph import _records

    x = np.array([[10.0, 10.0], [20.0, 30.0], [0.0, 5.0]])
    records, _, _ = _records(x, None, None, None, None, None, None, [{}] * 3)
    assert [r.outcome for r in records] == [
        "observed",
        "interval_censored",
        "interval_censored",
    ]
    records, _, _ = _records(
        x, [1, 2, -1], None, None, None, None, None, [{}] * 3
    )
    assert [r.outcome for r in records] == [
        "right_censored",
        "interval_censored",
        "left_censored",
    ]
    assert records[2].value == 5.0


def test_right_truncated_items_are_fitted():
    # only the pumps that failed by 600 days are in the data
    items, _, _ = simulate_split(400, seed=23)
    kept = [
        it.clone(right_truncation=600.0)
        for it in items
        if it.outcome == "observed" and it.value <= 600.0
    ]
    model = WeibullJointPH.fit(SurpyvalData.from_items(kept), **SPLIT)
    assert model.maximum == "verified"
    assert np.all(np.isfinite(model.standard_errors()))


def test_an_interval_needs_both_ends():
    with pytest.raises(ValueError, match="needs its \\[lower, upper\\]"):
        WeibullJointPH.fit(
            [100.0, 200.0, 300.0, 400.0],
            c=[0, 2, 0, 0],
            markers=[{0: 1.0, 50: 2.0}] * 4,
        )


# -- dynamic prediction ------------------------------------------------------


ITEM = dict(markers={0: 3.0, 90: 3.9, 180: 4.4}, given=200.0)


def _pred():
    return _small().predict(**ITEM)


def test_dynamic_prediction_is_the_models_array_prediction():
    model, pred = _small(), _pred()
    assert isinstance(pred, DynamicPrediction)
    x = np.array([100.0, 200.0, 250.0, 400.0, 900.0])
    np.testing.assert_array_equal(pred.sf(x), model.sf(x, **ITEM))
    np.testing.assert_allclose(pred.ff(x), model.ff(x, **ITEM), atol=1e-15)
    np.testing.assert_array_equal(pred.cb(x), model.cb(x, **ITEM))
    np.testing.assert_array_equal(
        pred.cb(x, on="ff", bound="lower"),
        model.cb(x, **ITEM, on="ff", bound="lower"),
    )
    # an item stands for all three
    item = ItemLifetime("P", right_censored=200.0, markers=ITEM["markers"])
    np.testing.assert_array_equal(model.predict(item).sf(x), pred.sf(x))
    assert pred.age == 200.0 and pred.markers == ITEM["markers"]


def test_the_hazard_is_the_derivative_of_the_survival():
    pred = _pred()
    x = np.array([210.0, 300.0, 500.0, 800.0])
    h = 1e-3 * x
    numeric = -(np.log(pred.sf(x + h)) - np.log(pred.sf(x - h))) / (2 * h)
    np.testing.assert_allclose(pred.hf(x), numeric, rtol=1e-5)
    np.testing.assert_allclose(pred.df(x), pred.hf(x) * pred.sf(x))
    np.testing.assert_allclose(pred.Hf(x), -np.log(pred.sf(x)))
    # known running up to its age: no hazard, no density
    assert pred.hf(150.0) == 0 and pred.df(200.0) == 0
    assert pred.Hf(100.0) == 0


def test_quantiles_invert_the_survival():
    pred = _pred()
    p = np.array([0.0, 1e-6, 0.05, 0.5, 0.95, 1.0])
    q = pred.qf(p)
    assert q[0] == 200.0 and q[-1] == np.inf
    np.testing.assert_allclose(pred.ff(q[1:-1]), p[1:-1], rtol=1e-8)
    np.testing.assert_allclose(pred.remaining_life(p), q - 200.0)
    with pytest.warns(UserWarning):
        assert np.isnan(pred.qf(1.5))


def test_draws_follow_the_prediction():
    pred = _pred()
    draws = pred.random(400, random_state=7)
    np.testing.assert_array_equal(draws, pred.random(400, random_state=7))
    assert np.all(draws > pred.age)
    # their empirical distribution is the prediction's (Kolmogorov-
    # Smirnov at 1%: the critical value for 400 draws is 0.081)
    xs = np.sort(draws)
    ecdf = np.arange(1, xs.size + 1) / xs.size
    assert np.max(np.abs(ecdf - pred.ff(xs))) < 0.081


def test_cumulative_hazard_bounds_are_the_survivals():
    pred = _pred()
    x = np.array([300.0, 600.0])
    sf = pred.cb(x)
    np.testing.assert_allclose(pred.cb(x, on="Hf"), -np.log(sf[:, ::-1]))
    np.testing.assert_allclose(
        pred.cb(x, on="Hf", bound="upper"), -np.log(pred.cb(x, bound="lower"))
    )


def test_the_marker_is_the_posterior_mean_of_the_path():
    model, pred = _small(), _pred()
    p = dict(zip(model.parameter_names, model.params))
    # brute force: a dense grid over the line's level and rate, weighted
    # by the readings and the survival to the age now
    v0, v1, r = p["sd_a"] ** 2, p["sd_b"] ** 2, p["corr_a_b"]
    D = np.array([[v0, r * np.sqrt(v0 * v1)], [r * np.sqrt(v0 * v1), v1]])
    g0 = np.linspace(-7, 7, 161) * np.sqrt(v0)
    g1 = np.linspace(-7, 7, 161) * np.sqrt(v1)
    B0, B1 = (a.ravel() for a in np.meshgrid(g0, g1, indexing="ij"))
    b = np.stack([B0, B1], 1)
    m0, m1 = p["a"] + B0, p["b"] + B1
    ages = np.array(list(ITEM["markers"]), dtype=float)
    ys = np.array(list(ITEM["markers"].values()))
    resid = ys[None, :] - m0[:, None] - m1[:, None] * ages[None, :]
    s = np.linspace(0, ITEM["given"], 2001)
    h = (p["beta"] / p["alpha"]) * (s / p["alpha"]) ** (p["beta"] - 1)
    h = h[None, :] * np.exp(
        p["marker_effect"]
        * (m0[:, None] + m1[:, None] * s[None, :] - model.marker_reference)
    )
    w = np.ones(s.size)
    w[1:-1:2], w[2:-1:2] = 4, 2
    H = h @ w * (s[1] - s[0]) / 3
    log_w = (
        -0.5 * np.einsum("ni,ij,nj->n", b, np.linalg.inv(D), b)
        - 0.5 * np.sum(resid**2, axis=1) / p["sd_noise"] ** 2
        - H
    )
    weight = np.exp(log_w - log_w.max())
    weight /= weight.sum()
    x = np.array([0.0, 100.0, 200.0, 400.0])
    line = m0[:, None] + m1[:, None] * x[None, :]
    mean = weight @ line
    sd = np.sqrt(weight @ (line - mean) ** 2)
    np.testing.assert_allclose(pred.marker(x), mean, rtol=1e-6)
    band = pred.marker_interval(x)
    np.testing.assert_allclose(
        band[:, 1] - band[:, 0], 2 * 1.959964 * sd, rtol=1e-4
    )


def test_dynamic_prediction_shapes_nan_and_refusals():
    pred = _pred()
    assert pred.sf(300.0).shape == ()
    assert pred.hf([[300.0, 400.0]]).shape == (1, 2)
    assert pred.cb([300.0, 400.0]).shape == (2, 2)
    assert pred.marker_interval([0.0, 300.0]).shape == (2, 2)
    assert np.isnan(pred.sf(np.nan)) and np.isnan(pred.hf(np.nan))
    assert pred.sf(np.inf) == 0
    with pytest.raises(ValueError, match="'on'"):
        pred.cb(300.0, on="hf")
    with pytest.raises(ValueError, match="Give 'given'"):
        _small().predict(markers={0: 3.0})
    text = repr(pred)
    assert "Median life left" in text and "Readings            : 3" in text
    # a missing covariate: nan everywhere
    model = _split()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unknown = model.predict({"site": np.nan, "load": 1.0}, **ITEM)
    assert np.isnan(unknown.sf(300.0)) and np.isnan(unknown.qf(0.5))
    assert np.isnan(unknown.marker(100.0))


def test_readings_move_the_dynamic_prediction():
    model = _small()
    hot = model.predict(markers={0: 3.0, 90: 4.6, 180: 5.8}, given=200)
    cool = model.predict(markers={0: 3.0, 90: 3.1, 180: 3.2}, given=200)
    assert hot.remaining_life() < cool.remaining_life()
    assert hot.hf(300) > cool.hf(300)
    assert hot.marker(400) > cool.marker(400)
