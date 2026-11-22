"""Competing failure modes in the joint model: ``WeibullJointPH.fit``
with the failures' event types (``e``).

The fleet is ``_fleet.simulate_causes``'s: pumps failing by the first of
a bearing failure the vibration warns of (0.9 per mm/s) and a seal
failure it says nothing about. The checks: one mode is today's model
exactly; the likelihood equals the one-mode model's where the two must
agree (equal hazards for the modes; a mode the marker does not follow
factors out as a plain Weibull); every kind of item agrees with an
independent dense-grid integral; a fit recovers the truth; a shared
effect is nested in the modes' own; and the model summarises, saves,
loads and refuses as the rest of the package does.
"""

import functools
import json
import pickle

import numpy as np
import pandas as pd
import pytest

import surpyval
from surpyval import (
    JointPH,
    SurpyvalData,
    Weibull,
    WeibullJointAFT,
    WeibullJointPH,
)
from surpyval.tests.univariate.joint._fleet import (
    CAUSES,
    CENTRE,
    TRUTH,
    fleet,
    simulate_causes,
)
from surpyval.univariate.joint import JointModel
from surpyval.univariate.joint._likelihood import (
    JointArrays,
    Layout,
    Model,
    PathFunction,
    cause_thetas,
    log_cause_fails_between,
)
from surpyval.univariate.joint.joint_ph import search_space


def _data(n=150, seed=3, **kwargs):
    return SurpyvalData.from_items(simulate_causes(n, seed=seed, **kwargs)[0])


@functools.cache
def _model():
    return WeibullJointPH.fit(_data(), marker_reference=CENTRE)


@functools.cache
def _recovered():
    return WeibullJointPH.fit(_data(300, seed=0), marker_reference=CENTRE)


def _arrays(data):
    """The data's arrays, as the fit's array form takes them."""
    readings = data.markers
    out = dict(
        x=np.asarray(data.x, dtype=float),
        c=np.asarray(data.c),
        n=np.asarray(data.n, dtype=float),
        markers=[dict(readings.get(i, {})) for i in data.i],
    )
    if data.e is not None:
        out["e"] = np.asarray(data.e, dtype=object)
    return out


def _with_params(model, **values):
    d = model.to_dict()
    for name, value in values.items():
        d["params"][model.parameter_names.index(name)] = value
    d.pop("covariance", None)
    return JointModel.from_dict(d)


def _one_mode(model, alpha, beta, effect):
    """The model of one kind of failure with the same marker part."""
    d = model.to_dict()
    names = model.parameter_names
    marker = names.index("sd_noise") + 1
    d["parameter_names"] = names[:marker] + ["alpha", "beta", "marker_effect"]
    d["params"] = d["params"][:marker] + [alpha, beta, effect]
    for key in ("causes", "shared", "n_events_by_cause", "covariance"):
        d.pop(key, None)
    return JointModel.from_dict(d)


# -- one mode is today's model -----------------------------------------------


def test_one_mode_is_the_model_without_modes_exactly():
    data = fleet(80, seed=4)
    plain = WeibullJointPH.fit(data)
    arrays = _arrays(data)
    e = np.array([None if c == 1 else "wear" for c in arrays["c"]], object)
    one = WeibullJointPH.fit(**arrays, e=e)
    none = WeibullJointPH.fit(**arrays)
    assert one.causes is None and one.shared == ()
    assert one.parameter_names == none.parameter_names
    np.testing.assert_array_equal(one.params, none.params)
    np.testing.assert_array_equal(one.covariance(), none.covariance())
    assert one.neg_ll() == none.neg_ll()
    np.testing.assert_array_equal(one.params, plain.params)
    assert "causes" not in one.to_dict()


# -- the likelihood, where one-mode models give it ---------------------------


def test_modes_with_equal_hazards_are_the_pooled_model():
    # Two modes with one hazard each: the failure hazard is twice it, and
    # each failure's mode a fair coin, so the likelihood is the pooled
    # model's at twice the baseline, less log 2 a failure.
    model = _model()
    data = _arrays(_data())
    alpha, beta, effect = 1500.0, 1.7, 0.6
    same = _with_params(
        model,
        **{
            f"{cause}: {name}": value
            for cause in model.causes
            for name, value in (
                ("alpha", alpha),
                ("beta", beta),
                ("marker_effect", effect),
            )
        },
    )
    f, u, _ = search_space(same, **data)
    pooled = _one_mode(model, alpha * 2 ** (-1 / beta), beta, effect)
    plain = {k: v for k, v in data.items() if k != "e"}
    g, v, _ = search_space(pooled, **plain)
    failures = float(np.sum(data["n"][data["c"] == 0]))
    assert float(f(u)) == pytest.approx(
        float(g(v)) + failures * np.log(2.0), rel=1e-10
    )


def test_a_mode_the_marker_does_not_follow_factors_out():
    # With an effect of 0 a mode's hazard is the same on every path: its
    # failures and everyone's survival of it are a plain Weibull's
    # likelihood, and what is left is the one-mode joint model of the
    # other mode's failures, this mode's censored.
    model = _model()
    data = _arrays(_data())
    values = {
        "bearing: alpha": 1700.0,
        "bearing: beta": 2.2,
        "bearing: marker_effect": 0.7,
        "seal: alpha": 2500.0,
        "seal: beta": 1.4,
        "seal: marker_effect": 0.0,
    }
    f, u, _ = search_space(_with_params(model, **values), **data)
    seal = np.array([label == "seal" for label in data["e"]])
    bearing = _one_mode(model, 1700.0, 2.2, 0.7)
    censored = dict(
        x=data["x"],
        c=np.where(seal, 1, data["c"]),
        n=data["n"],
        markers=data["markers"],
    )
    g, v, _ = search_space(bearing, **censored)
    weibull = Weibull.from_params([2500.0, 1.4])
    seal_ll = np.sum(
        data["n"]
        * (
            np.where(seal, np.log(weibull.hf(data["x"])), 0.0)
            - weibull.Hf(data["x"])
        )
    )
    assert float(f(u)) == pytest.approx(float(g(v)) - seal_ll, rel=1e-10)


# -- every kind of item, against a dense grid --------------------------------

#: One of each kind of item with a failure mode: failed exactly, found
#: failed between two inspections and before the first, running; some
#: counted twice, some that entered late or are in the data because they
#: failed by an age.
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
            [90.0, 180.0],
        ]
    ),
    c=np.array([0, 1, 2, 2, 0, 0, 2, -1, 2]),
    n=np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0]),
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
            [-np.inf, np.inf],
        ]
    ),
    e=np.array(
        [
            "bearing",
            None,
            "seal",
            "bearing",
            "seal",
            "bearing",
            "seal",
            "bearing",
            "seal",
        ],
        dtype=object,
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
        {0: 3.8, 60: 4.1},
    ],
)


def _simpson(values, step):
    w = np.ones(values.shape[-1])
    w[1:-1:2], w[2:-1:2] = 4, 2
    return values @ w * step / 3


def _brute_force_neg_ll(model, data):
    """The negative log-likelihood of ``data`` by a dense grid over the
    random effects (121 points a dimension, 7 sds each way) and
    Simpson's rule in age, from the model's parameters alone: no
    quadrature or code of the model's. A failure in an interval by a
    mode is that mode's hazard times the survival of both, integrated
    over the interval (Simpson on 401 ages, the cumulative hazard to
    each by Simpson on the pairs before it)."""
    p = dict(zip(model.parameter_names, model.params))
    v0, v1, r = p["sd_a"] ** 2, p["sd_b"] ** 2, p["corr_a_b"]
    D = np.array([[v0, r * np.sqrt(v0 * v1)], [r * np.sqrt(v0 * v1), v1]])
    g0 = np.linspace(-7, 7, 121) * np.sqrt(v0)
    g1 = np.linspace(-7, 7, 121) * np.sqrt(v1)
    B0, B1 = (a.ravel() for a in np.meshgrid(g0, g1, indexing="ij"))
    b = np.stack([B0, B1], 1)
    log_prior = (
        -0.5 * np.einsum("ni,ij,nj->n", b, np.linalg.inv(D), b)
        - np.log(2 * np.pi)
        - 0.5 * np.log(np.linalg.det(D))
    )
    cell = (g0[1] - g0[0]) * (g1[1] - g1[0])
    m0, m1 = p["a"] + B0, p["b"] + B1

    def log_sum(terms):
        top = terms.max()
        return top + np.log(np.sum(np.exp(terms - top)) * cell)

    def hazard(cause, t):
        """At ages ``t`` ``(A,)``: ``(grid, A)``."""
        t = np.atleast_1d(t)[None, :]
        alpha, beta = p[f"{cause}: alpha"], p[f"{cause}: beta"]
        h0 = (beta / alpha) * (t / alpha) ** (beta - 1)
        m = m0[:, None] + m1[:, None] * t
        return h0 * np.exp(
            p[f"{cause}: marker_effect"] * (m - model.marker_reference)
        )

    def total(t):
        return sum(hazard(cause, t) for cause in model.causes)

    def survival(t):
        if not np.isfinite(t):
            return np.zeros(B0.size)
        if t <= 0:
            return np.ones(B0.size)
        s = np.linspace(0, t, 1001)
        return np.exp(-_simpson(total(s), s[1] - s[0]))

    def fails_between(cause, lo, hi):
        """``int_lo^hi h_cause(s) S(s) ds``."""
        s = np.linspace(lo, hi, 801)
        step = s[1] - s[0]
        h = total(s)
        pairs = (h[:, 0:-2:2] + 4 * h[:, 1:-1:2] + h[:, 2::2]) * step / 3
        H = np.concatenate(
            [np.zeros((B0.size, 1)), np.cumsum(pairs, axis=1)], axis=1
        )
        even = s[::2]
        g = hazard(cause, even) * np.exp(-H) * survival(lo)[:, None]
        return _simpson(g, even[1] - even[0])

    out = 0.0
    for k, (lo, hi) in enumerate(data["x"]):
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
        c, cause = data["c"][k], data["e"][k]
        if c == 0:
            life = np.log(hazard(cause, lo)[:, 0] * survival(lo))
        elif c == 1:
            life = np.log(survival(lo))
        elif c == 2:
            life = np.log(fails_between(cause, lo, hi))
        else:  # failed by hi, having reached its entry
            life = np.log(fails_between(cause, max(entry, 1e-9), hi))
        ll = log_sum(log_y + log_prior + life)
        if np.isfinite(tl) or np.isfinite(tr):
            ll -= log_sum(np.log(survival(entry) - survival(tr)) + log_prior)
        out += data["n"][k] * ll
    return -out


@pytest.mark.parametrize("moved", [False, True], ids=["fitted", "moved"])
def test_every_kind_of_item_agrees_with_a_dense_grid_integral(moved):
    model = _model()
    if moved:
        # away from the maximum, where the nodes are placed afresh
        model = _with_params(
            model,
            **{
                "bearing: alpha": 1200.0,
                "bearing: beta": 1.4,
                "bearing: marker_effect": 1.4,
                "seal: alpha": 900.0,
                "seal: beta": 2.6,
                "seal: marker_effect": -0.5,
                "sd_noise": 0.4,
                "corr_a_b": -0.5,
            },
        )
    f, u, _ = search_space(model, **KINDS)
    with np.errstate(divide="ignore"):
        brute = _brute_force_neg_ll(model, KINDS)
    # 9 rows, 12 items: to within 1e-5 an item
    assert float(f(u)) == pytest.approx(brute, abs=1.2e-4)


def test_an_interval_split_at_a_lag_is_the_interval():
    # The rule of a failure by a mode in an interval, split where a lag
    # puts a kink: with a lag too short to change the path, the same
    # value as the rule in one piece.
    layout = Layout(2, 0, (0, 1), n_causes=2)
    theta = layout.unpack(
        layout.pack(
            {
                "mu": [3.0, 0.003],
                "Delta": np.zeros((2, 0)),
                "L": np.array([[0.6, 0.0], [0.0003, 0.0012]]),
                "sigma": 0.7,
                "alpha": [900.0, 1500.0],
                "beta": [2.0, 1.3],
                "gamma": np.zeros((2, 0)),
                "zeta": [[0.9], [-0.3]],
            }
        )
    )
    arrays = JointArrays.build(
        np.ones(3),
        np.array([0.0, 200.0, 400.0]),
        np.zeros(3),
        np.zeros(3),
        np.zeros((3, 0)),
        [np.zeros(0)] * 3,
        [np.zeros(0)] * 3,
        U=np.array([150.0, 380.0, 900.0]),
        e=np.array([0, 1, 0]),
    )
    params = np.array([3.0, 0.003]) + np.array(
        [[0.0, 0.0], [0.8, 0.001], [-0.5, -0.0005]]
    )
    params = np.broadcast_to(params[None], (3, 3, 2))
    from surpyval.degradation import LinearPath

    values = []
    for lag in (0.0, 1e-7):
        path = PathFunction(LinearPath, ("value",), lag)
        model = Model(layout, path, 4.0)
        values.append(
            np.asarray(
                log_cause_fails_between(
                    cause_thetas(theta, layout),
                    model.path,
                    params,
                    arrays,
                    arrays.T,
                    arrays.U,
                    4.0,
                ),
                dtype=float,
            )
        )
    assert np.all(np.isfinite(values[0])) and np.all(values[0] < 0)
    np.testing.assert_allclose(values[1], values[0], rtol=0, atol=2e-6)


# -- recovery ----------------------------------------------------------------


def _truth(names):
    out = dict(TRUTH)
    for cause, values in CAUSES.items():
        out.update({f"{cause}: {k}": v for k, v in values.items()})
    return np.array([out[name] for name in names])


def test_recovers_the_modes():
    model = _recovered()
    assert model.maximum == "verified"
    assert model.causes == ["bearing", "seal"]
    z = (model.params - _truth(model.parameter_names)) / (
        model.standard_errors()
    )
    # 12 parameters, each within 3.5 standard errors
    assert np.all(np.abs(z) < 3.5), dict(zip(model.parameter_names, z))
    # the reading speaks to the bearing, not to the seal
    table = model.summary().droplevel("part")
    assert table.loc["bearing: marker_effect", "p"] < 0.001
    assert table.loc["seal: marker_effect", "p"] > 0.05


def test_recovers_the_modes_from_inspections():
    data = _data(300, seed=0, inspect=90.0)
    model = WeibullJointPH.fit(data, marker_reference=CENTRE)
    assert model.maximum == "verified"
    assert set(np.unique(data.c)) == {-1, 1, 2}
    z = (model.params - _truth(model.parameter_names)) / (
        model.standard_errors()
    )
    assert np.all(np.abs(z) < 3.5), dict(zip(model.parameter_names, z))


# -- shared parameters -------------------------------------------------------


def test_a_shared_effect_is_nested_and_tested():
    data = _data(300, seed=0)
    each = _recovered()
    one = WeibullJointPH.fit(
        data, marker_reference=CENTRE, shared=["marker_effect"]
    )
    assert one.maximum == "verified"
    assert one.shared == ("marker_effect",)
    assert one.parameter_names[-5:] == [
        "bearing: alpha",
        "bearing: beta",
        "seal: alpha",
        "seal: beta",
        "marker_effect",
    ]
    assert one.k == each.k - 1
    assert one.neg_ll() >= each.neg_ll() - 1e-6
    test = each.likelihood_ratio_test(one)
    assert test["df"] == 1
    assert test["statistic"] == pytest.approx(
        2 * (one.neg_ll() - each.neg_ll())
    )
    # the truth's effects are 0.9 and 0: the modes differ
    assert test["p_value"] < 0.01
    # the shared fit is the modes' own fit held equal: its likelihood
    # by the modes' own layout at the shared estimates
    p = dict(zip(one.parameter_names, one.params))
    held = _with_params(
        each,
        **{k: p[k] for k in one.parameter_names[:-1]},
        **{f"{c}: marker_effect": p["marker_effect"] for c in each.causes},
    )
    arrays = _arrays(data)
    f, u, _ = search_space(held, **arrays)
    g, v, _ = search_space(one, **arrays)
    assert float(f(u)) == pytest.approx(float(g(v)), rel=1e-12)
    assert float(g(v)) == pytest.approx(one.neg_ll(), rel=1e-9)


def test_shared_beta_and_what_cannot_be_shared():
    data = _data()
    model = WeibullJointPH.fit(data, shared=["beta", "marker_effect"])
    assert model.parameter_names[-4:] == [
        "bearing: alpha",
        "seal: alpha",
        "beta",
        "marker_effect",
    ]
    assert model.shared == ("beta", "marker_effect")
    restored = surpyval.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.shared == model.shared
    item = dict(markers={0: 3.2, 60: 3.9}, given=60.0)
    np.testing.assert_array_equal(
        restored.sf([200.0, 500.0], **item), model.sf([200.0, 500.0], **item)
    )
    with pytest.raises(ValueError, match="cannot hold 'alpha'"):
        WeibullJointPH.fit(data, shared=["alpha"])
    with pytest.raises(ValueError, match="slope_effect"):
        WeibullJointPH.fit(data, shared=["slope_effect"])
    with pytest.raises(ValueError, match="fewer than two modes"):
        WeibullJointPH.fit(fleet(60, seed=1), shared=["marker_effect"])
    with pytest.raises(ValueError, match="fewer parameters"):
        model.likelihood_ratio_test(_model())
    with pytest.raises(ValueError, match="same"):
        _model().likelihood_ratio_test(_recovered())


# -- the model ---------------------------------------------------------------


def test_summary_is_grouped_by_mode():
    model = _model()
    table = model.summary()
    parts = list(dict.fromkeys(table.index.get_level_values("part")))
    assert parts[3:] == [
        "survival, bearing: Weibull PH, alpha at marker = 4",
        "link, bearing: current value",
        "survival, seal: Weibull PH, alpha at marker = 4",
        "link, seal: current value",
    ]
    names = list(table.index.get_level_values("name"))
    assert names == model.parameter_names
    plain = table.droplevel("part")
    # positive parameters have log-scale intervals and no z
    for name in ("bearing: alpha", "seal: beta"):
        assert plain.loc[name, "coef lower 95%"] > 0
        assert np.isnan(plain.loc[name, "z"])
    assert np.isfinite(plain.loc["seal: marker_effect", "z"])
    assert plain.loc["bearing: marker_effect", "exp(coef)"] == pytest.approx(
        np.exp(plain.loc["bearing: marker_effect", "coef"])
    )
    text = repr(model)
    counts = model.n_events_by_cause
    assert (
        "Failure modes       : bearing ({:g} failures), seal ({:g} "
        "failures)".format(counts["bearing"], counts["seal"]) in text
    )
    assert "Survival, seal: Weibull PH" in text
    assert sum(counts.values()) == model.n_events
    assert model.k == len(model.params) == 12


def test_survival_is_of_every_mode_together():
    model = _model()
    item = dict(markers={0: 3.2, 60: 3.9, 120: 4.3}, given=120.0)
    x = np.array([200.0, 400.0, 700.0])
    sf = model.sf(x, **item)
    assert np.all(np.diff(sf) < 0) and np.all((sf > 0) & (sf < 1))
    # a mode made far less likely leaves more survival
    safer = _with_params(model, **{"seal: alpha": 1e6})
    assert np.all(safer.sf(x, **item) > sf)
    pred = model.predict(**item)
    np.testing.assert_array_equal(pred.sf(x), sf)
    # the hazard is of failing at all: the survival's derivative
    step = 1e-3
    slope = (np.log(pred.sf(x + step)) - np.log(pred.sf(x - step))) / (
        2 * step
    )
    np.testing.assert_allclose(pred.hf(x), -slope, rtol=1e-5)
    np.testing.assert_allclose(pred.sf(pred.qf([0.2, 0.6])), [0.8, 0.4])
    band = pred.cb(x)
    assert np.all((band[:, 0] < sf) & (sf < band[:, 1]))
    assert len(pred.to_dict()["sf"]) == 101


def test_round_trips_through_json_and_pickle():
    model = _model()
    d = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    assert d["schema"] == 5
    assert d["causes"] == ["bearing", "seal"] and d["shared"] == []
    item = dict(markers={0: 3.2, 60: 3.9}, given=60.0)
    x = [100.0, 300.0, 900.0]
    for restored in (
        surpyval.from_dict(d),
        pickle.loads(pickle.dumps(model)),
    ):
        assert restored.causes == model.causes
        assert restored.parameter_names == model.parameter_names
        assert restored.n_events_by_cause == model.n_events_by_cause
        np.testing.assert_array_equal(restored.params, model.params)
        np.testing.assert_array_equal(
            restored.sf(x, **item), model.sf(x, **item)
        )
        np.testing.assert_array_equal(
            restored.cb(x, **item), model.cb(x, **item)
        )
        assert repr(restored) == repr(model)
    wrong = dict(d, parameter_names=list(reversed(d["parameter_names"])))
    with pytest.raises(ValueError, match="'causes'"):
        JointModel.from_dict(wrong)
    with pytest.raises(ValueError, match="two or more"):
        JointModel.from_dict(dict(d, causes=["bearing"]))


def test_labels_of_any_kind_and_the_array_and_frame_forms():
    data = _data(120, seed=5)
    arrays = _arrays(data)
    from_data = WeibullJointPH.fit(data)
    from_arrays = WeibullJointPH.fit(**arrays)
    np.testing.assert_array_equal(from_arrays.params, from_data.params)
    # labels that do not sort together, and a missing one as nan
    relabel = {"bearing": 2, "seal": ("seal", "lip"), None: np.nan}
    e = np.empty(len(arrays["e"]), dtype=object)
    for k, label in enumerate(arrays["e"]):
        e[k] = relabel[label]
    odd = WeibullJointPH.fit(**dict(arrays, e=e))
    assert odd.causes == [2, ("seal", "lip")]
    np.testing.assert_array_equal(odd.params, from_data.params)
    assert odd.parameter_names[-1] == "('seal', 'lip'): marker_effect"
    restored = surpyval.from_dict(json.loads(json.dumps(odd.to_dict())))
    assert restored.causes == odd.causes
    frame = pd.DataFrame(
        {
            "x": arrays["x"],
            "c": arrays["c"],
            "markers": arrays["markers"],
            "mode": arrays["e"],
        }
    )
    from_frame = WeibullJointPH.fit_from_df(
        frame, x_col="x", markers_col="markers", c_col="c", e_col="mode"
    )
    np.testing.assert_array_equal(from_frame.params, from_data.params)
    assert from_frame.causes == from_data.causes


def test_covariates_enter_each_modes_hazard():
    items, _ = simulate_causes(200, seed=8)
    data = SurpyvalData.from_items(items)
    arrays = _arrays(data)
    z = (np.arange(len(arrays["x"])) % 2).astype(float)
    frame = pd.DataFrame({"site": z})
    model = WeibullJointPH.fit(
        **arrays, Z=frame, hazard_covariates=["site"], path_covariates=[]
    )
    assert model.parameter_names[-8:] == [
        "bearing: alpha",
        "bearing: beta",
        "bearing: site",
        "bearing: marker_effect",
        "seal: alpha",
        "seal: beta",
        "seal: site",
        "seal: marker_effect",
    ]
    shared = WeibullJointPH.fit(
        **arrays,
        Z=frame,
        hazard_covariates=["site"],
        path_covariates=[],
        shared=["site"],
    )
    assert shared.parameter_names[-1] == "site"
    assert shared.k == model.k - 1
    assert shared.neg_ll() >= model.neg_ll() - 1e-6
    item = dict(markers={0: 3.2, 60: 3.9}, given=60.0)
    for fitted in (model, shared):
        restored = surpyval.from_dict(fitted.to_dict())
        np.testing.assert_array_equal(
            restored.sf([300.0], Z={"site": 1.0}, **item),
            fitted.sf([300.0], Z={"site": 1.0}, **item),
        )
    # the reported alpha is at site = 0: a model moved to another
    # reference predicts the same
    lower = model.sf([300.0], Z={"site": 0.0}, **item)
    assert model.sf([300.0], Z={"site": 1.0}, **item) != lower


# -- refusals ----------------------------------------------------------------


def test_what_the_fit_of_modes_cannot_take_is_refused():
    data = _data(120, seed=5)
    arrays = _arrays(data)
    with pytest.raises(ValueError, match="JointAFT.*JointPH"):
        WeibullJointAFT.fit(**arrays)
    with pytest.raises(ValueError, match="two-stage.*how='MLE'"):
        WeibullJointPH.fit(**arrays, how="two-stage")
    failed = np.flatnonzero(arrays["c"] == 0)
    running = np.flatnonzero(arrays["c"] == 1)
    e = arrays["e"].copy()
    e[failed[:3]] = None
    with pytest.raises(ValueError, match="3 failure.*no failure mode"):
        WeibullJointPH.fit(**dict(arrays, e=e))
    e = arrays["e"].copy()
    e[running[:2]] = "seal"
    with pytest.raises(ValueError, match="2 item.*still running"):
        WeibullJointPH.fit(**dict(arrays, e=e))
    e = arrays["e"].copy()
    e[failed[0]], e[failed[1]] = "gasket", "gasket"
    with pytest.raises(ValueError, match="'gasket': 2") as error:
        WeibullJointPH.fit(**dict(arrays, e=e))
    assert "3: alpha, beta, marker_effect" in str(error.value)
    with pytest.raises(ValueError, match="one failure mode per item"):
        WeibullJointPH.fit(**dict(arrays, e=arrays["e"][:-1]))
    with pytest.raises(ValueError, match="one label per item"):
        WeibullJointPH.fit(**dict(arrays, e=np.stack([arrays["e"]] * 2)))


def test_the_fitter_with_a_link_takes_modes():
    data = _data(200, seed=9)
    model = JointPH(Weibull, link=["value", "slope"]).fit(
        data, shared=["slope_effect"]
    )
    assert model.parameter_names[-7:] == [
        "bearing: alpha",
        "bearing: beta",
        "bearing: marker_effect",
        "seal: alpha",
        "seal: beta",
        "seal: marker_effect",
        "slope_effect",
    ]
    parts = list(dict.fromkeys(model.summary().index.get_level_values(0)))
    assert parts[-1] == "shared by the failure modes"
