"""Competing failure modes in the joint model of several markers:
``JointPH(Weibull, {name: path, ...}).fit(..., e=, shared=)``.

The fleet is ``_multi_fleet.simulate_causes``'s: items read with a
vibration and a temperature, failing by the first of a bearing failure
whose hazard follows the vibration (0.8 per unit) and a seal failure
whose hazard follows the temperature (0.08 per unit). The checks:

- one mode is the several-marker model without modes, exactly;
- one marker given as a dict, with modes, is the one-marker model of
  modes (``WeibullJointPH.fit(..., e=)``) to round-off;
- the likelihood equals simpler models' where it must: two modes with
  one hazard are the pooled model at twice the hazard, and a mode that
  follows no marker factors out as a plain Weibull, leaving the
  several-marker model of the other mode with its failures censored;
- a fit recovers each mode's effect of each marker, a shared effect is
  nested and tested, and the model summarises, predicts, saves, loads,
  takes a lag and a weight by marker and compares one marker's links
  with its modes.

With every mode following some marker, the likelihood does not factor by
mode: an item's path is integrated over once, with every mode's hazard
in the integrand, so a mode's fit with the others censored is not this
fit's answer (the docs say so).
"""

import functools
import json
import pickle

import numpy as np
import pandas as pd
import pytest

import surpyval
from surpyval import JointPH, Measurements, Weibull, WeibullJointPH
from surpyval.degradation import LinearPath
from surpyval.tests.univariate.joint._fleet import (
    CENTRE,
)
from surpyval.tests.univariate.joint._fleet import (
    simulate_causes as one_marker_causes,
)
from surpyval.tests.univariate.joint._multi_fleet import (
    CAUSES,
    CENTRES,
    TRUTH,
    simulate,
    simulate_causes,
)
from surpyval.univariate.joint import JointModel
from surpyval.univariate.joint.joint_ph import search_space

PATHS = {"vibration": LinearPath, "temperature": LinearPath}
REFERENCE = {"vibration": CENTRES[0], "temperature": CENTRES[1]}


def _fitter(**kwargs):
    """Both markers in the hazard, the vibration's start and rate and
    the temperature's start varying, each marker's correlated with its
    own only (three dimensions an item, for speed)."""
    return JointPH(
        Weibull,
        PATHS,
        varying=kwargs.pop(
            "varying", {"vibration": ["a", "b"], "temperature": ["a"]}
        ),
        random_effects="by_marker",
        **kwargs,
    )


def _data(n=150, seed=3):
    x, c, e, m, _ = simulate_causes(n, seed=seed)
    return dict(
        x=np.asarray(x), c=np.asarray(c), e=np.array(e, object), markers=m
    )


@functools.cache
def _model():
    return _fitter().fit(**_data(), marker_reference=REFERENCE)


def _with_params(model, **values):
    d = model.to_dict()
    for name, value in values.items():
        d["params"][model.parameter_names.index(name)] = value
    d.pop("covariance", None)
    return JointModel.from_dict(d)


def _one_mode(model, alpha, beta, effects):
    """The several-marker model of one kind of failure with the same
    marker part, the survival part ``alpha``, ``beta`` and ``effects``
    (by marker)."""
    d = model.to_dict()
    names = model.parameter_names
    marker = names.index("temperature.sd_noise") + 1
    d["parameter_names"] = names[:marker] + [
        "alpha",
        "beta",
        "vibration.marker_effect",
        "temperature.marker_effect",
    ]
    d["params"] = d["params"][:marker] + [
        alpha,
        beta,
        effects["vibration"],
        effects["temperature"],
    ]
    for key in ("causes", "shared", "n_events_by_cause", "covariance"):
        d.pop(key, None)
    return JointModel.from_dict(d)


# -- the reductions -----------------------------------------------------------


def test_one_mode_is_the_several_marker_model_without_modes_exactly():
    x, c, m, _ = simulate(n=60, seed=4)
    fitter = _fitter(varying={"vibration": ["a"], "temperature": ["a"]})
    e = np.array([None if k == 1 else "wear" for k in c], dtype=object)
    one = fitter.fit(x, c=c, markers=m, e=e)
    none = fitter.fit(x, c=c, markers=m)
    assert one.causes is None and one.shared == ()
    assert one.parameter_names == none.parameter_names
    np.testing.assert_array_equal(one.params, none.params)
    np.testing.assert_array_equal(one.covariance(), none.covariance())
    assert one.neg_ll() == none.neg_ll()
    assert "causes" not in one.to_dict()


#: The one-marker model's names in a dict model of the one marker.
ONE_NAMES = {
    "a": "vibration.a",
    "b": "vibration.b",
    "sd_a": "sd_vibration.a",
    "sd_b": "sd_vibration.b",
    "corr_a_b": "corr_vibration.a_vibration.b",
    "sd_noise": "vibration.sd_noise",
}


def _as_one_marker(name):
    if ": " in name:
        cause, own = name.split(": ")
        if own == "marker_effect":
            return f"{cause}: vibration.marker_effect"
        return name
    return ONE_NAMES.get(name, name)


def test_one_marker_as_a_dict_is_the_one_marker_model_of_modes():
    items, _ = one_marker_causes(150, seed=3)
    data = surpyval.SurpyvalData.from_items(items)
    single = WeibullJointPH.fit(data, marker_reference=CENTRE)
    several = JointPH(Weibull, {"vibration": LinearPath}).fit(
        data.x,
        c=data.c,
        n=data.n,
        e=data.e,
        markers=[{"vibration": r} for r in data._markers_by_row()],
        marker_reference={"vibration": CENTRE},
    )
    assert several.causes == single.causes == ["bearing", "seal"]
    assert several.parameter_names == [
        _as_one_marker(n) for n in single.parameter_names
    ]
    assert several.maximum == single.maximum == "verified"
    np.testing.assert_allclose(several.params, single.params, rtol=1e-9)
    np.testing.assert_allclose(
        several.standard_errors(), single.standard_errors(), rtol=1e-6
    )
    assert several.neg_ll() == pytest.approx(single.neg_ll(), rel=1e-12)
    readings = {0.0: 3.0, 60.0: 3.4, 120.0: 3.9}
    np.testing.assert_allclose(
        several.sf([150, 300], markers={"vibration": readings}, given=120),
        single.sf([150, 300], markers=readings, given=120),
        rtol=1e-9,
    )


# -- the likelihood, where simpler models give it ----------------------------


def test_modes_with_equal_hazards_are_the_pooled_model():
    # Two modes with one hazard each: the failure hazard is twice it, and
    # each failure's mode a fair coin, so the likelihood is the pooled
    # model's at twice the hazard, less log 2 a failure.
    model = _model()
    data = _data()
    alpha, beta = 1500.0, 1.7
    effects = {"vibration": 0.5, "temperature": 0.05}
    values = {}
    for cause in model.causes:
        values[f"{cause}: alpha"] = alpha
        values[f"{cause}: beta"] = beta
        for marker, effect in effects.items():
            values[f"{cause}: {marker}.marker_effect"] = effect
    f, u, _ = search_space(_with_params(model, **values), **data)
    pooled = _one_mode(model, alpha * 2 ** (-1 / beta), beta, effects)
    plain = {k: v for k, v in data.items() if k != "e"}
    g, v, _ = search_space(pooled, **plain)
    failures = float(np.sum(data["c"] == 0))
    assert float(f(u)) == pytest.approx(
        float(g(v)) + failures * np.log(2.0), rel=1e-10
    )


def test_a_mode_no_marker_follows_factors_out():
    # With both its effects 0 the seal's hazard is the same on every
    # path: its failures and everyone's survival of it are a plain
    # Weibull's likelihood, and what is left is the several-marker model
    # of the bearing's failures, the seal's censored.
    model = _model()
    data = _data()
    values = {
        "bearing: alpha": 1200.0,
        "bearing: beta": 2.1,
        "bearing: vibration.marker_effect": 0.7,
        "bearing: temperature.marker_effect": 0.02,
        "seal: alpha": 1600.0,
        "seal: beta": 1.4,
        "seal: vibration.marker_effect": 0.0,
        "seal: temperature.marker_effect": 0.0,
    }
    f, u, _ = search_space(_with_params(model, **values), **data)
    seal = np.array([label == "seal" for label in data["e"]])
    bearing = _one_mode(
        model, 1200.0, 2.1, {"vibration": 0.7, "temperature": 0.02}
    )
    censored = dict(
        x=data["x"], c=np.where(seal, 1, data["c"]), markers=data["markers"]
    )
    g, v, _ = search_space(bearing, **censored)
    weibull = Weibull.from_params([1600.0, 1.4])
    seal_ll = np.sum(
        np.where(seal, np.log(weibull.hf(data["x"])), 0.0)
        - weibull.Hf(data["x"])
    )
    assert float(f(u)) == pytest.approx(float(g(v)) - seal_ll, rel=1e-10)


# -- the fit -----------------------------------------------------------------


def _truth(names):
    out = dict(TRUTH)
    for cause, values in CAUSES.items():
        out.update({f"{cause}: {k}": v for k, v in values.items()})
    return np.array([out[n] for n in names])


def test_recovers_each_modes_effect_of_each_marker():
    # A fleet of 300 (about 95 bearing and 75 seal failures), every path
    # parameter varying and correlated across the markers, alpha at the
    # truth's centres: every parameter within 3.5 standard errors (the
    # cross-correlations are not compared: the truth's matrix is the
    # fleet's, the model's names its own).
    x, c, e, m, _ = simulate_causes(300, seed=0)
    model = JointPH(Weibull, PATHS).fit(
        x, c=c, e=np.array(e, object), markers=m, marker_reference=REFERENCE
    )
    assert model.maximum == "verified"
    assert model.causes == ["bearing", "seal"]
    names = [n for n in model.parameter_names if not n.startswith("corr_")]
    where = [model.parameter_names.index(n) for n in names]
    z = (model.params[where] - _truth(names)) / (
        model.standard_errors()[where]
    )
    assert np.all(np.abs(z) < 3.5), dict(zip(names, z))
    table = model.summary().droplevel("part")
    # each marker speaks to its own mode
    assert table.loc["bearing: vibration.marker_effect", "p"] < 0.001
    assert table.loc["seal: temperature.marker_effect", "p"] < 0.001


def test_a_shared_effect_is_nested_and_tested():
    data = _data()
    each = _model()
    one = _fitter().fit(
        **data,
        marker_reference=REFERENCE,
        shared=["beta", "vibration.marker_effect"],
    )
    assert one.maximum == "verified"
    assert one.shared == ("beta", "vibration.marker_effect")
    assert one.parameter_names[-6:] == [
        "bearing: alpha",
        "bearing: temperature.marker_effect",
        "seal: alpha",
        "seal: temperature.marker_effect",
        "beta",
        "vibration.marker_effect",
    ]
    assert one.k == each.k - 2
    assert one.neg_ll() >= each.neg_ll() - 1e-6
    test = each.likelihood_ratio_test(one)
    assert test["df"] == 2
    assert test["statistic"] == pytest.approx(
        2 * (one.neg_ll() - each.neg_ll())
    )
    # the shared fit is the modes' own fit held equal: its likelihood by
    # the modes' own layout at the shared estimates
    p = dict(zip(one.parameter_names, one.params))
    held = _with_params(
        each,
        **{k: p[k] for k in one.parameter_names[:-6]},
        **{
            f"{c}: {name}": p[k]
            for c in each.causes
            for name, k in (
                ("alpha", f"{c}: alpha"),
                (
                    "temperature.marker_effect",
                    f"{c}: temperature.marker_effect",
                ),
                ("beta", "beta"),
                ("vibration.marker_effect", "vibration.marker_effect"),
            )
        },
    )
    f, u, _ = search_space(held, **data)
    g, v, _ = search_space(one, **data)
    assert float(f(u)) == pytest.approx(float(g(v)), rel=1e-12)
    assert float(g(v)) == pytest.approx(one.neg_ll(), rel=1e-9)
    restored = surpyval.from_dict(json.loads(json.dumps(one.to_dict())))
    assert restored.shared == one.shared
    question = dict(markers={"vibration": {0.0: 3.0, 60.0: 3.4}}, given=60)
    np.testing.assert_array_equal(
        restored.sf([200.0, 500.0], **question),
        one.sf([200.0, 500.0], **question),
    )


def test_what_cannot_be_shared_is_refused():
    data = _data(80, seed=5)
    fitter = _fitter()
    with pytest.raises(ValueError, match="cannot hold 'alpha'"):
        fitter.fit(**data, shared=["alpha"])
    # an effect is a marker's: the plain name is not one
    with pytest.raises(ValueError, match="vibration.marker_effect"):
        fitter.fit(**data, shared=["marker_effect"])
    with pytest.raises(ValueError, match="fewer than two modes"):
        no_modes = {k: v for k, v in data.items() if k != "e"}
        fitter.fit(**no_modes, shared=["vibration.marker_effect"])


# -- the model ---------------------------------------------------------------


def test_summary_and_printout_are_grouped_by_mode():
    model = _model()
    table = model.summary()
    parts = list(dict.fromkeys(table.index.get_level_values("part")))
    at = "alpha at vibration = 4, temperature = 65"
    links = "vibration's current value, temperature's current value"
    assert parts[3:] == [
        f"survival, bearing: Weibull PH, {at}",
        f"link, bearing: {links}",
        f"survival, seal: Weibull PH, {at}",
        f"link, seal: {links}",
    ]
    assert list(table.index.get_level_values("name")) == (
        model.parameter_names
    )
    plain = table.droplevel("part")
    for name in ("bearing: alpha", "seal: beta"):
        assert plain.loc[name, "coef lower 95%"] > 0
        assert np.isnan(plain.loc[name, "z"])
    assert np.isfinite(plain.loc["seal: vibration.marker_effect", "z"])
    text = repr(model)
    counts = model.n_events_by_cause
    assert (
        "Failure modes       : bearing ({:g} failures), seal ({:g} "
        "failures)".format(counts["bearing"], counts["seal"]) in text
    )
    assert "Survival, seal: Weibull PH" in text
    assert sum(counts.values()) == model.n_events
    # 4 path, 3 spreads and 1 correlation, 2 noise, 4 a mode
    assert model.k == len(model.params) == 10 + 2 * 4


def test_survival_is_of_every_mode_together():
    model = _model()
    question = dict(
        markers={
            "vibration": {0.0: 3.2, 60.0: 3.5, 120.0: 3.9},
            "temperature": {0.0: 61.0, 120.0: 63.0},
        },
        given=120.0,
    )
    x = np.array([200.0, 400.0, 700.0])
    sf = model.sf(x, **question)
    assert np.all(np.diff(sf) < 0) and np.all((sf > 0) & (sf < 1))
    # a mode made far less likely leaves more survival
    safer = _with_params(model, **{"seal: alpha": 1e6})
    assert np.all(safer.sf(x, **question) > sf)
    pred = model.predict(**question)
    np.testing.assert_array_equal(pred.sf(x), sf)
    step = 1e-3
    slope = (np.log(pred.sf(x + step)) - np.log(pred.sf(x - step))) / (
        2 * step
    )
    # the survival's cumulative hazards are by the 15-point rule, whose
    # error is about 1e-5 of itself (the docs); the hazard is exact, so
    # the two agree to that error (1.1e-5 at 700 days here)
    np.testing.assert_allclose(pred.hf(x), -slope, rtol=2e-5)
    np.testing.assert_allclose(pred.sf(pred.qf([0.2, 0.6])), [0.8, 0.4])
    band = model.cb(x, **question)
    assert np.all((band[:, 0] < sf) & (sf < band[:, 1]))


def test_round_trips_through_json_and_pickle():
    model = _model()
    d = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    assert d["schema"] == 5
    assert d["causes"] == ["bearing", "seal"] and d["shared"] == []
    question = dict(markers={"vibration": {0.0: 3.2, 60.0: 3.9}}, given=60)
    x = [100.0, 300.0, 900.0]
    for restored in (surpyval.from_dict(d), pickle.loads(pickle.dumps(model))):
        assert restored.causes == model.causes
        assert restored.markers == model.markers
        assert restored.parameter_names == model.parameter_names
        assert restored.n_events_by_cause == model.n_events_by_cause
        np.testing.assert_array_equal(restored.params, model.params)
        np.testing.assert_array_equal(
            restored.sf(x, **question), model.sf(x, **question)
        )
        np.testing.assert_array_equal(
            restored.cb(x, **question), model.cb(x, **question)
        )
        assert repr(restored) == repr(model)


def test_the_measurements_data_and_frame_forms_agree():
    data = _data(80, seed=5)
    fitter = _fitter(varying={"vibration": ["a"], "temperature": ["a"]})
    arrays = fitter.fit(**data)
    ids = [f"u{k}" for k in range(len(data["x"]))]
    m = Measurements(
        {
            name: {i: dict(r[name]) for i, r in zip(ids, data["markers"])}
            for name in PATHS
        }
    )
    sdata = surpyval.SurpyvalData(
        data["x"], c=data["c"], e=data["e"], i=ids, m=m
    )
    np.testing.assert_array_equal(fitter.fit(sdata).params, arrays.params)
    frame = pd.DataFrame(
        {
            "x": data["x"],
            "c": data["c"],
            "markers": data["markers"],
            "mode": data["e"],
        }
    )
    from_frame = fitter.fit_from_df(
        frame, x_col="x", markers_col="markers", c_col="c", e_col="mode"
    )
    np.testing.assert_array_equal(from_frame.params, arrays.params)
    assert from_frame.causes == arrays.causes == ["bearing", "seal"]


def test_a_lag_and_a_weight_by_marker_take_modes():
    data = _data(100, seed=6)
    fitter = _fitter(
        varying={"vibration": ["a"], "temperature": ["a"]},
        link={"vibration": "value", "temperature": "cumulative"},
        lag={"vibration": 30.0},
        weight={"temperature": ("window", 180.0)},
    )
    model = fitter.fit(**data)
    assert model.maximum == "verified"
    assert model.causes == ["bearing", "seal"]
    assert "seal: temperature.cumulative_effect" in model.parameter_names
    parts = list(dict.fromkeys(model.summary().index.get_level_values(0)))
    assert parts[4] == (
        "link, bearing: vibration's current value, temperature's "
        "cumulative area, lags vibration 30, weights: temperature the "
        "area over the last 180 only (window)"
    )
    restored = surpyval.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.lag == {"vibration": 30.0, "temperature": 0.0}
    question = dict(markers={"vibration": {0.0: 3.0, 60.0: 3.4}}, given=60)
    np.testing.assert_array_equal(
        restored.sf([100.0, 300.0], **question),
        model.sf([100.0, 300.0], **question),
    )


def test_compare_links_of_one_marker_takes_modes():
    data = _data(100, seed=6)
    fitter = _fitter(varying={"vibration": ["a"], "temperature": ["a"]})
    comparison = fitter.compare_links(
        **data, links=["value", "slope"], marker="vibration"
    )
    assert sorted(comparison.models) == ["slope", "value"]
    value = comparison.models["value"]
    assert value.causes == ["bearing", "seal"]
    np.testing.assert_allclose(
        value.params, fitter.fit(**data).params, rtol=1e-6
    )
    slope = comparison.models["slope"]
    assert "bearing: vibration.slope_effect" in slope.parameter_names
    assert "seal: temperature.marker_effect" in slope.parameter_names


def test_what_the_fit_of_modes_cannot_take_is_refused():
    from surpyval import JointAFT

    data = _data(60, seed=5)
    with pytest.raises(ValueError, match="JointAFT.*JointPH"):
        JointAFT(Weibull, PATHS).fit(**data)
    with pytest.raises(ValueError, match="two-stage.*how='MLE'"):
        _fitter().fit(**data, how="two-stage")
    e = data["e"].copy()
    e[np.flatnonzero(data["c"] == 0)[:2]] = None
    with pytest.raises(ValueError, match="2 failure.*no failure mode"):
        _fitter().fit(**dict(data, e=e))
