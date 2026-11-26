"""Joint models of several markers: ``JointPH(Weibull, {name: path, ...})``.

The checks:

- a model of one marker given as a dict is the single-marker fit (the
  same numbers to round-off: the likelihood and the search are the
  same);
- two markers whose random effects are uncorrelated across them
  (``random_effects="by_marker"``) and the second of which the hazard does
  not follow (``link={"temperature": []}``) are the first marker's joint
  fit plus a linear mixed model of the second: the first's parameters are
  the single-marker fit's, the second's those of R's
  ``nlme::lme(..., method = "ML")``, and the log-likelihood the sum;
- a fit of two markers, each with its own effect, recovers them
  (``_multi_fleet.py``), and its predictions, bounds, printout and saved
  form keep the package's rules;
- the readings by marker reach the fit the same way from arrays, a
  ``Measurements`` and a ``SurpyvalData``;
- what the model does not take is refused, saying what to do.
"""

import json
import pickle

import numpy as np
import pytest

import surpyval
from surpyval import (
    JointPH,
    Measurements,
    SurpyvalData,
    Weibull,
    WeibullJointPH,
)
from surpyval.degradation import LinearPath, QuadraticPath
from surpyval.tests.univariate.joint._fleet import fleet
from surpyval.tests.univariate.joint._multi_fleet import simulate

#: nlme::lme(y ~ t, random = ~ 1 | id, method = "ML") of the temperature
#: readings of simulate(n=80, seed=7), by
#: scripts/reference/reference_r_joint_markers.R (R 4.6.2, nlme 3.1):
#: the fixed effects, the random intercept's sd, the noise sd and the
#: log-likelihood. (R 4.6.2, nlme 3.1.171.)
LME = {
    "temperature.a": 60.6227719464963,
    "temperature.b": 0.0140339451330678,
    "sd_temperature.a": 3.75549373746069,
    "temperature.sd_noise": 2.41304351572552,
    "loglik": -1917.3460256,
}

ONE_NAMES = {
    "a": "vibration.a",
    "b": "vibration.b",
    "sd_a": "sd_vibration.a",
    "sd_b": "sd_vibration.b",
    "corr_a_b": "corr_vibration.a_vibration.b",
    "sd_noise": "vibration.sd_noise",
    "alpha": "alpha",
    "beta": "beta",
    "marker_effect": "vibration.marker_effect",
}


def _by_marker(rows, name="vibration"):
    return [{name: dict(r)} for r in rows]


@pytest.fixture(scope="module")
def one():
    data = fleet(n=60, seed=5)
    single = WeibullJointPH.fit(data)
    several = JointPH(Weibull, {"vibration": LinearPath}).fit(
        data.x, c=data.c, n=data.n, markers=_by_marker(data._markers_by_row())
    )
    return data, single, several


@pytest.fixture(scope="module")
def reduced():
    x, c, m, _ = simulate(n=80, seed=7)
    fitter = JointPH(
        Weibull,
        {"vibration": LinearPath, "temperature": LinearPath},
        varying={"temperature": ["a"]},
        link={"vibration": "value", "temperature": []},
        random_effects="by_marker",
    )
    two = fitter.fit(x, c=c, markers=m)
    vibration = WeibullJointPH.fit(x, c=c, markers=[r["vibration"] for r in m])
    return (x, c, m), two, vibration


@pytest.fixture(scope="module")
def pair():
    """Two markers, every parameter of each varying (4), correlated
    across them, both in the hazard."""
    x, c, m, _ = simulate(n=150, seed=1)
    fitter = JointPH(
        Weibull, {"vibration": LinearPath, "temperature": LinearPath}
    )
    return (x, c, m), fitter.fit(x, c=c, markers=m)


# -- the reductions -----------------------------------------------------------


def test_one_marker_is_the_single_marker_fit(one):
    data, single, several = one
    assert several.markers == ["vibration"]
    assert several.parameter_names == [
        ONE_NAMES[n] for n in single.parameter_names
    ]
    np.testing.assert_allclose(several.params, single.params, rtol=1e-10)
    np.testing.assert_allclose(
        several.standard_errors(), single.standard_errors(), rtol=1e-8
    )
    assert several.neg_ll() == pytest.approx(single.neg_ll(), rel=1e-12)
    assert several.maximum == single.maximum == "verified"
    readings = {0.0: 3.0, 60.0: 3.4, 120.0: 3.9}
    np.testing.assert_allclose(
        several.sf([150, 300], markers={"vibration": readings}, given=120),
        single.sf([150, 300], markers=readings, given=120),
        rtol=1e-10,
    )


def test_an_unlinked_uncorrelated_marker_is_a_separate_mixed_model(reduced):
    _, two, vibration = reduced
    assert two.maximum == vibration.maximum == "verified"
    got = dict(zip(two.parameter_names, two.params))
    # the first marker and the survival part: the single-marker fit's
    for name, value in zip(vibration.parameter_names, vibration.params):
        assert got[ONE_NAMES[name]] == pytest.approx(value, rel=1e-6), name
    # the second: R's linear mixed model by maximum likelihood
    for name in ("temperature.a", "temperature.b", "sd_temperature.a"):
        assert got[name] == pytest.approx(LME[name], rel=1e-7), name
    assert got["temperature.sd_noise"] == pytest.approx(
        LME["temperature.sd_noise"], rel=1e-7
    )
    # and the log-likelihood the sum
    assert -two.neg_ll() == pytest.approx(
        -vibration.neg_ll() + LME["loglik"], abs=1e-5
    )
    assert "temperature.marker_effect" not in two.parameter_names
    assert not any(
        n.startswith("corr_") and "temperature" in n
        for n in two.parameter_names
    )


# -- two markers, both followed -----------------------------------------------


def test_two_markers_recover_their_effects(pair):
    _, model = pair
    assert model.maximum == "verified"
    got = dict(zip(model.parameter_names, model.params))
    se = dict(zip(model.parameter_names, model.standard_errors()))
    # the truth (_multi_fleet.TRUTH) within 3 standard errors
    for name, truth in (
        ("vibration.marker_effect", 0.8),
        ("temperature.marker_effect", 0.08),
        ("vibration.sd_noise", 0.5),
        ("temperature.sd_noise", 2.0),
        ("corr_vibration.a_temperature.a", 0.5),
        ("corr_vibration.b_temperature.b", 0.3),
        ("beta", 2.0),
    ):
        assert abs(got[name] - truth) < 3 * se[name], name
    assert model.effect_units == {
        "vibration.marker_effect": "marker unit",
        "temperature.marker_effect": "marker unit",
    }


def test_the_printout_and_the_summary(pair):
    _, model = pair
    text = repr(model)
    assert "Markers             : 2" in text
    assert "temperature.marker_effect" in text
    assert "alpha at vibration = " in text
    table = model.summary()
    parts = list(dict.fromkeys(table.index.get_level_values(0)))
    assert parts[0].startswith("paths: vibration: Linear")
    assert parts[2] == "noise"
    links = table.loc[parts[-1]]
    assert list(links.index) == [
        "vibration.marker_effect",
        "temperature.marker_effect",
    ]
    assert np.all(np.isfinite(links["p"]))


def test_predictions_take_every_markers_readings(pair):
    _, model = pair
    readings = {
        "vibration": {0.0: 3.5, 30.0: 3.8, 60.0: 4.0},
        "temperature": {0.0: 62.0, 60.0: 63.0},
    }
    pred = model.predict(markers=readings, given=60)
    x = np.array([100.0, 200.0, 400.0])
    np.testing.assert_allclose(
        pred.sf(x), model.sf(x, markers=readings, given=60), rtol=1e-12
    )
    assert np.all(np.diff(pred.sf(x)) < 0)
    assert pred.markers == readings
    # a hotter item with the same vibration fails sooner (temperature's
    # effect is positive)
    hot = dict(readings, temperature={0.0: 70.0, 60.0: 72.0})
    assert model.sf(400, markers=hot, given=60) < model.sf(
        400, markers=readings, given=60
    )
    band = model.cb(x, markers=readings, given=60)
    assert band.shape == (3, 2)
    assert np.all(band[:, 0] <= pred.sf(x)) and np.all(
        pred.sf(x) <= band[:, 1]
    )
    q = model.qf([0.5], markers=readings, given=60)
    assert pred.sf(q) == pytest.approx(0.5, abs=1e-6)
    assert np.isfinite(model.mean(markers=readings, given=60, tau=2000))
    path = pred.marker([0.0, 60.0], "temperature")
    assert 55 < path[0] < 70
    with pytest.raises(ValueError, match="name the one whose path"):
        pred.marker([0.0])
    out = pred.to_dict()
    json.dumps(out, allow_nan=False)
    assert set(out["marker"]) == {"vibration", "temperature"}


def test_saved_and_pickled(pair):
    _, model = pair
    d = model.to_dict()
    restored = surpyval.from_dict(json.loads(json.dumps(d, allow_nan=False)))
    assert restored.markers == model.markers
    assert restored.parameter_names == model.parameter_names
    readings = {"vibration": {0.0: 3.5}, "temperature": {0.0: 62.0}}
    np.testing.assert_allclose(
        restored.sf([100, 300], markers=readings, given=10),
        model.sf([100, 300], markers=readings, given=10),
        rtol=1e-12,
    )
    again = pickle.loads(pickle.dumps(model))
    assert again.aic() == model.aic()


def test_likelihood_ratio_tests_of_an_effect_and_of_the_cross_correlations(
    pair,
):
    (x, c, m), model = pair
    paths = {"vibration": LinearPath, "temperature": LinearPath}
    without = JointPH(
        Weibull, paths, link={"vibration": "value", "temperature": []}
    ).fit(x, c=c, markers=m)
    test = model.likelihood_ratio_test(without)
    assert test["df"] == 1
    assert test["p_value"] < 0.01
    apart = JointPH(Weibull, paths, random_effects="by_marker").fit(
        x, c=c, markers=m
    )
    test = model.likelihood_ratio_test(apart)
    assert test["df"] == 4
    # the starts are correlated 0.5 across the markers
    assert test["p_value"] < 0.01


# -- the readings -------------------------------------------------------------


def test_the_readings_by_marker_from_arrays_measurements_and_data(reduced):
    (x, c, m), two, _ = reduced
    ids = [f"u{k}" for k in range(len(x))]
    readings = Measurements(
        {
            name: {i: dict(r[name]) for i, r in zip(ids, m)}
            for name in ("vibration", "temperature")
        },
        units={"vibration": "mm/s", "temperature": "degC"},
    )
    fitter = JointPH(
        Weibull,
        {"vibration": LinearPath, "temperature": LinearPath},
        varying={"temperature": ["a"]},
        link={"vibration": "value", "temperature": []},
        random_effects="by_marker",
    )
    from_m = fitter.fit(x, c=c, i=ids, m=readings)
    from_data = fitter.fit(SurpyvalData(x, c=c, i=ids, m=readings))
    for model in (from_m, from_data):
        np.testing.assert_allclose(model.params, two.params, rtol=1e-10)
        assert model.marker_unit == {
            "vibration": "mm/s",
            "temperature": "degC",
        }
    assert from_m.effect_units == {"vibration.marker_effect": "mm/s"}
    # a single-marker fitter refuses them, saying how to fit both
    with pytest.raises(ValueError, match="several markers"):
        WeibullJointPH.fit(x, c=c, i=ids, m=readings)
    with pytest.raises(ValueError, match="several markers"):
        WeibullJointPH.fit(SurpyvalData(x, c=c, i=ids, m=readings))


# -- refusals -----------------------------------------------------------------


def test_refusals():
    paths = {"vibration": LinearPath, "temperature": QuadraticPath}
    with pytest.raises(ValueError, match="takes up to 8"):
        JointPH(
            Weibull,
            {"a": QuadraticPath, "b": QuadraticPath, "c": QuadraticPath},
        )
    with pytest.raises(ValueError, match="without a dot"):
        JointPH(Weibull, {"oil.debris": LinearPath})
    with pytest.raises(ValueError, match="not markers"):
        JointPH(Weibull, paths, link={"pressure": "value"})
    with pytest.raises(ValueError, match="no marker"):
        JointPH(Weibull, paths, link={"vibration": [], "temperature": []})
    with pytest.raises(ValueError, match="'<marker>.<parameter>'"):
        JointPH(Weibull, paths, varying=["a"])
    fitter = JointPH(Weibull, paths)
    with pytest.raises(ValueError, match="compares the links of one marker"):
        fitter.compare_links([1.0], markers=[{}])
    x, c, m, _ = simulate(n=20, seed=2)
    # failure modes are taken (test_joint_marker_causes.py); a mode on an
    # item still running is refused as with one marker
    with pytest.raises(ValueError, match="still running"):
        fitter.fit(x, c=c, markers=m, e=["a"] * len(x))
    with pytest.raises(ValueError, match=r"\{marker: \{age: value\}\}"):
        fitter.fit(x, c=c, markers=[r["vibration"] for r in m])
    with pytest.raises(ValueError, match="dict \\{marker: value\\}"):
        fitter.fit(x, c=c, markers=m, marker_reference=3.0)


def test_a_prediction_from_an_item_record_is_refused(reduced):
    from surpyval import ItemLifetime

    _, two, _ = reduced
    item = ItemLifetime("A", right_censored=50.0, markers={0.0: 3.0})
    with pytest.raises(ValueError, match="markers=\\{marker"):
        two.sf(100, item)
    with pytest.raises(ValueError, match="not markers of the model"):
        two.sf(100, markers={"pressure": {0.0: 1.0}}, given=10)


def test_the_accelerated_part_takes_several_markers():
    from surpyval import JointAFT

    x, c, m, _ = simulate(n=80, seed=11)
    paths = {"vibration": LinearPath, "temperature": LinearPath}
    with pytest.raises(ValueError, match="already accumulates"):
        JointAFT(Weibull, paths, link={"vibration": "cumulative"})
    fitter = JointAFT(Weibull, paths, varying=["vibration.a", "temperature.a"])
    model = fitter.fit(x, c=c, markers=m)
    assert model.survival == "AFT"
    assert model.maximum == "verified"
    effects = dict(zip(model.parameter_names, model.params))
    # both markers age the items faster (the fleet's hazard rises in both)
    assert effects["vibration.marker_effect"] > 0
    assert effects["temperature.marker_effect"] > 0
    assert "accelerated failure time" in repr(model)


# -- a lag and a weight of each marker's own -------------------------------


def _reduced_fit(**kw):
    x, c, m, _ = simulate(n=80, seed=7)
    fitter = JointPH(
        Weibull,
        {"vibration": LinearPath, "temperature": LinearPath},
        varying={"temperature": ["a"]},
        link=kw.pop("link", {"vibration": "value", "temperature": []}),
        random_effects="by_marker",
        **kw,
    )
    return fitter, fitter.fit(x, c=c, markers=m)


@pytest.fixture(scope="module")
def lagged():
    """A lag on the vibration only (the temperature not in the hazard),
    given once as one lag for all and once by marker."""
    return _reduced_fit(lag=30.0), _reduced_fit(lag={"vibration": 30.0})


def test_a_lag_by_marker_of_the_one_followed_is_the_lag_for_all(lagged):
    (_, every), (fitter, by_marker) = lagged
    assert fitter.lag == {"vibration": 30.0, "temperature": 0.0}
    assert np.array_equal(by_marker.params, every.params)
    assert by_marker.neg_ll() == every.neg_ll()
    assert "lags vibration 30" in repr(by_marker)
    assert "lag: vibration 30" in repr(fitter)
    restored = surpyval.from_dict(json.loads(json.dumps(by_marker.to_dict())))
    assert restored.lag == {"vibration": 30.0, "temperature": 0.0}
    question = dict(markers={"vibration": {0.0: 3.0, 60.0: 3.4}}, given=60)
    assert np.array_equal(
        restored.sf([100.0, 300.0], **question),
        by_marker.sf([100.0, 300.0], **question),
    )


def test_each_marker_follows_its_own_lag_and_weight():
    from surpyval.univariate.joint._likelihood import MarkerPaths
    from surpyval.univariate.joint._multi import MarkerSet

    paths = MarkerPaths(
        MarkerSet(["vibration", "temperature"], [LinearPath, LinearPath]),
        (("vibration", "value"), ("temperature", "cumulative")),
        {"vibration": 30.0, "temperature": 0.0},
        {"vibration": None, "temperature": ("window", 90.0)},
    )
    vibration, temperature = paths.parts
    assert (vibration.lag, vibration.weight) == (30.0, None)
    assert (temperature.lag, temperature.weight) == (0.0, ("window", 90.0))
    # the integrals split at each marker's lag and window
    assert paths.cuts == (30.0, 90.0)
    ages = np.array([10.0, 50.0, 200.0])
    params = np.broadcast_to([1.0, 0.1, 2.0, 0.05], (3, 4))
    value = paths.terms[0][2](params, ages)
    # the vibration's value 30 later, its age-0 value before
    np.testing.assert_allclose(value, [1.0, 3.0, 18.0])
    area = paths.terms[1][2](params, ages)
    window = lambda t: (  # noqa: E731
        2.0 * (t - max(t - 90, 0)) + 0.025 * (t**2 - max(t - 90, 0) ** 2)
    )
    np.testing.assert_allclose(area, [window(t) for t in ages])


def test_a_lag_or_weight_by_marker_is_checked():
    paths = {"vibration": LinearPath, "temperature": LinearPath}
    with pytest.raises(ValueError, match="not markers"):
        JointPH(Weibull, paths, lag={"pressure": 3.0})
    with pytest.raises(ValueError, match="'lag' is how long"):
        JointPH(Weibull, paths, lag={"vibration": -1.0})
    with pytest.raises(ValueError, match="'temperature' has no such link"):
        JointPH(
            Weibull,
            paths,
            link={
                "vibration": ["value", "cumulative"],
                "temperature": "value",
            },
            weight={"temperature": ("window", 30.0)},
        )
    fitter = JointPH(
        Weibull,
        paths,
        link={"vibration": ["value", "cumulative"], "temperature": "value"},
        weight={"vibration": ("exponential", 50.0)},
        lag={"temperature": 7.0},
    )
    assert fitter.weight == {
        "vibration": ("exponential", 50.0),
        "temperature": None,
    }
    assert "weight: vibration exponential 50" in repr(fitter)
    assert "lag: temperature 7" in repr(fitter)


# -- compare_links of one marker ---------------------------------------------


def test_compare_links_of_one_marker_the_others_as_given(reduced):
    """Each candidate is the fit with the named marker's link and the
    other's as the fitter has them: the value candidate is the reduced
    fit itself."""
    (x, c, m), two, _ = reduced
    fitter = JointPH(
        Weibull,
        {"vibration": LinearPath, "temperature": LinearPath},
        varying={"temperature": ["a"]},
        link={"vibration": "value", "temperature": []},
        random_effects="by_marker",
    )
    comparison = fitter.compare_links(
        x, c=c, markers=m, links=["value", "slope"], marker="vibration"
    )
    assert comparison.marker == "vibration"
    assert sorted(comparison.models) == ["slope", "value"]
    value = comparison.models["value"]
    assert value.link == two.link
    np.testing.assert_allclose(value.params, two.params, rtol=1e-6)
    slope = comparison.models["slope"]
    assert slope.link == (("vibration", "slope"),)
    assert comparison.best_link in (("value",), ("slope",))
    assert "vibration" in comparison.verdict
    assert "the marker's" not in comparison.verdict
    tests = comparison.likelihood_ratio_tests()
    assert len(tests) == 0  # neither link nested in the other


def test_compare_links_of_several_markers_names_the_marker():
    paths = {"vibration": LinearPath, "temperature": LinearPath}
    fitter = JointPH(Weibull, paths)
    with pytest.raises(ValueError, match="marker='vibration'"):
        fitter.compare_links([1.0], markers=[{}])
    with pytest.raises(ValueError, match="marker='vibration'"):
        fitter.compare_links([1.0], markers=[{}], marker="pressure")
    with pytest.raises(ValueError, match="leave marker out"):
        JointPH(Weibull, LinearPath).compare_links([1.0], marker="vibration")


# -- a spline marker's bends without spread ---------------------------------


def test_no_bends_without_spread_warning_by_marker():
    """The warning on a varying bend whose spread is estimated at 0 was
    dropped (one marker or several, ``"<marker>.s1"``): the readings
    rarely tell a small spread from none, so the warning misled. The two
    fits are compared by ``aic()`` instead."""
    from surpyval.univariate.joint import joint_ph

    assert not hasattr(joint_ph, "_warn_bends_without_spread")
    assert not hasattr(joint_ph, "_flat_bends")
