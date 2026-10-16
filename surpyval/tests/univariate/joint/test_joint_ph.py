"""The joint model of a marker and a lifetime, ``WeibullJointPH`` (#891).

The fleet is the issue's: pumps whose vibration is a straight line with
its own start and rate, read monthly with noise sd 0.7, failing at a
Weibull PH hazard in the line's current value with 0.9 per mm/s
(``_fleet.py``); and fleets whose marker follows a nonlinear path
(exponential, and quadratic with a common curvature). The checks: the
fit recovers the truth; with nearly noise-free readings it agrees with
``WeibullPH.fit_tvc`` given the true lines; its predictions keep the
package's rules, agree with an independent brute-force integral and,
for a constant marker, with a plain Weibull PH, and the record and array
forms agree; and it saves, loads and refuses as the rest of the package
does.
"""

import functools
import json
import pickle

import numpy as np
import pytest

import surpyval
from surpyval import (
    ItemLifetime,
    JointModel,
    JointPH,
    LogNormal,
    SurpyvalData,
    Weibull,
    WeibullJointPH,
    WeibullPH,
)
from surpyval.degradation import (
    ExponentialPath,
    LinearPath,
    LinkedPathModel,
    PathModel,
    QuadraticPath,
)
from surpyval.tests.univariate.joint._fleet import (
    CENTRE,
    TRUTH,
    fleet,
    simulate,
    simulate_path,
)


def _truth(model, **extra):
    """The truth in the model's parameterisation: its baseline is at the
    data's mean reading, the truth's at ``CENTRE``."""
    out = dict(TRUTH, **extra)
    shift = model.marker_reference - CENTRE
    out["alpha"] = TRUTH["alpha"] * np.exp(
        -TRUTH["marker_effect"] * shift / TRUTH["beta"]
    )
    return np.array([out[name] for name in model.parameter_names])


@functools.lru_cache(maxsize=None)
def _model():
    return WeibullJointPH.fit(fleet(300, seed=0))


@functools.lru_cache(maxsize=None)
def _small():
    return WeibullJointPH.fit(fleet(80, seed=4))


# -- recovery ---------------------------------------------------------------


def test_recovers_the_fleet():
    model = _model()
    assert model.maximum == "verified"
    truth = _truth(model)
    se = model.standard_errors()
    # every parameter within 3 standard errors of the truth
    z = (model.params - truth) / se
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))
    effect = model.params[-1]
    assert abs(effect / TRUTH["marker_effect"] - 1) < 0.15


def test_recovers_covariates_in_both_parts_and_late_entry():
    data = fleet(300, seed=1, gamma=0.5, delta=-0.4, entry=True)
    assert np.isfinite(data.tl).sum() > 100
    model = WeibullJointPH.fit(data)
    assert model.maximum == "verified"
    assert model.parameter_names == [
        "a",
        "a_z",
        "b",
        "b_z",
        "sd_a",
        "sd_b",
        "corr_a_b",
        "sd_noise",
        "alpha",
        "beta",
        "z",
        "marker_effect",
    ]
    truth = _truth(model, z=0.5, a_z=-0.4, b_z=0.0)
    z = (model.params - truth) / model.standard_errors()
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))


@pytest.mark.calibration
def test_coverage_of_the_marker_effect():
    # A modest study: 60 fleets of 150 pumps. The issue's prototype gave
    # +1% bias and 96% coverage on 300 fleets of 300.
    estimates, covered = [], []
    for seed in range(60):
        model = WeibullJointPH.fit(fleet(150, seed=100 + seed))
        truth = _truth(model)
        se = model.standard_errors()
        estimates.append(model.params[-1])
        covered.append(abs(model.params - truth) <= 1.96 * se)
    bias = np.mean(estimates) / TRUTH["marker_effect"] - 1
    coverage = np.mean(covered, axis=0)
    assert abs(bias) < 0.05, bias
    # 60 fleets: a 95% interval's coverage is within 0.87-1.0 by chance
    assert np.all(coverage >= 0.85), coverage


# -- the limit: readings without noise ---------------------------------------


def test_noise_free_readings_agree_with_fit_tvc_on_the_true_lines():
    # With readings every 10 days and noise sd 0.01, each item's line is
    # known: the joint fit's survival part is then the Weibull PH fit with
    # the true line as a time-varying covariate (given in 1-day steps).
    truth = dict(TRUTH, sd_noise=0.01)
    items, lines = simulate(n=150, seed=11, every=10.0, truth=truth)
    joint = WeibullJointPH.fit(SurpyvalData.from_items(items))
    centre = joint.marker_reference
    i, xl, xr, c, Z = [], [], [], [], []
    for item, (m0, m1) in zip(items, lines):
        end = item.value
        edges = np.append(np.arange(0.0, end, 1.0), end)
        for a, b in zip(edges[:-1], edges[1:]):
            i.append(item.id)
            xl.append(a)
            xr.append(b)
            c.append(int(not (b == end and item.outcome == "observed")))
            Z.append([m0 + m1 * (a + b) / 2 - centre])
    tvc = WeibullPH.fit_tvc(
        np.array(i), np.array(xl), np.array(xr), np.array(c), np.array(Z)
    )
    survival = [
        joint.parameter_names.index(n)
        for n in ("alpha", "beta", "marker_effect")
    ]
    np.testing.assert_allclose(joint.params[survival], tvc.params, rtol=3e-3)
    np.testing.assert_allclose(
        joint.standard_errors()[survival], tvc.standard_errors(), rtol=0.02
    )
    # and the marker's line: noise and lines as simulated
    noise = joint.parameter_names.index("sd_noise")
    assert abs(joint.params[noise] / 0.01 - 1) < 0.05


# -- prediction --------------------------------------------------------------


def _item(**changes):
    item = ItemLifetime(
        "new",
        right_censored=200,
        markers={0: 3.0, 60: 3.3, 120: 3.4, 180: 3.9},
    )
    return item.clone(**changes) if changes else item


def test_survival_at_or_before_the_age_now_is_one():
    model = _small()
    item = _item()
    np.testing.assert_array_equal(model.sf([0, 100, 200], item), 1.0)
    np.testing.assert_array_equal(model.ff([0, 100, 200], item), 0.0)
    np.testing.assert_array_equal(model.cb([0, 200], item), 1.0)
    later = model.sf([201, 300, 500, 900], item)
    assert np.all((later > 0) & (later < 1)) and np.all(np.diff(later) < 0)


def test_shape_in_shape_out():
    model = _small()
    item = _item()
    assert isinstance(model.sf(300, item), np.floating)
    assert model.sf(300.0, item).shape == ()
    assert model.sf([300.0], item).shape == (1,)
    x = np.array([[250.0, 300.0, 350.0], [400.0, 500.0, 600.0]])
    out = model.sf(x, item)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out.ravel(), model.sf(x.ravel(), item))
    assert model.sf(np.empty((0, 2)), item).shape == (0, 2)
    assert model.cb(x, item).shape == (2, 3, 2)
    assert model.cb(x, item, bound="lower").shape == (2, 3)
    assert model.cb(300, item).shape == (2,)
    assert np.ndim(model.cb(300, item, bound="upper")) == 0
    assert model.cb(np.empty(0), item).shape == (0, 2)


def test_nan_in_nan_out():
    model = _small()
    out = model.sf([np.nan, 300.0], _item())
    assert np.isnan(out[0]) and np.isfinite(out[1])
    assert np.isnan(model.cb([np.nan], _item())).all()


def test_bounds_contain_the_estimate_and_follow_their_options():
    model = _small()
    item = _item()
    x = np.array([250.0, 400.0, 700.0])
    sf = model.sf(x, item)
    two = model.cb(x, item)
    assert np.all(two[:, 0] <= sf) and np.all(sf <= two[:, 1])
    assert np.all((two > 0) & (two < 1))
    # a one-sided bound at alpha is that end of the two-sided at 2 alpha
    np.testing.assert_allclose(
        model.cb(x, item, alpha_ci=0.1, bound="lower"),
        two[:, 0] * 0 + model.cb(x, item, alpha_ci=0.2)[:, 0],
    )
    wider = model.cb(x, item, alpha_ci=0.01)
    assert np.all(wider[:, 0] <= two[:, 0]) and np.all(
        wider[:, 1] >= two[:, 1]
    )
    ff = model.cb(x, item, on="ff")
    np.testing.assert_allclose(ff, 1 - two[:, ::-1])
    np.testing.assert_allclose(model.cb(x, item, on="R"), two)
    with pytest.raises(ValueError, match="'on'"):
        model.cb(x, item, on="hf")
    with pytest.raises(ValueError, match="'bound'"):
        model.cb(x, item, bound="both")


def test_bound_derivatives_agree_with_finite_differences():
    # cb's delta method differentiates log(-log sf) by autograd
    import autograd.numpy as anp
    from autograd import make_jvp

    model = _small()
    x = np.array([250.0, 400.0, 700.0])
    f, _, _ = model._log_sf_function(x, _item(), None, None)
    params = np.array(model.params)
    jvp = make_jvp(lambda v: anp.log(-f(v)))(params)
    for j, e in enumerate(np.eye(params.size)):
        h = 1e-6 * abs(params[j])
        up, down = params + h * e, params - h * e
        numeric = (np.log(-f(up)) - np.log(-f(down))) / (2 * h)
        np.testing.assert_allclose(jvp(e)[1], numeric, rtol=1e-4, atol=1e-7)


def test_far_beyond_the_data_without_raw_warnings():
    # The survival need not reach 0: a line whose slope may be negative
    # takes the hazard to 0 far out, as the model says. It falls, to a
    # level, out to ten times the data's ages (beyond, where it has
    # levelled off, it moves by the quadrature's error, 1e-5 of itself).
    import warnings

    model = _small()
    falling = _item(markers={0: 2.0, 90: 1.8, 180: 1.7})
    x = np.geomspace(201, 7300, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for item in (_item(), falling):
            sf = model.sf(x, item)
            assert np.all((sf >= 0) & (sf <= 1))
            assert np.all(np.diff(sf) <= 1e-12)
            far = model.sf([4e4, 1e7], item)
            assert np.all((far >= 0) & (far <= 1))
            model.cb([1e7, 400.0], item)
    assert model.sf(4e4, falling) > 1e-3


def test_readings_move_the_prediction():
    model = _small()
    calm = model.sf(500, _item())
    hot = model.sf(500, _item(markers={0: 3.0, 60: 4.5, 120: 5.5, 180: 6.5}))
    none = model.sf(500, _item(markers=None))
    assert hot < calm
    assert 0 < none < 1


def _brute_force_sf(model, item, x):
    """``P(T > x | T > u, y)`` by a dense grid over the random effects
    and Simpson's rule in time, from the model's parameters alone."""
    names = model.parameter_names
    p = dict(zip(names, model.params))
    v0, v1, r = p["sd_a"] ** 2, p["sd_b"] ** 2, p["corr_a_b"]
    D = np.array([[v0, r * np.sqrt(v0 * v1)], [r * np.sqrt(v0 * v1), v1]])
    Dinv = np.linalg.inv(D)
    g0 = np.linspace(-7, 7, 161) * np.sqrt(v0)
    g1 = np.linspace(-7, 7, 161) * np.sqrt(v1)
    B0, B1 = (a.ravel() for a in np.meshgrid(g0, g1, indexing="ij"))
    m0 = p["a"] + B0
    m1 = p["b"] + B1
    readings = item.markers or {}
    ages = np.array(list(readings), dtype=float)
    ys = np.array(list(readings.values()), dtype=float)
    resid = ys[None, :] - m0[:, None] - m1[:, None] * ages[None, :]
    log_y = -0.5 * np.sum(resid**2, axis=1) / p["sd_noise"] ** 2
    b = np.stack([B0, B1], 1)
    log_prior = -0.5 * np.einsum("ni,ij,nj->n", b, Dinv, b)

    def H(t):
        s = np.linspace(0, t, 801)
        h0 = (p["beta"] / p["alpha"]) * (s / p["alpha"]) ** (p["beta"] - 1)
        h = h0[None, :] * np.exp(
            p["marker_effect"]
            * (m0[:, None] + m1[:, None] * s[None, :] - model.marker_reference)
        )
        w = np.ones(s.size)
        w[1:-1:2], w[2:-1:2] = 4, 2
        return h @ w * (s[1] - s[0]) / 3

    u = item.value
    log_q = log_y + log_prior
    log_q -= log_q.max()
    den = np.sum(np.exp(log_q - H(u)))
    return np.array([np.sum(np.exp(log_q - H(t))) / den for t in x])


def test_prediction_agrees_with_a_brute_force_integral():
    model = _small()
    item = _item(markers={0: 3.0, 90: 3.9, 180: 4.4})
    x = np.array([250.0, 400.0, 600.0])
    np.testing.assert_allclose(
        model.sf(x, item), _brute_force_sf(model, item, x), rtol=2e-4
    )


def _with_params(model, **values):
    d = model.to_dict()
    for name, value in values.items():
        d["params"][model.parameter_names.index(name)] = value
    d.pop("covariance", None)
    return JointModel.from_dict(d)


def test_a_constant_known_marker_is_a_weibull_ph():
    # Readings that are constant, many and almost noise-free: the item's
    # line is known (level, at 3.5), and its survival is that of a
    # Weibull PH with the marker as a constant covariate.
    model = _with_params(_small(), sd_noise=1e-3)
    level, now = 3.5, 300.0
    item = ItemLifetime(
        "flat",
        right_censored=now,
        markers={float(a): level for a in np.arange(0, now + 1, 10)},
    )
    p = dict(zip(model.parameter_names, model.params))
    hr = np.exp(p["marker_effect"] * (level - model.marker_reference))
    x = np.array([350.0, 500.0, 800.0])

    def S(t):
        return Weibull.sf(t, p["alpha"], p["beta"]) ** hr

    np.testing.assert_allclose(model.sf(x, item), S(x) / S(now), rtol=1e-4)


def test_prediction_refusals():
    model = _small()
    with pytest.raises(ValueError, match="not still running"):
        model.sf(300, ItemLifetime("x", observed=200))
    with pytest.raises(ValueError, match="fitted without any"):
        model.sf(300, _item(covariates=[1.0]))
    with pytest.raises(ValueError, match="leave out markers and given"):
        model.sf(300, _item(), markers={0: 1.0})
    with pytest.raises(ValueError, match="Give 'given'"):
        model.sf(300, markers={0: 3.0, 60: 3.1})
    with pytest.raises(ValueError, match="give given=90 or later"):
        model.sf(300, markers={0: 3.0, 90: 3.1}, given=60)
    with pytest.raises(ValueError, match="'given' is an age"):
        model.sf(300, given=-1.0)
    with pytest.raises(ValueError, match="one finite number"):
        model.sf(300, markers={0: np.nan}, given=10)
    with pytest.raises(ValueError, match="leave Z out"):
        model.sf(300, [1.0], given=10)
    with pytest.raises(ValueError, match="a dict"):
        model.sf(300, markers=[3.0], given=10)


def test_prediction_reads_named_covariates_in_the_models_order():
    data = fleet(120, seed=2, gamma=0.5, delta=-0.4)
    model = WeibullJointPH.fit(data)
    item = _item(covariates={"z": 1.0})
    assert 0 < model.sf(400, item) < 1
    np.testing.assert_allclose(
        model.sf(400, item), model.sf(400, _item(covariates=[1.0]))
    )
    with pytest.raises(ValueError, match="no covariates"):
        model.sf(400, _item())
    with pytest.raises(ValueError, match="fitted on"):
        model.sf(400, _item(covariates={"w": 1.0}))
    # the array form: by name, or a row
    by_name = model.sf(400, {"z": 1.0}, markers=_item().markers, given=200)
    assert by_name == model.sf(400, item)
    assert by_name == model.sf(400, [1.0], markers=_item().markers, given=200)
    with pytest.raises(ValueError, match="covariates \\['z'\\]: give"):
        model.sf(400)
    with pytest.raises(ValueError, match="Z names"):
        model.sf(400, {"w": 1.0})
    # a missing covariate gives nan, without warnings
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(model.sf([400.0, 500.0], [np.nan])).all()


# -- the model ---------------------------------------------------------------


def test_comparison_quantities():
    model = _small()
    k = len(model.params)
    assert model.k == k == 9
    assert model.log_likelihood == -model.neg_ll()
    assert model.aic() == pytest.approx(2 * k + 2 * model.neg_ll())
    failures = model.n_events
    assert model.bic() == pytest.approx(
        k * np.log(failures) + 2 * model.neg_ll()
    )
    np.testing.assert_allclose(
        model.standard_errors(), np.sqrt(np.diag(model.covariance()))
    )
    assert model.covariance().shape == (k, k)


def test_neg_ll_is_in_the_datas_units():
    # Rescaling time and the marker changes the log-likelihood by the
    # Jacobian only: per failure log(scale), per reading log(scale).
    items, _ = simulate(n=60, seed=7)
    base = WeibullJointPH.fit(SurpyvalData.from_items(items))
    scaled = [
        it.clone(
            **{it.outcome: it.value * 24.0},
            markers={a * 24.0: 10 * v for a, v in it.markers.items()},
        )
        for it in items
    ]
    other = WeibullJointPH.fit(SurpyvalData.from_items(scaled))
    expected = (
        base.neg_ll()
        + base.n_events * np.log(24.0)
        + base.n_readings * np.log(10)
    )
    assert other.neg_ll() == pytest.approx(expected, rel=1e-6)
    p, q = dict(zip(base.parameter_names, base.params)), dict(
        zip(other.parameter_names, other.params)
    )
    assert q["marker_effect"] == pytest.approx(
        p["marker_effect"] / 10, rel=1e-4
    )
    assert q["alpha"] == pytest.approx(p["alpha"] * 24, rel=1e-4)
    assert q["b"] == pytest.approx(p["b"] * 10 / 24, rel=1e-4)


def test_summary_and_repr():
    model = _small()
    table = model.summary()
    path = "path: Linear (a + b t)"
    survival = "survival: Weibull PH, alpha at marker = {:.6g}".format(
        model.marker_reference
    )
    assert list(table.index) == [
        (path, "a"),
        (path, "b"),
        ("spread between items", "sd_a"),
        ("spread between items", "sd_b"),
        ("spread between items", "corr_a_b"),
        ("noise", "sd_noise"),
        (survival, "alpha"),
        (survival, "beta"),
        ("link: current value", "marker_effect"),
    ]
    np.testing.assert_allclose(table["coef"], model.params)
    effect = table.loc[("link: current value", "marker_effect")]
    assert effect["exp(coef)"] == pytest.approx(np.exp(model.params[-1]))
    lower, upper = table["coef lower 95%"], table["coef upper 95%"]
    assert np.all(lower.to_numpy() < model.params)
    assert np.all(model.params < upper.to_numpy())
    assert lower[("spread between items", "sd_b")] > 0
    assert -1 < lower[("spread between items", "corr_a_b")]
    text = repr(model)
    assert "Path: Linear (a + b t)" in text and "Survival: Weibull PH" in text
    assert "marker_effect" in text


def test_round_trips_through_json_and_pickle():
    model = _small()
    item = _item()
    x = np.array([150.0, 250.0, 400.0, 800.0])
    text = json.dumps(model.to_dict(), allow_nan=False)
    for restored in (
        surpyval.from_dict(json.loads(text)),
        JointModel.from_json(model.to_json()),
        pickle.loads(pickle.dumps(model)),
    ):
        assert type(restored) is JointModel
        np.testing.assert_array_equal(restored.params, model.params)
        np.testing.assert_array_equal(restored.sf(x, item), model.sf(x, item))
        np.testing.assert_array_equal(restored.cb(x, item), model.cb(x, item))
        assert restored.aic() == model.aic() and restored.bic() == model.bic()
        assert restored.maximum == model.maximum


def test_unit_is_kept():
    model = WeibullJointPH.fit(fleet(60, seed=8), unit="days")
    assert model.unit == "day"
    assert JointModel.from_dict(model.to_dict()).unit == "day"


# -- the fitter --------------------------------------------------------------


def test_fitter_repr_and_factory():
    assert repr(WeibullJointPH) == (
        "WeibullJointPH: joint model fitter (Weibull baseline, Linear "
        "path, a + b t)"
    )
    # the path defaults to a line
    assert repr(JointPH(Weibull)) == repr(WeibullJointPH)
    assert repr(JointPH(Weibull, LinearPath)) == repr(WeibullJointPH)
    assert repr(JointPH(Weibull, QuadraticPath, varying=("b", "a"))) == (
        "WeibullQuadraticJointPH: joint model fitter (Weibull baseline, "
        "Quadratic path, a + b t + c t^2, varying: a, b)"
    )
    with pytest.raises(ValueError, match="Weibull baseline for now"):
        JointPH(LogNormal)
    with pytest.raises(ValueError, match="degradation path model"):
        JointPH(Weibull, "linear")
    with pytest.raises(ValueError, match="at least one, each once"):
        JointPH(Weibull, LinearPath, varying=("a", "d"))
    with pytest.raises(ValueError, match="at least one, each once"):
        JointPH(Weibull, LinearPath, varying=())


def test_arrays_give_the_same_fit_as_records():
    data = fleet(60, seed=9)
    by_data = WeibullJointPH.fit(data)
    by_arrays = WeibullJointPH.fit(
        data.x, c=data.c, markers=[data.markers[i] for i in data.ids]
    )
    np.testing.assert_allclose(by_arrays.params, by_data.params, rtol=1e-8)


def test_a_dataframe_names_the_covariates():
    import pandas as pd

    data = fleet(80, seed=10, gamma=0.5, delta=-0.4)
    Z = pd.DataFrame({"site": data.Z[:, 0]})
    model = WeibullJointPH.fit(
        data.x, c=data.c, Z=Z, markers=[data.markers[i] for i in data.ids]
    )
    assert {"a_site", "b_site", "site"} <= set(model.parameter_names)
    unnamed = WeibullJointPH.fit(
        data.x, c=data.c, Z=data.Z, markers=[data.markers[i] for i in data.ids]
    )
    assert {"a_coef_0", "b_coef_0", "coef_0"} <= set(unnamed.parameter_names)


def _base(k, **changes):
    return ItemLifetime(
        k,
        observed=100.0 + k,
        markers={0.0: 1.0 + 0.1 * k, 50.0: 2.0 + 0.05 * k},
    ).clone(**changes)


@pytest.mark.parametrize(
    "changes, match",
    [
        (dict(interval_censored=[50, 90]), "left or interval censored"),
        (dict(left_censored=90), "left or interval censored"),
        (dict(right_truncation=500.0), "right truncated"),
    ],
)
def test_what_v1_cannot_fit_is_refused(changes, match):
    items = [_base(k) for k in range(10)]
    items[3] = items[3].clone(**changes)
    with pytest.raises(ValueError, match=match):
        WeibullJointPH.fit(SurpyvalData.from_items(items))


def test_data_that_cannot_identify_the_model_is_refused():
    running = [_base(k, right_censored=100.0 + k) for k in range(10)]
    with pytest.raises(ValueError, match="at least one failure"):
        WeibullJointPH.fit(SurpyvalData.from_items(running))
    one_line = [_base(0)] + [_base(k, markers={0: 1.0}) for k in range(1, 9)]
    with pytest.raises(ValueError, match="2 or more ages"):
        WeibullJointPH.fit(SurpyvalData.from_items(one_line))
    with pytest.raises(ValueError, match="needs the items' marker readings"):
        WeibullJointPH.fit([10.0, 20.0, 30.0])
    with pytest.raises(ValueError, match="one \\{age: value\\} per item"):
        WeibullJointPH.fit([10.0, 20.0], markers=[{0: 1.0}])


def test_a_surpyval_data_holds_the_markers():
    data = SurpyvalData.from_items([_base(k) for k in range(10)])
    with pytest.raises(TypeError, match="leave out 'markers'"):
        WeibullJointPH.fit(data, markers=[{}] * 10)


def test_time_varying_covariates_are_refused():
    from surpyval import StepSchedule

    steps = StepSchedule.from_changepoints([0, 20], [[0.0], [1.0]])
    items = [_base(k, time_varying_covariates=steps) for k in range(10)]
    with pytest.raises(ValueError, match="time-varying covariates"):
        WeibullJointPH.fit(SurpyvalData.from_items(items))


@pytest.mark.parametrize(
    "name", ["beta", "a", "sd_noise", "marker_effect", "corr_a_b"]
)
def test_a_covariate_named_as_a_parameter_is_refused(name):
    items = [_base(k, covariates={name: float(k % 2)}) for k in range(10)]
    with pytest.raises(ValueError, match="rename them"):
        WeibullJointPH.fit(SurpyvalData.from_items(items))


def test_a_missing_covariate_leaves_its_item_out_with_one_warning():
    data = fleet(80, seed=12, gamma=0.5, delta=-0.4)
    Z = np.array(data.Z, dtype=float)
    Z[[3, 7]] = np.nan
    markers = [data.markers[i] for i in data.ids]
    with pytest.warns(UserWarning, match="Dropped 2 of 80 rows") as w:
        model = WeibullJointPH.fit(data.x, c=data.c, Z=Z, markers=markers)
    caught = [m for m in w if "Dropped" in str(m.message)]
    assert len(caught) == 1
    assert caught[0].filename == __file__  # at the caller's line
    assert model.n_items == 78


# -- other paths -------------------------------------------------------------


def _path_truth(model, truth):
    return np.array([truth.get(name, 0.0) for name in model.parameter_names])


def test_recovers_an_exponential_path():
    # a exp(b t): vibration from 3 mm/s growing 0.08% a day, both varying
    data = simulate_path(
        ExponentialPath, [3.0, 0.0008], [0.4, 0.0003], 0.3, n=300, seed=3
    )
    model = JointPH(Weibull, ExponentialPath).fit(data, marker_reference=4.0)
    assert model.maximum == "verified"
    assert model.parameter_names == [
        "a",
        "b",
        "sd_a",
        "sd_b",
        "corr_a_b",
        "sd_noise",
        "alpha",
        "beta",
        "marker_effect",
    ]
    truth = dict(
        a=3.0,
        b=0.0008,
        sd_a=0.4,
        sd_b=0.0003,
        corr_a_b=0.0,
        sd_noise=0.3,
        alpha=800.0,
        beta=2.0,
        marker_effect=0.9,
    )
    z = (model.params - _path_truth(model, truth)) / model.standard_errors()
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))


#: The exponential path with its rate on a log link: log(b) normal
#: between items (b log-normal, so positive), about log(0.0008), sd 0.3.
LOG_RATE = dict(
    a=3.0,
    **{"log(b)": np.log(0.0008)},
    sd_a=0.4,
    **{"sd_log(b)": 0.3, "corr_a_log(b)": 0.0},
    sd_noise=0.3,
    alpha=800.0,
    beta=2.0,
    marker_effect=0.9,
)


def _log_rate_path():
    return LinkedPathModel(ExponentialPath, {"b": "log"})


@functools.lru_cache(maxsize=None)
def _log_rate_model():
    data = simulate_path(
        _log_rate_path(),
        [3.0, np.log(0.0008)],
        [0.4, 0.3],
        0.3,
        n=300,
        seed=1,
    )
    return JointPH(Weibull, _log_rate_path()).fit(data, marker_reference=4.0)


def test_recovers_a_path_with_its_rate_on_a_log_link():
    model = _log_rate_model()
    assert model.maximum == "verified"
    # the path's own names: the log-linked rate, its spread and its
    # correlation are on the log scale, and named so
    assert model.parameter_names == list(LOG_RATE)
    z = (model.params - _path_truth(model, LOG_RATE)) / (
        model.standard_errors()
    )
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))
    assert "a exp(b t); b on a log link" in repr(model)
    assert "a exp(b t); b on a log link" in str(model.summary().index[0][0])


def test_a_path_on_a_log_link_predicts_and_round_trips():
    model = _log_rate_model()
    item = ItemLifetime(
        "P", right_censored=365, markers={0: 3.0, 120: 3.3, 240: 3.7}
    )
    x = [425.0, 545.0]
    ff = model.ff(x, item)
    assert np.array_equal(ff, model.ff(x, markers=item.markers, given=365))
    assert np.all((ff > 0) & (ff < 1)) and ff[0] < ff[1]
    lower, upper = model.cb(x, item, on="ff").T
    assert np.all((lower < ff) & (ff < upper))
    new = model.ff([365.0, 730.0])
    assert np.all((new > 0) & (new < 1))
    for copy in (
        pickle.loads(pickle.dumps(model)),
        JointModel.from_dict(json.loads(json.dumps(model.to_dict()))),
    ):
        assert copy.parameter_names == model.parameter_names
        assert np.allclose(copy.ff(x, item), ff, rtol=1e-12, atol=0)
        assert np.allclose(
            copy.cb(x, item), model.cb(x, item), rtol=1e-12, atol=0
        )
        assert copy.path_model.links == {"a": "identity", "b": "log"}


def test_a_log_linked_rate_shifted_by_a_covariate_is_named_so():
    import pandas as pd

    data = simulate_path(
        _log_rate_path(), [3.0, np.log(0.0008)], [0.4, 0.3], 0.3, n=120
    )
    markers = [data.markers[i] for i in data.ids]
    z = np.arange(len(markers)) % 2.0
    names = (
        JointPH(Weibull, _log_rate_path(), varying=("a",))
        .fit(data.x, c=data.c, Z=pd.DataFrame({"z": z}), markers=markers)
        .parameter_names
    )
    assert names[:4] == ["a", "a_z", "log(b)", "log(b)_z"]
    assert "sd_a" in names and "sd_log(b)" not in names


def test_recovers_a_quadratic_path_with_a_common_curvature():
    # a + b t + c t^2, the level and rate varying, the curvature common
    data = simulate_path(
        QuadraticPath,
        [3.0, 0.5 / 365, 1.0 / 365**2],
        [0.5, 0.3 / 365, 0.0],
        0.4,
        n=200,
        seed=2,
    )
    fitter = JointPH(Weibull, QuadraticPath, varying=("a", "b"))
    model = fitter.fit(data, marker_reference=4.0)
    assert model.maximum == "verified"
    assert model.parameter_names == [
        "a",
        "b",
        "c",
        "sd_a",
        "sd_b",
        "corr_a_b",
        "sd_noise",
        "alpha",
        "beta",
        "marker_effect",
    ]
    truth = dict(
        a=3.0,
        b=0.5 / 365,
        c=1.0 / 365**2,
        sd_a=0.5,
        sd_b=0.3 / 365,
        corr_a_b=0.0,
        sd_noise=0.4,
        alpha=800.0,
        beta=2.0,
        marker_effect=0.9,
    )
    z = (model.params - _path_truth(model, truth)) / model.standard_errors()
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))
    # predictions of a nonlinear path agree in both forms
    item = ItemLifetime("q", right_censored=150, markers={0: 3.1, 150: 3.6})
    x = np.array([200.0, 300.0])
    np.testing.assert_array_equal(
        model.sf(x, item), model.sf(x, markers=item.markers, given=150)
    )
    assert np.all(np.diff(model.sf([150, 200, 300, 400], item)) < 0)


class SquareRootPath(PathModel):
    """y = a + b sqrt(t): a path of one's own (module level, to pickle)."""

    name = "Square-root"
    parameter_names = ["a", "b"]

    def path(self, x, a, b):
        return a + b * np.sqrt(np.asarray(x, dtype=float))

    def inv_path(self, y, a, b):
        return ((np.asarray(y, dtype=float) - a) / b) ** 2

    def _initial_guess(self, x, y):
        return [float(y[0]), 0.1]


def test_a_path_model_of_ones_own():
    # no analytic Jacobian: the base class's (central differences)
    root = SquareRootPath()
    items, _ = simulate(n=80, seed=5)
    data = SurpyvalData.from_items(items)
    model = JointPH(Weibull, root).fit(data)
    assert model.maximum == "verified"
    assert model.summary().index[0] == (
        "path: Square-root (path(t; a, b))",
        "a",
    )
    item = _item()
    assert 0 < model.sf(400, item) < 1
    again = pickle.loads(pickle.dumps(model))
    assert again.sf(400, item) == model.sf(400, item)
    with pytest.raises(NotImplementedError, match="not a built-in"):
        model.to_dict()


def test_path_parameter_names_must_not_clash():
    from surpyval.degradation import PathModel

    class Clash(PathModel):
        name = "Clash"
        parameter_names = ["alpha", "b"]

        def path(self, x, a, b):
            return a + b * np.asarray(x, dtype=float)

        def inv_path(self, y, a, b):
            return (y - a) / b

    with pytest.raises(ValueError, match="rename them in the path model"):
        JointPH(Weibull, Clash())


# -- the marker reference ----------------------------------------------------


def test_the_marker_reference_moves_alpha_only():
    data = fleet(80, seed=4)
    default = _small()
    at_four = WeibullJointPH.fit(data, marker_reference=4.0)
    assert at_four.marker_reference == 4.0
    assert default.marker_reference == pytest.approx(
        np.mean(
            np.concatenate([list(m.values()) for m in data.markers.values()])
        )
    )
    i = default.parameter_names.index("alpha")
    others = np.arange(len(default.params)) != i
    np.testing.assert_allclose(
        at_four.params[others], default.params[others], rtol=1e-7
    )
    p = dict(zip(default.parameter_names, default.params))
    shift = p["marker_effect"] * (4.0 - default.marker_reference) / p["beta"]
    assert at_four.params[i] == pytest.approx(
        default.params[i] * np.exp(-shift), rel=1e-7
    )
    assert at_four.neg_ll() == pytest.approx(default.neg_ll(), rel=1e-10)
    np.testing.assert_allclose(
        at_four.sf([250.0, 500.0], _item()),
        default.sf([250.0, 500.0], _item()),
        rtol=1e-7,
    )
    assert "alpha at marker = 4" in at_four.summary().index[6][0]
    assert JointModel.from_dict(at_four.to_dict()).marker_reference == 4.0
    with pytest.raises(ValueError, match="one finite number"):
        WeibullJointPH.fit(data, marker_reference=[4.0])


# -- the array form of prediction --------------------------------------------


def test_the_item_and_array_forms_are_identical():
    model = _small()
    item = _item()
    x = np.array([[150.0, 250.0], [400.0, 800.0]])
    args = dict(markers=item.markers, given=item.value)
    for name in ("sf", "ff", "cb"):
        f = getattr(model, name)
        np.testing.assert_array_equal(f(x, item), f(x, **args))
        np.testing.assert_array_equal(f(x, item), f(x, None, **args))
    np.testing.assert_array_equal(
        model.cb(x, item, on="ff", alpha_ci=0.1, bound="lower"),
        model.cb(x, on="ff", alpha_ci=0.1, bound="lower", **args),
    )


def test_a_new_item_is_the_population_average():
    model = _small()
    x = np.array([200.0, 500.0, 900.0])
    new = model.sf(x)
    # the brute force with no readings, from age 0
    blank = ItemLifetime("blank", right_censored=1e-9)
    np.testing.assert_allclose(
        new, _brute_force_sf(model, blank, x), rtol=2e-4
    )
    # known only to have reached 300: conditional on that
    later = model.sf([300.0, 500.0, 900.0], given=300)
    np.testing.assert_allclose(later[1:], new[1:] / model.sf(300.0), rtol=1e-6)
    assert later[0] == 1.0
    assert np.all(np.diff(new) < 0)


# -- aliasing ----------------------------------------------------------------


def test_a_repeated_covariate_is_aliased():
    import warnings

    data = fleet(120, seed=2, gamma=0.5, delta=-0.4)
    markers = [data.markers[i] for i in data.ids]
    Z = np.asarray(data.Z, dtype=float)
    ref = WeibullJointPH.fit(data.x, c=data.c, Z=Z, markers=markers)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = WeibullJointPH.fit(
            data.x, c=data.c, Z=np.c_[Z, Z], markers=markers
        )
    said = [
        str(w.message)
        for w in caught
        if "cannot be estimated" in str(w.message)
    ]
    assert len(said) == 1 and said[0].startswith("Covariate column(s) 1 of Z")
    np.testing.assert_array_equal(model.aliased, [1])
    names = model.parameter_names
    for name in ("a_coef_1", "b_coef_1", "coef_1"):
        assert np.isnan(model.params[names.index(name)])
    kept = [names.index(n) for n in ref.parameter_names]
    np.testing.assert_allclose(model.params[kept], ref.params, rtol=1e-8)
    assert model.k == ref.k
    item = _item(covariates=[1.0])
    np.testing.assert_allclose(
        model.sf([300.0, 500.0], [1.0, 7.0], markers=item.markers, given=200),
        ref.sf([300.0, 500.0], item),
        rtol=1e-8,
    )
