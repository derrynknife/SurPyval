"""The joint model with a lag: the hazard at age ``t`` follows the
marker's path at ``max(t - lag, 0)``, the path held at its age-0 value
until the lag has passed (its slope and area 0 there).

The checks: the split rule and the lagged path against closed forms; the
likelihood of every kind of item, for each link and both survival parts,
against an independent dense-grid integral split at the lag; ``lag=0``
the model without one, number for number; recovery of a fleet simulated
with a lag (``_fleet.simulate_links``, seed 23, chosen before the fit
was seen) and lags compared by AIC; the prediction against a
brute-force integral; saving and loading; the printouts; and the
refusals. The comparison with R's JM on the PBC trial is in
``tests/reference/test_joint_lag.py``; the coverage study in
``tests/calibration/test_coverage_joint_lag.py``.
"""

import functools
import pickle
import warnings

import numpy as np
import pytest

import surpyval
from surpyval import JointAFT, JointPH, Weibull
from surpyval.degradation import LloydLipowPath, LogarithmicPath
from surpyval.degradation.path_models import PATH_MODELS, PathModel
from surpyval.serialisation import required_schema
from surpyval.tests.univariate.joint import test_joint_aft as aft
from surpyval.tests.univariate.joint import test_joint_links as links
from surpyval.tests.univariate.joint._fleet import simulate_links
from surpyval.univariate.joint import JointModel
from surpyval.univariate.joint._likelihood import (
    LaggedPath,
    LinkPath,
    lag_rule,
)
from surpyval.univariate.joint.compare import _candidates
from surpyval.univariate.joint.joint_ph import search_space

#: The lag (years) of the dense-grid checks: inside most of the items'
#: lives and inside two of their intervals.
LAG = 0.6

# -- the rule and the lagged path ---------------------------------------------


def test_the_split_rule_integrates_a_step_at_the_lag_exactly():
    lo, hi = np.array([0.2, 0.0, 0.7]), np.array([1.0, 0.5, 2.0])
    s, logw = lag_rule(lo, hi, 0.6)
    step = (s > 0.6).astype(float)
    np.testing.assert_allclose(
        np.sum(np.exp(logw) * step, axis=1), [0.4, 0.0, 1.3], atol=1e-14
    )
    np.testing.assert_allclose(np.sum(np.exp(logw), axis=1), hi - lo)
    # a piece of no width has no weight
    assert np.all(np.isneginf(logw[1, 15:])) and np.all(
        np.isneginf(logw[2, :15])
    )


def test_the_lagged_path_is_the_path_held_at_age_0_until_the_lag():
    line = PATH_MODELS["linear"]
    x = np.array([0.2, 1.0, 1.5, 4.0])
    a, b = 3.0, 0.4
    value = LaggedPath(line, 1.0, False)
    np.testing.assert_allclose(
        value.path(x, a, b), a + b * np.maximum(x - 1.0, 0)
    )
    slope = LaggedPath(LinkPath(line, "slope"), 1.0, True)
    np.testing.assert_allclose(slope.path(x, a, b), [0, 0, b, b])
    area = LaggedPath(LinkPath(line, "cumulative"), 1.0, True)
    u = np.maximum(x - 1.0, 0)
    np.testing.assert_allclose(
        area.path(x, a, b), a * u + b * u**2 / 2, atol=1e-14
    )
    full = [np.full(x.size, v) for v in (a, b)]
    np.testing.assert_allclose(
        value.jacobian(x, *full), np.column_stack([np.ones(4), u])
    )
    np.testing.assert_allclose(
        slope.jacobian(x, *full),
        np.column_stack([np.zeros(4), [0, 0, 1, 1]]),
    )
    np.testing.assert_allclose(
        area.jacobian(x, *full), np.column_stack([u, u**2 / 2]), atol=1e-14
    )


# -- the likelihood against a dense grid --------------------------------------


@pytest.mark.parametrize("name", sorted(links.LINK_FITS))
def test_every_kind_of_item_agrees_with_a_dense_grid_integral(name):
    model = links._model_at(links.LINK_FITS[name])
    model.lag = LAG
    data = {k: v for k, v in links.YEARS.items() if k != "Z"}
    f, u, _ = search_space(model, **data)
    with np.errstate(divide="ignore"):
        brute = links._brute_force_neg_ll(model, data, lag=LAG)
    # 8 rows, 11 items: to within 1e-5 an item
    assert float(f(u)) == pytest.approx(brute, abs=1.1e-4)


@pytest.mark.parametrize(
    "link, beta",
    [(("value",), 1.4), (("slope",), 0.8), (("value", "slope"), 1.4)],
    ids=["value", "slope-beta-below-1", "value+slope"],
)
def test_every_kind_of_item_agrees_with_a_dense_grid_integral_aft(link, beta):
    model = aft._model_at(link, beta)
    model.lag = LAG
    f, u, _ = search_space(model, **aft.YEARS)
    with np.errstate(divide="ignore"):
        brute = aft._brute_force_neg_ll(model, aft.YEARS, lag=LAG)
    assert float(f(u)) == pytest.approx(brute, abs=1.1e-4)


# -- fits ---------------------------------------------------------------------


@functools.cache
def _lagged():
    """A fleet whose hazard follows the line's value half a year
    earlier."""
    return simulate_links(
        300, seed=23, alpha=3.0, beta=2.0, value=0.9, lag=0.5
    )


@functools.cache
def _fit(lag):
    return JointPH(Weibull, lag=lag).fit(_lagged(), marker_reference=4.0)


def test_a_lag_of_0_is_the_model_without_one():
    data = simulate_links(80, seed=11, alpha=3.0, beta=2.0, value=0.9)
    plain = JointPH(Weibull).fit(data, marker_reference=4.0)
    zero = JointPH(Weibull, lag=0).fit(data, marker_reference=4.0)
    np.testing.assert_array_equal(zero.params, plain.params)
    np.testing.assert_array_equal(zero.covariance(), plain.covariance())
    assert zero.neg_ll() == plain.neg_ll()
    assert zero.lag == 0.0
    assert "lag" not in zero.to_dict()
    np.testing.assert_array_equal(
        zero.sf([0.5, 1.0], markers={0.0: 3.1, 0.5: 3.6}, given=0.5),
        plain.sf([0.5, 1.0], markers={0.0: 3.1, 0.5: 3.6}, given=0.5),
    )


def test_recovers_a_fleet_simulated_with_a_lag():
    model = _fit(0.5)
    assert model.maximum == "verified"
    assert model.lag == 0.5
    truth = dict(
        a=3.0,
        b=1.0,
        sd_a=0.6,
        sd_b=0.5,
        corr_a_b=0.2,
        sd_noise=0.7,
        alpha=3.0,
        beta=2.0,
        marker_effect=0.9,
    )
    target = np.array([truth[n] for n in model.parameter_names])
    z = (model.params - target) / model.standard_errors()
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))


def test_lags_compare_by_aic_on_the_same_parameters():
    # The same parameters, so the criteria differ by the likelihood only.
    # Which lag wins is not asserted: on a straight-line marker a lag is
    # close to a shift of the line (m(t - L) = m(t) - b L), so the data
    # say little about it. On this fleet (simulated with 0.5) the AIC is
    # 16126.81 at 0, 16126.75 at 0.25 and 16127.28 at 0.5.
    with_lag, at_once = _fit(0.5), _fit(0.0)
    assert with_lag.k == at_once.k
    assert with_lag.aic() - at_once.aic() == pytest.approx(
        2 * (with_lag.neg_ll() - at_once.neg_ll())
    )


def test_an_accelerated_model_fits_with_a_lag():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = JointAFT(Weibull, lag=0.5).fit(_lagged(), marker_reference=4.0)
    assert model.lag == 0.5 and model.survival == "AFT"
    assert np.all(np.isfinite(model.params))


def test_prediction_agrees_with_a_brute_force_integral():
    model = links._model_at(("value", "slope"))
    model.lag = LAG
    p = dict(zip(model.parameter_names, model.params))
    readings = {0.0: 3.2, 0.25: 3.6, 0.5: 4.1}
    given, x = 0.5, np.array([0.8, 1.2, 2.0])
    v0, v1, r = p["sd_a"] ** 2, p["sd_b"] ** 2, p["corr_a_b"]
    D = np.array([[v0, r * np.sqrt(v0 * v1)], [r * np.sqrt(v0 * v1), v1]])
    g0 = np.linspace(-7, 7, 201) * np.sqrt(v0)
    g1 = np.linspace(-7, 7, 201) * np.sqrt(v1)
    B0, B1 = (a.ravel() for a in np.meshgrid(g0, g1, indexing="ij"))
    b = np.stack([B0, B1], 1)
    m0, m1 = p["a"] + B0, p["b"] + B1
    ages = np.array(list(readings))
    ys = np.array(list(readings.values()))
    resid = ys[None, :] - m0[:, None] - m1[:, None] * ages[None, :]
    log_w = (
        -0.5 * np.einsum("ni,ij,nj->n", b, np.linalg.inv(D), b)
        - 0.5 * np.sum(resid**2, axis=1) / p["sd_noise"] ** 2
    )

    def hazard(t):
        t = np.atleast_1d(t)[None, :]
        u = np.maximum(t - LAG, 0.0)
        eta = p["marker_effect"] * (
            m0[:, None] + m1[:, None] * u - model.marker_reference
        ) + p["slope_effect"] * m1[:, None] * (t > LAG)
        return (
            (p["beta"] / p["alpha"])
            * (t / p["alpha"]) ** (p["beta"] - 1)
            * np.exp(eta)
        )

    def log_S(t):
        return -links._simpson(hazard, 0.0, t, LAG)

    base = log_w + log_S(given)
    top = base.max()
    expected = [
        np.sum(np.exp(log_w + log_S(t) - top)) / np.sum(np.exp(base - top))
        for t in x
    ]
    got = model.sf(x, markers=readings, given=given)
    np.testing.assert_allclose(got, expected, rtol=2e-5)


# -- saving, printing ---------------------------------------------------------


def test_round_trips_through_json_and_pickle():
    model = _fit(0.5)
    saved = model.to_dict()
    assert saved["lag"] == 0.5
    restored = surpyval.from_dict(saved)
    assert restored.lag == 0.5
    question = dict(markers={0.0: 3.2, 0.5: 3.9}, given=0.6)
    expected = model.sf([1.0, 1.5], **question)
    np.testing.assert_allclose(restored.sf([1.0, 1.5], **question), expected)
    unpickled = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(unpickled.sf([1.0, 1.5], **question), expected)
    # the lag changes the prediction
    without = surpyval.from_dict(
        {k: v for k, v in saved.items() if k != "lag"}
    )
    assert without.lag == 0.0
    assert not np.allclose(without.sf([1.0, 1.5], **question), expected)


def test_a_lagged_model_saves_as_schema_5_which_0_28_refuses():
    # SurPyval 0.28 reads schema 4 and below; it would ignore "lag" and
    # predict as if the hazard followed the marker with no delay.
    assert _fit(0.5).to_dict()["schema"] == 5
    assert _fit(0.0).to_dict()["schema"] < 5


def test_more_than_three_varying_path_parameters_need_schema_5():
    # 0.28 has no prediction grid for four or five varying parameters;
    # the schema makes it refuse at the load rather than fail later.
    saved = _fit(0.0).to_dict()
    assert required_schema(saved) < 5
    for varying in (["a", "b", "c"], ["a", "b", "c", "d"]):
        bigger = dict(saved, varying=varying)
        expected = 5 if len(varying) > 3 else required_schema(saved)
        assert required_schema(bigger) == expected


def test_a_saved_negative_lag_is_refused():
    saved = _fit(0.5).to_dict()
    saved["lag"] = -1.0
    with pytest.raises(ValueError, match="'lag'"):
        JointModel.from_dict(saved)


def test_the_printouts_say_the_lag():
    assert repr(JointPH(Weibull, lag=0.5)).endswith("lag: 0.5)")
    assert "lag" not in repr(JointPH(Weibull))
    assert repr(JointAFT(Weibull, lag=2)).endswith("lag: 2)")
    model = _fit(0.5)
    assert "0.5 earlier (lag)" in repr(model)
    parts = model.summary().index.get_level_values("part")
    assert "link: current value, lag 0.5" in set(parts)


def test_compare_links_keeps_the_lag():
    fitters, _, _ = _candidates(JointPH(Weibull, lag=0.5), None)
    assert [f.lag for f in fitters] == [0.5] * len(fitters)


# -- refusals -----------------------------------------------------------------


@pytest.mark.parametrize("lag", [-1.0, np.nan, np.inf, [1.0, 2.0], "soon"])
def test_a_lag_that_is_not_a_number_of_0_or_more_is_refused(lag):
    with pytest.raises(ValueError, match="'lag' is how long"):
        JointPH(Weibull, lag=lag)


@pytest.mark.parametrize("path", [LogarithmicPath, LloydLipowPath])
def test_a_path_infinite_at_0_cannot_lag_its_value(path):
    with pytest.raises(ValueError, match="infinite at age 0.*lag=0"):
        JointPH(Weibull, path, lag=1.0)
    # with no lag it is the model it was
    JointPH(Weibull, path)
    # the area under a logarithmic path is finite from 0
    JointPH(Weibull, LogarithmicPath, link="cumulative", lag=1.0)


class _Reciprocal(PathModel):
    """y = a + b / t: infinite at age 0 (a path of one's own)."""

    name = "Reciprocal"
    parameter_names = ["a", "b"]

    def path(self, x, a, b):
        x = np.asarray(x, dtype=float)
        return a + b / x

    def inv_path(self, y, a, b):
        return np.full(np.shape(y), np.nan)

    def _initial_guess(self, x, y):
        return [float(np.mean(y)), 0.1]


def test_a_path_of_ones_own_infinite_at_0_is_refused_at_the_fit():
    rng = np.random.default_rng(2)
    items = []
    for k in range(30):
        ages = np.arange(0.25, 2.0, 0.25)
        y = 3.0 + 0.5 / ages + rng.normal(0, 0.1, ages.size)
        items.append(
            surpyval.ItemLifetime(
                k,
                **({"observed": 2.0} if k % 3 else {"right_censored": 2.0}),
                markers=dict(zip(ages, y)),
            )
        )
    data = surpyval.SurpyvalData.from_items(items)
    with pytest.raises(ValueError, match="not finite at age 0"):
        JointPH(Weibull, _Reciprocal(), varying="a", lag=0.5).fit(data)
