"""The degradation level of the process models, drawn paths and
``with_params``.

``level_at(t)`` is the distribution of the level at one time, from new or
from a reading ``y0`` at an age ``t0``; ``random_path`` draws the levels at
several times; ``with_params`` rebuilds a model from a flat parameter
vector in the order of ``covariance_names``. The checks: the closed forms
against scipy, the identities linking the level to the first-passage life
(exact for the monotone gamma process; for the Wiener process the
reflection-principle term that the level misses), the drawn paths against
``level_at`` (Kolmogorov-Smirnov and moments of the marginals and the
increments, fixed seeds), the round trip through ``with_params`` and the
spread of models rebuilt from covariance draws against the delta method.
"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from surpyval import StepSchedule
from surpyval.degradation import (
    GammaProcess,
    GammaProcessModel,
    ProcessLevel,
    WienerProcess,
    WienerProcessModel,
)

MU, SIGMA = 0.5, 1.0
ALPHA, BETA = 2.0, 4.0
THRESHOLD = 100.0

# Stress: z = 1/T, three levels; AF(z) = exp(G (z - z_ref)).
Z_LEVELS = 1.0 / np.array([323.0, 348.0, 373.0])
G = -5000.0
Z_REF = Z_LEVELS[0]


def _wiener(**kw):
    return WienerProcessModel(MU, SIGMA, THRESHOLD, **kw)


def _gamma(**kw):
    return GammaProcessModel(ALPHA, BETA, THRESHOLD, **kw)


def _stressed(kind):
    kw = dict(gamma=[G], stress_ref=[Z_REF])
    return _wiener(**kw) if kind == "wiener" else _gamma(**kw)


def _af(z):
    return float(np.exp(G * (z - Z_REF)))


# --- closed forms ------------------------------------------------------------


@pytest.mark.parametrize("y0, t0", [(None, 0.0), (30.0, 50.0), (-5.0, 0.0)])
def test_wiener_level_is_normal(y0, t0):
    model = _wiener()
    level = model.level_at(120.0, y0=y0, t0=t0)
    start = 0.0 if y0 is None else y0
    ref = stats.norm(start + MU * (120.0 - t0), SIGMA * np.sqrt(120.0 - t0))
    y = np.linspace(-20.0, 140.0, 33)
    p = np.array([0.001, 0.05, 0.5, 0.95, 0.999])
    np.testing.assert_allclose(level.sf(y), ref.sf(y), rtol=1e-12, atol=0)
    np.testing.assert_allclose(level.ff(y), ref.cdf(y), rtol=1e-12, atol=0)
    np.testing.assert_allclose(level.df(y), ref.pdf(y), rtol=1e-12, atol=0)
    np.testing.assert_allclose(level.qf(p), ref.ppf(p), rtol=1e-12)
    assert level.mean() == pytest.approx(ref.mean(), rel=1e-12)
    assert isinstance(level, ProcessLevel) and level.family == "normal"


@pytest.mark.parametrize("y0, t0", [(None, 0.0), (30.0, 50.0), (7.5, 0.0)])
def test_gamma_level_is_shifted_gamma(y0, t0):
    model = _gamma()
    level = model.level_at(120.0, y0=y0, t0=t0)
    start = 0.0 if y0 is None else y0
    ref = stats.gamma(ALPHA * (120.0 - t0), loc=start, scale=1.0 / BETA)
    y = np.linspace(start, 120.0, 33)
    p = np.array([0.001, 0.05, 0.5, 0.95, 0.999])
    np.testing.assert_allclose(level.sf(y), ref.sf(y), rtol=1e-12, atol=0)
    np.testing.assert_allclose(level.ff(y), ref.cdf(y), rtol=1e-12, atol=0)
    np.testing.assert_allclose(level.df(y), ref.pdf(y), rtol=1e-12, atol=0)
    np.testing.assert_allclose(level.qf(p), ref.ppf(p), rtol=1e-12)
    assert level.mean() == pytest.approx(ref.mean(), rel=1e-12)
    assert level.family == "gamma"


def test_the_default_start_is_the_models_y0():
    model = _gamma(y0=10.0)
    level = model.level_at(40.0)
    assert level.y0 == 10.0 and level.t0 == 0.0
    assert level.mean() == pytest.approx(10.0 + 40.0 * ALPHA / BETA)


@pytest.mark.parametrize("kind", ["wiener", "gamma"])
def test_constant_stress_scales_the_clock(kind):
    model = _stressed(kind)
    z = Z_LEVELS[1]
    level = model.level_at(80.0, [z], y0=3.0, t0=20.0)
    assert level.exposure == pytest.approx(60.0 * _af(z), rel=1e-12)
    base = (_wiener() if kind == "wiener" else _gamma()).level_at(
        3.0 + level.exposure, y0=3.0, t0=3.0
    )
    y = np.linspace(0.0, 40.0, 9)
    np.testing.assert_allclose(level.sf(y), base.sf(y), rtol=1e-12, atol=0)


@pytest.mark.parametrize("kind", ["wiener", "gamma"])
def test_step_schedule_integrates_the_clock(kind):
    model = _stressed(kind)
    profile = StepSchedule.from_changepoints([0.0, 100.0], Z_LEVELS[:2, None])
    # from 50 to 150: 50 at the first stress, 50 at the second
    level = model.level_at(150.0, profile, y0=1.0, t0=50.0)
    expected = 50.0 * _af(Z_LEVELS[0]) + 50.0 * _af(Z_LEVELS[1])
    assert level.exposure == pytest.approx(expected, rel=1e-12)


def test_a_dataframe_stress_is_read_by_name():
    rng = np.random.default_rng(3)
    rows = []
    for unit in range(12):
        z = Z_LEVELS[unit % 3]
        steps = rng.gamma(ALPHA * 10.0 * _af(z), 1.0 / BETA, 8)
        y = np.r_[0.0, np.cumsum(steps)]
        for k in range(9):
            rows.append(dict(x=10.0 * k, y=y[k], i=unit, inv_T=z))
    df = pd.DataFrame(rows)
    model = GammaProcess.fit_from_df(
        df, Z_cols="inv_T", threshold=THRESHOLD, stress_ref=[Z_REF]
    )
    by_name = model.level_at(30.0, pd.DataFrame({"inv_T": [Z_LEVELS[2]]}))
    by_row = model.level_at(30.0, [Z_LEVELS[2]])
    assert by_name.to_dict() == by_row.to_dict()


# --- the level and the life --------------------------------------------------


@pytest.mark.parametrize("y0", [0.0, 12.0])
def test_gamma_level_below_threshold_is_survival(y0):
    # A monotone path has not failed by t exactly when its level at t is
    # below the threshold: P(level(t) < D) = P(T > t).
    model = _gamma(y0=y0)
    for t in [1.0, 50.0, 150.0, 199.0, 220.0, 300.0]:
        level = model.level_at(t)
        assert level.ff(THRESHOLD) == pytest.approx(
            float(model.sf(t)), rel=1e-12, abs=1e-300
        )
        assert level.sf(THRESHOLD) == pytest.approx(
            float(model.ff(t)), rel=1e-10, abs=1e-15
        )


@pytest.mark.parametrize(
    "Z",
    [
        [Z_LEVELS[1]],
        StepSchedule.from_changepoints(
            [0, 30], [[Z_LEVELS[0]], [Z_LEVELS[2]]]
        ),
    ],
)
def test_gamma_level_is_survival_under_stress(Z):
    model = _stressed("gamma")
    for t in [10.0, 40.0, 80.0]:
        assert model.level_at(t, Z).ff(THRESHOLD) == pytest.approx(
            float(model.sf(t, Z)), rel=1e-12, abs=1e-300
        )


def test_wiener_level_misses_the_paths_that_fell_back():
    # The reflection principle: P(T <= t) = P(W(t) >= D) + exp(2 mu D /
    # sigma^2) Phi(-(D + mu t) / (sigma sqrt t)). The level at t is the
    # first term only, so it understates the probability of failure.
    model = _wiener(y0=10.0)
    D = THRESHOLD - 10.0
    for t in [60.0, 150.0, 180.0, 250.0]:
        level_sf = float(model.level_at(t).sf(THRESHOLD))
        missed = np.exp(
            2.0 * MU * D / SIGMA**2
            + stats.norm.logcdf(-(D + MU * t) / (SIGMA * np.sqrt(t)))
        )
        assert float(model.ff(t)) == pytest.approx(level_sf + missed, rel=1e-9)
        assert level_sf < float(model.ff(t))


# --- drawing -----------------------------------------------------------------


def test_level_random_inverts_qf():
    level = _gamma().level_at(30.0, y0=2.0, t0=10.0)
    u = np.random.default_rng(11).uniform(size=50)
    np.testing.assert_array_equal(
        level.random(50, random_state=11), level.qf(u)
    )
    draws = level.random(4000, random_state=12)
    assert stats.kstest(draws, level.ff).pvalue > 0.001


@pytest.mark.parametrize("kind", ["wiener", "gamma"])
@pytest.mark.parametrize("stressed", [False, True])
def test_random_path_marginals_match_level_at(kind, stressed):
    if stressed:
        model = _stressed(kind)
        Z = StepSchedule.from_changepoints(
            [0.0, 40.0], [[Z_LEVELS[0]], [Z_LEVELS[1]]]
        )
    else:
        model = _wiener() if kind == "wiener" else _gamma()
        Z = None
    times = np.array([25.0, 40.0, 60.0, 100.0])
    paths = model.random_path(
        times, 4000, Z, random_state=2026, y0=4.0, t0=10.0
    )
    assert paths.shape == (4000, 4)
    for k, t in enumerate(times):
        level = model.level_at(t, Z, y0=4.0, t0=10.0)
        col = paths[:, k]
        assert stats.kstest(col, level.ff).pvalue > 0.001, (kind, t)
        sd = level.to_dict()["sd"]
        # the mean to within 4 standard errors
        assert abs(col.mean() - level.mean()) < 4.0 * sd / np.sqrt(col.size)
        # the variance to within 15%
        assert col.var() == pytest.approx(sd**2, rel=0.15)


@pytest.mark.parametrize("kind", ["wiener", "gamma"])
def test_random_path_increments_are_independent(kind):
    model = _wiener() if kind == "wiener" else _gamma()
    times = np.array([20.0, 50.0, 60.0, 90.0])
    paths = model.random_path(times, 5000, random_state=7)
    steps = np.diff(np.c_[np.zeros(5000), paths], axis=1)
    edges = np.r_[0.0, times]
    for k in range(times.size):
        # each step is a fresh process run over its own interval
        step = model.level_at(edges[k + 1], y0=0.0, t0=edges[k])
        assert stats.kstest(steps[:, k], step.ff).pvalue > 0.001, k
    corr = np.corrcoef(steps, rowvar=False)
    off = corr[~np.eye(times.size, dtype=bool)]
    # |r| < 4 / sqrt(n) for independent columns
    assert np.abs(off).max() < 4.0 / np.sqrt(5000)


def test_gamma_paths_only_rise():
    paths = _gamma().random_path(np.linspace(1, 100, 50), 200, random_state=1)
    assert (np.diff(paths, axis=1) >= 0).all()


def test_paths_are_reproducible():
    model = _wiener()
    a = model.random_path([1.0, 2.0], 5, random_state=3)
    b = model.random_path([1.0, 2.0], 5, random_state=3)
    np.testing.assert_array_equal(a, b)
    np.random.seed(4)
    c = model.random_path([1.0, 2.0], 5)
    np.random.seed(4)
    d = model.random_path([1.0, 2.0], 5)
    np.testing.assert_array_equal(c, d)


def test_a_path_starting_at_t0_reads_y0_there():
    paths = _wiener().random_path(
        [5.0, 9.0], 10, random_state=1, y0=2.5, t0=5.0
    )
    np.testing.assert_array_equal(paths[:, 0], 2.5)


# --- the level at t0, and refusals -------------------------------------------


def test_the_level_at_t0_is_y0_exactly():
    level = _wiener().level_at(5.0, y0=2.0, t0=5.0)
    np.testing.assert_array_equal(level.sf([1.9, 2.0, 2.1]), [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(level.ff([1.9, 2.0, 2.1]), [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(level.qf([0.0, 0.5, 1.0]), [2.0, 2.0, 2.0])
    assert level.mean() == 2.0
    with pytest.raises(ValueError, match="no density"):
        level.df(2.0)


def test_missing_values_give_nan():
    level = _gamma().level_at(10.0)
    assert np.isnan(level.sf(np.nan)) and np.isnan(level.ff(np.nan))
    assert np.isnan(level.qf(np.nan))
    with pytest.warns(UserWarning):
        assert np.isnan(level.qf(1.5))


def test_level_above_threshold_is_allowed():
    # The level process does not stop at the threshold; a reading past it
    # is a level like any other.
    level = _wiener().level_at(20.0, y0=120.0, t0=10.0)
    assert level.mean() == pytest.approx(125.0)


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda m: m.level_at(5.0, y0=1.0, t0=10.0), "before t0"),
        (lambda m: m.level_at(20.0, t0=10.0), "needs y0"),
        (lambda m: m.level_at(np.nan), "finite time"),
        (lambda m: m.level_at(np.inf), "finite time"),
        (lambda m: m.level_at([1.0, 2.0]), "one time"),
        (lambda m: m.level_at(5.0, y0=np.nan), "finite"),
        (lambda m: m.level_at(5.0, y0=1.0, t0=-1.0), "at or after 0"),
        (lambda m: m.random_path([2.0, 1.0], 3), "strictly increasing"),
        (lambda m: m.random_path([1.0, np.nan], 3), "finite"),
        (lambda m: m.random_path([[1.0, 2.0]], 3), "one-dimensional"),
        (lambda m: m.random_path([1.0, 2.0], 0), "positive whole"),
        (lambda m: m.random_path([1.0, 2.0], 2.5), "positive whole"),
        (lambda m: m.random_path([1.0], 2, y0=0.0, t0=3.0), "before t0"),
        (lambda m: m.level_at(5.0, [Z_LEVELS[0]]), "without stress"),
    ],
)
def test_refusals_say_what_to_do(call, match):
    with pytest.raises(ValueError, match=match):
        call(_wiener())


def test_a_stressed_model_needs_its_stress():
    model = _stressed("gamma")
    with pytest.raises(ValueError, match="pass Z"):
        model.level_at(5.0)
    with pytest.raises(ValueError, match="missing"):
        model.level_at(5.0, [np.nan])
    with pytest.raises(ValueError, match="pass Z"):
        model.random_path([1.0, 2.0], 3)


def test_to_dict_is_json_ready():
    out = _gamma().level_at(20.0, y0=1.0, t0=4.0).to_dict()
    assert json.loads(json.dumps(out)) == out
    assert out["params"] == {"shape": ALPHA * 16.0, "rate": BETA}
    assert out["mean"] == pytest.approx(1.0 + 16.0 * ALPHA / BETA)


# --- with_params -------------------------------------------------------------


def _paths(kind, seed, units=30, Z=False):
    rng = np.random.default_rng(seed)
    xs, ys, ii, zz = [], [], [], []
    for u in range(units):
        z = Z_LEVELS[u % 3]
        af = _af(z) if Z else 1.0
        dtau = np.full(10, 5.0) * af
        if kind == "wiener":
            steps = rng.normal(0.4 * dtau, 0.8 * np.sqrt(dtau))
        else:
            steps = rng.gamma(0.8 * dtau, 1.0 / 2.0)
        xs.append(np.arange(11) * 5.0)
        ys.append(np.r_[0.0, np.cumsum(steps)])
        ii.append(np.full(11, u))
        zz.append(np.full(11, z))
    out = [np.concatenate(v) for v in (xs, ys, ii)]
    return out + [np.concatenate(zz)[:, None]] if Z else out


FIT = {"wiener": WienerProcess, "gamma": GammaProcess}


def _fit(kind, seed, Z=False):
    data = _paths(kind, seed, Z=Z)
    kw = dict(threshold=40.0, y0=0.0)
    if Z:
        x, y, i, z = data
        return FIT[kind].fit(x, y, i, Z=z, stress_ref=[Z_REF], **kw)
    return FIT[kind].fit(*data, **kw)


def _predictions(model, Z):
    t = np.array([5.0, 20.0, 40.0, 60.0])
    return np.concatenate(
        [
            model.sf(t, Z),
            model.df(t, Z),
            model.qf([0.1, 0.5, 0.9], Z),
            [model.mean(Z)],
            model.level_at(30.0, Z).qf([0.1, 0.9]),
        ]
    )


@pytest.mark.parametrize("kind", ["wiener", "gamma"])
@pytest.mark.parametrize("stressed", [False, True])
def test_with_params_round_trip(kind, stressed):
    model = _fit(kind, 1, Z=stressed)
    Z = [Z_LEVELS[1]] if stressed else None
    again = model.with_params(model._estimates())
    assert type(again) is type(model)
    assert again.covariance_names == model.covariance_names
    assert again.y0 == model.y0 and again.threshold == model.threshold
    np.testing.assert_allclose(
        _predictions(again, Z), _predictions(model, Z), rtol=1e-12, atol=0
    )
    # built from parameters: nothing of the fit's data
    assert again.maximum == "not applicable"
    with pytest.raises(ValueError, match="built from its parameters"):
        again.covariance()
    with pytest.raises(ValueError, match="built from its parameters"):
        again.neg_ll()
    with pytest.raises(ValueError, match="built from its parameters"):
        again.cb([10.0], Z)


@pytest.mark.parametrize("kind", ["wiener", "gamma"])
@pytest.mark.parametrize("stressed", [False, True])
def test_with_params_of_a_refit_is_the_refit(kind, stressed):
    model = _fit(kind, 1, Z=stressed)
    refit = _fit(kind, 2, Z=stressed)
    Z = [Z_LEVELS[2]] if stressed else None
    np.testing.assert_allclose(
        _predictions(model.with_params(refit._estimates()), Z),
        _predictions(refit, Z),
        rtol=1e-12,
        atol=0,
    )


def test_with_params_keeps_names_and_unit():
    x, y, i, z = _paths("wiener", 4, Z=True)
    df = pd.DataFrame(dict(x=x, y=y, i=i, inv_T=z[:, 0]))
    model = WienerProcess.fit_from_df(
        df, Z_cols="inv_T", threshold=40.0, stress_ref=[Z_REF], unit="hours"
    )
    other = model.with_params(model._estimates() * [1.1, 0.9, 1.0])
    assert other.covariance_names == ["mu", "sigma", "inv_T"]
    assert other.unit == model.unit == "hour"
    assert other.mu == pytest.approx(1.1 * model.mu)
    np.testing.assert_array_equal(other.stress_ref, model.stress_ref)
    # the original is untouched
    assert model.mu == pytest.approx(model._estimates()[0])


@pytest.mark.parametrize(
    "params, match",
    [
        ([0.5], "flat vector of 2 values"),
        ([0.5, 1.0, 2.0], "flat vector of 2 values"),
        ([[0.5, 1.0]], "flat vector of 2 values"),
        ([0.5, np.nan], "finite"),
    ],
)
def test_with_params_refuses_the_wrong_vector(params, match):
    with pytest.raises(ValueError, match=match):
        _wiener().with_params(params)


def test_with_params_refuses_impossible_values():
    with pytest.raises(ValueError, match="sigma must be positive"):
        _wiener().with_params([0.5, -1.0])
    with pytest.raises(ValueError, match="alpha must be positive"):
        _gamma().with_params([-2.0, 4.0])
    with pytest.raises(ValueError, match="beta must be positive"):
        _gamma().with_params([2.0, 0.0])
    with pytest.raises(ValueError, match="flat vector of 3 values"):
        _stressed("gamma").with_params([2.0, 4.0])


@pytest.mark.parametrize("kind", ["wiener", "gamma"])
def test_covariance_draws_agree_with_the_delta_method(kind):
    # The spread of ff(t) over models rebuilt from covariance draws matches
    # the delta-method standard error behind cb (formed on the logit of ff,
    # so it is read back from the bound's width there).
    model = _fit(kind, 5)
    t = float(model.qf(0.3))
    rng = np.random.default_rng(20261105)
    draws = rng.multivariate_normal(
        model._estimates(), model.covariance(), 4000
    )
    ff = np.array([float(model.with_params(d).ff(t)) for d in draws])
    lo, hi = model.cb([t], on="ff")[0]
    p = float(model.ff(t))
    z = stats.norm.ppf(0.975)
    logit = stats.logistic.ppf
    se = (logit(hi) - logit(lo)) / (2.0 * z) * p * (1.0 - p)
    assert ff.std() == pytest.approx(se, rel=0.1)
