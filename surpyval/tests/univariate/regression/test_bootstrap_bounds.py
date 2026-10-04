"""Parametric bootstrap bounds of the parametric regression models (#617).

``cb``, ``param_cb``, ``quantile_cb`` and ``cb_tvc`` take
``method="bootstrap"``: each resample simulates every unit from the
fitted model at its own covariates (and within its truncation window),
censors it as the unit was censored, and refits; the bounds are the BCa
intervals of the refits. The tests check what is resampled (the design
kept, the censoring), the BCa interval and its acceleration, the refits
shared
between calls and dropped on pickling, and how failed refits are counted.
The coverage on #583's accelerated life test is in
``calibration/test_coverage_regression.py``.
"""

import copy
import pickle
import warnings

import numpy as np
import pytest
from scipy.stats import norm, skew

import surpyval as sp
from surpyval import CovariatePath, WeibullAFT, WeibullPH
from surpyval.tests._helpers import quietly
from surpyval.univariate.regression import _bootstrap
from surpyval.utils.linalg import numerical_gradient, percentile_bounds

BOOT = {"method": "bootstrap", "n_boot": 40, "random_state": 3}


def _ph_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    Z = np.column_stack([rng.binomial(1, 0.5, n), rng.uniform(-1, 1, n)])
    t = 10 * rng.weibull(1.8, n) * np.exp(-(Z @ [0.6, 0.4]) / 1.8)
    cens = rng.uniform(5, 25, n)
    return np.minimum(t, cens), (t > cens).astype(int), Z


@pytest.fixture(scope="module")
def ph():
    x, c, Z = _ph_data()
    return quietly(WeibullPH.fit, x, Z, c=c)


def _recording(monkeypatch, model):
    """Record every refit's data (and pass it on)."""
    calls = []
    fit = model.model.fit

    def recorded(x, Z, **kwargs):
        calls.append((np.array(x), np.array(Z), kwargs))
        return fit(x, Z, **kwargs)

    monkeypatch.setattr(model.model, "fit", recorded)
    return calls


def test_617_bootstrap_is_an_option_and_wald_stays_the_default(ph):
    wald = ph.cb([5.0, 10.0], [1, 0.0])
    np.testing.assert_array_equal(
        wald, ph.cb([5.0, 10.0], [1, 0.0], method="wald")
    )
    boot = ph.cb([5.0, 10.0], [1, 0.0], **BOOT)
    assert boot.shape == wald.shape and not np.allclose(boot, wald)
    with pytest.raises(ValueError, match="'bootstrap'"):
        ph.cb(5.0, [1, 0.0], method="boot")
    with pytest.raises(ValueError, match="'n_boot' must be a positive"):
        ph.cb(5.0, [1, 0.0], method="bootstrap", n_boot=0)


def _bca(draws, est, a, alpha_ci=0.05):
    """Efron's BCa interval, written out: the quantiles of the draws at
    Phi(z0 + (z0 + z) / (1 - a (z0 + z)))."""
    draws = np.asarray(draws, dtype=float)
    share = (np.sum(draws < est) + 0.5 * np.sum(draws == est)) / len(draws)
    z0 = norm.ppf(share)
    levels = [
        norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        for z in norm.ppf([alpha_ci / 2, 1 - alpha_ci / 2])
    ]
    return np.quantile(draws, levels)


def _least_favourable_skewness(model, fits, grad):
    """A sixth of the skewness of grad' cov S over the resamples' scores."""
    L = fits.scores @ (model.covariance() @ grad)
    return skew(L) / 6


def test_617_cb_is_the_bca_interval_of_the_refits(ph):
    x = np.array([2.0, 5.0, 10.0, 20.0])
    z = np.array([1, 0.3])
    sf = ph.cb(x, z, **BOOT)
    fits = ph._bootstrap_refits[(40, 3)]
    assert fits.params.shape == fits.scores.shape == (40, len(ph.params))
    H = np.array([ph.model.Hf(x, z, *p) for p in fits.params])
    H_hat = ph.Hf(x, z)
    for j in range(x.size):
        grad = numerical_gradient(
            lambda p: ph.model.Hf(x[j], z, *p), np.asarray(ph.params)
        )
        a = _least_favourable_skewness(ph, fits, grad)
        lo, hi = _bca(H[:, j], H_hat[j], a)
        np.testing.assert_allclose(sf[j], np.exp([-hi, -lo]), rtol=1e-6)
    # sf, ff and Hf are one interval
    ff = ph.cb(x, z, on="ff", **BOOT)
    Hf = ph.cb(x, z, on="Hf", **BOOT)
    np.testing.assert_allclose(ff, 1 - sf[:, ::-1], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(Hf, -np.log(sf[:, ::-1]), rtol=1e-12)
    # a one-sided bound is the two-sided end at twice the tail
    lower = ph.cb(x, z, alpha_ci=0.05, bound="lower", **BOOT)
    np.testing.assert_allclose(
        lower, ph.cb(x, z, alpha_ci=0.1, **BOOT)[:, 0], rtol=1e-12
    )
    hf = ph.cb(x, z, on="hf", **BOOT)
    h = np.array([ph.model.hf(x, z, *p) for p in fits.params])
    assert np.all((hf[:, 0] >= h.min(0)) & (hf[:, 1] <= h.max(0)))


def test_617_param_cb_and_quantile_cb_bootstrap(ph):
    fits = _bootstrap.refits(ph, 40, 3)
    for i, name in enumerate(ph.parameter_names):
        a = _least_favourable_skewness(ph, fits, np.eye(len(ph.params))[i])
        np.testing.assert_allclose(
            ph.param_cb(name, **BOOT),
            _bca(fits.params[:, i], ph.params[i], a),
            rtol=1e-9,
        )
    upper = ph.param_cb("beta_0", bound="upper", **BOOT)
    assert upper.shape == (1,)
    q = ph.quantile_cb([0.1, 0.5], [1, 0.0], **BOOT)
    # about the refits' own quantiles
    refit = copy.copy(ph)
    t = []
    for p in fits.params:
        refit.params = p
        t.append(refit.qf([0.1, 0.5], [1, 0.0]))
    t = np.array(t)
    t_hat = ph.qf([0.1, 0.5], [1, 0.0])
    assert np.all((q[:, 0] >= t.min(0)) & (q[:, 1] <= t.max(0)))
    assert np.all((q[:, 0] < t_hat) & (t_hat < q[:, 1]))


def test_617_with_no_skew_or_bias_bca_is_the_percentile_interval():
    draws = np.concatenate([np.arange(1.0, 51.0), -np.arange(1.0, 51.0)])
    np.testing.assert_allclose(
        _bootstrap.bca_bounds(
            draws[:, None], np.array([0.0]), 0.0, 0.1, "two-sided"
        )[0],
        percentile_bounds(draws, 0.1),
        rtol=1e-12,
    )


def test_617_the_acceleration_is_efrons_for_an_exponential_scale():
    # With the shape held at 1 and a group indicator, alpha is the scale
    # of an exponential estimated from the n0 units of group 0 alone; the
    # skewness of its score is 2 / sqrt(n0), so a = 1 / (3 sqrt(n0)) (Efron
    # 1987, the exponential/gamma example).
    rng = np.random.default_rng(5)
    Z = np.repeat([0.0, 1.0], 20)
    x = 10 * rng.exponential(size=40) * np.exp(-0.5 * Z)
    model = quietly(WeibullPH.fit, x, Z, fixed={"beta": 1.0})
    fits = _bootstrap.refits(model, 800, 0)
    grad = np.zeros((1, 3))
    grad[0, 0] = 1.0
    a = _bootstrap.acceleration(model, fits, grad)[0]
    assert abs(a - 1 / (3 * np.sqrt(20))) < 0.05, a


def test_617_the_calls_share_the_refits_per_seed(ph, monkeypatch):
    model = quietly(WeibullPH.fit, *_ph_data()[:1], _ph_data()[2])
    calls = _recording(monkeypatch, model)
    model.cb(5.0, [1, 0.0], **BOOT)
    assert len(calls) == 40
    model.param_cb("beta", **BOOT)
    model.quantile_cb(0.1, [0, 0.0], **BOOT)
    model.cb_tvc([5.0], CovariatePath.from_points([0], [[1, 0.0]]), **BOOT)
    assert len(calls) == 40
    # another seed or size draws again; no seed draws every time
    model.cb(5.0, [1, 0.0], method="bootstrap", n_boot=40, random_state=4)
    assert len(calls) == 80
    model.cb(5.0, [1, 0.0], method="bootstrap", n_boot=5)
    model.cb(5.0, [1, 0.0], method="bootstrap", n_boot=5)
    assert len(calls) == 90
    # and new parameters make the kept refits stale
    model.params = model.params * 1.01
    model.cb(5.0, [1, 0.0], **BOOT)
    assert len(calls) == 130


def test_617_the_seed_rule():
    model = quietly(WeibullPH.fit, *_ph_data(30)[:1], _ph_data(30)[2])
    kw = {"method": "bootstrap", "n_boot": 10}
    np.random.seed(0)
    a = model.cb(5.0, [1, 0.0], **kw)
    np.random.seed(0)
    b = model.cb(5.0, [1, 0.0], **kw)
    np.random.seed(1)
    c = model.cb(5.0, [1, 0.0], **kw)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    np.random.seed(1)
    d = model.cb(5.0, [1, 0.0], random_state=7, **kw)
    after = np.random.uniform(size=3)
    np.random.seed(1)
    np.testing.assert_array_equal(after, np.random.uniform(size=3))
    e = model.cb(5.0, [1, 0.0], random_state=np.random.default_rng(7), **kw)
    np.testing.assert_array_equal(d, e)


def test_617_resamples_keep_the_design_and_the_censoring(ph, monkeypatch):
    model = quietly(WeibullPH.fit, *_ph_data()[:1], _ph_data()[2], c=None)
    x, c, Z = _ph_data()
    # Type I: every unit still running at 12 is censored there
    x1, c1 = np.minimum(x, 12.0), (x > 12.0).astype(int)
    model = quietly(WeibullPH.fit, x1, Z, c=c1, n=np.full(len(x), 2))
    calls = _recording(monkeypatch, model)
    model.cb(5.0, [1, 0.0], method="bootstrap", n_boot=20, random_state=0)
    for xs, Zs, kwargs in calls:
        # each row of n = 2 is two units, at the row's covariates
        np.testing.assert_array_equal(Zs, np.repeat(Z, 2, axis=0))
        cs = kwargs["c"]
        assert np.all(xs[cs == 1] == 12.0) and np.all(xs[cs == 0] <= 12.0)
        np.testing.assert_allclose(kwargs["init"], model.params)
        assert kwargs["t"] is None


def test_617_conditional_censoring_of_the_failed_units():
    # Davison & Hinkley's conditional bootstrap: a censored unit keeps its
    # censoring time; a failed unit's is drawn from the product-limit
    # censoring distribution given that it is past the failure.
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    c = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    Z = np.array([0.0, 1, 0, 1, 0, 1, 0, 1])
    model = quietly(WeibullPH.fit, x, Z, c=c)
    design = _bootstrap._Design(model)
    rng = np.random.default_rng(0)
    draws = np.array([design.censoring(rng) for _ in range(20000)])
    np.testing.assert_array_equal(
        draws[:, c == 1], np.tile(x[c == 1], (20000, 1))
    )
    # The censoring distribution: censorings at 2, 4 and 7 (at risk 7, 5
    # and 2; failures are its censored), mass 1/7, 6/35, 12/35, rest 12/35
    # past 7.
    mass = {2.0: 1 / 7, 4.0: 6 / 35, 7.0: 12 / 35, np.inf: 12 / 35}
    for i in np.flatnonzero(c == 0):
        later = {k: v for k, v in mass.items() if k >= x[i]}
        total = sum(later.values())
        for k, v in later.items():
            share = np.mean(draws[:, i] == k)
            assert abs(share - v / total) < 0.015, (x[i], k, share)
        assert np.all(np.isin(draws[:, i], list(later)))


def test_617_truncated_resamples_stay_in_their_windows(monkeypatch):
    rng = np.random.default_rng(1)
    Z = rng.uniform(0, 1, 80)
    t = 10 * rng.weibull(2, 80) * np.exp(-0.5 * Z)
    tl = rng.uniform(0, 4, 80)
    keep = t > tl
    x, Z, tl = t[keep], Z[keep], tl[keep]
    tt = np.column_stack([tl, np.full(x.size, np.inf)])
    model = quietly(WeibullAFT.fit, x, Z, t=tt)
    calls = _recording(monkeypatch, model)
    model.cb(5.0, [0.5], method="bootstrap", n_boot=10, random_state=0)
    for xs, _, kwargs in calls:
        np.testing.assert_array_equal(kwargs["t"], tt)
        assert np.all(xs > tl)


def test_617_refuses_what_it_cannot_resample(ph):
    x, c, Z = _ph_data()
    c2 = c.copy()
    c2[:3] = -1
    left = quietly(WeibullPH.fit, x, Z, c=c2)
    with pytest.raises(ValueError, match="left- or interval-censored"):
        left.cb(5.0, [1, 0.0], method="bootstrap", n_boot=5)
    restored = type(ph).from_dict(ph.to_dict())
    with pytest.raises(ValueError, match="does not keep"):
        restored.cb(5.0, [1, 0.0], method="bootstrap", n_boot=5)
    tvc = quietly(
        WeibullPH.fit_tvc,
        i=[1, 1, 2, 3, 3, 4, 5],
        xl=[0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0],
        xr=[3.0, 5.0, 9.0, 4.0, 8.0, 6.0, 7.0],
        c=[1, 0, 0, 1, 1, 0, 0],
        Z=[[0.0], [1.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
    )
    with pytest.raises(ValueError, match="time-varying covariates"):
        tvc.cb(5.0, [1.0], method="bootstrap", n_boot=5)


def test_617_refits_without_a_maximum_are_kept_counted_and_warned():
    # Few failures in the Z = 1 group: many resamples have none there, so
    # the coefficient runs off (no finite maximum). Those refits are kept
    # at the estimate they reached, at their end of the bounds, and
    # counted in one warning.
    rng = np.random.default_rng(2)
    Z = np.repeat([0.0, 1.0], 20)
    t = 10 * rng.weibull(1.5, 40) * np.exp(-1.5 * Z)
    x, c = np.minimum(t, 10.0), (t > 10.0).astype(int)
    c[Z == 1] = 1
    c[np.flatnonzero(Z == 1)[:2]] = 0
    model = quietly(WeibullPH.fit, x, Z, c=c)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lo, hi = model.param_cb(
            "beta_0", method="bootstrap", n_boot=40, random_state=0
        )
    fits = model._bootstrap_refits[(40, 0)]
    assert fits.no_maximum > 0.02 * 40 and fits.params.shape[0] == 40
    msgs = [str(w.message) for w in caught]
    assert len(msgs) == 1, msgs
    assert f"{fits.no_maximum} with no finite maximum" in msgs[0]
    assert caught[0].filename == __file__
    assert lo < model.params[2] < hi


def test_617_refits_that_raise_are_left_out_and_counted(ph, monkeypatch):
    model = quietly(WeibullPH.fit, *_ph_data()[:1], _ph_data()[2])
    fit = model.model.fit
    count = {"n": 0}

    def flaky(x, Z, **kwargs):
        count["n"] += 1
        if count["n"] % 4 == 0:
            raise ValueError("no")
        return fit(x, Z, **kwargs)

    monkeypatch.setattr(model.model, "fit", flaky)
    with pytest.warns(UserWarning, match="5 failed and left out"):
        model.cb(5.0, [1, 0.0], method="bootstrap", n_boot=20, random_state=0)
    assert model._bootstrap_refits[(20, 0)].params.shape[0] == 15


def test_617_refits_are_not_pickled(ph):
    ph.cb(5.0, [1, 0.0], **BOOT)
    assert ph._bootstrap_refits
    restored = pickle.loads(pickle.dumps(ph))
    assert restored._bootstrap_refits is None
    np.testing.assert_array_equal(
        restored.cb(5.0, [1, 0.0], **BOOT), ph.cb(5.0, [1, 0.0], **BOOT)
    )


def test_617_cb_tvc_bootstrap_along_a_constant_path_is_cb(ph):
    flat = CovariatePath.from_points([0], [[1, 0.3]])
    x = np.array([4.0, 9.0])
    np.testing.assert_allclose(
        ph.cb_tvc(x, flat, **BOOT),
        ph.cb(x, [1, 0.3], **BOOT),
        rtol=1e-9,
    )
    lower = ph.cb_tvc(x, flat, on="Hf", bound="lower", **BOOT)
    assert lower.shape == (2,)


def test_617_bootstrap_of_held_and_centred_parameters():
    x, c, Z = _ph_data()
    held = quietly(WeibullPH.fit, x, Z, c=c, fixed={"beta": 1.8})
    np.testing.assert_array_equal(held.param_cb("beta", **BOOT), [1.8, 1.8])
    assert np.all(np.isfinite(held.param_cb("beta_1", **BOOT)))
    centred = quietly(WeibullAFT.fit, x, Z, c=c, center=True)
    plain = quietly(WeibullAFT.fit, x, Z, c=c)
    # the same predictions, whatever point the baseline is at
    np.testing.assert_allclose(
        centred.cb(5.0, [1, 0.0], **BOOT),
        plain.cb(5.0, [1, 0.0], **BOOT),
        rtol=1e-5,
    )


def test_617_an_accelerated_life_model():
    model = sp.tests._helpers.fitted_accelerated_life_model()
    lo, hi = model.cb(
        5.0, [1.5], method="bootstrap", n_boot=20, random_state=0
    )
    assert lo < model.sf(5.0, [1.5]) < hi
    name = model.parameter_names[-1]
    assert np.all(
        np.isfinite(
            model.param_cb(name, method="bootstrap", n_boot=20, random_state=0)
        )
    )


def test_617_a_model_with_no_finite_maximum_says_its_bounds_mean_nothing():
    # No failures in the Z = 1 group: the coefficient runs off, and data
    # simulated from the fit run off the same way.
    rng = np.random.default_rng(3)
    Z = np.repeat([0.0, 1.0], 15)
    x = 10 * rng.weibull(1.5, 30)
    c = (Z == 1).astype(int)
    model = quietly(WeibullPH.fit, x, Z, c=c)
    assert model.maximum == "no finite maximum"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.cb(5.0, [1.0], method="bootstrap", n_boot=10, random_state=0)
    msgs = [str(w.message) for w in caught]
    assert len(msgs) == 1 and "not a confidence bound" in msgs[0], msgs
