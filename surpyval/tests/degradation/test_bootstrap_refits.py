"""The degradation bootstrap's refits (#522).

``DegradationAnalysis.cb(method="bootstrap")`` refits the whole analysis
on each resample of the units. A resampled unit is one of the model's
own, so its path fit is the same computation every time: it is now done
once per unit and reused. The life distribution is warm started from the
full-data estimate, and searched from its default start as well only
where that does not reach a verified maximum (``warm_starts``).
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
import surpyval.degradation.degradation_analysis as da_module
from surpyval.univariate.parametric import parametric_fitter
from surpyval.utils import refits
from surpyval.utils.refits import warm_starts


def _analysis(units=30, path="linear", seed=7):
    rng = np.random.default_rng(seed)
    times = np.arange(1.0, 7.0)
    x, y, i = [], [], []
    for u in range(units):
        slope = np.exp(rng.normal(0.0, 0.3))
        if path == "exponential":
            level = 0.5 * np.exp(0.25 * slope * times)
        else:
            level = slope * times
        x.append(times)
        y.append(level * np.exp(rng.normal(0.0, 0.03, times.size)))
        i.append(np.full(times.size, u))
    x, y, i = map(np.concatenate, (x, y, i))
    threshold = 10.0 if path == "linear" else 5.0
    with warnings.catch_warnings():
        # (the noise-corrected path covariance of so few units is clipped)
        warnings.simplefilter("ignore")
        return sp.DegradationAnalysis.fit(
            x, y, i, threshold=threshold, path=path
        )


@pytest.mark.parametrize("path", ["linear", "exponential"])
def test_each_unit_path_is_fitted_once(monkeypatch, path):
    # 30 units, 20 resamples: 600 path fits before, 30 now.
    model = _analysis(path=path)
    calls = []
    cls = type(model.path_model)
    fit = cls.fit

    def counted(self, *args, **kwargs):
        calls.append(1)
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(cls, "fit", counted)
    model.cb(np.array([6.0, 10.0]), method="bootstrap", n_boot=20)
    assert len(calls) == len(model.units)


def test_every_refit_reaches_a_verified_maximum(monkeypatch):
    model = _analysis()
    lives = []
    fit = da_module.DegradationAnalysis.fit

    def kept(*args, **kwargs):
        out = fit(*args, **kwargs)
        lives.append(out.life_model)
        return out

    monkeypatch.setattr(da_module.DegradationAnalysis, "fit", kept)
    model.cb(np.array([6.0, 10.0]), method="bootstrap", n_boot=20)
    assert len(lives) == 20
    assert all(m.maximum == "verified" for m in lives)
    # A refit outside the bootstrap is cold, as before.
    assert refits.DEGRADATION_REFIT.get() is None


def test_a_warm_start_is_searched_once(monkeypatch):
    # From a start near the maximum, a fit given init no longer also
    # searches from the default start (it still does where the search from
    # init is not verified: ``results["_verified"]``).
    x = np.array([3.1, 4.7, 5.2, 6.9, 8.4, 9.0, 11.3, 12.8, 14.0, 15.5])
    full = sp.Weibull.fit(x)
    searches = []
    mle = parametric_fitter.METHOD_FUNC_DICT["MLE"]

    def counted(model):
        searches.append(1)
        return mle(model)

    monkeypatch.setitem(parametric_fitter.METHOD_FUNC_DICT, "MLE", counted)
    cold = sp.Weibull.fit(x[1:], init=full.params)
    assert len(searches) == 2
    searches.clear()
    with warm_starts():
        warm = sp.Weibull.fit(x[1:], init=full.params)
    assert len(searches) == 1
    assert warm.maximum == "verified"
    np.testing.assert_allclose(warm.params, cold.params, rtol=1e-6)


def test_the_bounds_are_those_of_cold_refits(monkeypatch):
    # The same resamples refitted from scratch: the bounds agree to the
    # optimiser's tolerance.
    model = _analysis()
    t = np.array([6.0, 10.0, 14.0])
    warm = model.cb(t, method="bootstrap", n_boot=40, random_state=3)
    real = refits.DEGRADATION_REFIT

    class Cold:
        # The bootstrap's context, which the refits do not see.
        def set(self, value):
            return real.set(value)

        def get(self):
            return None

        def reset(self, token):
            real.reset(token)

    monkeypatch.setattr(refits, "DEGRADATION_REFIT", Cold())
    cold = model.cb(t, method="bootstrap", n_boot=40, random_state=3)
    np.testing.assert_allclose(warm, cold, rtol=1e-6)
