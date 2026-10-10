"""A parametric additive hazards model queried outside its support
(#376, #828).

``h(x | Z) = h_0(x) + beta'Z`` is a distribution only where it and its
integral are non-negative: the fit keeps every observed point there, but
for a protective covariate row early on the cumulative hazard falls below
0, where ``sf`` would exceed 1 and ``ff`` and ``df`` go negative. Those
predictions are nan, and every prediction method says so with one warning
per call.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.regression import StepSchedule

# A row outside the support at the first two times (H < 0 there).
NEGATIVE = [-3.0, 0.0]
POSITIVE = [0.5, 1.0]
TIMES = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 100.0])


def _data(size=200, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=size)
    g = rng.integers(0, 2, size).astype(float)
    t = rng.weibull(1.5, size) * 10 * np.exp(-(0.8 * z + 0.6 * g) / 1.5)
    cens = rng.uniform(0, 25, size)
    x = np.round(np.minimum(t, cens), 3) + 1e-3
    c = (cens < t).astype(int)
    return x, np.column_stack([z, g]), c


@pytest.fixture(scope="module")
def model():
    x, Z, c = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.WeibullAH.fit(x, Z, c=c)


def _record(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, rec


@pytest.mark.parametrize("name", ["sf", "ff", "df", "hf", "Hf"])
def test_one_warning_per_call_outside_the_support(model, name):
    out, rec = _record(getattr(model, name), TIMES, NEGATIVE)
    assert len(rec) == 1
    w = rec[0]
    assert issubclass(w.category, RuntimeWarning)
    assert w.filename == __file__  # points at the caller
    msg = str(w.message)
    assert "negative at 2 of the 6 queried points" in msg
    assert "the values there are nan" in msg
    assert "WeibullPH" in msg
    out = np.asarray(out)
    assert np.isnan(out[:2]).all() and np.isfinite(out[2:]).all()


def test_inside_the_support_the_values_are_the_models(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sf = model.sf(TIMES, NEGATIVE)
        Hf = model.Hf(TIMES, NEGATIVE)
        ff = model.ff(TIMES, NEGATIVE)
    raw = np.ravel(model.model.Hf(TIMES, np.array([NEGATIVE]), *model.params))
    # (the model's raw H is negative where the values are nan)
    assert (raw[:2] < 0).all()
    np.testing.assert_allclose(Hf[2:], raw[2:])
    np.testing.assert_allclose(sf, np.exp(-Hf))
    np.testing.assert_allclose(ff, 1 - sf)
    assert np.all(sf[2:] <= 1) and np.all(ff[2:] >= 0)


@pytest.mark.parametrize("name", ["sf", "ff", "df", "hf", "Hf"])
def test_no_warning_where_the_hazard_is_positive(model, name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        getattr(model, name)(TIMES, POSITIVE)


def test_below_the_support_does_not_count(model):
    # Negative times are below a Weibull's support: sf is 1 there by
    # definition, not because the hazard fell.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.sf([-5.0, -1.0], NEGATIVE)


@pytest.mark.parametrize("on", ["sf", "ff", "Hf", "hf", "df"])
def test_cb_warns_once_and_is_nan_outside(model, on):
    out, rec = _record(model.cb, TIMES, NEGATIVE, on=on)
    assert len(rec) == 1 and rec[0].filename == __file__
    out = np.asarray(out)
    assert np.isnan(out[:2]).all() and np.isfinite(out[2:]).all()


def test_time_varying_paths_warn_once(model):
    # The covariate turns strongly protective at t = 5, so the hazard is
    # negative after it and the cumulative hazard falls: outside the
    # support from there on.
    path = StepSchedule.from_changepoints([0, 5], [[0.0, 0.0], [-5.0, 0.0]])
    for fn, kwargs in (
        (model.Hf_tvc, {}),
        (model.sf_tvc, {}),
        (model.sf_tvc, {"given": 1.0}),
    ):
        out, rec = _record(fn, TIMES, path, **kwargs)
        assert len(rec) == 1, fn.__name__
        assert rec[0].filename == __file__
        assert "negative at 3 of the 6" in str(rec[0].message)
        out = np.asarray(out)
        assert np.isnan(out[3:]).all() and np.isfinite(out[:3]).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.sf_tvc(
            TIMES, StepSchedule.from_changepoints([0, 5], [POSITIVE, POSITIVE])
        )


def test_proportional_hazards_never_warns():
    x, Z, c = _data()
    ph = sp.WeibullPH.fit(x, Z, c=c)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in ("sf", "ff", "df", "hf", "Hf"):
            getattr(ph, name)(TIMES, [-5.0, 0.0])
