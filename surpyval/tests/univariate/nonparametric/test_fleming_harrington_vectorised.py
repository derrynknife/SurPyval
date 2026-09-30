"""The Fleming-Harrington tie ladder, vectorised (#515).

``_fleming_harrington`` and ``fleming_harrington_variance`` called the
scalar ``fh_h`` / ``fh_var_h`` once per step in a list comprehension. It is
the default estimator of the Turnbull EM, which evaluates it on every
iteration: that loop was 98% of a Turnbull fit (12.8 s on 1000 random
intervals). ``_fh_ladder`` does the same arithmetic on whole arrays, and
must be bit-identical to the scalar functions, which stay as the reference
-- including on the round-off cases the snap exists for, the divergent
ladders, the closed form for long ladders, and NaN.
"""

import importlib

import numpy as np
import pytest

from surpyval import FlemingHarrington, Turnbull
from surpyval.univariate.nonparametric.fleming_harrington import (
    _MAX_TIE_LOOP,
    _fh_ladder,
    fh_h,
    fh_var_h,
)

fh_module = importlib.import_module(
    "surpyval.univariate.nonparametric.fleming_harrington"
)


def _scalar_ladder(r, d, variance):
    """The list comprehension ``_fh_ladder`` replaced."""
    f = fh_var_h if variance else fh_h
    return np.array([f(r_i, d_i) for r_i, d_i in zip(r, d)], dtype=float)


def _assert_bit_identical(new, old):
    np.testing.assert_array_equal(new, old)
    # assert_array_equal counts 0.0 and -0.0 as equal; the signs must match.
    np.testing.assert_array_equal(np.signbit(new), np.signbit(old))


def _cases():
    rng = np.random.default_rng(515)
    r = np.arange(300, 0, -1).astype(float)
    yield "integer counts", r, rng.binomial(4, 0.3, 300).astype(float)
    # Turnbull-style fractional expected counts, some a whole number up to
    # round-off, some fractional ties climbing the ladder.
    rf = rng.uniform(0.5, 80, 400)
    df = rf * rng.uniform(0, 1, 400)
    df[::9] = 0.0
    df[3] = rf[3] * (1 + 2e-16)
    df[5] = 1e-16
    df[7] = 3 + 4e-16
    df[11] = 2.0 - 3e-16
    rf[13] = 5 + 1e-15
    yield "fractional with round-off", rf, df
    # Ladder lengths either side of the switch to the closed form.
    rl = rng.uniform(_MAX_TIE_LOOP + 5, 500, 60)
    dl = np.concatenate(
        [
            np.full(20, _MAX_TIE_LOOP + 1.0),
            np.full(10, _MAX_TIE_LOOP + 1.5),
            np.full(10, _MAX_TIE_LOOP + 2.0),
            rng.uniform(1, _MAX_TIE_LOOP + 20, 20),
        ]
    )
    yield "short and long ladders", rl, dl
    yield "huge but valid", np.array([1e16, 1e9, 1e6]), np.array(
        [1e12, 1e8, 999_999.5]
    )
    # Divergent, degenerate and missing values.
    yield "edges", np.array(
        [10.0, 10.0, 0.3, 0.0, -1.0, np.nan, 5.0, 3.0, 3.0, 2.5, 0.0, 4.0]
    ), np.array(
        [1e15, np.inf, 2.0, 1.0, 2.0, 1.0, np.nan, 3.0, 4.0, 3.0, 0.0, -np.inf]
    )
    yield "empty", np.array([]), np.array([])


@pytest.mark.parametrize("variance", [False, True])
@pytest.mark.parametrize("label,r,d", list(_cases()))
def test_ladder_bit_identical_to_the_scalar_functions(label, r, d, variance):
    with np.errstate(all="ignore"):
        old = _scalar_ladder(r, d, variance)
    _assert_bit_identical(_fh_ladder(r, d, variance), old)


def _intervals(rng, size):
    a = rng.uniform(0, 10, size)
    return a, a + rng.uniform(0.1, 3, size)


def _turnbull_cases():
    rng = np.random.default_rng(12)
    a, b = _intervals(rng, 60)
    yield "intervals", dict(xl=a, xr=b)
    a, b = _intervals(rng, 40)
    yield "rounded intervals with counts", dict(
        xl=np.round(a),
        xr=np.round(a) + np.ceil(b - a),
        n=rng.integers(1, 4, 40),
    )
    x = np.round(rng.weibull(1.5, 50) * 10)
    c = rng.choice([0, 1, -1], 50)
    yield "left, right and exact with ties", dict(x=x, c=c)
    x = rng.weibull(1.5, 50) * 10
    yield "left truncated", dict(x=x + 1, tl=rng.uniform(0, 1, 50))
    yield "right truncated", dict(x=x, tr=x.max() + rng.uniform(0, 5, 50))


def _model_arrays(model):
    return [model.R, model.H, model.r, model.d, model.x]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "label,kwargs",
    list(_turnbull_cases()),
    ids=[c[0] for c in _turnbull_cases()],
)
def test_turnbull_fit_bit_identical(monkeypatch, label, kwargs):
    new = Turnbull.fit(**kwargs, max_iter=200)
    with monkeypatch.context() as patch:
        patch.setattr(fh_module, "_fh_ladder", _scalar_ladder)
        old = Turnbull.fit(**kwargs, max_iter=200)
    for a, b in zip(_model_arrays(new), _model_arrays(old)):
        _assert_bit_identical(a, b)
    with np.errstate(all="ignore"):
        _assert_bit_identical(new.sf(new.x), old.sf(old.x))
        for new_cb, old_cb in zip(
            new.cb(new.x, alpha_ci=0.1), old.cb(old.x, alpha_ci=0.1)
        ):
            _assert_bit_identical(new_cb, old_cb)


def test_fleming_harrington_fit_bit_identical(monkeypatch):
    rng = np.random.default_rng(3)
    x = np.round(rng.weibull(1.5, 500) * 20)
    c = (rng.uniform(size=500) < 0.3).astype(int)
    n = rng.integers(1, 6, 500)
    new = FlemingHarrington.fit(x, c, n)
    with monkeypatch.context() as patch:
        patch.setattr(fh_module, "_fh_ladder", _scalar_ladder)
        old = FlemingHarrington.fit(x, c, n)
    for a, b in zip(_model_arrays(new), _model_arrays(old)):
        _assert_bit_identical(a, b)
    for new_cb, old_cb in zip(new.cb(new.x), old.cb(old.x)):
        _assert_bit_identical(new_cb, old_cb)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_turnbull_em_does_not_loop_over_steps_in_python(monkeypatch):
    # The regression: the EM must not call the scalar functions per step.
    def refuse(*args, **kwargs):
        raise AssertionError("per-step scalar call")

    monkeypatch.setattr(fh_module, "fh_h", refuse)
    monkeypatch.setattr(fh_module, "fh_var_h", refuse)
    a, b = _intervals(np.random.default_rng(0), 50)
    model = Turnbull.fit(xl=a, xr=b, max_iter=50)
    model.cb(model.x, alpha_ci=0.1)
    FlemingHarrington.fit([1, 2, 2, 3, 5, 5, 5]).cb([2, 5])
