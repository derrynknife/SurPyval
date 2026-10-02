"""Sorting and handling ``xcnt`` data once (performance sweep).

``xcnt_sort`` sorted the data three times, stably, by ``c``, then the
lower truncation bound, then ``x``, each with a gather of all four
arrays; one stable sort on the three keys gives the same order. And a
Kaplan-Meier, Nelson-Aalen or Fleming-Harrington fit handled its data
(validated, grouped and sorted it) and then handled it again in
``xcnt_to_xrd``: a third of a fit at 1e6 rows (0.72 s, now 0.49 s).
"""

import importlib

import numpy as np
import pytest

import surpyval as sp
from surpyval.utils import xcnt_sort

utils_module = importlib.import_module("surpyval.utils")
np_fitter = importlib.import_module(
    "surpyval.univariate.nonparametric.nonparametric_fitter"
)


def _three_sorts(x, c, n, t):
    """``xcnt_sort`` before, verbatim, as the reference."""
    idx_c = np.argsort(c, kind="stable")
    x, c, n, t = x[idx_c], c[idx_c], n[idx_c], t[idx_c]
    key = t if t.ndim == 1 else t.min(axis=1)
    idx = np.argsort(key, kind="stable")
    x, c, n, t = x[idx], c[idx], n[idx], t[idx]
    key = x if x.ndim == 1 else x.mean(axis=1)
    idx = np.argsort(key, kind="stable")
    return x[idx], c[idx], n[idx], t[idx]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("two_d", [False, True])
def test_one_sort_orders_as_three(seed, two_d):
    rng = np.random.default_rng(seed)
    size = 400
    x = rng.choice([1.0, 2.0, 3.0, np.inf], size)
    if two_d:
        x = np.column_stack([x, x + rng.choice([0.0, 1.0], size)])
    c = rng.choice([-1, 0, 1, 2], size)
    n = np.arange(size)  # a row's identity
    t = np.column_stack(
        [
            rng.choice([-np.inf, 0.0, 0.5], size),
            rng.choice([2.0, np.inf], size),
        ]
    )
    for got, want in zip(xcnt_sort(x, c, n, t), _three_sorts(x, c, n, t)):
        np.testing.assert_array_equal(got, want)


def test_one_sort(monkeypatch):
    calls = []
    argsort = np.argsort

    def counting(*args, **kwargs):
        calls.append(1)
        return argsort(*args, **kwargs)

    monkeypatch.setattr(np, "argsort", counting)
    x = np.array([3.0, 1.0, 2.0, 1.0])
    t = np.zeros((4, 2))
    xcnt_sort(x, np.zeros(4, int), np.ones(4, int), t)
    assert calls == []


@pytest.mark.parametrize(
    "fitter", [sp.KaplanMeier, sp.NelsonAalen, sp.FlemingHarrington]
)
def test_a_fit_handles_its_data_once(monkeypatch, fitter):
    calls = []
    handler = utils_module.xcnt_handler

    def counting(*args, **kwargs):
        calls.append(1)
        return handler(*args, **kwargs)

    monkeypatch.setattr(utils_module, "xcnt_handler", counting)
    monkeypatch.setattr(np_fitter, "xcnt_handler", counting)
    rng = np.random.default_rng(1)
    x = np.round(10 * rng.weibull(1.5, 300), 1) + 0.1
    c = rng.choice([0, 1], 300)
    tl = np.where(rng.uniform(size=300) < 0.3, 0.5 * x, 0.0)
    model = fitter.fit(x=x, c=c, n=rng.integers(1, 3, 300), tl=tl)
    assert len(calls) == 1
    # The same ladder as from the public conversion
    xrd = sp.utils.xcnt_to_xrd(
        model.data["x"], model.data["c"], model.data["n"], model.data["t"]
    )
    for got, want in zip((model.x, model.r, model.d), xrd):
        np.testing.assert_array_equal(got, want)
