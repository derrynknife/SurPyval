"""
Along a step schedule, ``Hf_tvc`` (and ``sf_tvc``, ``cb_tvc`` and
``mean_tvc``, which use it) evaluates every segment in one call of the
model's ``Hf`` (or rate) rather than one segment at a time. These tests
hold it to the segment-at-a-time evaluation, which is still what a model
reading its covariates other than through ``beta' z`` gets: the same
numbers, bit for bit, for every family, with one and several covariates,
centred or not, and query times before, at and between the changes.
"""

import copy
import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval import StepSchedule
from surpyval.univariate.regression._tvc_evaluation import TVCEvaluationMixin
from surpyval.univariate.regression.accelerated_life import (
    AcceleratedLife,
    Power,
)

FAMILIES = [
    "WeibullPH",
    "ExponentialPH",
    "LogNormalPH",
    "WeibullAH",
    "WeibullAFT",
    "LogNormalAFT",
    "WeibullPO",
    "LogisticPH",
    "NormalAFT",
    "NormalAH",
]


def _fit(name, p, seed, center):
    rng = np.random.default_rng(seed)
    n = 150
    Z = rng.uniform(0, 2, (n, p))
    beta = np.linspace(-0.6, 0.4, p)
    x = rng.weibull(1.5, n) * 10 * np.exp(Z @ beta)
    if name.startswith(("Normal", "Logistic")):
        x = x - 2.0
    fitter = getattr(surv, name)
    kwargs = {"center": True} if center else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fitter.fit(x=x, Z=Z, **kwargs)


def _schedule(p, seed, n_seg, start=0.0):
    rng = np.random.default_rng(seed + 100)
    changes = np.concatenate(
        [[start], np.sort(rng.uniform(0.5, 25, n_seg - 1))]
    )
    schedule = StepSchedule.from_changepoints(
        changes, rng.uniform(0, 2, (n_seg, p))
    )
    return schedule, changes


def _one_at_a_time(monkeypatch):
    monkeypatch.setattr(
        TVCEvaluationMixin, "_tvc_linear", lambda self, *a: None
    )


def _both(monkeypatch, function):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = function()
        with monkeypatch.context() as patched:
            _one_at_a_time(patched)
            old = function()
    return np.asarray(new), np.asarray(old)


def _same(new, old):
    assert new.shape == old.shape
    assert np.array_equal(new, old, equal_nan=True), np.max(np.abs(new - old))


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("p", [1, 3])
@pytest.mark.parametrize("center", [False, True])
def test_hf_tvc_bit_identical(monkeypatch, name, p, center):
    model = _fit(name, p, 3, center)
    real_line = float(model.distribution.support[0]) < 0
    for n_seg in (1, 2, 17):
        schedule, changes = _schedule(
            p, n_seg, n_seg, start=-np.inf if real_line else 0.0
        )
        # Times at changes, between them, at 0 and missing.
        t = np.concatenate(
            [[0.0, np.nan], changes[1:3], np.linspace(0.1, 30, 41)]
        )
        if real_line:
            t = np.concatenate([t, [-3.0, -0.5]])
        _same(*_both(monkeypatch, lambda: model.Hf_tvc(t, schedule)))
        _same(*_both(monkeypatch, lambda: model.sf_tvc(t, schedule)))
        _same(
            *_both(monkeypatch, lambda: model.sf_tvc(t, schedule, given=5.0))
        )


@pytest.mark.parametrize("name", ["WeibullPH", "WeibullAFT", "WeibullPO"])
def test_cb_and_mean_tvc_bit_identical(monkeypatch, name):
    model = _fit(name, 2, 7, False)
    schedule, _ = _schedule(2, 7, 9)
    t = np.linspace(0.5, 30, 13)
    _same(*_both(monkeypatch, lambda: model.cb_tvc(t, schedule)))
    _same(*_both(monkeypatch, lambda: model.mean_tvc(schedule)))


def test_array_schedule_bit_identical(monkeypatch):
    model = _fit("WeibullPH", 2, 11, True)
    rows = [[0.0, 1.0], [1.5, 0.2], [0.3, 0.3]]
    t = [1.0, 4.0, 9.0, 20.0]
    _same(
        *_both(
            monkeypatch,
            lambda: model.Hf_tvc(t, rows, xl=[0.0, 3.0, 8.0]),
        )
    )


def test_accelerated_life_takes_the_segments_one_at_a_time():
    # Its life model reads the covariate row itself, so there is no
    # linear predictor to batch on.
    rng = np.random.default_rng(0)
    Z = rng.uniform(1, 3, (100, 1))
    x = rng.weibull(1.5, 100) * 40 / Z[:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AcceleratedLife(surv.Weibull, Power).fit(x=x, Z=Z)
    assert model._tvc_linear(model._eval_params(), None, [[1.0]]) is None


def test_custom_link_takes_the_segments_one_at_a_time():
    model = _fit("WeibullPH", 1, 0, False)
    model.model = copy.copy(model.model)
    model.model.phi = lambda Z, *b: 1.0 + np.dot(Z, np.array(b)) ** 2
    assert model._tvc_linear(model._eval_params(), None, [[1.0]]) is None
