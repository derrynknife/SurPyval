"""Every penalised fit keeps the same contract (the second wave: Fine-Gray,
Lin and Ying's additive hazards, the semi-parametric proportional odds
model, the parametric shared frailty and the parametric fits with
time-varying covariates).

Each case in :data:`CASES` is one penalised fit on a small seeded data
set, and each test checks one property of it:

- **no inference**: the covariance and standard errors refuse, saying the
  model was fitted with a penalty, and the summary's standard errors are
  ``nan`` (principle 17, the every-bound rule);
- **saving**: the model restored from its JSON dictionary has the same
  penalty and predictions, and still refuses;
- **pickling**: likewise through ``pickle``;
- **row order**: the fit does not depend on the order of the rows;
- **covariate units**: with ``standardise=True`` a covariate given in
  other units changes its coefficient by the factor and nothing else;
- **time units**: the coefficients are log ratios and do not change with
  the unit of time, except Lin and Ying's, rates, which are divided by
  the factor (the penalty is put on them in the unit of the mean
  follow-up);
- **large penalty**: a lasso penalty large enough sets every coefficient
  to 0, and the fit says it reached its minimum.

The references and optimality conditions of each fit are in
``univariate/regression/test_penalised_second_wave.py``.
"""

import json
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest

import surpyval as sp

REFUSED = "fitted with a penalty"


@dataclass
class PenalisedCase:
    name: str
    #: The rows: a dict of the fit's data arguments
    data: Callable[[], dict]
    #: fit(rows, **options) -> model
    fit: Callable[..., Any]
    #: The covariates' key in the rows, and the time keys
    Z: str = "Z"
    times: tuple = ("x",)
    #: Whether the coefficients are rates (1 / time)
    rates: bool = False
    #: The predictions a restored model must give
    predict: Callable[[Any, dict], Any] = lambda m, d: None
    #: model -> its coefficients
    coefficients: Callable[[Any], Any] = field(
        default=lambda m: np.asarray(m.beta, dtype=float)
    )


def _competing():
    rng = np.random.default_rng(31)
    Z = rng.normal(size=(150, 3))
    t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * Z[:, 0])))
    t_b = rng.exponential(1 / 0.05, 150)
    t_c = rng.uniform(0, 20, 150)
    x = np.minimum(np.minimum(t_a, t_b), t_c)
    e = np.where(
        t_c < np.minimum(t_a, t_b), None, np.where(t_a < t_b, "a", "b")
    )
    return {"x": x, "Z": Z, "e": e.astype(object)}


def _survival():
    rng = np.random.default_rng(32)
    Z = rng.normal(size=(150, 3))
    t = 10 * rng.weibull(1.5, 150) * np.exp(-0.5 * Z[:, 0])
    u = rng.uniform(0, 20, 150)
    return {"x": np.minimum(t, u), "Z": Z, "c": (u < t).astype(int)}


def _additive():
    rng = np.random.default_rng(33)
    Z = rng.normal(size=(150, 3))
    t = rng.exponential(1 / (0.2 + 0.05 * Z[:, 0]))
    u = rng.uniform(0, 15, 150)
    return {"x": np.minimum(t, u), "Z": Z, "c": (u < t).astype(int)}


def _frailty():
    rng = np.random.default_rng(34)
    groups = np.repeat(np.arange(25), 6)
    w = rng.gamma(2.0, 0.5, 25)[groups]
    Z = rng.normal(size=(150, 3))
    H = rng.exponential(1, 150) / (w * np.exp(0.5 * Z[:, 0]))
    return {"x": 10 * H**0.5, "Z": Z, "groups": groups}


def _start_stop():
    rng = np.random.default_rng(35)
    n = 120
    z = rng.normal(size=n)
    T = rng.weibull(2.0, n) * 5 * np.exp(-0.5 * z)
    other = rng.normal(size=(n, 2))
    i = np.repeat(np.arange(n), 2)
    xl = np.column_stack([0 * T, T / 2]).ravel()
    xr = np.column_stack([T / 2, T]).ravel()
    c = np.tile([1, 0], n)
    Z = np.column_stack([np.column_stack([z, z + 0.5]).ravel(), other[i]])
    return {"i": i, "xl": xl, "xr": xr, "c": c, "Z": Z}


def _tvc_case(fitter):
    return PenalisedCase(
        f"{fitter}.fit_tvc_penalised",
        _start_stop,
        lambda d, **kw: getattr(sp, fitter).fit_tvc_penalised(
            d["i"], d["xl"], d["xr"], d["c"], d["Z"], **kw
        ),
        times=("xl", "xr"),
        predict=lambda m, d: m.sf([1.0, 3.0], d["Z"][:2]),
        coefficients=lambda m: np.asarray(m.params, dtype=float)[2:],
    )


CASES = [
    PenalisedCase(
        "FineGray.fit_penalised",
        _competing,
        lambda d, **kw: sp.FineGray.fit_penalised(
            d["x"], d["Z"], d["e"], event="a", **kw
        ),
        predict=lambda m, d: m.cif([2.0, 5.0], d["Z"][:2]),
    ),
    PenalisedCase(
        "AdditiveHazards.fit_penalised",
        _additive,
        lambda d, **kw: sp.AdditiveHazards.fit_penalised(
            d["x"], d["Z"], c=d["c"], **kw
        ),
        rates=True,
        predict=lambda m, d: m.sf([2.0, 5.0], d["Z"][:2]),
    ),
    PenalisedCase(
        "ProportionalOdds.fit_penalised",
        _survival,
        lambda d, **kw: sp.ProportionalOdds.fit_penalised(
            d["x"], d["Z"], c=d["c"], **kw
        ),
        predict=lambda m, d: m.sf([2.0, 5.0], d["Z"][:2]),
    ),
    PenalisedCase(
        "WeibullFrailty.fit_penalised",
        _frailty,
        lambda d, **kw: sp.WeibullFrailty.fit_penalised(
            d["x"], d["Z"], groups=d["groups"], **kw
        ),
        predict=lambda m, d: m.sf([5.0, 10.0], d["Z"][0]),
    ),
    _tvc_case("WeibullPH"),
    _tvc_case("WeibullPO"),
    _tvc_case("WeibullAFT"),
]
IDS = [case.name for case in CASES]
OPTIONS = dict(penalizer=0.03, l1_ratio=0.5)


def _fit(case, data=None, **kw):
    return case.fit(case.data() if data is None else data, **{**OPTIONS, **kw})


@pytest.fixture(scope="module")
def fitted():
    return {case.name: _fit(case) for case in CASES}


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_no_inference(case, fitted):
    model = fitted[case.name]
    assert model.maximum == "verified"
    for call in (model.covariance, model.standard_errors):
        with pytest.raises(ValueError, match=REFUSED):
            call()
    assert np.all(np.isnan(np.asarray(model.summary()["se(coef)"], float)))
    assert "Penalty" in repr(model)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_saving(case, fitted):
    model = fitted[case.name]
    data = case.data()
    restored = sp.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.penalty.to_dict() == model.penalty.to_dict()
    np.testing.assert_allclose(
        case.predict(restored, data), case.predict(model, data), rtol=1e-12
    )
    with pytest.raises(ValueError, match=REFUSED):
        restored.covariance()


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_pickling(case, fitted):
    model = fitted[case.name]
    data = case.data()
    copy = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        case.predict(copy, data), case.predict(model, data)
    )
    with pytest.raises(ValueError, match=REFUSED):
        copy.covariance()


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_row_order(case, fitted):
    data = case.data()
    rows = len(data[case.Z])
    order = np.random.default_rng(7).permutation(rows)
    shuffled = {k: np.asarray(v)[order] for k, v in data.items()}
    np.testing.assert_allclose(
        case.coefficients(_fit(case, shuffled)),
        case.coefficients(fitted[case.name]),
        rtol=1e-7,
        atol=1e-10,
    )


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_covariate_units(case, fitted):
    data = case.data()
    data[case.Z] = data[case.Z] * np.array([10.0, 1.0, 1.0])
    expected = case.coefficients(fitted[case.name]) / np.array([10.0, 1, 1])
    np.testing.assert_allclose(
        case.coefficients(_fit(case, data)), expected, rtol=1e-6, atol=1e-10
    )


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_time_units(case, fitted):
    data = case.data()
    for key in case.times:
        data[key] = 24.0 * np.asarray(data[key], dtype=float)
    expected = case.coefficients(fitted[case.name])
    if case.rates:
        expected = expected / 24.0
    np.testing.assert_allclose(
        case.coefficients(_fit(case, data)), expected, rtol=1e-6, atol=1e-12
    )


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_large_lasso_penalty_sets_every_coefficient_to_zero(case):
    model = _fit(case, penalizer=1e3, l1_ratio=1.0)
    assert model.maximum == "verified"
    np.testing.assert_array_equal(case.coefficients(model), 0.0)
