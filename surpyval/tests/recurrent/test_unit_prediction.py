"""Prediction from each unit's current state for the renewal models
(#615): ``RenewalModel.unit_states``, ``next_failure_sf`` /
``next_failure_hf`` and ``surpyval.forecast`` of a renewal model.

The checks are against the models' definitions written out here one unit
and one failure at a time: the virtual age (or intensity reduction) a
unit's own failures leave it in, and a plain simulation of its future
from there.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

import surpyval as sp
from surpyval.recurrent import (
    ARA,
    ARI,
    CrowAMSAA,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
)

# (fitter, fit options, the true model's restoration parameter)
FAMILIES = {
    "GRP Kijima I": (GeneralizedRenewal, {"kijima": "i"}, 0.4),
    "GRP Kijima II": (GeneralizedRenewal, {"kijima": "ii"}, 0.4),
    "G1": (GeneralizedOneRenewal, {}, 0.1),
    "ARA m=2": (ARA, {"m": 2}, 0.6),
    "ARA m=inf": (ARA, {"m": np.inf}, 0.6),
    "ARI m=2": (ARI, {"m": 2}, 0.4),
}


def _truth(fitter, options, restoration):
    if fitter is ARI:
        return ARI.fit_from_parameters(
            [20.0, 1.8], restoration, m=options["m"], baseline=CrowAMSAA
        )
    if fitter is GeneralizedRenewal:
        return fitter.fit_from_parameters(
            [20.0, 1.8], restoration, kijima=options["kijima"]
        )
    if fitter is ARA:
        return fitter.fit_from_parameters(
            [20.0, 1.8], restoration, m=options["m"]
        )
    return fitter.fit_from_parameters([20.0, 1.8], restoration)


def _fitted(key):
    fitter, options, restoration = FAMILIES[key]
    truth = _truth(fitter, options, restoration)
    sim = truth.time_terminated_simulation_data(
        80.0, items=12, random_state=615
    )
    return fitter.fit(sim.x, sim.i, sim.c, **options)


def _histories(model):
    data = model.data
    out = []
    for item in data.items:
        mask = data.i == item
        x, c = data.x[mask], data.c[mask]
        out.append((np.sort(x[c == 0]), float(x.max())))
    return out


def _weibull(model):
    alpha, beta = (float(p) for p in model.model.params)
    return alpha, beta


def _state_by_definition(model, failures, now):
    """``(age the lifetime sees now, scale of the next gap)``, or for ARI
    ``(now, intensity reduction)``, from the model's recursion."""
    kind = model.kind
    r = float(model.restoration)
    since = now - (failures[-1] if failures.size else 0.0)
    if kind == "Generalized Renewal":
        v = 0.0
        previous = 0.0
        for t in failures:
            gap = t - previous
            previous = t
            v = v + r * gap if model.kijima_type == "i" else r * (v + gap)
        return v + since, 1.0
    if kind == "G1 Renewal":
        scale = (1.0 + r) ** failures.size
        return since / scale, scale
    if kind == "ARA Renewal":
        m = model.m
        recent = failures[::-1]
        terms = recent if np.isinf(m) else recent[: int(m)]
        weights = (1.0 - r) ** np.arange(terms.size)
        return now - r * np.sum(weights * terms), 1.0
    alpha, beta = (float(p) for p in model.model.params)
    lam = beta / alpha**beta * failures ** (beta - 1)
    m = model.m
    recent = lam[::-1]
    terms = recent if np.isinf(m) else recent[: int(m)]
    weights = (1.0 - r) ** np.arange(terms.size)
    return now, r * np.sum(weights * terms)


@pytest.mark.parametrize("key", list(FAMILIES))
def test_615_unit_states_follow_each_units_history(key):
    model = _fitted(key)
    table = model.unit_states()
    assert list(table.index) == list(model.data.items)
    for (failures, now), (_, row) in zip(_histories(model), table.iterrows()):
        assert row["time"] == now
        assert row["failures"] == failures.size
        first, second = _state_by_definition(model, failures, now)
        if model.kind == "ARI Recurrence":
            assert row["reduction"] == pytest.approx(second, rel=1e-10)
        else:
            assert row["virtual_age"] == pytest.approx(first, rel=1e-10)


@pytest.mark.parametrize("key", list(FAMILIES))
def test_615_next_failure_sf_and_hf_from_the_state(key):
    model = _fitted(key)
    x = np.array([0.0, 2.0, 7.5])
    sf = model.next_failure_sf(x)
    hf = model.next_failure_hf(x)
    assert sf.shape == hf.shape == (len(model.data.items), 3)
    for k, (failures, now) in enumerate(_histories(model)):
        first, second = _state_by_definition(model, failures, now)
        if model.kind == "ARI Recurrence":
            alpha, beta = (float(p) for p in model.model.params)
            cif = ((now + x) / alpha) ** beta - (now / alpha) ** beta
            expected_sf = np.exp(-(cif - second * x))
            expected_hf = beta / alpha**beta * (now + x) ** (beta - 1) - second
        else:
            alpha, beta = _weibull(model)
            age, scale = first, second
            expected_sf = np.exp(
                -(((age + x / scale) / alpha) ** beta - (age / alpha) ** beta)
            )
            expected_hf = (
                beta / alpha * ((age + x / scale) / alpha) ** (beta - 1)
            ) / scale
        np.testing.assert_allclose(sf[k], expected_sf, rtol=1e-10)
        np.testing.assert_allclose(hf[k], expected_hf, rtol=1e-10)
    # A scalar time gives one value per unit.
    assert model.next_failure_sf(2.0).shape == (len(model.data.items),)


def _future_count(model, failures, now, horizon, rng):
    """One simulated future of a unit, by the model's definition: its
    number of failures in ``(now, now + horizon]``."""
    kind = model.kind
    alpha, beta = (float(p) for p in model.model.params)
    failures = list(failures)
    t = now
    count = 0
    while True:
        u = rng.uniform()
        if kind == "ARI Recurrence":
            _, reduction = _state_by_definition(model, np.array(failures), t)

            def gain(x, t=t, reduction=reduction, e=-np.log(u)):
                cif = ((t + x) / alpha) ** beta - (t / alpha) ** beta
                return cif - reduction * x - e

            hi = 1.0
            while gain(hi) < 0:
                hi *= 2
            t = t + brentq(gain, 0.0, hi, xtol=1e-12)
        else:
            age, scale = _state_by_definition(model, np.array(failures), t)
            # Weibull residual life from `age` on the gap's time scale.
            base = alpha * ((age / alpha) ** beta - np.log(u)) ** (1 / beta)
            t = t + scale * (base - age)
        if t > now + horizon:
            return count
        failures.append(t)
        count += 1


@pytest.mark.parametrize("key", list(FAMILIES))
def test_615_forecast_simulates_each_unit_from_its_state(key):
    model = _fitted(key)
    result = sp.forecast(model, horizon=[5.0, 15.0], random_state=1)
    assert result.simulations == 1000
    assert list(result.units) == list(model.data.items)
    rng = np.random.default_rng(2)
    reps = 1500
    for k, (failures, now) in enumerate(_histories(model)[:4]):
        by_definition = np.array(
            [
                _future_count(model, failures, now, 15.0, rng)
                for _ in range(reps)
            ]
        )
        se = np.sqrt(by_definition.var() / reps + by_definition.var() / 1000)
        assert abs(result.per_unit[k, 1] - by_definition.mean()) < 4.5 * se
    # The chance of a failure within the horizon is the exact one.
    np.testing.assert_allclose(
        result.probability[:, 0], 1 - model.next_failure_sf(5.0)
    )
    np.testing.assert_allclose(result.unit_expected, result.per_unit)
    np.testing.assert_allclose(
        result.expected, result.per_unit.sum(axis=0), rtol=1e-12
    )
    assert np.all(result.lower <= result.expected)
    assert np.all(result.expected <= result.upper)


def test_615_new_units_and_limits():
    # Units given by their ages: no failure yet. A unit at or past its
    # limit contributes nothing; one with 2 left counts 2 hours only.
    model = GeneralizedRenewal.fit_from_parameters([20.0, 1.8], 0.4)
    table = model.unit_states(age=[0.0, 10.0])
    assert table["virtual_age"].tolist() == [0.0, 10.0]
    result = sp.forecast(
        model,
        age=[0.0, 10.0, 30.0],
        horizon=[5.0],
        limit=[100.0, 12.0, 30.0],
        random_state=3,
    )
    np.testing.assert_allclose(
        result.probability[:, 0],
        [
            1 - model.next_failure_sf(5.0, age=[0.0])[0],
            1 - model.next_failure_sf(2.0, age=[10.0])[0],
            0.0,
        ],
    )
    assert result.per_unit[2, 0] == 0
    # A new unit's simulated future is the model's own simulation from new.
    again = sp.forecast(
        model, age=[0.0], horizon=[40.0], items=4000, random_state=4
    )
    assert again.per_unit[0, 0] == pytest.approx(
        model.mcf(40.0, items=4000, random_state=5), rel=0.05
    )


def test_615_renewal_forecast_input_errors():
    model = GeneralizedRenewal.fit_from_parameters([20.0, 1.8], 0.4)
    with pytest.raises(ValueError, match="requires a model fitted"):
        sp.forecast(model, horizon=1.0)
    with pytest.raises(ValueError, match="n is not taken"):
        sp.forecast(model, age=[1.0], horizon=1.0, n=[3])
    with pytest.raises(ValueError, match="no covariates"):
        sp.forecast(model, age=[1.0], horizon=1.0, Z=[[1.0]])
    with pytest.raises(ValueError, match="cannot be negative"):
        model.next_failure_sf(-1.0, age=[1.0])
    with pytest.raises(ValueError, match="at least 0"):
        model.unit_states(age=[-1.0])


@pytest.mark.parametrize("m", [1, 3, np.inf])
def test_615_memory_from_units_histories(m):
    # The ARA/ARI memory of units with different numbers of failures
    # (fewer than m too) is each unit's own discounted sum, newest first.
    from surpyval.recurrent.renewal.renewal_model import DiscountedMemory

    histories = [np.array([1.0, 2.0]), np.arange(1.0, 6.0), np.zeros(0)]
    memory = DiscountedMemory.from_history(histories, 0.4, m)
    expected = []
    for values in histories:
        recent = values[::-1] if np.isinf(m) else values[::-1][: int(m)]
        expected.append(np.sum(0.6 ** np.arange(recent.size) * recent))
    np.testing.assert_allclose(memory.value(np.arange(3)), expected)
    # It carries on as the memory of sequences recorded round by round.
    memory.record(np.array([0, 1]), np.array([7.0, 8.0]))
    for k, new in ((0, 7.0), (1, 8.0)):
        values = np.append(histories[k], new)[::-1]
        recent = values if np.isinf(m) else values[: int(m)]
        assert memory.value(np.array([k]))[0] == pytest.approx(
            np.sum(0.6 ** np.arange(recent.size) * recent)
        )
