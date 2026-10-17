"""Every test planning result writes itself as strict JSON: native
values, tuples and arrays as lists, non-finite numbers as None with the
package's "non_finite" record."""

import dataclasses
import json

import numpy as np
import pytest

import surpyval
from surpyval import life_models
from surpyval.serialisation import decode_non_finite
from surpyval.test_planning import (
    accelerated_test_plan,
    accelerated_test_precision,
    demonstration_plan,
    life_test_precision,
    mtbf_demonstration_plan,
    sequential_demonstration_test,
    sequential_mtbf_test,
)
from surpyval.test_planning._serialise import result_dict


def _strict(d):
    text = json.dumps(d, allow_nan=False)
    assert json.loads(text) == d
    return d


def _same_fields(result, d):
    for f in dataclasses.fields(result):
        value = getattr(result, f.name)
        got = d[f.name]
        if isinstance(value, (tuple, list, np.ndarray)):
            np.testing.assert_allclose(got, np.asarray(value, dtype=float))
        elif isinstance(value, dict):
            assert got == {k: float(v) for k, v in value.items()}
        else:
            assert got == value, f.name


def test_demonstration_plans():
    plan = demonstration_plan(0.9, 0.95, alpha_ci=0.1)
    d = _strict(plan.to_dict())
    assert d["result"] == "DemonstrationPlan"
    _same_fields(plan, d)
    assert type(d["n"]) is int and d["test_time"] is None
    mtbf = mtbf_demonstration_plan(100.0, 300.0)
    d = _strict(mtbf.to_dict())
    _same_fields(mtbf, d)
    assert d["n"] is None and isinstance(d["test_time"], float)


def test_life_test_plan_and_simulated_precision():
    plan = life_test_precision(
        surpyval.Weibull, (1000.0, 2.0), 20, censor_time=800.0
    )
    d = _strict(plan.to_dict())
    assert d["result"] == "LifeTestPlan"
    _same_fields(plan, d)
    assert d["params"] == [1000.0, 2.0]
    sim = plan.simulate(30, random_state=1)
    d = _strict(sim.to_dict())
    assert d["result"] == "SimulatedPrecision"
    _same_fields(sim, d)
    assert type(d["n_sim"]) is int


def test_accelerated_test_plans():
    plan = accelerated_test_precision(
        surpyval.Weibull,
        life_models.Power,
        {"a": 1e6, "n": -2.0},
        2.0,
        use_stress=10.0,
        stresses=(40.0, 60.0),
        units=(30, 20),
        censor_time=1000.0,
    )
    d = _strict(plan.to_dict())
    assert d["result"] == "AcceleratedTestPlan"
    _same_fields(plan, d)
    assert d["units"] == [30, 20] and all(type(u) is int for u in d["units"])
    optimum = accelerated_test_plan(
        surpyval.Weibull,
        life_models.Power,
        {"a": 1e6, "n": -2.0},
        2.0,
        use_stress=10.0,
        high_stress=60.0,
        n=50,
        censor_time=1000.0,
    )
    _same_fields(optimum, _strict(optimum.to_dict()))


def test_sequential_mtbf_test():
    test = sequential_mtbf_test(100.0, 200.0, alpha_ci=0.1)
    d = _strict(test.to_dict())
    assert d["result"] == "SequentialMTBFTest"
    assert d["mtbf"] == 100.0 and d["good_mtbf"] == 200.0
    assert d["boundary_risks"] == [0.1, 0.2]
    assert d["consumer_risk"] == test.consumer_risk
    assert d["producer_risk"] == test.producer_risk
    table = d["boundaries"]
    # truncated: the table runs to the failure that rejects outright
    assert d["max_failures"] is not None
    assert table["failures"] == list(range(d["max_failures"] + 1))
    np.testing.assert_allclose(
        table["accept_time"], [test.accept_time(r) for r in table["failures"]]
    )
    # that failure rejects whenever it comes: inf, written as None
    assert table["reject_time"][-1] is None
    assert f"/boundaries/reject_time/{d['max_failures']}" in (
        d["non_finite"]["inf"]
    )
    restored = decode_non_finite(d)
    assert restored["boundaries"]["reject_time"][-1] == np.inf
    np.testing.assert_allclose(
        restored["boundaries"]["reject_time"],
        [test.reject_time(r) for r in table["failures"]],
    )
    untruncated = sequential_mtbf_test(100.0, 200.0, truncate=False)
    d = _strict(untruncated.to_dict())
    assert d["max_time"] is None and d["boundaries"]["failures"][-1] == 10
    assert len(_strict(untruncated.to_dict(3))["boundaries"]["failures"]) == 4


def test_sequential_demonstration_test():
    test = sequential_demonstration_test(
        0.9, 0.97, alpha_ci=0.1, producer_risk=0.1
    )
    d = _strict(test.to_dict())
    assert d["result"] == "SequentialDemonstrationTest"
    assert d["reliability"] == 0.9 and d["good_reliability"] == 0.97
    table = d["boundaries"]
    assert table["units"] == list(range(d["max_units"] + 1))
    assert table["accept_failures"] == [
        test.accept_failures(n) for n in table["units"]
    ]
    assert table["reject_failures"] == [
        test.reject_failures(n) for n in table["units"]
    ]
    untruncated = sequential_demonstration_test(0.9, 0.97, truncate=False)
    d = _strict(untruncated.to_dict())
    first = next(n for n in range(1000) if untruncated.accept_failures(n) >= 0)
    assert d["boundaries"]["units"][-1] == max(2 * first, 10)
    assert _strict(untruncated.to_dict(units=5))["boundaries"]["units"] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
    with pytest.raises(ValueError, match="units"):
        test.to_dict(units=-1)


def test_non_finite_values_are_none_and_recorded():
    class Result:
        pass

    d = _strict(
        result_dict(
            Result(),
            {"a": np.float64(np.nan), "b": np.array([1.0, np.inf]), "c": (1,)},
        )
    )
    assert d == {
        "result": "Result",
        "a": None,
        "b": [1.0, None],
        "c": [1],
        "non_finite": {"nan": ["/a"], "inf": ["/b/1"]},
    }
