"""Metamorphic invariances of the fits (#379).

Each property rewrites the fixture in a way that should not change the
answer, refits, and compares every function of the two models:

- **units**: times multiplied by ``K`` give the same model in the new
  unit -- probabilities at ``K x`` equal those at ``x``, rates are
  divided by ``K``, quantiles multiplied by it ("fits depended on the
  data's units");
- **row order**: a permutation of the data rows (the input-order bugs);
- **counts**: a count ``n`` gives what that many repeated rows give.
"""

import numpy as np
import pytest

from surpyval.tests.conformance.registry import (
    cases_for,
    fitted,
    predictions,
    refit,
    scramble,
)

K = 7.3  # an awkward unit change, so it cannot hide in rounding

# How each function scales with the time unit: value(K x) * K**power
# in the new unit equals value(x) in the old one.
_POWER = {"hf": 1, "df": 1, "iif": 1, "pdf": 2, "qf": -1}


def _power(case, key):
    fname = key.split("[")[0]
    if fname in case.jump_functions:
        return 0  # a jump is a probability, whatever the unit
    return _POWER.get(fname, 0)


def _compare(case, got, ref, scale=None):
    assert got.keys() == ref.keys()
    for key in ref:
        g = got[key]
        if scale is not None:
            g = g * scale ** _power(case, key)
        np.testing.assert_allclose(
            g,
            ref[key],
            rtol=case.rtol,
            atol=case.rtol * 1e-2,
            err_msg=key,
        )


def _rescaled(case, data, k):
    out = dict(data)
    for key in case.times:
        out[key] = np.asarray(data[key], dtype=float) * k
    return out


def _permuted(case, data):
    size = len(data[case.rows[0]])
    perm = scramble(size)
    return {
        key: np.asarray(v)[perm] if key in case.rows else v
        for key, v in data.items()
    }


def _expanded(case, data):
    n = np.asarray(data["n"])
    return {
        key: np.repeat(np.asarray(v), n, axis=0) if key in case.rows else v
        for key, v in data.items()
        if key != "n"
    }


@pytest.mark.parametrize("case", cases_for("units"))
def test_change_of_units(case):
    ref = predictions(case, fitted(case))
    model = refit(case, _rescaled(case, case.data(), K))
    got = predictions(case, model, x=case.x * K)
    _compare(case, got, ref, scale=K)


@pytest.mark.parametrize("case", cases_for("row_order"))
def test_data_row_order(case):
    ref = predictions(case, fitted(case))
    got = predictions(case, refit(case, _permuted(case, case.data())))
    _compare(case, got, ref)


@pytest.mark.parametrize("case", cases_for("counts"))
def test_counts_equal_repeated_rows(case):
    data = case.data()
    assert np.any(np.asarray(data["n"]) > 1), "the fixture needs counts"
    ref = predictions(case, fitted(case))
    got = predictions(case, refit(case, _expanded(case, data)))
    _compare(case, got, ref)
