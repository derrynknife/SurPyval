"""Shapes and vectorisation (#379).

A model's functions are elementwise in the query:

- a scalar query gives one value, the same as the one-element array;
- a 2-D query keeps its shape and agrees with the flattened query;
- an empty query gives an empty result (not an error);
- permuting the query permutes the result (the input-order class of bugs:
  unsorted times broke competing-risks Cox, Cox predictions and the
  integrated Brier score);
- with covariates, the rows are independent: evaluating several rows
  together gives what each gives alone, and one vector broadcasts over
  every time (the survival-tree routing bug).
"""

import numpy as np
import pytest

from surpyval.tests.conformance.registry import (
    WITH_COVARIATES,
    call,
    call_native,
    calls,
    cases_for,
    fitted,
    query,
    scramble,
)


@pytest.mark.parametrize("case", cases_for("scalar"))
def test_scalar_query(case):
    model = fitted(case)
    for fname, event in calls(case):
        x = query(case, fname)
        k = len(x) // 2
        if case.interface in WITH_COVARIATES:
            z = case.Z[k]
            got = call_native(case, model, fname, x[k], z, event)
            ref = call_native(case, model, fname, x[k : k + 1], z, event)
        else:
            got = call_native(case, model, fname, x[k], event=event)
            ref = call_native(case, model, fname, x[k : k + 1], event=event)
        assert np.size(got) == 1, (fname, np.shape(got))
        np.testing.assert_allclose(
            np.ravel(np.asarray(got, float)),
            np.ravel(np.asarray(ref, float)),
            rtol=1e-12,
            atol=1e-15,
        )


@pytest.mark.parametrize("case", cases_for("array2d"))
def test_2d_query_keeps_its_shape(case):
    model = fitted(case)
    for fname, event in calls(case):
        x = query(case, fname)[:4]
        grid = x.reshape(2, -1)
        got = np.asarray(call(case, model, fname, grid, event=event), float)
        flat = np.asarray(call(case, model, fname, x, event=event), float)
        assert got.shape == grid.shape, (fname, got.shape)
        np.testing.assert_array_equal(got.ravel(), flat)


@pytest.mark.parametrize("case", cases_for("empty"))
def test_empty_query(case):
    model = fitted(case)
    for fname, event in calls(case):
        x = np.array([], dtype=float)
        if case.interface in WITH_COVARIATES:
            Z = np.empty((0, case.Z.shape[1]))
            if case.z_style == "single":
                Z = case.Z[0]
            got = call_native(case, model, fname, x, Z, event)
        else:
            if case.interface == "bivariate" and fname != "qf":
                x = np.empty((0, 2))
            got = call_native(case, model, fname, x, event=event)
        assert np.size(got) == 0, (fname, np.shape(got))


@pytest.mark.parametrize("case", cases_for("query_order"))
def test_query_order_equivariance(case):
    model = fitted(case)
    for fname, event in calls(case):
        x = query(case, fname)
        perm = scramble(len(x))
        Z = None if case.Z is None or fname == "qf" else case.Z
        ref = np.asarray(call(case, model, fname, x, Z, event), float)
        got = np.asarray(
            call(
                case,
                model,
                fname,
                x[perm],
                None if Z is None else Z[perm],
                event,
            ),
            float,
        )
        np.testing.assert_array_equal(got, ref[perm], err_msg=fname)


@pytest.mark.parametrize("case", cases_for("row_independence"))
def test_covariate_rows_are_independent(case):
    model = fitted(case)
    x, Z = case.x, case.Z
    for fname, event in calls(case):
        if fname in case.jump_functions:
            continue
        together = np.asarray(call(case, model, fname, x, Z, event), float)
        alone = np.array(
            [
                np.asarray(
                    call(
                        case, model, fname, x[k : k + 1], Z[k : k + 1], event
                    ),
                    float,
                ).item()
                for k in range(len(x))
            ]
        )
        np.testing.assert_allclose(
            together, alone, rtol=1e-12, atol=0, err_msg=fname
        )
        # One vector is used at every time, as the same row repeated.
        z = Z[1]
        one = np.asarray(call(case, model, fname, x, z, event), float)
        tiled = np.asarray(
            call(case, model, fname, x, np.tile(z, (len(x), 1)), event),
            float,
        )
        np.testing.assert_allclose(one, tiled, rtol=1e-12, atol=0)
