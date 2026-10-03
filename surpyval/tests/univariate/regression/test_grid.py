"""Survival curves for many subjects at many times (#488).

``sf(x, Z)`` pairs row ``i`` of ``Z`` with ``x[i]`` (one row is used at
every time, one time for every row). Three times with four rows was a raw
numpy broadcast error, and three with three silently paired them -- the
same call a grid or a mistake depending on the number of subjects. Other
counts are now refused, naming the rule, and ``grid=True`` gives every
time for every row, ``(len(Z),) + x.shape``, as a survival forest does.
"""

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.tests._helpers import rossi_with_censoring

TIMES = np.array([10.0, 30.0, 50.0])
SUBJECTS = np.array([[0.0, 20.0], [1.0, 20.0], [0.0, 40.0], [1.0, 35.0]])
FITTERS = ["CoxPH", "WeibullPH", "WeibullAFT", "WeibullPO", "WeibullAH"]
FUNCTIONS = ["sf", "ff", "df", "hf", "Hf"]


def _fit(name):
    df = rossi_with_censoring()
    fitter = getattr(sp, name)
    return fitter.fit(
        df.week.values, df[["fin", "age"]].values, df.censored.values
    )


@pytest.mark.parametrize("name", FITTERS)
def test_mismatched_rows_are_refused(name):
    model = _fit(name)
    for fn in FUNCTIONS:
        with pytest.raises(ValueError, match="4 covariate rows for 3 times"):
            getattr(model, fn)(TIMES, SUBJECTS)
    if name != "CoxPH":
        with pytest.raises(ValueError, match="one row at a time"):
            model.cb(TIMES, SUBJECTS)
    # The pairings that were always allowed still are.
    assert getattr(model, "sf")(TIMES, SUBJECTS[:3]).shape == (3,)
    assert getattr(model, "sf")(TIMES, SUBJECTS[0]).shape == (3,)
    assert getattr(model, "sf")(TIMES[:1], SUBJECTS).shape == (4,)


@pytest.mark.parametrize("name", FITTERS)
def test_grid(name):
    model = _fit(name)
    for fn in FUNCTIONS:
        f = getattr(model, fn)
        grid = f(TIMES, SUBJECTS, grid=True)
        assert grid.shape == (4, 3)
        # Row i is subject i at every time.
        for i, row in enumerate(SUBJECTS):
            np.testing.assert_allclose(grid[i], f(TIMES, row), rtol=1e-12)
    # Shape in, shape out, with the rows in front.
    x2 = TIMES.reshape(1, 3)
    assert model.sf(x2, SUBJECTS, grid=True).shape == (4, 1, 3)
    assert model.sf(20.0, SUBJECTS, grid=True).shape == (4,)
    assert model.sf(TIMES, SUBJECTS[0], grid=True).shape == (1, 3)
    assert model.sf([], SUBJECTS, grid=True).shape == (4, 0)
    # A square grid is still a grid, not the pairs.
    square = model.sf(TIMES, SUBJECTS[:3], grid=True)
    np.testing.assert_allclose(
        np.diag(square), model.sf(TIMES, SUBJECTS[:3]), rtol=1e-12
    )
    assert not np.allclose(square[0], model.sf(TIMES, SUBJECTS[:3]))


def test_grid_from_a_data_frame_and_a_stratum():
    df = rossi_with_censoring()
    model = sp.CoxPH.fit_from_df(
        df, x_col="week", c_col="censored", Z_cols=["fin", "age"]
    )
    new = pd.DataFrame(SUBJECTS, columns=["fin", "age"])
    np.testing.assert_allclose(
        model.sf(TIMES, new, grid=True), model.sf(TIMES, SUBJECTS, grid=True)
    )
    strat = sp.CoxPH.fit(
        df.week.values, df[["age"]].values, df.censored.values, strata=df.fin
    )
    grid = strat.sf(TIMES, [[20.0], [30.0]], stratum=1, grid=True)
    np.testing.assert_allclose(
        grid[1], strat.sf(TIMES, [30.0], stratum=1), rtol=1e-12
    )
    # One covariate: a 1-D Z is one value per subject on the grid.
    np.testing.assert_allclose(
        strat.sf(TIMES, [20.0, 30.0], stratum=1, grid=True), grid
    )
