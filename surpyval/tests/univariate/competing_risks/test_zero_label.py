"""A cause label of 0 with nothing censored (#486).

lifelines, scikit-survival and R's cmprsk code competing-risks data as one
integer column with 0 for a censored row. SurPyval marks a censored row
with a missing cause (``None`` / ``NaN``) or with ``c``, so such data were
read as a third cause called 0 and no censoring, and every incidence was
wrong, silently: on the issue's data the cause-1 incidence at 8 was 0.5
against 0.71875. The meaning of 0 is unchanged -- a SurPyval user may well
number causes from 0 -- but where the data look 0-coded (numeric labels
including 0, none missing, no ``c``) the fit warns and says how to convert
them.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from surpyval import gray_test
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    CompetingRisksProportionalHazards,
    FineGray,
    ParametricCompetingRisks,
)

X = [1, 2, 3, 4, 5, 6, 7, 8]
E = [1, 2, 1, 0, 1, 2, 0, 1]
ZERO = "Cause label 0 is taken as a cause, and no row is censored"


def _warnings(fit):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fit()
    return out, [w for w in caught if str(w.message).startswith(ZERO)]


def test_the_issue():
    model, found = _warnings(lambda: CompetingRisks.fit(X, E))
    assert len(found) == 1 and found[0].filename == __file__
    # The fit itself is unchanged: 0 is a cause.
    assert list(model.event_idx_map) == [0, 1, 2]
    assert model.cif([8], 1) == pytest.approx([0.5])
    # The conversion the warning gives.
    converted = np.where(np.asarray(E) == 0, None, E)
    model, found = _warnings(lambda: CompetingRisks.fit(X, converted))
    assert not found
    assert model.cif([8], 1) == pytest.approx([0.71875])


@pytest.mark.parametrize(
    "e, c",
    [
        (E, np.zeros(8)),  # c given: 0 is a cause, nothing censored
        ([1, 2, 1, None, 1, 2, 0, 1], None),  # None marks the censoring
        ([1, 2, 1, 3, 1, 2, 3, 1], None),  # no 0
        (["a", "b", "a", "0", "a", "b", "0", "a"], None),  # not a number
        ([True, False] * 4, None),  # booleans are not 0-coded causes
    ],
)
def test_no_warning(e, c):
    _, found = _warnings(lambda: CompetingRisks.fit(X, e, c=c))
    assert not found


def test_every_competing_risks_entry_point_warns_once():
    rng = np.random.default_rng(0)
    n = 60
    x = rng.exponential(size=n)
    e = rng.choice([0, 1, 2], n)
    Z = rng.normal(size=(n, 1))
    fits = [
        lambda: CompetingRisks.fit(x, e),
        lambda: FineGray.fit(x, Z, e, event=1),
        lambda: CompetingRisksProportionalHazards.fit(x, Z, e),
        lambda: ParametricCompetingRisks.fit(x, e),
        lambda: gray_test(x, e, np.arange(n) % 2, event=1),
        lambda: CompetingRisksProportionalHazards.fit_from_df(
            pd.DataFrame({"x": x, "e": e, "z": Z[:, 0]}), "x", "e", Z_cols="z"
        ),
    ]
    for fit in fits:
        _, found = _warnings(fit)
        assert len(found) == 1
        assert found[0].filename == __file__
