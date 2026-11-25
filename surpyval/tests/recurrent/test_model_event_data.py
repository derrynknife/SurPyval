"""The package reads a fitted recurrent model's data through
``event_data(model)``, so what ``model.data`` hands users can become a
SurpyvalData with units ``i`` without moving a number; and the recurrent
diagnostics that take data take those rows as well as an
EventHistoryData, with the same answers."""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import SurpyvalData
from surpyval.recurrent import diagnostics
from surpyval.utils.model_data import as_event_history, event_data

X = [2.0, 5.0, 8.0, 3.0, 9.0, 12.0, 1.0, 4.0, 15.0]
I = [1, 1, 1, 2, 2, 2, 3, 3, 3]  # noqa: E741 - the package's name for ids
C = [0, 0, 1, 0, 0, 1, 0, 0, 1]


@pytest.fixture(scope="module")
def model():
    return sp.CrowAMSAA.fit(X, i=I, c=C)


@pytest.fixture(scope="module")
def rows():
    return SurpyvalData(X, C, i=I)


def test_event_data_is_the_data_the_fit_worked_on(model):
    assert event_data(model) is model.data
    assert event_data(sp.CrowAMSAA.from_params([2.0, 1.5])) is None


def test_rows_read_as_the_fits_read_them(model, rows):
    assert as_event_history(rows).equals(model.data, rtol=0)
    assert as_event_history(None) is None
    assert as_event_history(model.data) is model.data


def test_item_windows_take_the_rows(model, rows):
    a = diagnostics.item_windows(rows)
    b = diagnostics.item_windows(model.data)
    assert len(a) == len(b) == 3
    for one, other in zip(a, b):
        assert len(one) == len(other)
        for u, v in zip(one, other):
            assert np.array_equal(u, v)


@pytest.mark.parametrize(
    "residuals",
    [
        diagnostics.cumulative_hazard_residuals,
        diagnostics.martingale_residuals,
    ],
)
def test_residuals_from_the_rows_are_the_models(model, rows, residuals):
    # Bit-identical: the rows are read back into the same histories.
    assert np.array_equal(
        residuals(rows, model.cif), residuals(model.data, model.cif)
    )


@pytest.mark.parametrize("test", ["laplace", "mil_hdbk_189c"])
def test_trend_test_of_the_rows_is_the_models(model, rows, test):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        a = diagnostics.trend_test(rows, test=test)
    b = diagnostics.trend_test(model.data, test=test)
    assert a.statistic == b.statistic and a.p_value == b.p_value
