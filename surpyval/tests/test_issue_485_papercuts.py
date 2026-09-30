"""Regression tests for the small API papercuts in issue #485."""

import math

import numpy as np
import pytest

from surpyval import KaplanMeier, Turnbull, Weibull, fit_best, from_json
from surpyval.utils import xcnt_handler


def test_how_is_case_insensitive():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    lower = Weibull.fit(x, how="mle")
    upper = Weibull.fit(x, how="MLE")
    np.testing.assert_allclose(lower.params, upper.params)


def test_fit_best_metric_typo_message():
    with pytest.raises(ValueError, match="must be one of"):
        fit_best([1.0, 2.0, 3.0], metric="not-a-metric")


def test_to_json_without_path_returns_string():
    model = Weibull.from_params([10.0, 2.0])
    payload = model.to_json()
    assert isinstance(payload, str)
    restored = from_json(payload)
    np.testing.assert_allclose(restored.params, model.params)


def test_from_json_accepts_string_and_file(tmp_path):
    model = Weibull.from_params([5.0, 1.5])
    path = tmp_path / "m.json"
    written = model.to_json(path)
    assert written is None
    from_file = from_json(path)
    from_text = from_json(path.read_text())
    np.testing.assert_allclose(from_file.params, model.params)
    np.testing.assert_allclose(from_text.params, model.params)


def test_kaplan_meier_interval_censoring_message():
    x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    with pytest.raises(ValueError, match="Kaplan-Meier can't handle"):
        KaplanMeier.fit(x)
    Turnbull.fit(x)


def test_column_vector_x_is_accepted():
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    parsed, c, n, t = xcnt_handler(x=x)
    np.testing.assert_array_equal(parsed, [1.0, 2.0, 3.0, 4.0])
    model = Weibull.fit(x)
    assert model.params is not None


def test_qf_outside_unit_interval_is_nan():
    model = Weibull.from_params([1.0, 2.0])
    assert math.isnan(float(model.qf(1.5)))
    assert math.isnan(float(model.qf(-0.1)))
    qs = model.qf([0.2, 1.5, 0.8])
    assert math.isfinite(qs[0])
    assert math.isnan(qs[1])
    assert math.isfinite(qs[2])
