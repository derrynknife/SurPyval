"""``how`` (the estimation method) is accepted case-insensitively.

The method is a short string that is often typed by hand, so ``"mle"`` or
``"Mpp"`` should behave exactly like the canonical ``"MLE"`` / ``"MPP"``
rather than raising. The parametric fitters normalise the argument before
use. An unknown method is still rejected.
"""

import numpy as np
import pytest

import surpyval as surv

W = surv.Weibull


def _sample():
    np.random.seed(1)
    return W.random(50, 10, 2)


def test_how_lowercase_matches_canonical():
    x = _sample()
    canonical = W.fit(x, how="MLE")
    assert W.fit(x, how="mle").params == pytest.approx(canonical.params)


def test_how_mixed_case_matches_canonical():
    x = _sample()
    canonical = W.fit(x, how="MLE")
    assert W.fit(x, how="Mle").params == pytest.approx(canonical.params)


def test_how_still_rejects_an_unknown_method():
    x = _sample()
    with pytest.raises(ValueError, match='"how" must be one of'):
        W.fit(x, how="not-a-method")


def test_fit_best_metric_error_is_spelled_correctly():
    with pytest.raises(ValueError, match="`metric` must be one of"):
        surv.fit_best(_sample(), metric="not-a-metric")
