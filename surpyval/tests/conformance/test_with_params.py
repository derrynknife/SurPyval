"""``with_params`` rebuilds a model from its own estimates.

The shared contract of ``with_params``: every model that has it, and the
private ``_estimates()`` (its fitted values in the order of
``covariance_names``), rebuilds itself from that vector --
``model.with_params(model._estimates())`` predicts what ``model`` does --
and a vector of the wrong length (one entry too many, or none) is refused
with a ``ValueError``. Every registered case is tried and those whose
fitted model lacks either method are skipped, so a model that gains the
two is checked without editing this file.
"""

import warnings

import numpy as np
import pytest

from surpyval.tests.conformance.registry import CASES, fitted, predictions

ALL_CASES = [pytest.param(case, id=case.name) for case in CASES]


def _model_with_contract(case):
    """The case's fitted model, or a skip where it has no ``with_params``
    or ``_estimates``."""
    model = fitted(case)
    if not (
        callable(getattr(model, "with_params", None))
        and callable(getattr(model, "_estimates", None))
    ):
        pytest.skip(f"{case.name}: no with_params and _estimates")
    return model


@pytest.mark.parametrize("case", ALL_CASES)
def test_with_params_of_estimates_reproduces_predictions(case):
    model = _model_with_contract(case)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        rebuilt = model.with_params(model._estimates())
        ref = predictions(case, model)
        new = predictions(case, rebuilt)
    assert type(rebuilt) is type(model)
    assert rebuilt is not model
    assert ref.keys() == new.keys()
    for key in ref:
        np.testing.assert_allclose(
            new[key], ref[key], rtol=1e-12, atol=1e-14, err_msg=key
        )


@pytest.mark.parametrize("case", ALL_CASES)
def test_with_params_refuses_a_vector_of_the_wrong_length(case):
    model = _model_with_contract(case)
    estimates = np.asarray(model._estimates(), dtype=float)
    with pytest.raises(ValueError):
        model.with_params(np.append(estimates, 1.0))
    with pytest.raises(ValueError):
        model.with_params(estimates[:0])
