"""``parameter_bounds`` gives the range ``with_params`` keeps to.

A model with the public ``parameter_bounds`` gives one ``(low, high)``
pair per entry of the vector its ``with_params`` takes (``None`` for an
unbounded end). For every registered case whose fitted model has both,
each finite end is tried from the fitted estimates, one parameter at a
time:

- **just outside** (a relative 1e-6 beyond the end), ``with_params``
  refuses with a ``ValueError``, unless it ignores that entry (a
  parameter held fixed, or an accelerated life model's life parameter),
  which the rebuilt model shows by keeping the fitted value;
- **just inside**, ``with_params`` accepts it, unless the model also
  constrains its parameters jointly (:data:`JOINT`, each with the
  constraint): the per-parameter range cannot say that, and moving one
  parameter of a fitted set to the edge can break it.

Every model with the property also gives exactly one pair per estimate,
in the order of ``covariance_names`` (or, without that, of its
parameters).
"""

import warnings

import numpy as np
import pytest

from surpyval.tests.conformance.registry import CASES, fitted

ALL_CASES = [pytest.param(case, id=case.name) for case in CASES]

#: Cases whose parameters are also constrained jointly, so that a value
#: just inside one parameter's range, with the others at their estimates,
#: can still be refused. The constraint each one has.
JOINT = {
    "Uniform": "a < b",
    "Beta4": "a < b",
    "Binomial": "n is a whole number",
    "MixtureModel": "the weights sum to 1",
    "RoystonParmar": "the spline does not fall",
}


def _model(case):
    model = fitted(case)
    if not hasattr(type(model), "parameter_bounds"):
        pytest.skip(f"{case.name}: no parameter_bounds")
    return model


def _reference_size(model):
    for name in ("_estimates", "covariance_names", "_mle", "params"):
        value = getattr(model, name, None)
        if callable(value):
            value = value()
        if value is not None:
            return int(np.size(value))
    raise AssertionError("no parameter vector to compare with")


@pytest.mark.parametrize("case", ALL_CASES)
def test_one_range_per_parameter(case):
    model = _model(case)
    bounds = model.parameter_bounds
    assert isinstance(bounds, list)
    assert len(bounds) == _reference_size(model)
    for low, high in bounds:
        for end in (low, high):
            assert end is None or (isinstance(end, float) and np.isfinite(end))
        if low is not None and high is not None:
            assert low < high


def _try(model, values):
    """The rebuilt model, or the exception ``with_params`` raised."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            return model.with_params(values)
        except Exception as error:  # noqa: BLE001 -- checked by the caller
            return error


def _ends(model):
    """``(index, end, outward)`` for every finite end: ``outward`` is -1
    for a lower end and +1 for an upper."""
    for i, (low, high) in enumerate(model.parameter_bounds):
        if low is not None:
            yield i, low, -1.0
        if high is not None:
            yield i, high, 1.0


@pytest.mark.parametrize("case", ALL_CASES)
def test_with_params_keeps_to_the_ranges(case):
    model = _model(case)
    if not (
        callable(getattr(model, "with_params", None))
        and callable(getattr(model, "_estimates", None))
    ):
        pytest.skip(f"{case.name}: no with_params and _estimates")
    estimates = np.asarray(model._estimates(), dtype=float)
    problems = []
    for i, end, outward in _ends(model):
        step = 1e-6 * max(1.0, abs(end))
        names = getattr(model, "covariance_names", None)
        name = names[i] if names is not None else i

        outside = estimates.copy()
        outside[i] = end + outward * step
        result = _try(model, outside)
        if not isinstance(result, ValueError):
            if isinstance(result, Exception):
                problems.append(f"{name} = {outside[i]!r}: {result!r}")
            else:
                kept = np.asarray(result._estimates(), dtype=float)[i]
                if kept != estimates[i]:
                    problems.append(f"{name} = {outside[i]!r} accepted")

        if case.name in JOINT:
            continue
        inside = estimates.copy()
        inside[i] = end - outward * step
        result = _try(model, inside)
        if isinstance(result, Exception):
            problems.append(f"{name} = {inside[i]!r} refused: {result}")
    assert not problems, "; ".join(problems)
