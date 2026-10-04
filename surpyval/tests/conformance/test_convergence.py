"""Failure is never silent (principle 13, #401).

An optimiser that does not converge warns, or the fit raises a
``ValueError`` saying why; a fit never quietly returns its starting
values. Each registered case whose fit iterates declares in ``starve``
how to make it fail (see ``Case.starve`` in ``registry.py``):

- a public iteration limit set to 1 (``max_iter``);
- a start (``init``) far from the maximum -- the fitted value times a
  million -- for a fitter with no iteration limit;
- data whose likelihood has no maximum: a covariate level with no
  events, dependence at its limit, noise-free degradation readings.

The starved fit must then raise ``ValueError``, or give a deliberate
warning the normal fit does not, or -- having found the maximum after
all, which a robust fitter may from a far start -- give the same model
as the normal fit (never the case for data with no maximum). Only
silence with a different model fails.

A raw numerical warning leaking from a starved fit (an overflow at a
start a million times too large) is not this property's concern and does
not fail it; ``conftest.py`` checks for leaks everywhere else.

``test_a_fit_leaves_its_initial_guess`` checks the other half of the
principle where the fitter exposes its initial guess (the univariate
parametric fits, ``fitting_info["init"]``): the fitted parameters are
not the start. A case with nothing to starve (a closed-form or exact
estimator) excludes ``"convergence"`` with the reason, and
``test_every_case_is_starved_or_excluded`` keeps that decision explicit.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.conformance import leaks
from surpyval.tests.conformance.checks import compare
from surpyval.tests.conformance.registry import (
    CASES,
    cases_for,
    fitted,
    offset_data,
    predictions,
)

# The starved fit's answer, where it is silent, must be the normal fit's
# to this relative tolerance: a search from far away stops at its own
# tolerance, not at the normal fit's, and a model within 1% of the
# maximum everywhere has converged in every sense that matters here.
RTOL = 1e-2


def _deliberate(fit):
    """``fit()``, its result and the deliberate warnings it gave (raw
    numerical warnings are dropped; see the module docstring)."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with leaks.deliberate() as found:
            out = fit()
    return out, {message for _, message in found}


@pytest.mark.parametrize("case", cases_for("convergence"))
def test_a_fit_that_cannot_converge_says_so(case):
    _, usual = _deliberate(lambda: case.fit(case.data()))
    try:
        model, said = _deliberate(lambda: case.starve(case.data()))
    except ValueError:
        return
    if said - usual:
        return
    ref = predictions(case, fitted(case))
    try:
        # (A starved model's predictions may overflow too.)
        got, _ = _deliberate(lambda: predictions(case, model))
        compare(case, got, ref, rtol=RTOL)
    except AssertionError as error:
        pytest.fail(
            "the starved fit returned without a warning or an error, and "
            "not with the normal fit's model: {}\n{}".format(
                _params(model), str(error).strip().splitlines()[0]
            ),
            pytrace=False,
        )


def _params(model):
    for name in ("params", "beta", "coefficients"):
        value = getattr(model, name, None)
        if value is not None:
            return "{} {}".format(name, np.round(np.asarray(value), 4))
    return type(model).__name__


# ---------------------------------------------------------------------------
# The initial guess
# ---------------------------------------------------------------------------
_START = "convergence[initial guess]"


def _exposes_start(case):
    return case.model_class == "surpyval.Parametric"


def _start_cases():
    # Not ``cases_for``: its marks are the starved fit's known failures.
    params = []
    for case in CASES:
        if not (case.applies("convergence") and _exposes_start(case)):
            continue
        if _START in case.exclude:
            continue
        marks = [pytest.mark.slow] if "*" in case.slow else []
        if _START in case.xfail:
            marks.append(
                pytest.mark.xfail(strict=True, reason=case.xfail[_START])
            )
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params


def _fitted_vector(model):
    """The fitted parameters in ``init``'s order (see
    ``registry._parametric_start``)."""
    out = ([model.gamma] if model.offset else []) + list(model.params)
    out += [model.lfp_p] if model.lfp else []
    return np.array(out + ([model.f0] if model.zi else []), dtype=float)


@pytest.mark.parametrize("case", _start_cases())
def test_a_fit_leaves_its_initial_guess(case):
    model = fitted(case)
    info = getattr(model, "fitting_info", None)
    assert info is not None and info.get("init") is not None, (
        "an optimised parametric fit should expose its initial guess in "
        "fitting_info['init']"
    )
    start = np.asarray(info["inv_trans"](info["const"](info["init"])), float)
    got = _fitted_vector(model)
    assert start.shape == got.shape, (start, got)
    assert not np.allclose(
        got, start, rtol=1e-10, atol=0
    ), f"the fit returned its initial guess {start}"


def test_every_case_is_starved_or_excluded():
    undecided = [
        case.name
        for case in CASES
        if case.starve is None and "convergence" not in case.exclude
    ]
    assert not undecided, (
        "give these cases a `starve` (registry.py) or exclude "
        f"'convergence' with the reason: {undecided}"
    )


def test_init_is_checked_in_its_own_order():
    # An offset fit's ``init`` is [gamma, *params], the parameters'
    # positions, but its check read the names in the order they were
    # added, [*params, gamma]: it refused this valid start (gamma 6.0 is
    # below the first observation, 7.411) as "gamma = 10.0", and passed
    # an offset beyond the first observation, which then failed as "MLE
    # Failed" and "non-finite parameters". Found starving
    # Exponential[offset].
    data = offset_data()
    model = sp.Exponential.fit(**data, offset=True, init=[6.0, 10.0])
    np.testing.assert_allclose(
        [model.gamma, *model.params], [7.411, 0.1391], rtol=1e-3
    )
    with pytest.raises(ValueError, match="gamma = 100.0 lies outside"):
        sp.Exponential.fit(**data, offset=True, init=[100.0, 0.1])
