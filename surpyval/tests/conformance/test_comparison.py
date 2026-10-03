"""Every fitted model spells its model-comparison values the same way
(#572; principle 21).

Comparing candidate models -- one Weibull against a mixture, an NHPP
against an HPP, an AFT against a PH model -- needs the same values from
each, and they were spelt differently by family: ``aic`` a method on a
parametric model and a property on a recurrence model, and a mixture's
``loglike`` the *negative* log-likelihood. The convention (Conventions,
"Parametric models") is the one most families follow:

- ``neg_ll()``, ``aic()``, ``aic_c()`` and ``bic()`` are methods taking
  no argument, each returning a number;
- ``log_likelihood`` is the number itself, not a method, and is
  ``-neg_ll()``;
- ``aic()`` is ``2 k + 2 neg_ll()`` for a whole number ``k >= 1`` of
  parameters, and ``bic()`` and ``aic_c()`` penalise the same likelihood.

A model need not have every one (a non-parametric estimate has none),
but those it has must be spelt so. One that is not available for a fit
(a model with no likelihood) raises ``ValueError`` saying so. The old
spellings still work, with a ``DeprecationWarning``
(``surpyval.utils.deprecation``), which this property treats as an
error.
"""

import numbers
import warnings

import numpy as np
import pytest

from surpyval.tests.conformance.registry import cases_for, fitted

METHODS = ("neg_ll", "aic", "aic_c", "bic")


def _has(model, name):
    try:
        getattr(model, name)
    except AttributeError:
        return False
    return True


@pytest.mark.parametrize("case", cases_for("comparison"))
def test_model_comparison_values_are_spelt_alike(case):
    model = fitted(case)
    values = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for name in METHODS:
            if not _has(model, name):
                continue
            method = getattr(model, name)
            assert callable(method), f"{name} is not a method"
            try:
                value = method()
            except ValueError:
                # Not available for this fit (fitted without a
                # likelihood, or to no data), and it says so
                continue
            assert isinstance(value, numbers.Real), (name, value)
            values[name] = float(value)
        ll = None
        try:
            if _has(model, "log_likelihood"):
                ll = model.log_likelihood
        except ValueError:
            pass
        if ll is not None:
            assert isinstance(ll, numbers.Real), "log_likelihood is a method"
            values["log_likelihood"] = float(ll)
    if "log_likelihood" in values and "neg_ll" in values:
        np.testing.assert_allclose(
            values["log_likelihood"], -values["neg_ll"], rtol=1e-12
        )
    nll = values.get("neg_ll", -values.get("log_likelihood", np.nan))
    if "aic" in values and np.isfinite(nll):
        k = (values["aic"] - 2 * nll) / 2
        assert round(k) >= 1 and k == pytest.approx(round(k), abs=1e-6), k
        for name in ("bic", "aic_c"):
            if name in values and np.isfinite(values[name]):
                # The same likelihood, penalised by a positive amount
                assert values[name] > 2 * nll, name
