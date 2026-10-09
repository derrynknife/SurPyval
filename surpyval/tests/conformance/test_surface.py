"""One spelling for each quantity across the fitted models (#572).

Comparing candidate models is the core of a life-data study -- one
Weibull against a two-population mixture, an NHPP against an HPP, an
accelerated failure time model against a proportional hazards one -- and
the analyst reaches for the same names on each. Those names must mean
the same kind of thing on every model: a method called as ``model.aic()``
on one and a value read as ``model.aic`` on another breaks every script
that compares the two (``TypeError: 'float' object is not callable``,
or a bound method compared with a number).

- **one kind**: each name in :data:`NAMES` is either a method on every
  registered model that has it, or a value on every one;
- **comparable**: every full-likelihood fit (the scope of the
  ``maximum`` property, less the partial likelihoods) offers ``aic`` and
  ``bic``, so any two of them can be ranked.

``test_comparison.py`` checks each model's own spelling of the
comparison values against the convention; this module checks the names
against each other across the models, and that none of the
full-likelihood fits is left without them. A name spelt both ways is a
strict xfail in :data:`INCONSISTENT`, and a fit that cannot be ranked one
in :data:`NOT_COMPARABLE`, each led by its issue; it turns red the day
the gap is closed, as the reminder to delete the entry. This module reads
the fitted models only: it costs nothing beyond the registry's cached
fits.
"""

import functools
import inspect
from collections import defaultdict

import pytest

from surpyval.tests.conformance.registry import CASES, fitted
from surpyval.tests.conformance.registry_families import NOT_A_LIKELIHOOD_FIT

# The quantities a model comparison or a report reads off a fitted model.
NAMES = (
    "aic",
    "aic_c",
    "bic",
    "neg_ll",
    "log_likelihood",
    "covariance",
    "cov_matrix",
    "standard_errors",
    "summary",
    "params",
    "parameter_names",
    "maximum",
)
# A full-likelihood fit offers these, so any two can be ranked.
COMPARABLE = ("aic", "bic")

# Names spelt both ways, each reason led by its issue (none since #572:
# aic, bic and covariance are methods on every model, log_likelihood a
# value).
INCONSISTENT: dict[str, str] = {}

# Fits whose likelihood is a partial (or profile) likelihood: its value
# depends on the risk sets, not on a full model of the times, so an
# information criterion would not rank it against the parametric fits.
PARTIAL_LIKELIHOOD = frozenset(
    {
        "surpyval.SemiParametricRegressionModel",
        "surpyval.CoxFrailtyModel",
        "surpyval.ProportionalOddsModel",
        "surpyval.univariate.competing_risks."
        "CompetingRisksProportionalHazards",
        "surpyval.univariate.competing_risks.regression.fine_gray."
        "FineGrayModel",
    }
)
# Full-likelihood fits that cannot be ranked against the others, and why.
# (MixtureModel has aic and bic since #572; the degradation process models,
# DestructiveDegradation and CauseSpecificNHPP since #711.) The reason
# leads with its issue, or with NO_ISSUE until one is filed.
NO_ISSUE = "no issue yet: "
NOT_COMPARABLE: dict[str, str] = {}


def kind(model, name):
    """``"method"`` or ``"value"`` for ``model.<name>``, or ``None``.

    Read statically where it can be: a property is a value even when
    reading it raises (``log_likelihood`` of a model fitted without a
    likelihood raises ``ValueError`` saying so)."""
    try:
        found = inspect.getattr_static(model, name)
    except AttributeError:
        try:  # (an attribute made by __getattr__)
            found = getattr(model, name)
        except Exception:
            return None
    if isinstance(found, (property, functools.cached_property)):
        return "value"
    if isinstance(found, (staticmethod, classmethod)):
        return "method"
    return "method" if inspect.isroutine(found) else "value"


def _models():
    out = []
    for case in CASES:
        try:
            out.append((case, fitted(case)))
        except Exception:  # a case whose own fit fails is reported there
            continue
    return out


def _name_params():
    return [
        pytest.param(
            name,
            id=name,
            marks=(
                [pytest.mark.xfail(strict=True, reason=INCONSISTENT[name])]
                if name in INCONSISTENT
                else []
            ),
        )
        for name in NAMES
    ]


@pytest.mark.parametrize("name", _name_params())
def test_one_kind_per_name(name):
    kinds = defaultdict(list)
    for case, model in _models():
        found = kind(model, name)
        if found is not None:
            kinds[found].append(case.name)
    assert len(kinds) <= 1, f"{name!r} is " + " and ".join(
        f"a {k} on {len(v)} models ({', '.join(v[:4])}, ...)"
        for k, v in kinds.items()
    )


def _likelihood_cases():
    params = []
    for case in CASES:
        if case.model_class in {*NOT_A_LIKELIHOOD_FIT, *PARTIAL_LIKELIHOOD}:
            continue
        if not case.applies("maximum"):
            continue
        reason = NOT_COMPARABLE.get(case.name)
        marks = (
            [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        )
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params


@pytest.mark.parametrize("case", _likelihood_cases())
def test_likelihood_fits_are_comparable(case):
    model = fitted(case)
    missing = [name for name in COMPARABLE if not hasattr(model, name)]
    assert not missing, f"{case.name} has no {', '.join(missing)}"


def test_known_inconsistencies_name_their_issue():
    import re

    for reason in {**INCONSISTENT, **NOT_COMPARABLE}.values():
        assert re.match(rf"#\d+: |{NO_ISSUE}", reason), reason
