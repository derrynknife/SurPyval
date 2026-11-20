"""``rescale_time(factor)``: a model on another clock (principle 6).

Every registered parametric distribution (with an offset, a
limited-failure population and zero inflation), mixture and point mass
either moves to a new clock or refuses, saying what to do instead:

- **identities**: the rescaled model's ``sf``, ``ff`` and ``Hf`` at
  ``t`` are the model's at ``factor * t``, its ``df`` and ``hf`` are
  ``factor`` times the model's there, and its ``qf`` is the model's over
  ``factor``, to 1e-13, for a duty cycle (0.37) and a change of unit
  (24);
- **refit**: rescaling the fit by ``1 / K`` is the fit to the data
  multiplied by ``K`` (the ``units`` property's refit) -- its
  predictions, covariance, Wald bounds and log-likelihood, to the
  optimiser's tolerance (the case's ``rtol``);
- **saving**: the rescaled model round-trips through ``to_dict`` with
  its predictions and its new unit label;
- **refusals**: the distributions in :data:`REFUSED` raise a
  ``ValueError`` naming ``rescale_time``.

A new distribution of time declares how its parameters move
(``ParametricFitter._time_roles``); until it does, its case fails here,
unless it is listed in :data:`REFUSED` with the reason.
"""

import warnings

import numpy as np
import pytest

import surpyval
from surpyval.tests.conformance.checks import rescaled
from surpyval.tests.conformance.registry import (
    CASES,
    Q_PROBS,
    cases_for,
    fitted,
    predictions,
    refit,
)

K = 7.3  # as the units property: an awkward change of unit
FACTORS = (0.37, 24.0)  # a duty cycle, and hours to days
RTOL = 1e-13

#: The model classes that have ``rescale_time``.
CLASSES = frozenset(
    {
        "surpyval.Parametric",
        "surpyval.MixtureModel",
        "surpyval.NeverOccurs",
        "surpyval.InstantlyOccurs",
    }
)

_COUNTS = "a distribution of counts: a count has no unit of time"
#: The cases whose distribution refuses, and why.
REFUSED = {
    "Beta": "its support (0, 1) would move: it is a Beta4 on a new clock",
    "ConformanceGompertz": (
        "a CustomDistribution does not say which parameters are times"
    ),
    **dict.fromkeys(
        (
            "Poisson",
            "Geometric",
            "NegativeBinomial",
            "DiscreteWeibull",
            "BetaGeometric",
            "Discretize(Weibull)",
            "Binomial",
            "Bernoulli",
            "FixedEventProbability",
        ),
        _COUNTS,
    ),
}

_OURS = [c for c in CASES if c.model_class in CLASSES]
RESCALED = [pytest.param(c, id=c.name) for c in _OURS if c.name not in REFUSED]
REFUSING = [pytest.param(c, id=c.name) for c in _OURS if c.name in REFUSED]


def test_refused_cases_are_registered():
    names = {c.name for c in _OURS}
    assert set(REFUSED) <= names, set(REFUSED) - names


def _query(case):
    """The case's query and 20 points across it."""
    x = np.asarray(case.x, dtype=float).ravel()
    finite = x[np.isfinite(x)]
    grid = np.linspace(finite.min(), finite.max(), 20)
    return np.concatenate([x, grid])


@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("case", RESCALED)
def test_rescaled_functions_are_the_models_at_factor_t(case, factor):
    model = fitted(case)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        new = model.rescale_time(factor)
        t = _query(case)
        for fname in ("sf", "ff", "Hf", "df", "hf"):
            if fname not in case.functions:
                continue
            got = getattr(new, fname)(t / factor)
            want = getattr(model, fname)(t)
            if fname in ("df", "hf"):
                want = factor * np.asarray(want)
            np.testing.assert_allclose(
                got, want, rtol=RTOL, atol=0, err_msg=fname
            )
        if "qf" in case.functions:
            np.testing.assert_allclose(
                new.qf(Q_PROBS),
                np.asarray(model.qf(Q_PROBS)) / factor,
                rtol=RTOL,
                atol=0,
                err_msg="qf",
            )
    if case.model_class != "surpyval.Parametric":
        return
    assert new.dist is model.dist or new.dist.name == model.dist.name
    assert new.extras.keys() == model.extras.keys()
    for key in ("lfp_p", "f0"):
        assert new.extras.get(key) == model.extras.get(key)


def _refit_cases():
    return cases_for(
        "units",
        where=lambda c: c.model_class
        in {"surpyval.Parametric", "surpyval.MixtureModel"}
        and c.name not in REFUSED,
    )


@pytest.mark.parametrize("case", _refit_cases())
def test_rescaled_fit_is_the_fit_in_the_new_unit(case):
    model = fitted(case)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        new = model.rescale_time(1 / K)
        ref = refit(case, rescaled(case, case.data(), K))
        x = np.asarray(case.x, dtype=float) * K
        got = predictions(case, new, x=x)
        want = predictions(case, ref, x=x)
    for key in want:
        np.testing.assert_allclose(
            got[key],
            want[key],
            rtol=case.rtol,
            atol=case.rtol * 1e-2,
            err_msg=key,
        )
    assert new.maximum == ref.maximum
    _same_covariance(case, new, ref)
    _same_likelihood(case, new, ref)


def _same_covariance(case, new, ref):
    try:
        want = np.asarray(ref.covariance(), dtype=float)
    except ValueError:
        with pytest.raises(ValueError):
            new.covariance()
        return
    got = np.asarray(new.covariance(), dtype=float)
    scale = np.max(np.abs(want))
    np.testing.assert_allclose(
        got,
        want,
        rtol=case.rtol,
        atol=case.rtol * scale,
        err_msg="covariance",
    )
    if case.model_class == "surpyval.Parametric":
        x = np.asarray(case.x, dtype=float) * K
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.testing.assert_allclose(
                new.cb(x, on="sf"),
                ref.cb(x, on="sf"),
                rtol=case.rtol,
                atol=case.rtol * 1e-2,
                err_msg="cb",
            )


def _same_likelihood(case, new, ref):
    try:
        want = ref.neg_ll()
    except ValueError:
        return
    np.testing.assert_allclose(
        new.neg_ll(),
        want,
        rtol=case.rtol,
        atol=case.rtol,
        err_msg="neg_ll",
    )


@pytest.mark.parametrize("case", RESCALED)
def test_rescaled_model_saves_and_loads(case):
    model = fitted(case)
    if not hasattr(model, "to_dict") or not hasattr(model, "unit"):
        pytest.skip(f"{case.name}: the class itself is the model")
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        new = model.rescale_time(24.0, unit="days")
        back = surpyval.from_dict(new.to_dict())
        got = predictions(case, back)
        want = predictions(case, new)
    assert back.unit == new.unit == "days"
    for key in want:
        np.testing.assert_allclose(
            got[key], want[key], rtol=1e-14, atol=0, err_msg=key
        )


@pytest.mark.parametrize("case", REFUSING)
def test_refused_distributions_say_so(case):
    model = fitted(case)
    with pytest.raises(ValueError, match="rescale_time"):
        model.rescale_time(2.0)
