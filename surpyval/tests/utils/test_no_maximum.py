"""One warning for a fit that did not reach a verified maximum,
``surpyval.utils.no_maximum.warn_unverified`` (principles 13 and 22).

The univariate, accelerated life, additive hazards, regression-ladder,
NHPP, proportional-intensity, semi-parametric proportional odds and
mixture fits each worded it their own way ("did not converge", "may not
be the maximum-likelihood estimates", "Precision was lost", "EM
algorithm reached max iterations ..."), pointed at different frames, and
only the univariate one was held back by ``quiet_maximum_warnings``.
"""

import os
import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import surpyval as sp
from surpyval.tests.conformance.registry import CASE_BY_NAME
from surpyval.univariate.regression import _fit_skeleton
from surpyval.univariate.regression.accelerated_life import (
    parameter_substitution,
)
from surpyval.univariate.regression.additive_hazards import (
    additive_hazards_fitter,
)
from surpyval.utils.no_maximum import (
    quiet_maximum_warnings,
    warn_unverified,
)

UNVERIFIED = "did not reach a verified maximum of the likelihood"


def _unverified(caught):
    return [w for w in caught if UNVERIFIED in str(w.message)]


def test_message_and_quiet():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_unverified("The fit", "it stopped", "rescale")
        with quiet_maximum_warnings():
            warn_unverified("The fit")
    assert len(caught) == 1
    assert caught[0].category is UserWarning
    assert str(caught[0].message) == (
        "The fit did not reach a verified maximum of the likelihood (a "
        "point where the gradient is zero and the log-likelihood curves "
        "down in every direction); the parameters returned are the best "
        "point it found. The likelihood may have no maximum -- a parameter "
        "running off to a limit of its range -- or the search may have "
        "stalled (it stopped): rescale."
    )
    assert caught[0].filename == __file__


def _starve_al(monkeypatch):
    # Every search ends unverified.
    real = parameter_substitution.verify_or_polish

    def unverified(*args, **kwargs):
        res, _ = real(*args, **kwargs)
        return res, False

    monkeypatch.setattr(parameter_substitution, "verify_or_polish", unverified)


def _starve_ah(monkeypatch):
    real = additive_hazards_fitter.verify_or_polish

    def unverified(*args, **kwargs):
        res, _ = real(*args, **kwargs)
        return res, False

    monkeypatch.setattr(
        additive_hazards_fitter.AdditiveHazardsFitter,
        "_gradient_first",
        lambda self, fun, true_neg_ll, init, n_obs, **kw: (None, False),
    )
    monkeypatch.setattr(
        additive_hazards_fitter, "verify_or_polish", unverified
    )


def _starve_ladder(monkeypatch):
    # The regression optimiser ladder stops short, unconverged.
    def stuck(fun, x0, *args, **kwargs):
        return OptimizeResult(
            x=np.asarray(x0), fun=fun(x0), success=False, message="forced"
        )

    monkeypatch.setattr(_fit_skeleton, "preconditioned_bfgs", stuck)
    monkeypatch.setattr(_fit_skeleton, "minimize", stuck)
    monkeypatch.setattr(_fit_skeleton, "minimize_with_gradient", stuck)
    # and the baseline's profile walk (#710), which re-optimises with its
    # own Newton steps and would otherwise rescue the starved search
    monkeypatch.setattr(_fit_skeleton, "walk_profile", lambda *a, **k: None)


_CASES = {
    "WeibullAL[Power]": _starve_al,
    "WeibullAH": _starve_ah,
    "WeibullPH": _starve_ladder,
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_one_warning_at_the_caller_held_back_when_quiet(name, monkeypatch):
    case = CASE_BY_NAME[name]
    _CASES[name](monkeypatch)
    data = case.data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        case.fit(data)
    found = _unverified(caught)
    assert len(found) == 1, [str(w.message) for w in caught]
    # At the caller: the registry's call of the fitter (in one of the
    # registry_*.py modules), not package code.
    assert os.path.basename(found[0].filename).startswith("registry")
    # Inside ``quiet_maximum_warnings`` (``fit_best``) it is not given.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with quiet_maximum_warnings():
            case.fit(data)
    assert not _unverified(caught)


def test_the_univariate_precision_loss_says_so_in_the_same_words(
    monkeypatch,
):
    # "Precision was lost, try: ..." was a second wording of an
    # unverified maximum-likelihood answer.
    from surpyval.univariate.parametric.fitters import mle

    def lost(real):
        def search(*args, **kwargs):
            res = real(*args, **kwargs)
            res.success = False
            res.message = (
                "Desired error not necessarily achieved due to precision "
                "loss."
            )
            return res

        return search

    monkeypatch.setattr(
        mle, "minimize_with_gradient", lost(mle.minimize_with_gradient)
    )
    monkeypatch.setattr(
        mle, "preconditioned_bfgs", lost(mle.preconditioned_bfgs)
    )
    monkeypatch.setattr(mle, "is_local_minimum", lambda *a, **k: False)
    x = sp.Weibull.random(30, 10, 2, random_state=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sp.Gamma.fit(x)
    found = _unverified(caught)
    assert model.maximum == "unverified"
    assert len(found) == 1, [str(w.message) for w in caught]
    assert "loss of precision" in str(found[0].message)
    assert not hasattr(model, "_unverified_reason")
