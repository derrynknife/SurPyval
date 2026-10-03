"""fit_best ranks only regular maxima (#492).

AIC, AIC_c and BIC assume a regular maximum of the likelihood. The Uniform
and the Beta4 (support ends among the parameters) are left out of the
default candidates, and any candidate that is not a verified maximum (it
warns "No finite maximum", or that its search did not reach a verified
maximum) is set aside: ranked only when no regular candidate fitted, and
named in one warning.

And fit_best takes its data as ``fit`` does (#570): checked once, so an
input error raises as ``fit`` raises it, with ``tl``, ``tr``, ``xl`` and
``xr`` passed through; a candidate that cannot be fitted is named in the
warning with a short reason, not the data.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
import surpyval as surv

SEVEN = np.arange(1, 8.0)
WEIBULL_50 = np.random.default_rng(5).weibull(2, 50) * 100


def _fit_best(*args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model = sp.fit_best(*args, **kwargs)
    return model, [str(w.message) for w in rec]


@pytest.mark.parametrize("metric", ["aic", "aic_c", "bic"])
def test_beta4_no_longer_wins_on_seven_points(metric):
    # Before: 'Beta4' under every metric, neg_ll -24.75 at a shape of 0.11
    model, _ = _fit_best(SEVEN, metric=metric)
    assert model.dist.name not in ("Beta4", "Uniform")


def test_uniform_no_longer_wins_on_weibull_data():
    # Before: 'Uniform' on [3.8, 147.7], neg_ll 248.5 against 255.3
    model, _ = _fit_best(WEIBULL_50)
    assert model.dist.name == "Rayleigh"  # a Weibull with shape 2


def test_a_named_non_regular_family_is_set_aside_with_a_warning():
    model, messages = _fit_best(WEIBULL_50, include=["Uniform", "Weibull"])
    assert model.dist.name == "Weibull"
    aside = [m for m in messages if m.startswith("fit_best set aside")]
    assert len(aside) == 1
    assert "Uniform (its support ends are parameters" in aside[0]


def test_a_fit_with_no_maximum_is_set_aside_and_its_warning_replaced():
    model, messages = _fit_best(SEVEN, include=["Beta4", "Weibull"])
    assert model.dist.name == "Weibull"
    assert not [m for m in messages if m.startswith("No finite maximum")]
    assert [m for m in messages if "Beta4 (its likelihood has no" in m]


def test_a_runaway_fit_is_set_aside():
    # The ExpoWeibull runs towards a limit of its shapes on this sample
    # (beta to infinity, mu to 0; the profile log-likelihood rises from
    # -253.47 at beta = 3 to -248.64 at beta = 1000) and its AIC (503.5)
    # used to beat every regular fit. Its search used to end "unverified"
    # after the whole ladder; it now finds the runaway (#584).
    model, messages = _fit_best(WEIBULL_50, include=["ExpoWeibull", "Weibull"])
    assert model.dist.name == "Weibull"
    assert [m for m in messages if "ExpoWeibull (its likelihood has no" in m]
    assert not [m for m in messages if m.startswith("No finite maximum")]


def test_set_aside_candidates_are_ranked_when_nothing_else_fits():
    model, messages = _fit_best(SEVEN, include=["Uniform"])
    assert model.dist.name == "Uniform"
    assert [m for m in messages if "Chosen: Uniform" in m]


def test_an_ordinary_call_warns_nothing_extra():
    x = sp.Weibull.random(40, 10, 2, random_state=3)
    model, messages = _fit_best(x, include=["Weibull", "Gamma", "LogNormal"])
    assert messages == []
    assert model.dist.name in ("Weibull", "Gamma", "LogNormal")


# ---------------------------------------------------------------------------
# ``fit_best`` checks the distribution names.
# ---------------------------------------------------------------------------


def test_fit_best_checks_distribution_names():
    x = [1.0, 2, 3, 4, 5, 6]
    assert surv.fit_best(x, include=["weibull"]).dist.name == "Weibull"
    assert surv.fit_best(x, include="Weibull").dist.name == "Weibull"
    with pytest.raises(ValueError, match="Unknown distribution"):
        surv.fit_best(x, include=["Weibul"])
    with pytest.raises(ValueError, match="Unknown distribution"):
        surv.fit_best(x, exclude=["Geometric"])


# -- the data, as fit takes them (#570) ------------------------------------
def test_570_an_input_error_raises_as_fit_raises_it():
    # A right-censored row written as [xl, inf] with c=1: Weibull.fit
    # raises; fit_best returned None, warning the data eleven times.
    x = np.array([[1.0, np.inf], [2.0, 3.0], [4.0, 4.0]])
    c = np.array([1, 2, 0])
    with pytest.raises(ValueError) as single:
        sp.Weibull.fit(x=x, c=c)
    with pytest.raises(ValueError) as best:
        sp.fit_best(x=x, c=c)
    assert str(best.value) == str(single.value)


def test_570_a_failure_every_candidate_shares_is_raised():
    # No failure at all: every family refuses alike, so it is the data
    x = WEIBULL_50
    with pytest.raises(ValueError, match="only right censored") as error:
        sp.fit_best(x, c=np.ones_like(x))
    with pytest.raises(ValueError) as single:
        sp.Weibull.fit(x, c=np.ones_like(x))
    assert str(error.value) == str(single.value)


def test_570_data_outside_every_candidate_support_raises():
    with pytest.raises(ValueError, match="outside the support of every"):
        sp.fit_best(-WEIBULL_50, include=["Weibull", "Gamma"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tl": 20.0},
        {"tl": np.where(np.arange(50) % 2 == 0, 20.0, 0.0)},
        {"tr": 400.0},
    ],
    ids=["tl-scalar", "tl-array", "tr"],
)
def test_570_truncation_is_passed_through(kwargs):
    x = WEIBULL_50[WEIBULL_50 > 20]
    kwargs = {
        k: v[WEIBULL_50 > 20] if np.ndim(v) else v for k, v in kwargs.items()
    }
    model = sp.fit_best(x, include=["Weibull"], **kwargs)
    np.testing.assert_array_equal(
        model.params, sp.Weibull.fit(x, **kwargs).params
    )


def test_570_interval_ends_are_passed_through():
    xl, xr = np.floor(WEIBULL_50 / 10) * 10, np.ceil(WEIBULL_50 / 10) * 10
    model = sp.fit_best(xl=xl, xr=xr, include=["Weibull", "Gamma"])
    single = getattr(sp, model.dist.name).fit(xl=xl, xr=xr)
    np.testing.assert_array_equal(model.params, single.params)


def test_570_a_skipped_candidate_is_named_with_a_short_reason(monkeypatch):
    # A message that quotes the data (as the censoring check's does) is
    # cut to its first line; the warning names the family and the reason.
    def refuse(*args, **kwargs):
        raise ValueError("Gamma cannot be fitted here.\nx:\n" + "1.0 " * 500)

    monkeypatch.setattr(sp.Gamma, "fit", refuse)
    model, messages = _fit_best(WEIBULL_50, include=["Weibull", "Gamma"])
    assert model.dist.name == "Weibull"
    (skipped,) = [m for m in messages if m.startswith("fit_best skipped")]
    assert "Gamma (ValueError: Gamma cannot be fitted here.)" in skipped
    assert "\n" not in skipped and len(skipped) < 200
