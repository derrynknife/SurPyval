"""Targeted review of ``parametric/fitters/mpp.py`` (#399).

Each test pins a bug found by reading the module and its entry points
(``fit(how='MPP')``, ``fit_from_ecdf``, ``fit_from_non_parametric``)
adversarially. They are strict expected failures until the bug is fixed.
"""

import numpy as np
import pytest

import surpyval as surv


@pytest.mark.xfail(
    strict=True,
    reason="#438: fit_from_non_parametric keeps the KM's censored times, so "
    "on censored data it matches fit(how='MPP', heuristic='Kaplan-Meier', "
    "on_d_is_0=True) (alpha 10.711) and not the documented default fit "
    "(10.597)",
)
def test_fit_from_non_parametric_matches_the_documented_mpp_fit():
    np.random.seed(3)
    x = np.round(surv.Weibull.random(30, 10, 2), 1)
    c = (np.random.rand(30) < 0.3).astype(int)
    km = surv.KaplanMeier.fit(x, c)
    from_km = surv.Weibull.fit_from_non_parametric(km).params
    mpp = surv.Weibull.fit(x, c, how="MPP", heuristic="Kaplan-Meier").params
    np.testing.assert_allclose(from_km, mpp, rtol=1e-10)


@pytest.mark.xfail(
    strict=True,
    reason="#438: fit_from_ecdf silently drops an F outside [0, 1] or nan "
    "(F = [0.1, 0.3, 1.2, 0.9] fits alpha = 2.886) instead of raising",
)
@pytest.mark.parametrize("bad", [1.2, -0.1, np.nan])
def test_fit_from_ecdf_refuses_an_invalid_probability(bad):
    with pytest.raises(ValueError, match="F"):
        surv.Weibull.fit_from_ecdf([1, 2, 3, 4], [0.1, 0.3, bad, 0.9])


@pytest.mark.xfail(
    strict=True,
    reason="#438: fit_from_ecdf with x and F of different lengths raises "
    "IndexError, not a ValueError naming them",
)
def test_fit_from_ecdf_refuses_mismatched_lengths():
    with pytest.raises(ValueError):
        surv.Weibull.fit_from_ecdf([1, 2, 3, 4], [0.1, 0.3, 0.6])
