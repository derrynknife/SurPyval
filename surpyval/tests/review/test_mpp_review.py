"""Targeted review of ``parametric/fitters/mpp.py`` (#399).

Each test pins a bug found by reading the module and its entry points
(``fit(how='MPP')``, ``fit_from_ecdf``, ``fit_from_non_parametric``)
adversarially (fixed under #438; they were strict expected failures
until then).
"""

import numpy as np
import pytest

import surpyval as surv


def test_fit_from_non_parametric_matches_the_documented_mpp_fit():
    # The KM's censored times were plotted too: alpha 10.711 where the
    # documented equivalent gives 10.597.
    np.random.seed(3)
    x = np.round(surv.Weibull.random(30, 10, 2), 1)
    c = (np.random.rand(30) < 0.3).astype(int)
    km = surv.KaplanMeier.fit(x, c)
    from_km = surv.Weibull.fit_from_non_parametric(km).params
    mpp = surv.Weibull.fit(x, c, how="MPP", heuristic="Kaplan-Meier").params
    np.testing.assert_allclose(from_km, mpp, rtol=1e-10)
    np.testing.assert_allclose(from_km[0], 10.597, atol=1e-3)
    na = surv.NelsonAalen.fit(x, c)
    np.testing.assert_allclose(
        surv.Weibull.fit_from_non_parametric(na).params,
        surv.Weibull.fit(x, c, how="MPP", heuristic="Nelson-Aalen").params,
        rtol=1e-10,
    )


@pytest.mark.parametrize("bad", [1.2, -0.1, np.nan])
def test_fit_from_ecdf_refuses_an_invalid_probability(bad):
    # It was dropped silently: F = [0.1, 0.3, 1.2, 0.9] fitted alpha 2.886
    with pytest.raises(ValueError, match=r"F must lie in \[0, 1\]"):
        surv.Weibull.fit_from_ecdf([1, 2, 3, 4], [0.1, 0.3, bad, 0.9])


def test_fit_from_ecdf_refuses_mismatched_lengths():
    # an IndexError before
    with pytest.raises(ValueError, match="x has 4 values and F has 3"):
        surv.Weibull.fit_from_ecdf([1, 2, 3, 4], [0.1, 0.3, 0.6])
