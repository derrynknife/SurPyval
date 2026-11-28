"""Bias of the three bivariate survival estimators, and of Kendall's tau
from them, each under its assumptions.

The design, fixed before the first run: pairs from a Clayton copula
(theta 2, Kendall's tau 0.5) with Weibull(10, 2) and Weibull(20, 3)
margins, as ``test_coverage_dabrowska.py``. Two kinds of censoring:

- common: one follow-up time per unit, uniform on 0 to 45, censoring both
  lifetimes (all three estimators apply);
- independent: each lifetime its own uniform time, on 0 to 33 and 0 to 60
  (Dabrowska and Prentice-Cai apply; Lin-Ying must refuse every data set,
  and its formula forced on the data must be biased).

1,000 data sets of 200 and of 400 pairs each (seeds 7652 and 7653 plus the
sample size). The joint survival at (8, 18) and (4, 10), against the
copula's closed form; Kendall's tau given both fail by (12, 25), against
the sample tau of 200,000 draws inside that rectangle. A mean error
passes within four Monte Carlo standard errors (plus 0.005 for tau, the
truth's own simulation error). The reduced version (100 data sets of 200)
is in ``tests/multivariate/test_bivariate_estimators.py``.
"""

import numpy as np
import pytest

from surpyval.tests.multivariate.test_bivariate_estimators import (
    ESTIMATORS,
    _region_tau,
    _simulate,
)


@pytest.mark.parametrize("N", [200, 400])
def test_bias_common_censoring(N):
    out, refused = _simulate("common", N, 1000, 7652 + N)
    assert refused == 0
    tau = _region_tau()
    for e in ESTIMATORS:
        bias, mcse = out[e]
        assert np.all(np.abs(bias) < 4 * mcse), (e, bias, mcse)
        mean, mcse = out[f"tau {e}"]
        assert abs(mean - tau) < 4 * mcse + 0.005, (e, mean, tau, mcse)


@pytest.mark.parametrize("N", [200, 400])
def test_bias_independent_censoring(N):
    out, refused = _simulate("independent", N, 1000, 7653 + N)
    assert refused == 1000
    tau = _region_tau()
    for e in ("dabrowska", "prentice-cai"):
        bias, mcse = out[e]
        assert np.all(np.abs(bias) < 4 * mcse), (e, bias, mcse)
        mean, mcse = out[f"tau {e}"]
        assert abs(mean - tau) < 4 * mcse + 0.005, (e, mean, tau, mcse)
    bias, mcse = out["lin-ying (forced)"]
    assert np.any(np.abs(bias) > 4 * mcse), (bias, mcse)
