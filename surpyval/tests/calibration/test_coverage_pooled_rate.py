"""Coverage of ``pooled_rate``'s intervals on simulated sources.

The truth: 6 sources whose true rates are drawn from a gamma population
with shape 1 and mean 0.005 (a spread of about 60 times between its 5th
and 95th percentiles), each with a fixed exposure, log-spaced from 200 to
20,000 (so 1 to 100 expected events at the mean rate); the counts are
Poisson. Each replicate draws new rates and counts, and one more rate
from the population, the "new source". The checks, at 90% (``alpha_ci=
0.1``):

- the predictive interval (``predictive``) covers the new source's rate;
- the interval of the average (``average``) covers the population mean,
  0.005.

Under ``method="Bayes"`` the weak prior is not the truth, so its coverage
for one fixed population is near, not exactly, nominal; the check is two
sided (3 Monte Carlo standard errors + 0.03, as the intervals are
approximate for 6 sources). ``method="MLE"`` is reported (printed), not
checked: with 6 sources a plug-in or bootstrap population is known to be
too narrow, which is why ``"Bayes"`` is the default. 300 replicates
(MLE: 100, with 100 bootstrap refits each); the seed, sizes and
tolerance were fixed before any run.

Result on 15 Nov 2026: Bayes predictive 0.880, average 0.913 (300
replicates); MLE predictive 0.720, average 0.850 (100 replicates).
"""

from typing import Literal

import numpy as np
import pytest

from surpyval import pooled_rate
from surpyval.tests.calibration._montecarlo import check_rate

SEED = 20261115
SHAPE, MEAN = 1.0, 0.005
T = np.geomspace(200.0, 20000.0, 6)
NOMINAL = 0.9


def _replicates(
    method: Literal["Bayes", "MLE"], reps: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    pred, avg = [], []
    for r in range(reps):
        lam = rng.gamma(SHAPE, MEAN / SHAPE, size=T.size)
        x = rng.poisson(lam * T)
        new = rng.gamma(SHAPE, MEAN / SHAPE)
        if x.sum() == 0:
            continue
        with np.errstate(all="ignore"):
            res = pooled_rate(
                x,
                T,
                method=method,
                alpha_ci=0.1,
                n_boot=100,
                random_state=r,
            )
        p, a = res.predictive, res.average
        assert p is not None and a is not None
        pred.append(p["lower"] <= new <= p["upper"])
        avg.append(a["lower"] <= MEAN <= a["upper"])
    return np.array(pred), np.array(avg)


@pytest.mark.calibration
def test_pooled_rate_bayes_coverage() -> None:
    pred, avg = _replicates("Bayes", 300)
    check_rate(pred.sum(), pred.size, NOMINAL, "Bayes predictive", slack=0.03)
    check_rate(avg.sum(), avg.size, NOMINAL, "Bayes average", slack=0.03)


@pytest.mark.calibration
def test_pooled_rate_mle_coverage_reported() -> None:
    pred, avg = _replicates("MLE", 100)
    print(
        "MLE predictive coverage {:.3f}, average coverage {:.3f} "
        "(nominal 0.9, {} replicates; reported, not checked)".format(
            pred.mean(), avg.mean(), pred.size
        )
    )
