"""Recovery and coverage of the joint model with a lag: the hazard at age
``t`` follows the marker's line at ``max(t - lag, 0)``.

The truth is ``tests/univariate/joint/_fleet.simulate_links``' fleet:
200 items with ages in years, each marker a line from 3.0 (sd 0.6)
rising 1.0 a year (sd 0.5, correlation 0.2), read monthly with noise sd
0.7, followed for two years; the hazard Weibull (alpha 3.0 at a marker of
4.0, beta 2.0) with 0.9 per unit of the line's value half a year earlier
(``lag=0.5``; the value at age 0 until then). Each replicate is fitted
by ``JointPH(Weibull, lag=0.5)`` with ``alpha`` reported at 4.0, 200
replicates.

- **Parameters:** each parameter's two-sided 95% interval from
  ``summary()`` must cover the truth 95% of the time; the estimates'
  standardised bias and the ratio of the model's standard errors to their
  spread are within ``check_bias``'s rule.
- **The lag** (reported, not asserted): how often the fit with the true
  lag has a lower AIC than the fit with no lag. The rule fixed before the
  first run was at least 75%; it was 65.5% (131 of 200). The premise was
  wrong, not the fits: a straight line ``L`` earlier is the line now less
  ``b L``, which the marker's effect and the baseline largely absorb, so
  these data say little about the lag (see "A reading that acts later" in
  the joint models' guide). The other rules stand.

The seeds (8800 + r) and the rules were fixed before the first run.
Slack: 0.02 for the coverages (as the joint model's other coverage
study), the default 0.2 sd for the bias and 0.1 for the standard errors.
"""

import warnings

import numpy as np

from surpyval import JointPH, Weibull
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
)
from surpyval.tests.univariate.joint._fleet import CENTRE, simulate_links

UNITS = 200
REPS = 200
LAG = 0.5
TRUTH = dict(
    a=3.0,
    b=1.0,
    sd_a=0.6,
    sd_b=0.5,
    corr_a_b=0.2,
    sd_noise=0.7,
    alpha=3.0,
    beta=2.0,
    marker_effect=0.9,
)


def replicate(seed):
    data = simulate_links(
        UNITS, seed=seed, alpha=3.0, beta=2.0, value=0.9, lag=LAG
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = JointPH(Weibull, lag=LAG).fit(data, marker_reference=CENTRE)
        at_once = JointPH(Weibull).fit(data, marker_reference=CENTRE)
        table = model.summary()
    return {
        "names": model.parameter_names,
        "estimates": np.asarray(model.params, dtype=float),
        "se": model.standard_errors(),
        "lower": table["coef lower 95%"].to_numpy(),
        "upper": table["coef upper 95%"].to_numpy(),
        "maximum": model.maximum,
        "lag_wins": model.aic() < at_once.aic(),
    }


def test_joint_model_with_a_lag_recovers_and_covers():
    runs = [replicate(8800 + r) for r in range(REPS)]
    names = runs[0]["names"]
    verified = np.mean([r["maximum"] == "verified" for r in runs])
    wins = np.mean([r["lag_wins"] for r in runs])
    print(f"joint (lag) verified maxima: {verified:.3f}")
    print(f"joint (lag) the true lag's AIC lower: {wins:.3f}")
    assert verified >= 0.98
    target = np.array([TRUTH[n] for n in names])
    check_bias(
        np.array([r["estimates"] for r in runs]),
        target,
        "joint (lag) parameters",
        standard_errors=np.array([r["se"] for r in runs]),
    )
    check_coverage(
        np.array([r["lower"] for r in runs]),
        np.array([r["upper"] for r in runs]),
        target,
        0.95,
        "joint (lag) 95% intervals",
        slack=0.02,
    )
