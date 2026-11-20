"""Recovery and coverage of the joint model with a weight on its
accumulated exposure: the hazard follows the area under the marker's
line with older exposure counting for less.

The truth is ``tests/univariate/joint/_fleet.simulate_links``' fleet:
200 items with ages in years, each marker a line from 3.0 (sd 0.6)
rising 1.0 a year (sd 0.5, correlation 0.2), read monthly with noise sd
0.7, followed for two years; the hazard Weibull (alpha 3.0 at an area of
0, beta 2.0) with 0.9 per unit-year of the weighted area under the line:
exposure that fades over half a year (``("exponential", 0.5)``) in one
study, and only the last half year's (``("window", 0.5)``) in the
other. Each replicate is fitted by ``JointPH(Weibull,
link="cumulative", weight=...)`` with the true weight, 200 replicates
a weight.

- **Parameters:** each parameter's two-sided 95% interval from
  ``summary()`` must cover the truth 95% of the time; the estimates'
  standardised bias and the ratio of the model's standard errors to their
  spread are within ``check_bias``'s rule.
- **The weight** (reported, not asserted): how often the fit with the
  true weight has a lower AIC than the fit with none (all the exposure,
  equally).

The seeds (8900 + r for the exponential weight, 9100 + r for the window)
and the rules were fixed before the first run. Slack: 0.02 for the
coverages (as the joint model's other coverage studies), the default
0.2 sd for the bias and 0.1 for the standard errors.

The study is 800 fits, about four core-hours (a fit with the exponential
weight takes about a minute). It had not been run when it was written:
the first run's numbers go here, as they are.
"""

import warnings

import numpy as np
import pytest

from surpyval import JointPH, Weibull
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
)
from surpyval.tests.univariate.joint._fleet import simulate_links

UNITS = 200
REPS = 200
WEIGHTS = {
    "exponential": (("exponential", 0.5), 8900),
    "window": (("window", 0.5), 9100),
}
TRUTH = dict(
    a=3.0,
    b=1.0,
    sd_a=0.6,
    sd_b=0.5,
    corr_a_b=0.2,
    sd_noise=0.7,
    alpha=3.0,
    beta=2.0,
    cumulative_effect=0.9,
)


def replicate(seed, weight):
    data = simulate_links(
        UNITS, seed=seed, alpha=3.0, beta=2.0, cumulative=0.9, weight=weight
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = JointPH(Weibull, link="cumulative", weight=weight).fit(data)
        plain = JointPH(Weibull, link="cumulative").fit(data)
        table = model.summary()
    return {
        "names": model.parameter_names,
        "estimates": np.asarray(model.params, dtype=float),
        "se": model.standard_errors(),
        "lower": table["coef lower 95%"].to_numpy(),
        "upper": table["coef upper 95%"].to_numpy(),
        "maximum": model.maximum,
        "weight_wins": model.aic() < plain.aic(),
    }


@pytest.mark.parametrize("name", sorted(WEIGHTS))
def test_joint_model_with_a_weight_recovers_and_covers(name):
    weight, first = WEIGHTS[name]
    runs = [replicate(first + r, weight) for r in range(REPS)]
    names = runs[0]["names"]
    verified = np.mean([r["maximum"] == "verified" for r in runs])
    wins = np.mean([r["weight_wins"] for r in runs])
    print(f"joint ({name} weight) verified maxima: {verified:.3f}")
    print(f"joint ({name} weight) the true weight's AIC lower: {wins:.3f}")
    assert verified >= 0.98
    target = np.array([TRUTH[n] for n in names])
    check_bias(
        np.array([r["estimates"] for r in runs]),
        target,
        f"joint ({name} weight) parameters",
        standard_errors=np.array([r["se"] for r in runs]),
    )
    check_coverage(
        np.array([r["lower"] for r in runs]),
        np.array([r["upper"] for r in runs]),
        target,
        0.95,
        f"joint ({name} weight) 95% intervals",
        slack=0.02,
    )
