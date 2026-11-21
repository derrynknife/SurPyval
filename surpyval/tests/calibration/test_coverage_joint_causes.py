"""Recovery and coverage of the joint model with competing failure
modes.

The truth is ``tests/univariate/joint/_fleet.simulate_causes``'s fleet:
pumps whose vibration (mm/s) is a straight line with its own start (3.0,
sd 0.6) and rate (1.0 a year, sd 0.5, correlation 0.2), read monthly with
noise sd 0.7, failing by the first of two modes: a bearing failure whose
Weibull PH hazard (alpha 2020 days at 4.0 mm/s, beta 2) follows the
line's current value with 0.9 per mm/s, and a seal failure (alpha 2900
days, beta 1.3) that the vibration says nothing about (an effect of 0).
Each replicate is 300 pumps followed for two years, about 60 of them
failing by the bearing and 40 by the seal, fitted by ``WeibullJointPH``
with the failures' modes (``event_type``) and ``alpha`` reported at the
truth's 4.0 mm/s.

- **Parameters:** each parameter's two-sided 95% interval from
  ``summary()`` must cover the truth 95% of the time; the estimates'
  standardised bias and the ratio of the model's standard errors to
  their spread are within ``check_bias``'s rule. The two effects are
  the point: 0.9 for the bearing and 0 for the seal.

The design (the failure counts, from fleets of other seeds), the seeds
(45600 + r) and the rules were fixed before the first fit. Slack: 0.02
for the coverages, as the joint model's other studies (the Wald
intervals are asymptotic, the variance components' slowest to get
there), the default 0.2 sd for the bias and 0.1 for the standard errors.
"""

import warnings

import numpy as np

from surpyval import SurpyvalData, WeibullJointPH
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
    process_pool,
)
from surpyval.tests.univariate.joint._fleet import (
    CAUSES,
    CENTRE,
    TRUTH,
    simulate_causes,
)

UNITS = 300
REPS = 200
FIRST = 45600


def truth(names):
    out = dict(TRUTH)
    for cause, values in CAUSES.items():
        out.update({f"{cause}: {k}": v for k, v in values.items()})
    return np.array([out[n] for n in names])


def fitted(seed):
    """The fit of the fleet of ``seed``."""
    items, _ = simulate_causes(UNITS, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WeibullJointPH.fit(
            SurpyvalData.from_items(items), marker_reference=CENTRE
        )


def measured(model):
    """What the study reads of a fit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = model.summary()
    return {
        "names": model.parameter_names,
        "estimates": np.asarray(model.params, dtype=float),
        "se": model.standard_errors(),
        "lower": table["coef lower 95%"].to_numpy(),
        "upper": table["coef upper 95%"].to_numpy(),
        "maximum": model.maximum,
        "failures": dict(model.n_events_by_cause),
    }


def replicate(seed):
    return measured(fitted(seed))


def check(runs):
    names = runs[0]["names"]
    label = "joint causes"
    verified = np.mean([r["maximum"] == "verified" for r in runs])
    failures = {
        cause: np.mean([r["failures"][cause] for r in runs])
        for cause in CAUSES
    }
    print(f"{label} verified maxima: {verified:.3f}")
    print(f"{label} mean failures by mode: {failures}")
    print(f"{label} parameters: {names}")
    estimates = np.array([r["estimates"] for r in runs])
    print(f"{label} mean estimates: {estimates.mean(axis=0)}")
    print(f"{label} truth: {truth(names)}")
    assert verified >= 0.98
    target = truth(names)
    check_bias(
        estimates,
        target,
        label + " parameters",
        standard_errors=np.array([r["se"] for r in runs]),
    )
    check_coverage(
        np.array([r["lower"] for r in runs]),
        np.array([r["upper"] for r in runs]),
        target,
        0.95,
        label + " 95% intervals",
        slack=0.02,
    )


def test_joint_causes_recovery_and_coverage():
    with process_pool(2) as pool:
        runs = list(pool.map(replicate, range(FIRST, FIRST + REPS)))
    check(runs)
