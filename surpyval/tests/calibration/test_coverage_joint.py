"""Coverage of the joint model's intervals and dynamic-prediction bands
(#891).

The truth is ``tests/univariate/joint/_fleet.simulate_split``'s fleet:
pumps whose vibration (mm/s) is a straight line with its own start (3.0,
sd 0.6) and rate (1.0 a year, sd 0.5, correlation 0.2), read monthly with
noise sd 0.7, and a Weibull PH hazard (alpha 800 days at 4.0 mm/s, beta
2) with 0.9 per mm/s of the line's current value; two 0/1 covariates,
each for half the fleet: ``site`` in the hazard only (0.5) and ``load``
in the line's rate only (0.6 mm/s a year more). Each replicate is 200
pumps followed for two years, fitted by ``WeibullJointPH`` with each
covariate in its own part (``hazard_covariates=["site"]``,
``path_covariates={"b": ["load"]}``) and ``alpha`` reported at the
truth's 4.0 mm/s.

- **Exact failure times** and **inspections**: the same fleet, with the
  failures known exactly, and known only to have happened between two
  inspections 90 days apart (interval censored; left censored before the
  first), 200 replicates each.
- **Parameters:** each parameter's two-sided 95% interval from
  ``summary()`` (Wald; on the log scale for the sds, ``alpha`` and
  ``beta``, Fisher's scale for the correlation) must cover the truth 95%
  of the time; the estimates' standardised bias and the ratio of the
  model's standard errors to their spread are within ``check_bias``'s
  rule.
- **Dynamic prediction:** a pump outside the fleet (both covariates 1,
  four readings rising from 3.2 to 4.5 mm/s by 180 days, still running
  then) is predicted from each fit; the 95% band of its conditional
  survival (``predict(...).cb``) must cover the truth's own prediction
  (the same calculation with the true parameters) 60, 180 and 365 days
  on.

The seeds (7300 + r and 7600 + r) and the rules were fixed before the
first run. Slack: 0.02 for the coverages (the Wald intervals are
asymptotic, the variance components' slowest to get there), the default
0.2 sd for the bias and 0.1 for the standard errors.
"""

import warnings

import numpy as np
import pytest

from surpyval import JointModel, SurpyvalData, WeibullJointPH
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
)
from surpyval.tests.univariate.joint._fleet import (
    CENTRE,
    TRUTH,
    simulate_split,
)

GAMMA, DELTA_B = 0.5, 0.6 / 365
PARTS = dict(hazard_covariates=["site"], path_covariates={"b": ["load"]})
UNITS = 200
REPS = 200
#: The pump outside the fleet: its covariates, readings and age now, and
#: the ages to predict at.
HELD_OUT = dict(
    Z={"site": 1.0, "load": 1.0},
    markers={0.0: 3.2, 60.0: 3.6, 120.0: 4.1, 180.0: 4.5},
    given=180.0,
)
AGES = 180.0 + np.array([60.0, 180.0, 365.0])


def _truth(names):
    out = dict(TRUTH, site=GAMMA, b_load=DELTA_B)
    return np.array([out[n] for n in names])


def _with_params(model, params):
    d = model.to_dict()
    d["params"] = list(map(float, params))
    d.pop("covariance", None)
    return JointModel.from_dict(d)


def replicate(seed, inspect):
    items, _, _ = simulate_split(UNITS, seed=seed, inspect=inspect)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = WeibullJointPH.fit(
            SurpyvalData.from_items(items), marker_reference=CENTRE, **PARTS
        )
        table = model.summary()
        band = model.predict(**HELD_OUT).cb(AGES)
        truth = _with_params(model, _truth(model.parameter_names))
        true_sf = truth.predict(**HELD_OUT).sf(AGES)
    return {
        "names": model.parameter_names,
        "estimates": np.asarray(model.params, dtype=float),
        "se": model.standard_errors(),
        "lower": table["coef lower 95%"].to_numpy(),
        "upper": table["coef upper 95%"].to_numpy(),
        "band": band,
        "true_sf": true_sf,
        "maximum": model.maximum,
    }


@pytest.mark.parametrize(
    "inspect, first",
    [(None, 7300), (90.0, 7600)],
    ids=["exact", "inspections"],
)
def test_joint_model_coverage(inspect, first):
    runs = [replicate(first + r, inspect) for r in range(REPS)]
    names = runs[0]["names"]
    label = "joint ({})".format("exact" if inspect is None else "inspected")
    verified = np.mean([r["maximum"] == "verified" for r in runs])
    print(f"{label} verified maxima: {verified:.3f}")
    print(f"{label} parameters: {names}")
    assert verified >= 0.98
    target = _truth(names)
    check_bias(
        np.array([r["estimates"] for r in runs]),
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
    band = np.array([r["band"] for r in runs])
    true_sf = np.array([r["true_sf"] for r in runs])
    hits = (band[..., 0] <= true_sf) & (true_sf <= band[..., 1])
    # check_coverage with each replicate's own truth: hits as bounds
    check_coverage(
        np.where(hits, 0.0, 2.0),
        np.ones(hits.shape),
        1.0,
        0.95,
        label + " dynamic prediction cb(sf)",
        slack=0.02,
    )
