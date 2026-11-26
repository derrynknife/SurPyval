"""Coverage of the bounds on the chance of each failure mode for a
running item (``predict(...).cb(x, on="cif", mode=)``).

The truth is ``test_coverage_joint_causes``'s fleet (``_fleet.
simulate_causes``: 300 pumps a replicate, a bearing failure the vibration
warns of, 0.9 per mm/s, and a seal failure it says nothing about),
fitted by ``WeibullJointPH`` with the failures' modes. Each fit predicts
one pump read four times (3.1, 3.4, 3.6 and 3.9 mm/s at 0, 120, 240 and
360 days) and still running at 400 days: its chance of failing by each
mode by 600 and by 900 days, with two-sided 95% bounds. The truth is the
same chance at the true parameters (the model's own computation, which
``test_joint_cif`` checks against a brute-force integral to 1.4e-5).

The bounds must cover the truth 95% of the time at each of the four
(mode, age) points, with the joint studies' slack of 0.02 (Wald bounds
by the delta method are asymptotic). The design, the seeds (45900 + r),
the query and the rule were fixed before the first fit.
"""

import warnings

import numpy as np

from surpyval import SurpyvalData, WeibullJointPH
from surpyval.tests.calibration._montecarlo import (
    check_coverage,
    process_pool,
)
from surpyval.tests.calibration.test_coverage_joint_causes import truth
from surpyval.tests.univariate.joint._fleet import CENTRE, simulate_causes
from surpyval.univariate.joint import JointModel

UNITS = 300
REPS = 200
FIRST = 45900
ITEM = {
    "markers": {0.0: 3.1, 120.0: 3.4, 240.0: 3.6, 360.0: 3.9},
    "given": 400.0,
}
AGES = np.array([600.0, 900.0])
MODES = ("bearing", "seal")


def fitted(seed):
    items, _ = simulate_causes(UNITS, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WeibullJointPH.fit(
            SurpyvalData.from_items(items), marker_reference=CENTRE
        )


def at_truth(model):
    """``model`` with its parameters at the truth."""
    d = model.to_dict()
    d["params"] = truth(model.parameter_names).tolist()
    d.pop("covariance", None)
    return JointModel.from_dict(d)


def replicate(seed):
    model = fitted(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = model.predict(**ITEM)
        band = np.concatenate(
            [pred.cb(AGES, on="cif", mode=mode) for mode in MODES]
        )
        true = at_truth(model).predict(**ITEM)
        target = np.concatenate([true.cif(AGES, mode) for mode in MODES])
    return {
        "lower": band[:, 0],
        "upper": band[:, 1],
        "truth": target,
        "maximum": model.maximum,
    }


def check(runs):
    label = "joint cif"
    target = runs[0]["truth"]
    # the truth's chance is the same in every replicate
    assert np.allclose([r["truth"] for r in runs], target, rtol=1e-12)
    verified = np.mean([r["maximum"] == "verified" for r in runs])
    lower = np.array([r["lower"] for r in runs])
    upper = np.array([r["upper"] for r in runs])
    inside = np.mean((lower <= target) & (target <= upper), axis=0)
    print(f"{label} verified maxima: {verified:.3f}")
    print(f"{label} truth (bearing 600, 900; seal 600, 900): {target}")
    print(f"{label} coverage: {inside}")
    assert verified >= 0.98
    check_coverage(
        lower, upper, target, 0.95, label + " 95% bounds", slack=0.02
    )


def test_joint_cif_bounds_coverage():
    with process_pool(2) as pool:
        runs = list(pool.map(replicate, range(FIRST, FIRST + REPS)))
    check(runs)
