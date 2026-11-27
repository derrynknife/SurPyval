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
``test_joint_cif`` checks against a brute-force integral to
1.3e-5, the time rule over 1100 days).

The bounds must cover the truth 95% of the time at each of the four
(mode, age) points, with the joint studies' slack of 0.02 (Wald bounds
by the delta method are asymptotic). The design, the seeds (45900 + r),
the query and the rule were fixed before the first fit.

A second study (seeds 46100 + r, fixed with it, before any run) is the
same with the seal's marker effect -0.5 in place of 0: the two modes'
effects of opposite signs, so a high reading raises one mode's chance
and lowers the other's.
"""

import warnings

import numpy as np

from surpyval import SurpyvalData, WeibullJointPH
from surpyval.tests.calibration._montecarlo import (
    check_coverage,
    process_pool,
)
from surpyval.tests.univariate.joint._fleet import (
    CAUSES,
    CENTRE,
    TRUTH,
    simulate_causes,
)
from surpyval.univariate.joint import JointModel

UNITS = 300
REPS = 200
FIRST = 45900
#: the second study: the seal's effect opposite to the bearing's
OPPOSITE = {
    "bearing": dict(CAUSES["bearing"]),
    "seal": dict(CAUSES["seal"], marker_effect=-0.5),
}
FIRST_OPPOSITE = 46100
ITEM = {
    "markers": {0.0: 3.1, 120.0: 3.4, 240.0: 3.6, 360.0: 3.9},
    "given": 400.0,
}
AGES = np.array([600.0, 900.0])
MODES = ("bearing", "seal")


def truth(names, causes=None):
    out = dict(TRUTH)
    for cause, values in (CAUSES if causes is None else causes).items():
        out.update({f"{cause}: {k}": v for k, v in values.items()})
    return np.array([out[n] for n in names])


def fitted(seed, causes=None):
    items, _ = simulate_causes(UNITS, seed=seed, causes=causes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WeibullJointPH.fit(
            SurpyvalData.from_items(items), marker_reference=CENTRE
        )


def at_truth(model, causes=None):
    """``model`` with its parameters at the truth."""
    d = model.to_dict()
    d["params"] = truth(model.parameter_names, causes).tolist()
    d.pop("covariance", None)
    return JointModel.from_dict(d)


def replicate(seed, causes=None):
    model = fitted(seed, causes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = model.predict(**ITEM)
        band = np.concatenate(
            [pred.cb(AGES, on="cif", mode=mode) for mode in MODES]
        )
        true = at_truth(model, causes).predict(**ITEM)
        target = np.concatenate([true.cif(AGES, mode) for mode in MODES])
    return {
        "lower": band[:, 0],
        "upper": band[:, 1],
        "truth": target,
        "maximum": model.maximum,
    }


def replicate_opposite(seed):
    return replicate(seed, OPPOSITE)


def check(runs, label="joint cif"):
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


def test_joint_cif_bounds_coverage_opposite_effects():
    with process_pool(2) as pool:
        runs = list(
            pool.map(
                replicate_opposite,
                range(FIRST_OPPOSITE, FIRST_OPPOSITE + REPS),
            )
        )
    check(runs, "joint cif, opposite effects")
