"""Recovery and coverage of the joint model with competing failure
modes when some failures' modes were not recorded.

The fleet is ``test_coverage_joint_causes``'s (300 pumps, a bearing mode
the vibration warns of with 0.9 per mm/s and a seal mode it says nothing
about), with each failure's mode hidden (``None`` in ``e``) at random
with probability 0.3, independently of the mode and of everything else
(missing at random), by a generator of its own (seed ``seed + 10**6``).
Each replicate is fitted by ``WeibullJointPH`` with those labels; the
fit takes a failure of unknown mode as one by either mode.

- **Parameters:** each parameter's two-sided 95% interval from
  ``summary()`` must cover the truth 95% of the time, and the estimates'
  standardised bias and the standard errors' ratio to their spread are
  within ``check_bias``'s rule: the same rules as the study with every
  mode known.

The design, the seeds (46100 + r, 200 of them), the share hidden and the
rules were fixed before the first fit; the slack is the causes study's.

**It fails its standard-error rule, and is a strict xfail.** Run in full
(27 Nov 2026): every fit verified, 31.5 failures a fleet with no mode,
standardised biases -0.17 to +0.24, but ``bearing: alpha``'s rms
standard error over its estimates' sd is 1.298 against the rule's
1 +- 0.25. One replicate drives it: seed 46291, a verified fit with
``bearing: alpha`` 21,817 (truth 2,020; standard error 25,846),
``beta`` 0.71 and effect 1.52. Its fleet has 43 bearing failures known
and 33 failures of either mode; the fit with every label known gives
4,806 (se 2,561), 1.39 and 1.35, and those values lose only 1.39 in
log-likelihood on the hidden-label data, rising steadily along the way
from one to the other. So it is a flat ridge of the likelihood with the
labels hidden (the bearing's shape traded against its scale and the
marker's effect), not a wrong number. On the log scale, where ``alpha``'s
interval is taken, the ratio is within the rule (0.95). The rule is kept
as it is for every study; the fit's scale is not well determined from
so few known failures of a mode, which the docs say.
"""

import warnings

import numpy as np
import pytest

from surpyval import SurpyvalData, WeibullJointPH
from surpyval.tests.calibration._montecarlo import process_pool
from surpyval.tests.calibration.test_coverage_joint_causes import (
    UNITS,
    check,
    measured,
)
from surpyval.tests.univariate.joint._fleet import CENTRE, simulate_causes

REPS = 200
FIRST = 46100
#: The chance that a failure's mode was not recorded.
HIDDEN = 0.3


def hidden_labels(seed):
    """``(data, hidden)``: the fleet of ``seed`` with each failure's mode
    hidden with probability :data:`HIDDEN`, and the number hidden."""
    items, _ = simulate_causes(UNITS, seed=seed)
    data = SurpyvalData.from_items(items)
    e = np.array(data.e, dtype=object)
    failed = np.flatnonzero(np.array([label is not None for label in e]))
    rng = np.random.default_rng(seed + 10**6)
    drop = failed[rng.random(failed.size) < HIDDEN]
    e[drop] = None
    readings = data.markers
    arrays = dict(
        x=np.asarray(data.x, dtype=float),
        c=np.asarray(data.c),
        n=np.asarray(data.n, dtype=float),
        markers=[dict(readings.get(i, {})) for i in data.i],
        e=e,
    )
    return arrays, float(drop.size)


def replicate(seed):
    arrays, hidden = hidden_labels(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = WeibullJointPH.fit(**arrays, marker_reference=CENTRE)
    return dict(measured(model), hidden=hidden)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "bearing: alpha's rms standard error over its spread is 1.298 "
        "against 1 +- 0.25, from one replicate on a flat ridge of the "
        "likelihood with the labels hidden (seed 46291); see the module "
        "docstring"
    ),
)
def test_joint_unknown_modes_recovery_and_coverage():
    with process_pool(2) as pool:
        runs = list(pool.map(replicate, range(FIRST, FIRST + REPS)))
    print(
        "joint unknown modes: mean failures with no mode "
        f"{np.mean([r['hidden'] for r in runs]):.1f}"
    )
    check(runs)
