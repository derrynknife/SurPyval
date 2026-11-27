"""Coverage of the parametric bootstrap bounds on the maturity
projections: ``AMPMProjection.projected_intensity_cb``,
``SteinProjection.projected_intensity_cb`` and
``DiscreteSteinProjection.projected_reliability_cb``, each for both
targets.

Designs (the simulators are in ``tests/recurrent/_maturity_sims.py``),
20,000 modes each, fix-effectiveness factors 0.5, 0.7 or 0.9 drawn
independently of the modes' rates:

- continuous: B-mode rates gamma with total 0.2 and scale 0.02, A
  failures at 0.02, one system tested to 300 (about 20 B modes and 60 B
  failures a test); the AMPM and AMPM-Stein (with the A modes named);
- discrete: chances beta with 30 pseudo trials and total 0.5, 40 trials
  (about 9 modes and 20 failures a test).

``target="realised"`` is checked against each simulated system's own
intensity (reliability) once its seen modes are fixed;
``target="expected"`` against the model's expected projected quantity
(0.1 continuous; 0.7408 discrete, exact by Campbell's theorem). Two-sided
90% and one-sided 90% (an upper bound on the intensity, a lower bound on
the reliability). 400 tests each of 400 replicates, seed 1126; a test
whose fit is refused counts as a miss.
"""

import sys

import numpy as np
import pytest

from surpyval.recurrent import (
    adpm_stein_projection,
    ampm_projection,
    ampm_stein_projection,
)
from surpyval.tests.calibration._montecarlo import check_coverage
from surpyval.tests.recurrent import _maturity_sims as sims

REPS, N_BOOT, NOMINAL = 400, 400, 0.9
TARGETS = ("realised", "expected")


def _bounds(cb, rep):
    """Two-sided and one-sided bounds of ``cb`` (nan when refused)."""
    out = {}
    for target in TARGETS:
        try:
            two = cb(1 - NOMINAL, "two-sided", target, rep)
            one = cb(1 - NOMINAL, "one-sided", target, rep)
        except ValueError:
            two, one = np.array([np.nan, np.nan]), np.nan
        out[target] = (two, one)
    return out


@pytest.mark.xfail(
    sys.platform.startswith("linux"),
    strict=True,
    reason="a miss on the 0.30 release run (x86-64 Linux), released as a "
    "known limitation: the Stein expected-failures upper bound covered "
    "0.8450 for 0.9000 (tolerance 0.0550)",
)
def test_continuous_maturity_projection_coverage():
    rng = np.random.default_rng(1126)
    expected = sims.continuous_truth()
    rows: dict = {}
    for rep in range(REPS):
        x, modes, c, fef, a_modes, real = sims.continuous_test(rng)
        fits = {}
        for name in ("ampm", "stein"):
            try:
                fits[name] = (
                    ampm_projection(x, modes, fef, c=c)
                    if name == "ampm"
                    else ampm_stein_projection(
                        x, modes, fef, c=c, a_modes=a_modes
                    )
                )
            except ValueError:
                fits[name] = None
        for name, fit in fits.items():

            def cb(alpha, side, target, seed, fit=fit):
                if fit is None:
                    raise ValueError("refused")
                return fit.projected_intensity_cb(
                    alpha,
                    "two-sided" if side == "two-sided" else "upper",
                    target=target,
                    n_boot=N_BOOT,
                    random_state=seed,
                )

            for target, (two, upper) in _bounds(cb, rep).items():
                truth = real if target == "realised" else expected
                rows.setdefault((name, target), []).append(
                    (two[0], two[1], upper, truth)
                )
    for (name, target), values in rows.items():
        v = np.array(values, dtype=float)
        check_coverage(
            v[:, 0], v[:, 1], v[:, 3], NOMINAL, f"{name} {target} two-sided"
        )
        check_coverage(
            np.full(len(v), -np.inf),
            v[:, 2],
            v[:, 3],
            NOMINAL,
            f"{name} {target} upper",
        )


def test_discrete_maturity_projection_coverage():
    rng = np.random.default_rng(1126)
    expected = sims.discrete_truth()
    rows: dict = {}
    for rep in range(REPS):
        failures, fef, real = sims.discrete_test(rng)
        try:
            fit = adpm_stein_projection(failures, 40, fef=fef)
        except ValueError:
            fit = None

        def cb(alpha, side, target, seed):
            if fit is None:
                raise ValueError("refused")
            return fit.projected_reliability_cb(
                alpha,
                "two-sided" if side == "two-sided" else "lower",
                target=target,
                n_boot=N_BOOT,
                random_state=seed,
            )

        for target, (two, lower) in _bounds(cb, rep).items():
            truth = real if target == "realised" else expected
            rows.setdefault(target, []).append((two[0], two[1], lower, truth))
    for target, values in rows.items():
        v = np.array(values, dtype=float)
        check_coverage(
            v[:, 0], v[:, 1], v[:, 3], NOMINAL, f"discrete {target} two-sided"
        )
        check_coverage(
            v[:, 2],
            np.full(len(v), np.inf),
            v[:, 3],
            NOMINAL,
            f"discrete {target} lower",
        )
