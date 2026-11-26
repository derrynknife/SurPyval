"""Recovery and coverage of the joint model of several markers with
competing failure modes.

The truth is ``tests/univariate/joint/_multi_fleet.simulate_causes``'s
fleet: items whose vibration and temperature are straight lines with a
start and rate of their own, the four correlated across the markers (the
starts 0.5), vibration read every 30 days (noise sd 0.5) and temperature
every 60 (sd 2.0), failing by the first of two modes: a bearing failure
whose Weibull PH hazard (alpha 1100 days, beta 2.0 at vibration 4 and
temperature 65) follows the vibration's current value with 0.8 a unit
and not the temperature's, and a seal failure (alpha 1400, beta 1.5)
that follows the temperature with 0.08 a unit and not the vibration.
Each replicate is 250 items followed for two years, about 95 failing
by the bearing and 75 by the seal, fitted by ``JointPH(Weibull,
{"vibration": LinearPath, "temperature": LinearPath})`` (every path
parameter varying, every correlation estimated, as the truth) with the
failures' modes and ``alpha`` reported at the truth's centres.

- **Parameters:** each of the 26 parameters' two-sided 95% intervals
  must cover the truth 95% of the time; standardised bias and the ratio
  of standard errors to their spread within ``check_bias``'s rule. The
  four effects (0.8 and 0 for the bearing, 0 and 0.08 for the seal) are
  the point.

The design, the seeds (46000 + r) and the rules were fixed before the
first fit. Slack: 0.02 for the coverages, as the joint model's other
studies, the default 0.2 sd for the bias and 0.1 for the standard
errors. A reduced run (``reduced_run``, 12 fleets of 200, seeds 46000 to
46011) was made on 26 Nov 2026 to check the study runs (every maximum
verified, about two CPU minutes a fit; numbers in ``docs/Joint
Models.rst``); the full study was written and committed unrun.
"""

import warnings

import numpy as np

from surpyval import JointPH, Weibull
from surpyval.degradation import LinearPath
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
    process_pool,
)
from surpyval.tests.univariate.joint._multi_fleet import (
    CAUSES,
    CENTRES,
    TRUTH,
    simulate_causes,
)

UNITS = 250
REPS = 200
FIRST = 46000

#: The correlations' names in the model, and their place in the truth's
#: matrix (vibration a, b, temperature a, b).
_ORDER = ["vibration.a", "vibration.b", "temperature.a", "temperature.b"]


def truth(names):
    out = dict(TRUTH)
    for i, one in enumerate(_ORDER):
        for j, two in enumerate(_ORDER):
            out[f"corr_{one}_{two}"] = TRUTH["corr"][i, j]
    for cause, values in CAUSES.items():
        out.update({f"{cause}: {k}": v for k, v in values.items()})
    return np.array([out[n] for n in names])


def fitted(seed, units=UNITS):
    """The fit of the fleet of ``seed``."""
    x, c, e, m, _ = simulate_causes(units, seed=seed)
    fitter = JointPH(
        Weibull,
        {"vibration": LinearPath, "temperature": LinearPath},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fitter.fit(
            x,
            c=c,
            e=np.array(e, dtype=object),
            markers=m,
            marker_reference=dict(zip(("vibration", "temperature"), CENTRES)),
        )


def measured(model):
    """What the study reads of a fit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = model.summary()
    return {
        "names": list(model.parameter_names),
        "estimates": np.asarray(model.params, dtype=float),
        "se": model.standard_errors(),
        "lower": table["coef lower 95%"].to_numpy(),
        "upper": table["coef upper 95%"].to_numpy(),
        "maximum": model.maximum,
        "failures": dict(model.n_events_by_cause),
    }


def replicate(seed, units=UNITS):
    return measured(fitted(seed, units))


def _reduced(seed):
    return replicate(seed, 200)


def summarise(runs, label="joint marker causes"):
    """Print what the runs found: verified share, failures by mode, and
    each parameter's mean estimate, relative bias and 95% coverage."""
    names = runs[0]["names"]
    target = truth(names)
    estimates = np.array([r["estimates"] for r in runs])
    lower = np.array([r["lower"] for r in runs])
    upper = np.array([r["upper"] for r in runs])
    verified = np.mean([r["maximum"] == "verified" for r in runs])
    failures = {
        cause: np.mean([r["failures"][cause] for r in runs])
        for cause in CAUSES
    }
    print(f"{label}: {len(runs)} fleets, verified maxima {verified:.3f}")
    print(f"{label} mean failures by mode: {failures}")
    covered = np.mean((lower <= target) & (target <= upper), axis=0)
    for name, t, m, s, cov in zip(
        names,
        target,
        estimates.mean(axis=0),
        estimates.std(axis=0, ddof=1) / np.sqrt(len(runs)),
        covered,
    ):
        print(
            f"  {name:38s} truth {t:10.5g} mean {m:10.5g} "
            f"(MC se {s:.3g}) coverage {cov:.3f}"
        )
    return verified, estimates, lower, upper, target


def check(runs):
    label = "joint marker causes"
    verified, estimates, lower, upper, target = summarise(runs, label)
    assert verified >= 0.98
    check_bias(
        estimates,
        target,
        label + " parameters",
        standard_errors=np.array([r["se"] for r in runs]),
    )
    check_coverage(
        lower, upper, target, 0.95, label + " 95% intervals", slack=0.02
    )


def reduced_run(reps=12):
    """The reduced run: ``reps`` fleets of 200, seeds from ``FIRST``,
    summarised (no pass rule at this size)."""
    with process_pool(2) as pool:
        runs = list(pool.map(_reduced, range(FIRST, FIRST + reps)))
    return summarise(runs, "joint marker causes (reduced)")


def test_joint_marker_causes_recovery_and_coverage():
    # every core: one test, its fits the work (one worker in CI)
    with process_pool() as pool:
        runs = list(pool.map(replicate, range(FIRST, FIRST + REPS)))
    check(runs)


if __name__ == "__main__":
    reduced_run()
