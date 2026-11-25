"""
Test planning: designing reliability tests before they are run.

- **Demonstration tests** (#862) show, at a confidence level, that a
  design meets a reliability or MTBF target. Attribute (pass/fail) tests:
  :func:`demonstration_sample_size`, :func:`demonstrated_reliability`
  (the Clopper-Pearson bound; with no failures,
  :func:`surpyval.success_run`), :func:`demonstration_test_multiple`
  (in missions), :func:`demonstration_test_time` (in time) and
  :func:`demonstration_pass_probability`, with extended (Weibayes) tests
  that trade units for test time when the Weibull shape is known.
  Constant-failure-rate (chi-squared) tests: :func:`mtbf_test_time`,
  :func:`demonstrated_mtbf` and :func:`mtbf_pass_probability`. Plans that
  keep both the consumer's and the producer's risk:
  :func:`demonstration_plan` and :func:`mtbf_demonstration_plan`.
- **Sequential tests** (#865) decide as the evidence comes in, usually
  sooner than a fixed-length test with the same risks:
  :func:`sequential_mtbf_test` and :func:`sequential_demonstration_test`
  (Wald's sequential probability ratio test, truncated at the
  fixed-length test).
- **Estimation precision** (#863): how many units, and how long a test,
  to estimate a Weibull or lognormal quantile or mean to a given
  precision: :func:`life_test_precision`, :func:`life_test_sample_size`
  and :func:`life_test_length`.
- **Accelerated life tests** (#864): the stresses and the units at each
  that estimate a quantile at the use stress most precisely:
  :func:`accelerated_test_plan` (optimum and compromise plans) and
  :func:`accelerated_test_precision` for a test of your own.
- **Power and sample size, one function per design**: give every
  quantity of the design but one, and the one left ``None`` is solved
  for, in a :class:`PowerCalculation` that says which it was. A
  log-rank test of a hazard ratio (:func:`logrank_design`); a Cox
  model's coefficient for a continuous or adjusted covariate
  (:func:`cox_design`); non-inferiority and equivalence of a hazard ratio
  (:func:`noninferiority_design`, :func:`equivalence_design`); a
  difference in restricted mean survival time (:func:`rmst_design`); a
  log-rank test when some units are cured (:func:`cure_logrank_design`);
  a ratio of two mean lives, exponential or Weibull with a known common
  shape, by the exact F test (:func:`mean_ratio_design`);
  and a change in a failure rate at an intervention, by the exact test
  of :func:`surpyval.recurrent.intervention_test`
  (:func:`rate_ratio_design`).
- **Group-sequential designs**: a comparison looked at a few planned
  times, stopping early for efficacy or futility while keeping its
  level: :func:`group_sequential_design` (any normal test statistic),
  :func:`logrank_group_sequential` (the events and units of a log-rank
  test) and :func:`boundary_crossing` (the chances of crossing given
  boundaries).
- **Reliability growth planning** (MIL-HDBK-189C): the curve a growth
  test is planned against. :func:`idealised_growth_curve` (the
  AMSAA-Crow planning curve: solve for the test time, the growth rate or
  the final MTBF), with the chance that the test's own data demonstrate
  a requirement (:func:`growth_demonstration_probability`) and the test
  time that makes that chance a given value (:func:`growth_test_time`);
  :func:`pm2_continuous` and :func:`pm2_discrete`, planned from the
  management strategy and the fix effectiveness, with the MTBF (or
  reliability) step after each corrective action period.

Every confidence level is given as ``alpha_ci``, the significance level,
as everywhere in SurPyval: for a demonstration test it is the consumer's
risk, ``0.05`` for 95% confidence.

Examples
--------
>>> from surpyval.test_planning import (
...     demonstration_sample_size,
...     mtbf_test_time,
... )
>>> demonstration_sample_size(0.95)
59
>>> round(mtbf_test_time(1000.0, failures=1))
4744
>>> from surpyval.test_planning import logrank_design
>>> logrank_design(0.5, power=0.8).events
66
"""

from typing import Any

from surpyval.test_planning.accelerated import (
    AcceleratedTestPlan,
    accelerated_test_plan,
    accelerated_test_precision,
)
from surpyval.test_planning.demonstration import (
    DemonstrationPlan,
    demonstrated_mtbf,
    demonstrated_reliability,
    demonstration_pass_probability,
    demonstration_plan,
    demonstration_sample_size,
    demonstration_test_multiple,
    demonstration_test_time,
    mtbf_demonstration_plan,
    mtbf_pass_probability,
    mtbf_test_time,
)
from surpyval.test_planning.designs import (
    PowerCalculation,
    cox_design,
    cure_logrank_design,
    equivalence_design,
    logrank_design,
    mean_ratio_design,
    noninferiority_design,
    rate_ratio_design,
    rmst_design,
)
from surpyval.test_planning.estimation import (
    LifeTestPlan,
    SimulatedPrecision,
    life_test_length,
    life_test_precision,
    life_test_sample_size,
)
from surpyval.test_planning.group_sequential import (
    GroupSequentialDesign,
    LogrankGroupSequentialDesign,
    boundary_crossing,
    group_sequential_design,
    logrank_group_sequential,
)
from surpyval.test_planning.growth import (
    IdealisedGrowthCurve,
    PM2ContinuousPlan,
    PM2DiscretePlan,
    growth_demonstration_probability,
    growth_test_time,
    idealised_growth_curve,
    pm2_continuous,
    pm2_discrete,
)
from surpyval.test_planning.sequential import (
    SequentialDemonstrationTest,
    SequentialMTBFTest,
    sequential_demonstration_test,
    sequential_mtbf_test,
)

__all__ = [
    # Demonstration tests (#862)
    "DemonstrationPlan",
    "demonstrated_mtbf",
    "demonstrated_reliability",
    "demonstration_pass_probability",
    "demonstration_plan",
    "demonstration_sample_size",
    "demonstration_test_multiple",
    "demonstration_test_time",
    "mtbf_demonstration_plan",
    "mtbf_pass_probability",
    "mtbf_test_time",
    # Sequential tests (#865)
    "SequentialDemonstrationTest",
    "SequentialMTBFTest",
    "sequential_demonstration_test",
    "sequential_mtbf_test",
    # Estimation precision (#863)
    "LifeTestPlan",
    "SimulatedPrecision",
    "life_test_length",
    "life_test_precision",
    "life_test_sample_size",
    # Accelerated life tests (#864)
    "AcceleratedTestPlan",
    "accelerated_test_plan",
    "accelerated_test_precision",
    # Power and sample size: one function per design
    "PowerCalculation",
    "cox_design",
    "cure_logrank_design",
    "equivalence_design",
    "logrank_design",
    "mean_ratio_design",
    "noninferiority_design",
    "rate_ratio_design",
    "rmst_design",
    # Group-sequential designs
    "GroupSequentialDesign",
    "LogrankGroupSequentialDesign",
    "boundary_crossing",
    "group_sequential_design",
    "logrank_group_sequential",
    # Reliability growth planning
    "IdealisedGrowthCurve",
    "PM2ContinuousPlan",
    "PM2DiscretePlan",
    "growth_demonstration_probability",
    "growth_test_time",
    "idealised_growth_curve",
    "pm2_continuous",
    "pm2_discrete",
]

# The power and sample-size functions of one design each, replaced in
# v0.29 by the design's one function (each still reachable, with a
# FutureWarning naming the call that replaces it, until v0.30).
_REPLACED = {
    "logrank_events": (
        "comparison",
        "logrank_design(hazard_ratio, power=...).events",
    ),
    "logrank_sample_size": (
        "comparison",
        "logrank_design(hazard_ratio, power=..., event_probability=...).n",
    ),
    "logrank_power": (
        "comparison",
        "logrank_design(hazard_ratio, n=..., event_probability=...).power",
    ),
    "cox_events": (
        "sample_size",
        "cox_design(hazard_ratio, power=..., sd=...).events",
    ),
    "cox_sample_size": (
        "sample_size",
        "cox_design(hazard_ratio, power=..., event_probability=..., sd=...).n",
    ),
    "cox_power": (
        "sample_size",
        "cox_design(hazard_ratio, n=..., event_probability=..., sd=...).power",
    ),
    "cox_detectable": (
        "sample_size",
        "cox_design(power=..., n=..., event_probability=..., sd=...).value",
    ),
    "noninferiority_events": (
        "sample_size",
        "noninferiority_design(margin, power=...).events",
    ),
    "noninferiority_sample_size": (
        "sample_size",
        "noninferiority_design(margin, power=..., event_probability=...).n",
    ),
    "noninferiority_power": (
        "sample_size",
        "noninferiority_design(margin, n=..., event_probability=...).power",
    ),
    "equivalence_events": (
        "sample_size",
        "equivalence_design(margin, power=...).events",
    ),
    "equivalence_sample_size": (
        "sample_size",
        "equivalence_design(margin, power=..., event_probability=...).n",
    ),
    "equivalence_power": (
        "sample_size",
        "equivalence_design(margin, n=..., event_probability=...).power",
    ),
    "rmst_sample_size": (
        "sample_size",
        "rmst_design(survival, tau, power=...).n",
    ),
    "rmst_power": ("sample_size", "rmst_design(survival, tau, n=...).power"),
    "cure_logrank_sample_size": (
        "sample_size",
        "cure_logrank_design(cured, latency, power=..., follow_up=...).n",
    ),
    "cure_logrank_events": (
        "sample_size",
        "cure_logrank_design(cured, latency, power=..., follow_up=...).events",
    ),
    "cure_logrank_power": (
        "sample_size",
        "cure_logrank_design(cured, latency, n=..., follow_up=...).power",
    ),
    "rate_ratio_power": (
        "rate_ratio",
        "rate_ratio_design(ratio, rate=..., exposure_before=..., "
        "exposure_after=...).power",
    ),
    "rate_ratio_detectable": (
        "rate_ratio",
        "rate_ratio_design(rate=..., exposure_before=..., exposure_after=..., "
        "power=...).value",
    ),
    "rate_ratio_exposure": (
        "rate_ratio",
        "rate_ratio_design(ratio, rate=..., exposure_before=..., "
        "power=...).value",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _REPLACED:
        import importlib

        from surpyval.utils.deprecation import REMOVED_IN, renamed_name

        module, new = _REPLACED[name]
        value = getattr(
            importlib.import_module(f"surpyval.test_planning.{module}"),
            f"_{name}",
        )
        return renamed_name(
            "surpyval.test_planning",
            name,
            new,
            value,
            REMOVED_IN,
            FutureWarning,
        )
    raise AttributeError(
        f"module 'surpyval.test_planning' has no attribute {name!r}"
    )
