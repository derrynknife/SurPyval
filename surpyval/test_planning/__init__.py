"""
Test planning: designing reliability tests before they are run.

- **Demonstration tests** (#862) show, at a confidence level, that a
  design meets a reliability or MTBF target. Attribute (pass/fail) tests:
  :func:`demonstration_sample_size`, :func:`demonstrated_reliability`
  (the Clopper-Pearson bound; with no failures,
  :func:`surpyval.success_run`), :func:`demonstration_test_multiple` and
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
- **Comparing two groups** (#880): the events, units and power of a
  log-rank test of a hazard ratio: :func:`logrank_events`,
  :func:`logrank_sample_size` and :func:`logrank_power`.

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
"""

from surpyval.test_planning.accelerated import (
    AcceleratedTestPlan,
    accelerated_test_plan,
    accelerated_test_precision,
)
from surpyval.test_planning.comparison import (
    logrank_events,
    logrank_power,
    logrank_sample_size,
)
from surpyval.test_planning.demonstration import (
    DemonstrationPlan,
    demonstrated_mtbf,
    demonstrated_reliability,
    demonstration_pass_probability,
    demonstration_plan,
    demonstration_sample_size,
    demonstration_test_multiple,
    mtbf_demonstration_plan,
    mtbf_pass_probability,
    mtbf_test_time,
)
from surpyval.test_planning.estimation import (
    LifeTestPlan,
    SimulatedPrecision,
    life_test_length,
    life_test_precision,
    life_test_sample_size,
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
    # Comparing two groups (#880)
    "logrank_events",
    "logrank_power",
    "logrank_sample_size",
]
