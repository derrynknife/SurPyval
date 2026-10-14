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
]
