Test Planning
=============

Functions that plan reliability tests before they are run: demonstration
tests (fixed-length and sequential), life tests sized for the precision of
an estimate, accelerated life tests, and the sample size and power of a
comparison of two groups, or of a failure rate before and after a change. Import them from ``surpyval.test_planning``. Worked
examples are in :doc:`Test Planning with SurPyval`.

.. py:module:: surpyval.test_planning

Demonstration tests
-------------------

.. autofunction:: surpyval.test_planning.demonstration_sample_size

.. autofunction:: surpyval.test_planning.demonstrated_reliability

.. autofunction:: surpyval.test_planning.demonstration_test_multiple

.. autofunction:: surpyval.test_planning.demonstration_test_time

.. autofunction:: surpyval.test_planning.demonstration_pass_probability

.. autofunction:: surpyval.test_planning.mtbf_test_time

.. autofunction:: surpyval.test_planning.demonstrated_mtbf

.. autofunction:: surpyval.test_planning.mtbf_pass_probability

.. autofunction:: surpyval.test_planning.demonstration_plan

.. autofunction:: surpyval.test_planning.mtbf_demonstration_plan

.. autoclass:: surpyval.test_planning.DemonstrationPlan

Sequential tests
----------------

.. autofunction:: surpyval.test_planning.sequential_mtbf_test

.. autoclass:: surpyval.test_planning.SequentialMTBFTest
   :members:

.. autofunction:: surpyval.test_planning.sequential_demonstration_test

.. autoclass:: surpyval.test_planning.SequentialDemonstrationTest
   :members:

Estimation precision
--------------------

.. autofunction:: surpyval.test_planning.life_test_precision

.. autofunction:: surpyval.test_planning.life_test_sample_size

.. autofunction:: surpyval.test_planning.life_test_length

.. autoclass:: surpyval.test_planning.LifeTestPlan
   :members: simulate, to_dict

.. autoclass:: surpyval.test_planning.SimulatedPrecision

Accelerated life tests
----------------------

.. autofunction:: surpyval.test_planning.accelerated_test_plan

.. autofunction:: surpyval.test_planning.accelerated_test_precision

.. autoclass:: surpyval.test_planning.AcceleratedTestPlan
   :members: simulate, to_dict

Power and sample size
---------------------

One function per design: give every quantity but one, and the one left
``None`` is solved for. A log-rank test of a hazard ratio, a Cox model's
coefficient, non-inferiority and equivalence of a hazard ratio, a
difference in restricted mean survival time, a log-rank test when some
units are cured, and the exact test of a change in a failure rate at an
intervention (:func:`surpyval.recurrent.intervention_test`).

.. autofunction:: surpyval.test_planning.logrank_design

.. autofunction:: surpyval.test_planning.cox_design

.. autofunction:: surpyval.test_planning.noninferiority_design

.. autofunction:: surpyval.test_planning.equivalence_design

.. autofunction:: surpyval.test_planning.rmst_design

.. autofunction:: surpyval.test_planning.cure_logrank_design

.. autofunction:: surpyval.test_planning.rate_ratio_design

.. autofunction:: surpyval.test_planning.mean_ratio_design

.. autoclass:: surpyval.test_planning.PowerCalculation
   :members: to_dict

Group-sequential designs
------------------------

Designs looked at a few planned times, stopping early for efficacy or
futility while keeping their level.

.. autofunction:: surpyval.test_planning.logrank_group_sequential

.. autoclass:: surpyval.test_planning.LogrankGroupSequentialDesign
   :members: hazard_ratio_estimate, update, to_dict

.. autofunction:: surpyval.test_planning.group_sequential_design

.. autoclass:: surpyval.test_planning.GroupSequentialDesign
   :members: decide, update, p_value, estimate, repeated_interval, crossing_probabilities, max_information, to_dict, from_dict

.. autofunction:: surpyval.test_planning.boundary_crossing

Reliability growth planning
---------------------------

The curve a reliability growth test is planned against (MIL-HDBK-189C):
see :doc:`Reliability Growth Planning`.

.. autofunction:: surpyval.test_planning.idealised_growth_curve

.. autoclass:: surpyval.test_planning.IdealisedGrowthCurve
   :members: mtbf, expected_failures, cumulative_mtbf, phase_mtbf, demonstration_probability, oc_curve, plot, to_dict

.. autofunction:: surpyval.test_planning.growth_demonstration_probability

.. autofunction:: surpyval.test_planning.growth_test_time

.. autofunction:: surpyval.test_planning.pm2_continuous

.. autoclass:: surpyval.test_planning.PM2ContinuousPlan
   :members: goal_to_potential, lambda_a, lambda_b, mtbf, intensity, new_mode_rate, expected_modes, fraction_surfaced, targets, demonstration_probability, plot, to_dict

.. autofunction:: surpyval.test_planning.pm2_discrete

.. autoclass:: surpyval.test_planning.PM2DiscretePlan
   :members: r_a, r_b, reliability, expected_modes, fraction_surfaced, new_mode_probability, targets, plot, to_dict
