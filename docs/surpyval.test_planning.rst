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

Comparing two groups
--------------------

.. autofunction:: surpyval.test_planning.logrank_events

.. autofunction:: surpyval.test_planning.logrank_sample_size

.. autofunction:: surpyval.test_planning.logrank_power

Comparing two failure rates
---------------------------

The power of the exact test of a change in a failure rate at an
intervention, :func:`surpyval.recurrent.intervention_test`.

.. autofunction:: surpyval.test_planning.rate_ratio_power

.. autofunction:: surpyval.test_planning.rate_ratio_detectable

.. autofunction:: surpyval.test_planning.rate_ratio_exposure
