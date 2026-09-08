Competing Risks
===============

Models for units that can fail from one of several distinct causes,
where the occurrence of one cause removes the unit from risk of the
others. Import them from ``surpyval.univariate.competing_risks``.

For a narrative introduction with worked examples, see
:doc:`Competing Risks SurPyval Modelling`; for the statistical
background, see :doc:`Competing Risks Analysis`. The Gray test for
comparing cumulative incidence between groups is documented with the
other hypothesis tests in :doc:`comparison_and_validation`.

Non-Parametric (Aalen-Johansen)
-------------------------------

.. autoclass:: surpyval.univariate.competing_risks.nonparametric.competing_risks.CompetingRisks
   :members:

.. autofunction:: surpyval.univariate.competing_risks.aalen_johansen.aalen_johansen_iif

Parametric
----------

.. autoclass:: surpyval.univariate.competing_risks.parametric.parametric_competing_risks.ParametricCompetingRisks
   :members:

Regression
----------

.. autoclass:: surpyval.univariate.competing_risks.regression.fine_gray.FineGray_
   :members:

.. autoclass:: surpyval.univariate.competing_risks.regression.competing_risks_proportional_hazard.CompetingRisksProportionalHazards
   :members:
