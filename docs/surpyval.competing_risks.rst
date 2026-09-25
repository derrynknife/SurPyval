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

The cumulative incidence of each cause estimated without a model, and
the incidence-increment helper it (and Gray's test) is built on.

.. autoclass:: surpyval.univariate.competing_risks.nonparametric.competing_risks.CompetingRisks
   :members:

.. autofunction:: surpyval.univariate.competing_risks.aalen_johansen.aalen_johansen_iif

Parametric
----------

One distribution per cause, combined into cumulative incidence
functions.

.. autoclass:: surpyval.univariate.competing_risks.parametric.parametric_competing_risks.ParametricCompetingRisks
   :members:

Regression
----------

Covariate models for competing risks: the Fine-Gray subdistribution
hazards model (``FineGray`` is an instance of ``FineGray_`` below; its
``fit`` returns a ``FineGrayModel``), and
``CompetingRisksProportionalHazards``, which fits either a cause-specific
Cox model per cause or a Fine-Gray model per cause.

.. autoclass:: surpyval.univariate.competing_risks.regression.fine_gray.FineGray_
   :members:

.. autoclass:: surpyval.univariate.competing_risks.regression.fine_gray.FineGrayModel
   :members:

.. autoclass:: surpyval.univariate.competing_risks.regression.competing_risks_proportional_hazard.CompetingRisksProportionalHazards
   :members:
