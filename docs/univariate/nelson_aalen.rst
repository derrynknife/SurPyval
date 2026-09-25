Nelson-Aalen
============

Estimates the cumulative hazard as the running sum of deaths over the
number at risk, and the survival curve as :math:`e^{-H}`. Because
:math:`e^{-a} \geq 1 - a`, its survival curve never lies below
Kaplan-Meier's. ``NelsonAalen`` is an instance of the class below;
``NelsonAalen.fit`` returns a :doc:`NonParametric <non_parametric_class>`
model. Theory: :doc:`../Non-Parametric Estimation`.

.. autoclass:: surpyval.univariate.nonparametric.nelson_aalen.NelsonAalen_
   :members:
   :inherited-members:
