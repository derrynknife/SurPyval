Fleming-Harrington
==================

The Fleming-Harrington (tie-corrected Nelson-Aalen) estimator: tied
deaths are treated as happening one after another, so each removes one
unit from the risk set before the next is counted. It lies between
Nelson-Aalen and Kaplan-Meier. ``FlemingHarrington`` is an instance of
the class below; ``FlemingHarrington.fit`` returns a
:doc:`NonParametric <non_parametric_class>` model. Theory:
:doc:`../Non-Parametric Estimation`.

.. autoclass:: surpyval.univariate.nonparametric.fleming_harrington.FlemingHarrington_
   :members:
   :inherited-members:
