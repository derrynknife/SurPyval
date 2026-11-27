Turnbull
========

The Turnbull estimator: the non-parametric estimate for data with any
mix of observed, left-, right- and interval-censored and truncated
observations, found by an EM (self-consistency) algorithm that shares
each uncertain observation out over the intervals it could have fallen
in. ``Turnbull`` is an instance of the class below; ``Turnbull.fit``
returns a :doc:`NonParametric <non_parametric_class>` model. Theory:
:doc:`../Non-Parametric Estimation`.

.. autoclass:: surpyval.univariate.nonparametric.turnbull.Turnbull_
   :members:
   :inherited-members:
