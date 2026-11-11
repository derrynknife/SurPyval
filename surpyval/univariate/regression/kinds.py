"""The names of the parametric regression families.

A fitted ``ParametricRegressionModel`` says which family it is in its
public ``kind`` attribute, one of these strings, which ``to_dict``
stores. Compare a model's ``kind`` with these names rather than with
string literals, so a family is renamed or added in one place:

>>> import surpyval as surv
>>> from surpyval.univariate.regression import kinds
>>> x = [10.0, 14.0, 21.0, 26.0, 35.0, 41.0, 52.0, 60.0]
>>> Z = [[1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]]
>>> model = surv.WeibullAFT.fit(x, Z=Z)
>>> model.kind == kinds.ACCELERATED_FAILURE_TIME
True
>>> model.kind in kinds.TIME_SCALING
True

``TIME_SCALING`` holds the families whose covariates stretch or shrink
the time axis; ``HAZARD_BASED`` those whose covariates act on the
hazard. Proportional odds and distributional regression are in neither.
"""

__all__ = [
    "PROPORTIONAL_HAZARD",
    "ACCELERATED_FAILURE_TIME",
    "PROPORTIONAL_ODDS",
    "ADDITIVE_HAZARD",
    "ACCELERATED_LIFE",
    "DISTRIBUTIONAL",
    "TIME_SCALING",
    "HAZARD_BASED",
]

#: The covariates multiply the baseline hazard (``PH``, ``WeibullPH``).
PROPORTIONAL_HAZARD = "Proportional Hazard"
#: The covariates rescale time (``AFT``, ``WeibullAFT``).
ACCELERATED_FAILURE_TIME = "Accelerated Failure Time"
#: The covariates scale the baseline odds (``PO``, ``WeibullPO``; also
#: the semi-parametric ``ProportionalOdds``).
PROPORTIONAL_ODDS = "Proportional Odds"
#: The covariates add to the baseline hazard (``AH``, ``WeibullAH``).
ADDITIVE_HAZARD = "Additive Hazard"
#: A life model gives a distribution's life parameter at each stress
#: (``AcceleratedLife``).
ACCELERATED_LIFE = "Accelerated Life"
#: Every parameter of the distribution has its own linear predictor and
#: link (``Distributional``).
DISTRIBUTIONAL = "Distributional"

#: The families whose covariates rescale time: a unit under covariates
#: ``Z`` ages along the baseline's clock run faster or slower.
TIME_SCALING = frozenset({ACCELERATED_FAILURE_TIME, ACCELERATED_LIFE})
#: The families whose covariates act on the baseline hazard, by
#: multiplying it or adding to it.
HAZARD_BASED = frozenset({PROPORTIONAL_HAZARD, ADDITIVE_HAZARD})
