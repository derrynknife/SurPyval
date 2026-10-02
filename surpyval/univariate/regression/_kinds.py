"""The names of the parametric regression families.

A fitted ``ParametricRegressionModel`` says which family it is in its
public ``kind`` attribute, one of these strings, which ``to_dict``
stores. The package compares a model's ``kind`` with these names (or
through the model's predicates, ``_is_accelerated_life`` and
``_is_additive``) rather than with string literals, so a family is
renamed or added in one place.
"""

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
