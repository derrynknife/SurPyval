Semi-Parametric Proportional Odds
=================================

The proportional odds model with the baseline left to the data: the
covariates multiply the survival odds,
:math:`S(x \mid Z) / F(x \mid Z) = e^{\beta' Z} S_0(x) / F_0(x)`, and the
baseline failure odds :math:`G_0 = F_0 / S_0` are a step function that
jumps at the event times, estimated with :math:`\beta` by nonparametric
maximum likelihood (Murphy, Rossini and van der Vaart, 1997). A positive
coefficient means a longer life, as in the parametric ``PO`` family in
:doc:`parametric`. ``ProportionalOdds`` is an instance of the fitter class
below. Supports observed and right-censored data, with left truncation.

.. autoclass:: surpyval.univariate.regression.proportional_odds.proportional_odds.ProportionalOdds_
   :members:
   :inherited-members:

The fitted model:

.. autoclass:: surpyval.univariate.regression.proportional_odds.proportional_odds.ProportionalOddsModel
   :members:
