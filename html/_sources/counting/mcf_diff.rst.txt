Comparing Two MCFs
==================

The difference between two groups' mean cumulative functions, with
pointwise confidence bounds from the Lawless-Nadeau robust variance, and
the pseudo-score tests of Cook, Lawless and Nadeau (1996) of whether the
groups' event rates differ (constant and linear weights; robust or
Poisson variance). Fit a :doc:`NonParametricCounting <nonparametric_mcf>`
to each group and pass both. A worked example is in
:doc:`../Recurrent Event Modelling with SurPyval`.

.. autofunction:: surpyval.recurrent.mcf_diff

.. autoclass:: surpyval.recurrent.MCFDiffResult
   :members:
