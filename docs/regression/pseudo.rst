Pseudo-Observations
===================

Jackknife pseudo-values of the Kaplan-Meier survival, the restricted mean
survival time and a cause's cumulative incidence, and the regression on
them (estimating equations with robust standard errors). Importable
directly from ``surpyval``; the guide, with worked examples and the sign
of the coefficients, is :doc:`../Pseudo-Observations`.

.. autofunction:: surpyval.pseudo_survival

.. autofunction:: surpyval.pseudo_rmst

.. autofunction:: surpyval.pseudo_cif

.. autofunction:: surpyval.pseudo_regression

.. autofunction:: surpyval.pseudo_regression_from_df

.. autoclass:: surpyval.PseudoRegressionResult
    :members:
