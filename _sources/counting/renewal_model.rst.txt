Renewal Model
=============

The fitted-model object returned by the renewal / virtual-age fitters
(``GeneralizedRenewal``, ``GeneralizedOneRenewal``, ``ARA``, ``ARI``). It holds
the fitted lifetime distribution and restoration parameter and provides the
simulation, inference (``standard_errors``, ``param_cb``) and diagnostic
(``residuals``, ``trend_test``, ``cramer_von_mises``) behaviour.

.. autoclass:: surpyval.recurrent.renewal.renewal_model.RenewalModel
   :members:
   :inherited-members:
