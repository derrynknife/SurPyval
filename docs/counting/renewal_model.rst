Renewal Model
=============

The fitted-model object returned by the renewal / virtual-age fitters
(``GeneralizedRenewal``, ``GeneralizedOneRenewal``, ``ARA``, ``ARI``) and
by their ``fit_from_parameters``. It holds the fitted lifetime
distribution (``model``, a :doc:`Parametric <../univariate/parametric_class>`
model; for ``ARI`` the baseline intensity model) and the restoration
parameter (``q`` or ``rho``), and provides the simulation
(``count_terminated_simulation``, ``time_terminated_simulation``, and
``mcf`` and ``plot``, which are estimated by simulation because these
processes have no closed-form intensity), inference (``standard_errors``,
``param_cb``) and diagnostic (``residuals``, ``trend_test``,
``cramer_von_mises``) behaviour.

.. autoclass:: surpyval.recurrent.renewal.renewal_model.RenewalModel
   :members:
   :inherited-members:
