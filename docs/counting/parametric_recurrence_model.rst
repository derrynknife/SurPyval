Parametric Recurrence Model
===========================

The fitted model returned by the Poisson-process fitters (``HPP``,
``CrowAMSAA``, ``Duane``, ``CoxLewis``) and by their ``from_params``.
It evaluates the fitted process -- the cumulative intensity ``cif``
(which for a Poisson process is also the mean cumulative function,
``mcf``), the intensity ``iif`` and its inverse ``inv_cif`` -- with
confidence bounds (``cif_cb``, ``param_cb``), likelihood inference
(``log_likelihood``, ``aic``, ``bic``, ``covariance``,
``standard_errors``), goodness-of-fit diagnostics (``residuals``,
``trend_test``, ``cramer_von_mises``), simulation of new histories and
plotting. Models built with ``from_params`` or fitted with
``how="MSE"`` have no likelihood, so the inference methods raise.

The base class of the Poisson-process fitters, which defines the
functions every intensity model provides, is
:class:`~surpyval.recurrent.parametric.counting_process.CountingProcess`,
documented at the end of this page.

.. autoclass:: surpyval.recurrent.parametric.parametric_recurrence.ParametricRecurrenceModel
   :members:
   :inherited-members:

.. autoclass:: surpyval.recurrent.parametric.counting_process.CountingProcess
   :members:
