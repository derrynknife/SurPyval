Generalized Renewal Process
===========================

The generalized renewal (Kijima virtual-age) process: each repair
restores the system part of the way to new, by reducing its *virtual
age*, and the time to the next failure follows the lifetime
distribution conditional on that age. ``kijima="i"`` reduces only the
age added since the last repair, ``kijima="ii"`` the whole accumulated
age. The fit returns a :doc:`Renewal Model <renewal_model>`.

.. autodata:: surpyval.recurrent.renewal.generalized_renewal.GeneralizedRenewal
   :no-value:

   .. automethod:: surpyval.recurrent.renewal.generalized_renewal.GeneralizedRenewal.fit
   .. automethod:: surpyval.recurrent.renewal.generalized_renewal.GeneralizedRenewal.fit_from_recurrent_data
   .. automethod:: surpyval.recurrent.renewal.generalized_renewal.GeneralizedRenewal.fit_from_parameters
   .. automethod:: surpyval.recurrent.renewal.generalized_renewal.GeneralizedRenewal.kijima_i
   .. automethod:: surpyval.recurrent.renewal.generalized_renewal.GeneralizedRenewal.kijima_ii
