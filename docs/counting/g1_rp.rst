Generalized One Renewal Process
===============================

The G1 renewal process (Lam's geometric process, reparameterised): the
``j``-th inter-arrival time is the base lifetime scaled by
``(1 + q)**j``, so ``q < 0`` is a deteriorating system and ``q > 0`` an
improving one. The fit returns a :doc:`Renewal Model <renewal_model>`.

.. autodata:: surpyval.recurrent.renewal.generalized_one_renewal.GeneralizedOneRenewal
   :no-value:

   .. automethod:: surpyval.recurrent.renewal.generalized_one_renewal.GeneralizedOneRenewal.fit
   .. automethod:: surpyval.recurrent.renewal.generalized_one_renewal.GeneralizedOneRenewal.fit_from_recurrent_data
   .. automethod:: surpyval.recurrent.renewal.generalized_one_renewal.GeneralizedOneRenewal.fit_from_parameters
