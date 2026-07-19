Arithmetic Reduction of Intensity (ARI)
=======================================

The ``ARI`` imperfect-repair model reduces the *failure intensity* directly by
a fraction ``rho`` of the intensity built up over the last ``m`` inter-arrival
times (with ``m = numpy.inf`` the infinite-memory limit). It differs from
``ARA`` whenever the baseline intensity varies with time, and returns a
:doc:`Renewal Model <renewal_model>`.

.. class:: ARI

   .. automethod:: surpyval.recurrent.renewal.ari.ARI_.fit
   .. automethod:: surpyval.recurrent.renewal.ari.ARI_.fit_from_recurrent_data
   .. automethod:: surpyval.recurrent.renewal.ari.ARI_.fit_from_parameters
