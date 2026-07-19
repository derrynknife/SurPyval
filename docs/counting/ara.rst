Arithmetic Reduction of Age (ARA)
=================================

The ``ARA`` imperfect-repair model reduces a *virtual age* by a fraction
``rho`` of the age accumulated over the last ``m`` inter-arrival times (with
``m = numpy.inf`` the infinite-memory limit). ``rho = 1`` is as-good-as-new and
``rho = 0`` is as-bad-as-old. It returns a
:doc:`Renewal Model <renewal_model>`.

.. class:: ARA

   .. automethod:: surpyval.recurrent.renewal.ara.ARA_.fit
   .. automethod:: surpyval.recurrent.renewal.ara.ARA_.fit_from_recurrent_data
   .. automethod:: surpyval.recurrent.renewal.ara.ARA_.fit_from_parameters
