Rayleigh Defects
================

The Rayleigh defect model, an NHPP with
:math:`\Lambda(t) = K (1 - e^{-t^2 / (2 c^2)})`: the defects found
across the phases of a development rise to a peak at ``c`` and fall
away, ``K`` in all. ``RayleighDefects.fit`` returns a
:doc:`ParametricRecurrenceModel <parametric_recurrence_model>`; counts
that have not yet turned down are refused.
:func:`~surpyval.recurrent.phases_to_data` lays a table of counts by
phase out as the rows ``fit`` takes.

.. autodata:: surpyval.recurrent.parametric.rayleigh_defects.RayleighDefects
   :no-value:

   .. automethod:: surpyval.recurrent.parametric.rayleigh_defects.RayleighDefects.fit
   .. automethod:: surpyval.recurrent.parametric.rayleigh_defects.RayleighDefects.fit_from_df
   .. automethod:: surpyval.recurrent.parametric.rayleigh_defects.RayleighDefects.fit_bayes_from_df
   .. automethod:: surpyval.recurrent.parametric.rayleigh_defects.RayleighDefects.from_params
   .. automethod:: surpyval.recurrent.parametric.rayleigh_defects.RayleighDefects.cif
   .. automethod:: surpyval.recurrent.parametric.rayleigh_defects.RayleighDefects.iif
   .. automethod:: surpyval.recurrent.parametric.rayleigh_defects.RayleighDefects.log_iif
   .. automethod:: surpyval.recurrent.parametric.rayleigh_defects.RayleighDefects.inv_cif

.. autofunction:: surpyval.recurrent.phases_to_data
