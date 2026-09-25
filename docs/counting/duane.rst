Duane
=====

The Duane reliability-growth model, a power-law NHPP written as
:math:`\Lambda(t) = b\, t^{\alpha}` -- the same process as
Crow-AMSAA in a different parameterisation
(:math:`b = \alpha_{CA}^{-\beta_{CA}}`, :math:`\alpha = \beta_{CA}`).
``Duane.fit`` returns a
:doc:`ParametricRecurrenceModel <parametric_recurrence_model>`.

.. autodata:: surpyval.recurrent.parametric.duane.Duane
   :no-value:

   .. automethod:: surpyval.recurrent.parametric.duane.Duane.fit
   .. automethod:: surpyval.recurrent.parametric.duane.Duane.fit_from_recurrent_data
   .. automethod:: surpyval.recurrent.parametric.duane.Duane.from_params
   .. automethod:: surpyval.recurrent.parametric.duane.Duane.cif
   .. automethod:: surpyval.recurrent.parametric.duane.Duane.iif
   .. automethod:: surpyval.recurrent.parametric.duane.Duane.log_iif
   .. automethod:: surpyval.recurrent.parametric.duane.Duane.inv_cif
