Inflection S-Shaped
===================

Ohba's inflection S-shaped software reliability growth model, an NHPP
with :math:`\Lambda(t) = a (1 - e^{-b t}) / (1 + \psi e^{-b t})`: a
finite number of faults ``a``, with the inflection parameter ``psi``
(0 is Goel-Okumoto).
``InflectionS.fit`` returns a
:doc:`ParametricRecurrenceModel <parametric_recurrence_model>`; data
that show no reliability growth are refused.

.. autodata:: surpyval.recurrent.parametric.inflection_s.InflectionS
   :no-value:

   .. automethod:: surpyval.recurrent.parametric.inflection_s.InflectionS.fit
   .. automethod:: surpyval.recurrent.parametric.inflection_s.InflectionS.fit_from_recurrent_data
   .. automethod:: surpyval.recurrent.parametric.inflection_s.InflectionS.fit_from_df
   .. automethod:: surpyval.recurrent.parametric.inflection_s.InflectionS.from_params
   .. automethod:: surpyval.recurrent.parametric.inflection_s.InflectionS.cif
   .. automethod:: surpyval.recurrent.parametric.inflection_s.InflectionS.iif
   .. automethod:: surpyval.recurrent.parametric.inflection_s.InflectionS.log_iif
   .. automethod:: surpyval.recurrent.parametric.inflection_s.InflectionS.inv_cif
