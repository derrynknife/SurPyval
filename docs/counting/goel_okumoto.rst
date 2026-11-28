Goel-Okumoto
============

The Goel-Okumoto software reliability growth model, an NHPP with
:math:`\Lambda(t) = a (1 - e^{-b t})`: a finite number of faults ``a``,
each found at the rate ``b``.
``GoelOkumoto.fit`` returns a
:doc:`ParametricRecurrenceModel <parametric_recurrence_model>`; data
that show no reliability growth are refused.

.. autodata:: surpyval.recurrent.parametric.goel_okumoto.GoelOkumoto
   :no-value:

   .. automethod:: surpyval.recurrent.parametric.goel_okumoto.GoelOkumoto.fit
   .. automethod:: surpyval.recurrent.parametric.goel_okumoto.GoelOkumoto.fit_from_df
   .. automethod:: surpyval.recurrent.parametric.goel_okumoto.GoelOkumoto.fit_bayes_from_df
   .. automethod:: surpyval.recurrent.parametric.goel_okumoto.GoelOkumoto.from_params
   .. automethod:: surpyval.recurrent.parametric.goel_okumoto.GoelOkumoto.cif
   .. automethod:: surpyval.recurrent.parametric.goel_okumoto.GoelOkumoto.iif
   .. automethod:: surpyval.recurrent.parametric.goel_okumoto.GoelOkumoto.log_iif
   .. automethod:: surpyval.recurrent.parametric.goel_okumoto.GoelOkumoto.inv_cif
