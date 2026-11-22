Musa-Okumoto
============

The Musa-Okumoto logarithmic Poisson model, an NHPP with
:math:`\Lambda(t) = \log(1 + \lambda_0 \theta t) / \theta`: an
intensity that falls by the proportion ``theta`` with each failure, and
no limit to the number of failures.
``MusaOkumoto.fit`` returns a
:doc:`ParametricRecurrenceModel <parametric_recurrence_model>`; data
that show no reliability growth are refused.

.. autodata:: surpyval.recurrent.parametric.musa_okumoto.MusaOkumoto
   :no-value:

   .. automethod:: surpyval.recurrent.parametric.musa_okumoto.MusaOkumoto.fit
   .. automethod:: surpyval.recurrent.parametric.musa_okumoto.MusaOkumoto.fit_from_df
   .. automethod:: surpyval.recurrent.parametric.musa_okumoto.MusaOkumoto.from_params
   .. automethod:: surpyval.recurrent.parametric.musa_okumoto.MusaOkumoto.cif
   .. automethod:: surpyval.recurrent.parametric.musa_okumoto.MusaOkumoto.iif
   .. automethod:: surpyval.recurrent.parametric.musa_okumoto.MusaOkumoto.log_iif
   .. automethod:: surpyval.recurrent.parametric.musa_okumoto.MusaOkumoto.inv_cif
