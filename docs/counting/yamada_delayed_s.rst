Yamada Delayed S-Shaped
=======================

The delayed S-shaped software reliability growth model of Yamada, Ohba
and Osaki, an NHPP with
:math:`\Lambda(t) = a (1 - (1 + b t) e^{-b t})`: a finite number of
faults ``a``, found slowly at first.
``YamadaDelayedS.fit`` returns a
:doc:`ParametricRecurrenceModel <parametric_recurrence_model>`; data
that show no reliability growth are refused.

.. autodata:: surpyval.recurrent.parametric.yamada_delayed_s.YamadaDelayedS
   :no-value:

   .. automethod:: surpyval.recurrent.parametric.yamada_delayed_s.YamadaDelayedS.fit
   .. automethod:: surpyval.recurrent.parametric.yamada_delayed_s.YamadaDelayedS.fit_from_df
   .. automethod:: surpyval.recurrent.parametric.yamada_delayed_s.YamadaDelayedS.from_params
   .. automethod:: surpyval.recurrent.parametric.yamada_delayed_s.YamadaDelayedS.cif
   .. automethod:: surpyval.recurrent.parametric.yamada_delayed_s.YamadaDelayedS.iif
   .. automethod:: surpyval.recurrent.parametric.yamada_delayed_s.YamadaDelayedS.log_iif
   .. automethod:: surpyval.recurrent.parametric.yamada_delayed_s.YamadaDelayedS.inv_cif
