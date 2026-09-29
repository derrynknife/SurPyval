Cox-Lewis
=========

The Cox-Lewis (log-linear) NHPP, with intensity
:math:`\lambda(t) = e^{\alpha + \beta t}`: the rate changes by a
constant *factor* per unit time. ``CoxLewis.fit`` returns a
:doc:`ParametricRecurrenceModel <parametric_recurrence_model>`.

.. autodata:: surpyval.recurrent.parametric.cox_lewis.CoxLewis
   :no-value:

   .. automethod:: surpyval.recurrent.parametric.cox_lewis.CoxLewis.fit
   .. automethod:: surpyval.recurrent.parametric.cox_lewis.CoxLewis.fit_from_recurrent_data
   .. automethod:: surpyval.recurrent.parametric.cox_lewis.CoxLewis.from_params
   .. automethod:: surpyval.recurrent.parametric.cox_lewis.CoxLewis.cif
   .. automethod:: surpyval.recurrent.parametric.cox_lewis.CoxLewis.iif
   .. automethod:: surpyval.recurrent.parametric.cox_lewis.CoxLewis.log_iif
   .. automethod:: surpyval.recurrent.parametric.cox_lewis.CoxLewis.inv_cif
