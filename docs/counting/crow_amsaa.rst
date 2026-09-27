Crow-AMSAA
==========

The Crow-AMSAA (power-law) non-homogeneous Poisson process, with
cumulative intensity :math:`\Lambda(t) = (t / \alpha)^{\beta}`:
the event rate rises with age when :math:`\beta > 1` (deterioration),
falls when :math:`\beta < 1` (reliability growth) and is constant when
:math:`\beta = 1` (the HPP). ``CrowAMSAA.fit`` returns a
:doc:`ParametricRecurrenceModel <parametric_recurrence_model>`; the
methods below that take ``alpha, beta`` are the process's functions at
given parameters.

.. autodata:: surpyval.recurrent.parametric.crow_amsaa.CrowAMSAA
   :no-value:

   .. automethod:: surpyval.recurrent.parametric.crow_amsaa.CrowAMSAA.fit
   .. automethod:: surpyval.recurrent.parametric.crow_amsaa.CrowAMSAA.fit_from_recurrent_data
   .. automethod:: surpyval.recurrent.parametric.crow_amsaa.CrowAMSAA.from_params
   .. automethod:: surpyval.recurrent.parametric.crow_amsaa.CrowAMSAA.cif
   .. automethod:: surpyval.recurrent.parametric.crow_amsaa.CrowAMSAA.iif
   .. automethod:: surpyval.recurrent.parametric.crow_amsaa.CrowAMSAA.log_iif
   .. automethod:: surpyval.recurrent.parametric.crow_amsaa.CrowAMSAA.inv_cif
