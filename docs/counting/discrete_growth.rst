Discrete (Success/Fail) Reliability Growth
==========================================

Growth models for one-shot items tested in stages or configurations,
fitted to the failures and trials of each: the AMSAA (Crow) discrete
model, Lloyd-Lipow, and the Gompertz and modified Gompertz curves. Each
fit returns a :class:`~surpyval.recurrent.DiscreteGrowthModel`. A guide
with worked examples is :doc:`../Discrete Reliability Growth`.

.. autodata:: surpyval.recurrent.CrowAMSAADiscrete
   :no-value:

   .. automethod:: surpyval.recurrent.discrete_growth.CrowAMSAADiscrete_.fit

.. autodata:: surpyval.recurrent.LloydLipow
   :no-value:

.. autodata:: surpyval.recurrent.GompertzGrowth
   :no-value:

.. autodata:: surpyval.recurrent.ModifiedGompertzGrowth
   :no-value:

.. autoclass:: surpyval.recurrent.DiscreteGrowthModel
   :members:
