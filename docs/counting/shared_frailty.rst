Shared Frailty (Poisson Processes Whose Units Differ)
=====================================================

A fleet whose units differ by more than their covariates explain: each
unit (or cluster of units) has an unobserved frailty that multiplies its
intensity, gamma or log-normal, whose variance ``theta`` is estimated
with the intensity. It is fitted with ``frailty="gamma"`` (or
``"lognormal"``) and optionally ``cluster=`` on ``CrowAMSAA.fit``,
``Duane.fit``, ``CoxLewis.fit`` and ``ProportionalIntensityNHPP.fit``;
this is the model they return. A worked example is on
:doc:`/Recurrent Event Modelling with SurPyval`.

The fitted model
----------------

.. autoclass:: surpyval.recurrent.shared_frailty.SharedFrailtyModel
   :members:
   :inherited-members:

The frailty expectation
-----------------------

.. autofunction:: surpyval.recurrent.shared_frailty.mixed_poisson_log_moment

.. autofunction:: surpyval.recurrent.shared_frailty.frailty_moments
