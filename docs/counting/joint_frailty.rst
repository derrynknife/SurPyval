Joint Frailty Model (Recurrent and Terminal Events)
===================================================

Recurrent events that stop at a terminal event which depends on them: a
machine repaired again and again until it is scrapped, a patient
readmitted until death. Each unit has an unobserved frailty that raises
both its rate of recurrent events and its hazard of the terminal event,
so a unit with many recurrences tends to reach the terminal event sooner.
Both baselines are Weibull (the recurrent one is the power-law
intensity), on the unit's age or, for the recurrences, the time since
the last event. The theory is on :doc:`/Recurrent Event Regression
Analysis`, with worked examples on :doc:`/Recurrent Event Regression
Modelling with SurPyval`.

The fitter
----------

.. autodata:: surpyval.recurrent.joint_frailty.JointFrailty
   :no-value:

   .. automethod:: surpyval.recurrent.joint_frailty.JointFrailty.fit
   .. automethod:: surpyval.recurrent.joint_frailty.JointFrailty.fit_from_df
   .. automethod:: surpyval.recurrent.joint_frailty.JointFrailty.from_params

The fitted model
----------------

.. autoclass:: surpyval.recurrent.joint_frailty.JointFrailtyModel
   :members:
   :inherited-members:

The frailty integral
--------------------

.. autofunction:: surpyval.recurrent.joint_frailty.log_frailty_integral
