Joint Models
============

Joint models of a longitudinal marker and a failure time: a linear mixed
model of a marker read repeatedly on each unit and a Weibull proportional
hazards model of its failure, linked by the marker's current (noise-free)
value, fitted together by maximum likelihood. A fitted model gives the
survival of a new unit and, for a unit with readings, its dynamic
prediction: its survival given its readings so far, updated as readings
arrive. Import them from ``surpyval.joint``. Worked examples are in
:doc:`Joint Modelling with SurPyval`.

Joint Model
-----------

``JointModel`` is both the fitter (``JointModel.fit``,
``fit_from_df``, ``fit_from_items``, ``from_params``) and the fitted model
they return.

.. autoclass:: surpyval.joint.JointModel
   :members:
   :exclude-members: survival, link, path, maximum, Z_cols, path_Z_cols, data

Dynamic Prediction
------------------

.. autoclass:: surpyval.joint.DynamicPrediction
   :members:
