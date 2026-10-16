"""Joint models of a longitudinal marker and a failure time (#891).

A marker read repeatedly on each unit (a vibration level, a wear
measurement, a biomarker) and the unit's failure, modelled together: a
linear mixed model of the readings and a proportional hazards model of the
failure, linked by the marker's current (noise-free) value. A fitted model
gives each unit's survival given its own readings so far, its dynamic
prediction (:meth:`JointModel.predict`), updated as readings arrive.

Examples
--------
>>> from surpyval.joint import JointModel
>>> model = JointModel.fit_from_df(df, x_col="x", c_col="c", i_col="i",
...                                marker_x_col="t", marker_y_col="y")
...                                # doctest: +SKIP
>>> model.predict(t_readings, y_readings).remaining_life()  # doctest: +SKIP
"""

from .dynamic import DynamicPrediction
from .joint_model import JointModel

__all__ = ["JointModel", "DynamicPrediction"]
