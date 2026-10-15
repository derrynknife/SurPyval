"""Joint models of a marker -- a measurement read repeatedly on each item
-- and the item's lifetime (#891): ``JointPH(dist)`` builds a fitter,
``WeibullJointPH`` is the ready-made Weibull one, and a fit returns a
:class:`JointModel`."""

from surpyval.univariate.parametric import Weibull

from .joint_model import JointModel
from .joint_ph import JointPH, JointPHFitter

WeibullJointPH = JointPH(Weibull)

__all__ = ["JointModel", "JointPH", "JointPHFitter", "WeibullJointPH"]
