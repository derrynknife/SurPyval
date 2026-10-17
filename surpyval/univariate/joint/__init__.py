"""Joint models of a marker -- a measurement read repeatedly on each item
-- and the item's lifetime (#891): ``JointPH(dist, path_model)`` builds a
fitter (the marker following a degradation path model, a straight line by
default), ``WeibullJointPH`` is the ready-made Weibull one with a line,
and a fit returns a :class:`JointModel`, whose ``predict`` gives one
item's :class:`DynamicPrediction`."""

from surpyval.degradation.path_models import LinearPath
from surpyval.univariate.parametric import Weibull

from .dynamic import DynamicPrediction
from .joint_model import JointModel
from .joint_ph import JointPH, JointPHFitter

WeibullJointPH = JointPH(Weibull, LinearPath)

__all__ = [
    "DynamicPrediction",
    "JointModel",
    "JointPH",
    "JointPHFitter",
    "WeibullJointPH",
]
