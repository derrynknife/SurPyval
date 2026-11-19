"""Joint models of a marker -- a measurement read repeatedly on each item
-- and the item's lifetime (#891): ``JointPH(dist, path_model)`` builds a
fitter (the marker following a degradation path model, a straight line by
default), ``WeibullJointPH`` is the ready-made Weibull one with a line,
a fit returns a :class:`JointModel`, whose ``predict`` gives one
item's :class:`DynamicPrediction`, and a fitter's ``compare_links`` fits
several links to the same data and compares them
(:class:`LinkComparison`)."""

from surpyval.degradation.path_models import LinearPath
from surpyval.univariate.parametric import Weibull

from .compare import LinkComparison
from .dynamic import DynamicPrediction
from .joint_model import JointModel
from .joint_ph import JointPH, JointPHFitter

WeibullJointPH = JointPH(Weibull, LinearPath)

__all__ = [
    "DynamicPrediction",
    "JointModel",
    "JointPH",
    "JointPHFitter",
    "LinkComparison",
    "WeibullJointPH",
]
