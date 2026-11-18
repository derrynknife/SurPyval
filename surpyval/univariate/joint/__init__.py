"""Joint models of a marker -- a measurement read repeatedly on each item
-- and the item's lifetime (#891): ``JointPH(dist, path_model)`` builds a
fitter (the marker following a degradation path model, a straight line by
default) whose marker multiplies the hazard, ``JointAFT(dist,
path_model)`` one whose marker speeds up the item's ageing;
``WeibullJointPH`` and ``WeibullJointAFT`` are the ready-made Weibull
ones with a line, and a fit returns a :class:`JointModel`, whose
``predict`` gives one item's :class:`DynamicPrediction`."""

from surpyval.degradation.path_models import LinearPath
from surpyval.univariate.parametric import Weibull

from .dynamic import DynamicPrediction
from .joint_aft import JointAFT, JointAFTFitter
from .joint_model import JointModel
from .joint_ph import JointPH, JointPHFitter

WeibullJointPH = JointPH(Weibull, LinearPath)
WeibullJointAFT = JointAFT(Weibull, LinearPath)

__all__ = [
    "DynamicPrediction",
    "JointAFT",
    "JointAFTFitter",
    "JointModel",
    "JointPH",
    "JointPHFitter",
    "WeibullJointAFT",
    "WeibullJointPH",
]
