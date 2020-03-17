import autograd.numpy as np

NUM     = np.float64
TINIEST = np.finfo(NUM).tiny
EPS     = np.sqrt(np.finfo(NUM).eps)

from surpyval import nonparametric as nonp
from .surpyval_dist import SurpyvalDist
from .parametric_dist import Parametric

from .weibull import Weibull
from .weibull3p import Weibull3p
from .gumbel import Gumbel
from .exponential import Exponential
from .normal import Normal
from .lognormal import LogNormal
from .gamma import Gamma
from .uniform import Uniform
from .weibull_mix2 import Weibull_Mix_Two
from .wmm import WMM
from .logistic import Logistic
from .loglogistic import LogLogistic
from .mixture_model import MixtureModel

def round_sig(points, sig=2):
    places = sig - np.floor(np.log10(np.abs(points))) - 1
    output = []
    for p, i in zip(points, places):
        output.append(np.round(p, np.int(i)))
    return output