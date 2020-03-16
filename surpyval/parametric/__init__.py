import autograd.numpy as np

NUM     = np.float64
TINIEST = np.finfo(NUM).tiny
EPS     = np.sqrt(np.finfo(NUM).eps)

from surpyval import nonparametric as nonp
from surpyval.parametric.surpyval_dist import SurpyvalDist
from surpyval.parametric.parametric_dist import Parametric

from surpyval.parametric.weibull import Weibull
from surpyval.parametric.weibull3p import Weibull3p
from surpyval.parametric.gumbel import Gumbel
from surpyval.parametric.exponential import Exponential
from surpyval.parametric.normal import Normal
from surpyval.parametric.lognormal import LogNormal
from surpyval.parametric.gamma import Gamma
from surpyval.parametric.uniform import Uniform
from surpyval.parametric.weibull_mix2 import Weibull_Mix_Two
from surpyval.parametric.wmm import WMM
from surpyval.parametric.logistic import Logistic
from surpyval.parametric.loglogistic import LogLogistic

def round_sig(points, sig=2):
    places = sig - np.floor(np.log10(np.abs(points))) - 1
    output = []
    for p, i in zip(points, places):
        output.append(np.round(p, np.int(i)))
    return output