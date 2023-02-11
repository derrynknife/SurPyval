__version__ = "0.10.10"

from autograd import numpy as np

from surpyval.distribution import Distribution
from surpyval.nonparametric import (
    FlemingHarrington,
    KaplanMeier,
    NelsonAalen,
    NonParametric,
    Turnbull,
)
from surpyval.parametric import (
    Bernoulli,
    Beta,
    CustomDistribution,
    ExactEventTime,
    Exponential,
    ExpoWeibull,
    FixedEventProbability,
    Galton,
    Gamma,
    Gauss,
    Gumbel,
    Logistic,
    LogLogistic,
    LogNormal,
    MixtureModel,
    Normal,
    Parametric,
    Rayleigh,
    Uniform,
    Weibull,
)
from surpyval.regression import CoxPH, ExponentialPH, WeibullPH
from surpyval.utils import (
    fs_to_xcn,
    fs_to_xrd,
    fsl_to_xcn,
    fsli_to_xcn,
    round_sig,
    xcn_handler,
    xcn_sort,
    xcn_to_fs,
    xcn_to_xrd,
    xcnt_handler,
    xcnt_to_xrd,
    xrd_to_xcn,
)

from .fit_best import fit_best

NUM = np.float64
TINIEST = np.finfo(np.float64).tiny
EPS = np.sqrt(np.finfo(NUM).eps)
