__version__ = "0.10.8"

from autograd import numpy as np

from surpyval.utils import (
    xcn_sort,
    xcn_handler,
    xcn_to_xrd,
    xrd_to_xcn,
    xcnt_handler,
    xcnt_to_xrd,
    fsli_to_xcn,
    fsl_to_xcn,
    fs_to_xcn,
    fs_to_xrd,
    round_sig,
    xcn_to_fs
)

from surpyval.parametric import (
    Parametric,
    Gumbel,
    Uniform,
    Rayleigh,
    Exponential,
    Weibull,
    ExpoWeibull,
    Normal, Gauss,
    LogNormal, Galton,
    Logistic,
    LogLogistic,
    Gamma,
    Beta,
    Distribution,
    MixtureModel,
    Bernoulli,
    FixedEventProbability,
    ExactEventTime
)

from surpyval.nonparametric import (
    KaplanMeier,
    NelsonAalen,
    FlemingHarrington,
    Turnbull,
    NonParametric
)

from surpyval.regression import (
    CoxPH
)

NUM = np.float64
TINIEST = np.finfo(np.float64).tiny
EPS = np.sqrt(np.finfo(NUM).eps)
