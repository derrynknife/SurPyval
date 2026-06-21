__version__ = "0.11.1"

from autograd import numpy as np

from surpyval.distribution import (
    Distribution,
    MultivariateDistribution,
    NonParametricDistribution,
    ParametricDistribution,
)
from surpyval.univariate.nonparametric import (
    FlemingHarrington,
    KaplanMeier,
    LogRankResult,
    NelsonAalen,
    NonParametric,
    Turnbull,
    logrank,
    success_run,
)
from surpyval.univariate.parametric import (
    Bernoulli,
    Beta,
    Beta4,
    Binomial,
    CustomDistribution,
    ExactEventTime,
    Exponential,
    ExpoWeibull,
    FixedEventProbability,
    Galton,
    Gamma,
    Gauss,
    Gumbel,
    GumbelLEV,
    InstantlyOccurs,
    Logistic,
    LogLogistic,
    LogNormal,
    MixtureModel,
    NeverOccurs,
    Normal,
    Parametric,
    Rayleigh,
    Uniform,
    Weibull,
)
from surpyval.utils import (
    fs_to_xcnt,
    fs_to_xrd,
    fsl_to_xcnt,
    fsli_handler,
    fsli_to_xcnt,
    round_sig,
    xcn_to_fs,
    xcnt_handler,
    xcnt_to_xrd,
    xrd_handler,
    xrd_to_xcnt,
)

from .fit_best import fit_best

from surpyval.utils.recurrent_event_data import (  # isort: skip
    RecurrentEventData,
)

# The univariate regression models (CoxPH, WeibullPH, the accelerated
# life models, etc.) are importable directly from `surpyval`. Everything
# else (competing risks, recurrent events, experimental features) is
# imported from its package. Competing risks lives under each paradigm
# it applies to: `surpyval.univariate.competing_risks` and
# `surpyval.recurrent.competing_risks`. Recurrent and experimental
# features live in `surpyval.recurrent` and `surpyval.experimental`.
from surpyval.univariate.regression import *  # isort: skip # noqa: F401,F403,E501

from surpyval.utils.surpyval_data import SurpyvalData  # isort: skip
from surpyval.utils.recurrent_utils import handle_xicn  # isort: skip

NUM = np.float64
TINIEST = np.finfo(np.float64).tiny
EPS = np.sqrt(np.finfo(NUM).eps)
