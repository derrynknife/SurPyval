__version__ = "0.21.0"

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
    rmst_diff,
    success_run,
)
from surpyval.univariate.parametric import (
    Bernoulli,
    Beta,
    Beta4,
    BetaGeometric,
    Binomial,
    CustomDistribution,
    DiscreteWeibull,
    Discretize,
    DiscretizedFitter,
    ExactEventTime,
    Exponential,
    ExpoWeibull,
    FixedEventProbability,
    Galton,
    Gamma,
    Gauss,
    Geometric,
    Gumbel,
    GumbelLEV,
    Hypoexponential,
    InstantlyOccurs,
    Logistic,
    LogLogistic,
    LogNormal,
    MixtureModel,
    NegativeBinomial,
    NeverOccurs,
    Normal,
    Parametric,
    Poisson,
    Rayleigh,
    RoystonParmar,
    RoystonParmarModel,
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
from surpyval.univariate.competing_risks import gray_test  # noqa: E402,F401
from surpyval.metrics import (  # noqa: E402,F401
    auc_td,
    brier_score,
    integrated_brier_score,
    survival_probability,
)

from surpyval.utils.recurrent_event_data import (  # isort: skip
    RecurrentEventData,
)

# The univariate regression models (CoxPH, WeibullPH, the accelerated
# life models, etc.) are importable directly from `surpyval`. Everything
# else (competing risks, recurrent events, pre-stable models) is
# imported from its package. Competing risks lives under each paradigm
# it applies to: `surpyval.univariate.competing_risks` and
# `surpyval.recurrent.competing_risks`; recurrent events live in
# `surpyval.recurrent`. Pre-stable models are tiered by maturity:
# `surpyval.beta` (functionally complete, interface not yet stable --
# the survival tree and random survival forest in `surpyval.beta.ml`)
# and `surpyval.alpha` (exploratory).
from surpyval.univariate.regression import *  # isort: skip # noqa: F401,F403,E501

from surpyval.utils.surpyval_data import SurpyvalData  # isort: skip
from surpyval.utils.recurrent_utils import handle_xicn  # isort: skip

# Package-level readers for serialised models: `surpyval.from_json` /
# `surpyval.from_dict` restore a model of the right class from any
# model's `to_json` file / `to_dict` dictionary.
from surpyval.serialisation import from_dict, from_json  # isort: skip

NUM = np.float64
TINIEST = np.finfo(np.float64).tiny
EPS = np.sqrt(np.finfo(NUM).eps)

from typing import TYPE_CHECKING, Any  # isort: skip # noqa: E402

# Models that live in a subpackage, not at the top level: asking for one
# here (``surpyval.CrowAMSAA``) says where it is, rather than only that
# it is missing (#485). The subpackages are not imported to find out.
_ELSEWHERE = {
    **dict.fromkeys(
        [
            "ARA",
            "ARI",
            "CauseSpecificMCF",
            "CauseSpecificNHPP",
            "CoxLewis",
            "CrowAMSAA",
            "Duane",
            "GeneralizedOneRenewal",
            "GeneralizedRenewal",
            "HPP",
            "NonParametricCounting",
            "ProportionalIntensityHPP",
            "ProportionalIntensityNHPP",
            "laplace",
            "mil_hdbk_189c",
        ],
        "surpyval.recurrent",
    ),
    **dict.fromkeys(
        [
            "CompetingRisks",
            "CompetingRisksProportionalHazards",
            "FineGray",
            "ParametricCompetingRisks",
        ],
        "surpyval.univariate.competing_risks",
    ),
    **dict.fromkeys(
        [
            "DegradationAnalysis",
            "DestructiveDegradation",
            "GammaProcess",
            "WienerProcess",
        ],
        "surpyval.degradation",
    ),
    **dict.fromkeys(
        ["Clayton", "Copula", "Frank", "Gaussian", "Independence"],
        "surpyval.multivariate",
    ),
}

if not TYPE_CHECKING:  # keep the type checker's view of the module exact

    def __getattr__(name: str) -> Any:
        if name in _ELSEWHERE:
            raise AttributeError(
                "module 'surpyval' has no attribute {n!r}: it is in "
                "{m} (from {m} import {n})".format(n=name, m=_ELSEWHERE[name])
            )
        raise AttributeError(
            "module 'surpyval' has no attribute {!r}".format(name)
        )
