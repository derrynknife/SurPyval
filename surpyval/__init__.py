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
    weibayes,
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

# The regression, competing-risks, recurrent-event and degradation models
# and the metrics are importable directly from `surpyval` too, but are
# imported on first use (PEP 562), with the pandas, formulaic and
# scipy.stats they need: a program that only fits and evaluates
# distributions does not pay for them (#470). They also stay in their
# packages, which is where the helper functions and result types live
# (`surpyval.recurrent.laplace`, `TrendTestResult`, ...) and the
# generically named copulas (`surpyval.multivariate.Gaussian`, `Frank`,
# ...). Competing risks lives under each paradigm it applies to:
# `surpyval.univariate.competing_risks` and
# `surpyval.recurrent.competing_risks`. Pre-stable models are tiered by
# maturity: `surpyval.beta` (functionally complete, interface not yet
# stable -- the survival tree and random survival forest in
# `surpyval.beta.ml`) and `surpyval.alpha` (exploratory).
_LAZY = {
    **dict.fromkeys(
        (
            "AFT",
            "AFTFitter",
            "AH",
            "AcceleratedLife",
            "AdditiveHazards",
            "AdditiveHazardsFitter",
            "AdditiveHazardsModel",
            "BuckleyJames",
            "BuckleyJamesModel",
            "CovariatePath",
            "CoxPH",
            "DualExponential",
            "DualPower",
            "ExponentialAFT",
            "ExponentialAH",
            "ExponentialFrailty",
            "ExponentialLifeModel",
            "ExponentialPH",
            "ExponentialPO",
            "Eyring",
            "Frailty",
            "FrailtyFitter",
            "FrailtyModel",
            "GammaAFT",
            "GammaAH",
            "GammaFrailty",
            "GammaPH",
            "GammaPO",
            "GeneralLogLinear",
            "GumbelAFT",
            "GumbelAH",
            "GumbelPH",
            "GumbelPO",
            "InverseExponential",
            "InverseEyring",
            "InversePower",
            "LifeModel",
            "Linear",
            "LogNormalAFT",
            "LogNormalAH",
            "LogNormalFrailty",
            "LogNormalPH",
            "LogNormalPO",
            "LogisticAFT",
            "LogisticAH",
            "LogisticPH",
            "LogisticPO",
            "NormalAFT",
            "NormalAH",
            "NormalPH",
            "NormalPO",
            "PH",
            "PO",
            "ParameterSubstitutionFitter",
            "ParametricRegressionModel",
            "Power",
            "PowerExponential",
            "ProportionalHazardsFitter",
            "ProportionalOddsFitter",
            "SemiParametricRegressionModel",
            "StepSchedule",
            "StepValuedError",
            "WeibullAFT",
            "WeibullAH",
            "WeibullFrailty",
            "WeibullPH",
            "WeibullPO",
        ),
        "surpyval.univariate.regression",
    ),
    **dict.fromkeys(
        (
            "CompetingRisks",
            "CompetingRisksProportionalHazards",
            "FineGray",
            "ParametricCompetingRisks",
            "gray_test",
        ),
        "surpyval.univariate.competing_risks",
    ),
    **dict.fromkeys(
        (
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
        ),
        "surpyval.recurrent",
    ),
    **dict.fromkeys(
        (
            "DegradationAnalysis",
            "DestructiveDegradation",
            "GammaProcess",
            "WienerProcess",
        ),
        "surpyval.degradation",
    ),
    **dict.fromkeys(
        (
            "auc_td",
            "brier_score",
            "concordance_index",
            "integrated_brier_score",
            "survival_probability",
        ),
        "surpyval.metrics",
    ),
}

if TYPE_CHECKING:
    from surpyval import degradation, metrics, recurrent
    from surpyval.degradation import (
        DegradationAnalysis,
        DestructiveDegradation,
        GammaProcess,
        WienerProcess,
    )
    from surpyval.metrics import (
        auc_td,
        brier_score,
        concordance_index,
        integrated_brier_score,
        survival_probability,
    )
    from surpyval.recurrent import (
        ARA,
        ARI,
        HPP,
        CauseSpecificMCF,
        CauseSpecificNHPP,
        CoxLewis,
        CrowAMSAA,
        Duane,
        GeneralizedOneRenewal,
        GeneralizedRenewal,
        NonParametricCounting,
        ProportionalIntensityHPP,
        ProportionalIntensityNHPP,
    )
    from surpyval.univariate.competing_risks import (
        CompetingRisks,
        CompetingRisksProportionalHazards,
        FineGray,
        ParametricCompetingRisks,
        gray_test,
    )
    from surpyval.univariate.regression import *  # noqa: F401,F403

# The subpackages ``import surpyval`` used to import, so that
# ``surpyval.recurrent.laplace`` works without an import of its own.
_SUBPACKAGES = ("degradation", "metrics", "recurrent")

# Names that live only in a subpackage: asking for one here
# (``surpyval.laplace``) says where it is, rather than only that
# it is missing (#485). The subpackages are not imported to find out.
_ELSEWHERE = {
    **dict.fromkeys(
        ["TrendTestResult", "laplace", "mil_hdbk_189c"], "surpyval.recurrent"
    ),
    **dict.fromkeys(
        ["Clayton", "Copula", "Frank", "Gaussian", "Independence"],
        "surpyval.multivariate",
    ),
}

if not TYPE_CHECKING:  # keep the type checker's view of the module exact

    def __getattr__(name: str) -> Any:
        from importlib import import_module

        if name in _LAZY:
            value = getattr(import_module(_LAZY[name]), name)
            globals()[name] = value
            return value
        if name in _SUBPACKAGES:
            return import_module(f"surpyval.{name}")
        if name in _ELSEWHERE:
            raise AttributeError(
                "module 'surpyval' has no attribute {n!r}: it is in "
                "{m} (from {m} import {n})".format(n=name, m=_ELSEWHERE[name])
            )
        raise AttributeError(
            "module 'surpyval' has no attribute {!r}".format(name)
        )

    def __dir__() -> list[str]:
        return sorted(set(globals()) | set(_LAZY) | set(_SUBPACKAGES))
