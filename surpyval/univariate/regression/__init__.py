from typing import Any as _Any

from .accelerated_failure_time import (
    AFT,
    AFTFitter,
    ExponentialAFT,
    GammaAFT,
    GumbelAFT,
    LogisticAFT,
    LogNormalAFT,
    NormalAFT,
    WeibullAFT,
)
from .accelerated_life import (
    AcceleratedLife,
    DualExponential,
    DualPower,
    ExponentialLifeModel,
    Eyring,
    GeneralLinear,
    GeneralLogLinear,
    InverseExponential,
    InverseEyring,
    InversePower,
    LifeModel,
    Linear,
    ParameterSubstitutionFitter,
    Power,
    PowerExponential,
)
from .additive_hazards import (
    AH,
    AalenAdditive,
    AalenAdditiveModel,
    AdditiveHazards,
    AdditiveHazardsFitter,
    AdditiveHazardsModel,
    ExponentialAH,
    GammaAH,
    GumbelAH,
    LogisticAH,
    LogNormalAH,
    NormalAH,
    WeibullAH,
)
from .buckley_james import BuckleyJames, BuckleyJamesModel
from .cox_aalen import CoxAalen, CoxAalenModel
from .frailty import (
    CoxFrailty,
    CoxFrailtyFitter,
    CoxFrailtyModel,
    CoxMixedEffects,
    CoxMixedEffectsFitter,
    CoxMixedEffectsModel,
    ExponentialFrailty,
    Frailty,
    FrailtyFitter,
    FrailtyModel,
    GammaFrailty,
    LogNormalFrailty,
    RandomEffect,
    WeibullFrailty,
)
from .life_at import LifeAtCovariates
from .parametric_regression_model import ParametricRegressionModel
from .penalty_path import PenaltyPath, penalty_path
from .plot_coefficients import CoefficientPlot, plot_coefficients
from .proportional_hazards import (
    PH,
    CoxNPMLEModel,
    CoxPH,
    ExponentialPH,
    GammaPH,
    GumbelPH,
    IntervalCoxPH,
    IntervalCoxPHModel,
    LogisticPH,
    LogNormalPH,
    NormalPH,
    ProportionalHazardsFitter,
    WeibullPH,
)
from .proportional_odds import (
    PO,
    ExponentialPO,
    GammaPO,
    GumbelPO,
    LogisticPO,
    LogNormalPO,
    NormalPO,
    ProportionalOdds,
    ProportionalOddsFitter,
    ProportionalOddsModel,
    WeibullPO,
)
from .quantile import QuantileRegression, QuantileRegressionModel
from .royston_parmar_regression import (
    RoystonParmarPH,
    RoystonParmarPO,
    RoystonParmarRegression,
)
from .semi_parametric_regression_model import SemiParametricRegressionModel
from .tvc_path import CovariatePath
from .tvc_schedule import StepSchedule, StepValuedError

# The distributional, mixture-cure and zero-inflated regressions' base
# fitters and fitted models, and the pseudo-observation regression's
# result: their home for type hints and ``isinstance`` since v0.29, when
# they left the top level of ``surpyval``. Imported on first use, as
# ``surpyval`` imports their fitters (``surpyval.univariate.pseudo``
# imports competing risks, which imports this package).
_ON_FIRST_USE = {
    **dict.fromkeys(
        (
            "DistributionalFitter",
            "DistributionalRegressionModel",
            "MixtureCureFitter",
            "MixtureCureModel",
            "PromotionTimeCureFitter",
            "PromotionTimeCureModel",
            "ZeroInflatedFitter",
            "ZeroInflatedModel",
        ),
        "surpyval.univariate.regression.distributional",
    ),
    "PseudoRegressionResult": "surpyval.univariate.pseudo",
}


def __getattr__(name: str) -> _Any:
    if name in _ON_FIRST_USE:
        from importlib import import_module

        value = getattr(import_module(_ON_FIRST_USE[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ON_FIRST_USE))


__all__ = [
    # The base fitters and fitted models reached on first use
    *sorted(_ON_FIRST_USE),
    # The forest (coefficient) plot of any regression model
    "plot_coefficients",
    "CoefficientPlot",
    # The coefficient path of a penalised fit, and the penalty by AIC/BIC
    "penalty_path",
    "PenaltyPath",
    # Result classes that hold to_dict / from_dict / to_json / from_json
    "ParametricRegressionModel",
    "SemiParametricRegressionModel",
    "ProportionalOddsModel",
    "IntervalCoxPHModel",
    "CoxNPMLEModel",
    "FrailtyModel",
    # A regression's life at fixed covariates, model.at(Z)
    "LifeAtCovariates",
    "CoxFrailty",
    "CoxFrailtyFitter",
    "CoxFrailtyModel",
    "CoxMixedEffects",
    "CoxMixedEffectsFitter",
    "CoxMixedEffectsModel",
    # Shared-frailty proportional hazards (Gamma frailty)
    "Frailty",
    "FrailtyFitter",
    "ExponentialFrailty",
    "GammaFrailty",
    "LogNormalFrailty",
    "WeibullFrailty",
    "RandomEffect",
    # Time-varying-covariate paths for sf_tvc evaluation: steps, continuous
    "StepSchedule",
    "CovariatePath",
    "StepValuedError",
    # Buckley-James semi-parametric AFT
    "BuckleyJames",
    "BuckleyJamesModel",
    # The Cox-Aalen model (multiplicative terms on an additive baseline)
    "CoxAalen",
    "CoxAalenModel",
    # Censored quantile regression (Peng-Huang, Portnoy)
    "QuantileRegression",
    "QuantileRegressionModel",
    # Additive hazards — non-parametric (Aalen), semi-parametric
    # (Lin-Ying) and parametric
    "AH",
    "AalenAdditive",
    "AalenAdditiveModel",
    "AdditiveHazards",
    "AdditiveHazardsFitter",
    "AdditiveHazardsModel",
    "ExponentialAH",
    "GammaAH",
    "GumbelAH",
    "LogisticAH",
    "LogNormalAH",
    "NormalAH",
    "WeibullAH",
    # Accelerated life (parameter-substitution life models)
    "AcceleratedLife",
    "DualExponential",
    "DualPower",
    "ExponentialLifeModel",
    "Eyring",
    "GeneralLinear",
    "GeneralLogLinear",
    "InverseExponential",
    "InverseEyring",
    "InversePower",
    "LifeModel",
    "Linear",
    "ParameterSubstitutionFitter",
    "Power",
    "PowerExponential",
    # AFT
    "AFT",
    "AFTFitter",
    "ExponentialAFT",
    "GammaAFT",
    "GumbelAFT",
    "LogisticAFT",
    "LogNormalAFT",
    "NormalAFT",
    "WeibullAFT",
    # PH
    "CoxPH",
    "ExponentialPH",
    "GammaPH",
    "GumbelPH",
    "LogisticPH",
    "LogNormalPH",
    "NormalPH",
    "PH",
    "ProportionalHazardsFitter",
    "WeibullPH",
    # Royston-Parmar (spline baseline) regression
    "RoystonParmarPH",
    "RoystonParmarPO",
    "RoystonParmarRegression",
    # PO — semi-parametric (NPMLE) and parametric
    "ExponentialPO",
    "GammaPO",
    "GumbelPO",
    "LogisticPO",
    "LogNormalPO",
    "NormalPO",
    "PO",
    "ProportionalOdds",
    "ProportionalOddsFitter",
    "WeibullPO",
    # Cox for interval-censored data (NPMLE baseline)
    "IntervalCoxPH",
]
