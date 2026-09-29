"""IFM with a non-parametric margin: the semi-parametric pseudo-likelihood
(Genest, Ghoudi and Rivest, 1995)."""

import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval.multivariate import Clayton


@pytest.fixture(scope="module")
def sample():
    truth = Clayton.from_params(
        2.0,
        margins=[
            surv.Weibull.from_params([10.0, 2.0]),
            surv.LogNormal.from_params([2.5, 0.5]),
        ],
    )
    return truth.random(2000, random_state=1)


@pytest.mark.parametrize(
    "margins",
    [
        [surv.KaplanMeier, surv.KaplanMeier],
        [surv.Weibull, surv.KaplanMeier],
    ],
)
def test_ifm_accepts_a_nonparametric_margin(sample, margins):
    parametric = Clayton.fit(sample, margins=[surv.Weibull, surv.LogNormal])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = Clayton.fit(sample, margins=margins)
        aic = model.aic()
    # rank-based and parametric margins agree closely on a large sample
    assert model.params[0] == pytest.approx(parametric.params[0], abs=0.05)
    assert np.isfinite(aic)
    # the criteria count only the estimated parametric parameters
    n_parametric = sum(m is surv.Weibull for m in margins) * 2
    assert model.k == 1 + n_parametric


def test_ifm_with_nonparametric_margins_handles_censoring(sample):
    thresholds = np.array([14.0, 16.0])
    c = (sample > thresholds).astype(int)
    x = np.minimum(sample, thresholds)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = Clayton.fit(
            x, c=c, margins=[surv.KaplanMeier, surv.KaplanMeier]
        )
    assert model.params[0] == pytest.approx(2.0, abs=0.2)


def test_mle_refuses_a_nonparametric_margin(sample):
    with pytest.raises(ValueError, match="parametric"):
        Clayton.fit(
            sample, margins=[surv.KaplanMeier, surv.Weibull], how="MLE"
        )
