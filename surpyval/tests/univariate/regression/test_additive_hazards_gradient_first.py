"""Parametric additive hazards fits take the exact gradient first (#515).

``AdditiveHazardsFitter.fit`` ran Nelder-Mead and then TNC on finite
differences: 2.3 s for a WeibullAH at 10 000 rows, against 0.36 s for the
WeibullPH. It now runs BFGS on the likelihood's autograd gradient from the
start (inside the valid region) and keeps that answer when it is a verified
maximum, as the AFT and PO fits do (#499). A fit on the positivity boundary
or with no maximum still ends on the derivative-free search, with its
warnings (#376, #392).
"""

import warnings
from unittest import mock

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.regression.additive_hazards import (
    additive_hazards_fitter,
)


def _positive_data():
    rng = np.random.default_rng(2)
    n = 400
    Z = rng.uniform(0, 1, (n, 2))
    x = rng.exponential(1 / (0.05 + Z @ [0.05, 0.02]))
    c = (x > 15).astype(int)
    return dict(x=np.minimum(x, 15), Z=Z, c=c)


def _doc_data():
    rng = np.random.RandomState(1)
    Z = rng.binomial(1, 0.5, 100).reshape(-1, 1)
    x = sp.Weibull.qf(rng.uniform(size=100), 10, 2) * np.exp(-0.5 * Z[:, 0])
    return dict(x=x, Z=Z, c=np.zeros(100))


# The negative log-likelihood the Nelder-Mead-then-TNC search reached
# (before #515).
_OLD_NEG_LL = {
    ("positive", "WeibullAH"): 974.2367457938165,
    ("positive", "ExponentialAH"): 974.3385172069866,
    ("positive", "GammaAH"): 974.2940760703287,
    ("positive", "LogNormalAH"): 977.2681312431007,
}


@pytest.mark.parametrize("case,name", sorted(_OLD_NEG_LL))
def test_reaches_at_least_the_old_searchs_optimum(case, name):
    model = getattr(sp, name).fit(**_positive_data())
    assert model._neg_ll <= _OLD_NEG_LL[(case, name)] + 1e-6


@pytest.mark.parametrize("name", ["WeibullAH", "ExponentialAH", "GammaAH"])
def test_no_nelder_mead_when_the_gradient_search_converges(name):
    methods = []
    real = additive_hazards_fitter.minimize

    def recording(*args, **kwargs):
        methods.append(kwargs.get("method"))
        return real(*args, **kwargs)

    with mock.patch.object(additive_hazards_fitter, "minimize", recording):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = getattr(sp, name).fit(**_doc_data())
    assert np.isfinite(model._neg_ll)
    assert "Nelder-Mead" not in methods, methods


def test_the_gradient_search_has_a_budget():
    # On a likelihood with no maximum the line searches chase the runaway
    # coefficient with thousands of gradients; past the budget the fit
    # falls back to its derivative-free search, and still warns once.
    rng = np.random.default_rng(2)
    Z = np.r_[np.zeros(50), np.ones(20)].reshape(-1, 1)
    x = np.r_[rng.weibull(1.5, 50) * 5, np.full(20, 6.0)]
    c = np.r_[np.zeros(50), np.ones(20)]
    calls = []
    real = additive_hazards_fitter.AdditiveHazardsFitter._gradient_first

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out[1])
        return out

    with mock.patch.object(
        additive_hazards_fitter.AdditiveHazardsFitter,
        "_gradient_first",
        staticmethod(spy),
    ):
        with pytest.warns(UserWarning) as record:
            sp.ExponentialAH.fit(x=x, Z=Z, c=c)
    assert calls == [False]
    assert len(record) == 1
    assert "No finite maximum" in str(record[0].message)
