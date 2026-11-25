"""The pytensor likelihood of the regression fitters against SurPyval's
own.

At fixed parameters, ``log_likelihood(fitter, ..., Z=...)`` must equal
minus the fitter's ``neg_ll`` (what ``fit`` minimises) for proportional
hazards, accelerated failure time and accelerated life models, every row
type and every truncation pattern, to 1e-10; its gradient must agree
with a central difference of ``neg_ll``. Only pytensor is needed; the
module is skipped where it is not installed (PyMC is not a test
dependency of SurPyval).
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.life_models import Eyring, Linear, Power

pt = pytest.importorskip("pytensor.tensor")
pytensor = pytest.importorskip("pytensor")

from surpyval.pymc import REGRESSION, log_likelihood  # noqa: E402

X = [2.0, 5.0, [6.0, 10.0], 9.0, 14.0, [22.0, 27.0], 18.0, 4.5]
C = [0, 1, 2, -1, 0, 2, 1, -1]
N = [1, 3, 2, 1, 4, 1, 2, 5]
Z = np.c_[
    np.linspace(-1.0, 1.0, 8), np.r_[0.3, -1.1, 0.4, 2.0, -0.2, 0.9, -0.6, 0.1]
]
STRESS = np.linspace(20.0, 40.0, 8)[:, None]
COEFS = {"coef_0": 0.3, "coef_1": -0.2}

LINEAR = {
    "WeibullPH": (sp.WeibullPH, {"alpha": 12.0, "beta": 1.6}),
    "LogNormalPH": (sp.LogNormalPH, {"mu": 2.3, "sigma": 0.7}),
    "ExponentialPH": (sp.ExponentialPH, {"failure_rate": 0.08}),
    "WeibullAFT": (sp.WeibullAFT, {"alpha": 12.0, "beta": 1.6}),
    "GammaAFT": (sp.AFT(sp.Gamma), {"alpha": 2.5, "beta": 0.2}),
    "LogLogisticAFT": (sp.AFT(sp.LogLogistic), {"alpha": 10.0, "beta": 2.5}),
}
LIFE = {
    "Weibull-Power": (
        sp.AcceleratedLife(sp.Weibull, Power),
        {"beta": 1.6},
        {"a": 2000.0, "n": -1.0},
    ),
    "LogNormal-Power": (
        sp.AcceleratedLife(sp.LogNormal, Power),
        {"sigma": 0.6},
        {"a": 2000.0, "n": -1.0},
    ),
    "Exponential-Power": (
        sp.AcceleratedLife(sp.Exponential, Power),
        {},
        {"a": 2000.0, "n": -1.0},
    ),
    "Gamma-Power": (
        sp.AcceleratedLife(sp.Gamma, Power),
        {"alpha": 2.5},
        {"a": 200.0, "n": -1.0},
    ),
    "Weibull-Eyring": (
        sp.AcceleratedLife(sp.Weibull, Eyring),
        {"beta": 1.6},
        {"a": 4000.0, "b": -4.0},
    ),
    "Weibull-Linear": (
        sp.AcceleratedLife(sp.Weibull, Linear),
        {"beta": 1.6},
        {"a": 40.0, "b": -0.5},
    ),
}
TRUNCATION = {"none": {}, "left": {"tl": 1.0}, "both": {"tl": 1.0, "tr": 30.0}}


def _names(fitter, Zm):
    return list(fitter.fit(np.linspace(5.0, 30.0, 8), Zm).parameter_names)


def _reference(fitter, Zm, values, trunc):
    data = sp.SurpyvalData(x=X, c=C, n=N, Z=Zm, group_and_sort=False, **trunc)
    params = [values.get(k, 1.0) for k in _names(fitter, Zm)]
    return -float(fitter.neg_ll(data, *params))


def test_the_regression_kinds():
    assert REGRESSION == (
        "proportional hazards",
        "accelerated failure time",
        "accelerated life",
    )


@pytest.mark.parametrize("trunc", list(TRUNCATION))
@pytest.mark.parametrize("case", list(LINEAR))
def test_linear_predictor_models_match_neg_ll(case, trunc):
    fitter, base = LINEAR[case]
    kw = TRUNCATION[trunc]
    data = sp.SurpyvalData(x=X, c=C, n=N, group_and_sort=False, **kw)
    got = float(log_likelihood(fitter, data, Z=Z, **base, **COEFS).eval())
    want = _reference(fitter, Z, {**base, **COEFS}, kw)
    assert got == pytest.approx(want, rel=1e-10, abs=1e-10)
    # the coefficients as one vector are the same term
    vec = float(
        log_likelihood(
            fitter, data, Z=Z, coefs=np.array([0.3, -0.2]), **base
        ).eval()
    )
    assert vec == pytest.approx(got, rel=1e-12)


@pytest.mark.parametrize("trunc", list(TRUNCATION))
@pytest.mark.parametrize("case", list(LIFE))
def test_accelerated_life_models_match_neg_ll(case, trunc):
    fitter, base, life = LIFE[case]
    kw = TRUNCATION[trunc]
    data = sp.SurpyvalData(x=X, c=C, n=N, group_and_sort=False, **kw)
    got = float(
        log_likelihood(fitter, data, Z=STRESS, life=life, **base).eval()
    )
    want = _reference(fitter, STRESS, {**base, **life}, kw)
    assert got == pytest.approx(want, rel=1e-10, abs=1e-10)


def test_covariates_held_by_the_data():
    fitter, base = LINEAR["WeibullPH"]
    held = sp.SurpyvalData(x=X, c=C, n=N, Z=Z, group_and_sort=False)
    got = float(log_likelihood(fitter, held, **base, **COEFS).eval())
    assert got == pytest.approx(_reference(fitter, Z, {**base, **COEFS}, {}))


def test_the_gradient_matches_a_central_difference():
    fitter = sp.WeibullPH
    names = ["alpha", "beta", "coef_0", "coef_1"]
    v = pt.dvector("v")
    ll = log_likelihood(
        fitter,
        x=X,
        c=C,
        n=N,
        tl=1.0,
        Z=Z,
        alpha=v[0],
        beta=v[1],
        coefs=v[2:],
    )
    grad = pytensor.function([v], pt.grad(ll, v))
    at = np.array([12.0, 1.6, 0.3, -0.2])
    got = grad(at)
    data = sp.SurpyvalData(x=X, c=C, n=N, Z=Z, tl=1.0, group_and_sort=False)
    want = np.empty(4)
    for j in range(4):
        h = np.zeros(4)
        h[j] = 1e-6 * max(abs(at[j]), 1.0)
        up = -float(fitter.neg_ll(data, *(at + h)))
        down = -float(fitter.neg_ll(data, *(at - h)))
        want[j] = (up - down) / (2 * h[j])
    np.testing.assert_allclose(got, want, rtol=1e-5, err_msg=str(names))


def test_refusals():
    fitter, base = LINEAR["WeibullPH"]
    with pytest.raises(TypeError, match="needs the covariates, Z="):
        log_likelihood(fitter, x=X, c=C, **base, **COEFS)
    with pytest.raises(ValueError, match="one row per data row"):
        log_likelihood(fitter, x=X, c=C, Z=Z[:3], **base, **COEFS)
    with pytest.raises(TypeError, match="missing coef_1"):
        log_likelihood(fitter, x=X, c=C, Z=Z, coef_0=0.1, **base)
    with pytest.raises(TypeError, match="not both"):
        log_likelihood(
            fitter, x=X, c=C, Z=Z, coefs=[0.1, 0.2], coef_0=0.1, **base
        )
    with pytest.raises(ValueError, match="scalar"):
        log_likelihood(
            fitter, x=X, c=C, Z=Z, alpha=np.ones(8), beta=1.5, **COEFS
        )
    with pytest.raises(TypeError, match="Z= is for a regression fitter"):
        log_likelihood(sp.Weibull, x=X, c=C, Z=Z, alpha=10.0, beta=2.0)
    life_fitter, base, life = LIFE["Weibull-Power"]
    with pytest.raises(TypeError, match="as a dict, life="):
        log_likelihood(life_fitter, x=X, c=C, Z=STRESS, **base)
    with pytest.raises(TypeError, match="is missing n"):
        log_likelihood(
            life_fitter, x=X, c=C, Z=STRESS, life={"a": 1.0}, **base
        )
    with pytest.raises(ValueError, match="no PyMC likelihood"):
        log_likelihood(
            sp.AFT(sp.ExpoWeibull),
            x=X,
            c=C,
            Z=Z,
            alpha=1.0,
            beta=1.0,
            mu=1.0,
            **COEFS,
        )
