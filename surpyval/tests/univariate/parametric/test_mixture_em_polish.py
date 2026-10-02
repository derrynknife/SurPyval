"""EM polished by direct maximum likelihood (#506).

On a censored two-Weibull mixture EM crawled along a flat direction of
the likelihood for all 1000 iterations and warned "EM algorithm reached
max iterations before converging" at what was already the maximum, after
17 s. After a short EM run the fit now polishes by BFGS on the observed
likelihood with its autograd gradient (the weights through a softmax),
accepts the answer when it is a verified maximum, and warns only when it
is not; the M-step's L-BFGS-B has the autograd gradient too.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize

import surpyval as sp
from surpyval.univariate.parametric.fitters import is_local_minimum


def _censored_mixture(seed=0):
    # The issue's case (#482's data): 25% right censored.
    rng = np.random.default_rng(seed)
    t = np.concatenate([10 * rng.weibull(0.8, 100), 100 * rng.weibull(4, 100)])
    cut = np.quantile(t, 0.75)
    return np.minimum(t, cut), (t > cut).astype(int)


def test_censored_mixture_is_fitted_quickly_without_a_false_alarm(
    monkeypatch,
):
    x, c = _censored_mixture()
    # The time went on EM iterations: all 1000 of them (17 s), now at most
    # the 20 before the polish (well under a second on a quiet machine).
    # Counted rather than timed, which a loaded machine would make flaky.
    steps = []
    em = sp.MixtureModel.EM

    def counted(self):
        steps.append(1)
        return em(self)

    monkeypatch.setattr(sp.MixtureModel, "EM", counted)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sp.MixtureModel.fit(x, c=c, dist=sp.Weibull, m=2)
    assert not caught, [str(w.message) for w in caught]
    assert len(steps) <= 20
    # The maximum is at least as good as EM's iterate was (738.047053),
    # and direct L-BFGS-B from it gains nothing.
    neg_ll = model.neg_ll_of(model.w, model.params)
    assert neg_ll <= 738.047053
    assert neg_ll == pytest.approx(738.046941, abs=1e-6)
    k = model.dist.k

    def obj(z):
        return model.neg_ll_of(np.r_[z[0], 1 - z[0]], z[1:].reshape(2, k))

    with np.errstate(all="ignore"):
        res = minimize(
            obj,
            np.r_[model.w[0], model.params.ravel()],
            bounds=[(1e-9, 1 - 1e-9)] + list(model.dist.bounds) * 2,
        )
    assert neg_ll - res.fun < 1e-6


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_answer_is_a_verified_maximum(seed):
    x, c = _censored_mixture(seed)
    model = sp.MixtureModel.fit(x, c=c, dist=sp.Weibull, m=2)
    theta = model._pack(model.w, model.params)
    from autograd import grad, hessian

    def fun(t):
        w, params = model._unpack(t)
        return model.neg_ll_of(w, params)

    assert is_local_minimum(
        fun, grad(fun), hessian(fun), theta, obj_scale=float(len(x))
    )
    np.testing.assert_allclose(model.w.sum(), 1.0)


def test_m_step_gradient_is_the_exact_one():
    x, c = _censored_mixture()
    model = sp.MixtureModel.fit(x, c=c, dist=sp.Weibull, m=2)
    model.expectation()
    p = model.params.ravel() * np.array([1.1, 0.9, 1.05, 0.95])
    jac = model._Q_jac()
    assert jac is not None
    step = 1e-6 * np.abs(p)
    fd = [
        (
            model.Q(p + np.eye(4)[j] * step[j])
            - model.Q(p - np.eye(4)[j] * step[j])
        )
        / (2 * step[j])
        for j in range(4)
    ]
    np.testing.assert_allclose(jac(p), fd, rtol=1e-5)


def test_pack_and_unpack_are_inverse():
    x, c = _censored_mixture()
    model = sp.MixtureModel.fit(x, c=c, dist=sp.Weibull, m=2)
    w, params = model._unpack(model._pack(model.w, model.params))
    np.testing.assert_allclose(w, model.w, rtol=1e-12)
    np.testing.assert_allclose(params, model.params, rtol=1e-12)


def test_warns_only_when_neither_em_nor_the_polish_reaches_a_maximum(
    monkeypatch,
):
    x, c = _censored_mixture()
    model = sp.MixtureModel(dist=sp.Weibull, m=2)
    monkeypatch.setattr(type(model), "_polish", lambda self: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.data = sp.utils.surpyval_data.SurpyvalData(x=x, c=c)
        model._truncated = False
        model.p = np.ones((2, len(x))) / 2
        model.initialise_params()
        reason = model._em(max_iter=4, budget=2)
    # ``_em`` says why; the fit gives the one warning (unless the
    # likelihood has no finite maximum, which says so instead)
    assert caught == []
    assert reason == "EM reached its iteration limit"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = sp.MixtureModel.fit(x, c=c, dist=sp.Weibull, m=2)
    assert len(caught) == 1
    assert "did not reach a verified maximum" in str(caught[0].message)
    assert fitted.maximum == "unverified"
