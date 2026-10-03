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
    # the 20 before the polish from each of the two starts (#582), well
    # under a second each on a quiet machine. Counted rather than timed,
    # which a loaded machine would make flaky.
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
    assert len(steps) <= 2 * 20
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
    model._expectation()
    p = model.params.ravel() * np.array([1.1, 0.9, 1.05, 0.95])
    jac = model._Q_jac()
    assert jac is not None
    step = 1e-6 * np.abs(p)
    fd = [
        (
            model._Q(p + np.eye(4)[j] * step[j])
            - model._Q(p - np.eye(4)[j] * step[j])
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
        model._initialise_params()
        reason = model._em(max_iter=4, budget=2)
    # ``_em`` says why; the fit gives the one warning (unless the
    # likelihood has no finite maximum, which says so instead)
    assert caught == []
    assert reason == "EM reached its iteration limit"
    # The same through ``fit``. SQUAREM runs EM on (#589): plain EM's run
    # to 1000 iterations was this test's 21 s, and the run-on is the
    # same code either way (the plain one is ``_em`` above).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = sp.MixtureModel.fit(
            x, c=c, dist=sp.Weibull, m=2, em="squarem"
        )
    assert len(caught) == 1
    assert "did not reach a verified maximum" in str(caught[0].message)
    assert fitted.maximum == "unverified"


def _warranty_counts(seed):
    # #582: Nevada-chart returns of 24 monthly shipment cohorts (about
    # 125k units) as monthly interval counts, each cohort censored at its
    # own age. Truth: 3% defectives Weibull(2, 0.7) plus wear-out
    # Weibull(120, 3).
    rng = np.random.default_rng(seed)
    x, c, n = [], [], []
    for k, shipped in enumerate(rng.integers(3500, 6500, 24), start=1):
        age = 24 - k + 1
        bad = rng.random(shipped) < 0.03
        life = np.where(
            bad, 2 * rng.weibull(0.7, shipped), 120 * rng.weibull(3, shipped)
        )
        month = np.ceil(life)
        for j in range(1, age + 1):
            r = int((month == j).sum())
            if r:
                x.append([j - 1, j])
                c.append(2)
                n.append(r)
        x.append([age, age])
        c.append(1)
        n.append(int((month > age).sum()))
    return np.array(x, float), np.array(c), np.array(n)


def _direct_neg_ll(x, c, n, w, params):
    # The mixture likelihood written out, independently of MixtureModel
    def F(t):
        (a1, b1), (a2, b2) = params
        return w[0] * (1 - np.exp(-((t / a1) ** b1))) + w[1] * (
            1 - np.exp(-((t / a2) ** b2))
        )

    like = np.where(c == 2, F(x[:, 1]) - F(x[:, 0]), 1 - F(x[:, 0]))
    return -np.sum(n * np.log(like))


@pytest.mark.parametrize(
    "seed, neg_ll",
    # The issue's seed, where EM ended 101 units short (23010.17, 26%
    # defective), and one where fixing the gradient alone left the fit
    # 76 units short, on a ridge with the wear-out component's beta at
    # 2222 (the second start finds the maximum).
    [(21, 22909.0735), (7, 23149.3417)],
)
def test_582_warranty_counts_reach_the_maximum(seed, neg_ll):
    x, c, n = _warranty_counts(seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sp.MixtureModel.fit(x=x, c=c, n=n, dist=sp.Weibull, m=2)
    assert not caught, [str(w.message) for w in caught]
    assert model.maximum == "verified"
    # The maximum found by Nelder-Mead on the written-out likelihood from
    # the truth (the issue's check), to its precision
    assert model.neg_ll() == pytest.approx(neg_ll, abs=1e-3)
    assert _direct_neg_ll(x, c, n, model.w, model.params) == pytest.approx(
        model.neg_ll(), rel=1e-12
    )
    defective = int(np.argmin(model.w))
    assert model.w[defective] == pytest.approx(0.03, abs=0.005)
    assert model.params[defective, 1] < 1 < model.params[1 - defective, 1]


def test_582_gradient_is_finite_on_an_interval_from_zero():
    # A Weibull's (0 / alpha) ** beta has a NaN gradient in alpha for
    # beta < 1, which stopped the polish after one evaluation: the
    # interval rows' CDF at a lower end of 0 is now taken as 0.
    from autograd import grad

    x, c, n = _warranty_counts(21)
    model = sp.MixtureModel(dist=sp.Weibull, m=2)
    model.data = sp.utils.surpyval_data.SurpyvalData(x=x, c=c, n=n)
    model._truncated = False
    w = np.array([0.0289, 0.9711])
    params = np.array([[1.811, 0.699], [175.191, 2.373]])

    def fun(theta):
        return model.neg_ll_of(*model._unpack(theta))

    theta = model._pack(w, params)
    assert np.all(np.isfinite(grad(fun)(theta)))
    assert fun(theta) == pytest.approx(
        _direct_neg_ll(x, c, n, w, params), rel=1e-12
    )


def _count_em(monkeypatch):
    steps = []
    em = sp.MixtureModel.EM

    def counted(self):
        steps.append(1)
        return em(self)

    monkeypatch.setattr(sp.MixtureModel, "EM", counted)
    return steps


def test_589_squarem_runs_on_to_the_maximum_in_few_iterations(monkeypatch):
    # With the polish failing, plain EM ran all 1000 iterations and
    # stalled at 738.047067, 1.3e-4 below the maximum (738.046941); its
    # M-steps stop at scipy's tolerances, which near the maximum wander by
    # more than EM's own progress. SQUAREM, on full-precision M-steps,
    # reaches the maximum and stops.
    x, c = _censored_mixture()
    monkeypatch.setattr(sp.MixtureModel, "_polish", lambda self: False)
    steps = _count_em(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.MixtureModel.fit(x, c=c, dist=sp.Weibull, m=2, em="squarem")
    assert len(steps) < 200
    assert model.neg_ll() == pytest.approx(738.046941, abs=2e-6)


def test_589_squarem_changes_nothing_where_the_short_run_verifies():
    # The first 20 iterations are plain EM either way, so a fit whose
    # polish verifies there is the same fit, to the bit.
    x, c = _censored_mixture()
    plain = sp.MixtureModel.fit(x, c=c, dist=sp.Weibull, m=2)
    fast = sp.MixtureModel.fit(x, c=c, dist=sp.Weibull, m=2, em="squarem")
    assert plain.maximum == fast.maximum == "verified"
    np.testing.assert_array_equal(fast.params, plain.params)
    np.testing.assert_array_equal(fast.w, plain.w)


def test_589_m_step_evaluates_q_only_with_its_gradient(monkeypatch):
    # scipy asked for Q and its gradient separately, so Q ran twice at
    # every point of the M-step's search (once plain, once traced by
    # autograd), and once more at the start; it now runs once, traced.
    from autograd.tracer import Box

    x, c = _censored_mixture()
    model = sp.MixtureModel.fit(x, c=c, dist=sp.Weibull, m=2)
    plain_calls = []
    q = sp.MixtureModel._Q

    def recorded(self, params):
        if not isinstance(params, Box):
            plain_calls.append(1)
        return q(self, params)

    monkeypatch.setattr(sp.MixtureModel, "Q", recorded)
    model._expectation()
    model._Q_jac()  # the one-off check that autograd can differentiate it
    plain_calls.clear()
    model._maximisation()
    assert plain_calls == []


def test_589_em_option_is_checked():
    x, c = _censored_mixture()
    with pytest.raises(ValueError, match="'em' must be one of"):
        sp.MixtureModel.fit(x, c=c, dist=sp.Weibull, m=2, em="fast")
