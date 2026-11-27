"""Bayesian fits of the mixture-cure regression, the refusal of a
mixture's, and the PyMC bridge's mixture likelihood.

Checked against:

- the maximum-likelihood fit (with flat priors the posterior's peak);
- the cured fraction's posterior against the draws' own values;
- the warnings of poor follow-up (Maller and Zhou's check) and of no
  cured fraction (the maximum-likelihood fit's), and the refusal of a
  flat prior on ``lfp_p`` where the data show none;
- the bridge's mixture log-likelihood against ``MixtureModel``'s own at
  given parameters, to 1e-10, with every row type (exact, right, left,
  interval censored, counts and truncation); its windows are moderate,
  away from the tails, where any accurate way of taking an interval's
  probability agrees;
- coverage of the credible intervals on simulated data is in
  ``tests/calibration/test_coverage_mixture_cure_bayes.py``.
"""

import json
import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.bayes.regression import BayesianMixtureCure


def _cure(T, seed=3, n=150):
    rng = np.random.default_rng(seed)
    Z = rng.binomial(1, 0.5, size=(n, 1)).astype(float)
    cured = rng.uniform(size=n) < np.where(Z[:, 0] > 0, 0.5, 0.25)
    life = sp.Weibull.random(n, 10.0, 2.0, random_state=seed + 1)
    x = np.where(cured, T, np.minimum(life, T))
    c = np.where(cured | (life > T), 1, 0)
    return x, Z, c, life


@pytest.fixture(scope="module")
def fitted():
    x, Z, c, _ = _cure(40.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.MixtureCure(sp.Weibull).fit_bayes(x, Z, c)


def test_a_mixture_cure_fit(fitted):
    assert isinstance(fitted, BayesianMixtureCure)
    assert fitted.method == "laplace"
    assert fitted.parameter_names == [
        "alpha",
        "beta",
        "lfp_p",
        "alpha_coef_0",
        "lfp_p_coef_0",
    ]
    # the coefficients' default: Normal(0, 2.5 / sd), sd 0.5 here; lfp_p's
    # a proper uniform
    assert sorted(fitted.default_priors) == [
        "alpha_coef_0",
        "lfp_p",
        "lfp_p_coef_0",
    ]
    assert "Normal(0, 5)" in fitted.default_priors["lfp_p_coef_0"]
    assert fitted.default_priors["lfp_p"] == "Uniform(0, 1)"
    lo, hi = fitted.param_cri("lfp_p")
    assert 0 < lo < fitted.params[2] < hi < 1


def test_flat_priors_peak_at_the_maximum_likelihood_fit():
    # lfp_p cannot be flat on its logit (no finite mass); its default is
    # uniform on lfp_p itself, flat on its own scale. With the
    # coefficients flat too, the posterior density on the parameters'
    # own scale (lfp_p's, the others' log or identity) peaks at the
    # maximum-likelihood fit.
    from surpyval.bayes.engines.grid import find_mode
    from surpyval.bayes.result import bind, log_posterior

    x, Z, c, _ = _cure(40.0)
    flat = {k: sp.prior.flat() for k in ("alpha_coef_0", "lfp_p_coef_0")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.MixtureCure(sp.Weibull).fit_bayes(x, Z, c, priors=flat)
    bound, _ = bind(model.target, flat)
    j = bound.free_names.index("lfp_p")
    t = bound.transforms[bound.names.index("lfp_p")]
    log_post = log_posterior(model.target, bound)

    def natural(phi):
        phi = np.atleast_2d(phi)
        return log_post(phi) - t.log_jacobian(phi[:, j])

    start = bound.to_phi(model.mle.params[None, :])[0]
    peak, _, _ = find_mode(natural, [start])
    np.testing.assert_allclose(
        bound.full_theta(peak[None, :])[0], model.mle.params, rtol=1e-4
    )


def test_cure_fraction_posterior(fitted):
    values = np.array(
        [
            float(fitted.mle.with_params(p).cure_probability([[1.0]]))
            for p in fitted.posterior.points
        ]
    )
    w = fitted.posterior.weights
    assert fitted.cure_probability([[1.0]]) == pytest.approx(w @ values)
    lo, hi = fitted.cure_cri([[1.0]])
    assert np.sum(w[values < lo]) == pytest.approx(0.025, abs=2e-3)
    assert np.sum(w[values > hi]) == pytest.approx(0.025, abs=2e-3)
    assert lo < float(fitted.mle.cure_probability([[1.0]])) < hi
    assert fitted.cure_cri([[1.0]], bound="upper") == pytest.approx(
        fitted.cure_cri([[1.0]], alpha_ci=0.1)[1]
    )


def test_poor_follow_up_warns():
    x, Z, c, _ = _cure(6.0)
    assert (
        not sp.MixtureCure(sp.Weibull).fit(x, Z, c).follow_up()["sufficient"]
    )
    # it warns, and here (Pareto k 0.75) the Laplace approximation is
    # not to be trusted either: refused
    with pytest.warns(UserWarning, match="too short to identify the cure"):
        with pytest.raises(ValueError, match="cannot be trusted"):
            sp.MixtureCure(sp.Weibull).fit_bayes(x, Z, c)


def test_a_flat_prior_on_lfp_p_is_always_refused():
    # also on data with a clear plateau: the cure model holds the no-cure
    # model as lfp_p goes to 1, so a flat prior on its logit has no
    # finite mass on any data
    x, Z, c, _ = _cure(40.0)
    with pytest.raises(ValueError, match="cannot have a prior flat"):
        sp.MixtureCure(sp.Weibull).fit_bayes(
            x, Z, c, priors={"lfp_p": sp.prior.flat()}
        )


def test_no_cured_fraction_warns():
    _, Z, _, life = _cure(40.0)
    x, c = np.minimum(life, 40.0), (life > 40.0).astype(int)
    with pytest.warns(UserWarning, match="No finite maximum.*lfp_p"):
        model = sp.MixtureCure(sp.Weibull).fit_bayes(
            x, Z, c, priors={"lfp_p": sp.prior.between(0.6, 0.95)}
        )
    assert model.params[2] > 0.6


def test_saved(fitted):
    back = sp.from_dict(json.loads(json.dumps(fitted.to_dict())))
    assert isinstance(back, BayesianMixtureCure)
    assert back.cure_probability([[0.0]]) == pytest.approx(
        fitted.cure_probability([[0.0]]), rel=1e-12
    )


def test_from_a_data_frame(fitted):
    import pandas as pd

    x, Z, c, _ = _cure(40.0)
    df = pd.DataFrame({"t": x, "z": Z[:, 0], "c": c})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.MixtureCure(sp.Weibull).fit_bayes_from_df(
            df, x_col="t", Z_cols=["z"], c_col="c"
        )
    np.testing.assert_allclose(model.params, fitted.params, rtol=1e-9)


@pytest.mark.parametrize(
    "call",
    [
        lambda: sp.MixtureModel.fit_bayes([1.0, 2, 3, 9], dist=sp.Weibull),
        lambda: sp.MixtureModel(sp.Weibull, 2).fit_bayes([1.0, 2, 3, 9]),
    ],
)
def test_a_mixture_refuses_and_says_where_to_go(call):
    with pytest.raises(ValueError, match="surpyval.pymc.mixture_likelihood"):
        call()


# -- the PyMC bridge --------------------

ROWS = dict(
    x=[1.5, 2.7, 4.0, 6.0, [3.0, 5.0], [8.0, 11.0], 9.5, 12.0, 15.0, 3.3],
    c=[0, 0, 1, -1, 2, 2, 0, 1, 0, 0],
    n=[1, 2, 1, 1, 1, 3, 1, 1, 1, 1],
    tl=[0.5, 0, 0, 0, 0, 1.0, 0, 0, 2.0, 0],
    tr=[np.inf] * 8 + [30.0, np.inf],
)


@pytest.mark.parametrize(
    "dist, values",
    [
        (sp.Weibull, [[3.0, 1.5], [12.0, 2.5]]),
        (sp.LogNormal, [[1.0, 0.6], [2.4, 0.4]]),
        (sp.Weibull, [[2.0, 0.8], [6.0, 1.2], [14.0, 3.0]]),
    ],
)
def test_bridge_mixture_likelihood_is_the_mixture_models(dist, values):
    pytest.importorskip("pytensor")
    from surpyval.pymc import mixture_log_likelihood

    values = np.asarray(values)
    m = len(values)
    w = np.linspace(1.0, 2.0, m) / np.linspace(1.0, 2.0, m).sum()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.MixtureModel(dist, m).fit(**ROWS)
    ours = -float(model.neg_ll_of(w, values))
    params = {
        name: values[:, k] for k, name in enumerate(dist.parameter_names)
    }
    term = mixture_log_likelihood(dist, weights=w, **ROWS, **params)
    assert float(term.eval()) == pytest.approx(ours, rel=1e-10, abs=1e-10)


def test_bridge_refusals_and_the_order():
    pm = pytest.importorskip("pymc")
    from surpyval.pymc import mixture_likelihood, mixture_log_likelihood

    with pytest.raises(TypeError, match="every parameter"):
        mixture_log_likelihood(
            sp.Weibull, x=[1.0, 2.0], weights=[0.5, 0.5], alpha=[1.0, 2.0]
        )
    with pytest.raises(ValueError, match="3 values for 2 components"):
        mixture_log_likelihood(
            sp.Weibull,
            x=[1.0, 2.0],
            weights=[0.5, 0.5],
            alpha=[1.0, 2.0],
            beta=[1.0, 2.0, 3.0],
        )
    with pytest.raises(TypeError, match="inside"):
        mixture_likelihood(
            sp.Weibull, x=[1.0], weights=[0.5, 0.5], alpha=[1, 2], beta=[1, 1]
        )
    with pm.Model() as model:
        with pytest.raises(ValueError, match="order='gamma'"):
            mixture_likelihood(
                sp.Weibull,
                x=[1.0],
                weights=[0.5, 0.5],
                alpha=[1.0, 2.0],
                beta=[1.0, 1.0],
                order="gamma",
            )
        mixture_likelihood(
            sp.Weibull,
            x=[1.0, 4.0],
            weights=[0.5, 0.5],
            alpha=[3.0, 2.0],
            beta=[1.0, 1.0],
            name="swapped",
        )
    order = [p for p in model.potentials if p.name == "swapped_order"][0]
    # alpha decreasing: outside the ordered region
    assert float(order.eval()) == -np.inf


# -- where the Laplace approximation cannot be trusted --------------------
#: The coefficients held at 0 (``fixed``), so that the posterior has
#: three free parameters and the grid gives it exactly: the references
#: below are ``method="grid"`` fits with the default priors (integration
#: errors 0.07% and 0.4% of the 95% widths), frozen here because each
#: takes one to two minutes.
FIXED = {"alpha_coef_0": 0.0, "lfp_p_coef_0": 0.0}
GRID = {
    # follow-up 9 (0.9 of the latency's scale), seed 3: Pareto k 0.71
    9.0: {
        "lfp_p": (0.4442, 0.988),
        "alpha": (7.6585, 14.0538),
        "beta": (1.7748, 3.0518),
    },
    # follow-up 20, seed 3: Pareto k 0.68
    20.0: {
        "lfp_p": (0.6212, 0.783),
        "alpha": (9.6059, 12.1911),
        "beta": (1.7684, 2.5663),
    },
}


def test_a_laplace_fit_that_cannot_be_trusted_is_refused():
    # (the normal's 95% interval of lfp_p is 0.245 to 0.924, against the
    # exact 0.444 to 0.988)
    x, Z, c, _ = _cure(9.0)
    with pytest.raises(ValueError, match="cannot be trusted"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp.MixtureCure(sp.Weibull).fit_bayes(x, Z, c, fixed=FIXED)


def test_a_poor_laplace_fit_is_flagged_and_reweighting_corrects_it():
    x, Z, c, _ = _cure(20.0)
    with pytest.warns(UserWarning, match="importance=True"):
        plain = sp.MixtureCure(sp.Weibull).fit_bayes(x, Z, c, fixed=FIXED)
    assert 0.5 < plain.diagnostics["pareto_k"] <= 0.7
    assert plain.diagnostics["approximation"] == "poor"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixed_up = sp.MixtureCure(sp.Weibull).fit_bayes(
            x, Z, c, fixed=FIXED, importance=True
        )
    for name, exact in GRID[20.0].items():
        exact = np.asarray(exact)
        width = exact[1] - exact[0]
        corrected = np.abs(fixed_up.param_cri(name) - exact)
        assert np.all(corrected <= 0.1 * width), name
    lfp = np.asarray(GRID[20.0]["lfp_p"])
    assert np.max(np.abs(fixed_up.param_cri("lfp_p") - lfp)) < np.max(
        np.abs(plain.param_cri("lfp_p") - lfp)
    )


# -- the bridge's default order --------------------


def test_every_family_has_an_order_that_puts_the_shortest_life_first():
    from surpyval.pymc._logp import FAMILIES
    from surpyval.pymc.mixture import ORDER, default_order

    examples = {
        "Weibull": [10.0, 2.0],
        "Exponential": [0.1],
        "LogNormal": [2.0, 0.5],
        "Gamma": [2.0, 0.2],
        "LogLogistic": [10.0, 3.0],
        "Normal": [20.0, 3.0],
        "Gumbel": [20.0, 3.0],
        "Logistic": [20.0, 3.0],
        "Rayleigh": [10.0],
        "InverseGaussian": [10.0, 20.0],
        "GeneralizedGamma": [2.0, 0.5, 1.0],
        "BirnbaumSaunders": [0.5, 10.0],
    }
    assert set(FAMILIES) == set(ORDER) | {"Gompertz"}
    assert default_order(sp.Gompertz) is None
    for name, params in examples.items():
        dist = getattr(sp, name)
        param, direction = default_order(dist)
        k = list(dist.parameter_names).index(param)
        second = list(params)
        second[k] = params[k] * (1.5 if direction == "increasing" else 0.5)
        first_life = dist.from_params(params).qf(0.5)
        second_life = dist.from_params(second).qf(0.5)
        assert first_life < second_life, name


def test_a_gompertz_mixture_needs_an_order():
    pm = pytest.importorskip("pymc")
    from surpyval.pymc import mixture_likelihood

    with pm.Model():
        with pytest.raises(ValueError, match="no default order"):
            mixture_likelihood(
                sp.Gompertz,
                x=[1.0, 2.0],
                weights=[0.5, 0.5],
                shape=[0.1, 0.2],
                rate=[1.0, 1.0],
            )
        mixture_likelihood(
            sp.Gompertz,
            x=[1.0, 2.0],
            weights=[0.5, 0.5],
            shape=[0.1, 0.2],
            rate=[1.0, 1.0],
            order=("rate", "decreasing"),
            name="ordered",
        )


def test_bridge_window_deep_in_the_upper_tail():
    pytest.importorskip("pytensor")
    import mpmath

    from surpyval.pymc import mixture_log_likelihood

    # [64, 67] under Weibull(10, 2) and Weibull(12, 2): probabilities of
    # 1e-18 and 1e-13, which a difference of CDFs loses entirely
    w = np.array([0.4, 0.6])
    alpha, beta = np.array([10.0, 12.0]), np.array([2.0, 2.0])
    mpmath.mp.dps = 50

    def window(a, b):
        return mpmath.exp(-((mpmath.mpf(64) / a) ** b)) - mpmath.exp(
            -((mpmath.mpf(67) / a) ** b)
        )

    exact = float(
        mpmath.log(
            sum(
                mpmath.mpf(float(w[k])) * window(alpha[k], beta[k])
                for k in range(2)
            )
        )
    )
    term = mixture_log_likelihood(
        sp.Weibull, x=[[64.0, 67.0]], c=[2], weights=w, alpha=alpha, beta=beta
    )
    assert float(term.eval()) == pytest.approx(exact, rel=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.MixtureModel(sp.Weibull, 2).fit(
            x=[[64.0, 67.0], 3.0, 9.0, 14.0, 5.0, 7.5, 11.0, 20.0],
            c=[2, 0, 0, 0, 0, 0, 0, 0],
        )
    ours = -float(model.neg_ll_of(w, np.column_stack([alpha, beta])))
    rest = mixture_log_likelihood(
        sp.Weibull,
        x=[3.0, 9.0, 14.0, 5.0, 7.5, 11.0, 20.0],
        weights=w,
        alpha=alpha,
        beta=beta,
    )
    assert ours == pytest.approx(exact + float(rest.eval()), rel=1e-10)
