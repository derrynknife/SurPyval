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
    # the coefficients' default: Normal(0, 2.5 / sd), sd 0.5 here
    assert sorted(fitted.default_priors) == ["alpha_coef_0", "lfp_p_coef_0"]
    assert "Normal(0, 5)" in fitted.default_priors["lfp_p_coef_0"]
    lo, hi = fitted.param_cri("lfp_p")
    assert 0 < lo < fitted.params[2] < hi < 1


def test_flat_priors_peak_at_the_maximum_likelihood_fit():
    x, Z, c, _ = _cure(40.0)
    flat = {k: sp.prior.flat() for k in ("alpha_coef_0", "lfp_p_coef_0")}
    model = sp.MixtureCure(sp.Weibull).fit_bayes(
        x, Z, c, priors=flat, method="laplace"
    )
    np.testing.assert_allclose(
        model.diagnostics["mode"], model.mle.params, rtol=1e-4, atol=1e-6
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
    with pytest.warns(UserWarning, match="too short to identify the cure"):
        sp.MixtureCure(sp.Weibull).fit_bayes(x, Z, c)


def test_no_cured_fraction_warns_and_a_flat_prior_is_refused():
    _, Z, _, life = _cure(40.0)
    x, c = np.minimum(life, 40.0), (life > 40.0).astype(int)
    with pytest.raises(ValueError, match="cure fraction is not identified"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp.MixtureCure(sp.Weibull).fit_bayes(x, Z, c)
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
