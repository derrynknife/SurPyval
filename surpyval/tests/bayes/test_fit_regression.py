"""``fit_bayes`` on the parametric regressions (proportional hazards,
accelerated failure time, accelerated life), by the Laplace engine: the
posterior against closed forms and the maximum-likelihood fit, priors on
coefficients, the predictive functions at covariates, saving, and the
refusals.

The rules were fixed before the first run, with the seeds (460001 on)
and sizes:

- an exponential proportional hazards model with one binary covariate
  and flat priors: the groups' rates are independent Gammas, so the
  Laplace normal is a closed form (on the log rates, centre ``log(d /
  T)`` and variance ``1 / d`` for each group), which the fit must match
  to 1e-4, and reweighted (``importance=True``) its 2.5%, 50% and 97.5%
  points must be within 3% of the exact interval's width;
- with flat priors and 400 units, each posterior median within 0.1
  posterior standard deviations of the maximum-likelihood estimate, and
  each posterior standard deviation within 10% of the estimate's
  standard error;
- as a prior on a coefficient widens (``Normal(0, s)``, ``s`` from 0.1
  to 100), its posterior median moves to the estimate: the gap shrinks
  at every step, and at ``s = 100`` is below 0.01 standard errors;
- a prior on a coefficient two of the estimate's standard errors from it,
  with that spread: the posterior median strictly between the prior's
  centre and the estimate.
"""

import math
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

import surpyval as sp
from surpyval.bayes import BayesianRegression
from surpyval.life_models import Power


def quietly(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(*args, **kwargs)


def weibull_ph_data(n=40, seed=460010):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 1))
    x = sp.Weibull.random(n, 50.0, 2.0, random_state=seed + 1)
    return x * np.exp(-0.25 * Z[:, 0]), Z


@pytest.fixture(scope="module")
def fitted():
    x, Z = weibull_ph_data()
    return quietly(
        sp.WeibullPH.fit_bayes,
        x,
        Z,
        priors={"coef_0": sp.Normal.from_params([0.5, 0.2])},
    )


def _two_groups():
    rng = np.random.default_rng(460001)
    z = np.repeat([0.0, 1.0], 30)
    life = rng.exponential(1.0 / np.where(z == 0, 0.01, 0.02))
    x = np.minimum(life, 100.0)
    c = (life > 100.0).astype(int)
    d0, d1 = (int(np.sum((c == 0) & (z == g))) for g in (0, 1))
    T0, T1 = (float(np.sum(x[z == g])) for g in (0, 1))
    return x, z[:, None], c, (d0, d1, T0, T1)


def test_exponential_ph_laplace_is_the_closed_form():
    x, Z, c, (d0, d1, T0, T1) = _two_groups()
    model = sp.ExponentialPH.fit_bayes(
        x, Z, c, priors={"coef_0": sp.prior.flat()}
    )
    assert model.method == "laplace"
    z = stats.norm.ppf([0.025, 0.5, 0.975])
    rate = np.exp(np.log(d0 / T0) + z / math.sqrt(d0))
    coef = (
        math.log(d1 / T1) - math.log(d0 / T0) + z * math.sqrt(1 / d0 + 1 / d1)
    )
    q = [0.025, 0.5, 0.975]
    np.testing.assert_allclose(
        model.posterior.quantile("failure_rate", q), rate, rtol=1e-4
    )
    np.testing.assert_allclose(
        model.posterior.quantile("coef_0", q), coef, rtol=1e-4, atol=1e-4
    )


def test_exponential_ph_reweighted_matches_the_exact_posterior():
    x, Z, c, (d0, d1, T0, T1) = _two_groups()
    model = sp.ExponentialPH.fit_bayes(
        x, Z, c, priors={"coef_0": sp.prior.flat()}, importance=True
    )
    assert model.diagnostics["importance"]
    q = np.array([0.025, 0.5, 0.975])
    rate = stats.gamma(d0, scale=1.0 / T0)
    exact = rate.ppf(q)
    got = model.posterior.quantile("failure_rate", q)
    assert np.all(np.abs(got - exact) / (exact[2] - exact[0]) < 0.03)
    g1 = stats.gamma(d1, scale=1.0 / T1)

    def coef_cdf(v):
        # P(log G1 - log G0 <= v) = E[P(G1 <= exp(v) G0)]
        lo, hi = rate.ppf(1e-12), rate.ppf(1 - 1e-12)
        return integrate.quad(
            lambda g: g1.cdf(math.exp(v) * g) * rate.pdf(g),
            lo,
            hi,
            limit=200,
            epsabs=1e-12,
        )[0]

    from scipy.optimize import brentq

    exact = np.array([brentq(lambda v: coef_cdf(v) - p, -5.0, 5.0) for p in q])
    got = model.posterior.quantile("coef_0", q)
    assert np.all(np.abs(got - exact) / (exact[2] - exact[0]) < 0.03)


def _aft_data():
    rng = np.random.default_rng(460003)
    Z = rng.normal(size=(400, 1))
    life = sp.Weibull.random(400, 100.0, 1.5, random_state=460004)
    life = life * np.exp(0.5 * Z[:, 0])
    return np.minimum(life, 200.0), Z, (life > 200.0).astype(int)


def _se(mle, name):
    lo, hi = mle.param_cb(name, alpha_ci=2 * stats.norm.sf(1.0))
    return float(np.ravel(hi)[0] - np.ravel(lo)[0]) / 2


def test_flat_priors_and_plenty_of_data_match_the_mle():
    x, Z, c = _aft_data()
    model = sp.WeibullAFT.fit_bayes(
        x, Z, c, priors={"coef_0": sp.prior.flat()}
    )
    assert model.default_priors == {}
    post = model.posterior
    for name, estimate in zip(model.parameter_names, model.mle.params):
        lo, med, hi = post.quantile(name, [stats.norm.cdf(-1), 0.5, 0.8413])
        sd = float(hi - lo) / 2
        assert abs(float(med) - float(estimate)) <= 0.1 * sd, name
        assert abs(sd / _se(model.mle, name) - 1) < 0.1, name


def test_a_flattening_prior_moves_to_the_mle():
    x, Z, c = _aft_data()
    mle = sp.WeibullAFT.fit(x, Z, c)
    i = mle.parameter_names.index("coef_0")
    se = _se(mle, "coef_0")
    gaps = []
    for s in (0.1, 1.0, 10.0, 100.0):
        model = sp.WeibullAFT.fit_bayes(
            x, Z, c, priors={"coef_0": sp.Normal.from_params([0.0, s])}
        )
        gaps.append(abs(float(model.params[i]) - float(mle.params[i])) / se)
    assert all(a > b for a, b in zip(gaps, gaps[1:])), gaps
    assert gaps[-1] < 0.01, gaps


def test_a_prior_on_the_life_model_exponent_pulls_the_estimate():
    """A small accelerated test (Weibull life, power law in the stress)
    and a prior on the exponent ``n``, two standard errors from the
    estimate with that spread."""
    stress = np.repeat([20.0, 30.0, 40.0], 5).reshape(-1, 1)
    life = sp.Weibull.random(15, 10.0, 3.0, random_state=460006)
    x = life * (100.0 / stress[:, 0])
    fitter = sp.AcceleratedLife(sp.Weibull, Power)
    mle = fitter.fit(x, Z=stress)
    i = mle.parameter_names.index("n")
    se = _se(mle, "n")
    centre = float(mle.params[i]) + 2 * se
    model = quietly(
        fitter.fit_bayes,
        x,
        stress,
        priors={
            "n": sp.Normal.from_params([centre, se]),
            "a": sp.prior.flat(),
        },
    )
    assert model.parameter_names == ["alpha", "beta", "a", "n"]
    assert model.params[0] == 1.0  # the accelerated life model's fixed
    median = float(model.params[i])
    assert float(mle.params[i]) < median < centre
    use = [[10.0]]
    lo, hi = model.quantile_cri(0.1, use)
    assert lo < float(model.qf(0.1, use)) < hi
    lo, hi = model.param_cri("n")
    assert lo < median < hi


def test_a_coefficient_with_no_prior_gets_the_stated_default(fitted):
    x, Z = weibull_ph_data()
    model = quietly(
        sp.WeibullPH.fit_bayes,
        x,
        100.0 * Z,
        priors={"beta": sp.prior.around(2, between=(1, 4))},
    )
    sd = float(np.std(100.0 * Z[:, 0]))
    said = model.default_priors["coef_0"]
    assert said.startswith(f"default: Normal(0, {2.5 / sd:.4g})")
    assert "coef_0: default: Normal(0," in model.summary()
    prior = model.priors.single["coef_0"]
    assert prior.loc == 0.0 and prior.scale == pytest.approx(2.5 / sd)
    # a prior given is not replaced
    assert fitted.default_priors == {}


@pytest.mark.parametrize("importance", [False, True])
def test_the_predictive_functions_at_covariates(fitted, importance):
    if importance:
        x, Z = weibull_ph_data()
        m = quietly(
            sp.WeibullPH.fit_bayes,
            x,
            Z,
            priors={"coef_0": sp.Normal.from_params([0.5, 0.2])},
            importance=True,
        )
        assert m.diagnostics["importance"]
    else:
        m = fitted
    w = m.posterior.weights
    t = np.array([10.0, 40.0, 90.0])
    z = [[0.5]]
    models = [m.mle.with_params(p) for p in m._draws()]
    mean_sf = w @ np.array([mm.sf(t, z) for mm in models])
    assert m.sf(t, z) == pytest.approx(mean_sf, rel=1e-12)
    assert m.ff(t, z) + m.sf(t, z) == pytest.approx(np.ones(3), rel=1e-12)
    mean_df = w @ np.array([mm.df(t, z) for mm in models])
    assert m.hf(t, z) == pytest.approx(mean_df / mean_sf, rel=1e-12)
    assert m.Hf(t, z) == pytest.approx(-np.log(mean_sf), rel=1e-12)
    q = m.qf([0.1, 0.5], z)
    assert m.ff(q, z) == pytest.approx([0.1, 0.5], abs=1e-9)
    assert m.sf(t, [[0.0], [0.5], [1.0]]).shape == (3,)
    assert m.sf(t, [[0.0], [1.0]], grid=True).shape == (2, 3)
    assert np.all(m.sf(t, [[1.0]]) < m.sf(t, [[0.0]]))
    band = m.cri(t, z)
    assert band.shape == (3, 2) and np.all(band[:, 0] < band[:, 1])
    assert m.cri(t, z, bound="lower").shape == (3,)
    lo, hi = m.param_cri("coef_0")
    assert lo < m.params[2] < hi
    life, rows = m.random(5, [[0.0], [1.0]], random_state=1)
    assert life.shape == (10,) and rows.shape == (10, 1)


def test_compare_and_summary(fitted):
    table = fitted.compare().table
    assert sorted(table) == ["alpha", "beta", "coef_0"]
    row = table["coef_0"]
    assert row["prior"].startswith("Normal(0.5, 0.2)")
    assert row["mle"] == pytest.approx(float(fitted.mle.params[2]))
    text = fitted.summary()
    assert "Weibull Proportional Hazard: Bayesian fit, 40 items" in text
    assert "Laplace" in text and "Pareto k" in text


def test_saving_and_loading(fitted):
    t, z = np.array([20.0, 60.0]), [[0.3]]
    for again in (
        sp.from_dict(fitted.to_dict()),
        pickle.loads(pickle.dumps(fitted)),
    ):
        assert isinstance(again, BayesianRegression)
        assert np.array_equal(again.sf(t, z), fitted.sf(t, z))
        assert again.parameter_names == fitted.parameter_names
        assert np.array_equal(
            again.param_cri("coef_0"), fitted.param_cri("coef_0")
        )
        assert again.default_priors == fitted.default_priors


def test_fits_are_reproducible():
    x, Z = weibull_ph_data()
    one = quietly(sp.WeibullPH.fit_bayes, x, Z)
    two = quietly(sp.WeibullPH.fit_bayes, x, Z)
    assert np.array_equal(one.posterior.points, two.posterior.points)


def test_columns_of_a_dataframe_name_the_coefficients():
    x, Z = weibull_ph_data()
    df = pd.DataFrame({"hours": x, "load": Z[:, 0]})
    model = quietly(
        sp.LogNormalAFT.fit_bayes_from_df,
        df,
        x_col="hours",
        Z_cols=["load"],
        priors={"load": sp.Normal.from_params([0.0, 0.5])},
    )
    assert model.parameter_names == ["mu", "sigma", "load"]
    assert model.default_priors == {}
    new = pd.DataFrame({"load": [0.0, 1.0]})
    assert model.sf(30.0, new).shape == (2,)


def test_refusals_say_what_to_do(fitted):
    x, Z = weibull_ph_data()
    with pytest.raises(ValueError, match="does not take w: weights"):
        sp.WeibullPH.fit_bayes(x, Z, w=np.ones(len(x)))
    with pytest.raises(ValueError, match="a prior on the coefficients"):
        sp.WeibullPH.fit_bayes(x, Z, penalizer=0.1)
    with pytest.raises(ValueError, match="does not take chains"):
        sp.WeibullPH.fit_bayes(x, Z, chains=4)
    with pytest.raises(ValueError, match="this is a Proportional Odds"):
        sp.WeibullPO.fit_bayes(x, Z)
    with pytest.raises(ValueError, match="semi-parametric"):
        sp.CoxPH.fit_bayes(x, Z)
    with pytest.raises(ValueError, match="determine coef_1 .*Leave the"):
        sp.WeibullPH.fit_bayes(x, np.c_[Z, np.ones(len(x))])
    with pytest.raises(ValueError, match="starts from the maximum-lik"):
        sp.WeibullPH.fit_bayes(
            x, Z, np.ones(len(x)), priors={"coef_0": sp.prior.flat()}
        )
    with pytest.raises(ValueError, match="use cri"):
        fitted.cb(10.0, [[0.0]])
    with pytest.raises(ValueError, match="model.mle.diagnose"):
        fitted.diagnose()
    with pytest.raises(ValueError, match="on must be one of"):
        fitted.cri(10.0, [[0.0]], on="mean")
