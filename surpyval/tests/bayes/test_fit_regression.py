"""``fit_bayes`` on the parametric regressions (proportional hazards,
accelerated failure time, accelerated life): the posterior against a
closed form and the maximum-likelihood fit, priors on coefficients, the
predictive functions at covariates, saving, and the refusals.

The rules were fixed before the first run, with the seeds (460001 on)
and sizes:

- against the closed form, the rule of ``test_sampler.py``: within 4
  Monte Carlo standard errors from the draws' own effective sample size
  (the mean, and the exact CDF at the draws' 2.5%, 25%, 50%, 75% and
  97.5% points);
- against the maximum-likelihood fit, with flat priors and 400 units:
  each posterior mean within 4 Monte Carlo standard errors plus 0.1
  posterior standard deviations of the estimate (the two differ by
  order ``1 / n``, a twentieth of a standard deviation here);
- a prior on a coefficient two of the estimate's standard errors from it,
  with that spread: the posterior median strictly between the prior's
  centre and the estimate.

The fits that check only the interface use short chains and do not ask
them to converge.
"""

import math
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats
from scipy.special import polygamma, psi

import surpyval as sp
from surpyval.bayes import BayesianRegression
from surpyval.bayes import convergence as cv
from surpyval.life_models import Power
from surpyval.tests.bayes.test_sampler import within_monte_carlo_error

SHORT = dict(chains=2, draws=100, warmup=100)


def chains_of(model, name):
    d = model.diagnostics
    col = model.posterior.parameter_names.index(name)
    return model.posterior.points[:, col].reshape(
        d["chains"], d["draws_per_chain"]
    )


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
def short_fit():
    x, Z = weibull_ph_data()
    return quietly(
        sp.WeibullPH.fit_bayes,
        x,
        Z,
        priors={"coef_0": sp.Normal.from_params([0.5, 0.2])},
        random_state=460011,
        **SHORT,
    )


def test_exponential_ph_with_flat_priors_matches_the_closed_form():
    """An exponential proportional hazards model with one binary
    covariate and flat priors on ``log failure_rate`` and the
    coefficient: the two groups' rates are independent, each
    ``Gamma(failures, total time)``, so the baseline rate's posterior is
    that Gamma, and the coefficient's the difference of their logs."""
    rng = np.random.default_rng(460001)
    z = np.repeat([0.0, 1.0], 30)
    life = rng.exponential(1.0 / np.where(z == 0, 0.01, 0.02))
    x = np.minimum(life, 100.0)
    c = (life > 100.0).astype(int)
    d0, d1 = (int(np.sum((c == 0) & (z == g))) for g in (0, 1))
    T0, T1 = (float(np.sum(x[z == g])) for g in (0, 1))
    model = sp.ExponentialPH.fit_bayes(
        x,
        z[:, None],
        c,
        priors={"coef_0": sp.prior.flat()},
        random_state=460002,
    )
    assert model.ok
    rate = stats.gamma(d0, scale=1.0 / T0)
    within_monte_carlo_error(
        chains_of(model, "failure_rate"),
        rate.cdf,
        float(rate.mean()),
        float(rate.std()),
    )
    g1 = stats.gamma(d1, scale=1.0 / T1)

    def coef_cdf(q):
        # P(log G1 - log G0 <= q) = E[P(G1 <= exp(q) G0)]
        lo, hi = rate.ppf(1e-12), rate.ppf(1 - 1e-12)
        return integrate.quad(
            lambda g: g1.cdf(math.exp(q) * g) * rate.pdf(g),
            lo,
            hi,
            limit=200,
            epsabs=1e-12,
        )[0]

    mean = psi(d1) - math.log(T1) - psi(d0) + math.log(T0)
    sd = math.sqrt(polygamma(1, d1) + polygamma(1, d0))
    within_monte_carlo_error(chains_of(model, "coef_0"), coef_cdf, mean, sd)


def test_flat_priors_and_plenty_of_data_match_the_mle():
    rng = np.random.default_rng(460003)
    Z = rng.normal(size=(400, 1))
    life = sp.Weibull.random(400, 100.0, 1.5, random_state=460004)
    life = life * np.exp(0.5 * Z[:, 0])
    x, c = np.minimum(life, 200.0), (life > 200.0).astype(int)
    model = sp.WeibullAFT.fit_bayes(
        x, Z, c, priors={"coef_0": sp.prior.flat()}, random_state=460005
    )
    assert model.ok
    assert model.default_priors == {}
    for name, estimate in zip(model.parameter_names, model.mle.params):
        draws = chains_of(model, name)
        sd = float(draws.std())
        gap = abs(float(draws.mean()) - float(estimate))
        allowed = 4 * sd / math.sqrt(cv.ess_bulk(draws)) + 0.1 * sd
        assert gap <= allowed, (name, gap, allowed)


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
    # one standard error: the half width of the Wald interval of 68.3%
    lo, hi = mle.param_cb("n", alpha_ci=2 * stats.norm.sf(1.0))
    se = float(hi - lo) / 2
    centre = float(mle.params[i]) + 2 * se
    model = fitter.fit_bayes(
        x,
        stress,
        priors={
            "n": sp.Normal.from_params([centre, se]),
            "a": sp.prior.flat(),
        },
        random_state=460007,
    )
    assert model.ok
    assert model.parameter_names == ["alpha", "beta", "a", "n"]
    assert model.params[0] == 1.0  # the accelerated life model's fixed
    median = float(model.params[i])
    assert float(mle.params[i]) < median < centre
    use = [[10.0]]
    lo, hi = model.quantile_cri(0.1, use)
    assert lo < float(model.qf(0.1, use)) < hi
    lo, hi = model.param_cri("n")
    assert lo < median < hi


def test_a_coefficient_with_no_prior_gets_the_stated_default(short_fit):
    x, Z = weibull_ph_data()
    model = quietly(
        sp.WeibullPH.fit_bayes,
        x,
        100.0 * Z,
        priors={"beta": sp.prior.around(2, between=(1, 4))},
        random_state=460012,
        **SHORT,
    )
    sd = float(np.std(100.0 * Z[:, 0]))
    said = model.default_priors["coef_0"]
    assert said.startswith(f"default: Normal(0, {2.5 / sd:.4g})")
    assert "coef_0: default: Normal(0," in model.summary()
    prior = model.priors.single["coef_0"]
    assert prior.loc == 0.0 and prior.scale == pytest.approx(2.5 / sd)
    # a prior given is not replaced
    assert short_fit.default_priors == {}


def test_the_predictive_functions_at_covariates(short_fit):
    m = short_fit
    t = np.array([10.0, 40.0, 90.0])
    z = [[0.5]]
    models = [m.mle.with_params(p) for p in m._draws()]
    mean_sf = np.mean([mm.sf(t, z) for mm in models], axis=0)
    assert m.sf(t, z) == pytest.approx(mean_sf, rel=1e-12)
    assert m.ff(t, z) + m.sf(t, z) == pytest.approx(np.ones(3), rel=1e-12)
    mean_df = np.mean([mm.df(t, z) for mm in models], axis=0)
    assert m.hf(t, z) == pytest.approx(mean_df / mean_sf, rel=1e-12)
    assert m.Hf(t, z) == pytest.approx(-np.log(mean_sf), rel=1e-12)
    q = m.qf([0.1, 0.5], z)
    assert m.ff(q, z) == pytest.approx([0.1, 0.5], abs=1e-9)
    # one row per time, and every time for every row
    assert m.sf(t, [[0.0], [0.5], [1.0]]).shape == (3,)
    assert m.sf(t, [[0.0], [1.0]], grid=True).shape == (2, 3)
    # a higher covariate, a higher hazard (the prior and data say so)
    assert np.all(m.sf(t, [[1.0]]) < m.sf(t, [[0.0]]))
    band = m.cri(t, z)
    assert band.shape == (3, 2) and np.all(band[:, 0] < band[:, 1])
    assert m.cri(t, z, bound="lower").shape == (3,)
    lo, hi = m.param_cri("coef_0")
    assert lo < m.params[2] < hi


def test_compare_and_summary(short_fit):
    table = short_fit.compare().table
    assert sorted(table) == ["alpha", "beta", "coef_0"]
    row = table["coef_0"]
    assert row["prior"].startswith("Normal(0.5, 0.2)")
    assert row["mle"] == pytest.approx(float(short_fit.mle.params[2]))
    text = short_fit.summary()
    assert "Weibull Proportional Hazard: Bayesian fit, 40 items" in text
    assert "No-U-Turn sampler" in text


def test_saving_and_loading(short_fit):
    t, z = np.array([20.0, 60.0]), [[0.3]]
    for again in (
        sp.from_dict(short_fit.to_dict()),
        pickle.loads(pickle.dumps(short_fit)),
    ):
        assert isinstance(again, BayesianRegression)
        assert np.array_equal(again.sf(t, z), short_fit.sf(t, z))
        assert again.parameter_names == short_fit.parameter_names
        assert again.diagnostics["rhat"] == short_fit.diagnostics["rhat"]
        assert again.default_priors == short_fit.default_priors


def test_seeded_fits_do_not_depend_on_n_jobs():
    x, Z = weibull_ph_data()
    kw = dict(chains=2, draws=30, warmup=30, random_state=3)
    one = quietly(sp.WeibullPH.fit_bayes, x, Z, **kw)
    two = quietly(sp.WeibullPH.fit_bayes, x, Z, n_jobs=2, **kw)
    other = quietly(sp.WeibullPH.fit_bayes, x, Z, **{**kw, "random_state": 4})
    assert np.array_equal(one.posterior.points, two.posterior.points)
    assert not np.array_equal(one.posterior.points, other.posterior.points)


def test_columns_of_a_dataframe_name_the_coefficients():
    x, Z = weibull_ph_data()
    df = pd.DataFrame({"hours": x, "load": Z[:, 0]})
    model = quietly(
        sp.LogNormalAFT.fit_bayes_from_df,
        df,
        x_col="hours",
        Z_cols=["load"],
        priors={"load": sp.Normal.from_params([0.0, 0.5])},
        random_state=460013,
        **SHORT,
    )
    assert model.parameter_names == ["mu", "sigma", "load"]
    assert model.default_priors == {}
    new = pd.DataFrame({"load": [0.0, 1.0]})
    assert model.sf(30.0, new).shape == (2,)


def test_a_failed_check_warns_with_the_parameter_and_the_number():
    x, Z = weibull_ph_data()
    with pytest.warns(UserWarning, match="effective sample size below 200"):
        model = sp.WeibullPH.fit_bayes(
            x, Z, chains=2, draws=20, warmup=20, random_state=1
        )
    assert model.ok is False
    with pytest.raises(ValueError, match="did not pass its checks"):
        model.raise_if_failed()


def test_refusals_say_what_to_do(short_fit):
    x, Z = weibull_ph_data()
    with pytest.raises(ValueError, match="does not take w: weights"):
        sp.WeibullPH.fit_bayes(x, Z, w=np.ones(len(x)))
    with pytest.raises(ValueError, match="a prior on the coefficients"):
        sp.WeibullPH.fit_bayes(x, Z, penalizer=0.1)
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
        short_fit.cb(10.0, [[0.0]])
    with pytest.raises(ValueError, match="model.mle.diagnose"):
        short_fit.diagnose()
    with pytest.raises(ValueError, match="on must be one of"):
        short_fit.cri(10.0, [[0.0]], on="mean")
