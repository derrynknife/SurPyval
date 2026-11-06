"""The Bayesian-fit family of ``diagnose()``: the posterior predictive
checks (against an independent calculation for the closed-form
Exponential-Gamma case, and their replicates' censoring design), the
prior-data conflict check, the grid-accuracy check, each advice rule
firing on data built to call for it and not on data that do not, the
refusals, the plots and the report.

Every seed below was fixed before the test was first run.
"""

import json
import warnings

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
from surpyval.bayes.engines import grid as grid_engine
from surpyval.diagnostics import Diagnosis
from surpyval.diagnostics.bayes import (
    GRID_FAIL,
    GRID_WARN,
    posterior_predictive,
    prior_data_conflict,
)

WEAK = {"beta": sp.prior.around(2, between=(1, 4))}


def _fit(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.Weibull.fit_bayes(*args, **kwargs)


def _rows(d, check):
    return d.table[d.table["check"] == check]


@pytest.fixture(scope="module")
def fits_well():
    # Weibull data, 25% Type I censored at 12, a weak prior on the shape
    x = sp.Weibull.random(40, 10, 2, random_state=1)
    c = (x > 12).astype(int)
    return _fit(np.minimum(x, 12), c, priors=WEAK)


@pytest.fixture(scope="module")
def report_well(fits_well):
    return fits_well.diagnose(n_boot=200, random_state=2)


# -- the report ---------------------------------------------------------------
def test_a_right_model_and_prior_pass(report_well):
    d = report_well
    assert isinstance(d, Diagnosis)
    assert d.family == "Bayesian fit"
    assert d.ok, d
    assert set(d.table["check"]) == {
        "posterior_predictive",
        "goodness_of_fit",
        "prior_data_conflict",
        "grid_accuracy",
    }
    assert not d.advice


def test_the_report_is_reproducible_and_json(fits_well, report_well):
    again = fits_well.diagnose(n_boot=200, random_state=2)
    assert again.table["p"].equals(report_well.table["p"])
    out = report_well.to_dict()
    back = json.loads(json.dumps(out, allow_nan=False))
    assert back["family"] == "Bayesian fit"


def test_plot_draws_the_visual_checks(report_well):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = report_well.plot()
    titles = [ax.get_title() for ax in fig.axes]
    assert any("realised discrepancy" in t for t in titles)
    assert any("failures in time bins" in t for t in titles)
    assert any("prior, data and posterior" in t for t in titles)
    plt.close(fig)


# -- posterior predictive checks ----------------------------------------------
def test_a_wrong_distribution_is_flagged():
    # log-normal data with a long right tail, fitted with a Weibull
    x = sp.LogNormal.random(100, 2, 1, random_state=7)
    model = _fit(x, priors=WEAK)
    d = model.diagnose(n_boot=500, random_state=8)
    # The report fails, through the maximum-likelihood fit's
    # goodness-of-fit tests (p = 0.005 each), and advises other
    # distributions. The posterior predictive checks, conservative by
    # construction, did not reach 0.05 on these data as first run (the AD
    # discrepancy's p = 0.14, the bins' 0.19); they are recorded, not
    # asserted to fail. Their power is in the calibration study.
    assert not d.ok, d
    assert (_rows(d, "goodness_of_fit")["verdict"] == "fail").all(), d
    ppc = _rows(d, "posterior_predictive")
    assert (ppc["p"] < 0.25).all(), d
    assert any("fit_best(return_table=)" in a.features for a in d.advice)


def test_a_badly_wrong_distribution_fails_the_ppc():
    # an exponential fitted to data from a Weibull of shape 3: far off
    x = sp.Weibull.random(60, 10, 3, random_state=9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Exponential.fit_bayes(
            x, priors={"failure_rate": sp.Gamma.from_params([1.0, 10.0])}
        )
    d = model.diagnose(n_boot=500, random_state=10)
    ppc = _rows(d, "posterior_predictive").set_index("test")
    assert ppc.loc["posterior predictive: Anderson-Darling", "verdict"] == (
        "fail"
    ), d
    assert any("fit_best(return_table=)" in a.features for a in d.advice)


def test_conjugate_ppc_matches_an_independent_calculation():
    # Complete data, Exponential with a Gamma prior (the closed form). For
    # a rate lam the replicate's u = 1 - exp(-lam x) are uniform whatever
    # lam, so the p-value is the posterior mean of P(A0 >= A(y, lam)),
    # A0 the Anderson-Darling statistic of n uniforms.
    x = sp.Weibull.random(25, 10, 1.4, random_state=11)
    a0, b0 = 2.0, 20.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Exponential.fit_bayes(
            x, priors={"failure_rate": sp.Gamma.from_params([a0, b0])}
        )
    assert model.method == "conjugate"
    n = x.size
    a, b = a0 + n, b0 + x.sum()
    assert model.posterior.mean("failure_rate") == pytest.approx(a / b)
    out = posterior_predictive(model, 4000, random_state=5)

    lam = stats.gamma(a, scale=1 / b).ppf((np.arange(2000) + 0.5) / 2000)
    i = np.arange(1, n + 1)

    def ad(u):  # the classical formula, rows sorted
        return (
            -n
            - ((2 * i - 1) * (np.log(u) + np.log(1 - u[:, ::-1]))).sum(1) / n
        )

    observed = ad(1 - np.exp(-lam[:, None] * np.sort(x)[None, :]))
    rng = np.random.default_rng(12)
    null = np.sort(ad(np.sort(rng.uniform(size=(200_000, n)), axis=1)))
    p = np.mean(1 - np.searchsorted(null, observed, side="left") / null.size)
    # Monte Carlo error of 4000 draws: about 0.008 at most
    assert out["p"]["AD"] == pytest.approx(p, abs=0.03)
    assert np.median(out["observed"]) == pytest.approx(
        np.median(observed), rel=0.05
    )


def test_replicates_keep_the_type_i_censoring():
    # every unit is censored at 12 if it survives that long, so the
    # replicates' expected failures are N times the posterior predictive
    # F(12), computed independently from the posterior's draws
    x = sp.Weibull.random(60, 10, 2, random_state=21)
    c = (x > 12).astype(int)
    model = _fit(np.minimum(x, 12), c, priors=WEAK)
    out = posterior_predictive(model, 2000, random_state=22)
    expected = out["expected"].sum()
    F12 = float(model.ff(12.0))
    assert expected == pytest.approx(60 * F12, abs=0.6)
    assert out["observed_counts"].sum() == (c == 0).sum()


def test_ppc_bins_follow_the_number_of_failures():
    x = np.r_[2100, 3400, 4050, np.linspace(1500, 6000, 37)]
    c = np.r_[0, 0, 0, np.ones(37)]
    model = _fit(x, c, priors={"beta": sp.prior.around(2, between=(1.5, 3))})
    out = posterior_predictive(model, 100, random_state=1)
    # three failures: one bin, the number of failures
    assert out["edges"].size == 0
    assert out["observed_counts"].tolist() == [3.0]


def test_interval_censored_data_have_no_ppc_and_warn():
    xl = np.array([0, 1, 2, 3, 4, 5, 6, 7.0])
    model = _fit(xl=xl, xr=xl + 1, priors=WEAK)
    d = model.diagnose(n_boot=50, random_state=1)
    ppc = _rows(d, "posterior_predictive")
    assert len(ppc) == 2
    assert (ppc["verdict"] == "warn").all()
    assert ppc["detail"].str.contains("inspection times").all()


def test_goodness_of_fit_reuses_the_univariate_tests(report_well):
    gof = _rows(report_well, "goodness_of_fit")
    assert list(gof["test"]) == [
        "Anderson-Darling (parametric bootstrap, ML fit)",
        "Kolmogorov-Smirnov (parametric bootstrap, ML fit)",
        "Cramer-von Mises (parametric bootstrap, ML fit)",
    ]


# -- prior-data conflict ------------------------------------------------------
def test_a_prior_far_from_the_data_is_flagged():
    x = sp.Weibull.random(40, 10, 2, random_state=31)
    model = _fit(
        x,
        priors={
            "beta": sp.prior.around(6, between=(5, 7)),
            "alpha": sp.prior.around(10, between=(8, 12)),
        },
    )
    d = model.diagnose(n_boot=200, random_state=32)
    rows = _rows(d, "prior_data_conflict").set_index("test")
    assert rows.loc["prior vs data: beta", "verdict"] == "fail"
    assert rows.loc["prior vs data: alpha", "verdict"] == "ok"
    assert (
        rows.loc["prior vs data: all parameters with a prior", "verdict"]
        == "fail"
    )
    assert any(
        "bayes.BayesianParametric.compare" in a.features
        and "disagree on beta" in a
        for a in d.advice
    )
    # the posterior is pulled off the data, and the advice blames the
    # prior, not the distribution
    if (_rows(d, "posterior_predictive")["verdict"] == "fail").any():
        assert any("prior pulls the posterior" in a for a in d.advice)
        assert not any("fit_best" in a for a in d.advice)


def test_conflict_statistic_is_the_normal_prior_predictive_check():
    # Exponential, one parameter: the gap between the prior's mean of
    # log(rate) and the log of the MLE, over the root of the sum of the
    # prior's variance and 1 / failures (the MLE's variance of log rate)
    x = sp.Exponential.random(30, 0.1, random_state=41)
    a0, b0 = 4.0, 10.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Exponential.fit_bayes(
            x, priors={"failure_rate": sp.Gamma.from_params([a0, b0])}
        )
    out = prior_data_conflict(model)
    from scipy.special import digamma, polygamma

    prior_mean = digamma(a0) - np.log(b0)
    prior_var = polygamma(1, a0)
    data_mean = np.log(x.size / x.sum())
    z = (data_mean - prior_mean) / np.sqrt(prior_var + 1 / x.size)
    assert out["z"]["failure_rate"] == pytest.approx(z, rel=1e-3)
    assert out["p"]["failure_rate"] == pytest.approx(
        2 * stats.norm.sf(abs(z)), rel=1e-2
    )


def test_no_failures_cannot_be_compared_and_warn():
    model = _fit(
        np.full(10, 1000.0),
        np.ones(10),
        priors={
            "beta": sp.prior.around(2, between=(1.5, 3)),
            "alpha": sp.prior.around(3000, between=(1000, 9000)),
        },
    )
    d = model.diagnose(n_boot=50, random_state=1)
    rows = _rows(d, "prior_data_conflict")
    assert len(rows) == 2 and (rows["verdict"] == "warn").all()
    assert rows["detail"].str.contains("no maximum-likelihood").all()
    assert _rows(d, "goodness_of_fit").empty
    assert any("Goodness of fit: not checked" in n for n in d.notes)
    assert d.ok


def test_flat_priors_are_not_checked():
    rng = np.random.default_rng(1)
    draws = np.c_[
        rng.lognormal(np.log(10), 0.05, 2000),
        rng.lognormal(np.log(2), 0.1, 2000),
    ]
    x = sp.Weibull.random(40, 10, 2, random_state=3)
    model = sp.bayes.BayesianParametric.from_draws(
        sp.Weibull, draws, ["alpha", "beta"], data=(x,)
    )
    d = model.diagnose(n_boot=50, random_state=1)
    assert _rows(d, "prior_data_conflict").empty
    assert _rows(d, "grid_accuracy").empty
    assert any("no parameter has a proper prior" in n for n in d.notes)
    assert any("another sampler's draws" in n for n in d.notes)


# -- grid accuracy ------------------------------------------------------------
def test_the_default_grid_is_accurate(report_well):
    rows = _rows(report_well, "grid_accuracy")
    assert list(rows["test"]) == [
        "posterior mass at the grid's edge",
        "summaries on a finer grid",
    ]
    assert (rows["verdict"] == "ok").all()
    assert rows["statistic"].iloc[1] < GRID_WARN


def test_a_coarse_grid_is_flagged(monkeypatch):
    # the engine made to accept a 4-point grid: the finer grid moves its
    # summaries by far more than GRID_FAIL
    x = sp.Weibull.random(40, 10, 2, random_state=3)
    loose = {1: 0.5, 2: 0.5, 3: 0.5}
    monkeypatch.setattr(grid_engine, "TOLERANCE", loose)
    coarse = _fit(x, priors=WEAK, points=3)
    monkeypatch.undo()
    d = coarse.diagnose(n_boot=50, random_state=1)
    row = _rows(d, "grid_accuracy").iloc[-1]
    assert row["test"] == "summaries on a finer grid"
    assert row["verdict"] == "fail" and row["statistic"] > GRID_FAIL
    assert any(
        "Weibull.fit_bayes(points=)" in a.features and "more points" in a
        for a in d.advice
    )


def test_a_laplace_approximation_is_checked_against_the_grid():
    # 30 units, few enough that the posterior is skewed on the log scale
    x = sp.Weibull.random(30, 10, 2, random_state=1)
    c = (x > 12).astype(int)
    model = _fit(np.minimum(x, 12), c, priors=WEAK, method="laplace")
    d = model.diagnose(n_boot=50, random_state=1)
    rows = _rows(d, "grid_accuracy")
    assert list(rows["test"]) == ["Laplace approximation against the grid"]
    assert rows["statistic"].iloc[0] > GRID_WARN
    assert any("Weibull.fit_bayes(method=)" in a.features for a in d.advice)


def test_a_closed_form_has_no_grid_check():
    x = sp.Exponential.random(30, 0.1, random_state=41)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Exponential.fit_bayes(
            x, priors={"failure_rate": sp.Gamma.from_params([2.0, 20.0])}
        )
    d = model.diagnose(n_boot=50, random_state=1)
    assert _rows(d, "grid_accuracy").empty
    assert any("closed form" in n for n in d.notes)


# -- refusals -----------------------------------------------------------------
def test_draws_without_data_refuse_saying_what_to_do():
    draws = np.c_[np.full(10, 10.0), np.linspace(1.5, 2.5, 10)]
    model = sp.bayes.BayesianParametric.from_draws(
        sp.Weibull, draws, ["alpha", "beta"]
    )
    with pytest.raises(ValueError, match=r"from_draws\(\.\.\., data="):
        model.diagnose()


def test_a_bernoulli_refuses_saying_what_to_do():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Bernoulli.fit_bayes(
            [1, 0, 1, 1], priors={"p": sp.Beta.from_params([1.0, 1.0])}
        )
    with pytest.raises(ValueError, match=r"model\.compare\(\)"):
        model.diagnose()
