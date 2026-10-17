"""The univariate-distribution family of ``diagnose()`` (#875): the
goodness-of-fit statistics against R, the bootstrap's null distribution
against the textbook tables, each advice rule firing on data built to call
for it, the refusals, the plots and the report.

The R values were computed with R 4.3.3, fitdistrplus 1.1.11 and survival
3.5.8:

.. code:: r

    library(fitdistrplus); library(survival)
    x <- c(3.1, 4.7, 5.2, 6.0, 6.9, 7.4, 8.1, 8.8, 9.3, 10.2, 10.9, 11.5,
           12.4, 13.0, 14.6, 15.1, 16.8, 18.2, 21.5, 25.7)
    for (d in c("weibull", "lnorm", "norm")) {
      f <- fitdist(x, d); g <- gofstat(f)
      print(f$estimate); print(c(g$ks, g$cvm, g$ad))
    }
    # right-censored: the Kaplan-Meier estimate of u against the uniform,
    # integrated numerically
    u <- c(0.05, 0.12, 0.2, 0.2, 0.31, 0.4, 0.47, 0.55, 0.61, 0.7, 0.72,
           0.9)
    status <- c(1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0)
    km <- survfit(Surv(u, status) ~ 1)
    Fn <- stepfun(km$time, c(0, 1 - km$surv)); N <- 12; tau <- 0.9
    N * integrate(function(s) (Fn(s) - s)^2, 0, tau,
                  subdivisions = 5000, rel.tol = 1e-10)$value
    N * integrate(function(s) (Fn(s) - s)^2 / (s * (1 - s)), 0, tau,
                  subdivisions = 5000, rel.tol = 1e-10)$value
    grid <- sort(c(km$time, km$time - 1e-12, tau, 0))
    max(abs(Fn(grid) - grid))
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.diagnostics import Diagnosis, univariate
from surpyval.diagnostics.univariate import bootstrap, gof_statistics

X = np.array(
    [3.1, 4.7, 5.2, 6.0, 6.9, 7.4, 8.1, 8.8, 9.3, 10.2, 10.9, 11.5]
    + [12.4, 13.0, 14.6, 15.1, 16.8, 18.2, 21.5, 25.7]
)

#: fitdistrplus::fitdist's estimates (in its parametrisation) and
#: gofstat's KS, CvM and AD.
R_GOFSTAT = {
    "Weibull": (
        (13.00020865890178, 2.14753969351683),
        (0.0678921216836759, 0.0156576186096179, 0.124969549748833),
    ),
    "LogNormal": (
        (2.310910918259331, 0.523613825622734),
        (0.0601666952829071, 0.0113253448218135, 0.0932642151813212),
    ),
    "Normal": (
        (11.47, 5.70386710925141),
        (0.0981919395165216, 0.0457503005387375, 0.321025569304243),
    ),
}


def _fit(dist, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dist.fit(*args, **kwargs)


def _rows(d, check, verdict=None):
    table = d.table[d.table["check"] == check]
    if verdict is not None:
        table = table[table["verdict"] == verdict]
    return table


def _kinds(d):
    """The alternatives that warned, by the advice rule kind."""
    rows = _rows(d, "alternatives", "warn")
    return set(rows["alternative"])


# -- the statistics -----------------------------------------------------------
@pytest.mark.parametrize("name", sorted(R_GOFSTAT))
def test_statistics_match_fitdistrplus(name):
    params, (ks, cvm, ad) = R_GOFSTAT[name]
    model = getattr(sp, name).from_params(list(params))
    out = gof_statistics(model.ff(X), np.ones(X.size, bool))
    np.testing.assert_allclose(
        [out["KS"], out["CvM"], out["AD"]], [ks, cvm, ad], rtol=1e-9
    )


def test_censored_statistics_match_r():
    u = np.array([0.05, 0.12, 0.2, 0.2, 0.31, 0.4, 0.47, 0.55, 0.61, 0.7])
    u = np.append(u, [0.72, 0.9])
    status = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], bool)
    out = gof_statistics(u, status)
    np.testing.assert_allclose(
        [out["KS"], out["CvM"], out["AD"]],
        [0.137499999999, 0.0350447917059, 0.183392794472],
        rtol=1e-8,
    )


def test_complete_data_give_the_classical_statistics():
    rng = np.random.default_rng(4)
    u = np.sort(rng.uniform(size=37))
    n, i = u.size, np.arange(1, u.size + 1)
    ad = -n - np.mean((2 * i - 1) * (np.log(u) + np.log1p(-u[::-1])))
    cvm = 1 / (12 * n) + np.sum((u - (2 * i - 1) / (2 * n)) ** 2)
    ks = max(np.max(i / n - u), np.max(u - (i - 1) / n))
    out = gof_statistics(rng.permutation(u), np.ones(n, bool))
    np.testing.assert_allclose(
        [out["AD"], out["CvM"], out["KS"]], [ad, cvm, ks], rtol=1e-9
    )


def test_counts_equal_repetition():
    u = np.array([0.1, 0.25, 0.4, 0.4, 0.7, 0.85])
    failed = np.array([1, 0, 1, 1, 1, 0], bool)
    counts = np.array([2, 1, 3, 1, 2, 4])
    one = gof_statistics(u, failed, counts)
    rep = gof_statistics(np.repeat(u, counts), np.repeat(failed, counts))
    for key in one:
        assert one[key] == pytest.approx(rep[key], rel=1e-12)


def test_truncation_conditions_the_transform():
    model = sp.Weibull.from_params([10, 2])
    x = np.array([5.0, 8.0, 12.0])
    t = np.array([[4.0, np.inf], [4.0, 20.0], [-np.inf, np.inf]])
    u = univariate.transform(model, x, t)
    F = model.ff
    expected = [
        (F(5.0) - F(4.0)) / (1 - F(4.0)),
        (F(8.0) - F(4.0)) / (F(20.0) - F(4.0)),
        F(12.0),
    ]
    np.testing.assert_allclose(u, expected, rtol=1e-12)


# -- the bootstrap ------------------------------------------------------------
@pytest.mark.parametrize(
    "dist, x, tables",
    [
        # D'Agostino & Stephens (1986), Goodness-of-Fit Techniques, tables
        # 4.7 (normal, both parameters estimated) and 4.11 (exponential,
        # scale estimated): the 5% points of the modified statistics.
        (
            sp.Normal,
            sp.Normal.random(20, 10, 2, random_state=1),
            {
                "AD": (lambda s, n: s * (1 + 0.75 / n + 2.25 / n**2), 0.752),
                "CvM": (lambda s, n: s * (1 + 0.5 / n), 0.126),
                "KS": (
                    lambda s, n: s * (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n)),
                    0.895,
                ),
            },
        ),
        (
            sp.Exponential,
            sp.Exponential.random(20, 0.1, random_state=1),
            {
                "AD": (lambda s, n: s * (1 + 0.6 / n), 1.341),
                "CvM": (lambda s, n: s * (1 + 0.16 / n), 0.222),
                "KS": (
                    lambda s, n: (s - 0.2 / n)
                    * (np.sqrt(n) + 0.26 + 0.5 / np.sqrt(n)),
                    1.094,
                ),
            },
        ),
    ],
)
def test_bootstrap_null_matches_the_textbook_tables(dist, x, tables):
    # With complete data and a location-scale (or scale) family the null
    # distribution of each statistic does not depend on the parameters, so
    # the refits of one sample give it: their 95% points are the tables'.
    out = bootstrap(_fit(dist, x), 600, random_state=3)
    keys = [key for _, key in univariate.TESTS]
    for j, key in enumerate(keys):
        modify, critical = tables[key]
        q = np.quantile(modify(out["draws"][:, j], x.size), 0.95)
        assert q == pytest.approx(critical, rel=0.15), key
    assert out["failed"] == 0


def test_bootstrap_p_values_are_uniform_under_the_null():
    # A modest calibration: exponential samples, randomly right-censored,
    # each tested with 29 refits; the heavier study, on Weibull samples
    # with truncation, is test_calibration_of_the_bootstrap below.
    rng = np.random.default_rng(12)
    ps = []
    for i in range(50):
        t = rng.exponential(10, 40)
        cens = rng.uniform(0, 30, 40)
        model = _fit(sp.Exponential, np.minimum(t, cens), c=(t > cens) * 1)
        ps.append(bootstrap(model, 29, random_state=i)["p"]["AD"])
    ps = np.array(ps)
    assert 0.35 < ps.mean() < 0.65
    assert np.mean(ps <= 0.1) < 0.25


@pytest.mark.calibration
def test_calibration_of_the_bootstrap():
    # Weibull samples, left-truncated at 2 and randomly right-censored:
    # the share of p-values at or below 0.1 is 0.1 for each statistic
    # (with 49 refits a p-value is a multiple of 1/50).
    rng = np.random.default_rng(2024)
    ps = []
    for i in range(150):
        t = sp.Weibull.random(120, 10, 1.5, random_state=rng)
        t = t[t > 2][:60]
        cens = rng.uniform(2, 30, t.size)
        model = _fit(sp.Weibull, np.minimum(t, cens), c=(t > cens) * 1, tl=2.0)
        p = bootstrap(model, 49, random_state=i)["p"]
        ps.append([p[key] for _, key in univariate.TESTS])
    rate = np.mean(np.array(ps) <= 0.1 + 1e-9, axis=0)
    # 99.9% binomial bounds about 0.1 for 150 samples
    assert np.all((rate > 0.03) & (rate < 0.19)), rate


def test_bootstrap_is_reproducible_and_follows_the_seed_rule():
    model = _fit(sp.Weibull, sp.Weibull.random(30, 10, 2, random_state=7))
    a = bootstrap(model, 15, random_state=3)
    b = bootstrap(model, 15, random_state=3)
    np.testing.assert_array_equal(a["draws"], b["draws"])
    np.random.seed(5)
    c = bootstrap(model, 15)
    np.random.seed(5)
    d = bootstrap(model, 15)
    np.testing.assert_array_equal(c["draws"], d["draws"])


@pytest.mark.parametrize("bad", [1, 0, "200", 2.5, True])
def test_n_boot_is_checked(bad):
    model = _fit(sp.Weibull, sp.Weibull.random(20, 10, 2, random_state=7))
    with pytest.raises(ValueError, match="n_boot"):
        model.diagnose(n_boot=bad)


# -- the report ---------------------------------------------------------------
@pytest.fixture(scope="module")
def fits_well():
    x = sp.Weibull.random(60, 10, 2, random_state=11)
    cens = np.random.default_rng(99).uniform(5, 25, 60)
    model = _fit(sp.Weibull, np.minimum(x, cens), c=(x > cens) * 1)
    return model.diagnose(n_boot=40, random_state=1)


def test_data_from_the_model_pass(fits_well):
    d = fits_well
    assert d.ok, d
    assert not d.advice
    assert d.family == "Univariate distribution"
    assert d.model == "Parametric"
    assert list(d.table.columns) == [
        "check",
        "alternative",
        "test",
        "covariates",
        "statistic",
        "df",
        "p",
        "verdict",
        "detail",
    ]
    gof = _rows(d, "goodness_of_fit")
    assert list(gof["test"].str.replace(" (parametric bootstrap)", "")) == [
        "Anderson-Darling",
        "Kolmogorov-Smirnov",
        "Cramer-von Mises",
    ]
    assert (gof["p"] > 0.1).all()
    alternatives = set(_rows(d, "alternatives")["test"])
    assert {
        "AIC: best other distribution (fit_best)",
        "AIC: mixture of two",
        "AIC: offset (3-parameter)",
        "AIC: limited failure population",
    } <= alternatives
    assert any("parametric bootstrap" in n for n in d.notes)


def test_to_dict_round_trips(fits_well):
    out = fits_well.to_dict()
    back = Diagnosis.from_dict(json.loads(json.dumps(out, allow_nan=False)))
    assert back.to_dict() == out
    pd.testing.assert_frame_equal(back.table, fits_well.table)
    assert out["table"][0]["alternative"] == ""


def test_the_report_is_reproducible_with_a_seed():
    model = _fit(sp.LogNormal, sp.LogNormal.random(30, 2, 0.5, random_state=4))
    a = model.diagnose(n_boot=10, random_state=8).to_dict()
    b = model.diagnose(n_boot=10, random_state=8).to_dict()
    assert a == b


def test_alpha_sets_the_verdicts():
    x = sp.LogNormal.random(100, 2, 0.8, random_state=6)
    model = _fit(sp.Weibull, x)
    loose = model.diagnose(alpha=0.05, n_boot=20, random_state=1)
    strict = model.diagnose(alpha=0.01, n_boot=20, random_state=1)
    gof = _rows(loose, "goodness_of_fit")
    assert ((gof["verdict"] == "fail") == (gof["p"] < 0.05)).all()
    # with 20 refits no p-value is below 1/21: nothing fails at 0.01
    assert (_rows(strict, "goodness_of_fit")["verdict"] != "fail").all()


def test_columns_names_no_covariates():
    model = _fit(sp.Weibull, sp.Weibull.random(20, 10, 2, random_state=7))
    with pytest.raises(ValueError, match="'columns'"):
        model.diagnose(columns=["x"])


# -- each advice rule on data that call for it --------------------------------
def test_a_bimodal_sample_is_advised_a_mixture():
    x = np.concatenate(
        [
            sp.Weibull.random(60, 5, 6, random_state=1),
            sp.Weibull.random(60, 30, 6, random_state=2),
        ]
    )
    d = _fit(sp.Weibull, x).diagnose(n_boot=30, random_state=1)
    assert not d.ok
    assert (_rows(d, "goodness_of_fit")["verdict"] == "fail").all()
    assert "MixtureModel(Weibull, 2)" in _kinds(d)
    mixture = [a for a in d.advice if "MixtureModel.fit(m=)" in a.features]
    assert len(mixture) == 1
    assert "dist=surpyval.Weibull, m=2" in mixture[0]
    assert "fits best" in mixture[0]


def test_an_offset_sample_is_advised_an_offset():
    x = 50 + sp.Weibull.random(80, 20, 1.5, random_state=3)
    d = _fit(sp.Weibull, x).diagnose(n_boot=30, random_state=1)
    assert not d.ok
    assert "Weibull offset" in _kinds(d)
    offset = [a for a in d.advice if "Weibull.fit(offset=)" in a.features]
    assert len(offset) == 1
    assert "Weibull.fit(x, c, n, t, offset=True)" in offset[0]
    assert "fits best" in offset[0]
    assert offset[0].check == "alternatives"
    # A Gamma names its own fit
    g = _fit(sp.Gamma, x).diagnose(n_boot=10, random_state=1)
    features = {f for a in g.advice for f in a.features}
    assert "Gamma.fit(offset=)" in features


def test_a_limited_failure_population_is_advised_lfp():
    rng = np.random.default_rng(3)
    t = sp.Weibull.random(150, 10, 2, random_state=4)
    t[rng.uniform(size=150) < 0.4] = np.inf
    c = (t > 25).astype(int)
    d = _fit(sp.Weibull, np.minimum(t, 25), c=c).diagnose(
        n_boot=30, random_state=1
    )
    assert "Weibull lfp" in _kinds(d)
    lfp = [a for a in d.advice if "Weibull.fit(lfp=)" in a.features]
    assert len(lfp) == 1 and "fits best" in lfp[0]
    # The lfp model itself has no lfp row and fits
    fitted = _fit(sp.Weibull, np.minimum(t, 25), c=c, lfp=True)
    d = fitted.diagnose(n_boot=20, random_state=1)
    assert "Weibull lfp" not in set(_rows(d, "alternatives")["alternative"])
    assert (_rows(d, "goodness_of_fit")["verdict"] != "fail").all(), d


def test_failures_at_zero_are_advised_zero_inflation():
    x = np.concatenate(
        [np.zeros(20), sp.Weibull.random(100, 10, 2, random_state=5)]
    )
    d = _fit(sp.Normal, x).diagnose(n_boot=20, random_state=1)
    zi = _rows(d, "alternatives", "warn")
    zi = zi[zi["test"] == "AIC: zero-inflated"]
    assert len(zi) == 1
    name = zi["alternative"].item().split(" ")[0]
    advice = [a for a in d.advice if a.features == (name + ".fit(zi=)",)]
    assert len(advice) == 1 and "zi=True" in advice[0]
    # The zeros are the zero-inflation row's, not the ties note's
    assert not any("tie" in n for n in d.notes)


def test_another_distribution_is_advised_fit_best():
    x = sp.LogNormal.random(100, 2, 0.8, random_state=6)
    d = _fit(sp.Weibull, x).diagnose(n_boot=30, random_state=1)
    assert not d.ok
    best = _rows(d, "alternatives", "warn")
    best = best[best["test"].str.contains("fit_best")]
    assert best["alternative"].item() == "LogNormal"
    advice = [a for a in d.advice if "fit_best(return_table=)" in a.features]
    assert len(advice) == 1 and advice[0].startswith("LogNormal fits")


def test_an_unexplained_failure_is_advised_the_plot(monkeypatch):
    x = sp.LogNormal.random(100, 2, 0.8, random_state=6)
    monkeypatch.setattr(univariate, "_alternatives", lambda ctx: [])
    d = _fit(sp.Weibull, x).diagnose(n_boot=30, random_state=1)
    assert not d.ok
    assert len(d.advice) == 1
    assert d.advice[0].check == "goodness_of_fit"
    assert "d.plot('probability_plot')" in d.advice[0]
    assert "KaplanMeier.fit" in d.advice[0].features


# -- what the family supports -------------------------------------------------
def test_truncated_and_fixed_fits_are_refitted_as_fitted():
    x = sp.Weibull.random(80, 10, 2, random_state=3)
    x = x[x > 4]
    d = _fit(sp.Weibull, x, tl=4).diagnose(n_boot=20, random_state=1)
    assert d.ok, d
    fixed = _fit(sp.Weibull, x, tl=4, fixed={"beta": 2.0})
    assert univariate._refit_options(fixed)["fixed"] == {"beta": 2.0}
    d = fixed.diagnose(n_boot=20, random_state=1)
    assert d.ok, d
    # A mixture is no alternative to a model with a fixed parameter
    assert "AIC: mixture of two" not in set(d.table["test"])


def test_a_fit_not_by_mle_is_compared_at_its_mle():
    x = sp.Weibull.random(40, 10, 2, random_state=1)
    d = _fit(sp.Weibull, x, how="MPP").diagnose(n_boot=10, random_state=1)
    assert any("fitted by MPP" in n for n in d.notes)
    assert univariate._refit_options(_fit(sp.Weibull, x, how="MPP"))[
        "how"
    ] == ("MPP")


def test_interval_censored_data_have_no_bootstrap():
    model = _fit(sp.Weibull, xl=[1, 2, 3, 4, 5, 6], xr=[2, 3, 5, 6, 8, 9])
    d = model.diagnose()
    gof = _rows(d, "goodness_of_fit")
    assert len(gof) == 3
    assert (gof["verdict"] == "warn").all() and gof["p"].isna().all()
    assert gof["detail"].str.contains("inspection times").all()
    assert not any("parametric bootstrap" in n for n in d.notes)
    assert d.ok


def test_ties_are_noted():
    x = np.round(sp.Weibull.random(60, 10, 2, random_state=2))
    x = x[x > 0]
    d = _fit(sp.Weibull, x).diagnose(n_boot=10, random_state=1)
    assert any("rounded" in n for n in d.notes)


# -- refusals -----------------------------------------------------------------
def test_a_model_without_data_is_refused():
    with pytest.raises(ValueError, match="with_data=True"):
        sp.Weibull.from_params([10, 2]).diagnose()


def test_no_finite_maximum_is_refused():
    model = _fit(sp.Beta4, [1, 2, 3, 4, 5, 6, 7])
    assert model.maximum == "no finite maximum"
    with pytest.raises(ValueError, match="no finite maximum"):
        model.diagnose()


@pytest.mark.parametrize(
    "fit, error, match",
    [
        (lambda: sp.Bernoulli.fit([0, 1, 1, 0, 1]), ValueError, "param_cb"),
        (
            lambda: sp.FixedEventProbability.fit([0, 1, 1]),
            ValueError,
            "param_cb",
        ),
        (
            lambda: sp.ExactEventTime.fit([1, 2, 3], c=[1, -1, -1]),
            ValueError,
            "point mass",
        ),
        (
            lambda: sp.Geometric.fit([1, 2, 3, 4, 2, 1]),
            NotImplementedError,
            "#875",
        ),
    ],
)
def test_models_without_a_distribution_to_test_are_refused(fit, error, match):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit()
    with pytest.raises(error, match=match):
        model.diagnose()


# -- plots --------------------------------------------------------------------
def test_plot_draws_the_visual_checks(fits_well):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = fits_well.plot()
    titles = [ax.get_title() for ax in fig.axes]
    assert len(titles) == 3
    assert titles[0].startswith("P-P plot (AD p = ")
    assert titles[2] == "Weibull probability plot"
    plt.close(fig)
    fig = fits_well.plot("probability_plot")
    assert len(fig.axes) == 1
    plt.close(fig)
    with pytest.raises(ValueError, match="'which'"):
        fits_well.plot("alternatives")


def test_plot_of_interval_censored_data_is_the_probability_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = _fit(sp.Weibull, xl=[1, 2, 3, 4, 5, 6], xr=[2, 3, 5, 6, 8, 9])
    fig = model.diagnose().plot()
    assert [ax.get_title() for ax in fig.axes] == ["Weibull probability plot"]
    plt.close(fig)
