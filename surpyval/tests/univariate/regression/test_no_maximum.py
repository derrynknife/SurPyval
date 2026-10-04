"""Regression fits whose likelihood has no finite maximum warn (#392).

A covariate that is 1 on exactly the censored rows is a group with no
events: its coefficient has no finite maximum-likelihood estimate, as
the likelihood keeps rising towards a supremum it never reaches. The
optimisers stopped where the rise fell below their tolerances (a
WeibullPH coefficient of -14.7, a PO one of +33) and the fit returned
silently. It now warns, once, pointing at the caller, and still
returns what it reached; an ordinary fit does not warn.

The check is Newton's (``fitters.runaway.runaway_coefficients``): along the
coefficient's profile the Kantorovich quantity ``h = |f'''| |f'| / f''^2``
is at the level of the optimiser's tolerance at a maximum and about 1 on
the way to a supremum.
"""

import warnings

import autograd.numpy as anp
import numpy as np
import pandas as pd
import pytest
from autograd import grad, hessian

import surpyval as sp
from surpyval import CoxPH
from surpyval.tests.conformance.registry import grouped_reg_data, reg_data
from surpyval.univariate.competing_risks import FineGray
from surpyval.univariate.competing_risks.regression import (
    CompetingRisksProportionalHazards,
)
from surpyval.univariate.parametric.fitters import runaway
from surpyval.univariate.parametric.fitters.runaway import (
    _flat_at_start,
    runaway_coefficients,
)
from surpyval.univariate.regression import _fit_skeleton as skeleton

NO_MAXIMUM = "No finite maximum: the likelihood keeps increasing"
MONOTONE = "No finite maximum: the partial likelihood"


def _no_events(d):
    """The first covariate 1 on exactly the censored rows."""
    Z = np.array(d["Z"], dtype=float)
    Z[:, 0] = np.asarray(d["c"]) == 1
    return {**d, "Z": Z}


def _fit(fit):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = fit()
    return model, [x for x in w if issubclass(x.category, UserWarning)]


# One of each family and link, and the baselines that needed care: the
# Weibull and Exponential PH are fitted on centred covariates (the runaway
# direction moves the scale too), LogNormal AFT's tail is Gaussian, and
# the frailty variance of these data runs to its boundary of 0.
PARAMETRIC = [
    "WeibullPH",
    "ExponentialPH",
    "GammaPH",
    "LogNormalAFT",
    "ExponentialAFT",
    "WeibullAFT",
    "LogisticPO",
    "LogNormalPO",
]


@pytest.mark.parametrize("name", PARAMETRIC)
def test_a_level_with_no_events_warns_once(name):
    fitter = getattr(sp, name)
    model, w = _fit(lambda: fitter.fit(**_no_events(reg_data())))
    assert len(w) == 1, [str(x.message) for x in w]
    message = str(w[0].message)
    assert message.startswith(NO_MAXIMUM)
    assert "coefficient(s) [0]" in message
    assert w[0].filename == __file__
    # The fit still returns what it reached: a large coefficient. There
    # is no optimum, so where the search stops is arbitrary; the gradient
    # ladder AFT and PO take first (#499) stops sooner along LogNormal
    # AFT's Gaussian tail (-3.7, was -4.8), well out on the runaway.
    assert np.all(np.isfinite(model.params))
    assert abs(model.phi_params[0]) > 3


@pytest.mark.parametrize("name", PARAMETRIC)
def test_an_ordinary_fit_does_not_warn(name):
    _, w = _fit(lambda: getattr(sp, name).fit(**reg_data()))
    assert not w, [str(x.message) for x in w]


def test_the_issue_example():
    # d["Z"][:, 0] = d["c"] == 1; WeibullPH.fit(**d) gave -16.31 silently.
    # On the runaway where the search stops is arbitrary (-15.78 with
    # scipy 1.17, -15.08 with 1.18): what matters is the warning, and that
    # the coefficient has run far out.
    model, w = _fit(lambda: sp.WeibullPH.fit(**_no_events(reg_data())))
    assert [str(x.message)[: len(NO_MAXIMUM)] for x in w] == [NO_MAXIMUM]
    assert model.phi_params[0] < -10


def test_fit_from_df_points_at_the_caller():
    d = _no_events(reg_data())
    df = pd.DataFrame(d["Z"], columns=["z0", "z1"])
    df["x"], df["c"], df["n"] = d["x"], d["c"], d["n"]
    _, w = _fit(
        lambda: sp.WeibullPH.fit_from_df(
            df, x_col="x", Z_cols=["z0", "z1"], c_col="c", n_col="n"
        )
    )
    assert len(w) == 1 and str(w[0].message).startswith(NO_MAXIMUM)
    assert w[0].filename == __file__


def test_a_fixed_coefficient_keeps_the_numbering():
    d = _no_events(reg_data())
    # The runaway coefficient is still called 0 with the other fixed ...
    _, w = _fit(lambda: sp.WeibullAFT.fit(**d, fixed={"coef_1": -0.3}))
    assert len(w) == 1 and "coefficient(s) [0]" in str(w[0].message)
    # ... and with it fixed there is nothing to run away.
    _, w = _fit(lambda: sp.WeibullAFT.fit(**d, fixed={"coef_0": -1.0}))
    assert not w, [str(x.message) for x in w]


@pytest.mark.parametrize(
    "name", ["WeibullAH", "LogNormalAH", "ExponentialAH", "LogisticAH"]
)
def test_additive_hazards_warn_once_that_there_is_no_maximum(name):
    # The additive model's likelihood rises linearly without bound as the
    # no-event level's coefficient falls; it said instead that the fit had
    # ended on the positivity boundary, or had not converged.
    model, w = _fit(lambda: getattr(sp, name).fit(**_no_events(reg_data())))
    assert len(w) == 1, [str(x.message) for x in w]
    assert str(w[0].message).startswith(NO_MAXIMUM)
    assert w[0].filename == __file__
    assert model.phi_params[0] < -100


@pytest.mark.parametrize(
    "name",
    [
        "WeibullFrailty",
        "ExponentialFrailty",
        "GammaFrailty",
        "LogNormalFrailty",
    ],
)
def test_frailty_warns_once(name):
    d = _no_events(grouped_reg_data())
    model, w = _fit(lambda: getattr(sp, name).fit(**d))
    assert len(w) == 1, [str(x.message) for x in w]
    assert str(w[0].message).startswith(NO_MAXIMUM)
    assert w[0].filename == __file__
    assert model.beta[0] < -9
    _, w = _fit(lambda: getattr(sp, name).fit(**grouped_reg_data()))
    assert not w, [str(x.message) for x in w]


def _competing(d):
    e = np.where(d["c"] == 1, None, np.where(np.arange(30) % 3, "a", "b"))
    return {"x": d["x"], "Z": d["Z"], "e": e, "n": d["n"]}


def test_fine_gray_warns_as_cox_does():
    # BFGS reported success at -12.2, and the fit returned silently.
    d = _competing(_no_events(reg_data()))
    model, w = _fit(lambda: FineGray.fit(**d, event="a"))
    assert len(w) == 1, [str(x.message) for x in w]
    assert str(w[0].message).startswith(MONOTONE)
    assert "coefficient(s) [0] grow" in str(w[0].message)
    assert w[0].filename == __file__
    assert model.beta[0] < -10
    _, w = _fit(lambda: FineGray.fit(**_competing(reg_data()), event="a"))
    assert not w, [str(x.message) for x in w]


def test_competing_risks_fine_gray_warns_once_for_both_causes():
    d = _competing(_no_events(reg_data()))
    _, w = _fit(
        lambda: CompetingRisksProportionalHazards.fit(**d, model="Fine-Gray")
    )
    assert len(w) == 1, [str(x.message) for x in w]
    message = str(w[0].message)
    assert message.startswith(MONOTONE)
    assert "[0] (cause 'a') and [0] (cause 'b')" in message
    assert w[0].filename == __file__


def test_collinear_covariates_are_not_taken_for_a_runaway():
    # Every level of a factor coded (as "0 + C(g)" does): the likelihood
    # does not depend on their sum at all, so along it the derivatives are
    # rounding, and the Fine-Gray fit warned of a monotone likelihood. The
    # direction is flat at the start too, which a runaway's is not. The
    # last level is aliased (#476): that is the one warning.
    d = reg_data()
    g = np.arange(30) // 10  # each level has events of both causes
    Z = np.column_stack([g == 0, g == 1, g == 2, d["Z"][:, 1]]).astype(float)
    e = _competing(d)["e"]
    _, w = _fit(lambda: FineGray.fit(d["x"], Z, e, n=d["n"], event="a"))
    aliased = "Covariate column(s) 2 of Z cannot be estimated"
    assert [str(x.message)[:46] for x in w] == [aliased]
    for fitter in (sp.WeibullPH, sp.LogNormalAFT):
        _, w = _fit(lambda: fitter.fit(x=d["x"], Z=Z, c=d["c"], n=d["n"]))
        assert [str(x.message)[:46] for x in w] == [aliased]
    # Scaling a Weibull's survival odds leaves the Weibull family, so the
    # proportional odds model has no intercept to alias the sum with: the
    # coefficients are identified (weakly), and nothing is said.
    _, w = _fit(lambda: sp.WeibullPO.fit(x=d["x"], Z=Z, c=d["c"], n=d["n"]))
    assert not w, [str(x.message) for x in w]


# -- the criterion itself -----------------------------------------------------


def test_a_supremum_at_infinity_is_found_along_the_profile():
    # f(t, u) = exp(t) + 50 (u - t)^2 approaches its infimum as t -> -inf
    # with u following t: along t alone (u held) it has a minimum, so the
    # profile, not the axis, is what shows it.
    def f(p):
        return anp.exp(p[0]) + 50.0 * (p[1] - p[0]) ** 2

    assert runaway_coefficients(f, [-15.0, -15.0], [0]) == [0]
    # A quadratic's minimum is found, and so is a sharper one near it.
    assert runaway_coefficients(lambda p: (p[0] - 1) ** 2, [1.0], [0]) == []
    assert runaway_coefficients(lambda p: anp.cosh(p[0]), [1e-7], [0]) == []


def test_a_linear_rise_has_no_maximum():
    assert runaway_coefficients(lambda p: -3.0 * p[0], [-50.0], [0]) == [0]


def test_a_parameter_the_likelihood_ignores_gives_no_verdict():
    # A column of zeros: nothing to say about its coefficient.
    assert runaway_coefficients(lambda p: p[1] ** 2, [0.0, 0.0], [0]) == []


def test_a_direction_flat_at_the_start_too_gives_no_verdict():
    # f depends on p0 + p1 only: along (1, -1) it is flat everywhere, so
    # what its derivatives show at a fit is rounding; the start says so.
    def f(p):
        return (p[0] + p[1] - 1.0) ** 2 + anp.exp(-(p[0] + p[1]))

    assert _flat_at_start(f, [0.0, 0.0], np.array([1.0, -1.0]))
    assert not _flat_at_start(f, [0.0, 0.0], np.array([1.0, 0.0]))
    # A runaway's direction is not flat there: it curves (the profile of
    # exp(t) + 50 (u - t)^2, along which u follows t) ...
    g = lambda p: anp.exp(p[0]) + 50.0 * (p[1] - p[0]) ** 2  # noqa: E731
    assert not _flat_at_start(g, [0.0, 0.0], np.array([1.0, 1.0]))
    assert runaway_coefficients(g, [-15.0, -15.0], [0], [0.0, 0.0]) == [0]
    # ... or slopes (a linear rise).
    assert not _flat_at_start(lambda p: -3.0 * p[0], [0.0], np.ones(1))


def test_collinear_fine_gray_formula_does_not_warn():
    # "0 + C(g)" codes every level, so the columns sum to 1: the partial
    # likelihood does not depend on their sum. Along it the derivatives at
    # the fit are rounding, and on these data they looked like a runaway
    # ("No finite maximum" for coefficient 2 of cause "u").
    rng = np.random.default_rng(3)
    n = 300
    g = rng.choice(["a", "b", "c"], n)
    rng.integers(1, 4, n)  # (the formula tests' draws, in their order)
    z = rng.uniform(1, 3, n)
    eff = 0.4 * z + np.select([g == "b", g == "c"], [0.5, -0.4], 0.0)
    t = 10 * rng.weibull(1.5, n) * np.exp(-eff / 1.5)
    c = (rng.uniform(size=n) < 0.2).astype(int)
    rng.uniform(0, 1, n)
    df = pd.DataFrame(
        {
            "t": t,
            "c": c,
            "z": z,
            "g": g,
            "cause": np.where(c == 1, None, rng.choice(["u", "v"], n)),
        }
    )
    _, w = _fit(
        lambda: CompetingRisksProportionalHazards.fit_from_df(
            df,
            "t",
            "cause",
            c_col="c",
            formula="0 + C(g) + poly(z, 2)",
            model="Fine-Gray",
        )
    )
    # The last level is aliased (#476), once for both causes, by name.
    assert len(w) == 1, [str(x.message) for x in w]
    assert str(w[0].message).startswith(
        "Covariate column(s) 2 ('C(g)[c]') of Z cannot be estimated"
    )


# -- the gate: profiles are read only where Newton has not converged ----------


def _count_profiles(monkeypatch):
    calls = []
    profile = runaway._profile

    def counted(neg_ll, x, H, j):
        calls.append(j)
        return profile(neg_ll, x, H, j)

    monkeypatch.setattr(runaway, "_profile", counted)
    return calls


@pytest.mark.parametrize(
    "fit",
    [
        lambda: sp.WeibullPH.fit(**reg_data()),
        lambda: sp.LogNormalAFT.fit(**reg_data()),
        lambda: sp.LogisticPO.fit(**reg_data()),
        lambda: sp.WeibullAH.fit(**reg_data()),
        lambda: FineGray.fit(**_competing(reg_data()), event="a"),
    ],
    ids=["PH", "AFT", "PO", "AH", "FineGray"],
)
def test_an_ordinary_fit_reads_no_profile(monkeypatch, fit):
    # Its Newton step is at the optimiser's tolerance in every coefficient,
    # so the third derivatives are never taken.
    calls = _count_profiles(monkeypatch)
    _, w = _fit(fit)
    assert calls == [] and not w, [str(x.message) for x in w]


def test_an_ordinary_frailty_fit_is_quiet():
    # The frailty variance of these data sits at its limit of 0. Whether
    # the Newton step then reaches the optimiser's tolerance in every
    # coefficient depends on where scipy's BFGS stops (1.17 yes; 1.18 stops
    # at a gradient of 1.6e-5 and reads two profiles): either way the
    # verdict is an ordinary maximum, without a warning.
    _, w = _fit(lambda: sp.WeibullFrailty.fit(**grouped_reg_data()))
    assert not w, [str(x.message) for x in w]


@pytest.mark.parametrize(
    "name", ["WeibullPH", "LogNormalAFT", "LogisticPO", "LogNormalPO"]
)
def test_a_runaway_has_its_profile_read(monkeypatch, name):
    # The runaway coefficient's (0) profile is read. The other's step is at
    # the gate's tolerance, so whether it is cleared without a profile
    # depends on the last bits of the search (it is not on some CPUs);
    # either way only coefficient 0 runs away. LogNormalAFT and the PO fits
    # run furthest onto the plateau (profile information 1e-13 to 1e-15 of
    # the start's). The baseline's parameters are checked after the
    # coefficients, and one the gate does not clear on the plateau has its
    # profile read too (#634).
    calls = _count_profiles(monkeypatch)
    _, w = _fit(lambda: getattr(sp, name).fit(**_no_events(reg_data())))
    assert len(w) == 1 and "coefficient(s) [0]" in str(w[0].message)
    k_dist = len(getattr(sp, name).parameter_names)
    assert calls[0] == k_dist
    assert set(calls) <= {k_dist, k_dist + 1, *range(k_dist)}


def test_the_gate_clears_a_maximum_and_nothing_else():
    # A maximum: the Newton step is 0, well within 1/709.8 of the value.
    quadratic = lambda p: (p[0] - 2.0) ** 2 + (p[1] + 3.0) ** 2  # noqa: E731
    x = np.array([2.0, -3.0])
    H, g = hessian(quadratic)(x), grad(quadratic)(x)
    assert runaway._cleared(x, H, g).tolist() == [True, True]
    # A runaway along t = u (exp(t) + 50 (u - t)^2 at t = u = -15): the
    # step is (1, 1), 1/15 of the value, so neither is cleared.
    running = lambda p: anp.exp(p[0]) + 50.0 * (p[1] - p[0]) ** 2  # noqa
    x = np.array([-15.0, -15.0])
    H, g = hessian(running)(x), grad(running)(x)
    assert runaway._cleared(x, H, g).tolist() == [False, False]
    # A linear rise has no curvature: no Hessian to trust, nothing cleared.
    linear = lambda p: -3.0 * p[0] + p[1] ** 2  # noqa: E731
    x = np.array([-50.0, 0.0])
    H, g = hessian(linear)(x), grad(linear)(x)
    assert runaway._cleared(x, H, g).tolist() == [False, False]
    # A parameter the likelihood does not depend on is left out of the
    # step, and the others are still cleared.
    ignores = lambda p: (p[0] - 1.0) ** 2 + 0.0 * p[1]  # noqa: E731
    x = np.array([1.0, 7.0])
    H, g = hessian(ignores)(x), grad(ignores)(x)
    assert runaway._cleared(x, H, g).tolist() == [True, False]


# -- the cost of reading a profile (#501) -------------------------------------


@pytest.mark.parametrize("name", ["LogNormalAFT", "WeibullAFT", "WeibullPH"])
def test_a_profile_is_read_without_more_hessians(monkeypatch, name):
    # The covariate is +1 and -1 on two copies of the same data, so its
    # coefficient is exactly 0 at the maximum: no Newton step is small
    # beside 0, and its profile is read. The curvatures a quarter step
    # either side of the fit come from Hessian-vector products there; the
    # full Hessian is formed once, at the fit, for the covariance (#501:
    # twice more for each profile, which on 100,000 rows took 65% of a
    # LogNormal AFT fit).
    rng = np.random.default_rng(0)
    x = np.exp(2 + 0.5 * rng.normal(size=20))
    c = (rng.uniform(size=20) < 0.3).astype(int)
    Z = np.concatenate([np.ones(20), -np.ones(20)])[:, None]
    calls = _count_profiles(monkeypatch)
    hessians = []
    full = skeleton.search_derivatives

    def counted(*args):
        hessians.append(1)
        return full(*args)

    monkeypatch.setattr(skeleton, "search_derivatives", counted)
    monkeypatch.setattr(runaway, "search_derivatives", counted)
    fitter = getattr(sp, name)
    model, w = _fit(lambda: fitter.fit(x=np.tile(x, 2), c=np.tile(c, 2), Z=Z))
    assert not w, [str(x.message) for x in w]
    assert calls == [len(fitter.parameter_names)]
    assert len(hessians) == 1
    assert model.params[-1] == pytest.approx(0.0, abs=1e-6)


def test_the_profile_curvature_from_products_is_exact():
    # exp(t) + 50 (u - t)^2: the profile of t is exp(t), and so is its
    # curvature. Formed from the full Hessian, H_tt - H_tu^2 / H_uu loses
    # the digits of exp(t) ~ 3e-7 that cancel against the 100s (1e-8 of
    # it); from Hessian-vector products along the profile it is exact.
    def f(p):
        return anp.exp(p[0]) + 50.0 * (p[1] - p[0]) ** 2

    at = np.array([-15.0, -15.0])
    H, _ = runaway.search_derivatives(f, at)
    v = np.array([1.0, 1.0])
    for t in (-14.9, -15.0, -15.2):
        point = np.array([t, -15.0])
        S = runaway._profile_curvature(f, point, 0, H, v)
        assert S == pytest.approx(np.exp(t), rel=1e-14, abs=0)


# -- all the failures in one cell of a two-stress test (#628) -----------------


def _alt(draw):
    """#583's accelerated life test (Arrhenius in temperature, inverse power
    in voltage, Weibull shape 2.2, 12 units a cell, ended at 3000 h) at
    c = 600, draw ``draw`` of #617's study."""
    T, V = np.meshgrid(
        np.array([85.0, 105.0, 125.0]) + 273.15, [450.0, 500.0], indexing="ij"
    )
    Z = np.repeat(np.column_stack([T.ravel(), V.ravel()]), 12, axis=0)
    rng = np.random.default_rng([617, 600, draw])
    life = 600.0 * np.exp(0.7 / 8.617e-5 / Z[:, 0]) * Z[:, 1] ** -3.0
    t = life * rng.weibull(2.2, len(Z))
    return np.minimum(t, 3000.0), (t > 3000.0).astype(int), Z


def _one_cell_alt():
    """Draw 54: all six failures are in the 125 C / 500 V cell, so both
    coefficients run off together (any direction that lengthens every
    other cell's life raises the likelihood)."""
    x, c, Z = _alt(54)
    assert np.all(Z[c == 0] == [398.15, 500.0]) and np.sum(c == 0) == 6
    return x, c, Z


@pytest.mark.parametrize(
    "fit",
    [
        lambda x, c, Z: sp.WeibullAFT.fit(x, _alt_terms(Z), c=c),
        lambda x, c, Z: sp.WeibullPH.fit(x, _alt_terms(Z), c=c),
        lambda x, c, Z: sp.LogNormalAFT.fit(x, _alt_terms(Z), c=c),
        lambda x, c, Z: sp.WeibullPO.fit(x, _alt_terms(Z), c=c),
    ],
    ids=["WeibullAFT", "WeibullPH", "LogNormalAFT", "WeibullPO"],
)
def test_628_all_failures_in_one_cell_have_no_finite_maximum(fit):
    # WeibullAFT reached a scale of 9.1e134 and coefficients -52674 and 70
    # and called it a verified maximum, without a word: the likelihood is
    # within 1e-5 of its supremum there, so the gradient test passes, and
    # the no-maximum check, made in the search's own units, could not see
    # the run-off (WeibullPH and WeibullPO likewise).
    model, w = _fit(lambda: fit(*_one_cell_alt()))
    assert model.maximum == "no finite maximum"
    assert len(w) == 1, [str(x.message) for x in w]
    assert str(w[0].message).startswith(NO_MAXIMUM)
    assert w[0].filename == __file__


def _alt_terms(Z):
    return np.column_stack([1.0 / Z[:, 0], np.log(Z[:, 1])])


def test_628_an_accelerated_life_fit_short_of_its_maximum_is_no_run_off():
    # Draw 4 has failures at enough stresses for a finite maximum (WeibullAFT
    # on the same model finds it at -log L = 100.28769). The accelerated
    # life fit, whose ``c`` is searched linearly, stopped 4e-4 short of it
    # and warned "No finite maximum" for ``c``: Newton's test made along
    # ``c`` rather than ``log c``.
    x, c, Z = _alt(4)
    fitter = sp.AcceleratedLife(sp.Weibull, sp.life_models.PowerExponential)
    model, w = _fit(lambda: fitter.fit(x, Z, c=c))
    assert model.maximum != "no finite maximum"
    assert not any(str(m.message).startswith(NO_MAXIMUM) for m in w)
    assert model.neg_ll() == pytest.approx(100.28769, abs=1e-3)


def test_628_a_one_bounded_parameter_is_judged_on_its_log_scale():
    # A likelihood quadratic in log(a), with ``a`` searched linearly beyond
    # 1 (as ``bounds_convert`` searches a scale): a point short of its
    # maximum at a = e^5 - 1 looks like a run-off along ``a`` read linearly
    # (an accelerated life ``c`` at 1e22 warned "No finite maximum"), not
    # on the log scale.
    def f(p):
        return (anp.log1p(p[0]) - 3.0) ** 2

    for u in (5.0, 20.0):
        x = np.array([np.expm1(u)])
        assert runaway_coefficients(f, x, [0], [0.0]) == [0]
        assert (
            runaway.runaways_in_units(f, x, [0], [0.0], one_sided=(0,)) == []
        )


def test_628_a_profile_flat_to_rounding_has_no_maximum():
    # exp(t) at t = -800 has underflowed: no curvature at all where the
    # likelihood depended on t at the start. A parameter that never enters
    # it is flat at the start too, and is not called a run-off.
    def f(p):
        return anp.exp(p[0]) + (p[1] - 1.0) ** 2

    x, start = np.array([-800.0, 1.0]), np.zeros(2)
    derivatives = runaway.search_derivatives(f, x)
    assert runaway.flat_profiles(f, x, [0], start, derivatives) == [0]

    def ignores(p):
        return (p[1] - 1.0) ** 2 + 0.0 * p[0]

    derivatives = runaway.search_derivatives(ignores, x)
    assert runaway.flat_profiles(ignores, x, [0], start, derivatives) == []


# -- no-maximum follow-ups (#634) ---------------------------------------------


def _po(x, c, Z):
    return x, _alt_terms(Z), c


def test_634_a_baseline_parameter_running_off_is_named():
    # All six failures in one cell of #583's test: the WeibullPO's alpha
    # runs off with the coefficients. The check looked at the coefficients
    # only; the baseline's parameters are checked too, and named with
    # their values.
    model, w = _fit(lambda: sp.WeibullPO.fit(*_po(*_one_cell_alt())))
    assert model.maximum == "no finite maximum"
    assert len(w) == 1, [str(x.message) for x in w]
    message = str(w[0].message)
    assert message.startswith(NO_MAXIMUM)
    assert "the Weibull baseline's alpha (" in message
    assert "compare the fits with other baselines" in message


def test_634_the_message_names_coefficients_and_baseline_together():
    verdict = skeleton.SearchVerdict(None, "no finite maximum", None, [1])
    what, advice = skeleton._no_maximum_message(verdict)
    assert what == skeleton.NO_MAXIMUM_WHAT.format([1])
    both = verdict._replace(baseline=("alpha",))
    what, advice = skeleton._no_maximum_message(
        both, "Weibull", {"alpha": 2.5e9}
    )
    assert "coefficient(s) [1]" in what
    assert "and as the Weibull baseline's alpha (2.5e+09) runs on" in what
    assert "other baselines" in advice


def test_634_a_flat_fit_short_of_its_maximum_is_not_verified():
    # A likelihood so flat in one parameter that the gradient test passes
    # 4e-3 nats below the maximum (a WeibullPO was "verified" 0.006 below
    # its maximum, alpha 14 times short of it): the Newton decrement
    # g' H^-1 g / 2 says how far, in nats.
    def f(p):
        return 4e-7 * (p[0] - 100.0) ** 2 + (p[1] - 1.0) ** 2

    x = np.array([0.0, 1.0])
    derivatives = runaway.search_derivatives(f, x)
    gain = skeleton.newton_gain(x, derivatives, [0, 1])
    assert gain == pytest.approx(4e-3)
    assert not skeleton.is_verified(x, derivatives, 1.0)
    # A point a hair short of the maximum is verified
    at = np.array([100.0 - 1e-3, 1.0])
    assert skeleton.is_verified(at, runaway.search_derivatives(f, at), 1.0)


def test_634_weibull_po_reaches_its_flat_maximum():
    # Draw 167: the gradient of the PO survival was nan at the censored
    # rows once alpha passed 1e9 (log1p(-1) in the branch np.where did not
    # take), and the fit was "verified" 0.006 below its maximum, alpha
    # 7.3e11 rather than 1.2e13.
    x, c, Z = _alt(167)
    model, w = _fit(lambda: sp.WeibullPO.fit(x, _alt_terms(Z), c=c))
    assert not w, [str(m.message) for m in w]
    assert model.maximum == "verified"
    assert model.neg_ll() == pytest.approx(105.02012, abs=1e-4)
    assert model.params[0] > 5e12


def test_634_weibull_po_derivatives_are_finite_far_out():
    # The survival as 1 / (1 + F0 / (phi S0)) had a second derivative of
    # 1 / (phi S0)^3, inf at phi S0 = 1e-115 (alpha at 6e43).
    x, c, Z = _alt(9)
    T = _alt_terms(Z)
    censored = c == 1

    def log_sf(q):
        return sp.WeibullPO.log_sf(x[censored], T[censored], *q).sum()

    far = np.array([6.39e43, 2.877, 1.0311e4, -47.06])
    assert np.all(np.isfinite(hessian(log_sf)(far)))
    nearer = np.array([2.4e9, 3.1, 1.2e4, -11.7])
    assert np.all(np.isfinite(grad(log_sf)(nearer)))


def test_634_the_refusal_at_z_0_says_there_may_be_no_maximum():
    # Draw 9: the coefficient of log V runs off, and moving the baseline
    # to Z = 0 overflows. The refusal stays, and says why it happened and
    # that center=True fits, with the "No finite maximum" warning.
    x, c, Z = _alt(9)
    with pytest.raises(ValueError) as info:
        sp.WeibullPH.fit(x, _alt_terms(Z), c=c)
    message = str(info.value)
    assert "cannot be represented" in message
    assert "may have no finite maximum" in message
    assert "coefficient(s) [1]" in message
    assert "center=True" in message
    model, w = _fit(
        lambda: sp.WeibullPH.fit(x, _alt_terms(Z), c=c, center=True)
    )
    assert model.maximum == "no finite maximum"
    assert [str(m.message)[:22] for m in w] == ["No finite maximum: the"]


def _power_exponential(draw):
    x, c, Z = _alt(draw)
    fitter = sp.AcceleratedLife(sp.Weibull, sp.life_models.PowerExponential)
    return _fit(lambda: fitter.fit(x, Z, c=c))


def test_634_an_accelerated_life_c_far_above_1_is_found():
    # Draw 0: c was searched linearly beyond 1 and stopped short at 1.3e22,
    # "unverified". On its log scale the fit reaches the maximum, as the
    # WeibullAFT of the same model does.
    model, w = _power_exponential(0)
    assert not w, [str(m.message) for m in w]
    assert model.maximum == "verified"
    x, c, Z = _alt(0)
    aft = sp.WeibullAFT.fit(x, _alt_terms(Z), c=c)
    assert model.neg_ll() == pytest.approx(aft.neg_ll(), abs=1e-4)


def test_634_an_accelerated_life_run_off_with_c_to_0_is_found():
    # Draw 288: c ran down to 1e-294 (a log below 1, but its life computed
    # from c itself, which underflows), and the fit could only say
    # "unverified", 1.2 below the WeibullAFT's run-off. With the life one
    # exponent of log c, the run-off is seen.
    model, w = _power_exponential(288)
    assert model.maximum == "no finite maximum"
    assert len(w) == 1 and str(w[0].message).startswith(NO_MAXIMUM)
    x, c, Z = _alt(288)
    aft, _ = _fit(lambda: sp.WeibullAFT.fit(x, _alt_terms(Z), c=c))
    assert model.neg_ll() == pytest.approx(aft.neg_ll(), abs=1e-3)


def test_634_an_accelerated_life_whose_arrhenius_factor_overflows():
    # Draw 54 (all failures in one cell): e^(a / U) overflowed though the
    # life was finite, and the fit said "unverified"; no raw numpy warning
    # escapes either.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model, w = _power_exponential(54)
    assert model.maximum == "no finite maximum"
    assert len(w) == 1 and str(w[0].message).startswith(NO_MAXIMUM)


def test_634_the_life_is_one_exponent():
    # e^(a / U) = e^800 overflows; c e^(a / U) is e^109
    Z = np.array([[1.0, 2.0]])
    life = sp.life_models.PowerExponential.phi(Z, 1e-300, 800.0, 0.0)
    assert life[0] == pytest.approx(np.exp(800.0 + np.log(1e-300)))


# ---------------------------------------------------------------------------
# Cox warns on a monotone likelihood.
# ---------------------------------------------------------------------------


def test_cox_warns_on_monotone_likelihood():
    rng = np.random.default_rng(2)
    x = np.r_[np.full(10, 1.0), np.full(10, 5.0)] + rng.uniform(0, 0.1, 20)
    Z = np.r_[np.ones(10), np.zeros(10)]
    c = np.r_[np.zeros(10), np.ones(10)]
    with pytest.warns(UserWarning, match=MONOTONE):
        CoxPH.fit(x=x, Z=Z, c=c)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CoxPH.fit(x=x, Z=Z, c=c, tl=np.r_[np.full(10, 0.5), np.zeros(10)])
    kinds = {type(w.message) for w in caught}
    assert kinds == {UserWarning}  # no RuntimeWarnings alongside it


def test_648_cox_runaway_coefficient_has_nan_standard_error():
    # The pseudo-inverse of the collapsed information gave the runaway
    # coefficient a standard error of 0; it is nan, as is its p-value,
    # and the other coefficient keeps its own (#648).
    rng = np.random.default_rng(0)
    x = np.r_[np.arange(1.0, 16.0), np.arange(20.0, 35.0)]
    Z = np.c_[np.r_[np.ones(15), np.zeros(15)], rng.normal(size=30)]
    c = np.r_[np.zeros(15), np.ones(15)]
    with pytest.warns(UserWarning, match=MONOTONE):
        model = CoxPH.fit(x=x, Z=Z, c=c)
    se = model.standard_errors()
    assert np.isnan(se[0]) and np.isnan(model.p_values[0])
    assert np.isfinite(se[1]) and se[1] > 0
    assert np.isnan(model.summary()["se(coef)"].iloc[0])
