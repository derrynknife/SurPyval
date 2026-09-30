"""Regression fits whose likelihood has no finite maximum warn (#392).

A covariate that is 1 on exactly the censored rows is a group with no
events: its coefficient has no finite maximum-likelihood estimate, as
the likelihood keeps rising towards a supremum it never reaches. The
optimisers stopped where the rise fell below their tolerances (a
WeibullPH coefficient of -14.7, a PO one of +33) and the fit returned
silently. It now warns, once, pointing at the caller, and still
returns what it reached; an ordinary fit does not warn.

The check is Newton's (``_fit_skeleton.runaway_coefficients``): along the
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
from surpyval.tests.conformance.registry import grouped_reg_data, reg_data
from surpyval.univariate.competing_risks import FineGray
from surpyval.univariate.competing_risks.regression import (
    CompetingRisksProportionalHazards,
)
from surpyval.univariate.regression import _fit_skeleton as skeleton
from surpyval.univariate.regression._fit_skeleton import (
    _flat_at_start,
    runaway_coefficients,
)

NO_MAXIMUM = "No finite maximum: the likelihood keeps increasing"
MONOTONE = "Monotone partial likelihood"


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
    # d["Z"][:, 0] = d["c"] == 1; WeibullPH.fit(**d) gave -16.31 (now
    # -14.72) silently.
    model, w = _fit(lambda: sp.WeibullPH.fit(**_no_events(reg_data())))
    assert [str(x.message)[: len(NO_MAXIMUM)] for x in w] == [NO_MAXIMUM]
    assert model.phi_params[0] == pytest.approx(-14.72, abs=0.01)


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
    _, w = _fit(lambda: sp.WeibullAFT.fit(**d, fixed={"beta_1": -0.3}))
    assert len(w) == 1 and "coefficient(s) [0]" in str(w[0].message)
    # ... and with it fixed there is nothing to run away.
    _, w = _fit(lambda: sp.WeibullAFT.fit(**d, fixed={"beta_0": -1.0}))
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
    # direction is flat at the start too, which a runaway's is not.
    d = reg_data()
    g = np.arange(30) // 10  # each level has events of both causes
    Z = np.column_stack([g == 0, g == 1, g == 2, d["Z"][:, 1]]).astype(float)
    e = _competing(d)["e"]
    _, w = _fit(lambda: FineGray.fit(d["x"], Z, e, n=d["n"], event="a"))
    assert not w, [str(x.message) for x in w]
    for fitter in (sp.WeibullPH, sp.LogNormalAFT, sp.WeibullPO):
        _, w = _fit(lambda: fitter.fit(x=d["x"], Z=Z, c=d["c"], n=d["n"]))
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
    # ("Monotone partial likelihood" for coefficient 2 of cause "u").
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
    assert not w, [str(x.message) for x in w]


# -- the gate: profiles are read only where Newton has not converged ----------


def _count_profiles(monkeypatch):
    calls = []
    profile = skeleton._profile

    def counted(neg_ll, x, H, j):
        calls.append(j)
        return profile(neg_ll, x, H, j)

    monkeypatch.setattr(skeleton, "_profile", counted)
    return calls


@pytest.mark.parametrize(
    "fit",
    [
        lambda: sp.WeibullPH.fit(**reg_data()),
        lambda: sp.LogNormalAFT.fit(**reg_data()),
        lambda: sp.LogisticPO.fit(**reg_data()),
        lambda: sp.WeibullAH.fit(**reg_data()),
        lambda: sp.WeibullFrailty.fit(**grouped_reg_data()),
        lambda: FineGray.fit(**_competing(reg_data()), event="a"),
    ],
    ids=["PH", "AFT", "PO", "AH", "Frailty", "FineGray"],
)
def test_an_ordinary_fit_reads_no_profile(monkeypatch, fit):
    # Its Newton step is at the optimiser's tolerance in every coefficient
    # (the frailty variance of these data sits at its limit of 0, which the
    # step leaves out), so the third derivatives are never taken.
    calls = _count_profiles(monkeypatch)
    _, w = _fit(fit)
    assert calls == [] and not w, [str(x.message) for x in w]


@pytest.mark.parametrize(
    "name", ["WeibullPH", "LogNormalAFT", "LogisticPO", "LogNormalPO"]
)
def test_a_runaway_has_its_profile_read(monkeypatch, name):
    # Only the runaway coefficient's (0): the other's step is at tolerance.
    # LogNormalAFT and the PO fits run furthest onto the plateau (profile
    # information 1e-13 to 1e-15 of the start's).
    calls = _count_profiles(monkeypatch)
    _, w = _fit(lambda: getattr(sp, name).fit(**_no_events(reg_data())))
    assert len(w) == 1 and "coefficient(s) [0]" in str(w[0].message)
    k_dist = len(getattr(sp, name).param_names)
    assert calls == [k_dist]


def test_the_gate_clears_a_maximum_and_nothing_else():
    # A maximum: the Newton step is 0, well within 1/709.8 of the value.
    quadratic = lambda p: (p[0] - 2.0) ** 2 + (p[1] + 3.0) ** 2  # noqa: E731
    x = np.array([2.0, -3.0])
    H, g = hessian(quadratic)(x), grad(quadratic)(x)
    assert skeleton._cleared(x, H, g).tolist() == [True, True]
    # A runaway along t = u (exp(t) + 50 (u - t)^2 at t = u = -15): the
    # step is (1, 1), 1/15 of the value, so neither is cleared.
    runaway = lambda p: anp.exp(p[0]) + 50.0 * (p[1] - p[0]) ** 2  # noqa
    x = np.array([-15.0, -15.0])
    H, g = hessian(runaway)(x), grad(runaway)(x)
    assert skeleton._cleared(x, H, g).tolist() == [False, False]
    # A linear rise has no curvature: no Hessian to trust, nothing cleared.
    linear = lambda p: -3.0 * p[0] + p[1] ** 2  # noqa: E731
    x = np.array([-50.0, 0.0])
    H, g = hessian(linear)(x), grad(linear)(x)
    assert skeleton._cleared(x, H, g).tolist() == [False, False]
    # A parameter the likelihood does not depend on is left out of the
    # step, and the others are still cleared.
    ignores = lambda p: (p[0] - 1.0) ** 2 + 0.0 * p[1]  # noqa: E731
    x = np.array([1.0, 7.0])
    H, g = hessian(ignores)(x), grad(ignores)(x)
    assert skeleton._cleared(x, H, g).tolist() == [True, False]
