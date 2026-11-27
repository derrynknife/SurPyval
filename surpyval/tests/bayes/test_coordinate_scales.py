"""The Bayesian peak search is the same whatever the unit of time.

The search (``find_mode``) uses absolute gradient tolerances and
finite-difference steps, so it works in each coordinate divided by a
typical size: the maximum-likelihood standard error on the unconstrained
scale (a regression coefficient keeps one over its covariate's spread).
Time multiplied by ``k`` is then a reparameterisation the posterior
follows exactly: a location (``mu``, a Normal's or a Gumbel's) and a
scale are multiplied by ``k``, a log-scale family's ``log alpha`` moves
by ``log k``, and every prediction at time ``k t`` is the one at ``t``.
Before, a Normal or Gumbel location in a unit of millionths was refused
("curvature not finite"), and in a unit of millions moved ``sf`` by up to
1e-3.

Rules fixed before the run: ``sf`` at the rescaled times and its
credible interval within 1e-6 of the unscaled fit's, relative, for
``k`` in 1e-6, 1e-3, 1e3 and 1e6.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.bayes.test_fit_regression import _binary_covariate_data
from surpyval.utils.warnings import ignoring_warnings

FACTORS = [1e-6, 1e-3, 1e3, 1e6]
T = np.array([0.5, 2.0, 7.0, 14.0, 30.0])


def quietly(f, *args, **kwargs):
    with ignoring_warnings():
        return f(*args, **kwargs)


@pytest.mark.parametrize("k", FACTORS)
@pytest.mark.parametrize(
    "fitter",
    # location families (on the time scale itself) and log-scale ones
    [sp.NormalAFT, sp.GumbelPH, sp.WeibullPH, sp.LogNormalAFT],
    ids=["NormalAFT", "GumbelPH", "WeibullPH", "LogNormalAFT"],
)
def test_a_regressions_posterior_follows_the_unit_of_time(fitter, k):
    x, Z, c, n = _binary_covariate_data()
    ref = quietly(fitter.fit_bayes, x, Z, c, n)
    got = quietly(fitter.fit_bayes, x * k, Z, c, n)
    for z in (0.0, 1.0):
        np.testing.assert_allclose(
            got.sf(T * k, [[z]]), ref.sf(T, [[z]]), rtol=1e-6
        )
        np.testing.assert_allclose(
            got.cri(T * k, [[z]], on="sf"),
            ref.cri(T, [[z]], on="sf"),
            rtol=1e-6,
        )


@pytest.mark.parametrize("k", FACTORS)
@pytest.mark.parametrize(
    "dist, method",
    [
        (sp.Normal, "grid"),
        (sp.Gumbel, "grid"),
        (sp.Weibull, "grid"),
        (sp.Normal, "laplace"),
        (sp.Weibull, "laplace"),
    ],
    ids=[
        "Normal-grid",
        "Gumbel-grid",
        "Weibull-grid",
        "Normal-laplace",
        "Weibull-laplace",
    ],
)
def test_a_distributions_posterior_follows_the_unit_of_time(dist, method, k):
    x, _, c, n = _binary_covariate_data()
    ref = quietly(dist.fit_bayes, x, c, n, method=method)
    got = quietly(dist.fit_bayes, x * k, c, n, method=method)
    np.testing.assert_allclose(got.sf(T * k), ref.sf(T), rtol=1e-6)
    np.testing.assert_allclose(
        got.cri(T * k, on="sf"), ref.cri(T, on="sf"), rtol=1e-6
    )


@pytest.mark.parametrize("k", [1e-6, 1e6])
def test_a_prior_on_a_location_is_in_the_users_unit_of_time(k):
    """The search's units are not the prior's: ``Normal(12, 0.5)`` on
    ``mu`` in the data's unit, and the same belief in a unit ``k`` times
    smaller, ``Normal(12 k, 0.5 k)``, give the same posterior (every
    interval within 1e-6, relative, the location and scale times k);
    and the tight prior pulls the posterior median to within 1 of 12."""
    x, _, c, n = _binary_covariate_data()
    ref = quietly(
        sp.Normal.fit_bayes,
        x,
        c,
        n,
        priors={"mu": sp.Normal.from_params([12.0, 0.5])},
    )
    got = quietly(
        sp.Normal.fit_bayes,
        x * k,
        c,
        n,
        priors={"mu": sp.Normal.from_params([12.0 * k, 0.5 * k])},
    )
    assert abs(float(ref.params[0]) - 12.0) < 1.0
    for name in ("mu", "sigma"):
        np.testing.assert_allclose(
            np.ravel(got.param_cri(name)) / k,
            np.ravel(ref.param_cri(name)),
            rtol=1e-6,
        )


# -- covariates far from 0: the engines work at the covariates' means --


def _non_centred(seed, mean):
    """40 Weibull lives whose hazard rises by exp(0.3) per unit of a
    covariate near ``mean`` (spread 1) and exp(0.4) with a binary one."""
    rng = np.random.default_rng(seed)
    n = 40
    z1 = rng.normal(mean, 1.0, n)
    z0 = np.tile([0.0, 1.0], n // 2)
    x = sp.Weibull.random(n, 10.0, 1.5, random_state=seed + 1)
    return x * np.exp(-0.3 * (z1 - mean) - 0.4 * z0), z1, z0


#: The (seed, mean) designs a Weibull PH fit refuses for overflowing on
#: the user's scale, as test_a_posterior_that_overflows_on_the_users_
#: scale_is_refused pins.
_OVERFLOWS = {(seed, 1000.0) for seed in (460211, 460212, 460214, 460215)}


@pytest.mark.parametrize("mean", [0.0, 50.0, 1000.0])
@pytest.mark.parametrize(
    "fitter",
    [sp.WeibullPH, sp.LogNormalAFT],
    ids=["WeibullPH", "LogNormalAFT"],
)
@pytest.mark.parametrize("seed", range(460210, 460216))
def test_a_covariates_origin_does_not_change_the_posterior(seed, fitter, mean):
    """Moving a covariate's origin is a change of the baseline the model
    has in closed form (Weibull PH, LogNormal AFT): with the covariate
    centred, or near ``mean``, the coefficient's posterior and every
    prediction at the same physical covariate values are the same.
    Rule (fixed before this run, after the predictions were moved to the
    covariates' means): the coefficient's interval, ``sf`` at t = 10 at
    the covariate's mean and its interval within 1e-5 relative, at every
    mean, for all six designs; a refusal only where it is pinned
    (``_OVERFLOWS``)."""
    x, z1, z0 = _non_centred(seed, mean)
    if (seed, mean) in _OVERFLOWS and fitter is sp.WeibullPH:
        return
    fits = []
    for z in (z1, z1 - z1.mean()):
        model = quietly(fitter.fit_bayes, x, np.column_stack([z, z0]))
        at = [[float(z.mean()), 0.5]]
        fits.append(
            (
                np.ravel(model.param_cri("coef_0")),
                np.ravel(model.sf(10.0, at)),
                np.ravel(model.cri(10.0, at, on="sf")),
            )
        )
    for got, ref in zip(*fits):
        np.testing.assert_allclose(got, ref, rtol=1e-5)


def test_a_covariate_too_far_from_0_for_the_fit_is_refused_by_it():
    # at a mean of 1e4 the maximum-likelihood fit itself refuses, and
    # the Bayesian fit with it, saying so
    x, z1, z0 = _non_centred(460210, 1e4)
    with pytest.raises(ValueError, match="maximum-likelihood fit"):
        quietly(sp.WeibullPH.fit_bayes, x, np.column_stack([z1, z0]))


@pytest.mark.parametrize("mean", [50.0, 1000.0])
@pytest.mark.parametrize("seed", range(460210, 460216))
def test_the_laplace_normal_at_the_means_has_the_exact_curvature(seed, mean):
    """Rule (the existing pin's, fixed before): every working
    coordinate's standard deviation within 3% of the exact Hessian's
    (Richardson-extrapolated central differences at 1e-3 and 5e-4
    standard deviations), for the six seeded designs at a covariate
    mean of 50 and of 1000 (strongly correlated in the parameters' own
    coordinates: their correlation matrix's smallest eigenvalue about
    3e-7), where the fit is not refused for overflowing on the user's
    scale (a refusal added after the first run; then the test checks
    the refusal's advice instead)."""
    from surpyval.bayes.regression import RegressionTarget
    from surpyval.bayes.result import bind, log_posterior

    x, z1, z0 = _non_centred(seed, mean)
    try:
        model = quietly(sp.WeibullPH.fit_bayes, x, np.column_stack([z1, z0]))
    except ValueError as e:
        # (at 1000, four of the six overflow on the user's scale and are
        # refused: test_a_posterior_that_overflows_on_the_users_scale_...)
        assert mean >= 1000 and "Centre the covariates" in str(e), e
        return
    working = model.diagnostics["working_normal"]
    mode = np.asarray(working["mean"])
    sd = np.sqrt(np.diag(np.asarray(working["cov"])))
    target = RegressionTarget(model.mle)
    bound, _ = bind(target, None)
    log_post = log_posterior(target, bound)
    rp = target.reparam(bound)

    def in_u(u):
        u = np.atleast_2d(u)
        return log_post(rp.inverse(u)) + rp.log_det(u)

    d = mode.size

    def hessian(h):
        out = np.empty((d, d))
        for i in range(d):
            for j in range(d):
                ei, ej = np.zeros(d), np.zeros(d)
                ei[i], ej[j] = h[i], h[j]
                out[i, j] = (
                    in_u((mode + ei + ej)[None])[0]
                    - in_u((mode + ei - ej)[None])[0]
                    - in_u((mode - ei + ej)[None])[0]
                    + in_u((mode - ei - ej)[None])[0]
                ) / (4 * h[i] * h[j])
        return out

    exact = (4 * hessian(5e-4 * sd) - hessian(1e-3 * sd)) / 3
    exact_sd = np.sqrt(np.diag(np.linalg.inv(-exact)))
    np.testing.assert_allclose(sd, exact_sd, rtol=0.03)


@pytest.mark.parametrize(
    "fitter", [sp.GammaPH, sp.LogNormalPH], ids=["GammaPH", "LogNormalPH"]
)
@pytest.mark.parametrize("mean", [5.0, 50.0, 1000.0])
def test_a_family_with_no_centring_is_refused_when_nearly_collinear(
    fitter, mean
):
    # These families have no closed-form baseline at the means; with a
    # covariate 5 or more spreads from 0 the estimates' correlation
    # matrix has a smallest eigenvalue below 3e-2 and the fit is refused,
    # saying to centre the covariate. (Uncentred, GammaPH at 50 had
    # given sf(10) at the mean 0.362 with a 95% interval [0, 0.945].)
    x, z1, z0 = _non_centred(460210, mean)
    with pytest.raises(ValueError, match="centre it"):
        quietly(fitter.fit_bayes, x, np.column_stack([z1, z0]))
    # centred it fits
    quietly(fitter.fit_bayes, x, np.column_stack([z1 - z1.mean(), z0]))


def test_a_well_conditioned_design_is_not_refused():
    # A rare binary covariate and a normal one (an eigenvalue well above
    # the cut), for a family with no centring
    rng = np.random.default_rng(470100)
    n = 60
    Z = np.column_stack(
        [(rng.random(n) < 0.1).astype(float), rng.normal(size=n)]
    )
    x = sp.Weibull.random(n, 10.0, 1.5, random_state=470101)
    x = x * np.exp(-0.5 * Z[:, 0] - 0.3 * Z[:, 1])
    quietly(sp.GammaPH.fit_bayes, x, Z)


#: The location-scale AFT designs refused at a covariate mean of 1000 for
#: overflowing on the user's scale (``mu = mu_c e^{beta' zbar}``), as
#: test_a_location_scale_aft_that_overflows_is_refused pins.
_AFT_OVERFLOWS = {
    (460214, "LogisticAFT"),
    (460215, "NormalAFT"),
    (460215, "GumbelAFT"),
    (460215, "LogisticAFT"),
}


@pytest.mark.parametrize("seed, name", sorted(_AFT_OVERFLOWS))
def test_a_location_scale_aft_that_overflows_is_refused(seed, name):
    # 1 to 13 of 1024 points have an infinite mu at covariates of 0
    x, z1, z0 = _non_centred(seed, 1000.0)
    with pytest.raises(ValueError, match="Centre the covariates"):
        quietly(getattr(sp, name).fit_bayes, x, np.column_stack([z1, z0]))


@pytest.mark.parametrize("mean", [50.0, 1000.0])
@pytest.mark.parametrize(
    "fitter",
    [sp.NormalAFT, sp.GumbelAFT, sp.LogisticAFT],
    ids=["NormalAFT", "GumbelAFT", "LogisticAFT"],
)
@pytest.mark.parametrize("seed", range(460210, 460216))
def test_the_default_location_prior_is_flat_at_the_means(seed, fitter, mean):
    """For the accelerated failure time families on the whole line the
    default prior on ``mu`` is flat at the covariates' means (the
    maintainer's decision, 27 Nov 2026): time is multiplicative, so one
    flat on the user's ``mu`` had tilted the coefficients by ``e^{beta'
    zbar}`` (NormalAFT with a covariate near 50: the coefficient's
    interval [0.603, 1.367] where the centred fit's is [0.146, 0.476]).
    Rule (fixed before the run): with the covariate centred or near
    ``mean``, the coefficient's interval, ``sf`` at t = 10 at the
    covariate's mean and its interval within 1e-5 relative; a refusal
    only where it is pinned (``_AFT_OVERFLOWS``)."""
    x, z1, z0 = _non_centred(seed, mean)
    name = {sp.NormalAFT: "NormalAFT", sp.GumbelAFT: "GumbelAFT"}.get(
        fitter, "LogisticAFT"
    )
    if mean >= 1000 and (seed, name) in _AFT_OVERFLOWS:
        return
    fits = []
    for z in (z1, z1 - z1.mean()):
        model = quietly(fitter.fit_bayes, x, np.column_stack([z, z0]))
        at = [[float(z.mean()), 0.5]]
        fits.append(
            (
                np.ravel(model.param_cri("coef_0")),
                np.ravel(model.sf(10.0, at)),
                np.ravel(model.cri(10.0, at, on="sf")),
            )
        )
    for got, ref in zip(*fits):
        np.testing.assert_allclose(got, ref, rtol=1e-5)
    assert "covariates' means" in model.default_priors["mu"]


def test_an_explicit_flat_prior_on_mu_is_the_users():
    # prior.flat() on mu is flat on the user's mu, as stated: no
    # adjustment, so with a covariate near 50 it differs from the default
    x, z1, z0 = _non_centred(460210, 50.0)
    Z = np.column_stack([z1, z0])
    default = quietly(sp.NormalAFT.fit_bayes, x, Z)
    stated = quietly(
        sp.NormalAFT.fit_bayes, x, Z, priors={"mu": sp.prior.flat()}
    )
    assert (
        np.ravel(stated.param_cri("coef_0"))[0]
        > np.ravel(default.param_cri("coef_0"))[1]
    )


@pytest.mark.parametrize(
    "dist, centre",
    [(sp.Normal, 12.0), (sp.Gumbel, 11.0)],
    ids=["Normal", "Gumbel"],
)
@pytest.mark.parametrize("k", [1e-6, 1e6])
def test_the_prior_data_conflict_check_follows_the_unit_of_time(
    dist, centre, k
):
    """``prior_data_conflict`` runs the same search: with times and the
    prior on ``mu`` multiplied by ``k`` its gaps and p-values are the
    unscaled check's (rule: 1e-6 relative). With times in millions it had
    said the data do not pin the parameters down."""
    from surpyval.diagnostics.bayes import prior_data_conflict

    x = sp.Normal.random(40, 10, 2, random_state=1)

    def check(scale):
        prior = sp.Normal.from_params([centre * scale, 0.5 * scale])
        model = quietly(dist.fit_bayes, x * scale, priors={"mu": prior})
        return quietly(prior_data_conflict, model)

    ref, got = check(1.0), check(k)
    assert got["reason"] is None
    assert got["z"]["mu"] == pytest.approx(ref["z"]["mu"], rel=1e-6)
    assert got["p"]["mu"] == pytest.approx(ref["p"]["mu"], rel=1e-6)


@pytest.mark.parametrize("seed", sorted(seed for seed, _ in _OVERFLOWS))
def test_a_posterior_that_overflows_on_the_users_scale_is_refused(seed):
    # At a covariate mean of 1000 the baseline at a covariate of 0 is
    # alpha e^{beta' zbar / shape}: for these four designs 1 to 3 of the
    # 1024 points overflow to an infinite alpha (for seed 460215, 248 of
    # 4000 draws, which gave an sf interval reaching 1), so the fit is
    # refused, saying to centre the covariates. (The posterior at the
    # covariates' means is right; only the user's baseline at 0 cannot
    # be held: a follow-up.)
    x, z1, z0 = _non_centred(seed, 1000.0)
    with pytest.raises(ValueError, match="Centre the covariates"):
        quietly(sp.WeibullPH.fit_bayes, x, np.column_stack([z1, z0]))
