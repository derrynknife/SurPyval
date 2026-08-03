import numpy as np
import pytest

from surpyval import (
    Beta,
    Beta4,
    Exponential,
    ExpoWeibull,
    Gamma,
    Gumbel,
    GumbelLEV,
    Logistic,
    LogLogistic,
    LogNormal,
    Normal,
    Rayleigh,
    Weibull,
)

DISTS = [
    Gumbel,
    GumbelLEV,
    Normal,
    Weibull,
    LogNormal,
    Logistic,
    LogLogistic,
    Beta,
    ExpoWeibull,
    Gamma,
    Exponential,
    Rayleigh,
]

parameter_sample_random_parameters = [
    ((1, 20), (0.5, 5)),
    ((1, 20), (0.5, 5)),
    ((1, 100), (0.5, 100)),
    ((1, 100), (0.5, 20)),
    ((1, 3), (0.2, 1)),
    ((1, 100), (0.5, 20)),
    ((1, 100), (0.5, 20)),
    ((0.1, 30), (0.1, 30)),
    ((1, 30), (0.1, 10), (0.5, 1.5)),
    ((1, 30), (0.1, 10)),
    ((0.1, 1),),
    ((1, 30),),
]
FIT_SIZES = [1_000, 10_000, 100_000]


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)


def generate_mle_test_cases():
    for idx, dist in enumerate(DISTS):
        random_parameters = parameter_sample_random_parameters[idx]
        for kind in [
            "full",
            "censored",
            "left_censored",
            "truncated",
            "interval",
        ]:
            yield dist, random_parameters, kind


def generate_small_mle_test_cases():
    for idx, dist in enumerate(DISTS):
        random_parameters = parameter_sample_random_parameters[idx]
        for kind in ["full"]:
            yield dist, random_parameters, kind


def generate_mpp_test_cases():
    for idx, dist in enumerate(DISTS):
        random_parameters = parameter_sample_random_parameters[idx]
        for rr in ["x", "y"]:
            yield dist, random_parameters, rr


def generate_mom_test_cases():
    for idx, dist in enumerate(DISTS):
        random_parameters = parameter_sample_random_parameters[idx]
        yield dist, random_parameters


def generate_mps_test_cases():
    for idx, dist in enumerate(DISTS):
        random_parameters = parameter_sample_random_parameters[idx]
        yield dist, random_parameters


def generate_mps_trunc_test_cases():
    for idx, dist in enumerate(DISTS):
        if dist.name in ["ExpoWeibull"]:
            continue
        random_parameters = parameter_sample_random_parameters[idx]
        yield dist, random_parameters


def generate_mse_test_cases():
    for idx, dist in enumerate(DISTS):
        random_parameters = parameter_sample_random_parameters[idx]
        yield dist, random_parameters


def idfunc(x):
    if type(x) is tuple:
        return "random_parameters"
    elif type(x) is str:
        return x
    else:
        return x.name


def interval_censor(x, n=100):
    n, xx = np.histogram(x, bins=n)
    x = np.vstack([xx[0:-1], xx[1::]]).T
    x = x[n > 0]
    n = n[n > 0]
    return x, n


def censor_at(x, q, where="right"):
    c = np.zeros_like(x)
    x = np.copy(x)
    if where == "right":
        x_q = np.quantile(x, 1 - q)
        mask = x > x_q
        c[mask] = 1
        x[mask] = x_q
        return x, c
    elif where == "left":
        x_q = np.quantile(x, q)
        mask = x < x_q
        c[mask] = -1
        x[mask] = x_q
        return x, c
    elif where == "both":
        x_u = np.quantile(x, 1 - q)
        x_l = np.quantile(x, q)
        mask_l = x < x_l
        mask_u = x > x_u
        c[mask_l] = -1
        c[mask_u] = 1
        x[mask_l] = x_l
        x[mask_u] = x_u
        return x, c
    else:
        raise ValueError("'where' parameter not correctly defined")


def truncate_at(x, q, where="right"):
    x = np.copy(x)
    if where == "right":
        x_q = np.quantile(x, 1 - q)
        x = x[x < x_q]
        return x, None, x_q
    elif where == "left":
        x_q = np.quantile(x, q)
        x = x[x > x_q]
        return x, x_q, None
    elif where == "both":
        x_u = np.quantile(x, 1 - q)
        x_l = np.quantile(x, q)
        x = x[x < x_u]
        x = x[x > x_l]
        return x, x_l, x_u
    else:
        raise ValueError("'where' parameter not correctly defined")


@pytest.mark.parametrize(
    "dist,random_parameters,kind", generate_mle_test_cases(), ids=idfunc
)
def test_mle_convergence(dist, random_parameters, kind):
    tol = 0.03
    for n in FIT_SIZES:
        test_params = []
        for b in random_parameters:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        if kind == "full":
            model = dist.fit(x)
        elif kind == "censored":
            x, c = censor_at(x, 0.025, "right")
            model = dist.fit(x, c=c)
        elif kind == "left_censored":
            x, c = censor_at(x, 0.025, "left")
            model = dist.fit(x, c=c)
        elif kind == "truncated":
            x, tl, tr = truncate_at(x, 0.05, "both")
            model = dist.fit(x, tl=tl, tr=tr)
        elif kind == "interval":
            x, n = interval_censor(x)
            model = dist.fit(x=x, n=n)
        if len(model.params) == 0:
            continue
        fitted_params = np.array(model.params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        # Decrease the tolerance for every parameter
        # e.g. Weibull (2 params) tol will be 5%
        # ExpoWeibull the tolerance will be 7.5%
        if (diff < tol * dist.k).all():
            break
    else:
        raise AssertionError("MLE convergence not good for %s\n" % dist.name)


@pytest.mark.parametrize(
    "dist,random_parameters,kind", generate_small_mle_test_cases(), ids=idfunc
)
def test_mle_convergence_small(dist, random_parameters, kind):
    tol = 0.09
    for n in [100, 250, 500]:
        test_params = []
        for b in random_parameters:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        if kind == "full":
            model = dist.fit(x)
        if len(model.params) == 0:
            continue
        fitted_params = np.array(model.params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        # Decrease the tolerance for every parameter
        # e.g. Weibull (2 params) tol will be 6%
        # ExpoWeibull the tolerance will be 9%
        if (diff < tol * dist.k).all():
            break
    else:
        raise AssertionError(
            "MLE fit for small data not good for "
            + f"{dist.name}: {fitted_params} :: {test_params}\n"
        )


@pytest.mark.parametrize(
    "dist,random_parameters,rr", generate_mpp_test_cases(), ids=idfunc
)
def test_mpp(dist, random_parameters, rr):
    if dist not in [Beta, ExpoWeibull]:
        for n in FIT_SIZES:
            test_params = []
            tol = 0.025
            for b in random_parameters:
                test_params.append(np.random.uniform(*b))
            test_params = np.array(test_params)
            x = dist.random(10000, *test_params)
            model = dist.fit(x=x, rr=rr, how="MPP", heuristic="Nelson-Aalen")
            fitted_params = np.array(model.params)
            max_params = np.max([fitted_params, test_params], axis=0)
            diff = np.abs(fitted_params - test_params) / max_params
            if (diff < tol * dist.k).all():
                break
        else:
            raise AssertionError("MPP fit not very good in %s\n" % dist.name)


@pytest.mark.parametrize(
    "dist,random_parameters", generate_mom_test_cases(), ids=idfunc
)
def test_mom(dist, random_parameters):
    if dist.name == "ExpoWeibull":
        return None
    for n in FIT_SIZES:
        test_params = []
        tol = 0.025
        for b in random_parameters:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        model = dist.fit(x=x, how="MOM")
        fitted_params = np.array(model.params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        if (diff < tol * dist.k).all():
            break
    else:
        raise AssertionError("MOM fit not very good in %s\n" % dist.name)


@pytest.mark.parametrize(
    "dist,random_parameters", generate_mps_test_cases(), ids=idfunc
)
def test_mps(dist, random_parameters):
    for n in FIT_SIZES:
        test_params = []
        if dist.name == "ExpoWeibull":
            tol = 0.02
        else:
            tol = 0.01
        for b in random_parameters:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        model = dist.fit(x=x, how="MPS")
        fitted_params = np.array(model.params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        if (diff < tol * dist.k).all():
            break
    else:
        raise AssertionError("MPS fit not very good in %s\n" % dist.name)


@pytest.mark.parametrize(
    "dist,random_parameters", generate_mps_trunc_test_cases(), ids=idfunc
)
def test_mps_truncated(dist, random_parameters):
    for n in [2_000]:
        test_params = []
        tol = 0.1
        for b in random_parameters:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        x, tl, tr = truncate_at(x, 0.05, "both")
        model = dist.fit(x=x, tl=tl, tr=tr, how="MPS")
        fitted_params = np.array(model.params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        if (diff < tol * dist.k).all():
            break
    else:
        raise AssertionError("MPS fit not very good in %s\n" % dist.name)


OFFSET_CASES = [
    (Weibull, (10.0, 2.0)),
    (Gamma, (3.0, 2.0)),
    (LogNormal, (1.0, 0.5)),
    (LogLogistic, (5.0, 2.0)),
    (Exponential, (0.5,)),
    (Rayleigh, (3.0,)),
]


@pytest.mark.parametrize(
    "dist,dist_params", OFFSET_CASES, ids=[d.name for d, _ in OFFSET_CASES]
)
def test_offset_initialiser_puts_the_offset_first(dist, dist_params):
    # `_initial_guess` overwrites slot 0 with the offset seed, so every
    # offset initialiser must return gamma first. Gamma returned it last,
    # which meant the overwrite landed on the shape and destroyed it --
    # the seed came back as (offset, shape-estimate, offset) and MSE fits
    # converged to nonsense.
    shift = 10.0
    x = dist.random(2000, *dist_params) + shift
    c = np.zeros(x.size, dtype=int)
    n = np.ones(x.size, dtype=np.int64)

    init = np.asarray(
        dist._initial_guess(x, c, n, True, False, False, "Nelson-Aalen"),
        dtype=float,
    )
    assert len(init) == len(dist_params) + 1

    # Slot 0 is the offset: just below the smallest observation.
    assert init[0] == pytest.approx(x.min() - 1.0)
    assert np.isfinite(init[1:]).all()
    assert (init[1:] > 0).all()

    # The tell-tale of an initialiser that returns the offset in the
    # wrong position: `_initial_guess` writes the offset into slot 0
    # while the initialiser's own copy of it is still sitting in a
    # parameter slot, so the value appears twice and one real parameter
    # estimate has been thrown away. Gamma returned
    # ``(alpha, 1/beta, offset)`` and seeded (9.07, 14.88, 9.07) where
    # the parameters are (3.0, 2.0).
    assert not np.isclose(init[1:], init[0]).any(), (
        f"{dist.name} seeded {init}: the offset {init[0]:.4g} also "
        f"appears in a parameter slot, so its initialiser is returning "
        f"the offset somewhere other than first"
    )


def test_offset_gamma_mse_recovers_the_shift():
    # The user-visible consequence of the seeding bug above. With the
    # offset written over the shape estimate, MSE returned a *negative*
    # shift for data shifted up by 10, with a shape of 63 against a true
    # 3 -- and on other samples it stopped after 0.03s at the seed
    # itself, reporting beta equal to the offset. Neither raised.
    np.random.seed(3)
    shift = 10.0
    x = Gamma.random(600, 3.0, 4.0) + shift

    model = Gamma.fit(x, offset=True, how="MSE")
    assert model.gamma == pytest.approx(shift, abs=0.5)
    assert model.params[0] == pytest.approx(3.0, rel=0.4)
    assert model.params[1] == pytest.approx(4.0, rel=0.4)


@pytest.mark.parametrize("how", ["MLE", "MPS", "MSE", "MPP"])
@pytest.mark.parametrize(
    "dist,dist_params", OFFSET_CASES, ids=[d.name for d, _ in OFFSET_CASES]
)
def test_offset_fit_recovers_gamma(dist, dist_params, how):
    if how == "MPP" and dist is Gamma:
        pytest.skip(
            "MPP shape estimation is unreliable for Gamma "
            "(probability-plotting limitation; see issue #158)"
        )
    gamma = 10.0
    x = dist.random(10_000, *dist_params) + gamma
    model = dist.fit(x, offset=True, how=how)
    assert abs(model.gamma - gamma) < 0.5
    assert np.allclose(model.params, dist_params, rtol=0.15)


@pytest.mark.parametrize("how", ["MLE", "MPS", "MSE", "MOM"])
def test_beta_cannot_be_offset(how):
    # Beta is supported on [0, 1]; a one-sided offset cannot move the
    # lower bound while pinning the upper bound at 1, so it must raise a
    # clear error rather than the opaque failure it used to.
    x = Beta.random(100, 2.0, 5.0)
    with pytest.raises(ValueError, match="cannot be offset"):
        Beta.fit(x, offset=True, how=how)


def test_beta_rejects_mpp():
    # Beta has no linearising probability plot (its CDF is the incomplete
    # beta function and it is not a location-scale family), so MPP fitting
    # must raise the clean ValueError rather than a raw NotImplementedError.
    assert Beta.supports_mpp is False
    x = Beta.random(200, 2.0, 5.0)
    with pytest.raises(ValueError, match="probability plot"):
        Beta.fit(x, how="MPP")


def test_beta4_recovers_parameters():
    # The four-parameter Beta estimates the support bounds (a, b) along
    # with the two shape parameters.
    np.random.seed(0)
    alpha, beta, a, b = 3.0, 4.0, 2.0, 7.0
    x = Beta4.random(10_000, alpha, beta, a, b)
    model = Beta4.fit(x)
    assert np.allclose(model.params, [alpha, beta, a, b], rtol=0.1)
    # Support is read off the fitted bounds.
    assert np.allclose(model.support, [model.params[2], model.params[3]])


def test_beta4_cannot_be_offset():
    x = Beta4.random(100, 2.0, 5.0, 1.0, 4.0)
    with pytest.raises(ValueError, match="cannot be offset"):
        Beta4.fit(x, offset=True)


def test_beta4_handles_right_censoring():
    np.random.seed(1)
    alpha, beta, a, b = 2.5, 3.0, 1.0, 5.0
    x = Beta4.random(5_000, alpha, beta, a, b)
    threshold = 4.0
    c = np.where(x > threshold, 1, 0)
    x = np.where(x > threshold, threshold, x)
    model = Beta4.fit(x, c=c)
    assert np.allclose(model.params, [alpha, beta, a, b], rtol=0.15)


@pytest.mark.parametrize(
    "dist,random_parameters", generate_mse_test_cases(), ids=idfunc
)
def test_mse(dist, random_parameters):
    for n in FIT_SIZES:
        test_params = []
        # 5% accuracy!!
        if dist.name == "ExpoWeibull":
            tol = 0.075
        else:
            tol = 0.05
        for b in random_parameters:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        model = dist.fit(x=x, how="MSE")
        fitted_params = np.array(model.params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        if (diff < tol).all():
            break
    else:
        raise AssertionError("MPS fit not very good in %s\n" % dist.name)


def test_raw_to_central_moment_transform():
    # mu_k = sum_j C(k, j) (-1)^(k-j) E[X^j] mean^(k-j), with the mean
    # kept in the leading slot. Checked against a shifted Gamma, whose
    # central moments are known exactly: a/b^2 and 2a/b^3.
    from math import comb

    from surpyval.univariate.parametric.fitters.mom import raw_to_central

    g, a, b = 10.0, 3.0, 4.0
    raw = []
    for k in (1, 2, 3):
        total = 0.0
        for j in range(k + 1):
            rising = 1.0
            for i in range(j):
                rising *= a + i
            total += comb(k, j) * g ** (k - j) * rising / b**j
        raw.append(total)

    mean, var, mu3 = raw_to_central(np.array(raw))
    assert mean == pytest.approx(g + a / b)
    assert var == pytest.approx(a / b**2)
    assert mu3 == pytest.approx(2 * a / b**3)


def test_offset_gamma_mom_recovers_the_shape():
    # MOM compares central moments, not raw ones. On an offset fit
    # E[X^k] is dominated by gamma^k -- the shape is a 0.5% correction
    # to E[X^3] -- so the raw-moment objective was nearly blind to it
    # and settled on a shape of 17.7 against a true 3.0, matching the
    # sample moments *better than the true parameters did*. MOM is not
    # in the offset parametrisation above, so nothing caught this.
    np.random.seed(11)
    x = Gamma.random(10_000, 3.0, 4.0) + 10.0

    model = Gamma.fit(x, offset=True, how="MOM")
    assert model.gamma == pytest.approx(10.0, abs=0.5)
    # Generous on the shape: the three-parameter moment estimator is
    # biased upward at finite n because alpha goes like 1 / skewness^2.
    # The point is that it is near 3 rather than near 17.
    assert model.params[0] == pytest.approx(3.0, rel=0.35)
    assert model.params[1] == pytest.approx(4.0, rel=0.35)


def test_expoweibull_offset_keeps_the_refined_seed():
    # ExpoWeibull seeds itself from a Gumbel fit to log(x). Without an
    # offset the probability plot alone is enough -- 54 parameter
    # combinations, plus right, left and heavily tied data, all reach the
    # same optimum -- so the nested optimiser ladder there was 15-30% of
    # the fit for nothing.
    #
    # With an offset it is not enough: five of 48 offset fits seeded from
    # the plot alone landed on a worse optimum, one of them at 685.85
    # against 622.26. So the offset path still refines -- on the shifted
    # data, see test_expoweibull_offset_seeds_from_shifted_data.
    np.random.seed(8)
    x = ExpoWeibull.random(300, 10.0, 2.0, 1.5) + 25.0
    model = ExpoWeibull.fit(x, offset=True)
    assert model.neg_ll() < 861.93


def test_expoweibull_offset_seeds_from_shifted_data():
    # The offset initialiser used to take log(x) before removing the
    # shift. A large offset compresses those logs into a narrow band, so
    # the Gumbel sigma collapses and beta = 1 / sigma explodes: with a
    # true beta of 2 the seed came back at 23.5, and the MLE then failed
    # outright, returning nan and silently falling back to MPP.
    #
    # 54 of 120 offset fits returned nan that way -- every configuration
    # at an offset of 100 or 1000. The offset is now estimated first and
    # the seed read off x - gamma.
    np.random.seed(11)
    x = 100.0 + ExpoWeibull.random(500, 10.0, 2.0, 1.0)

    gamma, alpha, beta, mu = ExpoWeibull._parameter_initialiser(x, offset=True)
    # Seeded from x - gamma these sit near the truth; seeded from x they
    # came back as alpha = 111, beta = 23.5.
    assert beta == pytest.approx(2.0, rel=0.5)
    assert alpha == pytest.approx(10.0, rel=0.5)

    # ...and the fit that seed feeds now converges rather than returning
    # nan and warning its way back to MPP.
    model = ExpoWeibull.fit(x, offset=True)
    assert np.isfinite(model.neg_ll())
    assert model.gamma == pytest.approx(100.0, abs=1.0)

    # The offset the fitter will actually install, so that the seed is
    # taken against the same shift it will be optimised under.
    assert gamma < x.min()
    assert gamma == pytest.approx(x.min() - 1.0)


# -- information criteria for every fit method --------------------------
#
# Only MLE and the closed forms used to report a log-likelihood, because
# only they compute one on the way to the answer. neg_ll, aic, bic and
# aic_c therefore raised AttributeError for MPS, MSE, MOM and MPP -- the
# usual way of choosing between distributions was unavailable for four
# of the five methods. The log-likelihood belongs to the parameters and
# the data, not to the search that found them.

ALL_HOWS = ["MLE", "MPS", "MSE", "MOM", "MPP"]


@pytest.mark.parametrize("how", ALL_HOWS)
def test_information_criteria_available_for_every_method(how):
    np.random.seed(1)
    x = Weibull.random(500, 10.0, 2.0)
    model = Weibull.fit(x, how=how)
    for name in ("neg_ll", "aic", "bic", "aic_c"):
        assert np.isfinite(getattr(model, name)()), f"{how}.{name}()"


@pytest.mark.parametrize("how", ALL_HOWS)
def test_reported_log_likelihood_is_the_one_at_the_fitted_parameters(how):
    # Computed independently of the fitter, so a method that reported
    # its own objective by mistake -- MPS's mean log-spacing, say, or
    # MSE's sum of squares -- would be caught.
    np.random.seed(1)
    x = Weibull.random(500, 10.0, 2.0)
    model = Weibull.fit(x, how=how)
    direct = -np.sum(np.log(Weibull.from_params(list(model.params)).df(x)))
    assert model.neg_ll() == pytest.approx(direct, rel=1e-10)


def test_maximum_likelihood_attains_the_largest_likelihood():
    # The defining property, and only checkable now that the other
    # methods report one: no other estimator on the same data can beat
    # MLE's log-likelihood.
    np.random.seed(1)
    x = Weibull.random(500, 10.0, 2.0)
    best = Weibull.fit(x, how="MLE").neg_ll()
    for how in ("MPS", "MSE", "MOM", "MPP"):
        assert Weibull.fit(x, how=how).neg_ll() >= best - 1e-9, how


# -- degenerate data ----------------------------------------------------
#
# Fitting three tied observations used to die with an IndexError raised
# from inside numdifftools, four steps removed from the cause. The
# probability plot has no slope through a single distinct abscissa, so
# polyfit returned a nan; that nan seeded the MLE, which started at nan,
# produced a nan hessian, and finally asked numdifftools for a numerical
# one, whose list of finite-difference steps came back empty.
#
# Gamma and Beta failed the same way but silently: their moment-based
# initialisers divide by a variance that is zero for tied data, and a
# failed optimiser returns its initial guess (#261), so (inf, inf) was
# handed back as a fitted model.


def test_degenerate_data_is_rejected_with_a_clear_error():
    for dist in (Weibull, Gamma):
        with pytest.raises(ValueError, match="free parameter"):
            dist.fit(np.array([10.0, 10.0, 10.0]))
    with pytest.raises(ValueError, match="free parameter"):
        Beta.fit(np.array([0.2, 0.2, 0.2]))


def test_fixing_a_parameter_buys_back_a_degree_of_freedom():
    # The count is of *free* parameters, not of the distribution's, so a
    # single observation is enough once beta is fixed. The MLE of alpha
    # with beta known is (sum x**beta / n) ** (1 / beta) = 10.
    model = Weibull.fit(np.array([10.0]), fixed={"beta": 2.0})
    assert model.params[0] == pytest.approx(10.0, rel=1e-6)
    assert model.params[1] == pytest.approx(2.0)

    tied = Weibull.fit(np.array([10.0, 10.0, 10.0]), fixed={"beta": 2.0})
    assert tied.params[0] == pytest.approx(10.0, rel=1e-6)


def test_one_parameter_distributions_fit_tied_data():
    # A tied sample identifies a single parameter perfectly well, so
    # these must not be caught by the degeneracy check.
    assert Exponential.fit(np.array([10.0] * 3)).params[0] == pytest.approx(
        0.1
    )
    assert Rayleigh.fit(np.array([10.0] * 3)).params[0] == pytest.approx(
        np.sqrt(50.0)
    )


def test_probability_plotting_survives_a_degenerate_regression():
    # MPP is exempt from the degeneracy check: it is a regression, not a
    # likelihood maximisation, and it is how several distributions seed
    # themselves. It must return finite parameters rather than nan.
    model = Weibull.fit(np.array([10.0, 10.0, 10.0]), how="MPP")
    assert np.isfinite(np.asarray(model.params, dtype=float)).all()


def test_a_fit_never_returns_non_finite_parameters():
    # The backstop: whatever the cause, non-finite parameters are not a
    # valid answer. Gamma used to return (inf, inf) here in silence.
    for dist, x in (
        (Weibull, np.array([10.0, 10.0, 10.0])),
        (Gamma, np.array([10.0, 10.0, 10.0])),
    ):
        with pytest.raises(ValueError):
            dist.fit(x)
