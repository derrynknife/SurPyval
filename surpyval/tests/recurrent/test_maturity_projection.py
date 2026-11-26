"""The maturity projection models: AMPM, AMPM-Stein and the discrete
AMPM-Stein, each in its many-modes form.

Checked against the maximum of each model's own likelihood found by a
general optimiser (or, for the discrete model, the exact finite-modes
beta-binomial likelihood with very many modes), and against simulations of
the models' assumptions (many modes with gamma rates or beta chances).
"""

import numpy as np
import pytest
from scipy import optimize, special

from surpyval.recurrent import (
    AMPMProjection,
    DiscreteSteinProjection,
    SteinProjection,
    adpm_stein_projection,
    ampm_projection,
    ampm_stein_projection,
)

X = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
MODES: list = ["b1", "a1", "b2", "b1", "b3", "a2", "b2", "b4", "b1", "a1"]
MODES = MODES + ["b5", None]
C = [0] * 11 + [1]
FEF = {"b1": 0.8, "b2": 0.7, "b3": 0.75, "b4": 0.6, "b5": 0.7}
T = 400.0


# -- AMPM ---------------------------------------------------------------------


def test_ampm_is_the_maximum_of_the_first_occurrence_likelihood():
    result = ampm_projection(X, MODES, FEF, c=C)
    assert isinstance(result, AMPMProjection)
    first = np.array([15, 60, 130, 260, 390.0])
    m = first.size

    def nll(v):
        lam, b = np.exp(v)
        return -(
            m * np.log(lam)
            - np.sum(np.log1p(b * first))
            - lam / b * np.log1p(b * T)
        )

    res = optimize.minimize(
        nll,
        [np.log(0.02), np.log(0.01)],
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000},
    )
    lam_b, beta = np.exp(res.x)
    assert result.lambda_B == pytest.approx(lam_b, rel=1e-6)
    assert result.beta == pytest.approx(beta, rel=1e-6)
    assert result.lambda_A == pytest.approx(3 / T)
    assert result.mean_fef == pytest.approx(0.71)
    assert result.failures == {"A": 3, "B": 8}
    # The expected modes at T equal the modes seen (the likelihood's
    # first equation).
    assert result.expected_modes(T) == pytest.approx(5, rel=1e-10)
    rho = 3 / T + 0.29 * lam_b + 0.71 * lam_b / (1 + beta * T)
    assert result.projected_intensity == pytest.approx(rho, rel=1e-6)
    assert result.projected_mtbf == pytest.approx(50.12, abs=0.005)
    assert result.growth_potential_mtbf == pytest.approx(66.58, abs=0.005)
    assert result.maturity(T) == pytest.approx(
        beta * T / (1 + beta * T), rel=1e-6
    )
    assert result.mtbf(T) == pytest.approx(result.projected_mtbf)
    assert "AMPM" in repr(result)
    np.testing.assert_array_equal(result.modes.index, list(FEF))


def test_ampm_refusals():
    with pytest.raises(ValueError, match="do not thin out"):
        ampm_projection(
            [10, 300, 350, 390, 400],
            ["b1", "b2", "b3", "b4", None],
            {"b1": 0.7, "b2": 0.7, "b3": 0.7, "b4": 0.7},
            c=[0, 0, 0, 0, 1],
        )
    with pytest.raises(ValueError, match="two B modes"):
        ampm_projection(X, MODES, {"b1": 0.8}, c=C)
    with pytest.raises(ValueError, match="one test clock"):
        ampm_projection(
            [10, 20, 50, 15, 50],
            ["b1", "b2", None, "b1", None],
            {"b1": 0.7, "b2": 0.7},
            c=[0, 0, 1, 0, 1],
            i=[1, 1, 1, 2, 2],
        )
    with pytest.raises(ValueError, match="no failures in the data"):
        ampm_projection(X, MODES, {**FEF, "b9": 0.5}, c=C)
    with pytest.raises(ValueError, match="fraction in"):
        ampm_projection(X, MODES, {**FEF, "b1": 1.5}, c=C)
    with pytest.raises(ValueError, match="Give c"):
        ampm_projection(X, MODES, FEF)
    with pytest.raises(ValueError, match="mode label"):
        ampm_projection(X, [None] * 12, FEF, c=C)


# -- AMPM-Stein ---------------------------------------------------------------


def test_stein_theta_is_the_log_series_maximum():
    result = ampm_stein_projection(X, MODES, FEF, c=C)
    assert isinstance(result, SteinProjection)
    counts = {"b1": 3, "a1": 2, "b2": 2, "b3": 1, "a2": 1, "b4": 1, "b5": 1}

    # In the many-modes limit a seen mode's count is a logarithmic-series
    # variate with p = beta T / (1 + beta T) = theta.
    def nll(p):
        return -sum(
            n * np.log(p) - np.log(-np.log1p(-p)) for n in counts.values()
        )

    p = optimize.minimize_scalar(
        nll,
        bounds=(1e-6, 1 - 1e-9),
        method="bounded",
        options={"xatol": 1e-12},
    ).x
    assert result.theta == pytest.approx(p, rel=1e-7)
    theta = result.theta
    rho = (
        sum((1 - FEF.get(k, 0.0)) * theta * n / T for k, n in counts.items())
        + (1 - theta) * 11 / T
    )
    assert result.projected_intensity == pytest.approx(rho, rel=1e-12)
    assert result.unseen_intensity == pytest.approx((1 - theta) * 11 / T)
    assert result.initial_mtbf == pytest.approx(T / 11)
    assert result.beta * T == pytest.approx(theta / (1 - theta))
    assert round(result.theta, 4) == 0.5704
    assert round(result.projected_mtbf, 2) == 52.2
    assert "AMPM-Stein" in repr(result)


def test_stein_with_a_modes_shrinks_the_b_modes_only():
    result = ampm_stein_projection(X, MODES, FEF, c=C, a_modes=["a1", "a2"])
    # B modes: counts 3, 2, 1, 1, 1 (8 failures, 5 modes).
    y = optimize.brentq(lambda y: 8 * np.log1p(y) - 5 * y, 1e-9, 100)
    theta = y / (1 + y)
    assert result.theta == pytest.approx(theta, rel=1e-12)
    b = {"b1": 3, "b2": 2, "b3": 1, "b4": 1, "b5": 1}
    rho = (
        3 / T
        + sum((1 - FEF[k]) * theta * n / T for k, n in b.items())
        + (1 - theta) * 8 / T
    )
    assert result.projected_intensity == pytest.approx(rho, rel=1e-12)
    assert result.lambda_A == pytest.approx(3 / T)
    assert list(result.modes["kind"]) == ["B", "A", "B", "B", "A", "B", "B"]


def test_stein_pools_systems():
    # Two systems to 200 each: the same counts over 400 of test.
    x = [15, 42, 60, 98, 130, 171, 200, 5, 10, 105, 190, 200]
    m = ["b1", "a1", "b2", "b1", "b3", "a2", None, "b2", "b4", "b1", "a1"]
    m = m + [None]
    c = [0] * 6 + [1] + [0] * 4 + [1]
    i = [1] * 7 + [2] * 5
    fef = {k: v for k, v in FEF.items() if k != "b5"}
    pooled = ampm_stein_projection(x, m, fef, c=c, i=i)
    assert pooled.T == 400 and pooled.systems == 2
    single = ampm_stein_projection(
        X[:-2] + [400], MODES[:-2] + [None], fef, c=[0] * 10 + [1]
    )
    assert pooled.theta == pytest.approx(single.theta, rel=1e-14)
    assert pooled.projected_intensity == pytest.approx(
        single.projected_intensity, rel=1e-14
    )


def test_stein_refusals():
    once = [10, 20, 30, 40]
    with pytest.raises(ValueError, match="more than once"):
        ampm_stein_projection(
            once, ["a", "b", "c", None], {"a": 0.5}, c=[0, 0, 0, 1]
        )
    with pytest.raises(ValueError, match="no fix-effectiveness"):
        ampm_stein_projection(X, MODES, FEF, c=C, a_modes=["b1"])
    with pytest.raises(ValueError, match="single string"):
        ampm_stein_projection(X, MODES, FEF, c=C, a_modes="a1")
    with pytest.raises(ValueError, match="same end"):
        ampm_stein_projection(
            [10, 20, 50, 15, 60],
            ["b1", "b1", None, "b1", None],
            {"b1": 0.7},
            c=[0, 0, 1, 0, 1],
            i=[1, 1, 1, 2, 2],
        )
    with pytest.raises(ValueError, match="one c=1 row"):
        ampm_stein_projection([10, 20], ["b1", "b1"], {}, c=[0, 0])


def test_ampm_and_stein_against_a_simulation_of_many_modes():
    # 20,000 modes with gamma rates (total 0.5, beta 0.01), tested to 300:
    # the expected intensity of the modes not seen is 0.5 / (1 + 3) = 0.125
    # and the share of the initial intensity seen 0.75. With every seen
    # mode fixed perfectly, the projection is that unseen intensity.
    rng = np.random.default_rng(1126)
    k, lam, beta, end = 20000, 0.5, 0.01, 300.0
    ampm, stein, truth, seen_share = [], [], [], []
    for _ in range(60):
        rates = rng.gamma(lam / (k * beta), beta, k)
        counts = rng.poisson(rates * end)
        seen = np.flatnonzero(counts)
        x, modes = [], []
        for j in seen:
            x += list(rng.uniform(0, end, counts[j]))
            modes += [j] * counts[j]
        x.append(end)
        modes.append(None)
        c = [0] * (len(x) - 1) + [1]
        fef = {j: 1.0 for j in seen}
        a = ampm_projection(x, modes, fef, c=c)
        ampm.append(a.projected_intensity)
        seen_share.append(float(a.maturity(end)))
        stein.append(
            ampm_stein_projection(x, modes, fef, c=c).projected_intensity
        )
        truth.append(rates[counts == 0].sum())
    assert np.mean(truth) == pytest.approx(0.125, rel=0.03)
    assert np.mean(ampm) == pytest.approx(np.mean(truth), rel=0.06)
    assert np.mean(stein) == pytest.approx(np.mean(truth), rel=0.06)
    assert np.mean(seen_share) == pytest.approx(0.75, rel=0.06)


# -- the discrete AMPM-Stein --------------------------------------------------


COUNTS = {"seal": 4, "igniter": 2, "fin": 1, "fuze": 1}
DFEF = {"seal": 0.8, "igniter": 0.7, "fin": 0.6}


def test_adpm_is_the_many_modes_beta_binomial_maximum():
    result = adpm_stein_projection(COUNTS, trials=40, fef=DFEF)
    assert isinstance(result, DiscreteSteinProjection)
    counts = np.array([4, 2, 1, 1.0])
    trials = 40

    # The exact likelihood of 200,000 modes with Beta(a, n - a) chances,
    # a = c / k, c at its maximum for each n: its maximum in n.
    def exact(n, k=200000):
        harmonic = np.sum(1 / (n + np.arange(trials)))
        a = 4 / harmonic / k

        def logp(j):
            return (
                special.betaln(j + a, trials - j + n - a)
                - special.betaln(a, n - a)
                - special.betaln(j + 1, trials - j + 1)
                - np.log(trials + 1)
            )

        return (k - 4) * logp(0) + sum(logp(j) for j in counts)

    n = optimize.minimize_scalar(
        lambda u: -exact(np.exp(u)),
        bounds=(np.log(5), np.log(50)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert result.pseudo_trials == pytest.approx(np.exp(n.x), rel=1e-3)
    theta = trials / (result.pseudo_trials + trials)
    assert result.theta == pytest.approx(theta)
    d = np.array([0.8, 0.7, 0.6, 0.0])
    expected = np.prod(1 - (1 - d) * theta * counts / trials) * np.exp(
        -(1 - theta) * 8 / trials
    )
    assert result.projected_reliability == pytest.approx(expected, rel=1e-12)
    assert round(result.theta, 3) == 0.721
    assert round(result.projected_reliability, 3) == 0.899
    assert round(result.initial_reliability, 3) == 0.820
    assert "discrete AMPM-Stein" in repr(result)


def test_adpm_against_a_simulation_of_many_modes():
    # 20,000 modes with Beta chances (pseudo trials 30, total chance 0.5),
    # 40 trials, every seen mode fixed perfectly: the projection is the
    # chance of no failure from the modes not seen.
    rng = np.random.default_rng(1126)
    k, n, total, trials = 20000, 30.0, 15.0, 40
    a = total / k
    est, truth = [], []
    for _ in range(100):
        p = rng.beta(a, n - a, k)
        failed = rng.binomial(trials, p)
        seen = np.flatnonzero(failed)
        try:
            result = adpm_stein_projection(
                {j: int(failed[j]) for j in seen},
                trials,
                fef={j: 1.0 for j in seen},
            )
        except ValueError:
            continue
        est.append(result.projected_reliability)
        truth.append(np.prod(1 - p[failed == 0]))
    assert len(est) >= 95
    assert np.mean(est) == pytest.approx(np.mean(truth), abs=0.01)


def test_adpm_refusals():
    with pytest.raises(ValueError, match="more than once"):
        adpm_stein_projection({"a": 1, "b": 1}, 10)
    with pytest.raises(ValueError, match="trials must be"):
        adpm_stein_projection(COUNTS, 0)
    with pytest.raises(ValueError, match="trials must be"):
        adpm_stein_projection(COUNTS, 40.5)
    with pytest.raises(ValueError, match="whole number"):
        adpm_stein_projection({"a": 2.5, "b": 1}, 10)
    with pytest.raises(ValueError, match="more trials"):
        adpm_stein_projection({"a": 12, "b": 1}, 10)
    with pytest.raises(ValueError, match="dict"):
        adpm_stein_projection([4, 2], 40)
    with pytest.raises(ValueError, match="no failures"):
        adpm_stein_projection(COUNTS, 40, fef={"wing": 0.5})


# -- the bootstrap bounds -----------------------------------------------------


def _fits():
    return (
        ampm_projection(X, MODES, FEF, c=C),
        ampm_stein_projection(X, MODES, FEF, c=C),
        adpm_stein_projection(COUNTS, trials=40, fef=DFEF),
    )


def _cb(fit, **kw):
    if isinstance(fit, DiscreteSteinProjection):
        return fit.projected_reliability_cb(**kw)
    return fit.projected_intensity_cb(**kw)


def _point(fit):
    if isinstance(fit, DiscreteSteinProjection):
        return fit.projected_reliability
    return fit.projected_intensity


@pytest.mark.parametrize("target", ["realised", "expected"])
@pytest.mark.parametrize("which", [0, 1, 2])
def test_bootstrap_bounds_are_seeded_and_consistent(which, target):
    fit = _fits()[which]
    kw = dict(target=target, n_boot=300, random_state=7)
    two = _cb(fit, alpha_ci=0.1, **kw)
    assert two.shape == (2,) and two[0] < _point(fit) < two[1]
    np.testing.assert_array_equal(two, _cb(fit, alpha_ci=0.1, **kw))
    # A one-sided bound at alpha is the two-sided one at 2 alpha.
    lower = _cb(fit, alpha_ci=0.05, bound="lower", **kw)
    upper = _cb(fit, alpha_ci=0.05, bound="upper", **kw)
    assert np.ndim(lower) == 0
    assert lower == pytest.approx(two[0], rel=1e-14)
    assert upper == pytest.approx(two[1], rel=1e-14)
    if not isinstance(fit, DiscreteSteinProjection):
        mtbf = fit.projected_mtbf_cb(alpha_ci=0.1, **kw)
        np.testing.assert_allclose(mtbf, 1 / two[::-1], rtol=1e-15)
        low = fit.projected_mtbf_cb(alpha_ci=0.05, bound="lower", **kw)
        assert low == pytest.approx(1 / two[1], rel=1e-14)


def test_ampm_bounds_over_time():
    fit = _fits()[0]
    t = np.array([200.0, 400.0, 800.0, np.inf])
    for target in ("realised", "expected"):
        b = fit.intensity_cb(t, target=target, n_boot=300, random_state=3)
        assert b.shape == (4, 2)
        point = fit.intensity(t)
        assert np.all(b[:, 0] < point) and np.all(point < b[:, 1])
    assert fit.intensity(np.inf) == pytest.approx(
        fit.growth_potential_intensity
    )
    b = fit.intensity_cb(400.0, n_boot=300, random_state=3)
    np.testing.assert_array_equal(
        b, fit.projected_intensity_cb(n_boot=300, random_state=3)
    )
    one = fit.mtbf_cb(t, bound="lower", n_boot=300, random_state=3)
    assert one.shape == (4,)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_bootstrap_bounds_refusals(which):
    fit = _fits()[which]
    with pytest.raises(ValueError, match="only bootstrap bounds"):
        _cb(fit, method="wald")
    with pytest.raises(ValueError, match="'bound' must be"):
        _cb(fit, bound="both")
    with pytest.raises(ValueError, match="'target' must be"):
        _cb(fit, target="true")
    with pytest.raises(ValueError, match="n_boot must be"):
        _cb(fit, n_boot=1)
    with pytest.raises(ValueError, match="alpha_ci"):
        _cb(fit, alpha_ci=1.5)
    if which == 0:
        with pytest.raises(ValueError, match="times from 0"):
            fit.intensity_cb(-1.0)


def test_discrete_bootstrap_population_matches_the_fit():
    # The simulated population's expected modes seen is the number seen,
    # m, and its expected failures a trial the fitted total chance c.
    fit = _fits()[2]
    T, n = fit.trials, fit.pseudo_trials
    j = np.arange(1, T + 1, dtype=float)
    c = 4 / (n * np.sum(1 / (n + np.arange(T))))
    mean_j = c * n * special.comb(T, j) * np.exp(special.betaln(j, n + T - j))
    assert mean_j.sum() == pytest.approx(4, rel=1e-12)
    assert np.sum(j * mean_j) / T == pytest.approx(c, rel=1e-12)


def test_the_unseen_modes_draws_have_their_exact_moments():
    from surpyval.recurrent.maturity_projection import (
        _beta_unseen,
        _gamma_realised,
    )

    rng = np.random.default_rng(1)
    # Beta population: -sum ln(1 - p) over the unseen modes has mean
    # c n psi'(n + T) and variance -c n psi''(n + T).
    c, n, T = 0.5, 30.0, 40
    draws = np.array([_beta_unseen(rng, c * n, n + T) for _ in range(40000)])
    mean = c * n * special.polygamma(1, n + T)
    var = -c * n * special.polygamma(2, n + T)
    assert draws.mean() == pytest.approx(mean, rel=0.01)
    assert draws.var() == pytest.approx(var, rel=0.03)
    # Gamma population (shape a, scale s): the unseen modes' intensity
    # once those surfacing in u more are fixed has mean
    # a s_u + (1 - mu) (a s - a s_u), s_u = s / (1 + s u).
    a, s, pool = 8.0, 0.004, np.array([0.5, 0.9])
    for u in (0.0, 300.0, np.inf):
        draws = np.array(
            [_gamma_realised(rng, a, s, pool, u) for _ in range(40000)]
        )
        s_u = 0.0 if np.isinf(u) else s / (1 + s * u)
        mean = a * s_u + 0.3 * (a * s - a * s_u)
        assert draws.mean() == pytest.approx(mean, rel=0.01), u


def _coverage(reps, n_boot, seed):
    """Two-sided 90% bounds of each model and target on ``reps``
    simulated tests: the share covering each target's truth."""
    from surpyval.tests.recurrent import _maturity_sims as sims

    rng = np.random.default_rng(seed)
    alpha = 0.1
    hits: dict = {}

    def record(name, target, bounds, truth):
        hits.setdefault((name, target), []).append(
            bounds[0] <= truth <= bounds[1]
        )

    expected = sims.continuous_truth()
    for rep in range(reps):
        x, modes, c, fef, a_modes, real = sims.continuous_test(rng)
        for name in ("ampm", "stein"):
            for target, truth in (("realised", real), ("expected", expected)):
                try:
                    if name == "ampm":
                        fit = ampm_projection(x, modes, fef, c=c)
                    else:
                        fit = ampm_stein_projection(
                            x, modes, fef, c=c, a_modes=a_modes
                        )
                    b = fit.projected_intensity_cb(
                        alpha, target=target, n_boot=n_boot, random_state=rep
                    )
                except ValueError:  # the fit refused: a miss
                    b = np.array([np.nan, np.nan])
                record(name, target, b, truth)
    expected = sims.discrete_truth()
    for rep in range(reps):
        failures, fef, real = sims.discrete_test(rng)
        for target, truth in (("realised", real), ("expected", expected)):
            try:
                b = adpm_stein_projection(
                    failures, 40, fef=fef
                ).projected_reliability_cb(
                    alpha, target=target, n_boot=n_boot, random_state=rep
                )
            except ValueError:
                b = np.array([np.nan, np.nan])
            record("discrete", target, b, truth)
    return {k: float(np.mean(v)) for k, v in hits.items()}


def test_bootstrap_coverage_reduced():
    # A reduced, seeded version of the calibration study (sizes fixed in
    # advance): 100 simulated tests of 20,000 modes each, 200 replicates,
    # two-sided 90% bounds, each target against its own truth.
    cover = _coverage(reps=100, n_boot=200, seed=1126)
    print(cover)
    for key, value in cover.items():
        assert abs(value - 0.9) <= 0.09, (key, value)


def test_stein_bounds_narrow_as_the_test_lengthens():
    # The realised projection is learnt ever better from the repeats; the
    # AMPM, which reads only first occurrences, does not narrow (see its
    # notes), so it is not checked here.
    from surpyval.tests.recurrent import _maturity_sims as sims

    widths: dict = {"stein": [], "discrete": []}
    for end in (300.0, 3000.0, 30000.0):
        rng = np.random.default_rng(1127)
        x, modes, c, fef, a_modes, _ = sims.continuous_test(rng, end=end)
        s = ampm_stein_projection(
            x, modes, fef, c=c, a_modes=a_modes
        ).projected_intensity_cb(n_boot=400, random_state=1)
        widths["stein"].append((s[1] - s[0]) / s.mean())
    for trials in (40, 400, 4000):
        rng = np.random.default_rng(1127)
        failures, fef, _ = sims.discrete_test(rng, trials=trials)
        d = adpm_stein_projection(
            failures, trials, fef=fef
        ).projected_reliability_cb(n_boot=400, random_state=1)
        widths["discrete"].append(d[1] - d[0])
    print(widths)
    for name, w in widths.items():
        assert w[0] > w[1] > w[2], (name, w)
    assert widths["stein"][2] < 0.1 and widths["discrete"][2] < 0.05


# -- the handbook's worked example --------------------------------------------


def test_ampm_handbook_crow_extended_example():
    # MIL-HDBK-189C TABLE XV (56 failures in 400 hours; 10 A, 12 BC and 16
    # BD modes), analysed by the AMPM with every BC and BD mode a B mode
    # at the BD modes' average FEF, 0.72 (MIL-HDBK-00189A section 7.6.8,
    # TABLE XIX): growth potential MTBF 15.24, projected MTBF 10.75.
    rows = (
        "0.7 BC1, 3.7 BC1, 13.2 BC1, 15 BD1, 17.6 BC2, 25.3 BD2, 47.5 BD3, "
        "54.0 BD4, 54.5 BC3, 56.4 BD5, 63.6 A, 72.2 BD5, 99.2 BC4, "
        "99.6 BD6, 100.3 BD7, 102.5 A, 112.0 BD8, 112.2 BC5, 120.9 BD2, "
        "121.9 BC6, 125.5 BD9, 133.4 BD10, 151.0 BC7, 163.0 BC8, 164.7 BD9, "
        "174.5 BC9, 177.4 BD10, 191.6 BC10, 192.7 BD11, 213.0 A, 244.8 A, "
        "249.0 BD12, 250.8 A, 260.1 BD1, 263.5 BD8, 273.1 A, 274.7 BD6, "
        "282.8 BC11, 285.0 BD13, 304.0 BD9, 315.4 BD4, 317.1 A, 320.6 A, "
        "324.5 BD12, 324.9 BD10, 342.0 BD5, 350.2 BD3, 355.2 BC12, "
        "364.6 BD10, 364.9 A, 366.3 BD2, 373.0 BD8, 379.4 BD14, 389.0 BD15, "
        "394.9 A, 395.2 BD16"
    )
    x, modes = [], []
    for k, row in enumerate(rows.split(", ")):
        t, mode = row.split()
        x.append(float(t))
        modes.append(f"A{k}" if mode == "A" else mode)
    assert len(x) == 56
    fef = {m: 0.72 for m in set(modes) if m.startswith("B")}
    result = ampm_projection(x + [400], modes + [None], fef, c=[0] * 56 + [1])
    assert result.failures == {"A": 10, "B": 46} and len(result.modes) == 28
    assert round(result.growth_potential_mtbf, 2) == 15.24
    # The handbook's projected MTBF, 10.75, came from a program option for
    # fixes made during the test whose formula is not given; the model's
    # own formula gives 10.68 (0.7% lower), with the same lambda_A,
    # lambda_B and mean FEF (the growth potential agrees).
    assert result.projected_mtbf == pytest.approx(10.75, rel=0.01)
    assert round(result.projected_mtbf, 2) == 10.68
