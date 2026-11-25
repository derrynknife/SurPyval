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
