"""``JelinskiMoranda``: a whole number of faults, each found at the same
rate.

- The likelihood against a sum written out here, the rate's closed form
  for a given number of faults, and the estimate against a search over
  every whole number.
- The condition for a finite estimate against the profile likelihood
  itself, and its refusal.
- A published fit: the Naval Tactical Data System (NTDS) data of
  Jelinski and Moranda (1972), the 26 failures of the production phase.
  The data and the fit are as tabulated by Liu, Xu and Zhou
  (arXiv:1503.00094, Tables 1 and 5), who give the maximum-likelihood
  estimate with the number of faults taken as a real number: N =
  31.2159, phi = 0.006849 (and quote 31.2 from an earlier text). That
  estimate is reproduced here to every digit they print; the model's own
  estimate is the better of the two whole numbers beside it.
- The likelihood-ratio bounds, recovery from simulated campaigns (seeds
  fixed before the results were seen), saving and loading.
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

import surpyval as sp
from surpyval.recurrent import JelinskiMoranda, JelinskiMorandaModel

#: Days between the 26 failures of the NTDS production phase (Jelinski
#: and Moranda 1972; as tabulated in arXiv:1503.00094, Table 1)
NTDS = [
    9, 12, 11, 4, 7, 2, 5, 8, 5, 7, 1, 6, 1,
    9, 4, 1, 3, 3, 6, 1, 11, 33, 7, 91, 2, 1,
]  # fmt: skip

TIMES = np.array([3.0, 8, 12, 20, 25, 33, 41, 47, 58, 70, 81, 99])


def _direct(gaps, N, phi, since_last=0.0):
    """The log-likelihood written out: each gap exponential with rate
    phi (N - i + 1), and no failure in ``since_last`` at the rate left."""
    total = 0.0
    for i, gap in enumerate(gaps, start=1):
        rate = phi * (N - i + 1)
        total += np.log(rate) - rate * gap
    return total - phi * (N - len(gaps)) * since_last


def _simulate(rng, N, phi, end=None, failures=None):
    """One campaign: the ordered lives of N faults, each exponential
    with rate phi, seen to a time or to a number of failures."""
    lives = np.sort(rng.exponential(1.0 / phi, N))
    if failures is not None:
        return lives[:failures], None
    return lives[lives <= end], end


def test_likelihood_is_the_sum_written_out():
    model = JelinskiMoranda.fit(TIMES)
    gaps = np.diff(TIMES, prepend=0.0)
    assert model.log_likelihood == pytest.approx(
        _direct(gaps, model.N, model.phi), rel=1e-13
    )
    for N in (12, 13, 15, 40, 1000):
        # the closed form of the rate given N: n over the fault-time tested
        phi = 12 / sum((N - i) * g for i, g in enumerate(gaps))
        assert model.profile_likelihood(N) == pytest.approx(
            _direct(gaps, N, phi), rel=1e-13
        )
        for other in (0.9 * phi, 1.1 * phi):
            assert _direct(gaps, N, other) < _direct(gaps, N, phi)
    assert model.profile_likelihood(11) == -np.inf
    # the limit of many faults is a constant rate n / T
    limit = 12 * np.log(12 / 99.0) - 12
    assert model.profile_likelihood(np.inf) == pytest.approx(limit, rel=1e-14)
    assert model.profile_likelihood(1e13) == pytest.approx(limit, abs=1e-9)


def test_end_of_testing_enters_the_likelihood():
    x = np.append(TIMES, 130.0)
    c = [0] * 12 + [1]
    model = JelinskiMoranda.fit(x, c)
    gaps = np.diff(TIMES, prepend=0.0)
    assert model.T == 130.0
    assert model.log_likelihood == pytest.approx(
        _direct(gaps, model.N, model.phi, since_last=31.0), rel=1e-13
    )
    # 31 more days without a failure: fewer faults left than without them
    assert model.N <= JelinskiMoranda.fit(TIMES).N
    as_gaps = JelinskiMoranda.fit(np.append(gaps, 31.0), c, gaps=True)
    assert (as_gaps.N, as_gaps.phi) == (model.N, model.phi)
    shuffled = JelinskiMoranda.fit(x[::-1], c[::-1])
    assert (shuffled.N, shuffled.phi) == (model.N, model.phi)


@pytest.mark.parametrize("seed", range(8))
def test_estimate_is_the_best_whole_number(seed):
    rng = np.random.default_rng(seed)
    times, end = _simulate(rng, 30, 0.02, end=120.0)
    stats_ok = times.mean() < (times.size + 1) * end / (2 * times.size)
    if not stats_ok:
        with pytest.raises(ValueError, match="no reliability growth"):
            JelinskiMoranda.fit(np.append(times, end), [0] * times.size + [1])
        return
    model = JelinskiMoranda.fit(np.append(times, end), [0] * times.size + [1])
    N = np.arange(times.size, 5000)
    profile = model.profile_likelihood(N)
    assert model.N == N[np.argmax(profile)]
    # the profile rises and then falls, so the bounds are an interval
    rising = np.diff(profile) > 0
    assert not np.any(rising[np.argmax(~rising) :])
    assert model.phi == pytest.approx(
        times.size / (model.N * end - model.K), rel=1e-14
    )


def test_finite_estimate_condition_is_where_the_profile_turns():
    # Just inside and just outside the condition mean(t) < (n + 1) T /
    # (2 n) = 6 (n = 5, T = 10, the test ending at the last failure):
    # inside, the profile comes down to its limit; outside it climbs to
    # it for ever.
    for shift, finite in ((-0.02, True), (0.02, False)):
        t = np.array([1.0, 3.5, 6.5, 9.0, 10.0])
        t[:4] += shift * 5 / 4  # the mean moves by shift
        assert (t.mean() < 6.0) == finite
        gaps = np.diff(t, prepend=0.0)
        K = float(np.sum(np.arange(5) * gaps))
        # (to 1e6 faults: beyond, the profile's steps are below the
        # rounding of a sum of five logs)
        N = np.unique(np.round(np.geomspace(5, 1e6, 4000)))
        profile = 5 * np.log(5 / (N * 10.0 - K)) + np.array(
            [np.sum(np.log(v - np.arange(5))) for v in N]
        )
        limit = 5 * np.log(5 / 10.0)
        if finite:
            assert profile.max() > limit and np.isfinite(N[profile.argmax()])
            assert JelinskiMoranda.fit(t).N == N[profile.argmax()]
        else:
            assert np.all(np.diff(profile) > 0) and profile.max() < limit
            with pytest.raises(ValueError, match="no reliability growth"):
                JelinskiMoranda.fit(t)


def test_no_growth_is_refused_saying_what_to_do():
    # evenly spread failures: a constant rate
    t = np.arange(1.0, 21.0)
    with pytest.raises(ValueError) as error:
        JelinskiMoranda.fit(t)
    message = str(error.value)
    assert "show no reliability growth" in message
    assert "the 20 failures' mean time, 10.5, is not earlier than" in message
    assert "(n + 1) T / (2 n) = 10.5 (T = 20" in message
    assert "N has no estimate" in message
    assert "HPP" in message and "GoelOkumoto" in message


def test_ntds_published_fit():
    assert len(NTDS) == 26 and sum(NTDS) == 250
    model = JelinskiMoranda.fit(NTDS, gaps=True)
    n, T, K = 26, 250.0, model.K
    assert K == sum(i * g for i, g in enumerate(NTDS))

    # The estimate with N a real number, as published: 31.2159, 0.006849
    def negative_profile(N):
        return -(
            n * np.log(n / (N * T - K)) + gammaln(N + 1) - gammaln(N - n + 1)
        )

    real = minimize_scalar(
        negative_profile,
        bounds=(26.001, 200.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert round(real.x, 4) == 31.2159
    assert round(n / (real.x * T - K), 6) == 0.006849
    # and the model's, the better whole number beside it
    assert model.N == 31
    assert model.profile_likelihood(31) > model.profile_likelihood(32)
    assert model.phi == pytest.approx(26 / (31 * 250.0 - K), rel=1e-15)
    assert model.remaining_faults() == 5
    assert model.current_rate() == pytest.approx(5 * model.phi)
    # 34 faults were found in the end (8 more in the later phases)
    lower, upper = model.param_cb("N")
    assert lower == 26 and upper >= 34


def test_bounds_are_the_likelihood_ratio_sets():
    model = JelinskiMoranda.fit(NTDS, gaps=True)
    best = model.profile_likelihood(model.N)
    for alpha_ci, quantile in (
        (0.05, 3.841458820694124),
        (0.5, 0.454936423119572),
    ):
        lower, upper = model.param_cb("N", alpha_ci=alpha_ci)
        N = np.arange(26, 2000)
        inside = N[2 * (best - model.profile_likelihood(N)) <= quantile]
        assert (lower, upper) == (inside.min(), inside.max())
        left = model.remaining_faults_cb(alpha_ci=alpha_ci)
        assert tuple(left) == (lower - 26, upper - 26)
        # phi: on a fine grid, the best whole N for each phi
        phi_lo, phi_hi = model.param_cb("phi", alpha_ci=alpha_ci)
        for edge in (phi_lo, phi_hi):
            Ns = np.arange(26, 20000)
            values = (
                26 * np.log(edge)
                + gammaln(Ns + 1.0)
                - gammaln(Ns - 25.0)
                - edge * (Ns * 250.0 - model.K)
            )
            assert 2 * (best - values.max()) == pytest.approx(
                quantile, abs=1e-6
            )
        assert phi_lo < model.phi < phi_hi
    # one-sided: the two-sided bound at twice the tail
    assert model.param_cb("N", 0.05, "upper") == model.param_cb("N", 0.1)[1]
    assert model.param_cb("N", 0.05, "lower") == model.param_cb("N", 0.1)[0]


def test_rate_and_mtbf_bounds():
    model = JelinskiMoranda.fit(NTDS, gaps=True)
    best = model.profile_likelihood(model.N)
    lower, upper = model.current_rate_cb()
    # no fault left (N = 26) is inside the bounds on N, so 0 is a rate
    assert model.param_cb("N")[0] == 26 and lower == 0.0
    assert model.current_rate() < upper
    # at the upper bound the best split of the rate over the faults left
    left = np.arange(1, 200000)
    values = (
        26 * np.log(upper / left)
        + gammaln(26 + left + 1.0)
        - gammaln(left + 1.0)
        - (upper / left) * ((26 + left) * 250.0 - model.K)
    )
    assert 2 * (best - values.max()) == pytest.approx(
        3.841458820694124, abs=1e-6
    )
    np.testing.assert_allclose(model.mtbf_cb(), [1 / upper, np.inf])
    assert model.mtbf_cb(bound="lower") == pytest.approx(
        1 / model.current_rate_cb(bound="upper")
    )
    assert model.mtbf() == pytest.approx(1 / model.current_rate())
    assert model.next_failure_sf(10.0) == pytest.approx(
        np.exp(-10 * model.current_rate())
    )


def test_an_unbounded_number_of_faults_is_reported_as_inf():
    # barely inside the condition: any number of faults is as likely
    model = JelinskiMoranda.fit(TIMES)
    assert model.N == 15
    assert model.param_cb("N")[1] == np.inf
    assert model.remaining_faults_cb(bound="upper") == np.inf
    assert model.param_cb("phi")[0] == 0.0
    # a wide enough level bounds it
    assert np.isfinite(model.param_cb("N", alpha_ci=0.6)[1])


def test_no_fault_left():
    model = JelinskiMoranda.fit([2, 5, 9, 14, 20, 27, 35, 46, 59, 75, 98, 130])
    assert model.N == 12 and model.remaining_faults() == 0
    assert model.current_rate() == 0.0 and model.mtbf() == np.inf
    assert model.next_failure_sf(1e9) == 1.0
    lower, upper = model.current_rate_cb()
    assert lower == 0.0 and upper > 0.0
    assert model.remaining_faults_cb()[0] == 0.0


@pytest.mark.parametrize("method", ["wald", "bootstrap"])
def test_other_bound_methods_are_refused_with_the_reason(method):
    model = JelinskiMoranda.fit(NTDS, gaps=True)
    for call in (
        lambda: model.param_cb("N", method=method),
        lambda: model.remaining_faults_cb(method=method),
        lambda: model.current_rate_cb(method=method),
        lambda: model.mtbf_cb(method=method),
    ):
        with pytest.raises(ValueError, match="Use method='lr'"):
            call()
    with pytest.raises(ValueError, match="method"):
        model.param_cb("N", method="exact")
    with pytest.raises(ValueError, match="alpha_ci"):
        model.param_cb("N", alpha_ci=1.5)
    with pytest.raises(ValueError, match="name"):
        model.param_cb("a")


#: (N, phi, end of test): seeds fixed before the results were seen
RECOVERY = [(40, 0.02, 120.0), (100, 0.01, 250.0), (400, 0.005, 500.0)]


@pytest.mark.parametrize("N, phi, end", RECOVERY)
def test_recovery_from_simulated_campaigns(N, phi, end):
    # 200 campaigns each: the true number of faults inside the 95%
    # bounds at least 90% of the time among the campaigns with an
    # estimate (the bounds are likelihood-ratio sets of a whole number,
    # not exact), and the median estimate within 15% of the truth.
    rng = np.random.default_rng(20261120)
    covered, estimates, refused = 0, [], 0
    for _ in range(200):
        times, _ = _simulate(rng, N, phi, end=end)
        try:
            model = JelinskiMoranda.fit(
                np.append(times, end), [0] * times.size + [1]
            )
        except ValueError:
            refused += 1
            continue
        lower, upper = model.param_cb("N")
        covered += lower <= N <= upper
        estimates.append(model.N)
    fitted = len(estimates)
    print(
        "N = {}: {} of 200 refused; coverage {:.3f}; median estimate "
        "{}".format(N, refused, covered / fitted, np.median(estimates))
    )
    assert refused <= 10
    assert covered / fitted >= 0.90
    assert abs(np.median(estimates) - N) <= 0.15 * N


def test_from_params_and_save_and_load():
    model = JelinskiMoranda.fit(NTDS, gaps=True, unit="days")
    back = JelinskiMorandaModel.from_dict(model.to_dict())
    assert (back.N, back.phi, back.n, back.unit) == (31, model.phi, 26, "days")
    np.testing.assert_array_equal(back.param_cb("N"), model.param_cb("N"))
    assert back.log_likelihood == model.log_likelihood
    assert isinstance(sp.from_json(model.to_json()), JelinskiMorandaModel)
    assert repr(back).splitlines()[0] == "Jelinski-Moranda SurPyval Model"

    given = JelinskiMoranda.from_params([20, 0.01], n=15)
    assert given.remaining_faults() == 5
    assert given.cif(np.inf) == 20.0
    np.testing.assert_allclose(given.iif([0.0]), [0.2])
    with pytest.raises(ValueError, match="JelinskiMoranda.fit"):
        given.param_cb("N")
    for bad in ([20.5, 0.01], [0, 0.01], [20, -1.0], [20, 0.01, 3]):
        with pytest.raises(ValueError):
            JelinskiMoranda.from_params(bad)
    with pytest.raises(ValueError, match="from 0 to N = 20"):
        JelinskiMoranda.from_params([20, 0.01], n=21)


def test_data_refusals():
    with pytest.raises(ValueError, match="at least 2 failures"):
        JelinskiMoranda.fit([5.0])
    with pytest.raises(ValueError, match="one campaign has one end"):
        JelinskiMoranda.fit([1.0, 2.0, 3.0, 4.0], c=[0, 0, 1, 1])
    with pytest.raises(ValueError, match="before the last failure"):
        JelinskiMoranda.fit([1.0, 2.0, 5.0, 4.0], c=[0, 0, 0, 1])
    with pytest.raises(ValueError, match="fit GoelOkumoto"):
        JelinskiMoranda.fit([1.0, 2.0, 5.0], c=[0, 2, 0])
    with pytest.raises(ValueError, match="missing or infinite"):
        JelinskiMoranda.fit([1.0, np.nan, 5.0])
    with pytest.raises(ValueError, match="cannot be negative"):
        JelinskiMoranda.fit([1.0, -2.0, 5.0], gaps=True)
    with pytest.raises(ValueError, match="only the last value"):
        JelinskiMoranda.fit([1.0, 2.0, 5.0], c=[0, 1, 0], gaps=True)
    with pytest.raises(ValueError, match="one value for each"):
        JelinskiMoranda.fit([1.0, 2.0, 5.0], c=[0, 1])
    assert repr(JelinskiMoranda) == "JelinskiMoranda: fault-count model fitter"
