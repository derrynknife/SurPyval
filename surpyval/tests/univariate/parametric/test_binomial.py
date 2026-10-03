"""
Tests for the Binomial distribution.

The Binomial is a discrete count distribution (number of events in a fixed
number of pass/fail trials) and, like the Bernoulli, sits outside the
gradient-based MLE machinery. These tests check the closed-form identities,
the closed-form fit, the reduction to the Bernoulli at ``n = 1``, input
validation and serialisation.
"""

import numpy as np
import pytest
from scipy.stats import binom

import surpyval as surv
from surpyval import Bernoulli, Binomial, FixedEventProbability, Parametric
from surpyval.tests._helpers import no_warnings

N, P = 5, 0.3


def test_from_params_repr_and_params():
    model = Binomial.from_params([N, P])
    assert np.allclose(model.params, [N, P])
    assert "Binomial" in repr(model)


def test_pmf_cdf_sf_match_scipy():
    model = Binomial.from_params([N, P])
    k = np.arange(0, N + 1)
    assert np.allclose(model.df(k), binom.pmf(k, N, P))
    assert np.allclose(model.ff(k), binom.cdf(k, N, P))
    assert np.allclose(model.sf(k), binom.sf(k, N, P))


def test_sf_plus_ff_is_one():
    model = Binomial.from_params([N, P])
    k = np.arange(0, N + 1)
    assert np.allclose(model.sf(k) + model.ff(k), 1.0)


def test_pmf_sums_to_one():
    model = Binomial.from_params([N, P])
    assert np.isclose(model.df(np.arange(0, N + 1)).sum(), 1.0)


def test_mean_var_moment_entropy():
    model = Binomial.from_params([N, P])
    assert np.isclose(model.mean(), N * P)
    assert np.isclose(model.var(), N * P * (1 - P))
    assert np.isclose(model.moment(2), binom.moment(2, N, P))
    assert np.isclose(model.entropy(), binom.entropy(N, P))


def test_hazard_and_cumulative_hazard():
    model = Binomial.from_params([N, P])
    k = np.arange(0, N)
    expected_hf = binom.pmf(k, N, P) / (binom.sf(k, N, P) + binom.pmf(k, N, P))
    assert np.allclose(model.hf(k), expected_hf)
    assert np.allclose(model.Hf(k), -np.log(binom.sf(k, N, P)))


def test_qf_roundtrip():
    model = Binomial.from_params([N, P])
    k = np.arange(0, N + 1)
    # The quantile of the cdf returns a value no smaller than k
    assert np.all(model.qf(binom.cdf(k, N, P)) >= k)


def test_conditional_survival():
    model = Binomial.from_params([N, P])
    assert np.isclose(model.cs(1, 2), model.sf(3) / model.sf(2))


def test_random_moments():
    model = Binomial.from_params([N, P])
    np.random.seed(1)
    samples = model.random(100_000)
    assert np.isclose(samples.mean(), N * P, atol=0.05)
    assert np.isclose(samples.var(), N * P * (1 - P), atol=0.05)
    assert samples.min() >= 0
    assert samples.max() <= N


def test_fit_closed_form():
    # 2 + 3 + 1 + 4 = 10 events out of 4 * 5 = 20 trials -> p = 0.5
    model = Binomial.fit([2, 3, 1, 4], n_trials=5)
    assert np.allclose(model.params, [5, 0.5])
    assert isinstance(model, Parametric)


def test_fit_with_counts():
    # 0 events once, 5 events once -> 5 / (2 * 5) = 0.5
    model = Binomial.fit([0, 5], n_trials=5, n=[1, 1])
    assert np.isclose(model.params[1], 0.5)


def test_reduces_to_bernoulli_at_n_one():
    # At n = 1 the binomial *is* the Bernoulli, and since 0.20.0 the two
    # agree exactly on the probability mass:
    binomial = Binomial.from_params([1, P])
    bernoulli = Bernoulli.from_params(P)
    assert np.isclose(binomial.df(1), P)
    assert np.isclose(binomial.df(0), 1 - P)
    np.testing.assert_allclose(
        np.asarray(bernoulli.df([0, 1]), dtype=float),
        np.asarray(binomial.df([0, 1]), dtype=float),
    )

    # Since 0.22 (#344) Bernoulli follows the package's discrete rule
    # R(k) = P(K > k) too, so every function agrees, not just the mass
    # (it used P(X >= x), offset by one from Binomial).
    for fn in ("sf", "ff", "hf", "Hf"):
        np.testing.assert_allclose(
            np.asarray(getattr(bernoulli, fn)([0, 1]), dtype=float),
            np.asarray(getattr(binomial, fn)([0, 1]), dtype=float),
            err_msg=fn,
        )
    u = np.array([0.1, 0.7, 0.75, 0.99])
    np.testing.assert_array_equal(bernoulli.qf(u), binomial.qf(u))

    # Before 0.20.0 Bernoulli was a flat "fixed event probability" model
    # with F(x) = p at every x, which lined up with neither. That model
    # still exists under its own name and is unchanged.
    fixed = FixedEventProbability.from_params(P)
    assert np.isclose(fixed.ff(0), P)
    assert np.isclose(fixed.ff(37.5), P)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x": [1, 6], "n_trials": 5},  # out of range
        {"x": [1, 2], "n_trials": 5, "c": [0, 1]},  # censoring unsupported
        {"x": [1.5, 2], "n_trials": 5},  # non-integer counts
    ],
)
def test_fit_input_validation(kwargs):
    with pytest.raises(ValueError):
        Binomial.fit(**kwargs)


@pytest.mark.parametrize(
    "params",
    [
        [5.5, 0.3],  # non-integer n
        [0, 0.3],  # n must be positive
        [5, 1.3],  # p out of bounds
        [5],  # wrong number of params
    ],
)
def test_from_params_validation(params):
    with pytest.raises(ValueError):
        Binomial.from_params(params)


def test_to_dict_roundtrip():
    model = Binomial.from_params([N, P])
    restored = Parametric.from_dict(model.to_dict())
    assert np.allclose(restored.params, [N, P])
    assert np.isclose(restored.mean(), N * P)


def test_support_brackets_the_outcomes_exclusively():
    # ``support`` is a pair of exclusive bounds -- ``_validate_fit_inputs``
    # rejects ``x <= support[0]`` and ``x >= support[1]`` -- so both must
    # sit one step outside the outcomes {0, ..., n}. Zero events and n
    # events are ordinary outcomes with real mass, and the bounds used to
    # exclude both. Nothing observed it because Binomial does not inherit
    # OptimisedFitMixin, where that check lives.
    n_trials = 5
    for model in (
        Binomial.from_params([n_trials, 0.3]),
        Binomial.fit([0, 2, 3, 5, 1], n_trials=n_trials),
    ):
        lower, upper = model.support
        for k in (0, n_trials):
            assert lower < k < upper, k
            assert Binomial.df(k, n_trials, 0.3) > 0


def test_class_level_support_admits_zero_events():
    # The class-level bound is checked before n is known, so only its
    # lower end is meaningful; it must still admit k = 0, as Poisson's
    # does. It read 0 -- Geometric's value, whose first mass is at k = 1.
    assert Binomial.support[0] < 0


# ---------------------------------------------------------------------------
# The Bernoulli fit takes general inputs (#257).
# ---------------------------------------------------------------------------


def test_bernoulli_fit_works_for_general_inputs():
    assert Bernoulli.fit([0, 1, 1, 0, 1]).params[0] == pytest.approx(0.6)
    assert Bernoulli.fit([1, 1]).params[0] == 1.0
    assert Bernoulli.fit([0, 1], n=[3, 1]).params[0] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        Bernoulli.fit([0, 2, 1])
    assert Bernoulli.from_params(0.3).params[0] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# The hazard beyond ``n`` is quiet.
# ---------------------------------------------------------------------------


def test_binomial_hazard_beyond_n_is_quiet():
    hf = no_warnings(surv.Binomial.hf, np.array([3.0, 6, 7]), 5, 0.3)
    assert hf[1:].tolist() == [0.0, 0.0]


# ----------------------------------------------------------------------------
# #580: confidence bounds on p
# ----------------------------------------------------------------------------


def _three_in_1200():
    # The issue's example: 3 failures (coded 1) in 1200 demands, as
    # Bernoulli outcomes, a FixedEventProbability and a Binomial count.
    return (
        Bernoulli.fit([1, 0], n=[3, 1197]),
        FixedEventProbability.fit([1, 0], n=[3, 1197]),
        Binomial.fit([3], n_trials=1200),
        Binomial.fit([1, 2, 0], n_trials=400),
    )


@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.5])
def test_580_exact_bounds_are_clopper_pearson(alpha):
    # They raised "the Hessian was singular at the optimum" (Wald) or "need
    # the original data" (lr). The default is now the exact interval, as
    # scipy's binomtest gives it (by root finding, to about 1e-9).
    from scipy.stats import binomtest

    ci = binomtest(3, 1200).proportion_ci(1 - alpha, method="exact")
    for model in _three_in_1200():
        np.testing.assert_allclose(
            model.param_cb("p", alpha_ci=alpha), [ci.low, ci.high], rtol=1e-8
        )
        np.testing.assert_allclose(
            model.param_cb("p", alpha_ci=alpha, method="exact"),
            [ci.low, ci.high],
            rtol=1e-8,
        )
    # The issue's 90% interval.
    np.testing.assert_allclose(
        _three_in_1200()[0].param_cb("p", alpha_ci=0.1),
        [0.00068, 0.00645],
        atol=5e-6,
    )


def test_580_one_sided_is_the_matching_end():
    model = Bernoulli.fit([1, 0], n=[3, 1197])
    two = model.param_cb("p", alpha_ci=0.2)
    np.testing.assert_allclose(
        model.param_cb("p", alpha_ci=0.1, bound="lower"), two[:1]
    )
    np.testing.assert_allclose(
        model.param_cb("p", alpha_ci=0.1, bound="upper"), two[1:]
    )


def test_580_zero_failures_give_the_success_run_bound():
    # No failures in 1200: the upper bound is 1 - alpha ** (1 / n), the
    # complement of success_run, and the lower bound 0.
    model = Bernoulli.fit([0], n=[1200])
    upper = model.param_cb("p", alpha_ci=0.1, bound="upper")
    np.testing.assert_allclose(upper, [1 - 0.1 ** (1 / 1200)], rtol=1e-12)
    np.testing.assert_allclose(
        1 - upper, [surv.success_run(1200, alpha_ci=0.1)], rtol=1e-12
    )
    assert model.param_cb("p", alpha_ci=0.1)[0] == 0.0
    # All failures: the upper bound is 1.
    assert Binomial.fit([5, 5], n_trials=5).param_cb("p")[1] == 1.0


def test_580_wald_is_the_logit_interval_and_undefined_at_zero():
    model = Bernoulli.fit([1, 0], n=[3, 1197])
    p, n = 3 / 1200, 1200
    from scipy.stats import norm

    half = norm.ppf(0.95) / np.sqrt(n * p * (1 - p))
    u = np.log(p / (1 - p))
    expected = 1 / (1 + np.exp(-(u + np.array([-half, half]))))
    np.testing.assert_allclose(
        model.param_cb("p", alpha_ci=0.1, method="wald"), expected
    )
    with pytest.warns(RuntimeWarning, match="Wald confidence bound") as w:
        out = Bernoulli.fit([0], n=[1200]).param_cb("p", method="wald")
    assert np.isnan(out).all() and len(w) == 1
    assert w[0].filename == __file__


def test_580_lr_bounds_sit_on_the_deviance_contour():
    from scipy.stats import chi2

    k, n = 3.0, 1200.0

    def loglik(q):
        return k * np.log(q) + (n - k) * np.log1p(-q)

    model = Binomial.fit([3], n_trials=1200)
    lo, hi = model.param_cb("p", alpha_ci=0.1, method="lr")
    crit = chi2.ppf(0.9, 1)
    for q in (lo, hi):
        assert 2 * (loglik(k / n) - loglik(q)) == pytest.approx(crit, rel=1e-8)
    # With no events the upper bound has a closed form, 1 - exp(-c / 2N).
    zero = Bernoulli.fit([0], n=[1200])
    np.testing.assert_allclose(
        zero.param_cb("p", alpha_ci=0.1, method="lr"),
        [0.0, 1 - np.exp(-crit / 2400)],
        rtol=1e-8,
    )


def test_580_other_bounds_say_to_use_param_cb():
    for model in _three_in_1200():
        for call in (
            lambda: model.cb([0, 1]),
            lambda: model.quantile_cb(0.5),
            lambda: model.mean_cb(),
        ):
            with pytest.raises(ValueError, match=r"param_cb\('p'\)"):
                call()
    with pytest.raises(ValueError, match="'method' must be one of"):
        Bernoulli.fit([1, 0]).param_cb("p", method="profile-ish")
    with pytest.raises(ValueError, match="Unknown parameter"):
        Bernoulli.fit([1, 0]).param_cb("q")


def test_580_binomial_n_is_known():
    model = Binomial.fit([3], n_trials=1200)
    np.testing.assert_array_equal(model.param_cb("n"), [1200.0, 1200.0])
    np.testing.assert_array_equal(model.param_cb("n", bound="lower"), [1200.0])


def test_580_bounds_survive_a_round_trip_and_need_counts():
    for model in _three_in_1200():
        restored = surv.from_dict(model.to_dict())
        np.testing.assert_array_equal(
            restored.param_cb("p"), model.param_cb("p")
        )
    with pytest.raises(ValueError, match="counts of events and trials"):
        Bernoulli.from_params(0.01).param_cb("p")
    old = Bernoulli.fit([1, 0]).to_dict()
    del old["event_counts"]
    with pytest.raises(ValueError, match="counts of events and trials"):
        surv.from_dict(old).param_cb("p")


# -- #608: a number of trials per row ---------------------------------------
LOTS_X, LOTS_TRIALS, LOTS_N = [1, 0, 3, 2], [20, 50, 80, 50], [1, 2, 1, 1]


def test_608_per_row_trials_estimate_p_from_all_the_trials():
    model = Binomial.fit(LOTS_X, n_trials=LOTS_TRIALS, n=LOTS_N)
    events = np.dot(LOTS_X, LOTS_N)
    trials = np.dot(LOTS_TRIALS, LOTS_N)
    assert model.params[1] == pytest.approx(events / trials, rel=1e-15)
    # The maximum of the per-row likelihood
    from scipy.optimize import minimize_scalar

    def nll(p):
        return -np.sum(
            np.asarray(LOTS_N) * binom.logpmf(LOTS_X, LOTS_TRIALS, p)
        )

    best = minimize_scalar(nll, bounds=(1e-6, 0.5), method="bounded")
    assert model.params[1] == pytest.approx(best.x, rel=1e-4)
    # No single number of trials
    assert np.isnan(model.params[0])


def test_608_exact_bounds_hold_for_unequal_trials():
    # The events in all the trials are Binomial(sum of trials, p) whatever
    # the rows' sizes, so Clopper-Pearson on the totals is exact: scipy's
    # binomtest on the same totals.
    from scipy.stats import binomtest

    model = Binomial.fit(LOTS_X, n_trials=LOTS_TRIALS, n=LOTS_N)
    events = int(np.dot(LOTS_X, LOTS_N))
    trials = int(np.dot(LOTS_TRIALS, LOTS_N))
    ci = binomtest(events, trials).proportion_ci(confidence_level=0.95)
    np.testing.assert_allclose(
        model.param_cb("p"), [ci.low, ci.high], rtol=1e-10
    )


def test_608_equal_trials_per_row_are_the_scalar_fit():
    a = Binomial.fit([2, 3, 1], n_trials=[5, 5, 5])
    b = Binomial.fit([2, 3, 1], n_trials=5)
    np.testing.assert_array_equal(a.params, b.params)
    assert a.to_dict() == b.to_dict()


def test_608_per_row_trials_round_trip_and_pickle():
    import json
    import pickle

    model = Binomial.fit(LOTS_X, n_trials=LOTS_TRIALS)
    text = json.dumps(model.to_dict(), allow_nan=False)
    for restored in (
        surv.from_dict(json.loads(text)),
        pickle.loads(pickle.dumps(model)),
    ):
        np.testing.assert_array_equal(restored.params, model.params)
        np.testing.assert_array_equal(
            restored.param_cb("p"), model.param_cb("p")
        )
        np.testing.assert_array_equal(restored.support, [-1, 81])


def test_608_functions_of_the_count_need_one_number_of_trials():
    model = Binomial.fit(LOTS_X, n_trials=LOTS_TRIALS)
    for call in (
        lambda: model.sf(1),
        lambda: model.df(1),
        lambda: model.hf(1),
        lambda: model.qf(0.5),
        lambda: model.mean(),
        lambda: model.random(3),
    ):
        with pytest.raises(ValueError, match="no single n"):
            call()
    ten = model.with_params([10, model.params[1]])
    np.testing.assert_allclose(
        ten.sf([0, 1]), binom.sf([0, 1], 10, model.params[1])
    )


@pytest.mark.parametrize(
    "n_trials", [[20, 50], [0, 50, 80, 50], [1.5, 50, 80, 50], [[20] * 4]]
)
def test_608_per_row_trials_are_checked(n_trials):
    with pytest.raises(ValueError, match="n_trials"):
        Binomial.fit(LOTS_X, n_trials=n_trials)


def test_608_a_row_with_more_events_than_trials_is_refused():
    with pytest.raises(ValueError, match="between 0 and 'n_trials'"):
        Binomial.fit([1, 0, 30], n_trials=[20, 50, 20])
