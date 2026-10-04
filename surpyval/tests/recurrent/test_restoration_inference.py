"""The restoration factor is printed with its uncertainty (#513), and
tested against perfect and minimal repair.

A generalized renewal fit printed ``q`` as a bare number, so a fit to
minimal-repair data (true ``q = 1``) reported ``q = 2.63`` -- "every
repair leaves the truck worse than before it failed" -- although its 95%
interval ran from 0.094 to 73.4. The model now prints every parameter
with its standard error and Wald interval, and ``repair_test`` tests the
fit against perfect repair (``q = 0``; ``rho = 1``) and minimal repair
(``q = 1``; ``rho = 0``) by likelihood ratio. The printed model gives
the tests' one-line conclusion in place of the earlier ad hoc flag (an
interval covering both ``q = 0.5`` and ``q = 2``).
"""

import warnings

import numpy as np
import pytest
from scipy.stats import chi2

import surpyval as sp
from surpyval import recurrent as rc
from surpyval.recurrent.renewal.renewal_model import RenewalModel


def _trucks():
    # The issue's eight haul trucks under minimal repair (power law).
    rows = []
    rng = np.random.default_rng(8)
    for k in range(8):
        T = rng.uniform(6000, 12000)
        N = rng.poisson(2e-4 * T**1.35)
        ts = np.sort(T * rng.random(N) ** (1 / 1.35))
        rows += [(h, k, 0) for h in ts] + [(T, k, 1)]
    return map(np.array, zip(*rows))


def _renewals(n_items=30, n_events=10, seed=0):
    # As good as new (q = 0): each gap a fresh Weibull(10, 3) life.
    rng = np.random.default_rng(seed)
    gaps = 10.0 * rng.weibull(3.0, (n_items, n_events))
    x = np.cumsum(gaps, axis=1).ravel()
    i = np.repeat(np.arange(n_items), n_events)
    return x, i, np.zeros(x.size, int)


def _kijima(q, seed=1):
    # Strong wear-out (beta = 3), 20 systems with 13 failures each.
    truth = rc.GeneralizedRenewal.fit_from_parameters([10, 3], q)
    d = truth.count_terminated_simulation_data(12, items=20, random_state=seed)
    return d.x, d.i, d.c


def _renewal_log_likelihood(model):
    # A renewal process: the distribution fitted to the times between
    # failures, the last gap of each system censored.
    data = model.data
    fit = sp.Weibull.fit(data.interarrival_times, data.c, data.n)
    return -fit.neg_ll()


def _text(model):
    return " ".join(repr(model).split())


def test_trucks_print_their_interval_and_the_repair_conclusion():
    x, i, c = _trucks()
    model = rc.GeneralizedRenewal.fit(x, i, c)
    table = model.summary()
    assert table.loc["q", "estimate"] == pytest.approx(2.626, abs=1e-3)
    assert table.loc["q", "se"] == pytest.approx(4.463, rel=1e-2)
    np.testing.assert_allclose(
        table.loc["q", ["lower 95%", "upper 95%"]], [0.0939, 73.44], rtol=1e-2
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cb = model.param_cb("q")
    np.testing.assert_allclose(
        table.loc["q", ["lower 95%", "upper 95%"]], cb, rtol=1e-12
    )
    # The ad hoc flag is gone; the tests' conclusion is printed instead.
    text = _text(model)
    assert "covers both" not in text
    assert (
        "Repair test: consistent with minimal repair; perfect repair "
        "rejected" in text
    )


def test_trucks_repair_test():
    x, i, c = _trucks()
    model = rc.GeneralizedRenewal.fit(x, i, c)
    result = model.repair_test()
    # q = 1 with a Weibull is the power-law NHPP: the restricted maximum is
    # Crow-AMSAA's.
    crow = rc.CrowAMSAA.fit(x, i, c)
    minimal = result.minimal
    assert minimal.value == 1.0 and not minimal.boundary
    assert minimal.log_likelihood == pytest.approx(
        crow.log_likelihood, abs=1e-4
    )
    assert minimal.statistic == pytest.approx(0.398, abs=2e-3)
    assert minimal.p_value == pytest.approx(0.528, abs=2e-3)
    assert minimal.df == 1
    assert minimal.params[0] == 1.0
    # q = 0 is a Weibull renewal process, on the edge of q's range.
    perfect = result.perfect
    assert perfect.value == 0.0 and perfect.boundary
    assert perfect.log_likelihood == pytest.approx(
        _renewal_log_likelihood(model), abs=1e-4
    )
    assert perfect.statistic == pytest.approx(34.91, abs=1e-2)
    assert perfect.p_value < 1e-8
    assert result.conclusion == (
        "consistent with minimal repair; perfect repair rejected"
    )
    assert result.log_likelihood == pytest.approx(model.log_likelihood)
    text = repr(result)
    assert "perfect repair" in text and "minimal repair" in text
    assert "Conclusion (5% level)" in text
    # At a 1e-10 level neither is rejected: not determined.
    strict = model.repair_test(alpha_ci=1e-10)
    assert strict.conclusion == (
        "not determined: the data are consistent with both perfect and "
        "minimal repair"
    )


def test_renewal_data_reject_minimal_repair():
    x, i, c = _renewals()
    model = rc.GeneralizedRenewal.fit(x, i, c)
    result = model.repair_test()
    assert result.minimal.p_value < 1e-6
    # The estimate is on the edge, q = 0: the statistic is 0, p = 1.
    assert result.perfect.statistic == pytest.approx(0, abs=1e-6)
    assert result.perfect.p_value == 1.0
    assert result.conclusion == (
        "consistent with perfect repair; minimal repair rejected"
    )
    assert result.perfect.log_likelihood == pytest.approx(
        _renewal_log_likelihood(model), abs=1e-4
    )
    assert "consistent with perfect repair" in _text(model)


def test_minimal_repair_data_with_wear_out_reject_perfect_repair():
    x, i, c = _kijima(1.0)
    model = rc.GeneralizedRenewal.fit(x, i, c)
    result = model.repair_test()
    assert result.perfect.p_value < 1e-10
    assert result.minimal.p_value > 0.5
    assert result.conclusion == (
        "consistent with minimal repair; perfect repair rejected"
    )
    assert result.minimal.log_likelihood == pytest.approx(
        rc.CrowAMSAA.fit(x, i, c).log_likelihood, abs=1e-4
    )


def test_intermediate_repair_rejects_both():
    x, i, c = _kijima(0.5)
    model = rc.GeneralizedRenewal.fit(x, i, c)
    result = model.repair_test()
    assert result.perfect.p_value < 1e-10
    assert result.minimal.p_value < 1e-3
    assert result.conclusion.startswith(
        "both perfect and minimal repair rejected: q = 0.48"
    )
    assert result.conclusion.endswith("is between perfect and minimal repair")


def test_worse_than_minimal_repair():
    # q = 2: each repair adds twice the age gained since the last one.
    x, i, c = _kijima(2.0, seed=3)
    model = rc.GeneralizedRenewal.fit(x, i, c)
    result = model.repair_test()
    assert model.q > 1
    assert result.conclusion.endswith("is worse than minimal repair")


def test_boundary_p_values_are_halved():
    x, i, c = _kijima(0.5)
    # Kijima q >= 0: q = 0 is on the edge, q = 1 is not.
    result = rc.GeneralizedRenewal.fit(x, i, c).repair_test()
    assert result.perfect.boundary and not result.minimal.boundary
    assert result.perfect.p_value == pytest.approx(
        0.5 * chi2.sf(result.perfect.statistic, 1), rel=1e-12
    )
    assert result.minimal.p_value == pytest.approx(
        chi2.sf(result.minimal.statistic, 1), rel=1e-12
    )
    # ARA 0 <= rho <= 1: both ends are on the edge.
    result = rc.ARA.fit(x, i, c).repair_test()
    for test in (result.perfect, result.minimal):
        assert test.boundary
        assert test.p_value == pytest.approx(
            0.5 * chi2.sf(test.statistic, 1), rel=1e-12
        )
    # G1 q > -1: q = 0 is inside the range.
    result = rc.GeneralizedOneRenewal.fit(x, i, c).repair_test()
    assert not result.perfect.boundary
    assert result.perfect.p_value == pytest.approx(
        chi2.sf(result.perfect.statistic, 1), rel=1e-12
    )


def test_ara_and_ari_test_rho():
    x, i, c = _renewals(20, 8, seed=1)
    model = rc.ARA.fit(x, i, c)
    # renewals: as good as new is rho = 1, the other edge of its range
    assert model.rho == pytest.approx(1.0, abs=1e-6)
    assert "rho = 1 is at the edge" in _text(model)
    result = model.repair_test()
    assert (result.perfect.value, result.minimal.value) == (1.0, 0.0)
    assert result.minimal.p_value < 1e-6
    assert result.perfect.p_value == 1.0
    assert result.perfect.log_likelihood == pytest.approx(
        _renewal_log_likelihood(model), abs=1e-4
    )
    assert "consistent with perfect repair; minimal repair rejected" in (
        _text(model)
    )

    x, i, c = _trucks()
    model = rc.ARA.fit(x, i, c)
    assert np.isnan(model.summary().loc["rho", "se"])  # rho -> 0
    result = model.repair_test()
    # rho = 0 with a Weibull is Crow-AMSAA, as q = 1 is.
    assert result.minimal.log_likelihood == pytest.approx(
        rc.CrowAMSAA.fit(x, i, c).log_likelihood, abs=1e-4
    )
    assert result.conclusion == (
        "consistent with minimal repair; perfect repair rejected"
    )

    # ARI: rho = 0 is its baseline NHPP; rho = 1 is "maximal" repair.
    model = rc.ARI.fit(x, i, c)
    result = model.repair_test()
    assert result.minimal.log_likelihood == pytest.approx(
        rc.CrowAMSAA.fit(x, i, c).log_likelihood, abs=1e-4
    )
    assert result.perfect.hypothesis == "maximal repair"
    assert "maximal repair" in result.conclusion


def test_g1_tests_perfect_repair_only():
    x, i, c = _renewals()
    model = rc.GeneralizedOneRenewal.fit(x, i, c)
    result = model.repair_test()
    assert result.minimal is None
    assert result.perfect.value == 0.0
    assert result.perfect.log_likelihood == pytest.approx(
        _renewal_log_likelihood(model), abs=1e-4
    )
    assert result.perfect.p_value > 0.05
    assert result.conclusion.startswith("consistent with perfect repair")
    assert "no minimal repair" in result.conclusion
    # Shrinking times between failures: a renewal process is rejected.
    x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])
    result = rc.GeneralizedOneRenewal.fit(x).repair_test()
    assert result.perfect.p_value < 0.05
    assert "deterioration" in result.conclusion


def test_the_refits_run_once_and_not_at_fit(monkeypatch):
    calls = []
    original = RenewalModel._restricted_fit

    def counting(self, *args):
        calls.append(args[0])
        return original(self, *args)

    monkeypatch.setattr(RenewalModel, "_restricted_fit", counting)
    x, i, c = _trucks()
    model = rc.GeneralizedRenewal.fit(x, i, c)
    assert calls == []
    repr(model)
    assert sorted(calls) == ["minimal", "perfect"]
    repr(model)
    model.repair_test()
    model.repair_test(alpha_ci=0.01)
    assert len(calls) == 2


def test_a_failed_refit_is_reported_not_raised():
    x, i, c = _trucks()
    model = rc.GeneralizedRenewal.fit(x, i, c)
    neg_ll = model._neg_ll

    def no_minimal(params):
        # No finite likelihood anywhere with q = 1.
        return np.inf if params[0] == 1.0 else neg_ll(params)

    model._neg_ll = no_minimal
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = _text(model)
        result = model.repair_test()
        repr(model)
    assert len(caught) == 1
    assert "repair test is not available" in str(caught[0].message)
    assert caught[0].filename == __file__
    assert not result.minimal.available
    assert np.isnan(result.minimal.p_value)
    assert result.perfect.available
    assert "(minimal repair test not available)" in text
    assert "not available" in repr(result)


def test_a_model_without_data_has_no_tests():
    x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])
    model = rc.GeneralizedRenewal.fit(x, dist=sp.Weibull)
    restored = RenewalModel.from_dict(model.to_dict())
    assert "Repair test : not available (no data)" in _text(restored)
    given = rc.GeneralizedRenewal.fit_from_parameters([10, 2], 0.2)
    assert "Restoration Factor  : 0.2" in repr(given)
    with pytest.raises(ValueError, match="Likelihood inference"):
        given.repair_test()


def test_q_at_its_edge_says_so():
    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
    c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
    model = rc.GeneralizedRenewal.fit(x, i, c)
    assert "at the edge of its range" in _text(model)
    assert np.isnan(model.summary().loc["q", "se"])


@pytest.mark.parametrize(
    "fitter", [rc.GeneralizedRenewal, rc.ARA, rc.GeneralizedOneRenewal]
)
def test_663_too_few_failures_is_said_in_recurrent_terms(fitter):
    with pytest.raises(ValueError, match="1 distinct time.s. between") as info:
        fitter.fit([50.0, 100.0], c=[0, 1])
    assert "fixed=" not in str(info.value)


def test_663_restored_models_say_they_carry_no_data():
    x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
    i = [1] * 6 + [2] * 5
    c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    for model in (
        rc.CrowAMSAA.fit(x, i, c),
        rc.GeneralizedRenewal.fit(x, i, c),
    ):
        restored = type(model).from_dict(model.to_dict())
        with pytest.raises(
            ValueError, match="restored with from_dict / from_json"
        ):
            restored.param_cb(restored._parameter_names()[0])
    restored = rc.CrowAMSAA.fit(x, i, c)
    restored = type(restored).from_dict(restored.to_dict())
    with pytest.raises(
        ValueError, match="restored with from_dict / from_json"
    ):
        restored.cif_cb(10.0)


def test_665_profile_interval_for_an_interior_restoration_parameter():
    # method="lr": the values the likelihood-ratio test does not reject,
    # from the estimate out on each side; the Wald interval on a Kijima q
    # under-covered (83% for a nominal 90%).
    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
    c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
    model = rc.GeneralizedOneRenewal.fit(x, i, c)
    q = float(model._mle[0])
    lower, upper = model.param_cb("q", alpha_ci=0.1, method="lr")
    assert lower < q < upper
    crit = chi2.ppf(0.9, 1)
    for end in (lower, upper):
        drop = 2 * (model.log_likelihood - model._profile_ll(end))
        assert drop == pytest.approx(crit, abs=1e-5)
    # A one-sided bound is the two-sided one's end at twice alpha.
    assert model.param_cb("q", 0.05, "upper", method="lr")[0] == (
        pytest.approx(upper, rel=1e-6)
    )
    with pytest.raises(ValueError, match="restoration parameter 'q' only"):
        model.param_cb("alpha", method="lr")
    with pytest.raises(ValueError, match="'method' must be one of"):
        model.param_cb("q", method="profile")
