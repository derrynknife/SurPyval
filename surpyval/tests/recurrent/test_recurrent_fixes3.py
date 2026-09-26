"""Regression tests for the third round of recurrent-event fixes."""

import warnings

import numpy as np
import pandas as pd
import pytest

from surpyval import LogNormal, handle_xicn
from surpyval.recurrent import (
    ARA,
    ARI,
    HPP,
    CauseSpecificNHPP,
    CoxLewis,
    CrowAMSAA,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
    NonParametricCounting,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
    RenewalModel,
)
from surpyval.recurrent.renewal.fit_mixin import RenewalFitMixin
from surpyval.utils.recurrent_event_data import RecurrentEventData

# --- simulation ---------------------------------------------------------


def _kijima_i_weibull_mcf(alpha, beta, q, t, items=4000, seed=0):
    """Exact-sampling reference: Kijima-I virtual age, Weibull lifetime,
    each gap from H(v + x) = H(v) + E in closed form."""
    rng = np.random.default_rng(seed)
    v = np.zeros(items)
    now = np.zeros(items)
    counts = np.zeros((items, len(t)))
    alive = np.ones(items, dtype=bool)
    while alive.any():
        e = rng.exponential(size=items)
        x = alpha * ((v / alpha) ** beta + e) ** (1 / beta) - v
        now = now + x
        v = v + q * x
        alive &= now <= t.max()
        counts += alive[:, None] & (now[:, None] <= t[None, :])
    return counts.mean(axis=0)


def test_grp_simulated_mcf_is_right_at_long_horizons():
    # q = 1 (Kijima-I) is minimal repair: the MCF is (t / 10) ** 3. The
    # old sampler, qf(1 - u * sf(v)), lost all precision once sf(v) fell
    # below 1e-16 and gave about [8, 37] with a spurious asymptote warning.
    model = GeneralizedRenewal.fit_from_parameters([10, 3], 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mcf = model.mcf([20, 40], items=200, seed=1)
    assert np.allclose(mcf, [8, 64], rtol=0.1)


def test_grp_partial_repair_matches_exact_sampling():
    t = np.array([40.0, 70.0])
    truth = _kijima_i_weibull_mcf(10.0, 3.0, 0.5, t)
    model = GeneralizedRenewal.fit_from_parameters([10, 3], 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mcf = model.mcf(t, items=200, seed=2)
    # the old sampler gave 77.5 at 70 (exact about 89.6), and stayed
    # there: its curve was flat from 70 on
    assert np.allclose(mcf, truth, rtol=0.05)


def test_ara_simulated_mcf_is_right_at_long_horizons():
    model = ARA.fit_from_parameters([10, 3], 0.0, m=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mcf = model.mcf([20, 40], items=200, seed=1)
    assert np.allclose(mcf, [8, 64], rtol=0.1)


def test_nhpp_simulation_does_not_underflow():
    # exp(-cif) underflowed once the expected count passed about 745.
    model = CrowAMSAA.from_params([10, 3])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim = model.time_terminated_simulation(100, items=200, seed=1)
    assert np.isclose(sim.mcf(100)[0], 1000, rtol=0.05)


def test_uniform_stream_needs_no_prior_simulation():
    # used to raise AttributeError: 'us' on a model that had not simulated
    model = CrowAMSAA.from_params([1.0, 1.0])
    assert 0 < model.get_uniform_random_number() < 1
    grp = GeneralizedRenewal.fit_from_parameters([10, 2], 0.2)
    restored = RenewalModel.from_dict(grp.to_dict())
    assert 0 < restored.get_uniform_random_number() < 1


# --- renewal fitting ----------------------------------------------------

_X = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60])
_I = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
_C = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])


@pytest.mark.parametrize("m", [1, 2, np.inf])
def test_ara_finds_the_perfect_repair_maximum(m):
    # The likelihood is largest at rho = 1 (an ordinary renewal process);
    # the fit used to return rho = 0 at a log-likelihood of -31.85.
    model = ARA.fit(_X, _I, c=_C, m=m)
    assert model.rho > 0.999
    assert model.log_likelihood == pytest.approx(-30.0999, abs=1e-3)
    assert np.allclose(model.model.params, [13.779, 1.917], atol=1e-3)


def test_grp_finds_the_perfect_repair_maximum():
    model = GeneralizedRenewal.fit(_X, _I, c=_C)
    assert model.q < 1e-6
    assert model.log_likelihood == pytest.approx(-30.0999, abs=1e-3)


def test_multistart_keeps_a_start_that_hit_the_evaluation_cap():
    from types import SimpleNamespace

    capped = SimpleNamespace(success=False, fun=1.0, x=np.array([0.0]))
    converged = SimpleNamespace(success=True, fun=2.0, x=np.array([1.0]))
    results = iter([converged, capped])
    polished = []

    def polish(res):
        polished.append(res)
        return SimpleNamespace(success=False, fun=1.5, x=res.x)

    best = RenewalFitMixin._multistart(
        lambda x0: next(results), [[0.1], [0.9]], None, polish=polish
    )
    # the capped start has the better likelihood and polishing did not
    # improve it, so it is kept
    assert best is capped
    assert polished == [capped]


def test_renewal_rejects_unknown_kijima_type():
    x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])
    with pytest.raises(ValueError, match="Unknown kijima_type"):
        GeneralizedRenewal.fit(x, kijima="iii")


def test_renewal_init_must_have_the_right_length():
    x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])
    with pytest.raises(ValueError, match="init must have 3 values"):
        GeneralizedRenewal.fit(x, init=[1.0, 2.0])
    with pytest.raises(ValueError, match="init must have 3 values"):
        GeneralizedOneRenewal.fit(x, init=[1.0])


def test_g1_rejects_tied_events_clearly():
    x = [1, 3, 3, 5, 2, 4, 6]
    i = [1, 1, 1, 1, 2, 2, 2]
    c = [0, 0, 0, 1, 0, 0, 1]
    with pytest.raises(ValueError, match="tied event times"):
        GeneralizedOneRenewal.fit(x, i, c)
    # the virtual-age and intensity models take tied events
    for fitter in (GeneralizedRenewal, ARA, ARI):
        fitter.fit(x, i, c)
    CrowAMSAA.fit(x, i, c)


def test_renewal_event_at_time_zero_is_a_clear_error():
    x = [0, 3, 5, 2, 4]
    i = [1, 1, 1, 2, 2]
    c = [0, 0, 1, 0, 1]
    for fitter in (GeneralizedRenewal, ARA, GeneralizedOneRenewal):
        with pytest.raises(ValueError, match="event at time 0"):
            fitter.fit(x, i, c)
    with pytest.raises(ValueError, match="event at t = 0"):
        ARI.fit(x, i, c)


def test_renewal_rejects_negative_times():
    with pytest.raises(ValueError, match="cannot be negative"):
        GeneralizedRenewal.fit_from_recurrent_data(
            handle_xicn([-1.0, 2.0, 3.0], [1, 1, 1], [0, 0, 1], tl=-2.0)
        )


_REPAIR_CASES = {
    "G1 q below -1": (GeneralizedOneRenewal, -1.5),
    "G1 q at -1": (GeneralizedOneRenewal, -1.0),
    "GRP q negative": (GeneralizedRenewal, -0.5),
    "GRP q nan": (GeneralizedRenewal, np.nan),
    "ARA rho above 1": (ARA, 1.5),
    "ARA rho negative": (ARA, -0.1),
    "ARI rho negative": (ARI, -0.1),
}


@pytest.mark.parametrize("case", sorted(_REPAIR_CASES))
def test_fit_from_parameters_checks_the_repair_parameter(case):
    fitter, value = _REPAIR_CASES[case]
    with pytest.raises(ValueError, match="must be finite and in"):
        fitter.fit_from_parameters([10, 2], value)


def test_fit_from_parameters_accepts_the_bounds():
    GeneralizedOneRenewal.fit_from_parameters([10, 2], -0.5)
    GeneralizedRenewal.fit_from_parameters([10, 2], 0.0)
    ARA.fit_from_parameters([10, 2], 0.0)
    ARA.fit_from_parameters([10, 2], 1.0)


@pytest.mark.parametrize("fitter", [GeneralizedRenewal, ARA])
def test_lognormal_renewal_fit_is_warning_free(fitter):
    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2.2, 5, 7.5, 9, 12])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = fitter.fit(x, i, dist=LogNormal)
        model.residuals()


def test_reloaded_renewal_model_keeps_how():
    x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])
    fitted = GeneralizedRenewal.fit(x)
    restored = RenewalModel.from_dict(fitted.to_dict())
    assert "Fitted by           : MLE" in repr(restored)
    given = GeneralizedRenewal.fit_from_parameters([10, 2], 0.2)
    restored = RenewalModel.from_dict(given.to_dict())
    assert "given parameters" in repr(restored)


# --- degenerate NHPP data -----------------------------------------------


@pytest.mark.parametrize("model", ["CrowAMSAA", "HPP", "PI-HPP", "PI-NHPP"])
def test_all_censored_data_is_a_clear_error(model):
    x, i, c, Z = [5, 6], [1, 2], [1, 1], [[0], [1]]
    with pytest.raises(ValueError, match="no events"):
        if model == "CrowAMSAA":
            CrowAMSAA.fit(x, i, c)
        elif model == "HPP":
            HPP.fit(x, i, c)
        elif model == "PI-HPP":
            ProportionalIntensityHPP.fit(x, Z, i, c)
        else:
            ProportionalIntensityNHPP.fit(x, Z, i, c, dist=CrowAMSAA)


def test_power_law_event_at_time_zero_is_a_clear_error():
    with pytest.raises(ValueError, match="event at t = 0"):
        CrowAMSAA.fit([0, 3, 5], [1, 1, 1], [0, 0, 1])
    # the constant-rate model is fine there
    assert np.isclose(HPP.fit([0, 3, 5], [1, 1, 1], [0, 0, 1]).params[0], 0.4)


def test_power_law_rejects_times_before_zero():
    x, i, c = [-3, -1, 2, 4], [1] * 4, [0, 0, 0, 1]
    with pytest.raises(ValueError, match="outside it"):
        CrowAMSAA.fit(x, i, c, tl=-5)
    # models defined on the whole line take the negative window
    assert np.isclose(HPP.fit(x, i, c, tl=-5).params[0], 3 / 9)
    CoxLewis.fit(x, i, c, tl=-5)


def test_single_failure_truncated_event_is_a_clear_error():
    with pytest.raises(ValueError, match="single event"):
        CrowAMSAA.fit([5])
    # a window closed after the event identifies the power law
    CrowAMSAA.fit([5, 10], c=[0, 1])
    # and the one-parameter HPP is fine
    assert np.isclose(HPP.fit([5]).params[0], 0.2)


def test_nhpp_rejects_bad_how_and_init():
    with pytest.raises(ValueError, match="how must be"):
        CrowAMSAA.fit([1, 2, 3, 4], how="bad")
    with pytest.raises(ValueError, match="init must have 2 values"):
        CrowAMSAA.fit([1, 2, 3, 4], init=[1])


def test_hpp_init_must_be_a_positive_rate():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="positive"):
            HPP.fit([1, 2, 3, 4], init=[0])
        with pytest.raises(ValueError, match="positive"):
            ProportionalIntensityHPP.fit(
                [1, 2, 3], [[0], [1], [0]], [1, 2, 3], init=[0, 0]
            )


def test_cox_lewis_zero_and_tiny_beta():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flat = CoxLewis.from_params([0.5, 0.0])
        assert np.allclose(flat.cif([1, 2]), np.exp(0.5) * np.array([1, 2]))
        assert np.allclose(
            flat.inv_cif([1, 2]), np.array([1, 2]) / np.exp(0.5)
        )
        tiny = CoxLewis.from_params([0.0, 1e-14])
        assert tiny.cif(1.0) == pytest.approx(1.0, rel=1e-12)
        assert tiny.inv_cif(1.0) == pytest.approx(1.0, rel=1e-12)
    # unchanged away from zero
    model = CoxLewis.from_params([0.2, 0.3])
    assert model.cif(2.0) == pytest.approx(
        np.exp(0.2) / 0.3 * (np.exp(0.6) - 1.0)
    )


# --- left-censored counts with tl ---------------------------------------


def test_hpp_left_censored_count_covers_the_window_from_entry():
    # three events in (5, 10]: rate 3 / 5, not 3 / 10
    model = HPP.fit([10], [1], [-1], n=[3], tl=[5])
    assert np.isclose(model.params[0], 0.6)


def test_nhpp_left_censored_count_uses_cif_from_entry():
    data = handle_xicn([10, 20], [1, 1], [-1, 1], n=[3, 1], tl=5)
    neg_ll = CrowAMSAA.create_negll_func(data)
    alpha, beta = 8.0, 1.3

    def cif(t):
        return (t / alpha) ** beta

    lam = cif(10) - cif(5)
    expected = 3 * np.log(lam) - lam - np.log(6) - (cif(20) - cif(10))
    assert np.isclose(-neg_ll(np.array([alpha, beta])), expected)


def test_proportional_intensity_left_censored_count_uses_entry():
    data = handle_xicn(
        [10, 20], [1, 1], [-1, 1], n=[3, 1], tl=5, Z=[[1.0], [1.0]]
    )
    rate, b = 0.4, 0.2
    phi = np.exp(b)
    lam = rate * phi * 5
    expected = 3 * np.log(lam) - lam - np.log(6) - rate * phi * 10
    neg_ll = ProportionalIntensityHPP.create_negll_func(data)
    assert np.isclose(-neg_ll(np.array([np.log(rate), b])), expected)

    neg_ll = ProportionalIntensityNHPP.create_negll_func(data, CrowAMSAA)
    alpha, beta = 8.0, 1.3

    def cif(t):
        return (t / alpha) ** beta

    lam = phi * (cif(10) - cif(5))
    expected = 3 * np.log(lam) - lam - np.log(6) - phi * (cif(20) - cif(10))
    assert np.isclose(-neg_ll(np.array([alpha, beta, b])), expected)


# --- input validation in handle_xicn ------------------------------------


def test_censoring_row_before_finite_tr_is_rejected():
    kwargs = dict(
        x=[1, 2, 3, 4],
        i=[1, 1, 1, 2],
        c=[0, 0, 1, 1],
        e=["a", "b", None, None],
        tr=[10, 10, 10, 10],
        dist=HPP,
    )
    with pytest.raises(ValueError, match="before its right truncation"):
        CauseSpecificNHPP.fit(**kwargs)
    with pytest.raises(ValueError, match="before its right truncation"):
        NonParametricCounting.fit([1, 2, 3], c=[0, 0, 1], tr=10)
    # a c=1 row at tr is the same close, and is fine
    NonParametricCounting.fit([1, 2, 10], c=[0, 0, 1], tr=10)


def test_tied_censoring_row_and_event_do_not_depend_on_order():
    a = NonParametricCounting.fit([1, 3, 3], [1, 1, 1], [0, 1, 0])
    b = NonParametricCounting.fit([1, 3, 3], [1, 1, 1], [0, 0, 1])
    assert np.array_equal(a.mcf_hat, b.mcf_hat)
    fit_a = CrowAMSAA.fit([1, 3, 3, 2, 4], [1, 1, 1, 2, 2], [0, 1, 0, 0, 1])
    fit_b = CrowAMSAA.fit([1, 3, 3, 2, 4], [1, 1, 1, 2, 2], [0, 0, 1, 0, 1])
    assert np.allclose(fit_a.params, fit_b.params)


def test_missing_or_mixed_item_ids_are_clear_errors():
    with pytest.raises(ValueError, match="must not be missing"):
        handle_xicn([1, 2], [None, None])
    with pytest.raises(ValueError, match="must not be missing"):
        handle_xicn([1, 2], [1, None])
    mixed = pd.Series([1, "a"], dtype=object).to_numpy()
    with pytest.raises(ValueError, match="one comparable kind"):
        handle_xicn([1, 2], mixed)


def test_covariates_must_be_constant_within_an_item():
    with pytest.raises(ValueError, match="change between its rows"):
        ProportionalIntensityHPP.fit([1, 2, 3], [[0], [1], [1]], [1, 1, 2])
    # the same values on every row of an item are fine
    ProportionalIntensityHPP.fit([1, 2, 3], [[0], [0], [1]], [1, 1, 2])


def test_to_xrd_matches_a_direct_count():
    rng = np.random.default_rng(3)
    x, i, c, tl, tr = [], [], [], [], []
    for item in range(12):
        entry = float(rng.choice([0.0, 1.0, 2.5]))
        close = float(rng.uniform(6, 12))
        times = np.sort(np.round(rng.uniform(entry, close, 5), 1))
        closed_by_row = item % 2 == 0
        x += times.tolist() + ([close] if closed_by_row else [])
        c += [0] * 5 + ([1] if closed_by_row else [])
        rows = 6 if closed_by_row else 5
        i += [item] * rows
        tl += [entry] * rows
        tr += [np.inf if closed_by_row else close] * rows
    data = handle_xicn(x, i, c, tl=tl, tr=tr)
    grid, r, d = data.to_xrd()
    entry, exit_ = data.item_observation_windows()
    events = data.c == 0
    for k, t in enumerate(grid):
        assert d[k] == data.n[(data.x == t) & events].sum()
        assert r[k] == ((entry <= t) & (t <= exit_)).sum()


# --- nonparametric MCF --------------------------------------------------


def test_linear_mcf_bounds_rise_from_zero_like_the_mcf():
    model = NonParametricCounting.fit(
        [2, 4, 6, 3, 5], [1, 1, 1, 2, 2], [0, 0, 1, 0, 1]
    )
    query = np.array([0.0, 1.0, 2.0])
    mcf = model.mcf(query, interp="linear")
    cb = model.mcf_cb(query, interp="linear")
    assert np.allclose(cb[0], [0.0, 0.0])
    assert np.all(cb[:, 0] <= mcf) and np.all(mcf <= cb[:, 1])
    upper = model.mcf_cb(query, interp="linear", bound="upper")
    assert upper[0] == 0 and np.all(upper >= mcf)
    # before the origin both are undefined
    assert np.isnan(model.mcf(-1.0, interp="linear")).all()
    assert np.isnan(model.mcf_cb(-1.0, interp="linear")).all()


def test_mcf_is_defined_at_negative_times_inside_a_negative_tl():
    model = NonParametricCounting.fit(
        [-3, -1, 2, 4], [1, 1, 1, 1], [0, 0, 0, 1], tl=-5
    )
    assert np.allclose(model.mcf([-4, -3, 0]), [0, 1, 2])
    assert np.isnan(model.mcf(-6)).all()
    assert np.allclose(model.mcf_cb([-4])[0], [0, 0])
    restored = NonParametricCounting.from_dict(model.to_dict())
    assert np.allclose(restored.mcf([-4, -3, 0]), [0, 1, 2])


def test_simulated_mcf_round_trip_has_no_variance():
    sim = CrowAMSAA.from_params([10, 2]).time_terminated_simulation(
        20, items=50, seed=1
    )
    as_dict = sim.to_dict()
    assert as_dict["var"] is None
    restored = NonParametricCounting.from_dict(as_dict)
    with pytest.raises(ValueError, match="no variance"):
        restored.mcf_cb([5])


def test_old_dict_with_nan_variance_reads_as_no_variance():
    old = NonParametricCounting.fit([1, 2, 3]).to_dict()
    old["var"] = float("nan")
    old.pop("origin")
    restored = NonParametricCounting.from_dict(old)
    assert restored.var is None
    assert restored.origin == 0.0


def test_recurrent_event_data_to_xrd_keeps_integer_counts():
    data = RecurrentEventData(
        np.array([1, 2, 1]), np.array([1, 1, 2]), np.zeros(3), np.ones(3, int)
    )
    _, r, d = data.to_xrd()
    assert d.dtype.kind == "i" and r.dtype.kind == "i"
