"""Fitting, simulating and checking the renewal / imperfect-repair models
(generalized renewal, G1, ARA, ARI).
"""

import warnings

import numpy as np
import pytest

from surpyval import LogNormal, handle_xicn
from surpyval.recurrent import (
    ARA,
    ARI,
    CrowAMSAA,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
    RenewalModel,
)
from surpyval.recurrent.renewal.fit_mixin import RenewalFitMixin
from surpyval.tests._helpers import (
    REPAIR_FLEET_C,
    REPAIR_FLEET_I,
    REPAIR_FLEET_X,
)

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
        mcf = model.mcf([20, 40], items=200, random_state=1)
    assert np.allclose(mcf, [8, 64], rtol=0.1)


def test_grp_partial_repair_matches_exact_sampling():
    t = np.array([40.0, 70.0])
    truth = _kijima_i_weibull_mcf(10.0, 3.0, 0.5, t)
    model = GeneralizedRenewal.fit_from_parameters([10, 3], 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mcf = model.mcf(t, items=200, random_state=2)
    # the old sampler gave 77.5 at 70 (exact about 89.6), and stayed
    # there: its curve was flat from 70 on
    assert np.allclose(mcf, truth, rtol=0.05)


def test_ara_simulated_mcf_is_right_at_long_horizons():
    model = ARA.fit_from_parameters([10, 3], 0.0, m=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mcf = model.mcf([20, 40], items=200, random_state=1)
    assert np.allclose(mcf, [8, 64], rtol=0.1)


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
    with pytest.raises(ValueError, match="'kijima_type' must be one of"):
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
    # Without a tl, times count from the start of the item's life.
    from surpyval import RecurrentEventData

    with pytest.raises(ValueError, match="cannot be negative"):
        GeneralizedRenewal.fit_from_recurrent_data(
            RecurrentEventData([-1.0, 2.0, 3.0], [1, 1, 1], [0, 0, 1], 1)
        )
    # With one they count from the entry, as new there (#615): a negative
    # time after a negative entry is a positive age (with a warning that a
    # negative entry age is probably a data error, #664).
    with pytest.warns(UserWarning, match="negative age"):
        entered = GeneralizedRenewal.fit_from_recurrent_data(
            handle_xicn([-1.0, 2.0, 3.0, 5.0], [1] * 4, [0, 0, 0, 1], tl=-2.0)
        )
    assert np.allclose(entered.data.x, [1.0, 4.0, 5.0, 7.0])


@pytest.mark.parametrize(
    "fitter", [GeneralizedRenewal, GeneralizedOneRenewal, ARA, ARI]
)
def test_664_a_negative_entry_age_warns_and_is_kept(fitter):
    x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60.0]
    i = ["a"] * 6 + ["b"] * 5
    c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    tl = [-50.0] * 6 + [0.0] * 5
    kw = {"baseline": CrowAMSAA, "m": 1} if fitter is ARI else {}
    with pytest.warns(UserWarning, match=r"1 item\(s\).*a \(tl=-50.0\)"):
        model = fitter.fit(x, i, c, tl=tl, **kw)
    # still fitted, item a as new at -50 (its times moved on by 50)
    assert np.allclose(model.data.x[:6], np.array(x[:6]) + 50.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitter.fit(x, i, c, tl=[0.0] * 11, **kw)


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


# ---------------------------------------------------------------------------
# Warning-free G1 fits.
# ---------------------------------------------------------------------------


def test_g1_weibull_fit_emits_no_warnings():
    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = GeneralizedOneRenewal.fit(x)
        GeneralizedOneRenewal.fit(
            REPAIR_FLEET_X, REPAIR_FLEET_I, REPAIR_FLEET_C
        )
    assert model.q == pytest.approx(0.2163, abs=1e-3)


@pytest.mark.parametrize(
    "fitter, name, init",
    [(GeneralizedRenewal, "q", [0.0, 0.2]), (ARA, "rho", [1.0, 0.2])],
)
def test_665_a_memoryless_life_warns_that_restoration_is_unestimable(
    fitter, name, init
):
    # With an Exponential life every value of the restoration parameter
    # has the HPP's likelihood: q = 17 was reported, in silence, and a
    # start on the edge of its range (q = 0) failed to converge.
    from surpyval import Exponential

    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
    c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
    for kwargs in ({}, {"init": init}):
        with pytest.warns(UserWarning, match=name + " cannot be estimated"):
            model = fitter.fit(x, i, c, dist=Exponential, **kwargs)
        # The HPP's rate: failures over the total time observed
        assert model.model.params[0] == pytest.approx(10 / 23, rel=1e-6)


def _kijima_i_sample(seed, q=0.4, alpha=100.0, beta=2.5, units=6, T=400.0):
    """Kijima-I failures of a Weibull life, ``units`` items to ``T``."""
    rng = np.random.default_rng(seed)
    x, i, c = [], [], []
    for unit in range(units):
        t = v = 0.0
        while True:
            # The gap from virtual age v: H(v + y) - H(v) = -log(u)
            h = (v / alpha) ** beta - np.log(rng.uniform())
            y = alpha * h ** (1 / beta) - v
            t += y
            if t > T:
                break
            x.append(t)
            i.append(unit)
            c.append(0)
            v += q * y
        x.append(T)
        i.append(unit)
        c.append(1)
    return np.array(x), np.array(i), np.array(c)


def test_630_kijima_ii_likelihood_keeps_the_gaps_of_aged_items():
    # Kijima-II on Kijima-I data: at q = 611 the virtual ages reach 1e30
    # within a dozen failures, v + x rounded to v, every survival drop
    # read 0 and the likelihood appeared to rise without bound (-172
    # there against -284 at the maximum, q = 0.95); larger samples ended
    # "unverified" at q = 4. The drops are integrated from the hazard
    # there now, and the likelihood falls away from its maximum.
    x, i, c = _kijima_i_sample(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = GeneralizedRenewal.fit(x, i, c, kijima="ii")
    assert model.maximum == "verified"
    assert model.q == pytest.approx(0.954, abs=1e-3)
    far = -model._neg_ll(np.array([611.0, 88.5, 1.04]))
    assert far < model.log_likelihood - 5
    # One aged gap against its exact drop, -(H(v + x) - H(v))
    from surpyval import Weibull
    from surpyval.recurrent.renewal.generalized_renewal import (
        virtual_age_log_likelihood,
    )

    v, gap = np.array([1e18]), np.array([11.0])
    exact = (-1.5 * gap * (v / 88.0) ** 0.5 / 88.0)[0]
    log_h = np.log(Weibull.hf(v + gap, 88.0, 1.5))[0]
    for c, density in ((1, 0.0), (0, log_h)):
        ll = virtual_age_log_likelihood(
            Weibull, (88.0, 1.5), v, gap, np.array([c]), np.array([False])
        )
        assert ll == pytest.approx(exact + density, rel=1e-12)


def _fleet_data(seed=0):
    rng = np.random.default_rng(seed)
    x, i, c = [], [], []
    for k in range(12):
        t = np.cumsum(rng.weibull(2.0, rng.integers(1, 9)) * 10)
        x += list(t) + [t[-1] + 2.0]
        i += [k] * (t.size + 1)
        c += [0] * t.size + [1]
    return handle_xicn(np.array(x), np.array(i), np.array(c))


@pytest.mark.parametrize(
    "fitter, kw, params",
    [
        (GeneralizedRenewal, {"kijima": "i"}, [0.4, 10.0, 0.7]),
        (GeneralizedRenewal, {"kijima": "ii"}, [0.4, 10.0, 2.0]),
        # q > 1, and q = 40: virtual ages that dwarf the gaps (Simpson)
        (GeneralizedRenewal, {"kijima": "ii"}, [1.7, 10.0, 2.0]),
        (GeneralizedRenewal, {"kijima": "ii"}, [40.0, 10.0, 2.0]),
        (ARA, {"m": 1}, [0.4, 10.0, 0.7]),
        (ARA, {"m": 3}, [0.4, 10.0, 2.0]),
        (ARA, {"m": np.inf}, [0.4, 10.0, 2.0]),
        (ARI, {"m": 1}, [0.3, 10.0, 1.5]),
        (ARI, {"m": np.inf}, [0.3, 10.0, 1.5]),
        (GeneralizedOneRenewal, {}, [0.3, 10.0, 1.5]),
    ],
)
def test_710_renewal_likelihoods_have_exact_derivatives(fitter, kw, params):
    # The renewal fits search on the likelihood's gradient (#710): the
    # Kijima-II, ARA and ARI recursions wrote arrays in place, which
    # autograd cannot follow. Their derivatives are exact now, to the
    # second order (the Hessian), even at a Weibull beta < 1, whose log
    # S(v) has an infinite derivative at an item's first age of 0.
    from autograd import grad, hessian

    from surpyval import Weibull
    from surpyval.utils.linalg import numerical_gradient, numerical_hessian

    data = _fleet_data()
    if fitter is ARI:
        neg_ll = fitter.create_negll_func(data, CrowAMSAA, **kw)
    elif fitter is GeneralizedOneRenewal:
        neg_ll = fitter.create_negll_func(
            data.interarrival_times, data.i, data.c, data.n, Weibull
        )
    else:
        neg_ll = fitter.create_negll_func(data, Weibull, **kw)
    p = np.array(params)

    def plain(u):
        return float(neg_ll(u))

    step = np.maximum(np.abs(p), 1.0)
    np.testing.assert_allclose(
        grad(neg_ll)(p),
        numerical_gradient(plain, p, 1e-6 * step),
        rtol=1e-6,
        atol=1e-6 * abs(plain(p)),
    )
    np.testing.assert_allclose(
        hessian(neg_ll)(p),
        numerical_hessian(plain, p, 1e-4 * step),
        rtol=1e-4,
        atol=1e-4 * abs(plain(p)),
    )


def test_710_polish_takes_the_exact_gradient():
    # A fit that is not a verified maximum is polished by BFGS on the
    # likelihood's exact gradient (#710), and on central differences
    # where autograd cannot differentiate it (a Cox-Lewis baseline is
    # written in plain numpy).
    from surpyval import Weibull
    from surpyval.recurrent import CoxLewis
    from surpyval.univariate.parametric.fitters import verified_maximum

    x, i, c = _kijima_i_sample(0)
    model = GeneralizedRenewal.fit(x, i, c, kijima="ii")
    neg_ll = GeneralizedRenewal.create_negll_func(
        model.data, Weibull, kijima="ii"
    )
    bounds = [(0, None), *Weibull.bounds]
    off = model._mle * np.array([1.0, 1.01, 0.99])
    assert not verified_maximum(neg_ll, off, bounds, model._n_obs)
    polished = RenewalFitMixin._polish_unverified(
        neg_ll, off, bounds, model._n_obs
    )
    assert verified_maximum(neg_ll, polished, bounds, model._n_obs)
    assert -neg_ll(polished) > model.log_likelihood - 1e-9

    x = np.array([3, 9, 20, 35, 56, 4, 11, 25, 44, 70.0])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    model = ARI.fit(x, i, baseline=CoxLewis)
    neg_ll = ARI.create_negll_func(model.data, CoxLewis, 1)
    bounds = [(0, 1), *CoxLewis.bounds]
    off = model._mle + np.array([0.0, 0.05, 0.001])
    polished = RenewalFitMixin._polish_unverified(
        neg_ll, off, bounds, model._n_obs
    )
    assert neg_ll(polished) < neg_ll(off)


def _728_neg_ll(fitter, life, kw, data):
    if fitter is ARI:
        return fitter.create_negll_func(data, life, **kw)
    if fitter is GeneralizedOneRenewal:
        return fitter.create_negll_func(
            data.interarrival_times, data.i, data.c, data.n, life
        )
    return fitter.create_negll_func(data, life, **kw)


@pytest.mark.parametrize("one_item", [False, True])
@pytest.mark.parametrize(
    "fitter, life, kw, params",
    [
        (GeneralizedRenewal, "Weibull", {"kijima": "i"}, [0.4, 10.0, 0.7]),
        (GeneralizedRenewal, "Weibull", {"kijima": "ii"}, [1.7, 10.0, 2.0]),
        # Virtual ages that dwarf the gaps (Simpson's rule)
        (GeneralizedRenewal, "Weibull", {"kijima": "ii"}, [40.0, 10.0, 0.6]),
        (GeneralizedRenewal, "LogNormal", {"kijima": "i"}, [0.7, 2.0, 0.5]),
        (GeneralizedRenewal, "LogNormal", {"kijima": "ii"}, [30.0, 2.0, 0.5]),
        (ARA, "Weibull", {"m": 1}, [0.4, 10.0, 0.7]),
        (ARA, "Weibull", {"m": 3}, [0.4, 10.0, 2.0]),
        (ARA, "Weibull", {"m": np.inf}, [1 - 1e-9, 10.0, 2.0]),
        (ARA, "LogNormal", {"m": 2}, [0.4, 2.0, 0.6]),
        (ARI, "CrowAMSAA", {"m": 1}, [0.3, 10.0, 1.5]),
        (ARI, "CrowAMSAA", {"m": np.inf}, [0.3, 10.0, 1.5]),
        (ARI, "Duane", {"m": 2}, [0.3, 1.5, 0.05]),
        (GeneralizedOneRenewal, "Weibull", {}, [-0.3, 10.0, 0.8]),
        (GeneralizedOneRenewal, "LogNormal", {}, [0.3, 2.0, 0.8]),
    ],
)
def test_728_hand_written_gradient_is_autograds(
    fitter, life, kw, params, one_item
):
    # The renewal fits search on a gradient written by hand in plain numpy
    # (#728): autograd's cost 8 to 30 likelihoods on these models' small
    # data. It is autograd's gradient, and the value the likelihood's.
    from autograd import grad

    import surpyval
    from surpyval import recurrent

    life = getattr(surpyval, life, None) or getattr(recurrent, life)
    if one_item:
        rng = np.random.default_rng(5)
        data = handle_xicn(np.cumsum(rng.weibull(2.0, 30) * 3))
    else:
        data = _fleet_data()
    neg_ll = _728_neg_ll(fitter, life, kw, data)
    p = np.array(params)
    value, gradient = neg_ll.value_and_grad(p)
    assert value == pytest.approx(neg_ll(p), rel=1e-13)
    expected = grad(neg_ll)(p)
    np.testing.assert_allclose(
        gradient, expected, rtol=1e-10, atol=1e-13 * np.abs(expected).max()
    )


def test_728_search_map_is_the_fits_map():
    # The gradient search's own map to the natural parameters, with its
    # slopes, is bounds_convert's (the one the starts and the answer are
    # mapped with).
    from surpyval.recurrent.renewal._search import SearchMap
    from surpyval.univariate.parametric.fitters import bounds_convert

    bounds = [(0, None), (0, 1), (None, None), (-1, None), (None, 3), (2, 5)]
    _, inv_trans, *_ = bounds_convert(
        None, bounds, {}, {str(k): k for k in range(len(bounds))}
    )
    to_natural = SearchMap(bounds)
    for u in np.random.default_rng(1).normal(0, 3, (20, len(bounds))):
        natural, slopes = to_natural(u)
        np.testing.assert_allclose(natural, inv_trans(u), rtol=1e-15)
        step = 1e-6
        numeric = (to_natural(u + step)[0] - to_natural(u - step)[0]) / (
            2 * step
        )
        np.testing.assert_allclose(slopes, numeric, rtol=1e-7)


def test_728_gradient_search_where_written_and_bounds_reached():
    # BFGS on the hand-written gradient for a Weibull life; Nelder-Mead,
    # as before, where there is none (a GumbelLEV life; #746 gave the
    # Gamma one, #760 the ExpoWeibull and Normal ones). The search still
    # carries a restoration parameter onto its bound: perfect repair.
    from surpyval import ExpoWeibull, Gamma, GumbelLEV, Normal, Weibull
    from surpyval.recurrent.renewal._search import (
        GradientSearch,
        SimplexSearch,
        renewal_search,
    )

    data = _fleet_data()
    for life in (Weibull, Gamma, ExpoWeibull, Normal):
        neg_ll = GeneralizedRenewal.create_negll_func(data, life)
        bounds = [(0, None), *life.bounds]
        assert isinstance(
            renewal_search(neg_ll, bounds, 1.0, None), GradientSearch
        )
    lev = GeneralizedRenewal.create_negll_func(data, GumbelLEV)
    bounds = [(0, None), *GumbelLEV.bounds]
    assert isinstance(renewal_search(lev, bounds, 1.0, None), SimplexSearch)

    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
    c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kijima = GeneralizedRenewal.fit(x, i, c)
        ara = ARA.fit(x, i, c, m=2)
    assert kijima.maximum == ara.maximum == "verified"
    assert kijima.q < 1e-12 and ara.rho > 1 - 1e-12
    # The perfect-repair maximum: a Weibull renewal process
    assert kijima.log_likelihood == pytest.approx(-12.5054426749, abs=1e-8)
    assert ara.log_likelihood == pytest.approx(-12.5054426749, abs=1e-8)


def test_728_kijima_finds_the_run_off():
    # On these data the Kijima-I likelihood rises without bound as q -> inf
    # (-29.37 at q = 2.8, -29.04 at 1e6, -28.93 at 1e300). The Nelder-Mead
    # search stopped at the local maximum q = 2.8 and called it verified;
    # the gradient search follows the run-off, and says the fit is not a
    # verified maximum.
    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2, 20, 20, 20])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 1, 2, 3])
    c = np.r_[np.zeros(12), 1, 1, 1]
    with pytest.warns(UserWarning, match="verified maximum"):
        model = GeneralizedRenewal.fit(x, i, c)
    assert model.maximum == "unverified"
    assert model.q > 1e6
    assert model.log_likelihood > -29.1


def _richardson_gradient(f, p, rel=1e-4):
    # Fourth-order central differences of f at p
    g = np.zeros(p.size)
    for k in range(p.size):
        e = np.zeros(p.size)
        e[k] = rel * max(abs(p[k]), 1e-2)
        g[k] = (8 * (f(p + e) - f(p - e)) - (f(p + 2 * e) - f(p - 2 * e))) / (
            12 * e[k]
        )
    return g


@pytest.mark.parametrize("one_item", [False, True])
@pytest.mark.parametrize(
    "fitter, life, kw, params",
    [
        (GeneralizedRenewal, "Gamma", {"kijima": "i"}, [0.4, 2.5, 0.25]),
        (GeneralizedRenewal, "Gamma", {"kijima": "i"}, [0.4, 0.6, 0.05]),
        (GeneralizedRenewal, "Gamma", {"kijima": "ii"}, [1.2, 2.5, 0.25]),
        (ARA, "Gamma", {"m": 1}, [0.4, 2.5, 0.25]),
        (ARA, "Gamma", {"m": np.inf}, [0.7, 0.8, 0.1]),
        (GeneralizedOneRenewal, "Gamma", {}, [-0.3, 2.5, 0.25]),
        (ARI, "CoxLewis", {"m": 1}, [0.3, -1.0, 0.05]),
        (ARI, "CoxLewis", {"m": 2}, [0.3, -2.0, 0.0]),
        (ARI, "CoxLewis", {"m": np.inf}, [0.2, -1.5, 0.02]),
        (GeneralizedRenewal, "LogLogistic", {"kijima": "ii"}, [1.7, 10, 0.7]),
        (ARA, "LogLogistic", {"m": 2}, [0.4, 10.0, 2.5]),
        (GeneralizedOneRenewal, "LogLogistic", {}, [-0.3, 10.0, 2.5]),
        (GeneralizedRenewal, "Rayleigh", {"kijima": "ii"}, [40.0, 8.0]),
        (GeneralizedOneRenewal, "Exponential", {}, [0.3, 0.1]),
        (ARI, "HPP", {"m": np.inf}, [0.3, 0.2]),
        # and the ExpoWeibull, Normal, Gumbel and Logistic (#760)
        (
            GeneralizedRenewal,
            "ExpoWeibull",
            {"kijima": "i"},
            [0.4, 10, 2, 1.3],
        ),
        (ARA, "ExpoWeibull", {"m": 2}, [0.4, 10.0, 0.7, 3.0]),
        (GeneralizedOneRenewal, "ExpoWeibull", {}, [0.1, 10.0, 2.0, 1.3]),
        (GeneralizedRenewal, "Normal", {"kijima": "i"}, [0.4, 9.0, 5.0]),
        (ARA, "Normal", {"m": np.inf}, [0.7, 9.0, 5.0]),
        (GeneralizedRenewal, "Gumbel", {"kijima": "i"}, [0.4, 9.0, 7.0]),
        (ARA, "Gumbel", {"m": 1}, [0.3, 9.0, 7.0]),
        (GeneralizedRenewal, "Logistic", {"kijima": "ii"}, [1.7, 9.0, 3.0]),
        (ARA, "Logistic", {"m": 2}, [0.4, 9.0, 3.0]),
    ],
)
def test_746_hand_written_gradients(fitter, life, kw, params, one_item):
    # The Gamma, LogLogistic, Rayleigh and Exponential lifetimes and the
    # Cox-Lewis and HPP baselines have hand-written gradients too (#746):
    # the likelihood's value, and the derivatives of its central
    # differences (autograd's of the Cox-Lewis likelihood fails, and its
    # Gamma shape derivative is itself a difference).
    import surpyval
    from surpyval import recurrent

    life = getattr(surpyval, life, None) or getattr(recurrent, life)
    if one_item:
        rng = np.random.default_rng(5)
        data = handle_xicn(np.cumsum(rng.weibull(2.0, 30) * 3))
    else:
        data = _fleet_data()
    neg_ll = _728_neg_ll(fitter, life, kw, data)
    p = np.array(params)
    value, gradient = neg_ll.value_and_grad(p)
    assert value == pytest.approx(neg_ll(p), rel=1e-13)
    expected = _richardson_gradient(neg_ll, p)
    np.testing.assert_allclose(
        gradient, expected, rtol=1e-8, atol=1e-9 * np.abs(expected).max()
    )


@pytest.mark.parametrize(
    "alpha, beta, t, d_log_sf, d_hf_alpha, d_hf_beta",
    [
        # S near 1, the body of the differences, their edge, and far into
        # the asymptotic series (mpmath, 50 digits)
        (
            0.3,
            2.0,
            1e-4,
            0.790993230187813,
            -1650.02396635922,
            46.6417442345232,
        ),
        (
            2.5,
            0.25,
            0.01,
            6.65947660112928e-7,
            -1.56983561371224e-4,
            2.34257581625340e-4,
        ),
        (
            2.5,
            0.25,
            10.0,
            0.627211804085023,
            -0.0607562894483123,
            0.861149917949485,
        ),
        (
            40.0,
            1.0,
            35.0,
            0.0615249734365236,
            -0.0119628381471453,
            0.478121390927338,
        ),
        (
            0.7,
            1.0,
            120.0,
            6.01576020053809,
            -0.00826534775089870,
            1.00002006735564,
        ),
        (2.5, 0.25, 4e8, 17.7175241133071, -2.49999997500000e-9, 1.0),
    ],
)
def test_746_gamma_shape_derivatives(
    alpha, beta, t, d_log_sf, d_hf_alpha, d_hf_beta
):
    # The Gamma's log S has no closed-form derivative in its shape: it is
    # a difference of log(-log S) in the body and the asymptotic series in
    # the tail (#746), each within 1e-11 of mpmath's.
    from surpyval.recurrent.renewal._derivatives import GammaDerivatives

    terms = GammaDerivatives()
    t = np.array([t])
    _, _, (sf_alpha, _) = terms.log_sf(t, (alpha, beta))
    _, _, (hf_alpha, hf_beta) = terms.hf(t, (alpha, beta))
    assert sf_alpha[0] == pytest.approx(d_log_sf, rel=1e-10)
    assert hf_alpha[0] == pytest.approx(d_hf_alpha, rel=1e-10)
    assert hf_beta[0] == pytest.approx(d_hf_beta, rel=1e-10)


def test_746_gamma_kijima_ii_deep_tail_gradient():
    # Kijima-II with q = 40 ages the item past 1e40 within a few dozen
    # failures, where Gamma.hf (f / S, each of size e**-y) has lost its
    # excess over beta. The hand-written likelihood takes the hazard there
    # from the upper incomplete gamma's asymptotic series: its value and
    # gradient are mpmath's (with the same Simpson's rule over the aged
    # gaps) to 1e-13.
    from surpyval import Gamma

    rng = np.random.default_rng(5)
    data = handle_xicn(np.cumsum(rng.weibull(2.0, 30) * 3))
    neg_ll = GeneralizedRenewal.create_negll_func(data, Gamma, "ii")
    value, gradient = neg_ll.value_and_grad(np.array([40.0, 3.0, 2.0]))
    assert value == pytest.approx(116.7551230645448, rel=1e-12)
    np.testing.assert_allclose(
        gradient, [6.51809079e-4, -1.22491737, 54.5728753], rtol=1e-8
    )


def test_746_cox_lewis_beta_in_its_own_units():
    # A Cox-Lewis beta is a rate per unit time, 8e-5 on data in thousands
    # of hours. BFGS stepped it in units of 1, overflowed e**(beta t) and
    # stopped where it started; and the check of the maximum differenced
    # it in steps of 1e-7, where its slope was read to 0.3 and a maximum
    # was verified or not by luck. Both now take its unit as one over the
    # longest time (#746).
    from surpyval.recurrent import CoxLewis
    from surpyval.recurrent.renewal._search import GradientSearch

    rng = np.random.default_rng(8)
    x, i, c = [], [], []
    for k in range(8):
        end = rng.uniform(6000, 12000)
        n = rng.poisson(2e-4 * end**1.35)
        times = np.sort(end * rng.random(n) ** (1 / 1.35))
        x += [*times, end]
        i += [k] * (n + 1)
        c += [0] * n + [1]
    data = handle_xicn(np.array(x), np.array(i), np.array(c))
    neg_ll = ARI.create_negll_func(data, CoxLewis, 1)
    search = GradientSearch(neg_ll, [(0, 1), (None, None), (None, None)], 1)
    found = search.bfgs(np.array([-2.2, -5.8, 8e-5]))
    assert found.success and found.nit > 5
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ARI.fit(x, i, c, baseline=CoxLewis, m=1)
    assert model.maximum == "verified"
    assert model.rho < 1e-10
    assert model.log_likelihood == pytest.approx(-2329.17357703663, abs=1e-6)


@pytest.mark.parametrize(
    "kind, point, expected",
    [
        # (value, d/dt, d/dalpha, d/dbeta, d/dmu), mpmath to 500 digits.
        # The hazard where u = (t / alpha)**beta is 36 and 900: its log
        # is log f - log S, whose terms in u (and in mu) cancel there.
        (
            "hf",
            (60.0, 10.0, 2.0, 3.0),
            [
                1.2,
                0.0200000000000003,
                -0.240000000000002,
                2.75011136307368,
                -1.4e-16,
            ],
        ),
        ("hf", (300.0, 10.0, 2.0, 0.3), [6.0, 0.02, -1.2, 23.40718428997, 0]),
        (
            "log_sf",
            (300.0, 10.0, 2.0, 0.3),
            [-901.203972804326, -6.0, 180.0, -3061.07764349594, 10 / 3],
        ),
        (
            "log_sf",
            (1e5, 10.0, 0.7, 0.4),
            [
                -631.873635212067,
                -0.00441670141136135,
                44.1670141136135,
                -5811.33190286081,
                2.5,
            ],
        ),
        (
            "log_df",
            (1e5, 10.0, 0.7, 0.4),
            [
                -637.295997360593,
                -0.00441970141136135,
                44.0970141136135,
                -5800.69299106026,
                2.5,
            ],
        ),
    ],
)
def test_760_expo_weibull_derivatives_in_the_tail(kind, point, expected):
    # The ExpoWeibull's hand-written derivatives (#760) against mpmath's,
    # far into the right tail where the hazard's own terms cancel.
    from surpyval.recurrent.renewal._derivatives import (
        ExpoWeibullDerivatives,
    )

    t, *params = point
    terms = ExpoWeibullDerivatives()
    tt = np.array([t])
    if kind == "hf":
        value, d_t, d_p = terms.hf(tt, params)
    elif kind == "log_sf":
        value, d_t, d_p = terms.log_sf(tt, params)
    else:
        value, d_t, d_p = terms.log_end(tt, params, np.array([True]))
    got = [value[0], d_t[0], *(d[0] for d in d_p)]
    # d/dmu of the hazard is below 1e-15 of it: absolute
    np.testing.assert_allclose(got[:4], expected[:4], rtol=1e-12)
    assert got[4] == pytest.approx(expected[4], rel=1e-12, abs=1e-15)
