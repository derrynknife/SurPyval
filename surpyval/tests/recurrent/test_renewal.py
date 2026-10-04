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
    # time after a negative entry is a positive age.
    entered = GeneralizedRenewal.fit_from_recurrent_data(
        handle_xicn([-1.0, 2.0, 3.0, 5.0], [1] * 4, [0, 0, 0, 1], tl=-2.0)
    )
    assert np.allclose(entered.data.x, [1.0, 4.0, 5.0, 7.0])


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
