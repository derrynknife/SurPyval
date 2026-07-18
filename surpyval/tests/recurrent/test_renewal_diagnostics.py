"""Residual and trend-test diagnostics for the renewal / virtual-age
imperfect-repair models (GeneralizedRenewal, GeneralizedOneRenewal, ARA, ARI).

Unlike the counting-process models, these have no marginal cumulative
intensity; the time-rescaling residuals come from each process's *conditional*
intensity -- the cumulative hazard accumulated over each interarrival given the
model's virtual age (Kijima / ARA), time scaling (G1R) or intensity reduction
(ARI). The tests check the residual shapes and identities, that the residuals
are genuinely iid Exp(1) under a well-specified simulate-and-refit, and that
the trend test delegates to the standalone statistic.
"""

import numpy as np
import pytest

from surpyval import Weibull
from surpyval.recurrent import (
    ARA,
    ARI,
    CrowAMSAA,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
)
from surpyval.recurrent.diagnostics import (
    GoodnessOfFitResult,
    _renewal_conditional_uniforms,
    cvm_statistic,
)


def _fit_each():
    # One fitted model per family/configuration, on shared multi-item data.
    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2.2, 5, 7.5, 9, 12])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])
    return {
        "GR-i": GeneralizedRenewal.fit(x, i, dist=Weibull, kijima="i"),
        "GR-ii": GeneralizedRenewal.fit(x, i, dist=Weibull, kijima="ii"),
        "G1R": GeneralizedOneRenewal.fit(x, i, dist=Weibull),
        "ARA-m2": ARA.fit(x, i, dist=Weibull, m=2),
        "ARI": ARI.fit(x, i, dist=CrowAMSAA, m=1),
        "_x": x,
        "_i": i,
    }


MODELS = _fit_each()
MODEL_KEYS = ["GR-i", "GR-ii", "G1R", "ARA-m2", "ARI"]


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_residual_shapes(key):
    model = MODELS[key]
    x, i = MODELS["_x"], MODELS["_i"]
    # one cumulative-hazard residual per observed event (all observed here)
    assert model.residuals("cumulative_hazard").size == x.size
    # one martingale residual per item
    assert model.residuals("martingale").size == np.unique(i).size


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_pit_is_transform_of_cumulative_hazard(key):
    model = MODELS[key]
    e = model.residuals("cumulative_hazard")
    pit = model.residuals("pit")
    assert np.allclose(pit, 1.0 - np.exp(-e))
    assert np.all((pit >= 0) & (pit <= 1))


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_cumulative_hazard_residuals_are_non_negative(key):
    # A cumulative-hazard increment is non-negative by construction.
    assert np.all(MODELS[key].residuals("cumulative_hazard") >= -1e-9)


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_martingale_residuals_near_zero_at_mle(key):
    # observed count minus compensator, summed, is ~0 at the fitted params.
    assert abs(MODELS[key].residuals("martingale").sum()) < 0.5


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_trend_test_returns_result(key):
    result = MODELS[key].trend_test(test="laplace")
    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")


def test_bad_kind_raises():
    with pytest.raises(ValueError, match="cumulative_hazard"):
        MODELS["GR-i"].residuals(kind="nonsense")


def test_fit_from_parameters_has_no_residuals():
    # A model built from parameters carries no data, so residuals must refuse.
    model = GeneralizedRenewal.fit_from_parameters(
        [10.0, 2.0], 0.3, kijima="i", dist=Weibull
    )
    with pytest.raises(ValueError, match="fitted from data"):
        model.residuals()
    with pytest.raises(ValueError, match="fitted from data"):
        model.trend_test()


# --- calibration: residuals are Exp(1) under a well-specified refit ---------


def _simulate_refit_residuals(truth, fitter, refit_kwargs, seed):
    data = truth.count_terminated_simulation_data(8, items=250, seed=seed)
    model = fitter.fit_from_recurrent_data(data, **refit_kwargs)
    return model.residuals("cumulative_hazard"), model


def test_age_reduction_residuals_are_exp1():
    # Kijima-I (age reduction): rescaled increments H(v+x) - H(v) are Exp(1).
    truth = GeneralizedRenewal.fit_from_parameters(
        [30.0, 1.6], 0.4, kijima="i", dist=Weibull
    )
    e, model = _simulate_refit_residuals(
        truth, GeneralizedRenewal, dict(dist=Weibull, kijima="i"), seed=0
    )
    assert e.size > 1500
    assert abs(e.mean() - 1.0) < 0.06
    assert abs(e.var() - 1.0) < 0.12
    # parameters recover
    assert abs(model.q - 0.4) < 0.1
    assert np.allclose(model.model.params, [30.0, 1.6], rtol=0.1)


def test_intensity_reduction_residuals_are_exp1():
    # ARI (intensity reduction): the reduced-intensity integral per interval
    # is Exp(1) under the fitted model -- a different construction entirely.
    truth = ARI.fit_from_parameters([20.0, 1.5], 0.5, m=1, dist=CrowAMSAA)
    e, model = _simulate_refit_residuals(
        truth, ARI, dict(dist=CrowAMSAA, m=1), seed=5
    )
    assert e.size > 1500
    assert abs(e.mean() - 1.0) < 0.06
    assert abs(e.var() - 1.0) < 0.15
    assert abs(model.rho - 0.5) < 0.12


# --- Cramer-von Mises goodness of fit ---------------------------------------
#
# The bootstrap refits the (multi-start) imperfect-repair fit per replicate, so
# these use small data and few replicates. Correctness of the underlying
# transform is covered by the Exp(1) residual calibration above (a unit-Poisson
# compensator is exactly what makes the conditional transforms uniform); one
# model per construction (age reduction, intensity reduction) exercises the
# renewal CvM plumbing.


def _small_fit(truth, fitter, refit_kwargs, seed):
    data = truth.count_terminated_simulation_data(6, items=35, seed=seed)
    return fitter.fit_from_recurrent_data(data, **refit_kwargs)


def test_cvm_age_reduction_runs_and_matches_statistic():
    truth = GeneralizedRenewal.fit_from_parameters(
        [30.0, 1.6], 0.4, kijima="i", dist=Weibull
    )
    model = _small_fit(
        truth, GeneralizedRenewal, dict(dist=Weibull, kijima="i"), seed=0
    )
    result = model.cramer_von_mises(n_boot=10, seed=1)
    assert isinstance(result, GoodnessOfFitResult)
    assert 0.0 < result.p_value <= 1.0
    assert result.n_systems == np.unique(model.data.i).size
    # the observed statistic is the CvM statistic of the compensator transforms
    inc = model._fitter._rescaled_increments(model, model.data)
    u, _ = _renewal_conditional_uniforms(model.data, inc)
    assert np.isclose(result.statistic, cvm_statistic(u))


def test_cvm_intensity_reduction_runs():
    truth = ARI.fit_from_parameters([20.0, 1.5], 0.5, m=1, dist=CrowAMSAA)
    model = _small_fit(truth, ARI, dict(dist=CrowAMSAA, m=1), seed=4)
    result = model.cramer_von_mises(n_boot=10, seed=2)
    assert isinstance(result, GoodnessOfFitResult)
    assert 0.0 < result.p_value <= 1.0


def test_cvm_is_reproducible_with_seed():
    truth = GeneralizedRenewal.fit_from_parameters(
        [30.0, 1.6], 0.4, kijima="i", dist=Weibull
    )
    model = _small_fit(
        truth, GeneralizedRenewal, dict(dist=Weibull, kijima="i"), seed=3
    )
    a = model.cramer_von_mises(n_boot=8, seed=99)
    b = model.cramer_von_mises(n_boot=8, seed=99)
    assert a.statistic == b.statistic
    assert a.p_value == b.p_value


def test_cvm_no_data_guard():
    model = GeneralizedRenewal.fit_from_parameters(
        [10.0, 2.0], 0.3, kijima="i", dist=Weibull
    )
    with pytest.raises(ValueError, match="fitted from data"):
        model.cramer_von_mises()
