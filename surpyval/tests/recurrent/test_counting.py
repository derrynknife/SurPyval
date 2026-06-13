import warnings

import numpy as np
import pytest

from surpyval import Exponential, Gamma, Normal, Weibull
from surpyval.recurrent import HPP
from surpyval.recurrent.renewal import (
    GeneralizedOneRenewal,
    GeneralizedRenewal,
)


def test_g1_renewal():
    # Solution from:
    # Kaminskiy, M. P., and V. V. Krivtsov.
    # "G1-renewal process as repairable system model."
    # Reliability: Theory & Applications 5.3 (18) (2010): 7-14.
    # Ref:
    # https://arxiv.org/pdf/1006.3718.pdf

    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()

    model = GeneralizedOneRenewal.fit(x, dist=Exponential)
    life = 1.0 / model.model.params[0]

    assert np.allclose([0.232], model.q, atol=1e-3)
    assert np.allclose([4.781], life, atol=1e-3)


def test_g1_renewal_scale_family_dist():
    # The time-axis formulation lets G1 fit any non-negative lifetime
    # distribution, not just Weibull/Exponential. Gamma is a scale family
    # whose scale parameter is not the first positional parameter, so this
    # would have been silently mishandled by the old parameter-scaling code.
    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()
    model = GeneralizedOneRenewal.fit(x, dist=Gamma)

    assert model.model.dist == Gamma
    assert np.all(np.isfinite(model.model.params))
    assert np.isfinite(model.q)


def test_g1_renewal_rejects_distribution_with_negative_support():
    # Distributions with support over negative values cannot be scaled into a
    # valid interarrival distribution and must be rejected up front.
    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()

    with pytest.raises(ValueError, match="non-negative lifetime"):
        GeneralizedOneRenewal.fit(x, dist=Normal)

    with pytest.raises(ValueError, match="non-negative lifetime"):
        GeneralizedOneRenewal.fit_from_parameters([10, 2], 0.2, dist=Normal)


@pytest.mark.parametrize("model", [GeneralizedOneRenewal, GeneralizedRenewal])
def test_renewal_rejects_unsupported_censoring(model):
    # The renewal likelihoods only define contributions for exact (c=0) and
    # right-censored (c=1) observations. Interval (c=2) and left (c=-1)
    # censoring must be rejected rather than silently dropped.
    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])

    c_interval = np.array([0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="censoring code"):
        model.fit(x, i, c=c_interval)

    c_left = np.array([-1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="censoring code"):
        model.fit(x, i, c=c_left)


@pytest.mark.parametrize(
    "model, module_path",
    [
        (
            GeneralizedOneRenewal,
            "surpyval.recurrent.renewal.generalized_one_renewal",
        ),
        (
            GeneralizedRenewal,
            "surpyval.recurrent.renewal.generalized_renewal",
        ),
    ],
)
def test_renewal_raises_when_no_start_converges(
    model, module_path, monkeypatch
):
    # Both renewal models share one contract: if no multi-start initial value
    # converges, raise rather than silently return an unconverged fit. Force
    # every optimizer call to report failure to exercise that path.
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module(module_path)

    def failing_minimize(*args, **kwargs):
        return SimpleNamespace(
            success=False, fun=np.inf, x=np.array([1.0, 1.0, 1.0])
        )

    monkeypatch.setattr(module, "minimize", failing_minimize)

    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
    c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

    with pytest.raises(ValueError, match="Could not find a good solution"):
        model.fit(x, i, c=c)


@pytest.mark.parametrize(
    "model, module_path",
    [
        (
            GeneralizedOneRenewal,
            "surpyval.recurrent.renewal.generalized_one_renewal",
        ),
        (
            GeneralizedRenewal,
            "surpyval.recurrent.renewal.generalized_renewal",
        ),
    ],
)
def test_renewal_raises_when_user_init_does_not_converge(
    model, module_path, monkeypatch
):
    # A user-supplied `init` that fails to converge must raise too, rather
    # than silently returning the unconverged result.
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module(module_path)

    def failing_minimize(*args, **kwargs):
        return SimpleNamespace(
            success=False, fun=np.inf, x=np.array([1.0, 1.0, 1.0])
        )

    monkeypatch.setattr(module, "minimize", failing_minimize)

    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
    c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

    with pytest.raises(ValueError, match="did not.*converge"):
        model.fit(x, i, c=c, init=[1.0, 1.0, 1.0])


def test_count_terminated_simulation_via_mixin():
    # The shared RecurrenceSimulationMixin drives count_terminated_simulation
    # for every recurrent model. Pin the documented G1 result so the refactor
    # stays behaviour-preserving.
    x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])
    model = GeneralizedOneRenewal.fit(x, dist=Weibull)
    np.random.seed(0)
    np_model = model.count_terminated_simulation(len(x), 5000)
    expected = np.array(
        [0.1696, 1.181, 2.287, 3.6694, 5.58237925, 8.54474531]
    )
    assert np.allclose(np_model.mcf(np.array([1, 2, 3, 4, 5, 6])), expected)


def test_time_terminated_tol_stops_decaying_sequence():
    # A G1 process with q < 0 has geometrically shrinking interarrival times,
    # so its cumulative time converges below any large T. The tol early-exit
    # must catch this and the simulation must terminate (not hang).
    model = GeneralizedOneRenewal.fit_from_parameters(
        [5.0, 1.5], q=-0.6, dist=Weibull
    )
    np.random.seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np_model = model.time_terminated_simulation(T=1e6, items=10)
    assert any("possible asymptote" in str(w.message) for w in caught)
    assert len(np_model.x) > 0


def test_time_terminated_max_events_backstop():
    # max_events guarantees termination even when tol never trips: a low-rate
    # process with a tiny cap and a huge T must stop at the cap and warn.
    x = Exponential.random(20, 1.0).cumsum()
    hpp = HPP.fit(x)
    np.random.seed(1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hpp.time_terminated_simulation(T=1e6, items=3, max_events=5)
    assert any("max_events" in str(w.message) for w in caught)


def test_time_terminated_reaching_sequence_does_not_warn():
    # A well-behaved process that comfortably reaches T must not trip either
    # the tol or max_events early-exit (no false positives).
    model = GeneralizedOneRenewal.fit_from_parameters(
        [5.0, 1.5], q=0.2, dist=Weibull
    )
    np.random.seed(2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np_model = model.time_terminated_simulation(T=60, items=30)
    assert caught == []
    assert len(np_model.x) > 0
