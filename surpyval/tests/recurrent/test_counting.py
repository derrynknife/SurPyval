import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from surpyval import Exponential, Gamma, Normal, Weibull  # noqa: E402
from surpyval.recurrent import HPP  # noqa: E402
from surpyval.recurrent.renewal import (  # noqa: E402
    ARA,
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


def test_count_terminated_simulation_data_is_recurrent_data():
    # The data simulator returns raw xicn events (RecurrentEventData), not the
    # aggregated MCF, with events+1 exact events per item.
    from surpyval.utils.recurrent_event_data import RecurrentEventData

    model = GeneralizedOneRenewal.fit_from_parameters(
        [5.0, 1.5], q=0.2, dist=Weibull
    )
    data = model.count_terminated_simulation_data(8, items=20, seed=0)
    assert isinstance(data, RecurrentEventData)
    assert len(data.x) == 20 * (8 + 1)
    assert len(set(data.i.tolist())) == 20
    assert set(data.c.tolist()) == {0}


def test_simulated_data_round_trips_through_fit():
    # Simulating from a known model and refitting recovers it in the right
    # neighbourhood (this is now possible because the simulator yields events).
    truth = ARA.fit_from_parameters([10.0, 2.0], rho=0.5, m=2, dist=Weibull)
    data = truth.count_terminated_simulation_data(events=8, items=400, seed=0)
    refit = ARA.fit(data.x, data.i, c=data.c, m=2)
    assert 0.0 < refit.rho < 1.0
    assert np.all(refit.model.params > 0)


def test_time_terminated_simulation_data_is_censored_at_T():
    from surpyval.utils.recurrent_event_data import RecurrentEventData

    model = GeneralizedOneRenewal.fit_from_parameters(
        [5.0, 1.5], q=0.2, dist=Weibull
    )
    data = model.time_terminated_simulation_data(T=60, items=20, seed=2)
    assert isinstance(data, RecurrentEventData)
    # Each reaching sequence ends in a right-censored row at T.
    assert (data.c == 1).any()
    assert data.x[data.c == 1].max() <= 60.0 + 1e-9


def test_parametric_recurrence_model_has_data_simulators():
    # The data simulators come from the shared mixin, so the intensity models
    # (HPP, CrowAMSAA, ...) get them too.
    from surpyval.utils.recurrent_event_data import RecurrentEventData

    x = Exponential.random(20, 1.0).cumsum()
    hpp = HPP.fit(x)
    data = hpp.count_terminated_simulation_data(10, items=15, seed=0)
    assert isinstance(data, RecurrentEventData)
    assert len(data.x) == 15 * (10 + 1)


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


def test_simulation_seed_is_reproducible():
    # An explicit seed gives a reproducible stream independent of global state;
    # a different seed gives a different result.
    model = GeneralizedOneRenewal.fit_from_parameters(
        [5.0, 1.5], q=0.2, dist=Weibull
    )
    xs = np.array([1, 2, 3, 4, 5])
    a = model.count_terminated_simulation(9, 500, seed=42).mcf(xs)
    b = model.count_terminated_simulation(9, 500, seed=42).mcf(xs)
    c = model.count_terminated_simulation(9, 500, seed=7).mcf(xs)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_seed_none_defers_to_global_rng():
    # seed=None must keep honouring np.random.seed (backward compatible with
    # the documented examples).
    x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])
    model = GeneralizedOneRenewal.fit(x, dist=Weibull)
    np.random.seed(0)
    got = model.count_terminated_simulation(len(x), 5000).mcf(
        np.array([1, 2, 3, 4, 5, 6])
    )
    expected = np.array(
        [0.1696, 1.181, 2.287, 3.6694, 5.58237925, 8.54474531]
    )
    assert np.allclose(got, expected)


@pytest.mark.parametrize(
    "model_cls", [GeneralizedOneRenewal, GeneralizedRenewal]
)
def test_renewal_mcf_convenience(model_cls):
    # mcf(x) estimates a sensible, non-decreasing MCF by simulation.
    model = model_cls.fit_from_parameters([5.0, 1.5], 0.2, dist=Weibull)
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mcf = model.mcf(xs, items=2000, seed=1)
    assert mcf.shape == xs.shape
    assert np.all(np.diff(mcf) >= -1e-9)
    assert np.all(mcf >= 0)


def test_plot_requires_fitted_data():
    # plot needs the empirical MCF, so a from-parameters model cannot plot.
    model = GeneralizedOneRenewal.fit_from_parameters(
        [5.0, 1.5], 0.2, dist=Weibull
    )
    with pytest.raises(ValueError, match="requires a model fitted from data"):
        model.plot()


def test_plot_returns_axes_when_fitted():
    model = GeneralizedRenewal.fit(
        np.array([1, 3, 6, 9, 10]), c=np.array([0, 0, 0, 0, 1])
    )
    ax = model.plot(items=300, seed=2)
    assert ax is not None


def test_renewal_information_criteria():
    # logL, AIC and BIC follow the standard definitions and logL matches the
    # optimiser's stored objective.
    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()
    model = GeneralizedOneRenewal.fit(x, dist=Exponential)
    k = model._mle.size
    n = model._n_obs
    ll = model.log_likelihood
    assert np.isclose(ll, -model.res.fun)
    assert np.isclose(model.aic, 2 * k - 2 * ll)
    assert np.isclose(model.bic, k * np.log(n) - 2 * ll)


def test_renewal_standard_errors_interior_optimum():
    # With q in the interior, standard errors are finite and positive.
    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()
    model = GeneralizedOneRenewal.fit(x, dist=Exponential)
    cov = model.covariance()
    se = model.standard_errors()
    assert cov.shape == (model._mle.size, model._mle.size)
    assert se.shape == (model._mle.size,)
    assert np.all(np.isfinite(se)) and np.all(se > 0)


def test_inference_requires_fit_from_data():
    # fit_from_parameters carries no likelihood, so inference must raise.
    model = GeneralizedOneRenewal.fit_from_parameters(
        [5.0, 1.5], 0.2, dist=Weibull
    )
    for attr in ("log_likelihood", "aic", "bic"):
        with pytest.raises(ValueError, match="fitted from data"):
            getattr(model, attr)
    with pytest.raises(ValueError, match="fitted from data"):
        model.standard_errors()
