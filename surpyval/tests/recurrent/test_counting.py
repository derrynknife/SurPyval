import numpy as np
import pytest

from surpyval import Exponential, Gamma, Normal
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
