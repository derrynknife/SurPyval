"""
Every surpyval model should participate in the ``Distribution`` ABC
hierarchy so that user code can dispatch on ``isinstance(model,
Distribution)``. These tests pin the adoption down across the model
families.
"""

import numpy as np

import surpyval as sp
from surpyval import (
    Distribution,
    NonParametricDistribution,
    ParametricDistribution,
)
from surpyval.alpha.parallel import ParallelModel
from surpyval.alpha.series import SeriesModel


def _parametric_model():
    np.random.seed(0)
    x = np.concatenate(
        [sp.Weibull.random(50, 10, 3), sp.Weibull.random(50, 50, 4)]
    )
    return sp.Weibull.fit(x), x


def test_parametric_is_parametric_distribution():
    model, _ = _parametric_model()
    assert isinstance(model, Distribution)
    assert isinstance(model, ParametricDistribution)


def test_nonparametric_is_nonparametric_distribution():
    model = sp.NelsonAalen.fit([1.0, 2.0, 3.0, 4.0, 5.0])
    assert isinstance(model, Distribution)
    assert isinstance(model, NonParametricDistribution)


def test_mixture_model_is_distribution():
    _, x = _parametric_model()
    mm = sp.MixtureModel(dist=sp.Weibull, m=2)
    mm.fit(x=x)
    assert isinstance(mm, Distribution)


def test_degenerate_models_are_distributions():
    assert issubclass(sp.NeverOccurs, Distribution)
    assert issubclass(sp.InstantlyOccurs, Distribution)


def test_composed_models_are_distributions():
    model, _ = _parametric_model()
    assert isinstance(SeriesModel([model, model]), Distribution)
    assert isinstance(ParallelModel([model, model]), Distribution)


def test_composition_accepts_any_distribution_leaf():
    # The | and & operators previously only recognised Parametric leaves.
    # They now accept any non-composite Distribution (here NonParametric).
    weibull, _ = _parametric_model()
    npm = sp.NelsonAalen.fit([1.0, 2.0, 3.0, 4.0, 5.0])
    grid = np.array([1.0, 2.0, 3.0])

    series = SeriesModel([weibull]) | npm
    assert len(series.models) == 2
    # Series survival is the product of component survival functions.
    assert np.allclose(series.sf(grid), weibull.sf(grid) * npm.sf(grid))

    parallel = ParallelModel([weibull]) & npm
    assert len(parallel.models) == 2
    # Parallel failure is the product of component failure functions.
    assert np.allclose(parallel.ff(grid), weibull.ff(grid) * npm.ff(grid))


def test_hf_default_derives_from_sf():
    # MixtureModel has no closed-form Hf, so it falls back to the
    # Distribution default Hf = -log(sf).
    _, x = _parametric_model()
    mm = sp.MixtureModel(dist=sp.Weibull, m=2)
    mm.fit(x=x)
    grid = np.array([1.0, 5.0, 10.0, 25.0])
    assert np.allclose(mm.Hf(grid), -np.log(mm.sf(grid)))
