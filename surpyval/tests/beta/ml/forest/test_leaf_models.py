"""Parametric leaves on observed and right-censored data are built from
the maximum the split search finds, not re-fitted: the same model as a
full fit, without its cost."""

from unittest import mock

import numpy as np
import pytest

from surpyval import Exponential, Weibull
from surpyval.beta.ml.forest import RandomSurvivalForest
from surpyval.beta.ml.forest.node import TerminalNode
from surpyval.utils.surpyval_data import SurpyvalData


def _leaf_data(seed, size=40):
    rng = np.random.default_rng(seed)
    x = rng.weibull(rng.uniform(0.5, 4.0), size) * rng.uniform(0.1, 100.0)
    c = (rng.uniform(size=size) < 0.4).astype(int)
    n = rng.integers(1, 3, size)
    return SurpyvalData(x, c, n)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize(
    "kind, dist", [("weibull", Weibull), ("exponential", Exponential)]
)
def test_leaf_is_the_full_fit(seed, kind, dist):
    data = _leaf_data(seed)
    leaf = TerminalNode(data, kind)
    full = dist.fit_from_surpyval_data(data)
    np.testing.assert_allclose(leaf.model.params, full.params, rtol=1e-4)
    # The leaf's is the maximum: no lower than the optimiser's
    # (gamma, f0, p) = (0, 0, 1): no offset, zero-inflation or cure
    leaf_nll = dist._neg_ll_func(data, *leaf.model.params, 0.0, 0.0, 1.0)
    full_nll = dist._neg_ll_func(data, *full.params, 0.0, 0.0, 1.0)
    assert leaf_nll <= full_nll + 1e-9


def test_leaves_are_not_refitted():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 3))
    x = rng.weibull(1.5, 300) * np.exp(0.5 * Z[:, 0])
    c = (rng.uniform(size=300) < 0.3).astype(int)
    forest = RandomSurvivalForest.fit(
        x=x, Z=Z, c=c, n_trees=3, kind="weibull", random_state=1
    )
    refitted = []
    fit = Weibull.fit_from_surpyval_data

    def spy(data, *args, **kwargs):
        refitted.append(np.unique(data.x[data.c == 0]).size)
        return fit(data, *args, **kwargs)

    with mock.patch.object(Weibull, "fit_from_surpyval_data", spy):
        forest.sf(np.array([1.0]), Z)
    # Only the leaves Weibull.fit refuses at once (one failure time, #462)
    # reach it, on their way to the exponential.
    assert set(refitted) <= {1}


def test_one_failure_time_is_exponential():
    # Weibull.fit refuses one distinct failure time (#462), and the leaf
    # falls back to the exponential, as before.
    data = SurpyvalData(
        np.array([2.0, 2.0, 2.0, 3.0, 4.0]), np.array([0, 0, 0, 1, 1])
    )
    leaf = TerminalNode(data, "weibull")
    assert leaf.model.dist is Exponential
    np.testing.assert_allclose(leaf.model.params, [3 / 13])


def test_steep_leaf_keeps_its_shape():
    # Failures bunched together: a shape far above the split search's
    # window (20), as Weibull.fit finds it.
    rng = np.random.default_rng(4)
    data = SurpyvalData(10.0 + rng.uniform(0, 0.05, 12), np.zeros(12, int))
    leaf = TerminalNode(data, "weibull")
    full = Weibull.fit_from_surpyval_data(data)
    assert full.params[1] > 100
    np.testing.assert_allclose(leaf.model.params, full.params, rtol=1e-4)
