"""Reproducible random draws (Conventions, "Random draws and seeds").

- With no seed, a draw comes from numpy's global generator, so
  ``np.random.seed`` reproduces it (and a different seed changes it).
- An explicit seed gives a stream of its own: the same seed gives the
  same draw, an int seed is ``np.random.default_rng(seed)``, and the
  draw neither depends on nor advances the global stream.

The draws are ``random()`` where a model has one, the recurrent-event
simulations, the simulated MCF of a renewal model and, for the random
survival forest, the fit itself.
"""

import numpy as np
import pytest

from surpyval.tests.conformance.registry import cases_for, fitted, flat


def _draw(case, seed):
    return flat(case.draw(fitted(case), seed))


@pytest.mark.parametrize("case", cases_for("seed_global"))
def test_global_seed_reproduces_a_draw(case):
    np.random.seed(0)
    a = _draw(case, None)
    np.random.seed(0)
    b = _draw(case, None)
    np.random.seed(1)
    c = _draw(case, None)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c, equal_nan=True), "the seed is ignored"


@pytest.mark.parametrize("case", cases_for("seed_explicit"))
def test_explicit_seed_is_its_own_stream(case):
    np.random.seed(1)
    a = _draw(case, 7)
    after = np.random.uniform(size=4)
    np.random.seed(1)
    untouched = np.random.uniform(size=4)
    np.random.seed(2)
    b = _draw(case, 7)
    np.testing.assert_array_equal(a, b)
    # Drawing with a seed of its own does not advance the global stream.
    np.testing.assert_array_equal(after, untouched)
    # An int seed means default_rng(seed).
    np.testing.assert_array_equal(_draw(case, np.random.default_rng(7)), a)
