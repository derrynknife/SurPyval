"""The level given survival of every registered process model.

For the fitted models whose ``level_at`` takes ``given_survival`` (see
``SURVIVING_LEVEL`` in ``registry_families.py``): at the median life, where
half of the new units have failed, the level of a working unit has all
its mass below the threshold, ``qf`` inverts ``ff``, and its draws are
reproducible under ``random_state``; at the age by which one unit in
10**12 has failed, it is the unconditioned level. The closed forms and a
simulation are checked in ``tests/degradation/test_surviving_level.py``.
"""

import numpy as np
import pytest

from surpyval.tests.conformance.registry import (
    CASES,
    SURVIVING_LEVEL,
    cases_for,
    fitted,
)

P = np.array([0.01, 0.1, 0.5, 0.9, 0.99])


@pytest.mark.parametrize("case", cases_for("surviving_level"))
def test_surviving_level(case):
    model = fitted(case)
    L = model.threshold
    t = float(np.ravel(model.qf(0.5))[0])
    level = model.level_at(t, given_survival=True)
    assert level.given_survival and level.threshold == L
    assert float(level.ff(L)) == 1.0 and float(level.sf(L)) == 0.0
    q = level.qf(P)
    assert np.all(q < L) and np.all(np.diff(q) > 0)
    np.testing.assert_allclose(level.ff(q), P, rtol=1e-9)
    a = level.random(500, random_state=3)
    np.testing.assert_array_equal(a, level.random(500, random_state=3))
    assert a.max() < L

    early = float(np.ravel(model.qf(1e-12))[0])
    free = model.level_at(early)
    given = model.level_at(early, given_survival=True)
    y = free.qf(P)
    np.testing.assert_allclose(given.ff(y), free.ff(y), atol=1e-10)


def test_every_surviving_level_model_is_registered():
    registered = {c.model_class for c in CASES}
    assert SURVIVING_LEVEL <= registered, SURVIVING_LEVEL - registered
