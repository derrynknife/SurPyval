"""Hypothesis settings for the property-based tests (#379).

Two profiles, chosen with the ``SURPYVAL_HYPOTHESIS_PROFILE`` environment
variable:

``fast`` (the default)
    A few examples per property, small data sets and the slower
    properties on fewer models: under a minute for the directory on one
    core, so the full suite can afford it. No per-example deadline: the
    CI suite runs under coverage, which made one Weibull example take 14 s
    against a 10 s deadline, and a slow machine is not a failure. The
    fixed number of examples is what bounds the run time.
``nightly``
    Many more examples and no deadline (a slow machine is not a
    failure), for the scheduled run::

        SURPYVAL_HYPOTHESIS_PROFILE=nightly python -m pytest \\
            surpyval/tests/properties

Both are derandomised (the examples are a function of the test alone)
and keep no example database, so a run is reproducible: a failure seen
in CI is seen again locally with the same command, and nothing a
previous run found changes what the next one tries. A counterexample
worth keeping is written into the test as an explicit ``@example``.

As in the conformance suite, a test fails if a raw numerical warning
leaks out of the package while it runs (see
``surpyval/tests/conformance/leaks.py``). The fits here stay silenced
(``common.quietly``: generated data are meant to be awkward), so it is
the predictions, the model's answers, that are checked. A leak fails the
test as a whole, after its examples: it is not shrunk to a minimal one.
"""

import os
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

from surpyval.tests.conformance import leaks

_COMMON: dict[str, Any] = dict(
    derandomize=True,
    database=None,
    print_blob=True,
    # The strategies build whole data sets; the fitters, not the
    # generation, are what should be slow.
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("fast", max_examples=5, deadline=None, **_COMMON)
settings.register_profile(
    "nightly", max_examples=400, deadline=None, **_COMMON
)
settings.load_profile(os.environ.get("SURPYVAL_HYPOTHESIS_PROFILE", "fast"))


@pytest.fixture(autouse=True)
def _restore_global_rng():
    # Some fitters draw from the global stream; put it back as found.
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    with leaks.watch() as found:
        result = yield
    leaks.fail_on_new(found)
    return result
