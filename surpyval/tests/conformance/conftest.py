"""Configuration of the conformance suite (see ``registry.py``).

The pull-request job runs this directory with ``-m "not slow"``; the
properties marked ``slow`` (refits of the less common variants, whose
fast counterparts already run) are left to the full suite.

Every test also fails if a raw numerical warning leaks out of the
package while it runs (a numpy "divide by zero", an overflow inside
autograd; principle 22). ``leaks.py`` says what counts as a leak and
why the check reads the call stack rather than using
``filterwarnings``; its ``KNOWN_LEAKS`` are the leaks not fixed yet,
which ``test_warnings.py`` reproduces as strict xfails.
"""

import numpy as np
import pytest

from surpyval.tests.conformance import leaks


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: a conformance check left out of the pull-request job "
        "(-m 'not slow'); the full suite runs it",
    )


@pytest.fixture(autouse=True)
def _restore_global_rng():
    # The seed properties seed the global stream; put it back as found.
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    # Around the test body only, so a leak is the test's failure (an
    # xfail mark applies to it) and not an error in a fixture.
    with leaks.watch() as found:
        result = yield
    leaks.fail_on_new(found)
    return result
