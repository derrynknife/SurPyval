"""Configuration of the conformance suite (see ``registry.py``).

The pull-request job runs this directory with ``-m "not slow"``; the
properties marked ``slow`` (refits of the less common variants, whose
fast counterparts already run) are left to the full suite.
"""

import numpy as np
import pytest


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
