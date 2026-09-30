"""``alpha_ci`` sets the level of the recurrent-event bounds (#422).

``alpha_ci`` replaced ``confidence`` (its complement) in v0.21; the old
name was removed in v0.22.0 (see test_removed_arguments.py). The level
is keyword-only, so an old positional ``confidence`` is refused rather
than read as its complement.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from surpyval.recurrent import (  # noqa: E402
    CauseSpecificMCF,
    CrowAMSAA,
    NonParametricCounting,
    ProportionalIntensityNHPP,
)

X = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60, 5, 18, 30, 50, 60.0]
I = [1] * 6 + [2] * 5 + [3] * 5  # noqa: E741
C = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
E = ["a", "b", "a", "b", "a", None, "b", "a", "a", "b", None]
E += ["a", "b", "b", "a", None]
Z = np.array([0.0] * 6 + [1.0] * 5 + [0.5] * 5).reshape(-1, 1)
GRID = [5.0, 20.0, 40.0]


def _nhpp():
    return CrowAMSAA.fit(X, i=I, c=C)


def _pi():
    return ProportionalIntensityNHPP.fit(X, Z, i=I, c=C, dist=CrowAMSAA)


def _mcf():
    return NonParametricCounting.fit(X, i=I, c=C)


def _cs_mcf():
    return CauseSpecificMCF.fit(X, i=I, c=C, e=E)


def test_level_is_keyword_only():
    # confidence used to be reachable by position (plot's first argument,
    # mcf_cb's fourth); the same number now means its complement, so a
    # positional level is refused rather than silently reinterpreted.
    mcf = _mcf()
    with pytest.raises(TypeError):
        mcf.mcf_cb(GRID, "two-sided", "step", 0.95)
    with pytest.raises(TypeError):
        mcf.plot(0.95)
    with pytest.raises(TypeError):
        _cs_mcf().plot(0.95)
    with pytest.raises(TypeError):
        _nhpp().plot(None, True, 0.95)
    with pytest.raises(TypeError):
        _pi().plot(None, True, 0.95)
    plt.close("all")


def test_alpha_ci_sets_the_level():
    # alpha_ci is the total tail probability: the default 0.05 gives the
    # 95% bounds the default confidence=0.95 gave, and the plots label it.
    mcf = _mcf()
    np.testing.assert_allclose(
        mcf.mcf_cb(GRID), mcf.mcf_cb(GRID, alpha_ci=0.05)
    )
    wide = mcf.mcf_cb(GRID, alpha_ci=0.01)
    narrow = mcf.mcf_cb(GRID, alpha_ci=0.2)
    assert np.all(wide[:, 0] <= narrow[:, 0])
    assert np.all(wide[:, 1] >= narrow[:, 1])
    _, ax = plt.subplots()
    mcf.plot(ax=ax, alpha_ci=0.1)
    assert ax.lines[1].get_label() == "90% Confidence Bounds"
    _, ax = plt.subplots()
    _nhpp().plot(ax=ax, alpha_ci=0.1)
    assert ax.collections[0].get_label() == "90% Confidence Band"
    plt.close("all")
