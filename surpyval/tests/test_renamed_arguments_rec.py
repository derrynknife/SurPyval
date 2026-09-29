"""Renamed arguments of the recurrent-event models (#422, principle 21).

``seed`` is now ``random_state``, ``confidence`` is ``alpha_ci`` (its
complement: ``alpha_ci = 1 - confidence``) and ``cause`` is ``event``. The
old names keep working until v0.22.0 with a ``DeprecationWarning`` that
points at the caller; each gives the same answer as the new one.
"""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from surpyval import Weibull  # noqa: E402
from surpyval.recurrent import diagnostics  # noqa: E402
from surpyval.recurrent import (  # noqa: E402
    ARA,
    CauseSpecificMCF,
    CauseSpecificNHPP,
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


def _renewal():
    truth = ARA.fit_from_parameters([20.0, 1.5], 0.5, dist=Weibull)
    data = truth.count_terminated_simulation_data(4, items=6, random_state=0)
    return ARA.fit_from_recurrent_data(data, dist=Weibull)


def _mcf():
    return NonParametricCounting.fit(X, i=I, c=C)


def _cs_mcf():
    return CauseSpecificMCF.fit(X, i=I, c=C, e=E)


def _cs_nhpp():
    return CauseSpecificNHPP.fit(X, i=I, c=C, e=E)


def _plotted(draw):
    # What a plot drew: its lines' y values and its filled bands' outlines.
    _, ax = plt.subplots()
    draw(ax)
    out = [np.asarray(line.get_ydata(), float) for line in ax.lines]
    out += [
        path.vertices
        for collection in ax.collections
        for path in collection.get_paths()
    ]
    plt.close("all")
    return out


def _data(d):
    return [d.x, d.i, d.c, d.n]


# (label, fit, call(model, **kw) -> numbers, old keyword, new keyword)
CASES = [
    # seed -> random_state: the simulations and the parametric bootstraps
    (
        "count_terminated_simulation",
        _nhpp,
        lambda m, **kw: m.count_terminated_simulation(5, 20, **kw).mcf_hat,
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "count_terminated_simulation_data",
        _nhpp,
        lambda m, **kw: _data(
            m.count_terminated_simulation_data(5, items=4, **kw)
        ),
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "time_terminated_simulation",
        _nhpp,
        lambda m, **kw: m.time_terminated_simulation(60.0, 20, **kw).mcf_hat,
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "time_terminated_simulation_data",
        _nhpp,
        lambda m, **kw: _data(
            m.time_terminated_simulation_data(60.0, items=4, **kw)
        ),
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "cramer_von_mises",
        _nhpp,
        lambda m, **kw: m.cramer_von_mises(n_boot=5, **kw).p_value,
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "diagnostics.cramer_von_mises",
        _nhpp,
        lambda m, **kw: diagnostics.cramer_von_mises(m, 5, **kw).p_value,
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "PI.count_terminated_simulation",
        _pi,
        lambda m, **kw: m.count_terminated_simulation(
            5, [0.5], items=20, **kw
        ).mcf_hat,
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "PI.count_terminated_simulation_data",
        _pi,
        lambda m, **kw: _data(
            m.count_terminated_simulation_data(5, [0.5], items=4, **kw)
        ),
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "PI.time_terminated_simulation",
        _pi,
        lambda m, **kw: m.time_terminated_simulation(
            60.0, [0.5], items=20, **kw
        ).mcf_hat,
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "PI.time_terminated_simulation_data",
        _pi,
        lambda m, **kw: _data(
            m.time_terminated_simulation_data(60.0, [0.5], items=4, **kw)
        ),
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "PI.mcf",
        _pi,
        lambda m, **kw: m.mcf(GRID, [0.5], items=50, **kw),
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "PI.cramer_von_mises",
        _pi,
        lambda m, **kw: m.cramer_von_mises(n_boot=3, **kw).p_value,
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "RenewalModel.mcf",
        _renewal,
        lambda m, **kw: m.mcf(GRID, items=50, **kw),
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "RenewalModel.plot",
        _renewal,
        lambda m, **kw: _plotted(lambda ax: m.plot(ax=ax, items=20, **kw)),
        {"seed": 3},
        {"random_state": 3},
    ),
    (
        "RenewalModel.cramer_von_mises",
        _renewal,
        lambda m, **kw: m.cramer_von_mises(n_boot=4, **kw).p_value,
        {"seed": 3},
        {"random_state": 3},
    ),
    # confidence -> alpha_ci (= 1 - confidence)
    (
        "NonParametricCounting.mcf_cb",
        _mcf,
        lambda m, **kw: m.mcf_cb(GRID, bound="upper", **kw),
        {"confidence": 0.9},
        {"alpha_ci": 0.1},
    ),
    (
        "NonParametricCounting.plot",
        _mcf,
        lambda m, **kw: _plotted(lambda ax: m.plot(ax=ax, **kw)),
        {"confidence": 0.8},
        {"alpha_ci": 0.2},
    ),
    (
        "ParametricRecurrenceModel.plot",
        _nhpp,
        lambda m, **kw: _plotted(lambda ax: m.plot(ax=ax, **kw)),
        {"confidence": 0.8},
        {"alpha_ci": 0.2},
    ),
    (
        "ProportionalIntensityModel.plot",
        _pi,
        lambda m, **kw: _plotted(lambda ax: m.plot(ax=ax, **kw)),
        {"confidence": 0.8},
        {"alpha_ci": 0.2},
    ),
    (
        "CauseSpecificMCF.plot",
        _cs_mcf,
        lambda m, **kw: _plotted(lambda ax: m.plot(ax=ax, **kw)),
        {"confidence": 0.8},
        {"alpha_ci": 0.2},
    ),
    (
        "CauseSpecificMCF.mcf_cb[confidence]",
        _cs_mcf,
        lambda m, **kw: m.mcf_cb(GRID, "a", **kw),
        {"confidence": 0.8},
        {"alpha_ci": 0.2},
    ),
    # cause -> event
    (
        "CauseSpecificMCF.mcf",
        _cs_mcf,
        lambda m, **kw: m.mcf(GRID, **kw),
        {"cause": "b"},
        {"event": "b"},
    ),
    (
        "CauseSpecificMCF.mcf_cb[cause]",
        _cs_mcf,
        lambda m, **kw: m.mcf_cb(GRID, **kw),
        {"cause": "b"},
        {"event": "b"},
    ),
    (
        "CauseSpecificNHPP.cif",
        _cs_nhpp,
        lambda m, **kw: m.cif(GRID, **kw),
        {"cause": "b"},
        {"event": "b"},
    ),
    (
        "CauseSpecificNHPP.iif",
        _cs_nhpp,
        lambda m, **kw: m.iif(GRID, **kw),
        {"cause": "b"},
        {"event": "b"},
    ),
    (
        "CauseSpecificNHPP.mcf",
        _cs_nhpp,
        lambda m, **kw: m.mcf(GRID, **kw),
        {"cause": "b"},
        {"event": "b"},
    ),
]


def _same(a, b):
    if isinstance(a, list):
        assert len(a) == len(b)
        for u, v in zip(a, b):
            _same(u, v)
        return
    # 1 - 0.9 is not 0.1 to the last bit; the answers agree to rounding.
    np.testing.assert_allclose(
        np.asarray(a, float), np.asarray(b, float), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize(
    "fit, call, old, new",
    [pytest.param(*case[1:], id=case[0]) for case in CASES],
)
def test_old_name_warns_and_agrees(fit, call, old, new):
    model = fit()
    with warnings.catch_warnings():
        # The simulations' own warnings (stalled sequences, a refit's
        # optimiser) are not what this test is about.
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", DeprecationWarning)
        expected = call(model, **new)
    [(old_name, _)] = old.items()
    [(new_name, _)] = new.items()
    with pytest.warns(DeprecationWarning) as caught:
        got = call(model, **old)
    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecations) == 1
    message = str(deprecations[0].message)
    assert f"'{old_name}' is deprecated" in message
    assert f"use '{new_name}'" in message
    assert "v0.22.0" in message
    # The warning points at the caller (this file), not at the package.
    assert deprecations[0].filename == __file__
    _same(got, expected)


def test_old_and_new_names_together_are_refused():
    model = _mcf()
    with pytest.raises(ValueError, match="alpha_ci"):
        model.mcf_cb(GRID, confidence=0.9, alpha_ci=0.1)
    with pytest.raises(ValueError, match="random_state"):
        _nhpp().count_terminated_simulation(3, 2, seed=1, random_state=1)
    with pytest.raises(ValueError, match="event"):
        _cs_nhpp().cif(GRID, cause="a", event="a")


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
