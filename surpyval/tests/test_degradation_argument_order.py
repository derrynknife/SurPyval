"""
#422 in the degradation models: ``Z`` comes straight after the query
(principle 21).

``DegradationModel.cb`` and the process models' ``random`` take ``Z``
second; ``induced_life`` and both ``predict_rul`` take everything after
the query by keyword. The old orders (``Z`` last) and the old names
(``t``, ``q``, ``seed``) were removed in v0.22: a call in the old order
now fails rather than being read with its old meaning (see also
test_removed_arguments.py).
"""

import warnings

import numpy as np
import pytest

from surpyval.degradation import (
    DegradationAnalysis,
    GammaProcessModel,
    WienerProcessModel,
)
from surpyval.tests.conformance.registry import CASE_BY_NAME, fitted

T = np.array([5.0, 20.0, 40.0])


def _quiet(call):
    """``call()``, which must not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return call()


def _model(name):
    return fitted(CASE_BY_NAME[name])


# -- DegradationModel.cb: Z comes second ------------------------------------


def test_cb_takes_Z_second():
    from surpyval.tests.degradation.test_degradation_analysis import adt_data

    x, y, i, Z = adt_data(seed=1)
    model = DegradationAnalysis.fit(x, y, i, threshold=100.0, Z=Z)
    # Z reaches Z (not ``on``): the analytic bound is not derived for an
    # accelerated model, and says so.
    with pytest.raises(NotImplementedError, match="method='bootstrap'"):
        _quiet(lambda: model.cb(T, [1.0]))
    got = _quiet(
        lambda: model.cb(
            T, [1.0], method="bootstrap", n_boot=4, random_state=3
        )
    )
    np.testing.assert_array_equal(
        got,
        model.cb(T, Z=[1.0], method="bootstrap", n_boot=4, random_state=3),
    )
    # The old order, with ``on`` second, now fails.
    with pytest.raises(ValueError, match="'on' must be one of"):
        model.cb(T, "sf", 0.05, "two-sided", "bootstrap", 4, 3, [1.0])


def test_cb_in_the_old_positional_order_fails():
    model = _model("DegradationAnalysis[linear]")
    with pytest.raises(ValueError, match="no covariates"):
        model.cb(T, "ff")
    with pytest.raises(ValueError, match="'on' must be one of"):
        model.cb(T, "ff", 0.1, "lower", "analytic", 200, None, None)


# -- the process models' random: (size, Z, random_state) --------------------


def _accelerated(kind):
    if kind == "wiener":
        return WienerProcessModel(
            0.5, 0.3, 10.0, gamma=[0.7], stress_ref=[0.0]
        )
    return GammaProcessModel(2.0, 1.0, 10.0, gamma=[0.7], stress_ref=[0.0])


@pytest.mark.parametrize("case", ["WienerProcess", "GammaProcess"])
def test_random_with_a_positional_seed_fails(case):
    # A model fitted without stress refuses the second argument, which is
    # ``Z``: it is no longer read as the old positional seed.
    model = _model(case)
    for seed in (4, np.random.default_rng(4)):
        with pytest.raises(ValueError, match="without stress"):
            model.random(6, seed)
    with pytest.raises(ValueError, match="without stress"):
        model.random(6, 4, None)


@pytest.mark.parametrize("kind", ["wiener", "gamma"])
def test_random_takes_Z_second(kind):
    model = _accelerated(kind)
    new = _quiet(lambda: model.random(6, Z=[1.0], random_state=5))
    # the new order, by position
    np.testing.assert_array_equal(
        _quiet(lambda: model.random(6, [1.0], random_state=5)), new
    )
    np.testing.assert_array_equal(
        _quiet(lambda: model.random(6, [1.0], 5)), new
    )
    # a scalar is a stress
    np.testing.assert_array_equal(
        _quiet(lambda: model.random(6, 1, random_state=5)), new
    )
    with pytest.raises(TypeError, match="multiple values"):
        model.random(6, 5, Z=[1.0])


# ``induced_life`` and both ``predict_rul`` took ``Z`` after the seed and
# the level; ``Z`` now comes straight after the query and the rest is
# keyword-only, so a positional argument after the query is refused.


def test_induced_life_is_keyword_only():
    model = _model("DegradationAnalysis[linear]")
    with pytest.raises(TypeError, match="positional"):
        model.induced_life(200, 3)
    assert np.size(model.induced_life(200, random_state=3).samples) == 200


def test_path_predict_rul_is_keyword_only():
    model = _model("DegradationAnalysis[linear]")
    x, y = [1.0, 2.0, 3.0], [1.0, 1.5, 2.0]
    with pytest.raises(TypeError, match="positional"):
        model.predict_rul(x, y, 0.1, 500, 4)


@pytest.mark.parametrize("case", ["WienerProcess", "GammaProcess"])
def test_process_predict_rul_is_keyword_only(case):
    model = _model(case)
    with pytest.raises(TypeError, match="positional"):
        model.predict_rul(1.0, 0.1)
    assert np.isfinite(model.predict_rul(1.0, alpha_ci=0.1).rul)
