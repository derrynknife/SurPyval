"""
#422 in the degradation models: one name per option (principle 21).

Each renamed argument still works under its old name until v0.22.0, with a
``DeprecationWarning`` pointing at the caller, and gives the answer the new
name gives. ``DegradationModel.cb`` and the process models' ``random`` now
take ``Z`` straight after the query; a call by position in the old order
keeps its old meaning, with a warning.
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


def _deprecated(call, old):
    """``call()``, checking it warns about ``old`` from this file."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = call()
    hits = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(hits) == 1, [str(w.message) for w in caught]
    assert old in str(hits[0].message)
    assert hits[0].filename == __file__  # points at the caller
    return out


def _quiet(call):
    """``call()``, which must not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return call()


def _model(name):
    return fitted(CASE_BY_NAME[name])


def _process(case):
    return _model(case)


# (label, model, old call, new call, old name)
RENAMES = [
    *[
        (
            f"{case}.{fn}(t=)",
            case,
            lambda m, fn=fn: getattr(m, fn)(t=T),
            lambda m, fn=fn: getattr(m, fn)(x=T),
            "'t'",
        )
        for case in ("WienerProcess", "GammaProcess", "DestructiveDegradation")
        for fn in ("sf", "ff", "df", "hf", "Hf")
        if not (case == "DestructiveDegradation" and fn == "hf")
    ],
    (
        "DestructiveDegradation.cb(t=)",
        "DestructiveDegradation",
        lambda m: m.cb(t=T, n_boot=5, random_state=1),
        lambda m: m.cb(x=T, n_boot=5, random_state=1),
        "'t'",
    ),
    (
        "DestructiveDegradation.cb(seed=)",
        "DestructiveDegradation",
        lambda m: m.cb(T, n_boot=5, seed=1),
        lambda m: m.cb(T, n_boot=5, random_state=1),
        "'seed'",
    ),
    (
        "DestructiveDegradation.median_degradation(t=)",
        "DestructiveDegradation",
        lambda m: m.median_degradation(t=T),
        lambda m: m.median_degradation(x=T),
        "'t'",
    ),
    (
        "DestructiveDegradation.degradation_quantile(q=)",
        "DestructiveDegradation",
        lambda m: m.degradation_quantile(q=0.1, x=T),
        lambda m: m.degradation_quantile(p=0.1, x=T),
        "'q'",
    ),
    (
        "DestructiveDegradation.degradation_quantile(t=)",
        "DestructiveDegradation",
        lambda m: m.degradation_quantile(0.1, t=T),
        lambda m: m.degradation_quantile(0.1, x=T),
        "'t'",
    ),
    (
        "DegradationModel.cb(seed=)",
        "DegradationAnalysis[linear]",
        lambda m: m.cb(T, method="bootstrap", n_boot=5, seed=1),
        lambda m: m.cb(T, method="bootstrap", n_boot=5, random_state=1),
        "'seed'",
    ),
]


@pytest.mark.parametrize(
    "case, old, new, name",
    [pytest.param(*row[1:], id=row[0]) for row in RENAMES],
)
def test_old_name_warns_and_agrees(case, old, new, name):
    model = _model(case)
    got = _deprecated(lambda: old(model), name)
    np.testing.assert_array_equal(got, _quiet(lambda: new(model)))


# -- DegradationModel.cb: Z comes second ------------------------------------


def test_cb_in_the_old_positional_order_keeps_its_meaning():
    model = _model("DegradationAnalysis[linear]")
    new = model.cb(T, on="ff", alpha_ci=0.1, bound="lower")
    old = _deprecated(
        lambda: model.cb(T, "ff", 0.1, "lower", "analytic", 200, None, None),
        "old order",
    )
    np.testing.assert_array_equal(old, new)
    boot = model.cb(T, method="bootstrap", n_boot=5, random_state=2)
    old = _deprecated(
        lambda: model.cb(T, "sf", 0.05, "two-sided", "bootstrap", 5, 2),
        "old order",
    )
    np.testing.assert_array_equal(old, boot)


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
    old = _deprecated(
        lambda: model.cb(T, "sf", 0.05, "two-sided", "bootstrap", 4, 3, [1.0]),
        "old order",
    )
    np.testing.assert_array_equal(old, got)


def test_cb_refuses_an_argument_given_twice():
    model = _model("DegradationAnalysis[linear]")
    with pytest.raises(TypeError, match="multiple values for argument 'on'"):
        model.cb(T, "sf", on="ff")


# -- the process models' random: (size, Z, random_state) --------------------


def _accelerated(kind):
    if kind == "wiener":
        return WienerProcessModel(
            0.5, 0.3, 10.0, gamma=[0.7], stress_ref=[0.0]
        )
    return GammaProcessModel(2.0, 1.0, 10.0, gamma=[0.7], stress_ref=[0.0])


@pytest.mark.parametrize("case", ["WienerProcess", "GammaProcess"])
def test_random_with_a_positional_seed_keeps_its_meaning(case):
    model = _process(case)
    new = model.random(6, random_state=4)
    # an int after size, for a model fitted without stress (which refuses Z)
    np.testing.assert_array_equal(
        _deprecated(lambda: model.random(6, 4), "old order"), new
    )
    # a Generator is only ever a seed
    np.testing.assert_array_equal(
        _deprecated(
            lambda: model.random(6, np.random.default_rng(4)), "old order"
        ),
        new,
    )
    # the old order in full
    np.testing.assert_array_equal(
        _deprecated(lambda: model.random(6, 4, None), "old order"), new
    )


@pytest.mark.parametrize("kind", ["wiener", "gamma"])
def test_random_takes_Z_second(kind):
    model = _accelerated(kind)
    new = _quiet(lambda: model.random(6, Z=[1.0], random_state=5))
    # the new order, by position
    np.testing.assert_array_equal(
        _quiet(lambda: model.random(6, [1.0], random_state=5)), new
    )
    # a scalar is a stress now (it raised before: Z was missing)
    np.testing.assert_array_equal(
        _quiet(lambda: model.random(6, 1, random_state=5)), new
    )
    # two positional arguments after size: the old (random_state, Z)
    np.testing.assert_array_equal(
        _deprecated(lambda: model.random(6, 5, [1.0]), "old order"), new
    )
    # one, with Z by name: the old random_state
    np.testing.assert_array_equal(
        _deprecated(lambda: model.random(6, 5, Z=[1.0]), "old order"), new
    )
    with pytest.raises(TypeError, match="multiple values"):
        model.random(6, 5, [1.0], Z=[1.0])


# ``induced_life`` and both ``predict_rul`` took ``Z`` after the seed and
# the level; ``Z`` now comes straight after the query and the rest is
# keyword-only, so any positional argument after the query is the old
# order, read with its old meaning and a warning.


def test_induced_life_old_positional_order():
    model = _model("DegradationAnalysis[linear]")
    old = _deprecated(lambda: model.induced_life(200, 3), "old order")
    new = _quiet(lambda: model.induced_life(200, random_state=3))
    np.testing.assert_array_equal(old.samples, new.samples)


def test_path_predict_rul_old_positional_order():
    model = _model("DegradationAnalysis[linear]")
    x, y = [1.0, 2.0, 3.0], [1.0, 1.5, 2.0]
    old = _deprecated(
        lambda: model.predict_rul(x, y, 0.1, 500, 4), "old order"
    )
    new = _quiet(
        lambda: model.predict_rul(
            x, y, alpha_ci=0.1, n_samples=500, random_state=4
        )
    )
    assert old.rul == new.rul
    np.testing.assert_array_equal(old.rul_interval, new.rul_interval)


@pytest.mark.parametrize("case", ["WienerProcess", "GammaProcess"])
def test_process_predict_rul_old_positional_order(case):
    model = _model(case)
    old = _deprecated(lambda: model.predict_rul(1.0, 0.1), "old order")
    new = _quiet(lambda: model.predict_rul(1.0, alpha_ci=0.1))
    assert old.rul == new.rul
    assert old.rul_interval == new.rul_interval
