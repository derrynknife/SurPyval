"""
#422 in the univariate models: one name per option (principle 21), and
#389: every ``random`` takes ``random_state`` (principle 19).

Each renamed argument still works under its old name until v0.22.0, with a
``DeprecationWarning`` pointing at the caller, and gives the answer the new
name gives. Every ``random`` of a univariate model or distribution takes a
keyword ``random_state``: ``None`` is numpy's global stream, as before, and
an int or a generator is a stream of its own.
"""

import warnings

import numpy as np
import pytest
from scipy.stats import uniform

import surpyval as sp
from surpyval.tests.conformance.registry import CASE_BY_NAME, CASES, fitted

X = np.array([5.0, 10.0, 20.0])
P = np.array([0.1, 0.5, 0.9])


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


# (label, case, old call, new call, old name)
RENAMES = [
    (
        "Parametric.cb(t=)",
        "Weibull",
        lambda m: m.cb(t=X, on="Hf", bound="lower"),
        lambda m: m.cb(x=X, on="Hf", bound="lower"),
        "'t'",
    ),
    *[
        (
            f"RoystonParmarModel.{fn}(t=)",
            "RoystonParmar",
            lambda m, fn=fn: getattr(m, fn)(t=X),
            lambda m, fn=fn: getattr(m, fn)(x=X),
            "'t'",
        )
        for fn in ("sf", "ff", "df", "hf", "Hf", "cb")
    ],
    (
        "RoystonParmarModel.qf(q=)",
        "RoystonParmar",
        lambda m: m.qf(q=P),
        lambda m: m.qf(p=P),
        "'q'",
    ),
    (
        "NeverOccurs.qf(u=)",
        "NeverOccurs",
        lambda m: m.qf(u=P),
        lambda m: m.qf(p=P),
        "'u'",
    ),
    (
        "InstantlyOccurs.qf(u=)",
        "InstantlyOccurs",
        lambda m: m.qf(u=P),
        lambda m: m.qf(p=P),
        "'u'",
    ),
    *[
        (
            f"{case}.bootstrap_cb(B=)",
            case,
            lambda m: m.bootstrap_cb(X, B=20, random_state=1),
            lambda m: m.bootstrap_cb(X, n_boot=20, random_state=1),
            "'B'",
        )
        for case in ("KaplanMeier", "Turnbull")
    ],
]


@pytest.mark.parametrize(
    "case, old, new, name",
    [pytest.param(*row[1:], id=row[0]) for row in RENAMES],
)
def test_old_name_warns_and_agrees(case, old, new, name):
    model = _model(case)
    got = _deprecated(lambda: old(model), name)
    np.testing.assert_array_equal(got, _quiet(lambda: new(model)))


def test_both_names_is_an_error():
    model = _model("Weibull")
    with pytest.raises(ValueError, match="pass 'x' only"):
        model.cb(t=X, x=X)
    with pytest.raises(ValueError, match="pass 'n_boot' only"):
        _model("KaplanMeier").bootstrap_cb(X, B=5, n_boot=5)


def test_positional_calls_are_unchanged():
    model = _model("RoystonParmar")
    np.testing.assert_array_equal(_quiet(lambda: model.cb(X)), model.cb(x=X))
    weibull = _model("Weibull")
    np.testing.assert_array_equal(
        _quiet(lambda: weibull.cb(X, "ff")), weibull.cb(x=X, on="ff")
    )


# ---------------------------------------------------------------------------
# random_state (#389)
# ---------------------------------------------------------------------------
_UNIVARIATE = [
    c.name
    for c in CASES
    if c.model_class
    in (
        "surpyval.Parametric",
        "surpyval.NeverOccurs",
        "surpyval.InstantlyOccurs",
        "surpyval.NonParametric",
        "surpyval.MixtureModel",
        "surpyval.RoystonParmarModel",
    )
]


def _draws(case):
    """The draws of a case's model, and of its distribution when it has
    one, each as ``draw(random_state)``."""
    model = _model(case)
    out = [lambda s: model.random(15, random_state=s)]
    if isinstance(model, sp.Parametric):
        out.append(lambda s: model.random_data(15, random_state=s))
        out.append(
            lambda s: model.dist.random(15, *model.params, random_state=s)
        )
    return out


def _flat(value):
    if isinstance(value, tuple):
        return np.concatenate([np.ravel(v) for v in value])
    return np.ravel(value)


@pytest.mark.parametrize("case", _UNIVARIATE)
def test_random_state_is_a_stream_of_its_own(case):
    for draw in _draws(case):
        np.random.seed(1)
        a = _flat(_quiet(lambda: draw(7)))
        after = np.random.uniform(size=4)
        np.random.seed(1)
        untouched = np.random.uniform(size=4)
        np.random.seed(2)
        b = _flat(draw(7))
        np.testing.assert_array_equal(a, b)
        # It neither depends on nor advances the global stream, and an
        # int seed is default_rng(seed).
        np.testing.assert_array_equal(after, untouched)
        np.testing.assert_array_equal(_flat(draw(np.random.default_rng(7))), a)


@pytest.mark.parametrize("case", _UNIVARIATE)
def test_no_random_state_is_the_global_stream(case):
    for draw in _draws(case):
        np.random.seed(0)
        a = _flat(draw(None))
        np.random.seed(0)
        np.testing.assert_array_equal(_flat(draw(None)), a)


def test_no_random_state_keeps_the_old_draws():
    # random_state=None is the global stream exactly as before: the same
    # numbers as qf of numpy's global uniforms.
    np.random.seed(1)
    got = sp.Weibull.random(5, 3, 4)
    np.random.seed(1)
    want = sp.Weibull.qf(uniform.rvs(size=5), 3, 4)
    np.testing.assert_array_equal(got, want)
    model = sp.Weibull.from_params([10, 3])
    np.random.seed(1)
    got = model.random(4)
    np.random.seed(1)
    np.testing.assert_array_equal(got, model.qf(np.random.random_sample(4)))


def test_random_state_is_keyword_only():
    # random(size, *params): a trailing number is a parameter, not a seed.
    with pytest.raises(TypeError):
        _model("Weibull").random(5, None, None, 7)
    with pytest.raises(TypeError):
        _model("RoystonParmar").random(5, 7)


def test_binomial_draw_takes_a_fitted_n():
    # A fitted Binomial holds n as a float; its draw raised a TypeError
    # ("Cannot cast scalar from dtype('float64') to dtype('int64')").
    model = _model("Binomial")
    np.random.seed(3)
    got = sp.Binomial.random(6, *model.params)
    np.random.seed(3)
    want = sp.Binomial.random(6, int(model.params[0]), model.params[1])
    np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError, match="whole number"):
        sp.Binomial.random(6, 5.5, 0.5)
