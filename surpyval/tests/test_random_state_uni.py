"""
#389: every ``random`` of a univariate model or distribution takes a
keyword ``random_state`` (principle 19): ``None`` is numpy's global
stream, as before, and an int or a generator is a stream of its own.
Also, the positional form of the query methods renamed in #422 (their
old names were removed in v0.22; see test_removed_arguments.py).
"""

import warnings

import numpy as np
import pytest
from scipy.stats import uniform

import surpyval as sp
from surpyval.tests.conformance.registry import CASE_BY_NAME, CASES, fitted

X = np.array([5.0, 10.0, 20.0])


def _quiet(call):
    """``call()``, which must not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return call()


def _model(name):
    return fitted(CASE_BY_NAME[name])


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
