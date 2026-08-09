"""
A method that every distribution implements must have one signature.

Not a style rule. A distribution is reached through a
``ParametricFitter`` reference all over the package -- ``fit_best``
iterates a list of them, ``Discretize`` and ``MixtureModel`` wrap one,
the regression fitters hold one as ``self.dist`` -- and code written
against that reference has to work for every member. When the same
positional slot is called ``p`` on one distribution and ``u`` on
another, a keyword call is correct for a subset and a ``TypeError`` for
the rest, with nothing to say which until it runs. That is what made
the narrow ``from_params`` overrides on ``Bernoulli``, ``Binomial`` and
``ExactEventTime`` a defect rather than a naming preference (#257), and
the same argument applies to every shared method.

These tests read the signatures rather than asserting a list of names,
so a distribution added later is covered without touching this file.
"""

import ast
import inspect
import pathlib
from collections import defaultdict

import numpy as np
import pytest

import surpyval
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter

DIST_DIR = pathlib.Path(surpyval.__file__).parent / (
    "univariate/parametric/distributions"
)

# The order of a moment, and the probability a quantile is taken at.
# ``m`` and ``u`` rather than ``n`` and ``p`` because ``n`` is
# Binomial's trial count and ``p`` is a parameter of Bernoulli,
# Binomial, Geometric and NegativeBinomial -- the two names that would
# otherwise read as the obvious choice are already taken.
CANONICAL_FIRST_ARG = {
    "moment": "m",
    "qf": "u",
    "mpp_x_transform": "x",
}


def _signatures(method):
    """{module stem: [parameter names]} for every implementation."""
    out = {}
    for path in sorted(DIST_DIR.glob("*.py")):
        if path.stem == "__init__":
            continue
        tree = ast.parse(path.read_text())
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                if fn.name != method:
                    continue
                args = fn.args.posonlyargs + fn.args.args
                out[path.stem] = [a.arg for a in args][1:]  # drop self
    return out


@pytest.mark.parametrize("method, first", sorted(CANONICAL_FIRST_ARG.items()))
def test_shared_method_first_argument_is_uniform(method, first):
    sigs = _signatures(method)
    assert len(sigs) > 5, f"expected many implementations of {method}"
    wrong = {
        mod: params[0]
        for mod, params in sigs.items()
        if params and params[0] != first
    }
    assert not wrong, (
        f"{method}'s first argument must be {first!r} everywhere; "
        f"these differ: {wrong}"
    )


def test_mpp_x_transform_takes_only_x():
    # It used to take a vestigial ``gamma`` on eleven distributions and
    # not on the other four. No caller ever passed it -- the MPP fitter
    # subtracts the offset from x before calling (fitters/mpp.py) -- so
    # a caller that did pass it would have subtracted twice.
    sigs = _signatures("mpp_x_transform")
    extra = {mod: params for mod, params in sigs.items() if params != ["x"]}
    assert not extra, f"mpp_x_transform must take x alone: {extra}"


def test_moment_order_is_a_scalar_everywhere():
    # The docstrings used to promise "integer or numpy array of
    # integers" on nine distributions; only six delivered, and the
    # other three raised. The contract is now a scalar order, which
    # every implementation honours and every caller passes.
    checked = 0
    for name in dir(surpyval):
        dist = getattr(surpyval, name)
        if not isinstance(dist, ParametricFitter):
            continue
        if not hasattr(dist, "moment"):
            continue
        params = list(inspect.signature(type(dist).moment).parameters)
        assert params[1] == "m", f"{name}.moment first arg is {params[1]!r}"
        checked += 1
    assert checked > 15, f"only checked {checked} distributions"


def test_mpp_override_and_supports_mpp_do_not_contradict():
    # ``mpp`` is an opt-in hook: fitters/mpp.py dispatches on
    # hasattr(dist, "mpp") and otherwise runs the generic probability
    # plotting path. Independently, fit() refuses how="MPP" outright
    # when supports_mpp is False.
    #
    # A distribution that sets supports_mpp = False *and* defines mpp is
    # carrying dead code -- the guard in fit() raises before dispatch can
    # reach the method. Beta and Beta4 each had one whose whole body was
    # `raise NotImplementedError`, which read as the thing doing the
    # refusing when it could never run.
    #
    # The converse is not an error: a distribution can support MPP and
    # use the generic path, which is what most of them do.
    contradictory = []
    for name in dir(surpyval):
        dist = getattr(surpyval, name)
        if not isinstance(dist, ParametricFitter):
            continue
        defines = any("mpp" in k.__dict__ for k in type(dist).__mro__)
        if defines and not dist.supports_mpp:
            contradictory.append(name)
    assert not contradictory, (
        "these declare supports_mpp = False but still define mpp, which "
        f"can never run: {contradictory}"
    )


def test_every_distribution_refusing_mpp_raises_the_same_way():
    # One refusal, one message, one place -- rather than each
    # distribution inventing its own NotImplementedError.
    #
    # Restricted to distributions whose fit() takes a ``how`` at all.
    # Bernoulli, Binomial and ExactEventTime override fit() with a
    # narrow signature that has no ``how``, so asking them for MPP is a
    # TypeError from argument binding rather than this refusal. That is
    # the separate fit() divergence, not this one.
    refusers = [
        name
        for name in dir(surpyval)
        if isinstance(getattr(surpyval, name), ParametricFitter)
        and not getattr(surpyval, name).supports_mpp
        and "how"
        in inspect.signature(type(getattr(surpyval, name)).fit).parameters
    ]
    assert len(refusers) > 3, f"expected several refusers, got {refusers}"
    for name in refusers:
        dist = getattr(surpyval, name)
        with pytest.raises(ValueError, match="does not work"):
            dist.fit(np.array([1.0, 2.0, 3.0, 4.0]), how="MPP")


def _param_names_by_module():
    """{module stem: set of that distribution's parameter names}."""
    out = defaultdict(set)
    for name in dir(surpyval):
        dist = getattr(surpyval, name)
        if isinstance(dist, ParametricFitter):
            stem = type(dist).__module__.rsplit(".", 1)[-1]
            out[stem].update(getattr(dist, "param_names", []) or [])
    return out


def test_no_shared_method_diverges_in_its_data_argument():
    # A guard for methods not yet in CANONICAL_FIRST_ARG: any method
    # implemented by five or more distributions must agree on the name
    # of its leading *data* argument. Catches the next occurrence of
    # this bug without needing the method listed above.
    #
    # Parameter names are skipped, because those are the one thing that
    # is legitimately per-distribution: ``Weibull.mean(alpha, beta)``
    # and ``Poisson.mean(mu)`` are not a divergence, they are what the
    # distributions are. What must agree is everything else -- the x a
    # function is evaluated at, the u a quantile is taken at, the m of a
    # moment.
    params_by_mod = _param_names_by_module()
    leading = defaultdict(dict)
    for path in sorted(DIST_DIR.glob("*.py")):
        if path.stem == "__init__":
            continue
        own = params_by_mod.get(path.stem, set())
        tree = ast.parse(path.read_text())
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                if fn.name.startswith("__"):
                    # Constructors: the wrappers (Discretize,
                    # CustomDistribution) take a distribution rather
                    # than a name, which is the point of them.
                    continue
                args = fn.args.posonlyargs + fn.args.args
                names = [a.arg for a in args][1:]
                data = [n for n in names if n not in own]
                if data:
                    leading[fn.name][path.stem] = data[0]

    diverging = {}
    for method, by_mod in leading.items():
        if len(by_mod) < 5:
            continue
        firsts = set(by_mod.values())
        if len(firsts) > 1:
            diverging[method] = {
                first: sorted(m for m, f in by_mod.items() if f == first)
                for first in sorted(firsts)
            }

    assert not diverging, (
        "these shared methods disagree on their leading data argument: "
        f"{diverging}"
    )
