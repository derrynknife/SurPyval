"""
Past the quantile limit, ``conditional_gaps`` inverts the cumulative
hazard by Newton's method with the hazard rate, kept inside a bracket.
It must agree with the bracketing solver it replaced (to rounding), with
mpmath roots, and take few evaluations of ``H`` and ``h`` per draw.

The mpmath roots are from ``scripts/reference/conditional_gaps_mpmath.py``
(50 digits), typed in here.
"""

from collections import Counter
from typing import Any

import numpy as np
import pytest

import surpyval as surv
from surpyval.recurrent.renewal import renewal_model
from surpyval.recurrent.renewal.renewal_model import (
    _bracket_and_solve,
    _invert_cumulative_hazard,
    _lifetime_functions,
    conditional_gaps,
)

# (lifetime, v, u, x) with H(x) = H(v) - log(u), from
# scripts/reference/conditional_gaps_mpmath.py.
ROOTS = [
    ("mixture", 5000.0, 0.9, 5006.083250453805),
    ("mixture", 5000.0, 0.3, 5068.441559907095),
    ("mixture", 5000.0, 0.01, 5250.267639985795),
    ("lognormal", 40000.0, 0.9, 40474.95812352477),
    ("lognormal", 40000.0, 0.3, 45712.40643137095),
    ("lognormal", 40000.0, 0.01, 65642.0900602999),
    ("lognormal", 100000.0, 0.5, 106907.69680713824),
    ("normal", 40.0, 0.9, 40.03128511027553),
    ("normal", 40.0, 0.3, 40.3556155469713),
    ("normal", 40.0, 0.01, 41.33881869606884),
]


def _mixture() -> surv.MixtureModel:
    model = surv.MixtureModel(surv.Weibull, m=2)
    model.params = np.array([[60.0, 0.8], [2000.0, 3.5]])
    model.w = np.array([0.25, 0.75])
    return model


class _WeibullWithoutRate:
    """A lifetime with ``Hf``, ``qf`` and ``df`` but no ``hf``: the
    hazard rate comes from ``exp(log f + H)``."""

    dist = surv.Weibull

    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha, self.beta = alpha, beta

    def Hf(self, x: Any) -> np.ndarray:
        return (np.asarray(x, dtype=float) / self.alpha) ** self.beta

    def df(self, x: Any) -> Any:
        return surv.Weibull.df(x, self.alpha, self.beta)

    def qf(self, p: Any) -> Any:
        return surv.Weibull.qf(p, self.alpha, self.beta)


class _WeibullWithLogDensity(_WeibullWithoutRate):
    """As ``_WeibullWithoutRate``, with ``log_df``, which stays finite
    where the density underflows."""

    def log_df(self, x: Any) -> Any:
        return surv.Weibull.log_df(x, self.alpha, self.beta)


LIFETIMES = {
    "mixture": (_mixture(), 5000.0),
    "lognormal": (surv.LogNormal.from_params([5, 0.8]), 40000.0),
    "lognormal far": (surv.LogNormal.from_params([5, 0.8]), 100000.0),
    "normal": (surv.Normal.from_params([10, 3]), 40.0),
    "gumbel": (surv.Gumbel.from_params([10, 3]), 40.0),
    "gamma": (surv.Gamma.from_params([2, 0.1]), 500.0),
    "weibull decreasing": (surv.Weibull.from_params([10, 0.5]), 4000.0),
    "weibull increasing": (surv.Weibull.from_params([10, 2.0]), 300.0),
    "loglogistic": (surv.LogLogistic.from_params([10, 3]), 1e5),
    "exponential": (surv.Exponential.from_params([0.1]), 300.0),
    # H(v) = 32, where the density (1e-14) has not underflowed.
    "no hf": (_WeibullWithoutRate(10.0, 2.5), 40.0),
    # H(v) = 1e4: the density underflows, its log does not.
    "no hf, log_df": (_WeibullWithLogDensity(10.0, 2.5), 400.0),
    "offset": (surv.Weibull.from_params([10, 2.0], gamma=5.0), 300.0),
}

U = np.random.default_rng(20261105).uniform(size=50)


def _targets(lifetime: Any, v: float) -> tuple:
    hazard, _, both = _lifetime_functions(lifetime)
    ages = np.full(U.size, v)
    return hazard, both, ages, hazard(ages) - np.log(U)


@pytest.mark.parametrize("name,v,u,x", ROOTS)
def test_newton_draw_matches_the_mpmath_root(name, v, u, x):
    lifetime = {
        "mixture": _mixture(),
        "lognormal": surv.LogNormal.from_params([5, 0.8]),
        "normal": surv.Normal.from_params([10, 3]),
    }[name]
    gap = conditional_gaps(lifetime, np.array([v]), np.array([u]))[0]
    # To 1e-13 relative in the age reached. The draw starts from the
    # double H(v), whose own rounding moves the root by up to about
    # eps * H / (x h): 3e-15 for the LogNormal here.
    assert v + gap == pytest.approx(x, rel=1e-13)


@pytest.mark.parametrize("name", sorted(LIFETIMES))
def test_newton_matches_the_bracketing_solver(name):
    lifetime, v = LIFETIMES[name]
    hazard, both, ages, target = _targets(lifetime, v)
    assert both is not None
    assert np.all(target > renewal_model._QF_HAZARD_LIMIT)
    old = _bracket_and_solve(hazard, ages, target)
    together = _invert_cumulative_hazard(hazard, ages, target, both=both)
    one_by_one = np.array(
        [
            _invert_cumulative_hazard(
                hazard, ages[k : k + 1], target[k : k + 1], both=both
            )[0]
            for k in range(U.size)
        ]
    )
    # Both solve to rounding: 1e-13 relative leaves room for an H as
    # flat as the LogNormal's far tail (x h / H about 0.1), where an eps
    # in H is 1e-15 in x.
    np.testing.assert_allclose(together, old, rtol=1e-13)
    # The one-draw path takes exactly the vectorised path's steps.
    np.testing.assert_array_equal(one_by_one, together)


def _count(monkeypatch: Any) -> Counter:
    calls: Counter = Counter()
    original = renewal_model._lifetime_functions

    def counting(lifetime: Any) -> tuple:
        hazard, quantile, both = original(lifetime)

        def counted_hazard(x: Any) -> Any:
            calls["H"] += 1
            return hazard(x)

        def counted_both(x: Any, H: Any = None) -> Any:
            calls["both"] += 1
            assert both is not None
            return both(x, H)

        return counted_hazard, quantile, counted_both

    monkeypatch.setattr(renewal_model, "_lifetime_functions", counting)
    return calls


@pytest.mark.parametrize("name", sorted(LIFETIMES))
def test_newton_takes_few_evaluations(name, monkeypatch):
    lifetime, v = LIFETIMES[name]
    calls = _count(monkeypatch)
    for u in U:
        conditional_gaps(lifetime, np.array([v]), np.array([u]))
    # Per draw: H at v, then the hazard pair at v and at each Newton
    # point. The bracketing solver took 12 to 15 evaluations of H.
    assert calls["H"] == U.size
    assert calls["both"] <= 5 * U.size
    assert max(calls.values()) / U.size <= 5
    per_draw = (calls["H"] + calls["both"]) / U.size
    assert per_draw <= 6


def test_newton_draws_are_deterministic_and_vectorise():
    lifetime = _mixture()
    ages = np.full(U.size, 5000.0)
    first = conditional_gaps(lifetime, ages, U)
    again = conditional_gaps(lifetime, ages, U)
    np.testing.assert_array_equal(first, again)
    one = [
        conditional_gaps(lifetime, np.array([5000.0]), np.array([u]))[0]
        for u in U
    ]
    np.testing.assert_array_equal(first, one)


def test_a_lifetime_without_a_rate_is_bracketed():
    # A non-parametric lifetime's hf is the jumps of a step function, not
    # a rate: no Newton steps.
    km = surv.KaplanMeier.fit([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _lifetime_functions(km)[2] is None


def test_past_a_bounded_support_newton_hands_over_to_bracketing():
    lifetime = surv.Beta.from_params([2.0, 3.0])
    ages = np.array([0.999, 0.999999])
    hazard, both, _, _ = _targets(lifetime, 0.5)
    target = hazard(ages) - np.log(np.array([0.5, 1e-30]))
    new = _invert_cumulative_hazard(hazard, ages, target, both=both)
    old = _bracket_and_solve(hazard, ages, target)
    np.testing.assert_allclose(new, old, rtol=1e-13)
    assert np.all((new > ages) & (new <= 1.0))


@pytest.mark.parametrize(
    "lifetime,age",
    [
        (_WeibullWithoutRate(10.0, 2.5), 40.0),
        (_WeibullWithLogDensity(10.0, 2.5), 400.0),
    ],
)
def test_newton_without_hf_matches_the_weibull_closed_form(lifetime, age):
    alpha, beta = 10.0, 2.5
    ages = np.full(U.size, age)
    gaps = conditional_gaps(lifetime, ages, U)
    target = (ages / alpha) ** beta - np.log(U)
    exact = alpha * target ** (1 / beta)
    np.testing.assert_allclose(ages + gaps, exact, rtol=1e-14)
