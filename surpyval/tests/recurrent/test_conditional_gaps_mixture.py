"""
``conditional_gaps`` for a mixture: the age it returns is the root of
``H(v + x) = H(v) - log(u)`` to the last bits or so, from new to past the
0.999 quantile and for ``u`` from 1e-9 to 1 - 1e-9; draws for other
lifetimes are as they were.

A mixture's quantile function is a root search on its failure function,
so inverting it lost about ``eps * exp(H)`` of the age (2e-11 relative at
``u = 1e-6`` for a log-normal mixture); its draws now take Newton's steps
on ``H`` from a bound found from the components.

The roots are from ``scripts/reference/conditional_gaps_mpmath.py`` (50
digits, by bisection), typed in here.
"""

import warnings
from typing import Any

import numpy as np
import pytest

import surpyval as surv
from surpyval.recurrent.renewal import renewal_model
from surpyval.recurrent.renewal.renewal_model import (
    _QF_HAZARD_LIMIT,
    _invert_cumulative_hazard,
    _lifetime_functions,
    _mixture_start,
    conditional_gaps,
)

# (name, v, u, x) with H(x) = H(v) - log(u), from
# scripts/reference/conditional_gaps_mpmath.py.
MIXTURE_DRAWS = [
    ("weibull2", 0.0, 1e-09, 4736.159313607215),
    ("weibull2", 0.0, 1e-06, 4209.5727666113235),
    ("weibull2", 0.0, 0.001, 3432.118087223781),
    ("weibull2", 0.0, 0.1, 2443.2041240551102),
    ("weibull2", 0.0, 0.5, 1545.3146686122122),
    ("weibull2", 0.0, 0.9, 25.91139397175796),
    ("weibull2", 0.0, 0.999999, 1.0733153125278793e-05),
    ("weibull2", 0.0, 0.999999999, 1.9086496863371715e-09),
    ("weibull2", 25.91, 1e-09, 4743.1229530437995),
    ("weibull2", 25.91, 1e-06, 4218.913917174642),
    ("weibull2", 25.91, 0.001, 3447.6362749505),
    ("weibull2", 25.91, 0.1, 2479.0427896561478),
    ("weibull2", 25.91, 0.5, 1650.73946732524),
    ("weibull2", 25.91, 0.9, 93.56240525102055),
    ("weibull2", 25.91, 0.999999, 25.9103804210939),
    ("weibull2", 25.91, 0.999999999, 25.910000380419387),
    ("weibull2", 1545.31, 1e-09, 4781.51191682557),
    ("weibull2", 1545.31, 1e-06, 4270.103056801896),
    ("weibull2", 1545.31, 0.001, 3531.1658185504148),
    ("weibull2", 1545.31, 0.1, 2658.5571618740855),
    ("weibull2", 1545.31, 0.5, 2054.467900959253),
    ("weibull2", 1545.31, 0.9, 1650.738891977747),
    ("weibull2", 1545.31, 0.999999, 1545.311088919251),
    ("weibull2", 1545.31, 0.999999999, 1545.3100010889195),
    ("weibull2", 3432.12, 1e-09, 5147.064905966794),
    ("weibull2", 3432.12, 1e-06, 4736.160168682794),
    ("weibull2", 3432.12, 0.001, 4209.5739146983915),
    ("weibull2", 3432.12, 0.1, 3737.661476222533),
    ("weibull2", 3432.12, 0.5, 3531.168191476409),
    ("weibull2", 3432.12, 0.9, 3447.638702958098),
    ("weibull2", 3432.12, 0.999999, 3432.1201481258904),
    ("weibull2", 3432.12, 0.999999999, 3432.1200001481257),
    ("weibull3", 0.0, 1e-09, 4246.901360421401),
    ("weibull3", 0.0, 1e-06, 3819.739960400221),
    ("weibull3", 0.0, 0.001, 3180.7015455275073),
    ("weibull3", 0.0, 0.1, 2313.9285140872785),
    ("weibull3", 0.0, 0.5, 1307.0160984536976),
    ("weibull3", 0.0, 0.9, 39.16272077350573),
    ("weibull3", 0.0, 0.999999, 1.0000100001455279e-09),
    ("weibull3", 0.0, 0.999999999, 9.99999953436137e-16),
    ("weibull3", 39.16, 1e-09, 4253.019149693452),
    ("weibull3", 39.16, 1e-06, 3827.281503549262),
    ("weibull3", 39.16, 0.001, 3193.7180343562786),
    ("weibull3", 39.16, 0.1, 2347.218717593569),
    ("weibull3", 39.16, 0.5, 1464.747831796245),
    ("weibull3", 39.16, 0.9, 153.5210846866124),
    ("weibull3", 39.16, 0.999999, 39.161038994525036),
    ("weibull3", 39.16, 0.999999999, 39.160001038989755),
    ("weibull3", 1307.02, 1e-09, 4288.1345903419515),
    ("weibull3", 1307.02, 1e-06, 3868.5628984265454),
    ("weibull3", 1307.02, 0.001, 3263.5690674287885),
    ("weibull3", 1307.02, 0.1, 2511.0630759026126),
    ("weibull3", 1307.02, 0.5, 1934.596881235657),
    ("weibull3", 1307.02, 0.9, 1464.753939071742),
    ("weibull3", 1307.02, 0.999999, 1307.0217893212946),
    ("weibull3", 1307.02, 0.999999999, 1307.020001789324),
    ("weibull3", 3180.7, 1e-09, 6415.290415384628),
    ("weibull3", 3180.7, 1e-06, 4246.900640606952),
    ("weibull3", 3180.7, 0.001, 3819.739067796428),
    ("weibull3", 3180.7, 0.1, 3434.8178343116174),
    ("weibull3", 3180.7, 0.5, 3263.5673857505517),
    ("weibull3", 3180.7, 0.9, 3193.716829198365),
    ("weibull3", 3180.7, 0.999999, 3180.7001243061354),
    ("weibull3", 3180.7, 0.999999999, 3180.700000124306),
    ("lognormal2", 0.0, 1e-09, 24273.349066847957),
    ("lognormal2", 0.0, 1e-06, 5318.568785237435),
    ("lognormal2", 0.0, 0.001, 680.0996711434741),
    ("lognormal2", 0.0, 0.1, 64.12958999097435),
    ("lognormal2", 0.0, 0.5, 6.918722978891898),
    ("lognormal2", 0.0, 0.9, 1.8334197309970828),
    ("lognormal2", 0.0, 0.999999, 0.07585288325230521),
    ("lognormal2", 0.0, 0.999999999, 0.01662023605098214),
    ("lognormal2", 1.83, 1e-09, 24780.473829450573),
    ("lognormal2", 1.83, 1e-06, 5458.069377314318),
    ("lognormal2", 1.83, 0.001, 707.0390508691403),
    ("lognormal2", 1.83, 0.1, 69.63743489114096),
    ("lognormal2", 1.83, 0.5, 9.103582737747077),
    ("lognormal2", 1.83, 0.9, 2.44049792852389),
    ("lognormal2", 1.83, 0.999999, 1.8300063248851979),
    ("lognormal2", 1.83, 0.999999999, 1.8300000063248916),
    ("lognormal2", 6.92, 1e-09, 27801.27832026527),
    ("lognormal2", 6.92, 1e-06, 6299.228695727886),
    ("lognormal2", 6.92, 0.001, 873.7558538857273),
    ("lognormal2", 6.92, 0.1, 105.5979265273548),
    ("lognormal2", 6.92, 0.5, 25.857761572875265),
    ("lognormal2", 6.92, 0.9, 9.117691692614645),
    ("lognormal2", 6.92, 0.999999, 6.920018195182213),
    ("lognormal2", 6.92, 0.999999999, 6.920000018195144),
    ("lognormal2", 680.1, 1e-09, 85432.58048123188),
    ("lognormal2", 680.1, 1e-06, 24273.35529386928),
    ("lognormal2", 680.1, 0.001, 5318.57049499515),
    ("lognormal2", 680.1, 0.1, 1488.461050517903),
    ("lognormal2", 680.1, 0.5, 873.7347156099964),
    ("lognormal2", 680.1, 0.9, 707.1805368088162),
    ("lognormal2", 680.1, 0.999999, 680.1002532225554),
    ("lognormal2", 680.1, 0.999999999, 680.1000002532224),
]


def _mixture(dist: Any, params: Any, w: Any) -> surv.MixtureModel:
    model = surv.MixtureModel(dist, m=len(w))
    model.params = np.array(params, dtype=float)
    model.w = np.array(w, dtype=float)
    return model


MIXTURES = {
    "weibull2": _mixture(
        surv.Weibull, [[60.0, 0.8], [2000.0, 3.5]], [0.25, 0.75]
    ),
    "weibull3": _mixture(
        surv.Weibull,
        [[10.0, 0.5], [300.0, 1.5], [2000.0, 4.0]],
        [0.1, 0.3, 0.6],
    ),
    "lognormal2": _mixture(
        surv.LogNormal, [[1.0, 0.5], [3.0, 1.2]], [0.4, 0.6]
    ),
}

#: The age found is the root to this much of itself.
RTOL = 1e-13


def _cases(name: str) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    rows = [row for row in MIXTURE_DRAWS if row[0] == name]
    v, u, x = (np.array([row[k] for row in rows]) for k in (1, 2, 3))
    return v, u, x


@pytest.mark.parametrize("name", sorted(MIXTURES))
def test_a_mixture_s_draws_at_once_are_the_roots(name):
    v, u, x = _cases(name)
    age = v + conditional_gaps(MIXTURES[name], v, u)
    np.testing.assert_allclose(age, x, rtol=RTOL, atol=0)


@pytest.mark.parametrize("name", sorted(MIXTURES))
def test_a_mixture_s_draws_one_at_a_time_are_the_roots(name):
    # One draw a call (as the simulations make them) runs the steps on
    # Python floats.
    v, u, x = _cases(name)
    age = np.array(
        [
            vk + conditional_gaps(MIXTURES[name], np.array([vk]), [uk])[0]
            for vk, uk in zip(v, u)
        ]
    )
    np.testing.assert_allclose(age, x, rtol=RTOL, atol=0)


@pytest.mark.parametrize("name", sorted(MIXTURES))
def test_the_start_is_at_or_below_the_root(name):
    # Newton's steps begin at a bound below the root, up to the rounding
    # of the components' quantiles where exp(-target) is small (relative
    # error about eps * exp(target), 1e-7 at u = 1e-9 from the 0.999
    # quantile); the solver finds which side the start is on from H.
    v, u, x = _cases(name)
    model = MIXTURES[name]
    target = np.asarray(model.Hf(v), dtype=float) - np.log(u)
    start = _mixture_start(model, v, target)
    assert np.all(start >= v)
    assert np.all(start <= x * (1 + 1e-6))


def test_a_mixture_takes_few_evaluations_a_draw(monkeypatch):
    model = MIXTURES["weibull2"]
    v, u, _ = _cases("weibull2")
    calls = []
    original = model._hazard_and_rate

    def counted(x: Any) -> Any:
        calls.append(np.size(x))
        return original(x)

    monkeypatch.setattr(model, "_hazard_and_rate", counted)
    for vk, uk in zip(v, u):
        conditional_gaps(model, np.array([vk]), [uk])
    assert len(calls) <= 3 * len(v)


def test_a_mixture_s_ends():
    model = MIXTURES["weibull2"]
    gaps = conditional_gaps(
        model, np.array([100.0, 100.0, 0.0]), np.array([1.0, 0.0, 1.0])
    )
    assert gaps[0] == 0.0
    assert gaps[1] == np.inf
    assert gaps[2] == 0.0


def _old_draw(lifetime: Any, v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The draw before mixtures took Newton's steps throughout: the
    quantile function below ``_QF_HAZARD_LIMIT``, Newton from ``v``
    beyond it."""
    hazard, quantile, both = _lifetime_functions(lifetime)
    H_v = hazard(v)
    target = H_v - np.log(u)
    easy = target <= _QF_HAZARD_LIMIT
    x = np.array(v, dtype=float)
    x[easy] = quantile(-np.expm1(-target[easy]))
    x[~easy] = _invert_cumulative_hazard(
        hazard, v[~easy], target[~easy], both=both, H_v=H_v[~easy]
    )
    return np.maximum(x - v, 0.0)


@pytest.mark.parametrize(
    "lifetime",
    [
        surv.Weibull.from_params([2000.0, 3.5]),
        surv.LogNormal.from_params([3.0, 1.2]),
        surv.Gamma.from_params([2.0, 0.01]),
    ],
    ids=["weibull", "lognormal", "gamma"],
)
def test_other_lifetimes_draw_as_before(lifetime):
    rng = np.random.default_rng(20261120)
    u = 10.0 ** rng.uniform(-12.0, 0.0, size=400)
    v = rng.choice([0.0, 50.0, 2000.0, 6000.0], size=400)
    expected = _old_draw(lifetime, v, u)
    np.testing.assert_array_equal(conditional_gaps(lifetime, v, u), expected)
    # One at a time against the old draw one at a time: numpy's loops
    # over many values may round differently from its one-value path (by
    # an ulp, on x86-64 Linux), so each is held to its own kind.
    one = [
        conditional_gaps(lifetime, v[k : k + 1], u[k : k + 1])[0]
        for k in range(40)
    ]
    old_one = [
        _old_draw(lifetime, v[k : k + 1], u[k : k + 1])[0] for k in range(40)
    ]
    np.testing.assert_array_equal(one, old_one)


def test_a_hazard_that_levels_off_below_the_target_never_fails():
    # With a limited-failure population H tops out at -log(1 - p): a draw
    # past it is an event that never comes (it was NaN beyond the
    # quantile limit).
    model = surv.Weibull.from_params([100.0, 2.0], lfp_p=0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gaps = conditional_gaps(
            model, np.array([0.0, 50.0, 200.0]), np.array([1e-9] * 3)
        )
    assert np.all(gaps == np.inf)
    # Below the level it is the root, as before.
    gap = conditional_gaps(model, np.array([0.0]), np.array([0.5]))[0]
    assert float(model.sf(gap)) == pytest.approx(0.5, rel=1e-14)


def test_the_module_keeps_the_quantile_limit():
    # Other lifetimes still use their quantile functions in the body.
    assert renewal_model._QF_HAZARD_LIMIT == 20.0
