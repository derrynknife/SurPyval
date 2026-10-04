"""``CustomDistribution``: its moments, quantiles and sampling, the methods
it refuses, and its round trip through the registry.
"""

import numpy as np
import pytest
from autograd import numpy as anp
from scipy.stats import gompertz

import surpyval as surv
from surpyval.tests._helpers import no_warnings


def _gompertz() -> surv.CustomDistribution:
    def Hf(x, *params):
        return params[0] * (anp.exp(params[1] * x) - 1)

    return surv.CustomDistribution(
        "Gompertz", Hf, ["nu", "b"], ((0, None), (0, None)), (0, np.inf)
    )


def test_custom_distribution_refuses_mpp_clearly():
    G = _gompertz()
    assert not G.supports_mpp
    with pytest.raises(ValueError, match="probability plot"):
        G.fit([0.5, 1.0, 1.2, 1.5, 2.0], how="MPP")


def test_custom_distribution_moments_mean_and_var():
    G = _gompertz()
    rv = gompertz(0.1, scale=1 / 0.5)
    model = G.from_params([0.1, 0.5])
    assert model.mean() == pytest.approx(rv.mean(), rel=1e-8)
    assert model.var() == pytest.approx(rv.var(), rel=1e-8)
    assert G.moment(2, 0.1, 0.5) == pytest.approx(rv.moment(2), rel=1e-8)


def test_custom_distribution_moments_on_other_supports():
    def H_shifted(x, *params):
        return ((x - 2) / params[0]) ** params[1]

    W = surv.CustomDistribution(
        "W2", H_shifted, ["a", "shape"], ((0, None), (0, None)), (2, np.inf)
    )
    assert W.moment(1, 3.0, 2.0) == pytest.approx(
        2 + surv.Weibull.mean(3.0, 2.0)
    )

    def H_gumbel(x, *params):
        return anp.exp((x - params[0]) / params[1])

    G = surv.CustomDistribution(
        "Gmb",
        H_gumbel,
        ["mu", "s"],
        ((None, None), (0, None)),
        (-np.inf, np.inf),
    )
    assert G.moment(1, -3.0, 2.0) == pytest.approx(surv.Gumbel.mean(-3.0, 2.0))
    assert G.moment(2, -3.0, 2.0) == pytest.approx(
        surv.Gumbel.moment(2, -3.0, 2.0)
    )


def test_custom_distribution_mom_is_fast_and_matches():
    G = _gompertz()
    x = gompertz(0.1, scale=1 / 0.5).rvs(size=100, random_state=0)
    model = G.fit(x, how="MOM")
    np.testing.assert_allclose(model.mean(), x.mean(), rtol=1e-4)
    np.testing.assert_allclose(model.moment(2), (x**2).mean(), rtol=1e-4)


# ---------------------------------------------------------------------------
# Moments at any scale, quantile and random, the registry.
# ---------------------------------------------------------------------------


W, E, G = surv.Weibull, surv.Exponential, surv.Geometric


def _custom_weibull(name="CustomW3"):
    def Hf(x, *params):
        return (x / params[0]) ** params[1]

    return surv.CustomDistribution(
        name, Hf, ["a", "b"], ((0, None), (0, None)), (0, np.inf)
    )


@pytest.mark.parametrize("alpha", [1e-4, 10.0, 1e5])
def test_custom_distribution_moments_at_any_scale(alpha):
    C = _custom_weibull()
    assert no_warnings(C.mean, alpha, 2.0) == pytest.approx(
        W.mean(alpha, 2.0), rel=1e-9
    )
    assert C.moment(2, alpha, 2.0) == pytest.approx(
        W.moment(2, alpha, 2.0), rel=1e-9
    )


def test_custom_distribution_has_a_quantile_and_random():
    C = _custom_weibull()
    u = np.array([0.1, 0.5, 0.9])
    assert np.allclose(C.qf(u, 10.0, 2.0), W.qf(u, 10.0, 2.0))
    model = C.fit([1.0, 2, 3, 4, 5])
    np.random.seed(1)
    assert model.random(4).shape == (4,)


def test_custom_distribution_round_trips_through_its_registry():
    C = _custom_weibull("RoundTripW")
    model = C.fit([1.0, 2, 3, 4, 5])
    d = model.to_dict()
    assert d["custom"] is True
    restored = surv.from_dict(d)
    assert restored.dist is C
    assert np.allclose(restored.sf([2, 4]), model.sf([2, 4]))


def test_unregistered_custom_distribution_gives_a_clear_error():
    C = _custom_weibull("NeverRegisteredAgain")
    d = C.fit([1.0, 2, 3, 4, 5]).to_dict()
    d["distribution"] = "SomeOtherCustom"
    with pytest.raises(ValueError, match="Construct it again"):
        surv.from_dict(d)


def test_custom_distribution_may_name_a_parameter_p():
    def Hf(x, *params):
        return (x / params[0]) ** params[1]

    C = surv.CustomDistribution(
        "CustomWithP", Hf, ["a", "p"], ((0, None), (0, None)), (0, np.inf)
    )
    np.random.seed(2)
    x = W.random(200, 10, 2)
    c = np.zeros(200)
    never = np.random.uniform(size=200) > 0.7
    x[never], c[never] = 30, 1
    model = C.fit(x, c, lfp=True)
    reference = W.fit(x, c, lfp=True)
    assert model.params == pytest.approx(reference.params, rel=1e-4)
    assert model.lfp_p == pytest.approx(reference.lfp_p, rel=1e-4)
    # ``p`` is the distribution's own parameter (#608)
    assert model.p == model.params[1]


_SUPPORTS = {
    # name: (Hf, bounds, support, params)
    "half line": (
        lambda x, nu, b: nu * anp.expm1(b * x),
        ((0, None), (0, None)),
        (0, np.inf),
        (1e-3, 0.5),
    ),
    "whole line": (
        lambda x, mu, s: anp.exp((x - mu) / s),
        ((None, None), (0, None)),
        (-np.inf, np.inf),
        (5.0, 2.0),
    ),
    "interval": (
        lambda x, a: -a * anp.log1p(-x),
        ((0, None),),
        (0, 1),
        (1.5,),
    ),
    "below zero": (
        lambda x, a: -a * anp.log(-anp.expm1(x)),
        ((0, None),),
        (-np.inf, 0),
        (2.0,),
    ),
}


@pytest.mark.parametrize("support", sorted(_SUPPORTS))
def test_596_qf_solves_every_probability_at_once(support):
    # A cumulative hazard that broadcasts is inverted for every
    # probability together: 300 points in a few dozen calls, where a
    # brentq per probability made thousands (0.5-1.2 s for 2000). The
    # answers are the one-at-a-time inversion's.
    fun, bounds, edges, params = _SUPPORTS[support]
    calls = []

    def Hf(x, *p):
        calls.append(1)
        return fun(x, *p)

    names = [f"p{i}" for i in range(len(params))]
    C = surv.CustomDistribution("Y596" + support, Hf, names, bounds, edges)
    u = np.concatenate(
        [np.random.default_rng(0).uniform(size=300), [0, 1e-9, 1 - 1e-9, 1]]
    )
    q = C.qf(u, *params)
    assert len(calls) < 200
    H = C._scalar_fn(C.Hf, list(params))
    lo, hi = map(float, edges)
    with np.errstate(divide="ignore"):
        target = -np.log1p(-u)
    alone = [C._invert_Hf(H, t, lo, hi) for t in target]
    np.testing.assert_allclose(q, alone, rtol=1e-13, atol=1e-300)


def _scalar_only(x, nu, b):
    # Takes one point at a time.
    return nu * np.expm1(b * float(np.ravel(x)[0]))


def _reduces(x, nu, b):
    # Broadcasts, but a point's value depends on the others.
    return nu * anp.expm1(b * x) + (anp.max(x) - x)


def _wrong_when_long(x, nu, b):
    # Right on short arrays only: passes the probe, not the answers.
    shift = 1.0 if np.size(x) > 50 else 0.0
    return nu * anp.expm1(b * x) + shift


@pytest.mark.parametrize("fun", [_scalar_only, _reduces, _wrong_when_long])
def test_596_qf_of_a_hazard_that_does_not_broadcast(fun):
    # Each probability is then solved alone, as before, and the
    # quantiles are those of the same hazard written to broadcast.
    bounds, support = ((0, None), (0, None)), (0, np.inf)
    C = surv.CustomDistribution(
        "Y596" + fun.__name__, fun, ["nu", "b"], bounds, support
    )
    G = surv.CustomDistribution(
        "Y596broadcasts",
        _SUPPORTS["half line"][0],
        ["nu", "b"],
        bounds,
        support,
    )
    u = np.random.default_rng(1).uniform(size=100)
    np.testing.assert_allclose(
        C.qf(u, 1e-3, 0.5), G.qf(u, 1e-3, 0.5), rtol=1e-13
    )


def test_596_qf_of_a_tiny_probability_does_not_raise():
    # nu (exp(b x) - 1) is 0 below x ~ 2e-16, where exp(b x) rounds to 1:
    # brentq never converged on a target of 1e-30 (RuntimeError); the
    # bracket now closes on the smallest x where the hazard is positive.
    q = _gompertz().qf(np.array([1e-300, 1e-30]), 1e-3, 0.5)
    assert np.all((q > 0) & (q < 1e-15))
