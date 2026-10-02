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
    assert model.p == pytest.approx(reference.p, rel=1e-4)
