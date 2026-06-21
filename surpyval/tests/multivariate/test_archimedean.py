import numpy as np

import surpyval as surv
from surpyval.multivariate import (
    Clayton,
    Frank,
    Gaussian,
    Gumbel,
    Independence,
)

MARGINS = [
    surv.Weibull.from_params([10, 2]),
    surv.LogNormal.from_params([3, 0.4]),
]


def test_independence_cdf_is_product_of_margins():
    ind = Independence.from_params([], margins=MARGINS)
    x = np.array([[10.0, 18.0], [5.0, 25.0], [15.0, 12.0]])
    prod = MARGINS[0].ff(x[:, 0]) * MARGINS[1].ff(x[:, 1])
    assert np.allclose(ind.cdf(x), prod)
    assert np.allclose(
        ind.pdf(x), MARGINS[0].df(x[:, 0]) * MARGINS[1].df(x[:, 1])
    )


def test_joint_sf_definition():
    m = Clayton.from_params(2.0, margins=MARGINS)
    x = np.array([[10.0, 18.0], [7.0, 20.0]])
    u = MARGINS[0].ff(x[:, 0])
    v = MARGINS[1].ff(x[:, 1])
    expected = 1.0 - u - v + m.cdf(x)
    assert np.allclose(m.sf(x), expected)


def test_cdf_bounds_and_monotonicity():
    m = Gumbel.from_params(2.5, margins=MARGINS)
    x = np.array([[1.0, 1.0], [10.0, 18.0], [1e3, 1e3]])
    c = m.cdf(x)
    assert np.all(c >= 0) and np.all(c <= 1)
    assert c[0] < c[1] < c[2]


def _recover(cop, true_p, seed):
    m = cop.from_params(true_p, margins=MARGINS)
    data = m.random(4000, random_state=seed)
    fit = cop.fit(
        [data[:, 0], data[:, 1]],
        margins=[surv.Weibull, surv.LogNormal],
        how="IFM",
    )
    return fit.params[0]


def test_ifm_recovers_clayton():
    assert np.isclose(_recover(Clayton, 2.5, 1), 2.5, rtol=0.15)


def test_ifm_recovers_gumbel():
    assert np.isclose(_recover(Gumbel, 2.0, 2), 2.0, rtol=0.15)


def test_ifm_recovers_frank():
    assert np.isclose(_recover(Frank, 5.0, 3), 5.0, rtol=0.15)


def test_ifm_recovers_gaussian():
    assert np.isclose(_recover(Gaussian, 0.6, 4), 0.6, rtol=0.15)


def test_mle_joint_recovers_clayton_and_margins():
    m = Clayton.from_params(2.5, margins=MARGINS)
    data = m.random(3000, random_state=5)
    fit = Clayton.fit(
        [data[:, 0], data[:, 1]],
        margins=[surv.Weibull, surv.LogNormal],
        how="MLE",
    )
    assert np.isclose(fit.params[0], 2.5, rtol=0.2)
    assert np.allclose(fit.margins[0].params, [10, 2], rtol=0.15)


def test_from_params_roundtrips_through_to_dict():
    m = Clayton.from_params(2.0, margins=MARGINS)
    d = m.to_dict()
    assert d["copula"] == "Clayton"
    assert np.isclose(d["params"][0], 2.0)
    assert len(d["margins"]) == 2
