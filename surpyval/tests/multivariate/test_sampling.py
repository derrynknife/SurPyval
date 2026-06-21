import numpy as np
from scipy.stats import kendalltau

import surpyval as surv
from surpyval.multivariate import Clayton, Frank, Gaussian, Gumbel

MARGINS = [
    surv.Weibull.from_params([10, 2]),
    surv.LogNormal.from_params([3, 0.4]),
]


def test_random_shape_and_margins():
    m = Clayton.from_params(2.0, margins=MARGINS)
    s = m.random(5000, random_state=0)
    assert s.shape == (5000, 2)
    # sampled margins match the specified margin quantiles
    assert np.isclose(np.median(s[:, 0]), MARGINS[0].qf(0.5), rtol=0.1)
    assert np.isclose(np.median(s[:, 1]), MARGINS[1].qf(0.5), rtol=0.1)


def test_sample_dependence_matches_model_tau():
    for cop, p in [
        (Clayton, 2.5),
        (Gumbel, 2.0),
        (Frank, 5.0),
        (Gaussian, 0.6),
    ]:
        m = cop.from_params(p, margins=MARGINS)
        s = m.random(20_000, random_state=1)
        tau_emp = kendalltau(s[:, 0], s[:, 1]).statistic
        assert np.isclose(tau_emp, m.kendall_tau(), atol=0.03)


def test_gaussian_direct_sampler_uniform_margins():
    u, v = Gaussian.sample_uv(20_000, [0.6], random_state=2)
    # copula samples have uniform margins
    assert np.isclose(u.mean(), 0.5, atol=0.02)
    assert np.isclose(v.mean(), 0.5, atol=0.02)
    assert np.isclose(
        kendalltau(u, v).statistic, Gaussian.kendall_tau(0.6), atol=0.03
    )


def test_conditional_cdf_is_in_unit_interval():
    m = Clayton.from_params(2.0, margins=MARGINS)
    x = np.array([[10.0, 18.0], [5.0, 25.0]])
    h = m.conditional_cdf(x, given_dim=0)
    assert np.all(h >= 0) and np.all(h <= 1)
