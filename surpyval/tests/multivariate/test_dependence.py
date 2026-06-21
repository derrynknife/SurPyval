import numpy as np

from surpyval.multivariate import (
    Clayton,
    Frank,
    Gaussian,
    Gumbel,
    Independence,
)


def test_clayton_kendall_tau_closed_form():
    for theta in [0.5, 2.0, 5.0]:
        assert np.isclose(Clayton.kendall_tau(theta), theta / (theta + 2.0))


def test_gumbel_kendall_tau_closed_form():
    for theta in [1.5, 2.0, 4.0]:
        assert np.isclose(Gumbel.kendall_tau(theta), 1.0 - 1.0 / theta)


def test_gaussian_kendall_and_spearman_closed_form():
    for rho in [-0.4, 0.3, 0.8]:
        assert np.isclose(
            Gaussian.kendall_tau(rho), 2 / np.pi * np.arcsin(rho)
        )
        assert np.isclose(
            Gaussian.spearman_rho(rho), 6 / np.pi * np.arcsin(rho / 2)
        )


def test_independence_has_zero_dependence():
    assert Independence.kendall_tau() == 0.0
    assert Independence.spearman_rho() == 0.0


def test_tail_dependence():
    # Clayton: lower tail only
    lo, up = Clayton.tail_dependence(2.0)
    assert np.isclose(lo, 2.0 ** (-1.0 / 2.0)) and up == 0.0
    # Gumbel: upper tail only
    lo, up = Gumbel.tail_dependence(2.0)
    assert lo == 0.0 and np.isclose(up, 2.0 - 2.0 ** (1.0 / 2.0))
    # Frank: none
    assert Frank.tail_dependence(3.0) == (0.0, 0.0)


def test_frank_kendall_tau_matches_empirical():
    # closed-form Debye-based tau agrees with the sampled estimate
    tau_cf = Frank.kendall_tau(5.0)
    u, v = Frank.sample_uv(40_000, [5.0], random_state=0)
    from scipy.stats import kendalltau

    assert np.isclose(tau_cf, kendalltau(u, v).statistic, atol=0.02)
