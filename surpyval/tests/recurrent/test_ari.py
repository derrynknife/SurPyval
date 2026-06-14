import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from surpyval.recurrent import ARI, CrowAMSAA, Duane  # noqa: E402
from surpyval.recurrent.renewal.ari import ari_reduction  # noqa: E402
from surpyval.utils.recurrent_utils import handle_xicn  # noqa: E402

X = np.array([3, 9, 20, 35, 56, 4, 11, 25, 44, 70], dtype=float)
I = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])


def test_ari_reduction_helper():
    lams = [0.2, 0.5, 0.9]
    rho = 0.4
    # m = 1 keeps only the most recent failure intensity.
    assert np.isclose(ari_reduction(lams, rho, 1), rho * 0.9)
    # m = 2 keeps the last two.
    assert np.isclose(
        ari_reduction(lams, rho, 2), rho * (0.9 + (1 - rho) * 0.5)
    )
    # m = inf keeps the whole memory-weighted history.
    assert np.isclose(
        ari_reduction(lams, rho, np.inf),
        rho * (0.9 + (1 - rho) * 0.5 + (1 - rho) ** 2 * 0.2),
    )
    assert ari_reduction([], rho, 1) == 0.0


@pytest.mark.parametrize("dist", [CrowAMSAA, Duane])
def test_ari_rho_zero_matches_nhpp(dist):
    # With rho = 0 there is no intensity reduction, so the ARI log-likelihood
    # must equal the plain NHPP log-likelihood of the baseline intensity.
    data = handle_xicn(X, I, as_recurrent_data=True)
    ari_negll = ARI.create_negll_func(data, dist, m=1)
    nhpp_negll = dist.create_negll_func(data)
    for params in ([100.0, 1.3], [50.0, 1.4]):
        a = ari_negll([0.0, *params])
        b = nhpp_negll(params)
        assert np.isfinite(a) and np.isclose(a, b)


def test_ari_fit_and_information_criteria():
    model = ARI.fit(X, I, m=1, dist=CrowAMSAA)
    assert 0.0 <= model.rho <= 1.0
    k = model._mle.size
    n = model._n_obs
    ll = model.log_likelihood
    assert np.isclose(ll, -model.res.fun)
    assert np.isclose(model.aic, 2 * k - 2 * ll)
    assert np.isclose(model.bic, k * np.log(n) - 2 * ll)
    assert model.parameter_names == ["rho", "alpha", "beta"]
    assert "ARI" in repr(model)


def test_ari_mcf_simulation_monotonic():
    model = ARI.fit_from_parameters([60.0, 2.0], rho=0.3, m=1, dist=CrowAMSAA)
    mcf = model.mcf(np.array([5.0, 10.0, 20.0, 30.0]), items=800, seed=0)
    assert np.all(np.diff(mcf) >= -1e-9)
    assert np.all(mcf >= 0)


def test_ari_validates_memory():
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            ARI.fit(X, I, m=bad)


def test_ari_rejects_unsupported_censoring():
    c = np.zeros_like(I)
    c[-1] = 2  # interval censoring not supported
    with pytest.raises(ValueError, match="censoring code"):
        ARI.fit(X, I, c=c, m=1)


def test_ari_inference_requires_fit_from_data():
    model = ARI.fit_from_parameters([60.0, 2.0], rho=0.3, m=1, dist=CrowAMSAA)
    with pytest.raises(ValueError, match="fitted from data"):
        model.aic
