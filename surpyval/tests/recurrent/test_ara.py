import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from surpyval import Weibull  # noqa: E402
from surpyval.recurrent import ARA, GeneralizedRenewal  # noqa: E402
from surpyval.recurrent.renewal.ara import ara_virtual_ages  # noqa: E402

X = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
C = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
I = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])


def test_ara_virtual_ages_reduces_to_kijima():
    # The ARA_m virtual age must reproduce the Kijima-I age at m=1 and the
    # Kijima-II age at m=inf.
    T = np.array([2.0, 5.0, 9.0, 14.0, 20.0])
    rho = 0.3
    q = 1 - rho

    kijima_i = np.array([0.0] + [q * T[k - 1] for k in range(1, len(T))])
    assert np.allclose(ara_virtual_ages(T, rho, 1), kijima_i)

    virtual = 0.0
    kijima_ii = [0.0]
    interarrival = np.diff(T, prepend=0)
    for k in range(1, len(T)):
        virtual = q * (virtual + interarrival[k - 1])
        kijima_ii.append(virtual)
    assert np.allclose(ara_virtual_ages(T, rho, np.inf), np.array(kijima_ii))


def test_ara_m1_matches_kijima_i():
    ara = ARA.fit(X, I, c=C, m=1)
    gr = GeneralizedRenewal.fit(X, I, c=C, kijima="i")
    assert np.isclose(ara.rho, 1 - gr.q, atol=1e-3)
    assert np.isclose(ara.log_likelihood, gr.log_likelihood, atol=1e-4)
    assert np.allclose(ara.model.params, gr.model.params, rtol=1e-3)


def test_ara_minf_matches_kijima_ii():
    ara = ARA.fit(X, I, c=C, m=np.inf)
    gr = GeneralizedRenewal.fit(X, I, c=C, kijima="ii")
    assert np.isclose(ara.log_likelihood, gr.log_likelihood, atol=1e-3)


def test_ara_general_memory_fits_and_simulates():
    model = ARA.fit(X, I, c=C, m=2)
    assert model.m == 2
    assert 0.0 <= model.rho <= 1.0
    assert np.isfinite(model.aic) and np.isfinite(model.bic)
    mcf = model.mcf(np.array([1.0, 2.0, 3.0, 4.0]), items=1000, seed=0)
    assert np.all(np.diff(mcf) >= -1e-9)
    assert "ARA" in repr(model)


def test_ara_validates_memory():
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            ARA.fit(X, I, c=C, m=bad)


def test_ara_rejects_unsupported_censoring():
    c_interval = np.array([0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="censoring code"):
        ARA.fit(X, I, c=c_interval, m=2)


def test_ara_inference_requires_fit_from_data():
    model = ARA.fit_from_parameters([10.0, 2.0], rho=0.4, m=2, dist=Weibull)
    with pytest.raises(ValueError, match="fitted from data"):
        model.aic
