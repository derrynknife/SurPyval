import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from surpyval import Weibull  # noqa: E402
from surpyval.recurrent import (  # noqa: E402
    ARA,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
)
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
    assert np.isfinite(model.aic()) and np.isfinite(model.bic())
    mcf = model.mcf(np.array([1.0, 2.0, 3.0, 4.0]), items=1000, random_state=0)
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
        model.aic()


def test_777_starts_where_the_lifetime_fit_runs_off():
    # An ExpoWeibull fitted to this item's gaps runs off to a power law
    # ending at the longest gap, so a gap from any later age had zero
    # likelihood at every default start, and the fit failed with "Could
    # not find a good solution". Each restoration start now has its own
    # lifetime, fitted to the gaps from the ages it leaves (#777).
    from surpyval import ExpoWeibull
    from surpyval.recurrent.renewal.fit_mixin import RenewalFitMixin

    x = np.array([1.87, 5.28, 5.82, 8.2, 10.77, 11.18, 14.6, 19.4])
    data = ARA.fit(x).data
    neg_ll = ARA.create_negll_func(data, ExpoWeibull, 2)
    life = RenewalFitMixin._initial_dist_params(data, ExpoWeibull)
    assert not any(
        RenewalFitMixin._finite_at(neg_ll, [rho, *life])
        for rho in (0.1, 0.5, 0.9, 0.99)
    )
    with pytest.warns(UserWarning, match="No finite maximum"):
        model = ARA.fit(x, dist=ExpoWeibull, m=2)
    assert model.maximum == "no finite maximum"
    assert np.isfinite(model.log_likelihood)
    # At least as high as perfect repair with the gaps' own lifetime,
    # whose fit stops at -12.5385
    assert model.log_likelihood > -12.5386


@pytest.mark.parametrize("family", ["kijima-ii", "g1"])
def test_777_a_life_running_off_has_no_finite_maximum(family):
    # The ExpoWeibull life runs off to its power-law limit (beta to 1e9,
    # mu to 1e-9), and the fits stopped on the ridge, "unverified" (with
    # the start's own fit warning of its run-off besides). Their life,
    # fitted with the restoration held, has no finite maximum: so the
    # fit says, once, in the package's words (#777).
    import warnings

    from surpyval import ExpoWeibull

    x = np.array([1.87, 5.28, 5.82, 8.2, 10.77, 11.18, 14.6, 19.4])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if family == "g1":
            model = GeneralizedOneRenewal.fit(x, dist=ExpoWeibull)
        else:
            model = GeneralizedRenewal.fit(x, dist=ExpoWeibull, kijima="ii")
    said = [str(w.message) for w in caught]
    assert len(said) == 1, said
    assert said[0].startswith("No finite maximum: the ")
    assert "likelihood keeps increasing as its ExpoWeibull life runs off" in (
        said[0]
    )
    assert "power law" in said[0]
    assert model.maximum == "no finite maximum"
