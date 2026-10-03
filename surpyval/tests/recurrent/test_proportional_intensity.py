"""The proportional-intensity regressions (``ProportionalIntensityHPP``,
``ProportionalIntensityNHPP``): starts, covariates and censored counts.
"""

import warnings

import numpy as np
import pytest

from surpyval import handle_xicn
from surpyval.recurrent import (
    CrowAMSAA,
    Duane,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)


def test_proportional_intensity_left_censored_count_uses_entry():
    data = handle_xicn(
        [10, 20], [1, 1], [-1, 1], n=[3, 1], tl=5, Z=[[1.0], [1.0]]
    )
    rate, b = 0.4, 0.2
    phi = np.exp(b)
    lam = rate * phi * 5
    expected = 3 * np.log(lam) - lam - np.log(6) - rate * phi * 10
    neg_ll = ProportionalIntensityHPP.create_negll_func(data)
    assert np.isclose(-neg_ll(np.array([np.log(rate), b])), expected)

    neg_ll = ProportionalIntensityNHPP.create_negll_func(data, CrowAMSAA)
    alpha, beta = 8.0, 1.3

    def cif(t):
        return (t / alpha) ** beta

    lam = phi * (cif(10) - cif(5))
    expected = 3 * np.log(lam) - lam - np.log(6) - phi * (cif(20) - cif(10))
    assert np.isclose(-neg_ll(np.array([alpha, beta, b])), expected)


# ---------------------------------------------------------------------------
# Duane reaches the Crow-AMSAA optimum from the default start;
# covariates as a dict of scalars or a 1-D array.
# ---------------------------------------------------------------------------


def _power_law_fleet(lam, beta, T, n_items=20, coef=0.7, seed=3):
    rng = np.random.default_rng(seed)
    x, i, c, z = [], [], [], []
    for k in range(n_items):
        zk = k % 2
        cumulative = 0.0
        while True:
            cumulative += rng.exponential()
            t = (cumulative / (lam * np.exp(coef * zk))) ** (1 / beta)
            if t > T:
                break
            x.append(t), i.append(k), c.append(0), z.append(zk)
        x.append(T), i.append(k), c.append(1), z.append(zk)
    return np.array(x), np.array(i), np.array(c), np.array(z).reshape(-1, 1)


@pytest.mark.parametrize(
    "lam, beta, T", [(1e-4, 2.0, 300.0), (1e-5, 2.5, 200.0)]
)
def test_duane_proportional_intensity_reaches_the_crow_amsaa_optimum(
    lam, beta, T
):
    # From a unit start the Duane fit stopped 15-30 AIC short
    x, i, c, Z = _power_law_fleet(lam, beta, T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        duane = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=Duane)
        crow = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=CrowAMSAA)
    assert duane.aic == pytest.approx(crow.aic, abs=0.01)
    assert duane.coeffs == pytest.approx(crow.coeffs, abs=1e-3)


def _small_covariate_fleet():
    x = [2.0, 5.0, 3.0, 7.0, 1.0, 4.0, 2.0, 6.0]
    i = [1, 1, 2, 2, 3, 3, 4, 4]
    c = [0, 1, 0, 1, 0, 1, 0, 1]
    z = {1: 0.2, 2: 0.5, 3: 0.8, 4: 0.3}
    return x, i, c, z


def test_covariates_accept_a_dict_of_scalars_and_a_1d_array():
    from surpyval.recurrent import ProportionalIntensityHPP

    x, i, c, z = _small_covariate_fleet()
    Z_rows = np.array([[z[k]] for k in i])
    as_rows = ProportionalIntensityHPP.fit(x, Z_rows, i=i, c=c)
    as_dict = ProportionalIntensityHPP.fit(x, z, i=i, c=c)
    as_1d = ProportionalIntensityHPP.fit(x, Z_rows.ravel(), i=i, c=c)
    np.testing.assert_allclose(as_dict.params, as_rows.params)
    np.testing.assert_allclose(as_1d.params, as_rows.params)
    with pytest.raises(ValueError, match="no covariates for item"):
        ProportionalIntensityHPP.fit(x, {1: 0.2, 2: 0.5}, i=i, c=c)


# ---------------------------------------------------------------------------
# The fitters honour ``init`` (#288).
# ---------------------------------------------------------------------------


class TestPIFittersHonourInit:
    @staticmethod
    def _data():
        np.random.seed(9)
        xs, iis, cs, Zs = [], [], [], []
        for it in range(30):
            z = np.random.binomial(1, 0.5)
            t = 0.0
            rate = 0.3 * np.exp(0.5 * z)
            while True:
                t += np.random.exponential(1 / rate)
                if t > 20:
                    break
                xs.append(t)
                iis.append(it)
                cs.append(0)
                Zs.append([z])
            xs.append(20.0)
            iis.append(it)
            cs.append(1)
            Zs.append([z])
        return xs, iis, cs, Zs

    def test_hpp_accepts_and_validates_init(self):
        xs, iis, cs, Zs = self._data()
        m = ProportionalIntensityHPP.fit(
            x=xs, Z=Zs, i=iis, c=cs, init=[0.3, 0.5]
        )
        assert np.all(np.isfinite(np.atleast_1d(m.params)))
        with pytest.raises(ValueError, match="init must have"):
            ProportionalIntensityHPP.fit(
                x=xs, Z=Zs, i=iis, c=cs, init=[0.3, 0.5, 0.9]
            )

    def test_nhpp_accepts_and_validates_init(self):
        xs, iis, cs, Zs = self._data()
        m = ProportionalIntensityNHPP.fit(
            x=xs, Z=Zs, i=iis, c=cs, dist=CrowAMSAA, init=[10.0, 1.0, 0.5]
        )
        assert np.all(np.isfinite(np.atleast_1d(m.params)))
        with pytest.raises(ValueError, match="init must have"):
            ProportionalIntensityNHPP.fit(
                x=xs, Z=Zs, i=iis, c=cs, dist=CrowAMSAA, init=[1.0]
            )
