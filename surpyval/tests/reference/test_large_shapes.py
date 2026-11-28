"""The distribution functions at large shapes, against 50-digit mpmath
values (``mp.gammainc``, ``mp.loggamma`` and ``mp.betainc`` at
``mp.dps = 50``, at the doubles below).

The log masses of the Poisson, binomial and negative binomial
distributions, the gamma density and the front factors of the incomplete
gamma and beta functions are formed by Loader's saddle-point form
(``surpyval.utils.saddle_point``), and the gamma tails at a shape of 100
or more by Temme's expansion or a series or fraction after that front
factor. Each value below was 1e-12 to 3e-9 off before; the tolerances are
the conditioning of each value and a few ulps, far below that (#1300).
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.utils.saddle_point import bd0, log_poisson_mass, stirlerr

EPS = np.finfo(float).eps

# (y, Q(1000, y)): the gamma survival at shape 1e3 and rate 1, so the
# time is the incomplete gamma's own argument. SciPy's gammaincc was
# 3.1e3 ulps off at 1e-30, 500 at 1e-100 and 1.4e3 at 1e-300.
GAMMA_SF = [
    (959.6939327288333, 0.90000000000000028904),
    (1100.578098, 0.0010000000293048639572),
    (1407.188640157539, 9.9999999999999926114e-31),
    (1830.8469620466, 1.0000000000165960564e-100),
    (2666.752073301995, 1.0000000000001364585e-300),
]


@pytest.mark.parametrize("y, true", GAMMA_SF)
def test_gamma_sf_at_shape_1e3(y, true):
    """Within ``2 |log Q| + 16`` ulps: the log of the tail is formed to a
    few ulps of itself, and exponentiated."""
    got = sp.Gamma.sf(y, 1e3, 1.0)
    allowed = (2.0 * abs(np.log(true)) + 16.0) * EPS
    assert abs(got / true - 1.0) <= allowed


# (k, log P(T = k)) for a Poisson of mean 1e6; scipy's logpmf was up to
# 1.4e-10 off, the form k log mu - mu - log k! 9e-11.
POISSON_LOG_MASS = [
    (978801.0, -234.11962454616224494),
    (994392.0, -23.578191658500238191),
    (999999.0, -7.8266938955201431272),
    (1003092.0, -12.603550287052265976),
    (1043114.0, -924.18015050848077807),
]


@pytest.mark.parametrize("k, true", POISSON_LOG_MASS)
def test_poisson_log_mass_at_mean_1e6(k, true):
    got = sp.Poisson.log_df(k, 1e6)
    assert abs(got / true - 1.0) <= 4 * EPS
    assert abs(log_poisson_mass(k, 1e6) / true - 1.0) <= 4 * EPS


# (k, log P(X = k)) for a binomial of 1e6 trials with p = 1/2; scipy's
# logpmf was up to 2e-10 off.
BINOMIAL_LOG_MASS = [
    (481478.0, -693.41883880178943322),
    (499999.0, -7.133548881624864487),
    (501545.0, -11.907599704770738636),
    (521402.0, -923.5037842006709716),
]


@pytest.mark.parametrize("k, true", BINOMIAL_LOG_MASS)
def test_binomial_log_mass_at_1e6_trials(k, true):
    got = sp.Binomial.log_df(k, 1e6, 0.5)
    assert abs(got / true - 1.0) <= 4 * EPS


# (k, R(k) = I_{1-p}(k, r)) for a negative binomial with r = 1e3 and
# p = 1e-6: 2e-10 and 2.2e-9 off before, where the incomplete beta's
# continued fraction cancels in every odd step.
NEGATIVE_BINOMIAL_SF = [
    (1100577049.0, 0.00099999995047026106358),
    (1187747721.0, 9.999998521490528546e-9),
]


@pytest.mark.parametrize("k, true", NEGATIVE_BINOMIAL_SF)
def test_negative_binomial_sf_at_r_1e3(k, true):
    got = sp.NegativeBinomial.sf(k, 1e3, 1e-6)
    assert abs(got / true - 1.0) <= 1e-13


def test_gamma_isf_inverts_the_new_sf_at_shape_1e3():
    """isf is polished on the new sf, so it gives back the time from it
    to an ulp or two (it was 17 ulps at s = 1e-30)."""
    t = np.array([y for y, _ in GAMMA_SF])
    back = sp.Gamma.isf(sp.Gamma.sf(t, 1e3, 1.0), 1e3, 1.0)
    assert np.all(np.abs(back - t) <= 4 * np.spacing(t))


def test_stirlerr_and_bd0_closed_forms():
    """delta(n) is ln n! - (n + 1/2) ln n + n - ln(2 pi) / 2 (exact at
    n = 1: 1 - ln(2 pi) / 2), and the deviance is 0 at x = m and m at
    x = 0."""
    assert stirlerr(1.0) == pytest.approx(1.0 - 0.5 * np.log(2 * np.pi))
    # Stirling's series: 1 / 12n - 1 / 360 n^3 + ...
    assert stirlerr(1e6) == pytest.approx(1 / 12e6, rel=1e-12)
    assert bd0(5.0, 5.0) == 0.0
    assert bd0(0.0, 3.0) == 3.0
    # the series (just inside |v| = 1/2) and the direct form agree
    x, m = 3.0 * (1 - 1e-9), 1.0
    direct = x * np.log(x / m) + m - x
    assert bd0(x, m) == pytest.approx(direct, rel=1e-14)
