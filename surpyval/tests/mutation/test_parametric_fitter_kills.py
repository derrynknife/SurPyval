"""Checks that kill mutants of the parametric fitters that the suite
let survive (#396).

Each test names the mutant it kills: the function, then the edit. The
data are small and the tests fast: they run in the normal suite.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp


def _lfp_zi_data():
    """120 units: a quarter never fail (a limited failure population), a
    tenth fail at 0 (zero inflation), the rest are Weibull, all censored
    at 70."""
    rng = np.random.default_rng(7)
    life = rng.weibull(1.8, 120) * 50 + 10
    immune = rng.random(120) < 0.25
    doa = rng.random(120) < 0.10
    x = life.copy()
    x[doa] = 0.0
    c = np.zeros(120, int)
    late = (x > 70.0) | immune
    c[late] = 1
    x[late] = 70.0
    return x, c


def _fit(x, c, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return sp.Weibull.fit(x, c=c, **options)


def test_lfp_and_zi_together():
    """Kills, in ``fitters/mle.py::_negative_log_likelihood.fun``, the
    ``if zi:`` and ``if lfp:`` unpacking blocks swapped: ``f0`` and the
    limited-failure proportion ``p`` then read each other's parameter,
    which gave ``lfp_p`` 0.548, a nan ``neg_ll`` and an unverified
    maximum on these data.

    With every exact 0 failed at 0 and every other time after it, the
    zero-inflation share has the closed form ``f0 = 11 / 120`` (the 11
    units dead on arrival); ``lfp_p`` and ``neg_ll`` are regression values
    (true ``p`` about 0.76). Adding ``lfp`` to a zero-inflated fit nests
    it, so its maximum cannot be lower.
    """
    x, c = _lfp_zi_data()
    assert np.sum(x == 0) == 11
    m = _fit(x, c, lfp=True, zi=True)
    assert m.maximum == "verified"
    assert m.f0 == pytest.approx(11 / 120, rel=1e-6)
    assert m.lfp_p == pytest.approx(0.764399, rel=1e-5)
    assert np.isfinite(m.neg_ll())
    assert m.neg_ll() == pytest.approx(357.404136, rel=1e-8)
    assert m.neg_ll() <= _fit(x, c, zi=True).neg_ll()
