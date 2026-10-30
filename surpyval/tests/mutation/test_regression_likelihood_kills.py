"""Checks that kill mutants of the parametric regressions' likelihood
that the suite let survive (#396)."""

import warnings

import numpy as np
import pytest

import surpyval as sp


def _mixed_windows():
    """60 rows: 10 truncated on both sides, 10 on the left only, 20 on
    the right only, 20 not at all."""
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(60, 1))
    x = rng.weibull(1.5, 60) * 10 * np.exp(-0.4 * Z[:, 0])
    rows = np.arange(60)
    tl = np.where(rows % 3 == 0, x * 0.3, -np.inf)
    tr = np.where(rows % 2 == 0, x * 2.0, np.inf)
    return x, Z, tl, tr


@pytest.mark.parametrize(
    "fitter, params",
    [
        ("WeibullAFT", [13.44837451, 1.71794673, 0.40049884]),
        ("LogNormalAFT", [2.27048842, 0.78575086, 0.48200481]),
    ],
)
def test_two_sided_windows_among_one_sided_ones(fitter, params):
    """Kills, in ``regression/_likelihood.py::truncation_correction``,
    ``stand_in(tl, both)`` -> ``stand_in(tl, None)``: the two-sided
    windows' branch then evaluates the distribution at the other rows'
    ``tl = -inf``, which ``np.where`` discards in value but not in the
    gradient, so the search stopped short of a verified maximum (with
    the warning) on these data. No test mixed two-sided windows with
    one-sided ones. The parameters are regression values.
    """
    x, Z, tl, tr = _mixed_windows()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = getattr(sp, fitter).fit(x, Z, tl=tl, tr=tr)
    assert m.maximum == "verified"
    np.testing.assert_allclose(m.params, params, rtol=1e-6)
