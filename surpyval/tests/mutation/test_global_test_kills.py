"""Checks that kill mutants of the regressions' global likelihood-ratio
test (``regression/_global_test.py``) that the suite let survive (#396).

The suite's tests of ``likelihood_ratio_test()`` all used one row per
unit and no entry times, so the null model's fit could drop the counts
``n`` or the truncation ``t`` and no test noticed: the statistic is then
measured against the null fitted to other data. The null here is the
univariate fit of the same data, written out.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp


def _data():
    rng = np.random.default_rng(306)
    z = rng.integers(0, 2, 30).astype(float)
    x = np.round(rng.weibull(1.5, 30) * 10 * np.exp(-0.5 * z), 1) + 0.1
    c = (rng.uniform(size=30) < 0.2).astype(int)
    n = rng.integers(1, 4, 30)
    tl = np.where(np.arange(30) % 3 == 0, np.minimum(x * 0.4, 2.0), 0.0)
    return x, z, c, n, tl


@pytest.mark.parametrize(
    "fitter, null",
    [
        (sp.WeibullPH, sp.Weibull),
        (sp.RoystonParmarPH, None),
    ],
)
def test_null_keeps_the_counts(fitter, null):
    """Kills, in ``parametric_likelihood_ratio_test``, the null fit given
    ``n=None`` or no ``n`` (the distribution's and the Royston-Parmar
    spline's): with counts, the test must be that of the rows repeated."""
    x, z, c, n, _ = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        counted = fitter.fit(x, z, c=c, n=n).likelihood_ratio_test()
        rows = np.repeat(np.arange(30), n)
        repeated = fitter.fit(x[rows], z[rows], c=c[rows])
        repeated = repeated.likelihood_ratio_test()
    assert_allclose(
        counted["log_likelihood_null"],
        repeated["log_likelihood_null"],
        rtol=1e-5,
    )
    assert_allclose(counted["statistic"], repeated["statistic"], rtol=1e-4)
    if null is not None:
        ll0 = -null.fit(x, c=c, n=n).neg_ll()
        assert_allclose(counted["log_likelihood_null"], ll0, rtol=1e-6)


@pytest.mark.parametrize("fitter", [sp.WeibullPH, sp.RoystonParmarPH])
def test_null_keeps_the_entry_times(fitter):
    """Kills, in ``parametric_likelihood_ratio_test``, the null fit given
    ``t=None`` or no ``t`` (the distribution's and the Royston-Parmar
    spline's): with delayed entry the null is the univariate fit of the
    left-truncated data (for Royston-Parmar, on the regression's knots
    and scale), not of the data as if every unit were seen from 0."""
    x, z, c, _, tl = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fitter.fit(x, z, c=c, tl=tl)
        res = model.likelihood_ratio_test()
        if fitter is sp.WeibullPH:
            null, plain = sp.Weibull.fit(x, c=c, tl=tl), sp.Weibull.fit(x, c=c)
        else:
            base = model.distribution
            kw = {"knots": base.knots, "scale": base.scale}
            null = sp.RoystonParmar.fit(x, c=c, tl=tl, **kw)
            plain = sp.RoystonParmar.fit(x, c=c, **kw)
    ll0, untruncated = -null.neg_ll(), -plain.neg_ll()
    assert not np.isclose(ll0, untruncated, rtol=1e-3)
    assert_allclose(res["log_likelihood_null"], ll0, rtol=1e-6)
