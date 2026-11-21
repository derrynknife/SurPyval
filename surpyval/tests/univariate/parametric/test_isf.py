"""The inverse survival function ``isf``: the time at which the survival
function is ``s``.

The values themselves are checked against 50-digit references far into
the tails in ``tests/reference/test_tails.py``; here, that it inverts
``sf`` (``isf(sf(t)) == t``, deep in the upper tail where ``qf`` cannot
go), agrees with ``qf(1 - s)`` where ``1 - s`` is exact, and that a
model's ``isf`` follows its offset, limited failure population and zero
inflation as its ``qf`` does.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp

EPS = np.finfo(float).eps

CONTINUOUS = [
    (sp.Weibull, (10.0, 3.0)),
    (sp.Weibull, (10.0, 0.5)),
    (sp.Exponential, (0.1,)),
    (sp.Rayleigh, (3.0,)),
    (sp.Gumbel, (5.0, 2.0)),
    (sp.GumbelLEV, (5.0, 2.0)),
    (sp.Logistic, (5.0, 2.0)),
    (sp.LogLogistic, (10.0, 3.0)),
    (sp.Normal, (5.0, 2.0)),
    (sp.LogNormal, (2.0, 0.5)),
    (sp.Gamma, (2.5, 0.5)),
    (sp.Gamma, (1e3, 1e-6)),
    (sp.Beta, (2.0, 3.0)),
    (sp.Beta4, (3.0, 4.0, 2.0, 3.0)),
    (sp.Uniform, (0.0, 10.0)),
    (sp.ExpoWeibull, (10.0, 2.0, 1.5)),
    (sp.Hypoexponential, (1.0, 2.0)),
]
S = np.array([0.9, 0.5, 1e-3, 1e-8, 1e-16, 1e-30, 1e-100, 1e-200, 1e-300])


def _id(case):
    dist, params = case
    return f"{dist.name}{params}"


@pytest.mark.parametrize("case", CONTINUOUS, ids=_id)
def test_isf_inverts_sf(case):
    """sf(isf(s)) is s to the rounding of the time: a relative error in
    ``t`` of ``eps`` moves ``log sf`` by ``eps t h(t)``."""
    dist, params = case
    t = dist.isf(S, *params)
    sf = dist.sf(t, *params)
    keep = np.isfinite(t) & (sf > 0) & (t > 0)
    h = dist.hf(t[keep], *params)
    allowed = 8 * EPS * np.maximum(1.0, t[keep] * h)
    err = np.abs(np.log(sf[keep]) - np.log(S[keep]))
    assert np.all(err <= allowed), (err, allowed)


@pytest.mark.parametrize("case", CONTINUOUS, ids=_id)
def test_isf_of_sf_gives_back_the_time(case):
    """isf(sf(t)) == t deep in the upper tail, to the conditioning of
    ``isf`` (``1 / (t h(t))`` per relative error of ``s``)."""
    dist, params = case
    t = dist.isf(S, *params)
    keep = np.isfinite(t) & (dist.sf(t, *params) > 0) & (t > 0)
    t = t[keep]
    back = dist.isf(dist.sf(t, *params), *params)
    h = dist.hf(t, *params)
    allowed = 8 * EPS * np.abs(t) * np.maximum(1.0, 1.0 / (np.abs(t) * h))
    assert np.all(np.abs(back - t) <= allowed)


@pytest.mark.parametrize("case", CONTINUOUS, ids=_id)
def test_isf_is_qf_of_the_complement(case):
    """Where ``1 - s`` is exact (``s >= 1/2``) isf(s) is qf(1 - s)."""
    dist, params = case
    s = np.array([0.5, 0.6, 0.9, 0.999, 1 - 1e-12])
    np.testing.assert_allclose(
        dist.isf(s, *params), dist.qf(1 - s, *params), rtol=1e-12
    )
    s = np.array([0.4, 0.1, 1e-3])
    np.testing.assert_allclose(
        dist.isf(s, *params), dist.qf(1 - s, *params), rtol=1e-10
    )


@pytest.mark.parametrize("case", CONTINUOUS, ids=_id)
def test_isf_at_the_ends_and_missing(case):
    dist, params = case
    lo, hi = dist.qf(np.array([0.0, 1.0]), *params)
    out = dist.isf([1.0, 0.0, np.nan], *params)
    assert out[0] == lo and out[1] == hi and np.isnan(out[2])
    with pytest.warns(UserWarning):
        assert np.isnan(dist.isf(1.5, *params))


@pytest.mark.parametrize(
    "dist, params",
    [
        (sp.Poisson, (4.0,)),
        (sp.Geometric, (0.3,)),
        (sp.Binomial, (20, 0.3)),
        (sp.NegativeBinomial, (3.0, 0.4)),
        (sp.DiscreteWeibull, (0.9, 1.5)),
        (sp.BetaGeometric, (2.0, 3.0)),
    ],
)
def test_discrete_isf_is_the_first_point_reached(dist, params):
    """The smallest k with sf(k) <= s, also where s is below 1e-16."""
    s = np.array([0.9, 0.5, 0.1, 1e-5, 1e-20])
    k = dist.isf(s, *params)
    assert np.all(dist.sf(k, *params) <= s)
    assert np.all(dist.sf(k - 1, *params) > s)


def test_model_isf_is_the_distribution_isf():
    model = sp.Weibull.from_params([10, 3])
    s = np.array([0.5, 1e-10, 1e-300])
    assert np.array_equal(model.isf(s), sp.Weibull.isf(s, 10, 3))
    assert model.isf(1e-300) == sp.Weibull.isf(1e-300, 10, 3)
    assert np.ndim(model.isf(0.3)) == 0


@pytest.mark.parametrize(
    "options",
    [
        dict(gamma=2.0),
        dict(lfp_p=0.6),
        dict(f0=0.1),
        dict(gamma=2.0, lfp_p=0.6, f0=0.1),
    ],
)
def test_model_isf_follows_the_options(options):
    """isf is qf(1 - s) where 1 - s is exact; sf(isf(s)) is s where
    the model's survival reaches it; inf at or below 1 - p, 0 at or
    above 1 - f0."""
    model = sp.Weibull.from_params([10, 3], **options)
    s = np.array([1.0, 0.97, 0.9, 0.75, 0.5])
    np.testing.assert_allclose(model.isf(s), model.qf(1 - s), rtol=1e-12)
    s = np.array([0.45, 0.42, 0.3, 1e-8, 1e-300])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = model.isf(s)
    never = 1 - model.lfp_p
    assert np.all(np.isinf(t[s <= never]))
    reached = s > never
    np.testing.assert_allclose(model.sf(t[reached]), s[reached], rtol=1e-11)
    if model.f0:
        assert np.all(model.isf([1.0, 1 - model.f0]) == 0.0)


def test_model_isf_refuses_a_probability_outside_the_unit_interval():
    model = sp.Weibull.from_params([10, 3], lfp_p=0.6)
    with pytest.warns(UserWarning):
        out = model.isf([-0.1, 0.5, 1.2])
    assert np.isnan(out[0]) and np.isnan(out[2]) and np.isfinite(out[1])
