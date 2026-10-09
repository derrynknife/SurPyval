"""A model with no offset, limited-failure or zero-inflation part gives
its distribution's own sf, ff and df, without the transforms (#642), and
its own qf, without them or a second check of the probabilities (#769)."""

import math
import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval.utils.validation import warn_outside_unit_interval

X = np.array([-1.0, 0.0, 0.5, 2.0, 7.5, 40.0, np.nan, np.inf])


@pytest.mark.parametrize(
    "dist, params",
    [
        (surv.Weibull, [10, 2]),
        (surv.LogNormal, [1, 0.5]),
        (surv.Exponential, [0.1]),
        (surv.Gamma, [2, 0.3]),
        (surv.Geometric, [0.3]),
    ],
)
@pytest.mark.parametrize("fn", ["sf", "ff", "df"])
def test_a_plain_model_is_its_distribution(dist, params, fn):
    model = dist.from_params(params)
    got = getattr(model, fn)(X)
    want = getattr(model.dist, fn)(X, *params)
    np.testing.assert_array_equal(got, want)
    # a scalar query gives a scalar, and a -0.0 below the support is 0.0
    value = getattr(model, fn)(-1.0)
    assert np.ndim(value) == 0
    assert math.copysign(1.0, float(value)) == 1.0


def test_the_transforms_still_apply_off_the_plain_model():
    base = surv.Weibull.from_params([10, 2])
    lfp = surv.Weibull.from_params([10, 2], lfp_p=0.8)
    zi = surv.Weibull.from_params([10, 2], f0=0.1)
    shifted = surv.Weibull.from_params([10, 2], gamma=1.5)
    x = np.array([0.5, 2.0, 7.5])
    np.testing.assert_allclose(lfp.ff(x), 0.8 * base.ff(x))
    np.testing.assert_allclose(zi.ff(x), 0.1 + 0.9 * base.ff(x))
    np.testing.assert_allclose(shifted.sf(x), base.sf(x - 1.5))


# --- qf and random (#769) ----------------------------------------------


def _qf_before_769(model, p):
    """``Parametric.qf`` as it was before #769, every step taken."""
    if isinstance(p, list):
        p = np.array(p)
    u = np.atleast_1d(np.asarray(p, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.clip((u - model.f0) / (model.lfp_p - model.f0), 0.0, 1.0)
        q = model.gamma + model.dist.qf(base, *model.params)
    at_zero = (model.f0 > 0) or getattr(model.dist, "discrete", False)
    q = np.where(at_zero & (u <= model.f0), 0.0, q)
    q = np.where((model.lfp_p < 1) & (u >= model.lfp_p), np.inf, q)
    q = np.where(warn_outside_unit_interval(u), np.nan, q)
    q = np.asarray(q, dtype=float)
    return q[0] if np.ndim(p) == 0 else q


def _outcome(f):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = f()
    messages = [str(w.message) for w in caught]
    return type(out), np.shape(out), np.asarray(out).tobytes(), messages


QF_DISTS = [
    (surv.Weibull, [10, 2]),
    (surv.LogNormal, [1, 0.5]),
    (surv.Normal, [0.0, 1.0]),
    (surv.Exponential, [0.1]),
    (surv.Uniform, [2, 7]),
    (surv.Poisson, [4.0]),
    (surv.Geometric, [0.3]),
    (surv.Binomial, [10, 0.3]),  # its qf is decorated again
]
QF_VARIANTS = [
    {},
    {"lfp_p": 0.7},
    {"f0": 0.1},
    {"lfp_p": 0.8, "f0": 0.05},
    {"gamma": 2.5},
    {"gamma": 2.5, "lfp_p": 0.8, "f0": 0.05},
]
INSIDE_U = [0.0, -0.0, 1.0, 0.05, 0.1, 0.7, 0.8, 5e-324, 1 - 1e-16]
OUTSIDE_U = [np.nan, -0.1, 1.5, np.inf]


def _qf_queries():
    rng = np.random.default_rng(769)
    queries = INSIDE_U + OUTSIDE_U + [np.array(0.3), [0.1, 0.5], []]
    queries += [np.array([[0.1, 0.2], [0.3, 0.99]]), np.array([0, 1])]
    for i in range(24):
        u = rng.random(int(rng.integers(1, 30)))
        if i % 3:
            idx = rng.integers(0, u.size, size=int(rng.integers(1, 4)))
            pool = INSIDE_U + (OUTSIDE_U if i % 3 == 2 else [])
            u[idx] = rng.choice(pool, size=idx.size)
        queries.append(u.reshape(1, -1) if i % 4 == 0 else u)
    return queries


@pytest.mark.parametrize("dist, params", QF_DISTS)
def test_769_qf_is_unchanged_by_its_fast_path(dist, params):
    # The same values bit for bit, types, shapes and warnings as when
    # every step was taken, inside and outside [0, 1] and with NaN.
    for variant in QF_VARIANTS:
        try:
            model = dist.from_params(params, **variant)
        except (ValueError, TypeError):
            continue  # an offset or LFP the family does not take
        for u in _qf_queries():
            assert _outcome(lambda: model.qf(u)) == _outcome(
                lambda: _qf_before_769(model, u)
            ), (dist.name, variant, u)


@pytest.mark.parametrize("dist, params", QF_DISTS[:4])
def test_769_a_plain_model_qf_is_its_distribution_qf(dist, params):
    model = dist.from_params(params)
    u = np.random.default_rng(1).random(1000)
    assert model.qf(u).tobytes() == dist.qf(u, *params).tobytes()
    if dist.support[0] == 0:
        # but for the sign of a zero, which the offset always made 0.0
        assert math.copysign(1.0, float(model.qf(-0.0))) == 1.0


def test_769_random_draws_as_before():
    from scipy.stats import uniform

    for variant in QF_VARIANTS:
        model = surv.Weibull.from_params([10, 2], **variant)
        for size in [1, 7, (2, 3)]:
            for seed in [None, 3]:
                np.random.seed(11)
                got = model.random(size, random_state=seed)
                after = np.random.random_sample()
                np.random.seed(11)
                state = None if seed is None else np.random.default_rng(seed)
                u = uniform.rvs(size=size, random_state=state)
                if variant:
                    want = np.reshape(_qf_before_769(model, u), size)
                else:
                    want = model.dist.qf(u, *model.params) + model.gamma
                assert got.tobytes() == want.tobytes()
                assert got.shape == np.shape(want)
                assert np.random.random_sample() == after
