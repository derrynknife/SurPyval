"""The sampler's log-posterior and gradient (``log_posterior_and_grad``).

Its value is held to the posterior the grid uses (``bound.log_prior`` plus
the public ``LogLikelihood``) to 1e-12, relative, and its gradient to
fourth-order central differences of that public posterior to 1e-6
(``rtol`` and ``atol``), at 20 random points, for every distribution
``fit_bayes`` takes, every kind of data, the offset, limited-failure and
zero-inflated forms and every kind of prior. The seeds were fixed before
any result was seen.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.bayes.fitting import build_bound
from surpyval.bayes.likelihood import LogLikelihood, log_posterior_and_grad
from surpyval.tests.bayes.test_likelihood import (
    CONTINUOUS,
    FORMS,
    KINDS,
    C,
    N,
    _point,
    _scaled,
)

#: The steps of the central differences; each component's reference is
#: the smaller step of the consecutive pair that agree best.
STEPS = (1e-2, 1e-3, 1e-4, 1e-5)


def _public(dist, data, bound):
    """The posterior the grid evaluates, vectorised in ``z``."""
    ll = LogLikelihood(dist, data, bound.offset, bound.lfp, bound.zi)

    def f(Z):
        Z = np.atleast_2d(Z)
        with np.errstate(all="ignore"):
            out = bound.log_prior(Z) + ll(bound.full_theta(Z))
        return np.where(np.isnan(out), -np.inf, out)

    return f


def _fd_at(f, z, h):
    d = len(z)
    E = np.eye(d) * h
    pts = np.concatenate([z + 2 * E, z + E, z - E, z - 2 * E])
    v = f(pts).reshape(4, d)
    return (-v[0] + 8 * v[1] - 8 * v[2] + v[3]) / (12 * h)


def _fd(f, z):
    """The converged central difference of each component, and how far
    it moved between its two steps (its error, roughly)."""
    est = np.array([_fd_at(f, z, h) for h in STEPS])
    moved = np.abs(np.diff(est, axis=0))
    best = np.argmin(np.where(np.isfinite(moved), moved, np.inf), axis=0)
    cols = np.arange(len(z))
    return est[best + 1, cols], moved[best, cols]


def _settled(fd, moved):
    """Whether the differences settled to the tolerance: where the public
    value itself is rounded to a step (an interval far in the upper tail
    of a limited-failure model, whose window loses its digits there) no
    difference is a reference for a gradient, and the point is skipped."""
    return bool(np.all(moved <= 1e-6 * (1 + np.abs(fd))))


def _check(dist, data, priors=None, n_points=20, seed=0, **forms):
    f = log_posterior_and_grad(data, dist, priors, **forms)
    bound = f.bound
    public = _public(dist, data, bound)
    rng = np.random.default_rng(seed)
    x_min = float(np.min(np.asarray(data.x, dtype=float)))
    checked = 0
    for _ in range(n_points):
        theta = _point(
            dist,
            rng,
            forms.get("offset", False),
            forms.get("lfp", False),
            forms.get("zi", False),
            x_min,
        )
        z = bound.to_phi(theta[None, :])[0]
        z = z[: bound.d] if len(z) > bound.d else z
        want = float(public(z)[0])
        value, gradient = f(z)
        if not np.isfinite(want):
            assert value == -np.inf
            continue
        np.testing.assert_allclose(value, want, rtol=1e-12, atol=0)
        assert np.all(np.isfinite(gradient))
        fd, moved = _fd(public, z)
        if not _settled(fd, moved):
            continue
        checked += 1
        np.testing.assert_allclose(gradient, fd, rtol=1e-6, atol=1e-6)
    assert checked >= n_points // 2, f"only {checked} points checked"
    return f


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("name", CONTINUOUS)
def test_value_and_gradient(name, kind):
    dist = getattr(sp, name)
    x, tl, tr = _scaled(dist)
    data = sp.SurpyvalData(**KINDS[kind](x, tl, tr))
    _check(dist, data)


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize(
    "name", [n for n in CONTINUOUS if getattr(sp, n).support[0] == 0]
)
def test_value_and_gradient_with_extra_parameters(name, form):
    dist = getattr(sp, name)
    x, tl, tr = _scaled(dist)
    data = sp.SurpyvalData(x=x, c=C, n=N, tl=tl, tr=tr)
    if name == "Beta" and "offset" in form:
        with pytest.raises(ValueError, match="offset"):
            log_posterior_and_grad(data, dist, **FORMS[form])
        return
    _check(dist, data, **FORMS[form])


def test_zero_inflated_zeros():
    data = sp.SurpyvalData(x=[0.0, 0.0, 1.5, 2.5, 4.0], c=[0, 0, 0, 1, 0])
    _check(sp.Weibull, data, zi=True)
    _check(sp.Weibull, data, zi=True, lfp=True)


DATA = sp.SurpyvalData(
    x=[2.1, 3.4, 4.05, 5.0, 6.2, 7.5, 9.0], c=[0, 0, 0, 1, 0, 1, 0]
)
OLD = sp.SurpyvalData(x=[2.5, 4.0, 6.5, 8.0], c=[0, 0, 0, 1])

PRIORS = {
    "around": lambda: {"beta": sp.prior.around(2, between=(1.2, 4))},
    "between both": lambda: {
        "alpha": sp.prior.between(3, 12),
        "beta": sp.prior.between(1, 5),
    },
    "flat": lambda: {"alpha": sp.prior.flat()},
    "distribution": lambda: {
        "beta": sp.prior.from_distribution(sp.Gamma.from_params([4.0, 2.0]))
    },
    "truncated distribution": lambda: {
        "beta": sp.prior.from_distribution(
            sp.Normal.from_params([2.0, 1.0]), truncate=True
        )
    },
    "fixed": lambda: {"beta": sp.prior.fixed(2.0)},
    "from model": lambda: sp.prior.from_model(
        sp.Weibull.fit([2.0, 3.0, 4.5, 5.5, 7.0, 8.0])
    ),
    "from data": lambda: sp.prior.from_data(OLD.x, c=OLD.c, weight=0.5),
    "B10 and beta": lambda: sp.prior.on_quantities(
        sp.Weibull,
        B10=sp.prior.around(2.5, between=(1.0, 5.0)),
        beta=sp.prior.between(1.2, 4),
    ),
    "R and median": lambda: sp.prior.on_quantities(
        sp.Weibull,
        R_at={3.0: sp.prior.between(0.6, 0.95)},
        median=sp.prior.around(5.0, between=(3.0, 8.0)),
    ),
    "mean and beta": lambda: sp.prior.on_quantities(
        sp.Weibull,
        mean=sp.prior.around(5.0, between=(3.0, 8.0)),
        beta=sp.prior.between(1.2, 4),
    ),
}


@pytest.mark.parametrize("which", sorted(PRIORS))
def test_every_kind_of_prior(which):
    f = _check(sp.Weibull, DATA, PRIORS[which]())
    assert f.hand_written


def test_quantity_without_a_traceable_form_is_said():
    f = log_posterior_and_grad(DATA, sp.Weibull, PRIORS["mean and beta"]())
    assert f.numeric_factors and "mean" in f.numeric_factors[0]
    g = log_posterior_and_grad(DATA, sp.Weibull, PRIORS["B10 and beta"]())
    assert g.numeric_factors == []


def test_fixed_parameter_drops_out_of_z():
    f = log_posterior_and_grad(DATA, sp.Weibull, fixed={"beta": 2.0})
    assert f.free_names == ["alpha"]
    value, gradient = f(np.array([np.log(5.0)]))
    assert gradient.shape == (1,)


def test_a_bound_prior_is_taken_as_it_is():
    bound = build_bound(
        sp.Weibull, DATA, PRIORS["around"](), False, False, False, None
    )
    f = log_posterior_and_grad(DATA, sp.Weibull, bound)
    assert f.bound is bound
    with pytest.raises(ValueError, match="pass the same"):
        log_posterior_and_grad(DATA, sp.Weibull, bound, lfp=True)


@pytest.mark.parametrize(
    "name", ["Weibull", "Exponential", "Normal", "LogNormal", "Gamma"]
)
def test_hand_written_gradient_equals_autograd(name):
    """The hand-written families against the autograd path they replace."""
    from autograd import value_and_grad

    from surpyval.bayes._lean import LeanLikelihood, _to_theta

    dist = getattr(sp, name)
    x, tl, tr = _scaled(dist)
    data = sp.SurpyvalData(x=x, c=C, n=N, tl=tl, tr=tr)
    f = log_posterior_and_grad(data, dist)
    assert f.hand_written
    ll = LeanLikelihood(dist, data)
    ts = f.bound.transforms
    slow = value_and_grad(
        lambda z: ll([_to_theta(ts[i], z[i]) for i in range(len(ts))])
    )
    rng = np.random.default_rng(1)
    for _ in range(20):
        theta = _point(dist, rng, False, False, False, 1.0)
        z = f.bound.to_phi(theta[None, :])[0]
        v, g = f(z)
        v0, g0 = slow(z)
        if not np.isfinite(v0):
            continue
        np.testing.assert_allclose(v, v0, rtol=1e-12)
        np.testing.assert_allclose(g, g0, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("name", CONTINUOUS)
def test_no_nan_at_extreme_z(name):
    """Far out on the unconstrained scale the value is finite or -inf, and
    the gradient is finite wherever the public posterior is."""
    dist = getattr(sp, name)
    x, tl, tr = _scaled(dist)
    data = sp.SurpyvalData(x=x, c=C, n=N, tl=tl, tr=tr)
    f = log_posterior_and_grad(data, dist)
    public = _public(dist, data, f.bound)
    grid = np.array(np.meshgrid(*[[-30.0, -8.0, 8.0, 30.0]] * f.d))
    for z in grid.reshape(f.d, -1).T:
        want = float(public(z)[0])
        value, gradient = f(z)
        assert not np.isnan(value)
        if np.isfinite(want):
            np.testing.assert_allclose(value, want, rtol=1e-12)
            assert np.all(np.isfinite(gradient)), (z, gradient)
        else:
            assert value == -np.inf


def test_refusals_say_what_to_do():
    f = log_posterior_and_grad(DATA, sp.Weibull)
    with pytest.raises(ValueError, match="2 values"):
        f(np.zeros(3))
    with pytest.raises(ValueError, match="closed form"):
        log_posterior_and_grad(
            sp.SurpyvalData(x=[0, 1, 1]), sp.Bernoulli, None
        )
    with pytest.raises(ValueError, match="from_params"):
        log_posterior_and_grad(
            DATA, sp.Weibull, fixed={"alpha": 5.0, "beta": 2.0}
        )
    with pytest.raises(ValueError, match="Uniform"):
        log_posterior_and_grad(DATA, sp.Uniform)
