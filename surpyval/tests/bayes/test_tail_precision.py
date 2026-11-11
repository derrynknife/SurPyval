"""The predictive functions of a Bayesian fit in the tails.

A failure probability of 1e-12 is as much a number as one of 0.5: ``ff``
and ``Hf`` must keep their digits there (they are not ``1 - sf`` and
``-log(sf)``, which keep none below about 1e-8), ``sf`` must keep its own
far past the data, ``qf`` must invert both ends, and the credible
intervals on these functions must do the same. Every engine is covered:
the closed forms (Exponential with a Gamma prior, whose predictive is a
Lomax; Normal with a Normal prior on its mean), the grid, the Laplace
approximation and a posterior of draws from another sampler, with the
offset, limited-failure and zero-inflated forms.

References do not depend on the platform: exact rational arithmetic
(``fractions.Fraction``) where the predictive has a closed form, and
60-digit decimal arithmetic (``decimal``) for the mixture over a
posterior's points, whose own definition is the reference
(``ff = sum(w_i ff_i)``).
"""

import functools
import math
import warnings
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
from scipy import optimize, stats

import surpyval as sp
from surpyval import prior
from surpyval.tests._helpers import (
    KS_BAND_20000,
    ks_distance,
    ks_distance_bracketed,
)

#: The failure probabilities of the lower-tail checks, and the survival
#: probabilities of the upper-tail ones.
TAILS = (1e-12, 1e-10, 1e-8, 1e-4)
#: Relative error allowed against an exact reference.
RTOL = 1e-12
#: Seed of every draw below, fixed before any result was seen.
SEED = 20261111
N_DRAWS = 20_000

RATE = {"failure_rate": sp.Gamma.from_params([2.0, 500.0])}
LIVES = [100.0, 250.0, 400.0]
#: The Exponential-Gamma posterior of LIVES under RATE: Gamma(2 + 3,
#: 500 + 750), so the predictive is a Lomax of shape 5 and scale 1250.
A, B = 5, 1250

# 40 pump seals: three failures, 37 running
X = np.r_[2100.0, 3400.0, 4050.0, np.linspace(1500, 6000, 37)]
C = np.r_[0, 0, 0, np.ones(37)]
SHAPE = {"beta": prior.around(2, between=(1.5, 3))}

#: Every continuous family fit_bayes takes, with parameters to simulate
#: 30 lives from (seed 7).
FAMILIES = {
    "Weibull": (sp.Weibull, [100.0, 2.0]),
    "Exponential": (sp.Exponential, [0.01]),
    "Gamma": (sp.Gamma, [3.0, 0.03]),
    "LogNormal": (sp.LogNormal, [4.5, 0.4]),
    "Normal": (sp.Normal, [100.0, 10.0]),
    "LogLogistic": (sp.LogLogistic, [100.0, 4.0]),
    "Logistic": (sp.Logistic, [100.0, 8.0]),
    "Gumbel": (sp.Gumbel, [100.0, 8.0]),
    "GumbelLEV": (sp.GumbelLEV, [100.0, 8.0]),
    "Rayleigh": (sp.Rayleigh, [80.0]),
    "ExpoWeibull": (sp.ExpoWeibull, [100.0, 2.0, 1.5]),
    "Beta": (sp.Beta, [3.0, 5.0]),
}


def _family_fit(name, method):
    dist, theta = FAMILIES[name]
    x = dist.random(30, *theta, random_state=7)
    return dist.fit_bayes(x, method=method)


def _draws_model(dist=sp.Weibull, centre=(100.0, 2.0)):
    """A posterior of 400 weighted draws, as another sampler gives."""
    rng = np.random.default_rng(1)
    draws = np.c_[
        rng.lognormal(np.log(centre[0]), 0.1, 400),
        rng.lognormal(np.log(centre[1]), 0.1, 400),
    ]
    weights = rng.uniform(0.5, 1.5, 400)
    return sp.bayes.BayesianParametric.from_draws(
        dist, draws, dist.parameter_names, weights=weights
    )


def _offset_model():
    x = np.array([520.0, 610, 700, 760, 830, 905, 990, 1100, 1250, 1400])
    return sp.Weibull.fit_bayes(
        x,
        offset=True,
        priors={
            "gamma": prior.between(200, 500),
            "beta": prior.between(0.8, 4),
            "alpha": prior.between(100, 2000),
        },
    )


def _lfp_model():
    x = sp.Weibull.random(30, 100, 2, random_state=4)
    return sp.Weibull.fit_bayes(
        np.minimum(x, 120),
        np.where(x > 120, 1, 0),
        lfp=True,
        priors={**SHAPE, "lfp_p": sp.Beta.from_params([4.0, 1.0])},
    )


def _zi_model():
    x = np.r_[np.zeros(4), [120.0, 200, 260, 330, 410, 520, 600, 700]]
    return sp.Weibull.fit_bayes(
        x,
        zi=True,
        priors={"f0": prior.between(0.1, 0.5), "beta": prior.between(0.8, 4)},
    )


#: Every engine and form: name -> builder.
BUILDERS = {
    "conjugate-exponential-gamma": lambda: sp.Exponential.fit_bayes(
        LIVES, priors=RATE
    ),
    "conjugate-normal-normal": lambda: sp.Normal.fit_bayes(
        [4.1, 5.3, 3.8, 6.0, 5.1, 4.7],
        priors={
            "mu": sp.Normal.from_params([3.0, 2.0]),
            "sigma": prior.fixed(0.8),
        },
    ),
    "grid-seals": lambda: sp.Weibull.fit_bayes(X, C, priors=SHAPE),
    "laplace-Weibull": lambda: _family_fit("Weibull", "laplace"),
    "laplace-ExpoWeibull": lambda: _family_fit("ExpoWeibull", "laplace"),
    "draws-Weibull": _draws_model,
    "draws-LogNormal": lambda: _draws_model(sp.LogNormal, (4.5, 0.4)),
    "draws-Normal": lambda: _draws_model(sp.Normal, (100.0, 10.0)),
    "draws-LogLogistic": lambda: _draws_model(sp.LogLogistic, (100.0, 4.0)),
    "grid-offset": _offset_model,
    "grid-lfp": _lfp_model,
    "grid-zi": _zi_model,
    **{
        f"grid-{name}": functools.partial(_family_fit, name, "grid")
        for name in FAMILIES
        if name != "ExpoWeibull"
    },
}
_CACHE: dict = {}


def model_of(name):
    if name not in _CACHE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _CACHE[name] = BUILDERS[name]()
    return _CACHE[name]


#: The models whose predictive is the mixture over the posterior's points
#: (every one but the two closed forms).
MIXTURES = [n for n in BUILDERS if not n.startswith("conjugate")]


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
def _point_values(model, fname, x):
    """``fname`` ('ff' or 'sf') of every posterior point at the time
    ``x``, from the distribution's own function, with the offset,
    limited-failure and zero-inflated forms."""
    post = model.posterior
    th = {n: post.points[:, i] for i, n in enumerate(post.parameter_names)}
    dp = [th[n] for n in model.dist.parameter_names]
    n = len(post.weights)
    gamma = th.get("gamma", np.zeros(n))
    p = th.get("lfp_p", np.ones(n))
    f0 = th.get("f0", np.zeros(n))
    xg = x - gamma
    below = xg < model.dist.support[0]
    with np.errstate(all="ignore"):
        if fname == "ff":
            base = np.where(below, 0.0, model.dist.ff(xg, *dp))
            return f0 + (p - f0) * base
        base = np.where(below, 1.0, model.dist.sf(xg, *dp))
        return (1 - p) + (p - f0) * base


def mixture(model, fname, x):
    """The predictive ``ff`` or ``sf`` at ``x`` by its definition, the
    posterior average of every point's, summed exactly (``math.fsum``)."""
    v = _point_values(model, fname, float(x))
    return math.fsum((model.posterior.weights * v).tolist())


def time_at(model, fname, target):
    """The time at which the mixture's ``fname`` is ``target``, found
    from the definition. The model's ``qf`` is used only as a first guess
    for the bracket; the root is that of the definition."""
    sign = 1.0 if fname == "ff" else -1.0

    def g(t):
        v = mixture(model, fname, t)
        return sign * (math.log(max(v, 1e-300)) - math.log(target))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        guess = float(model.qf(target if fname == "ff" else 1 - target))
    if not np.isfinite(guess):
        raise AssertionError(f"no first guess for {fname} = {target}")
    if guess > 0:
        # on the log of time: the bracket may span many decades
        lo = hi = math.log(guess)
        d = 1e-6
        while g(math.exp(lo)) > 0:
            lo -= d
            d *= 4
        d = 1e-6
        while g(math.exp(hi)) < 0:
            hi += d
            d *= 4
        if lo == hi:
            return guess
        root = optimize.brentq(
            lambda v: g(math.exp(v)), lo, hi, xtol=1e-300, rtol=1e-15
        )
        return float(math.exp(root))
    d = max(abs(guess) * 1e-6, 1e-12)
    lo = hi = guess
    while g(lo) > 0:
        lo -= d
        d *= 4
    d = max(abs(guess) * 1e-6, 1e-12)
    while g(hi) < 0:
        hi += d
        d *= 4
    if lo == hi:
        return guess
    return float(optimize.brentq(g, lo, hi, xtol=1e-300, rtol=1e-15))


def _dec_phi(z):
    """The standard normal CDF at the Decimal ``z`` by its Taylor series
    about 0, carried with enough extra digits (``z**2 / ln 10``) for the
    cancellation among its terms, which grow to about ``exp(z**2 / 2)``,
    so that it is good to about 1e-60 absolute."""
    with localcontext() as ctx:
        ctx.prec = 70 + int(float(z * z) / 2.3) + 10
        return +_dec_phi_series(+z)


def _dec_phi_series(z):
    term = z
    total = z
    k = 0
    while True:
        k += 1
        term = -term * z * z / (2 * k)
        add = term / (2 * k + 1)
        total += add
        if abs(add) < Decimal(10) ** -75:
            break
    pi = Decimal(
        "3.14159265358979323846264338327950288419716939937510582097494"
        "4592307816406286208998628034825342117067982148086513282306647"
    )
    return Decimal("0.5") + total / (2 * pi).sqrt()


def _dec_point(dist, fname, x, theta):
    """``ff`` or ``sf`` of ``dist`` at ``x`` in decimal arithmetic."""
    name = dist.name
    x = Decimal(x)
    th = [Decimal(float(v)) for v in theta]
    one = Decimal(1)
    if name in ("Weibull", "Exponential", "Rayleigh"):
        if name == "Exponential":
            z = th[0] * x
        elif name == "Rayleigh":
            z = x * x / (2 * th[0] * th[0])
        else:
            z = (th[1] * (x.ln() - th[0].ln())).exp()
        s = (-z).exp()
        return s if fname == "sf" else one - s
    if name in ("Normal", "LogNormal"):
        z = ((x.ln() if name == "LogNormal" else x) - th[0]) / th[1]
        return _dec_phi(z if fname == "ff" else -z)
    if name == "LogLogistic":
        r = (th[1] * (x.ln() - th[0].ln())).exp()  # (x / alpha) ** beta
        return r / (one + r) if fname == "ff" else one / (one + r)
    raise ValueError(name)


def decimal_mixture(model, x):
    """(ff, sf, Hf) of the mixture at ``x`` in 60-digit decimal
    arithmetic, rounded to the nearest double. ``Hf`` is ``-log(1 - ff)``
    where ``ff`` is below one half and ``-log(sf)`` above, the
    definitions the model documents."""
    post = model.posterior
    cols = [post.parameter_names.index(n) for n in model.dist.parameter_names]
    with localcontext() as ctx:
        ctx.prec = 60
        F = Decimal(0)
        S = Decimal(0)
        for w, row in zip(post.weights, post.points[:, cols]):
            w = Decimal(float(w))
            F += w * _dec_point(model.dist, "ff", x, row)
            S += w * _dec_point(model.dist, "sf", x, row)
        H = -(1 - F).ln() if F < Decimal("0.5") else -S.ln()
        return float(F), float(S), float(H)


def rel(got, want):
    return abs(float(got) - want) / abs(want)


# ---------------------------------------------------------------------------
# The closed form: Exponential with a Gamma prior is a Lomax
# ---------------------------------------------------------------------------
def _lomax_time(p):
    """A time (a double) at which the Lomax's ff is about ``p``."""
    return B * math.expm1(-math.log1p(-p) / A)


def _lomax_exact(x):
    """(ff, sf) of the Lomax at the double ``x``, exactly."""
    s = Fraction(B) ** A / (Fraction(B) + Fraction(x)) ** A
    return 1 - s, s


def _minus_log1m(f):
    """``-log(1 - f)`` of a small rational ``f``, by its series."""
    total = Fraction(0)
    k = 0
    while True:
        k += 1
        term = f**k / k
        total += term
        if term < Fraction(1, 10**40):
            return total


@pytest.mark.parametrize("p", TAILS)
def test_lomax_lower_tail_is_exact(p):
    model = model_of("conjugate-exponential-gamma")
    x = _lomax_time(p)
    f, s = _lomax_exact(x)
    assert rel(model.ff(x), float(f)) <= RTOL
    assert rel(model.Hf(x), float(_minus_log1m(f))) <= RTOL
    assert rel(model.sf(x), float(s)) <= RTOL
    # the hazard and density of a Lomax: a / (b + x), and that times sf
    h = Fraction(A) / (Fraction(B) + Fraction(x))
    assert rel(model.hf(x), float(h)) <= RTOL
    assert rel(model.df(x), float(h * s)) <= RTOL


@pytest.mark.parametrize("q", TAILS)
def test_lomax_upper_tail_is_exact(q):
    model = model_of("conjugate-exponential-gamma")
    x = B * (q ** (-1.0 / A) - 1.0)
    f, s = _lomax_exact(x)
    assert rel(model.sf(x), float(s)) <= RTOL
    assert rel(model.ff(x), float(f)) <= RTOL
    # -log(s) of the exact s, in decimal arithmetic
    with localcontext() as ctx:
        ctx.prec = 60
        H = -(Decimal(s.numerator) / Decimal(s.denominator)).ln()
        assert rel(model.Hf(x), float(H)) <= RTOL


@pytest.mark.parametrize("p", TAILS)
@pytest.mark.parametrize("on", ["ff", "Hf"])
def test_lomax_credible_interval_keeps_its_digits(p, on):
    """The rate's posterior is Gamma(5, 1250): the interval on ff at x is
    1 - exp(-rate x) at the rate's quantiles, about rate * x here."""
    model = model_of("conjugate-exponential-gamma")
    x = _lomax_time(p)
    rate = stats.gamma(A, scale=1.0 / B).ppf([0.025, 0.5, 0.975])
    want = -np.expm1(-rate * x) if on == "ff" else rate * x
    np.testing.assert_allclose(
        model.cri(x, on=on), want[[0, 2]], rtol=1e-11, atol=0
    )
    np.testing.assert_allclose(
        model.cri(x, on=on, point=True), want[1], rtol=1e-11, atol=0
    )


def test_normal_normal_closed_form_tails():
    """Normal data with sigma known and a Normal prior on mu: the
    predictive is Normal(m, sqrt(sigma^2 + s^2))."""
    model = model_of("conjugate-normal-normal")
    m, s = model.posterior.exact["params"]
    sd = math.hypot(0.8, s)
    for z in (-7.0, -5.0, 5.0, 7.0):
        x = m + z * sd
        with localcontext() as ctx:
            ctx.prec = 60
            zz = (Decimal(x) - Decimal(m)) / Decimal(sd)
            F, S = _dec_phi(zz), _dec_phi(-zz)
            H = float(-(1 - F).ln()) if z < 0 else float(-S.ln())
            F, S = float(F), float(S)
        assert rel(model.ff(x), F) <= 1e-11
        assert rel(model.sf(x), S) <= 1e-11
        assert rel(model.Hf(x), H) <= 1e-11


# ---------------------------------------------------------------------------
# The mixture over a posterior's points: grid, Laplace and draws
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", MIXTURES)
@pytest.mark.parametrize("p", TAILS)
def test_lower_tail_is_the_average_of_the_points(name, p):
    model = model_of(name)
    if model.zi:
        pytest.skip("a zero-inflated ff is never below its mass at 0")
    x = time_at(model, "ff", p)
    f = mixture(model, "ff", x)
    assert f == pytest.approx(p, rel=1e-6)
    assert rel(model.ff(x), f) <= RTOL
    assert rel(model.Hf(x), -math.log1p(-f)) <= RTOL
    # sf is within a rounding of 1 - ff there
    assert abs(float(model.sf(x)) - (1 - f)) <= 1e-13


@pytest.mark.parametrize("name", MIXTURES)
@pytest.mark.parametrize("q", TAILS)
def test_upper_tail_is_the_average_of_the_points(name, q):
    model = model_of(name)
    if model.lfp:
        pytest.skip("a limited-failure sf never falls below 1 - lfp_p")
    x = time_at(model, "sf", q)
    s = mixture(model, "sf", x)
    assert s == pytest.approx(q, rel=1e-6)
    assert rel(model.sf(x), s) <= RTOL
    assert rel(model.Hf(x), -math.log(s)) <= RTOL
    assert abs(float(model.ff(x)) - (1 - s)) <= 1e-13


@pytest.mark.parametrize(
    "name",
    [
        "grid-Exponential",
        "grid-Rayleigh",
        "grid-seals",
        "laplace-Weibull",
        "draws-Weibull",
        "draws-LogNormal",
        "draws-Normal",
        "draws-LogLogistic",
    ],
)
@pytest.mark.parametrize("tail", ["lower", "upper"])
@pytest.mark.parametrize("p", [1e-12, 1e-8])
def test_tails_against_decimal_arithmetic(name, tail, p):
    """ff, sf and Hf against the mixture computed in 60-digit decimal
    arithmetic from the families' closed forms (the grids of one and two
    parameters, the Laplace approximation, and draws for the families
    whose closed form needs the normal integral or a logistic)."""
    model = model_of(name)
    x = time_at(model, "ff" if tail == "lower" else "sf", p)
    F, S, H = decimal_mixture(model, x)
    small, want = ("ff", F) if tail == "lower" else ("sf", S)
    assert want == pytest.approx(p, rel=1e-6)
    assert rel(getattr(model, small)(x), want) <= RTOL
    assert rel(model.Hf(x), H) <= RTOL


def test_hf_stays_finite_where_sf_underflows():
    """Past the point where every posterior point's sf underflows, the
    cumulative hazard is still the log of their average."""
    model = model_of("grid-Exponential")
    # the first doubling of 1.5e5 at which the predictive sf underflows
    x = 1.5e5
    while float(model.sf(x)) > 0.0:
        x *= 2
    assert float(model.sf(x)) == 0.0
    post = model.posterior
    with localcontext() as ctx:
        ctx.prec = 60
        S = sum(
            Decimal(float(w)) * (-Decimal(float(r)) * Decimal(x)).exp()
            for w, r in zip(post.weights, post.points[:, 0])
        )
        want = float(-S.ln())
    assert np.isfinite(want)
    assert rel(model.Hf(x), want) <= 1e-11


@pytest.mark.parametrize("name", MIXTURES)
def test_nothing_has_failed_at_the_lower_edge(name):
    """ff is exactly 0 and sf exactly 1 below the support (they were the
    rounding of the weights' sum away, either side of it)."""
    model = model_of(name)
    lo = model.dist.support[0]
    x = -1e9 if not np.isfinite(lo) else lo - 1.0
    if model.offset:
        # below every posterior point's own offset
        x = float(np.min(model._theta()["gamma"])) + lo - 1.0
    if model.dist.name == "Beta":
        x = -0.5
    assert float(model.ff(x)) == 0.0
    assert float(model.Hf(x)) == 0.0
    assert float(model.sf(x)) == 1.0
    assert not np.signbit(model.Hf(x))
    t = np.linspace(
        time_at(model, "sf", 0.999) if not model.zi else 1.0,
        time_at(model, "sf", 0.5) if not model.lfp else 100.0,
        7,
    )
    assert np.all(model.ff(t) >= 0) and np.all(model.sf(t) <= 1)


# ---------------------------------------------------------------------------
# Credible intervals in the tails
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["grid-seals", "draws-Weibull"])
@pytest.mark.parametrize("p", TAILS)
def test_credible_interval_on_ff_keeps_its_digits(name, p):
    """The interval on ff at a time where it is tiny is a posterior
    quantile of the points' own ff: positive, ordered, inside their
    range, the same asked on ff or Hf (which is ff there), and still the
    complement of the interval on sf."""
    model = model_of(name)
    x = time_at(model, "ff", p)
    lo, hi = model.cri(x, on="ff")
    mid = float(model.cri(x, on="ff", point=True))
    values = _point_values(model, "ff", x)
    assert values.min() <= lo < mid < hi <= values.max()
    assert lo > 0
    want = model.posterior.wquantile(values, [0.025, 0.5, 0.975])
    if model.method == "draws":
        # a weighted quantile of the points' ff, to a rounding
        np.testing.assert_allclose([lo, mid, hi], want, rtol=1e-12)
    else:
        # The grid's quantile spreads each point's mass over its cell, on
        # the scale of sf; taken on the scale of ff itself it is another
        # approximation of the same posterior quantile, a few percent
        # away where ff spans decades between neighbouring points (5.5%
        # for the lower bound at 1e-12, where the interval spans 13
        # decades). The two must agree to a small part of the interval's
        # own width on the log scale.
        width = math.log(hi) - math.log(lo)
        gap = np.abs(np.log([lo, mid, hi]) - np.log(want))
        assert np.all(gap <= 0.01 * width), (gap, width)
    H = model.cri(x, on="Hf")
    np.testing.assert_allclose(H, -np.log1p(-np.array([lo, hi])), rtol=1e-12)
    s_lo, s_hi = model.cri(x, on="sf")
    assert abs((1 - s_hi) - lo) <= 1e-13 and abs((1 - s_lo) - hi) <= 1e-13
    # one-sided bounds are the same quantiles
    assert model.cri(x, on="ff", alpha_ci=0.025, bound="lower") == lo
    assert model.cri(x, on="ff", alpha_ci=0.025, bound="upper") == hi


@pytest.mark.parametrize("name", ["grid-seals", "draws-Weibull"])
def test_credible_interval_on_Hf_in_the_upper_tail(name):
    model = model_of(name)
    x = time_at(model, "sf", 1e-10)
    s_lo, s_hi = model.cri(x, on="sf")
    np.testing.assert_allclose(
        model.cri(x, on="Hf"), -np.log([s_hi, s_lo]), rtol=1e-12
    )


def test_compare_to_on_a_tiny_failure_probability():
    """Two posteriors compared on ff where it is about 1e-12: the answer
    is that of comparing the points' own ff, the complement of comparing
    on sf at an ordinary time, and not a coin toss between rounded
    values."""
    a = model_of("draws-Weibull")
    rng = np.random.default_rng(2)
    draws = np.c_[
        rng.lognormal(np.log(110), 0.1, 400),
        rng.lognormal(np.log(2), 0.1, 400),
    ]
    b = sp.bayes.BayesianParametric.from_draws(
        sp.Weibull, draws, ["alpha", "beta"]
    )
    x = time_at(a, "ff", 1e-12)
    fa, fb = _point_values(a, "ff", x), _point_values(b, "ff", x)
    wa, wb = a.posterior.weights, b.posterior.weights
    want = float(wa @ ((fb[None, :] < fa[:, None]) @ wb))
    got = a.compare_to(b, on="ff", x=x).probability
    assert got == pytest.approx(want, abs=1e-12)
    mle = sp.Weibull.fit(sp.Weibull.random(60, 110, 2.0, random_state=2))
    est = float(sp.Weibull.ff(x, *mle.params))
    r = a.compare_to(mle, on="ff", x=x)
    # the other model's standard error is that of its ff, not of noise
    h = 1e-6 * mle.params
    grad = np.array(
        [
            (
                sp.Weibull.ff(x, mle.params[0] + h[0], mle.params[1])
                - sp.Weibull.ff(x, mle.params[0] - h[0], mle.params[1])
            )
            / (2 * h[0]),
            (
                sp.Weibull.ff(x, mle.params[0], mle.params[1] + h[1])
                - sp.Weibull.ff(x, mle.params[0], mle.params[1] - h[1])
            )
            / (2 * h[1]),
        ]
    )
    se = float(np.sqrt(grad @ mle.covariance() @ grad))
    want = float(wa @ stats.norm.cdf((fa - est) / se))
    assert r.probability == pytest.approx(want, rel=1e-6)


# ---------------------------------------------------------------------------
# qf in both tails
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(BUILDERS))
def test_qf_round_trips_in_the_lower_tail(name):
    model = model_of(name)
    if model.zi:
        pytest.skip("a zero-inflated ff is never below its mass at 0")
    if name.startswith("conjugate"):
        x = float(model.qf(1e-12))
    else:
        x = time_at(model, "ff", 1e-12)
    p = float(model.ff(x))
    assert p == pytest.approx(1e-12, rel=1e-6)
    assert float(model.qf(p)) == pytest.approx(x, rel=1e-9)


def assert_solves(fn, t, target, rtol=1e-9):
    """``fn(t)`` is ``target`` to ``rtol`` at each ``t``, or as near as a
    double time allows: ``target`` lies between ``fn`` at the doubles
    either side of ``t`` (near the upper edge of a Beta, at 1, the
    doubles are 1.1e-16 apart, and a probability of 1e-12 there moves by
    about 1e-8 relative from one to the next)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    target = np.atleast_1d(np.asarray(target, dtype=float))
    got = np.asarray(fn(t), dtype=float)
    a = np.asarray(fn(np.nextafter(t, -np.inf)), dtype=float)
    b = np.asarray(fn(np.nextafter(t, np.inf)), dtype=float)
    close = np.abs(got - target) <= rtol * np.abs(target)
    between = (np.minimum(a, b) <= target) & (target <= np.maximum(a, b))
    assert np.all(close | between), (t, got, target)


@pytest.mark.parametrize("name", list(BUILDERS))
def test_qf_round_trips_in_the_upper_tail(name):
    """At ``p = 1 - 2**-40`` (``1 - p`` is exact in doubles, about
    9.1e-13), qf solves ``sf(t) = 1 - p`` and gives the time at which the
    posterior average of the points' sf is that."""
    model = model_of(name)
    if model.lfp:
        pytest.skip("a limited-failure ff never reaches 1 - 1e-12")
    s = 2.0**-40
    t = float(model.qf(1 - s))
    assert np.isfinite(t)
    assert_solves(model.sf, t, s)
    if not name.startswith("conjugate"):
        x = time_at(model, "sf", s)
        assert t == pytest.approx(x, rel=1e-9)


@pytest.mark.parametrize("name", list(BUILDERS))
def test_qf_inverts_ff_over_the_whole_range(name):
    model = model_of(name)
    lower = np.array([1e-12, 1e-10, 1e-8, 1e-4, 0.01, 0.3, 0.5])
    upper = 1 - lower[:-1]
    if model.zi:
        f0 = float(model.ff(0.0))
        lower = lower[lower > f0]
    if model.lfp:
        top = float(model.ff(np.inf))
        upper = upper[upper < top]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t_lo = model.qf(lower)
        t_hi = model.qf(upper)
    assert np.all(np.isfinite(t_lo)) and np.all(np.isfinite(t_hi))
    assert_solves(model.ff, t_lo, lower)
    assert_solves(model.sf, t_hi, 1 - upper)
    assert np.all(np.diff(np.r_[t_lo, t_hi[::-1]]) > 0)


def test_qf_of_a_limited_failure_population_above_its_proportion():
    model = model_of("grid-lfp")
    top = float(model.ff(np.inf))
    assert top < 1
    assert model.qf(min(1.0, top + 1e-6)) == np.inf
    t = float(model.qf(top * (1 - 1e-9)))
    assert np.isfinite(t)
    assert float(model.ff(t)) == pytest.approx(top * (1 - 1e-9), rel=1e-12)


def test_qf_of_a_zero_inflated_model_at_its_mass():
    model = model_of("grid-zi")
    f0 = float(model.ff(0.0))
    assert model.qf(f0 * 0.5) == 0.0
    t = float(model.qf(f0 + 1e-12))
    assert t > 0
    assert float(model.ff(t)) - f0 == pytest.approx(1e-12, rel=1e-3)


# ---------------------------------------------------------------------------
# random: one uniform per life, and still the predictive
# ---------------------------------------------------------------------------
def _uniforms_used(model, size):
    """How many doubles ``random(size)`` took from its generator."""
    used = np.random.default_rng(SEED)
    model.random(size, random_state=used)
    for n in range(0, 4 * size + 2):
        fresh = np.random.default_rng(SEED)
        fresh.random(n)
        if fresh.bit_generator.state == used.bit_generator.state:
            return n
    raise AssertionError("more than four uniforms per life")


#: The models whose predictive ``ff`` is cheap enough to evaluate at
#: every draw, to check the bracketed distance against the plain one.
CHEAP = [
    "conjugate-exponential-gamma",
    "conjugate-normal-normal",
    "draws-Weibull",
    "draws-LogNormal",
    "draws-Normal",
    "draws-LogLogistic",
    "grid-Exponential",
    "grid-Rayleigh",
]


@pytest.mark.parametrize("name", CHEAP)
def test_bracketed_ks_is_the_supremum_over_every_draw(name):
    """The KS check below takes the supremum over all 20,000 draws by
    bracketing ``ff`` between the times at which it is evaluated; where
    evaluating it at every draw is cheap, the two are the same number."""
    model = model_of(name)
    draws = model.random(N_DRAWS, random_state=SEED)
    calls = []

    def ff(t):
        calls.append(np.size(t))
        return model.ff(t)

    got = ks_distance_bracketed(draws, ff)
    assert got == ks_distance(draws, model.ff)
    # and from far fewer than the 40,001 times the plain one evaluates
    assert sum(calls) < 4_000


def _step_ff(t):
    """A failure function with a jump of 0.3 at 0 and an Exponential of
    mean 10 after it, reaching 0.9: a tenth never fails."""
    t = np.asarray(t, dtype=float)
    with np.errstate(over="ignore"):
        return np.where(t < 0, 0.0, 0.3 + 0.6 * -np.expm1(-t / 10))


def test_bracketed_ks_with_ties_jumps_and_infinite_lives():
    rng = np.random.default_rng(SEED)
    u = rng.random(N_DRAWS)
    draws = np.where(
        u < 0.3, 0.0, np.where(u < 0.9, rng.exponential(10, N_DRAWS), np.inf)
    )
    assert ks_distance_bracketed(draws, _step_ff) == ks_distance(
        draws, _step_ff
    )
    # a curve some way off: the supremum is found where it is
    for shifted in (lambda t: _step_ff(t * 1.2), lambda t: _step_ff(t - 5)):
        want = ks_distance(draws, shifted)
        assert want > KS_BAND_20000
        assert ks_distance_bracketed(draws, shifted) == want


def test_bracketed_ks_refuses_a_curve_that_falls():
    """The brackets hold only for a curve that never falls: one that
    does (here a survival function passed by mistake) is an error, not a
    distance."""
    draws = np.random.default_rng(SEED).exponential(10, 1000)
    with pytest.raises(AssertionError, match="falls"):
        ks_distance_bracketed(draws, lambda t: np.exp(-np.asarray(t) / 10))


@pytest.mark.parametrize("name", list(BUILDERS))
def test_random_takes_one_uniform_per_life(name):
    assert _uniforms_used(model_of(name), 50) == 50


@pytest.mark.parametrize("name", list(BUILDERS))
def test_random_follows_the_predictive(name):
    model = model_of(name)
    draws = model.random(N_DRAWS, random_state=SEED)
    assert draws.shape == (N_DRAWS,)
    assert not np.isnan(draws).any()
    d = ks_distance_bracketed(draws, model.ff)
    assert d <= KS_BAND_20000, (
        f"{name}: largest gap {d:.4f} between the {N_DRAWS} draws' ECDF "
        f"and ff is beyond the KS 0.1% band {KS_BAND_20000:.4f}"
    )
    # reproducible from its seed, and shaped as asked
    np.testing.assert_array_equal(
        draws, model.random(N_DRAWS, random_state=SEED)
    )
    assert model.random((3, 4), random_state=SEED).shape == (3, 4)


@pytest.mark.parametrize("name", ["grid-seals", "conjugate-exponential-gamma"])
def test_random_is_the_quantile_transform_of_its_uniforms(name):
    """One uniform per life, and more of it gives a longer life within a
    posterior point: the draws' quantiles match qf's."""
    model = model_of(name)
    draws = model.random(N_DRAWS, random_state=SEED)
    for p in (0.01, 0.1, 0.5, 0.9, 0.99):
        se = math.sqrt(p * (1 - p) / N_DRAWS)
        share = float(np.mean(draws <= float(model.qf(p))))
        assert abs(share - p) <= 3.3 * se


def test_random_with_the_global_seed_is_reproducible():
    model = model_of("grid-seals")
    np.random.seed(5)
    a = model.random(10)
    np.random.seed(5)
    b = model.random(10)
    np.testing.assert_array_equal(a, b)
