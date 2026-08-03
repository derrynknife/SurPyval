"""A wide net over the fitting API, asserting only what must always hold.

Every defect found in the v0.18.x round slipped past the whole suite, and
each one lived at an *intersection* of dimensions the suite tests one at
a time:

- a censored observation that is also truncated (the likelihood was
  unbounded, and returned garbage with ``success=True``)
- an offset combined with a particular method (Gamma's moment
  initialiser returned ``(inf, inf)``, which #261's fallback then
  reported as the fit)
- an offset combined with a large *shift magnitude* (54 of 120
  ExpoWeibull fits returned ``nan``)
- a sample smaller than the 1000 floor of ``FIT_SIZES`` (a rank
  deficient probability plot seeded the optimiser with ``nan``)

The full cross of distributions, methods, censoring, truncation,
structural flags, sizes and scales is around 600,000 cells, so this does
not attempt it. Instead it asserts cheap invariants -- finiteness,
boundedness, monotonicity -- over a broad seeded sample of that space.
Four of the five defects above would have failed the first two
assertions here, without anyone having to guess where to look.

Accuracy is deliberately not asserted. Recovery of known parameters is
already covered by ``test_fit.py``; the point of this file is to catch
answers that are not answers at all.

Opt in with ``--run-invariants``.
"""

import numpy as np
import pytest

from surpyval import (
    Exponential,
    ExpoWeibull,
    Gamma,
    Gumbel,
    Logistic,
    LogLogistic,
    LogNormal,
    Normal,
    Rayleigh,
    Weibull,
)

DISTS = [
    Weibull,
    Gamma,
    LogNormal,
    Normal,
    Logistic,
    LogLogistic,
    Gumbel,
    ExpoWeibull,
    Exponential,
    Rayleigh,
]
TRUE = {
    "Weibull": (10.0, 2.0),
    "Gamma": (3.0, 2.0),
    "LogNormal": (2.0, 0.5),
    "Normal": (10.0, 2.0),
    "Logistic": (10.0, 2.0),
    "LogLogistic": (10.0, 3.0),
    "Gumbel": (10.0, 2.0),
    "ExpoWeibull": (10.0, 2.0, 1.5),
    "Exponential": (0.1,),
    "Rayleigh": (5.0,),
}
POSITIVE_ONLY = {
    "Weibull",
    "Gamma",
    "LogNormal",
    "LogLogistic",
    "ExpoWeibull",
    "Exponential",
    "Rayleigh",
}

METHODS = ["MLE", "MPS", "MSE", "MOM", "MPP"]
SHAPES = ["plain", "right", "left", "interval", "tl", "tr", "right+tl"]
# Scale is the least covered axis in the suite, and the MLE failure
# warning itself advises rescaling towards 1 -- so exercise both ends.
SCALES = [1e-3, 1.0, 1e3]
# One representative size per cell. Sweeping sizes inside each cell
# multiplied the run by three for almost no extra coverage -- the size
# axis is exercised on its own in ``test_tiny_samples_never_return_junk``.
N = 120
TINY = [1, 2, 5, 40]


def _make(dist, n, shape, scale, seed):
    """Return (kwargs for fit) for one cell of the grid, or None to skip."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    x = np.asarray(dist.random(n, *TRUE[dist.name]), dtype=float) * scale
    if dist.name not in POSITIVE_ONLY:
        pass
    c = np.zeros(n, dtype=int)
    kw = {}
    if shape == "right":
        cut = np.quantile(x, 0.75)
        c = np.where(x > cut, 1, 0)
        x = np.minimum(x, cut)
    elif shape == "left":
        cut = np.quantile(x, 0.25)
        c = np.where(x < cut, -1, 0)
        x = np.maximum(x, cut)
    elif shape == "interval":
        counts, edges = np.histogram(x, bins=max(3, n // 4))
        keep = counts > 0
        xi = np.vstack([edges[:-1], edges[1:]]).T[keep]
        return dict(x=xi, n=counts[keep])
    elif shape == "tl":
        lo = float(np.quantile(x, 0.10))
        x = x[x > lo]
        if x.size < 3:
            return None
        c = np.zeros(x.size, dtype=int)
        kw["tl"] = lo
    elif shape == "tr":
        hi = float(np.quantile(x, 0.90))
        x = x[x < hi]
        if x.size < 3:
            return None
        c = np.zeros(x.size, dtype=int)
        kw["tr"] = hi
    elif shape == "right+tl":
        # The combination that #310/#311 was about: an observation that
        # is both censored and truncated.
        lo = float(np.quantile(x, 0.10))
        x = x[x > lo]
        if x.size < 4:
            return None
        cut = float(np.quantile(x, 0.75))
        c = np.where(x > cut, 1, 0)
        x = np.minimum(x, cut)
        kw["tl"] = lo
    _ = rng
    return dict(x=x, c=c, **kw)


def _fit_or_reason(dist, how, kwargs, **extra):
    """Fit, returning (model, None) or (None, exception).

    A raised exception is not a failure here. Several combinations are
    legitimately rejected -- a distribution that cannot be offset, data
    that cannot identify the free parameters, a method that does not
    support left censoring. What must never happen is a *returned* model
    that is not a model.
    """
    try:
        return dist.fit(how=how, **kwargs, **extra), None
    except Exception as exc:  # noqa: BLE001 - any rejection is acceptable
        return None, exc


def _assert_model_is_usable(model, label):
    params = np.atleast_1d(np.asarray(model.params, dtype=float))
    assert np.isfinite(params).all(), f"{label}: non-finite params {params}"

    for name in ("gamma", "p", "f0"):
        value = getattr(model, name, None)
        if value is not None:
            assert np.isfinite(float(value)), f"{label}: {name}={value}"

    nll = float(model.neg_ll())
    assert np.isfinite(nll), f"{label}: neg_ll={nll}"


def _assert_survival_is_a_survival_function(model, label):
    lo, hi = model.dist.support
    probe = np.asarray(model.surv_data.x, dtype=float)
    probe = probe[np.isfinite(probe)]
    if probe.size == 0:
        return
    grid = np.quantile(np.ravel(probe), np.linspace(0.05, 0.95, 12))
    grid = np.clip(grid, max(lo, -1e300), min(hi, 1e300))
    with np.errstate(all="ignore"):
        sf = np.asarray(model.sf(grid), dtype=float)
    assert np.isfinite(sf).all(), f"{label}: non-finite sf {sf}"
    inside = ((sf >= -1e-9) & (sf <= 1 + 1e-9)).all()
    assert inside, f"{label}: sf outside [0, 1]"
    assert np.all(np.diff(sf) <= 1e-9), f"{label}: sf not non-increasing"


@pytest.mark.parametrize("dist", DISTS, ids=lambda d: d.name)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("scale", SCALES, ids=lambda s: f"scale{s:g}")
def test_a_returned_fit_is_always_a_usable_model(dist, shape, scale):
    """Finite parameters, finite neg_ll, and a valid survival function.

    This is the assertion that four of the five v0.18.x defects would
    have tripped.
    """
    for how in METHODS:
        kwargs = _make(dist, N, shape, scale, seed=N + len(shape))
        if kwargs is None:
            continue
        label = f"{dist.name}|{how}|{shape}|scale={scale:g}|n={N}"
        model, exc = _fit_or_reason(dist, how, kwargs)
        if model is None:
            continue  # a refusal is a valid outcome; a bad model is not
        _assert_model_is_usable(model, label)
        _assert_survival_is_a_survival_function(model, label)


@pytest.mark.parametrize("dist", DISTS, ids=lambda d: d.name)
@pytest.mark.parametrize("n", TINY)
def test_tiny_samples_never_return_junk(dist, n):
    """Below the 1000 floor of ``FIT_SIZES``, where the crashes were.

    A refusal is fine -- one or two points cannot identify two free
    parameters, and the fitter now says so. What must not happen is a
    returned model carrying nan or inf, which is what Gamma and Beta
    used to do here.
    """
    for how in METHODS:
        kwargs = _make(dist, n, "plain", 1.0, seed=n)
        if kwargs is None:
            continue
        model, _ = _fit_or_reason(dist, how, kwargs)
        if model is None:
            continue
        _assert_model_is_usable(model, f"{dist.name}|{how}|n={n}")


@pytest.mark.parametrize("dist", DISTS, ids=lambda d: d.name)
def test_structural_flags_pairwise(dist):
    """Offset, lfp, zi and fixed, alone and in pairs.

    The suite exercises these almost exclusively on their own -- only a
    handful of places combine two. Pairwise is where interaction bugs
    live, and it is a small fraction of the full cross.
    """
    combos = [
        {"offset": True},
        {"lfp": True},
        {"zi": True},
        {"offset": True, "lfp": True},
        {"offset": True, "zi": True},
        {"lfp": True, "zi": True},
    ]
    if dist.k >= 2:
        first = dist.param_names[0]
        combos.append({"fixed": {first: float(TRUE[dist.name][0])}})
        combos.append(
            {"offset": True, "fixed": {first: float(TRUE[dist.name][0])}}
        )
    for extra in combos:
        for shape in ("plain", "right"):
            kwargs = _make(dist, 200, shape, 1.0, seed=7)
            if kwargs is None:
                continue
            label = f"{dist.name}|{shape}|{extra}"
            model, exc = _fit_or_reason(dist, "MLE", kwargs, **extra)
            if model is None:
                continue
            _assert_model_is_usable(model, label)


@pytest.mark.parametrize("dist", DISTS, ids=lambda d: d.name)
def test_mle_attains_the_lowest_negative_log_likelihood(dist):
    """The defining property of the estimator, checkable since #316.

    Maximum likelihood minimises the negative log-likelihood, so no other
    method may beat it on its own objective. A tolerance is allowed
    because the others stop on different criteria, but a real defeat
    means the MLE did not converge.
    """
    np.random.seed(3)
    x = np.asarray(dist.random(500, *TRUE[dist.name]), dtype=float)
    mle, exc = _fit_or_reason(dist, "MLE", dict(x=x))
    if mle is None:
        pytest.skip(f"{dist.name}: MLE unavailable ({exc})")
    best = float(mle.neg_ll())
    for how in ("MPS", "MSE", "MOM", "MPP"):
        other, _ = _fit_or_reason(dist, how, dict(x=x))
        if other is None:
            continue
        rival = float(other.neg_ll())
        if not np.isfinite(rival):
            continue
        assert rival >= best - 1e-6 * max(
            abs(best), 1.0
        ), f"{dist.name}: {how} reached {rival} against MLE's {best}"
