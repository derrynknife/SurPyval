"""The Bayesian fit's vectorised likelihood equals the fitters' own.

``surpyval.bayes.log_likelihood_fn`` is written separately from the
fitters' ``_log_likelihood`` so that the fitters are untouched; these
tests hold the two to 1e-10 (relative) at 50 random parameter points,
for every continuous distribution, for each kind of data alone and all
kinds mixed, and for the offset, limited-failure and zero-inflated forms.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.bayes import log_likelihood_fn
from surpyval.univariate.parametric import ParametricFitter

#: Families whose support moves with their parameters (fit_bayes refuses
#: them: the grid does not hold a moving support), and a point mass with
#: no density, and the Hypoexponential (given its rates, not fitted),
#: whose likelihoods are not used.
NOT_USED = {"Uniform", "Beta4", "ExactEventTime", "Hypoexponential"}

CONTINUOUS = sorted(
    {
        getattr(sp, name).name
        for name in sp.__all__
        if name not in sp._RENAMED_TOP_LEVEL
        and isinstance(getattr(sp, name, None), ParametricFitter)
        and not getattr(sp, name).discrete
        and getattr(sp, name).name not in NOT_USED
        and type(getattr(sp, name)).__name__ != "CustomDistribution"
    }
)

# One row of each kind: exact, right, left, interval; counts; left and
# right truncation on some rows.
X = np.array([1.2, 2.5, 3.1, 4.4, 5.0, 6.2, 7.7, 2.2, 3.3, 9.0])
C = np.array([0, 0, 1, 0, -1, 1, 0, 2, 0, 1])
XR = np.where(C == 2, X + 1.5, X)
N = np.array([1, 2, 1, 1, 3, 1, 1, 2, 1, 1])
TL = np.where(np.arange(10) % 3 == 0, 0.5, -np.inf)
TR = np.where(np.arange(10) % 4 == 0, 30.0, np.inf)


def _scaled(dist):
    """The data, put inside the distribution's support."""
    if dist.name == "Beta":
        k = 12.0
        tl = np.where(np.isfinite(TL), TL / k, -np.inf)
        tr = np.where(np.isfinite(TR), 0.99, np.inf)
        return np.c_[X / k, XR / k], tl, tr
    return np.c_[X, XR], TL, TR


def _point(dist, rng, offset, lfp, zi, x_min):
    p = []
    if offset:
        p.append(rng.uniform(-2.0, 0.3 * x_min))
    for lo, hi in dist.bounds:
        if lo == 0 and hi is None:
            p.append(rng.uniform(0.5, 4.0))
        elif lo is None and hi is None:
            p.append(rng.uniform(1.0, 6.0))
        else:
            p.append(rng.uniform(0.5, 4.0))
    if lfp:
        p.append(rng.uniform(0.3, 0.99))
    if zi:
        p.append(rng.uniform(0.01, 0.3))
    return np.array(p)


def _reference(dist, data, p, offset, lfp, zi):
    j = 1 if offset else 0
    dp = p[j : j + dist.k]
    gamma = p[0] if offset else 0.0
    lfp_p = p[j + dist.k] if lfp else 1.0
    f0 = p[-1] if zi else 0.0
    with np.errstate(all="ignore"):
        return float(dist._log_likelihood(data, *dp, gamma, f0, lfp_p))


def _check(dist, data, offset=False, lfp=False, zi=False, seed=0):
    ll = log_likelihood_fn(dist, data, offset=offset, lfp=lfp, zi=zi)
    rng = np.random.default_rng(seed)
    x_min = float(np.min(data.x))
    points = np.array(
        [_point(dist, rng, offset, lfp, zi, x_min) for _ in range(50)]
    )
    got = ll(points)
    want = np.array(
        [_reference(dist, data, p, offset, lfp, zi) for p in points]
    )
    both = np.isfinite(want)
    assert np.all(np.isfinite(got) == both), (got, want)
    np.testing.assert_allclose(got[both], want[both], rtol=1e-10, atol=0)


KINDS = {
    "exact": lambda x, tl, tr: dict(x=x[[0, 1, 3, 6]][:, 0]),
    "right": lambda x, tl, tr: dict(x=x[[0, 2, 5, 9]][:, 0], c=[0, 1, 1, 1]),
    "left": lambda x, tl, tr: dict(x=x[[0, 4, 3]][:, 0], c=[0, -1, 0]),
    "interval": lambda x, tl, tr: dict(x=x[[7, 1, 3]], c=[2, 0, 0]),
    "counts": lambda x, tl, tr: dict(x=x[:, 0], n=N),
    "left truncated": lambda x, tl, tr: dict(
        x=x[[1, 3, 6]][:, 0], tl=np.full(3, x[0, 0] * 0.5)
    ),
    "right truncated": lambda x, tl, tr: dict(
        x=x[[0, 1, 3]][:, 0], tr=np.full(3, x[9, 0])
    ),
    "mixed": lambda x, tl, tr: dict(x=x, c=C, n=N, tl=tl, tr=tr),
}


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("name", CONTINUOUS)
def test_equals_the_fitters_likelihood(name, kind):
    dist = getattr(sp, name)
    x, tl, tr = _scaled(dist)
    args = KINDS[kind](x, tl, tr)
    data = sp.SurpyvalData(**args)
    _check(dist, data)


FORMS = {
    "offset": dict(offset=True),
    "lfp": dict(lfp=True),
    "zi": dict(zi=True),
    "offset lfp": dict(offset=True, lfp=True),
    "lfp zi": dict(lfp=True, zi=True),
}


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize(
    "name", [n for n in CONTINUOUS if getattr(sp, n).support[0] == 0]
)
def test_equals_the_fitters_likelihood_with_extra_parameters(name, form):
    dist = getattr(sp, name)
    x, tl, tr = _scaled(dist)
    data = sp.SurpyvalData(x=x, c=C, n=N, tl=tl, tr=tr)
    _check(dist, data, **FORMS[form])


def test_zero_inflated_zeros():
    data = sp.SurpyvalData(x=[0.0, 0.0, 1.5, 2.5, 4.0], c=[0, 0, 0, 1, 0])
    _check(sp.Weibull, data, zi=True)
    _check(sp.Weibull, data, zi=True, lfp=True)


def test_shape_and_names():
    ll = log_likelihood_fn(sp.Weibull, [1.0, 2.0, 3.0], offset=True)
    assert ll.parameter_names == ["gamma", "alpha", "beta"]
    out = ll(np.ones((4, 5, 3)) * [0.1, 2.0, 1.5])
    assert out.shape == (4, 5)
    with pytest.raises(ValueError, match="3 values"):
        ll(np.ones((2, 2)))


def test_discrete_is_refused():
    with pytest.raises(ValueError, match="discrete"):
        log_likelihood_fn(sp.Poisson, [1, 2, 3])
