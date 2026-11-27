"""``MixtureModel``'s window probabilities against 50-digit arithmetic.

An interval-censored row's probability under a component, and a
truncation window's, is ``F(r) - F(l)``. Deep in the component's upper
tail both CDFs are near 1 and their difference loses its digits (a
window of probability 8e-11 under a BirnbaumSaunders was 4.4e-9 too
large), so there the mixture takes it from the survival function in log
space, as the univariate fit does. These check every component's log
probability of windows deep in its lower tail, across its median and
deep in its upper tail, and the mixture's negative log-likelihood, with
and without truncation, against mpmath at 50 digits. The references are
worked out at test time from the closed forms, so nothing depends on the
platform.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.utils.warnings import ignoring_warnings

# mpmath is a test dependency (the tests extra); a bare install of the
# wheel skips these.
mp = pytest.importorskip("mpmath")


@pytest.fixture(autouse=True)
def _fifty_digits():
    with mp.workdps(50):
        yield


def _mp_sf(name, x, p):
    """The survival function of ``name`` at ``x``, in mpmath."""
    x = mp.mpf(x)
    a, b = (mp.mpf(v) for v in p)
    if name == "Weibull":
        return mp.exp(-((x / a) ** b))
    if name == "LogNormal":
        return mp.ncdf(-(mp.log(x) - a) / b)
    if name == "Gamma":  # shape a, rate b
        return mp.gammainc(a, b * x, mp.inf, regularized=True)
    if name == "Gumbel":
        return mp.exp(-mp.exp((x - a) / b))
    if name == "BirnbaumSaunders":
        return mp.ncdf(-(mp.sqrt(x / b) - mp.sqrt(b / x)) / a)
    raise KeyError(name)


def _mp_ff(name, x, p):
    """The CDF, as its own closed form (``1 - S`` would cancel in the
    lower tail even at 50 digits, where S is within 1e-40 of 1)."""
    x = mp.mpf(x)
    a, b = (mp.mpf(v) for v in p)
    if name == "Weibull":
        return -mp.expm1(-((x / a) ** b))
    if name == "LogNormal":
        return mp.ncdf((mp.log(x) - a) / b)
    if name == "Gamma":
        return mp.gammainc(a, 0, b * x, regularized=True)
    if name == "Gumbel":
        return -mp.expm1(-mp.exp((x - a) / b))
    if name == "BirnbaumSaunders":
        return mp.ncdf((mp.sqrt(x / b) - mp.sqrt(b / x)) / a)
    raise KeyError(name)


def _mp_df(name, x, p):
    return mp.diff(lambda t: _mp_ff(name, t, p), mp.mpf(x))


def _mp_window(name, lo, hi, p):
    """``P(lo < T <= hi)``, from the CDF below the median and the
    survival function above it (each exact to 50 digits; the difference
    keeps 40 of them for a window of 1e-10)."""
    if np.isfinite(lo) and _mp_sf(name, lo, p) < 0.5:
        upper = _mp_sf(name, hi, p) if np.isfinite(hi) else 0
        return _mp_sf(name, lo, p) - upper
    lower = _mp_ff(name, lo, p) if np.isfinite(lo) else 0
    return (_mp_ff(name, hi, p) if np.isfinite(hi) else 1) - lower


def _mp_row(name, row, p):
    """One component's likelihood of one row ``(x or (l, r), c)``."""
    x, c = row
    if c == 0:
        return _mp_df(name, x, p)
    if c == 1:
        return _mp_sf(name, x, p)
    if c == -1:
        return _mp_ff(name, x, p)
    return _mp_window(name, x[0], x[1], p)


#: Two components of each family, and windows: deep in the first's lower
#: tail (F about 1e-12), across its median, deep in its upper tail (S
#: about 1e-9 to 1e-11), and deep in the second's upper tail.
CASES = {
    "Weibull": (
        [[10.0, 2.0], [30.0, 4.0]],
        [[1e-5, 2e-5], [6.0, 11.0], [46.0, 50.0], [64.0, 67.0]],
    ),
    "LogNormal": (
        [[2.0, 0.4], [3.0, 0.3]],
        [[0.45, 0.5], [6.0, 9.0], [82.0, 108.0], [122.0, 150.0]],
    ),
    "Gamma": (
        [[3.0, 0.5], [8.0, 0.4]],
        [[0.001, 0.002], [4.0, 7.0], [54.0, 63.0], [95.0, 108.0]],
    ),
    "Gumbel": (
        [[8.0, 3.0], [20.0, 4.0]],
        [[-75.0, -68.0], [5.0, 9.0], [17.1, 17.7], [32.1, 32.9]],
    ),
    "BirnbaumSaunders": (
        [[0.3, 4.0], [0.5, 18.0]],
        [[0.64, 0.68], [3.4, 4.7], [20.2, 23.5], [197.0, 237.0]],
    ),
}
W = [0.7, 0.3]


def _model(name, components, w, **data):
    model = sp.MixtureModel.from_params(getattr(sp, name), components, w)
    model.data = sp.SurpyvalData(**data)
    model._truncated = bool(np.isfinite(model.data.t).any())
    return model


def _mp_neg_ll(name, components, w, rows, n, tl=None):
    """The mixture's negative log-likelihood in mpmath: each row's
    mixture probability, conditioned on ``(tl, inf)`` if given."""
    total = mp.mpf(0)
    for row, count in zip(rows, n):
        like = sum(
            mp.mpf(wi) * _mp_row(name, row, p) for wi, p in zip(w, components)
        )
        if tl is not None:
            like /= sum(
                mp.mpf(wi) * _mp_sf(name, tl, p)
                for wi, p in zip(w, components)
            )
        total -= count * mp.log(like)
    return float(total)


@pytest.mark.parametrize("name", list(CASES))
def test_each_components_window_is_exact_in_both_tails(name):
    components, windows = CASES[name]
    model = _model(
        name,
        components,
        W,
        xl=[w[0] for w in windows],
        xr=[w[1] for w in windows],
    )
    for p in components:
        got = model._component_log_likelihood(np.asarray(p))
        expected = [
            float(mp.log(_mp_window(name, lo, hi, p))) for lo, hi in windows
        ]
        # The log to 1e-12: each window's probability to 1e-12 relative,
        # however small it is. The relative term was added after the first
        # run: one window, the second Gumbel's under the first, has a log
        # of -3082 (far below the smallest float), where 1e-12 absolute is
        # within a few spacings of doubles; there the log is held to 1e-15
        # relative, a few of its last digits, instead.
        np.testing.assert_allclose(got, expected, rtol=1e-15, atol=1e-12)


@pytest.mark.parametrize("name", list(CASES))
def test_the_mixture_likelihood_is_exact(name):
    components, windows = CASES[name]
    median = float(getattr(sp, name).qf(0.5, *components[0]))
    rows = [(w, 2) for w in windows] + [
        (median, 0),
        (windows[2][0], 1),
        (median, -1),
    ]
    n = [1, 2, 1, 3, 2, 1, 4]
    model = _model(
        name,
        components,
        W,
        x=[r[0] for r in rows],
        c=[r[1] for r in rows],
        n=n,
    )
    expected = _mp_neg_ll(name, components, W, rows, n)
    got = float(model.neg_ll_of(model.w, model.params))
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("name", list(CASES))
def test_a_truncation_deep_in_the_upper_tail_is_exact(name):
    # Left truncated where the first component's survival is about 1e-9:
    # its window (tl, inf) had been 1 - F(tl)
    components, windows = CASES[name]
    tl = windows[2][0] - 0.5 * (windows[2][1] - windows[2][0])
    rows = [(windows[2], 2), (windows[3], 2), (windows[3][0], 0)]
    n = [2, 1, 1]
    model = _model(
        name,
        components,
        W,
        x=[r[0] for r in rows],
        c=[r[1] for r in rows],
        n=n,
        tl=[tl] * len(rows),
    )
    expected = _mp_neg_ll(name, components, W, rows, n, tl=tl)
    got = float(model.neg_ll_of(model.w, model.params))
    assert got == pytest.approx(expected, rel=1e-12)


#: A component weighted within 2e-35 of 1 whose window [22, 27] (and
#: Gumbel's [18, 29]) is deep in its upper tail: the case that found the
#: loss of digits (BirnbaumSaunders was 121.77324154, 4.4e-9 high).
DOMINANT = {
    "BirnbaumSaunders": [[0.3, 4.0], [0.5, 18.0], [1.0, 10.0]],
    "Gumbel": [[8.0, 3.0], [20.0, 4.0], [14.0, 6.0]],
}


@pytest.mark.parametrize("name", list(DOMINANT))
def test_a_dominant_components_upper_tail_window_is_exact(name):
    x = [2.0, 5.0, [6.0, 10.0], 9.0, 14.0, [22.0, 27.0], 18.0, 4.5]
    c = [0, 1, 2, -1, 0, 2, 1, -1]
    n = [1, 3, 2, 1, 4, 1, 2, 5]
    if name == "Gumbel":
        x[5] = [18.0, 29.0]
    logits = np.array([40.0, -40.0, -40.0])
    w = np.exp(logits - np.logaddexp.reduce(logits))
    model = _model(name, DOMINANT[name], w, x=x, c=c, n=n)
    rows = [(xi if ci != 2 else tuple(xi), ci) for xi, ci in zip(x, c)]
    expected = _mp_neg_ll(name, DOMINANT[name], w, rows, n)
    got = float(model.neg_ll_of(model.w, model.params))
    assert got == pytest.approx(expected, rel=1e-12)
    if name == "BirnbaumSaunders":
        # the 50-digit value the torch loss is checked against
        assert expected == pytest.approx(121.77324099699673, rel=1e-15)


@pytest.mark.parametrize("tl", [None, 0.4])
def test_a_window_with_no_probability_leaves_the_gradient_finite(tl):
    # (25 / 1) ** 400 overflows: the first component gives the window
    # [25, 35] probability 0, a log of -inf, floored; the gradient must
    # still be finite (with and without a truncation)
    from autograd import grad

    data = dict(x=[[25.0, 35.0], 0.9, [0.5, 0.95]], c=[2, 0, 2])
    if tl is not None:
        data["tl"] = [tl] * 3
    model = _model("Weibull", [[1.0, 400.0], [20.0, 2.0]], [0.5, 0.5], **data)
    theta = model._pack(model.w, model.params)

    def fun(th):
        return model.neg_ll_of(*model._unpack(th))

    assert np.isfinite(fun(theta))
    assert np.all(np.isfinite(grad(fun)(theta)))


def _left_truncated_gamma_intervals():
    """200 draws from an even mixture of Gamma(3, 0.5) and Gamma(30,
    0.5), as inspection intervals of width 2, left truncated at the
    intervals' 20% point (the rows below it dropped)."""
    true = sp.MixtureModel.from_params(
        sp.Gamma, [[3, 0.5], [30, 0.5]], [0.5, 0.5]
    )
    x = np.sort(true.random(200, random_state=7))
    lo = np.floor(x / 2) * 2
    tl = np.quantile(lo, 0.2)
    keep = lo > tl
    return lo[keep], lo[keep] + 2, np.full(keep.sum(), tl)


def test_a_left_truncated_gamma_mixture_leaves_its_degenerate_point():
    # Before the windows were exact this fit stopped, "verified", at a
    # degenerate first component (shape 7.5e-7, weight 1 - 1.3e-6); the
    # rule, fixed before the run: the fit's log-likelihood is above that
    # point's by more than 0.1, and it is not reported "verified" at a
    # component with a shape below 1e-4
    xl, xr, tl = _left_truncated_gamma_intervals()
    with ignoring_warnings():
        fit = sp.MixtureModel.fit(xl=xl, xr=xr, tl=tl, dist=sp.Gamma, m=2)
    before = -float(
        fit.neg_ll_of(
            np.array([0.9999987263365633, 1.2736634370616663e-06]),
            np.array(
                [
                    [7.486189541834617e-07, 0.09361714794481045],
                    [30.109185115800617, 0.4982037144700529],
                ]
            ),
        )
    )
    assert fit.log_likelihood > before + 0.1
    degenerate = np.min(fit.params[:, 0]) < 1e-4
    assert not (degenerate and fit.maximum == "verified")


def test_rows_no_component_explains_are_counted_in_a_warning():
    # (25 / 1) ** 400 overflows: neither component gives the window
    # [25, 35] any probability, even in log space
    model = _model(
        "Weibull",
        [[1.0, 400.0], [1.1, 400.0]],
        [0.5, 0.5],
        x=[[25.0, 35.0], 0.9, 1.0],
        c=[2, 0, 0],
        n=[3, 1, 1],
    )
    with pytest.warns(UserWarning, match=r"1 row\(s\) \(3 observation"):
        model._warn_if_unexplained_rows()
