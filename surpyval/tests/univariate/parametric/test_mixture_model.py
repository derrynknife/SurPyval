"""
Tests for the ``MixtureModel`` fitter.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
import surpyval as surv
from surpyval.tests._helpers import no_warnings


def _fitted_model(m=2, seed=0):
    np.random.seed(seed)
    x = np.concatenate(
        [sp.Weibull.random(100, 10, 3), sp.Weibull.random(100, 50, 4)]
    )
    mm = sp.MixtureModel(dist=sp.Weibull, m=m)
    mm.fit(x=x)
    return mm


def test_unfitted_attributes_are_none():
    mm = sp.MixtureModel(dist=sp.Weibull, m=2)
    assert mm.params is None
    assert mm.w is None
    assert mm.data is None
    # #482: the pre-fit repr said "Unable to fit values"
    assert repr(mm) == (
        "Unfitted Parametric Mixture SurPyval Model (Weibull, m = 2)"
    )


def test_fit_populates_params_and_weights():
    mm = _fitted_model()
    assert mm.params is not None
    assert mm.params.shape == (2, sp.Weibull.k)
    # Weights are a valid probability vector.
    assert np.isclose(mm.w.sum(), 1.0)
    assert np.all(mm.w >= 0)


def test_distribution_functions_are_consistent():
    mm = _fitted_model()
    grid = np.array([1.0, 5.0, 10.0, 25.0, 50.0])
    # sf and ff are complements.
    assert np.allclose(mm.sf(grid), 1 - mm.ff(grid))
    # ff is a valid CDF: non-decreasing and within [0, 1].
    ff = mm.ff(grid)
    assert np.all(np.diff(ff) >= 0)
    assert np.all(ff >= 0) and np.all(ff <= 1)
    # df is non-negative.
    assert np.all(mm.df(grid) >= 0)


def test_random_returns_requested_size_for_m_gt_2():
    # Regression test for the slice-accumulation bug in random(): for
    # m > 2 the per-component samples must be laid down contiguously
    # without overwriting earlier components.
    mm = _fitted_model(m=3)
    rvs = mm.random(500)
    assert rvs.shape == (500,)
    # All draws should be populated (Weibull draws are strictly positive),
    # i.e. no slot was left as the initial zero from an overwritten slice.
    assert np.all(rvs > 0)


def test_r_cb_is_removed():
    # R_cb was dead code that raised AttributeError on every fitted model;
    # it has been removed rather than silently shipped.
    mm = _fitted_model()
    assert not hasattr(mm, "R_cb")


def test_too_few_data_points_raises():
    mm = sp.MixtureModel(dist=sp.Weibull, m=2)
    with pytest.raises(ValueError):
        mm.fit(x=[1.0, 2.0])


def test_tied_data_recovers_mixture_weights():
    # SurpyvalData groups duplicate values into counts n > 1; the counts
    # must multiply the *mixture* log-likelihood, not power the
    # per-component likelihood before mixing (#254).
    np.random.seed(9)
    x = np.concatenate(
        [sp.Weibull.random(300, 5, 4), sp.Weibull.random(300, 30, 3)]
    )
    xr = np.round(x, 0)
    xr = xr[xr > 0]
    mm = sp.MixtureModel(dist=sp.Weibull, m=2)
    mm.fit(xr)
    assert np.sort(mm.w)[0] == pytest.approx(0.5, abs=0.1)


def test_truncation_correction_recovers_components():
    # Left-truncated sampling must be conditioned on the window; the naive
    # fit is biased and truncation used to be silently ignored (#254).
    np.random.seed(3)
    x = np.concatenate(
        [sp.Weibull.random(800, 5, 3), sp.Weibull.random(800, 20, 4)]
    )
    xt = x[x > 4.0]

    naive = sp.MixtureModel(dist=sp.Weibull, m=2)
    naive.fit(xt)
    corrected = sp.MixtureModel(dist=sp.Weibull, m=2)
    corrected.fit(xt, tl=4.0)

    # The corrected fit must differ from the naive fit (truncation was
    # previously a silent no-op) and recover the true first component.
    assert not np.allclose(np.sort(naive.w), np.sort(corrected.w), atol=1e-6)
    alphas = np.sort(corrected.params[:, 0])
    assert alphas[0] == pytest.approx(5.0, rel=0.15)
    assert alphas[1] == pytest.approx(20.0, rel=0.15)
    assert np.sort(corrected.w)[0] == pytest.approx(0.5, abs=0.1)


def test_interval_only_input_fits():
    # ``xl``/``xr``-only input previously crashed on ``len(None)`` (#254).
    np.random.seed(4)
    x = np.concatenate(
        [sp.Weibull.random(150, 5, 3), sp.Weibull.random(150, 20, 4)]
    )
    mm = sp.MixtureModel(dist=sp.Weibull, m=2)
    mm.fit(xl=np.floor(x), xr=np.floor(x) + 1)
    assert np.isclose(mm.w.sum(), 1.0)
    grid = np.array([1.0, 5.0, 10.0, 25.0])
    assert np.all(np.diff(mm.ff(grid)) >= 0)


def test_df_accepts_integer_input():
    mm = _fitted_model()
    vals = mm.df([1, 5, 10])
    assert np.all(np.isfinite(vals)) and np.all(vals >= 0)


# -- #482: fit returns the model, and works on the class ------------------

_X482 = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]


def test_fit_on_a_model_returns_that_model():
    # ``model = mm.fit(x)`` gave None, so ``model.sf`` raised AttributeError
    mm = sp.MixtureModel(dist=sp.Weibull, m=2)
    model = mm.fit(_X482)
    assert model is mm
    assert np.isfinite(model.sf(10))


def test_fit_on_the_class_builds_and_fits():
    old = sp.MixtureModel(dist=sp.Weibull, m=2)
    old.fit(_X482)  # the in-place form still works
    new = sp.MixtureModel.fit(_X482, dist=sp.Weibull, m=2)
    assert isinstance(new, sp.MixtureModel)
    assert new.m == 2 and new.dist is sp.Weibull
    np.testing.assert_allclose(new.params, old.params)
    np.testing.assert_allclose(new.w, old.w)
    three = sp.MixtureModel.fit(_X482, dist=sp.Weibull, m=3)
    assert three.params.shape == (3, 2)


def test_fit_on_the_class_passes_every_data_argument():
    c = [0] * 15 + [1, 1]
    old = sp.MixtureModel(dist=sp.Weibull, m=2)
    old.fit(_X482, c, tl=0.5)
    new = sp.MixtureModel.fit(_X482, c, tl=0.5, dist=sp.Weibull)
    np.testing.assert_allclose(new.params, old.params)
    # truncated data is fitted by direct maximisation, and says so
    assert "Fitted by           : MLE" in repr(new)
    assert "Fitted by           : EM" in repr(
        sp.MixtureModel.fit(_X482, dist=sp.Weibull)
    )


def test_fit_on_the_class_needs_dist():
    with pytest.raises(ValueError, match="dist"):
        sp.MixtureModel.fit(_X482)


def test_unbound_call_with_a_model_still_works():
    mm = sp.MixtureModel(dist=sp.Weibull, m=2)
    assert sp.MixtureModel.fit(mm, _X482) is mm
    assert mm.params is not None


def test_fit_signatures():
    import inspect

    on_class = inspect.signature(sp.MixtureModel.fit).parameters
    assert list(on_class)[:2] == ["x", "c"]
    assert on_class["dist"].kind is inspect.Parameter.KEYWORD_ONLY
    assert on_class["m"].default == 2
    on_model = inspect.signature(sp.MixtureModel(sp.Weibull).fit).parameters
    assert "dist" not in on_model and "self" not in on_model


# ---------------------------------------------------------------------------
# EM on interval data, a Geometric mixture, a restored mixture.
# ---------------------------------------------------------------------------


W, E, G = surv.Weibull, surv.Exponential, surv.Geometric


def test_mixture_em_on_interval_data_reaches_the_optimum():
    np.random.seed(0)
    x = np.concatenate([W.random(300, 5, 3), W.random(300, 30, 4)])
    mm = surv.MixtureModel(W, 2)
    no_warnings(mm.fit, xl=np.floor(x), xr=np.floor(x) + 1)
    truth = mm.neg_ll_of(np.array([0.5, 0.5]), np.array([[5, 3], [30, 4.0]]))
    # It stalled 114 units above the truth's negative log-likelihood
    assert mm.loglike <= truth + 1e-6


def test_geometric_mixture_fits_without_warnings():
    np.random.seed(0)
    x = np.concatenate([G.random(300, 0.5), G.random(300, 0.05)])
    mm = surv.MixtureModel(G, 2)
    no_warnings(mm.fit, x)
    assert sorted(mm.params.ravel()) == pytest.approx([0.05, 0.5], abs=0.03)


def test_restored_mixture_needs_its_data_for_plots_and_takes_lists_in_cs():
    x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
    mm = surv.MixtureModel(W, 2)
    mm.fit(x)
    restored = surv.from_dict(mm.to_dict())
    assert np.allclose(restored.cs([1, 2], 5), mm.cs(np.array([1, 2]), 5))
    for method in (restored.plot, restored.get_plot_data):
        with pytest.raises(ValueError, match="needs the data"):
            method()


# ---------------------------------------------------------------------------
# #544: a censored row with a finite truncation bound on its censored side
# is the interval between them (#310); the EM's pieces lost it.
# ---------------------------------------------------------------------------


def _data_544():
    rng = np.random.default_rng(544)
    x = np.sort(
        np.concatenate([rng.weibull(3, 40) * 5, rng.weibull(4, 40) * 20])
    )
    t = np.column_stack([np.zeros(80), np.full(80, np.inf)])
    return x, t


@pytest.mark.parametrize("kind", ["right", "left"])
def test_544_censored_row_with_truncation_is_its_interval(kind):
    x, t = _data_544()
    c = np.zeros(80, int)
    xl, xr, ci = x.copy(), x.copy(), c.copy()
    rows = [3, 50]
    if kind == "right":
        # right censored at x, truncated at tr: the interval [x, tr]
        c[rows] = 1
        t[rows, 1] = x[rows] + 5
        xr[rows] = t[rows, 1]
    else:
        # left censored at x, truncated at tl: the interval [tl, x]
        c[rows] = -1
        t[rows, 0] = x[rows] / 2
        xl[rows] = t[rows, 0]
    ci[rows] = 2

    # It raised IndexError: the converted rows fell out of the row order
    coded = sp.MixtureModel.fit(x, c=c, t=t, dist=sp.Weibull)
    explicit = sp.MixtureModel.fit(xl=xl, xr=xr, c=ci, t=t, dist=sp.Weibull)

    # The same likelihood at any parameters: row by row (the two forms
    # sort the rows differently), and in total
    for params in (explicit.params, [[3.0, 2.0], [10.0, 1.0]]):
        params = np.asarray(params)
        np.testing.assert_allclose(
            np.sort(coded.likelihood(params[0])),
            np.sort(explicit.likelihood(params[0])),
            rtol=1e-12,
        )
        # each row's log-likelihood is its likelihood's log
        np.testing.assert_allclose(
            coded.log_likelihood(params[1]),
            np.log(coded.likelihood(params[1])),
            rtol=1e-12,
        )
        assert coded.neg_ll_of(explicit.w, params) == pytest.approx(
            explicit.neg_ll_of(explicit.w, params), rel=1e-12
        )
    # and so the same fit: the maximum to 1e-8; the parameters to the
    # tolerance of the truncated path's optimiser (L-BFGS-B), as the
    # maximum is flat: the two fits' log-likelihoods differ by 3e-8 (of
    # 243) where their parameters differ by 2e-5
    assert coded.loglike == pytest.approx(explicit.loglike, rel=1e-8)
    np.testing.assert_allclose(coded.params, explicit.params, rtol=1e-4)
    np.testing.assert_allclose(coded.w, explicit.w, rtol=1e-4)


@pytest.mark.parametrize("kind", ["right", "left"])
def test_560_truncated_fit_is_a_verified_maximum(kind):
    # The truncated path took L-BFGS-B's answer unverified: the two forms
    # of the same data reached parameters 2e-5 apart on a flat maximum.
    # Polished and verified as the EM path is (#506), they agree to 1e-6,
    # and each is a verified maximum, in silence.
    x, t = _data_544()
    c = np.zeros(80, int)
    xl, xr, ci = x.copy(), x.copy(), c.copy()
    rows = [3, 50]
    if kind == "right":
        c[rows] = 1
        t[rows, 1] = x[rows] + 5
        xr[rows] = t[rows, 1]
    else:
        c[rows] = -1
        t[rows, 0] = x[rows] / 2
        xl[rows] = t[rows, 0]
    ci[rows] = 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coded = sp.MixtureModel.fit(x, c=c, t=t, dist=sp.Weibull)
        explicit = sp.MixtureModel.fit(
            xl=xl, xr=xr, c=ci, t=t, dist=sp.Weibull
        )
    assert coded.maximum == explicit.maximum == "verified"
    np.testing.assert_allclose(coded.params, explicit.params, rtol=1e-6)
    np.testing.assert_allclose(coded.w, explicit.w, rtol=1e-6)
