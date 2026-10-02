"""
Tests for the ``MixtureModel`` fitter.
"""

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
