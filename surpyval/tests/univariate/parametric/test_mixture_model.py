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
    assert mm.neg_ll() <= truth + 1e-6


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
            np.sort(coded._likelihood(params[0])),
            np.sort(explicit._likelihood(params[0])),
            rtol=1e-12,
        )
        # each row's log-likelihood is its likelihood's log
        np.testing.assert_allclose(
            coded._component_log_likelihood(params[1]),
            np.log(coded._likelihood(params[1])),
            rtol=1e-12,
        )
        assert coded.neg_ll_of(explicit.w, params) == pytest.approx(
            explicit.neg_ll_of(explicit.w, params), rel=1e-12
        )
    # and so the same fit: the maximum to 1e-8; the parameters to the
    # tolerance of the truncated path's optimiser (L-BFGS-B), as the
    # maximum is flat: the two fits' log-likelihoods differ by 3e-8 (of
    # 243) where their parameters differ by 2e-5
    assert coded.neg_ll() == pytest.approx(explicit.neg_ll(), rel=1e-8)
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
    # Both are verified maxima of a flat likelihood: they agree to 5e-7
    # here and 1.1e-6 on other CPUs (they differed by 1.8e-5 before #560).
    np.testing.assert_allclose(coded.params, explicit.params, rtol=1e-5)
    np.testing.assert_allclose(coded.w, explicit.w, rtol=1e-5)


def test_a_point_mass_component_warns_once():
    # A component collapsed onto a point mass has no finite maximum, which
    # is also why EM ran to its iteration limit: one warning, "No finite
    # maximum", where the fit used to give that one and "did not reach a
    # verified maximum" as well (principle 22).
    x = np.r_[np.full(10, 3.0), np.linspace(20.0, 40.0, 10)]
    n = np.ones(20, int)
    n[[2, 15]] = 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sp.MixtureModel.fit(x, n=n, dist=sp.Weibull)
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1, messages
    assert messages[0].startswith("No finite maximum"), messages
    assert model.maximum == "no finite maximum"


def test_572_log_likelihood_is_the_fitted_value_with_the_criteria():
    # The issue's example: ``loglike`` was +219.73, the *negative*
    # log-likelihood, and there was no aic/bic to compare with a Weibull
    x = 100 * np.random.default_rng(0).weibull(1.5, 40)
    mm = no_warnings(lambda: surv.MixtureModel.fit(x, dist=surv.Weibull, m=2))
    density = sum(w * surv.Weibull.df(x, *p) for w, p in zip(mm.w, mm.params))
    ll = np.log(density).sum()
    assert mm.log_likelihood == pytest.approx(ll, rel=1e-12)
    assert mm.log_likelihood < 0
    assert mm.neg_ll() == pytest.approx(-ll, rel=1e-12)
    # As Parametric defines them: k = 2 * 2 + 1 free parameters, and the
    # sample size of every SurPyval BIC (40 failures)
    k, d = 5, 40
    assert mm.aic() == pytest.approx(2 * k - 2 * ll, rel=1e-12)
    assert mm.bic() == pytest.approx(k * np.log(d) - 2 * ll, rel=1e-12)
    assert mm.aic_c() == pytest.approx(
        mm.aic() + (2 * k**2 + 2 * k) / (d - k - 1), rel=1e-12
    )
    # ... so the comparison with one Weibull is the right way round
    weibull = surv.Weibull.fit(x)
    assert mm.aic() > weibull.aic()


def test_572_old_spellings_are_gone():
    # Deprecated in v0.23, removed in v0.24
    mm = _fitted_model()
    assert not hasattr(mm, "loglike")
    assert type(mm.log_likelihood) is float
    with pytest.raises(TypeError):
        mm.log_likelihood(mm.params[0])


def test_572_criteria_survive_a_round_trip_and_a_refit():
    mm = _fitted_model()
    restored = surv.from_dict(mm.to_dict())
    for name in ("neg_ll", "aic", "aic_c", "bic"):
        assert getattr(restored, name)() == getattr(mm, name)()
    assert restored.log_likelihood == mm.log_likelihood
    # A refit in place recomputes them
    aic = mm.aic()
    np.random.seed(1)
    mm.fit(
        x=np.concatenate(
            [sp.Weibull.random(50, 10, 3), sp.Weibull.random(50, 50, 4)]
        )
    )
    assert mm.aic() != aic
    assert mm.aic() == pytest.approx(2 * 5 + 2 * mm.neg_ll(), rel=1e-12)


@pytest.mark.parametrize(
    "old",
    [
        "likelihood",
        "Q",
        "expectation",
        "maximisation",
        "EM",
        "initialise_params",
    ],
)
def test_605_em_steps_are_internal(old):
    # Their public names, deprecated in v0.23, are gone in v0.24
    x = surv.Weibull.random(100, 10, 2, random_state=0)
    model = sp.MixtureModel.fit(x, dist=surv.Weibull, m=2)
    assert not hasattr(model, old)


def test_626_responsibilities_are_internal():
    # ``p`` held the EM responsibilities, a meaning ``p`` has nowhere
    # else; its public name warns until v0.25 and the fit does not use it.
    x = surv.Weibull.random(100, 10, 2, random_state=0)
    model = no_warnings(sp.MixtureModel.fit, x, dist=surv.Weibull, m=2)
    with pytest.warns(DeprecationWarning, match="internal to the fit") as rec:
        resp = model.p
    assert rec[0].filename == __file__
    assert resp is model._resp and resp.shape == (2, len(model.data.x))
    np.testing.assert_allclose(resp.sum(axis=0), 1.0)


def test_650_a_component_past_the_data_is_no_finite_maximum():
    # A second Weibull component ran off past the data (scale 33,561, the
    # largest observation 1,150) and the fit called it verified; its limit
    # is a one-component limited-failure Weibull, which is 2e-8 higher.
    import scipy.stats as ss

    rng = np.random.default_rng(21)
    t = ss.weibull_min(1.6, scale=500, loc=100).rvs(40, random_state=rng)
    cen = rng.uniform(300, 1500, 40)
    x, c = np.minimum(t, cen), (t > cen).astype(int)
    with pytest.warns(UserWarning, match="explains no failure"):
        mm = surv.MixtureModel(surv.Weibull, 2).fit(x, c)
    assert mm.maximum == "no finite maximum"


def test_650_an_ordinary_mixture_is_still_verified():
    rng = np.random.default_rng(3)
    x = np.concatenate(
        [
            surv.Weibull.random(60, 10, 3, random_state=rng),
            surv.Weibull.random(60, 60, 5, random_state=rng),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mm = surv.MixtureModel(surv.Weibull, 2).fit(x)
    assert mm.maximum == "verified"


# -- hf, qf and the Wald inference (#651) ----------------------------------


def _two_weibulls():
    x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
    return sp.MixtureModel.fit(x, dist=sp.Weibull, m=2)


def test_hf_is_df_over_sf():
    mm = _two_weibulls()
    grid = np.array([0.5, 5.0, 15.0, 30.0])
    assert np.allclose(mm.hf(grid), mm.df(grid) / mm.sf(grid))
    assert np.ndim(mm.hf(5.0)) == 0
    # Finite where the survival has underflowed to 0
    assert np.isfinite(mm.hf(1e4)) and mm.hf(1e4) > 0


def test_sf_hf_and_cs_keep_their_precision_in_the_upper_tail():
    # #671: sf was 1 - ff, 0 (and Hf inf) once the survival fell below
    # about 1e-16; the components' summed survival keeps it.
    g = np.random.default_rng(7)
    mm = sp.MixtureModel(sp.Weibull, 2)
    mm.fit(np.r_[20 * g.weibull(1, 15), 150 * g.weibull(3, 45)])
    x = np.array([1e-6, 600.0, 1000.0, 1e4])
    exact = sum(w * sp.Weibull.sf(x, *p) for w, p in zip(mm.w, mm.params))
    assert np.allclose(mm.sf(x), exact, rtol=1e-12, atol=0)
    # (near 0, -log1p(-ff) is the exact reference: log(sf) cancels there)
    F = sum(w * sp.Weibull.ff(x, *p) for w, p in zip(mm.w, mm.params))
    with np.errstate(divide="ignore"):
        H = np.where(F < 0.5, -np.log1p(-F), -np.log(exact))
    assert np.allclose(mm.Hf(x), H, rtol=1e-12, atol=0)
    assert np.allclose(mm.sf(x) + mm.ff(x), 1.0)
    # Past the survival's underflow, Hf is the components' log-sum-exp
    log_s = [
        np.log(w) + sp.Weibull.log_sf(1e5, *p) for w, p in zip(mm.w, mm.params)
    ]
    assert mm.sf(1e5) == 0
    assert np.isclose(mm.Hf(1e5), -np.logaddexp(*log_s), rtol=1e-12)
    assert mm.Hf(0.0) == 0 and mm.Hf(np.inf) == np.inf
    # cs from the cumulative hazard: finite where sf(given) is 0
    assert np.isclose(mm.cs(10.0, 1000.0), mm.sf(1010.0) / exact[2])
    assert 0 < mm.cs(1.0, 1e5) < 1
    assert np.isclose(mm.cs(3.0, 5.0), mm.sf(8.0) / mm.sf(5.0))
    assert np.isnan(mm.cs(1.0, np.inf))


def test_qf_inverts_ff():
    mm = _two_weibulls()
    p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    assert np.allclose(mm.ff(mm.qf(p)), p, atol=1e-10)
    assert mm.qf([[0.1], [0.5]]).shape == (2, 1)
    assert mm.qf(0.0) == 0.0 and mm.qf(1.0) == np.inf


def test_qf_outside_unit_interval_is_nan_with_one_warning():
    mm = _two_weibulls()
    with pytest.warns(UserWarning, match="outside") as caught:
        q = mm.qf([10.0, -0.1, np.nan, 0.5])
    assert len(caught) == 1
    assert np.isnan(q[:3]).all() and np.isfinite(q[3])


def test_covariance_matches_direct_hessian():
    from surpyval.utils.linalg import numerical_hessian

    mm = _two_weibulls()
    assert mm.covariance_names == [
        "alpha_0",
        "beta_0",
        "alpha_1",
        "beta_1",
        "w_0",
        "w_1",
    ]

    def nll(v):
        w = np.array([v[4], 1 - v[4]])
        return float(mm.neg_ll_of(w, v[:4].reshape(2, 2)))

    v = np.r_[mm.params.ravel(), mm.w[0]]
    se = np.sqrt(np.diag(np.linalg.inv(numerical_hessian(nll, v))))
    assert np.allclose(mm.standard_errors()[:5], se, rtol=1e-3)
    assert np.isclose(mm.standard_errors()[4], mm.standard_errors()[5])
    assert mm.covariance().shape == (6, 6)


def test_param_cb_names_and_scales():
    mm = _two_weibulls()
    lo, hi = mm.param_cb("alpha_1")
    assert 0 < lo < mm.params[1, 0] < hi
    lo, hi = mm.param_cb("w_0")
    assert 0 < lo < mm.w[0] < hi < 1
    with pytest.raises(ValueError, match="'alpha_0'"):
        mm.param_cb("alpha")
    with pytest.raises(ValueError, match="Wald"):
        mm.param_cb("alpha_0", method="lr")


def test_cb_and_quantile_cb_bracket_the_estimate():
    mm = _two_weibulls()
    grid = np.array([2.0, 8.0, 16.0])
    for on in ("sf", "ff", "Hf", "hf", "df"):
        band = mm.cb(grid, on=on)
        value = getattr(mm, on)(grid)
        assert band.shape == (3, 2)
        assert np.all(band[:, 0] <= value) and np.all(value <= band[:, 1])
    assert np.allclose(mm.cb(grid, on="ff"), 1 - mm.cb(grid)[:, ::-1])
    lower = mm.cb(grid, bound="lower")
    assert np.allclose(lower, mm.cb(grid, alpha_ci=0.1)[:, 0])
    lo, hi = mm.quantile_cb(0.1)
    assert lo < mm.qf(0.1) < hi
    assert mm.quantile_cb([0.1, 0.5]).shape == (2, 2)


def test_626_quantile_cb_outside_0_1_is_nan_with_one_warning():
    mm = _two_weibulls()
    with pytest.warns(UserWarning, match=r"quantile_cb: 1 of the 2") as rec:
        got = mm.quantile_cb([0.1, 1.5])
    assert len(rec) == 1 and rec[0].filename == __file__
    np.testing.assert_allclose(got[0], mm.quantile_cb(0.1))
    assert np.isnan(got[1]).all()


def test_covariance_survives_to_dict_and_refit():
    mm = _two_weibulls()
    restored = sp.MixtureModel.from_dict(mm.to_dict())
    assert np.allclose(restored.standard_errors(), mm.standard_errors())
    assert np.allclose(restored.cb([5.0]), mm.cb([5.0]))
    before = mm.standard_errors()
    mm.fit(np.arange(1.0, 30.0))
    assert not np.allclose(mm.standard_errors(), before)
