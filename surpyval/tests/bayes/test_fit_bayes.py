"""``fit_bayes`` and the model it returns: priors, refusals, the
predictive functions, credible intervals, the workflow (update, earlier
fits and data as priors, comparisons), saving and plotting."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.bayes import BayesianParametric

# 40 pump seals: three failures, 37 running
X = np.r_[2100.0, 3400.0, 4050.0, np.linspace(1500, 6000, 37)]
C = np.r_[0, 0, 0, np.ones(37)]
SHAPE = {"beta": sp.prior.around(2, between=(1.5, 3))}
RATE = {"failure_rate": sp.Gamma.from_params([2.0, 500.0])}


@pytest.fixture(scope="module")
def seals():
    return sp.Weibull.fit_bayes(X, C, priors=SHAPE)


# ---------------------------------------------------------------------------
# Priors and refusals
# ---------------------------------------------------------------------------
def test_around_is_lognormal_on_the_interval():
    from surpyval.bayes.transforms import Transform

    b = sp.prior.around(2, between=(1.5, 3), mass=0.9)
    t = Transform.for_bounds((0, None))
    m, s = b._resolved(t)
    assert np.exp(m) == pytest.approx(np.sqrt(1.5 * 3))
    from scipy import stats

    lo, hi = np.exp(stats.norm.ppf([0.05, 0.95], m, s))
    assert (lo, hi) == pytest.approx((1.5, 3.0))
    assert "LogNormal(median 2.121" in b.describe(t)


def test_a_normal_prior_on_a_positive_parameter_is_refused():
    with pytest.raises(ValueError, match="mass outside the range of beta"):
        sp.Weibull.fit_bayes(
            X, C, priors={"beta": sp.Normal.from_params([1.0, 1.0])}
        )
    # (with a flat alpha too the posterior would be improper: a Normal
    # cut at 0 leaves mass near beta = 0, where alpha's tail is too long)
    model = sp.Weibull.fit_bayes(
        X,
        C,
        priors={
            "beta": sp.prior.from_distribution(
                sp.Normal.from_params([2.0, 0.5]), truncate=True
            ),
            "alpha": sp.prior.around(1e4, between=(3e3, 3e4)),
        },
    )
    assert model.params[1] > 0


def test_unknown_parameter_is_refused():
    with pytest.raises(ValueError, match="No parameter 'shape'"):
        sp.Weibull.fit_bayes(X, C, priors={"shape": sp.prior.flat()})


def test_a_number_is_not_a_prior():
    with pytest.raises(ValueError, match="prior.fixed"):
        sp.Weibull.fit_bayes(X, C, priors={"beta": 2.0})


def test_flat_with_no_failures_is_refused():
    with pytest.raises(ValueError, match="no failures.*alpha"):
        sp.Weibull.fit_bayes([100.0, 200, 300], c=[1, 1, 1], priors=SHAPE)


def test_all_flat_needs_a_verified_maximum():
    # one failure: the likelihood rises without bound in beta
    with pytest.raises(ValueError, match="do not determine them"):
        sp.Weibull.fit_bayes([100.0, 200, 300], c=[0, 1, 1])


def test_all_flat_with_enough_data_is_the_reference_posterior():
    x = sp.Weibull.random(30, 100, 2, random_state=1)
    model = sp.Weibull.fit_bayes(x)
    mle = sp.Weibull.fit(x)
    # the posterior's mode is the maximum (flat on the log scale)
    assert model.diagnostics["ok"]
    np.testing.assert_allclose(model.params, mle.params, rtol=0.05)


def test_more_than_three_free_parameters_is_refused():
    with pytest.raises(ValueError, match="1 to 3"):
        sp.Weibull.fit_bayes(X, C, offset=True, lfp=True, priors=SHAPE)


def test_every_parameter_fixed_is_refused():
    with pytest.raises(ValueError, match="nothing to fit"):
        sp.Weibull.fit_bayes(
            X,
            C,
            priors={"alpha": sp.prior.fixed(1e4), "beta": sp.prior.fixed(2.0)},
        )


@pytest.mark.parametrize("name", ["Uniform", "Beta4", "Poisson"])
def test_families_not_taken(name):
    with pytest.raises(ValueError, match=name):
        getattr(sp, name).fit_bayes([1.0, 2.0, 3.0])


def test_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        sp.Weibull.fit_bayes(X, C, priors=SHAPE, method="pymc")


@pytest.mark.parametrize("bad", [1, 0, -3, "many", 2.5, True])
def test_bad_points_is_refused_naming_points(bad):
    """audit B17: an unusable ``points`` raises a ValueError that names
    the argument (principle 2)."""
    x = sp.Weibull.random(30, 100, 2, random_state=5)
    with pytest.raises(ValueError, match="points"):
        sp.Weibull.fit_bayes(x, priors=SHAPE, points=bad)


def test_two_priors_for_one_parameter_are_refused():
    """audit B18: a list of priors naming one parameter twice."""
    with pytest.raises(ValueError, match="Two priors"):
        sp.Weibull.fit_bayes(
            X, C, priors=[SHAPE, {"beta": sp.prior.uniform(1, 4)}]
        )


def test_a_prior_keyed_by_a_non_string_is_refused():
    """audit B18: the keys of a priors dict are parameter names."""
    with pytest.raises(ValueError, match="parameter names"):
        sp.Weibull.fit_bayes(X, C, priors={1: sp.prior.flat()})


def test_a_bare_number_as_priors_is_refused():
    """audit B18: priors that are neither a dict, a joint prior nor a
    list of them."""
    with pytest.raises(ValueError, match="priors must be"):
        sp.Weibull.fit_bayes(X, C, priors=3.0)


def test_data_disagreeing_with_the_prior_warns():
    x = sp.Weibull.random(200, 100, 6, random_state=2)
    with pytest.warns(UserWarning, match="disagree with the prior on beta"):
        sp.Weibull.fit_bayes(x, priors=SHAPE)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def test_functions_agree(seals):
    t = np.array([500.0, 2000.0, 5000.0, 20000.0])
    np.testing.assert_allclose(seals.sf(t) + seals.ff(t), 1.0)
    np.testing.assert_allclose(seals.Hf(t), -np.log(seals.sf(t)))
    np.testing.assert_allclose(seals.hf(t), seals.df(t) / seals.sf(t))
    p = np.array([0.05, 0.1, 0.5, 0.9])
    np.testing.assert_allclose(seals.ff(seals.qf(p)), p, rtol=1e-9)


def test_shapes(seals):
    assert np.ndim(seals.sf(1000.0)) == 0
    assert seals.sf([[1000.0, 2000.0]]).shape == (1, 2)
    assert seals.cri([1000.0, 2000.0]).shape == (2, 2)
    assert seals.cri(1000.0).shape == (2,)
    assert seals.cri([1000.0, 2000.0], bound="lower").shape == (2,)
    assert seals.quantile_cri([0.1, 0.5]).shape == (2, 2)
    assert seals.param_cri().shape == (2, 2)
    assert seals.mean_cri().shape == (2,)
    assert np.isnan(seals.sf(np.nan))


def test_credible_intervals_contain_the_median_and_nest(seals):
    t = np.array([1000.0, 3000.0, 6000.0])
    med = seals.cri(t, point=True)
    for a in (0.01, 0.05, 0.2):
        lo, hi = seals.cri(t, alpha_ci=a).T
        assert np.all(lo <= med) and np.all(med <= hi)
    wide = seals.cri(t, alpha_ci=0.01)
    narrow = seals.cri(t, alpha_ci=0.2)
    assert np.all(wide[:, 0] <= narrow[:, 0])
    assert np.all(wide[:, 1] >= narrow[:, 1])
    # one-sided is an end of the two-sided at twice the level
    np.testing.assert_allclose(
        seals.cri(t, alpha_ci=0.05, bound="lower"),
        seals.cri(t, alpha_ci=0.1)[:, 0],
    )
    # ... on every function, the hazard too
    for on in ("hf", "df"):
        lo, hi = seals.cri(t, on=on).T
        mid = seals.cri(t, on=on, point=True)
        assert np.all(lo <= mid) and np.all(mid <= hi)
    # as alpha_ci -> 1 the interval closes onto the median
    np.testing.assert_allclose(
        seals.cri(t, alpha_ci=1 - 1e-9), np.c_[med, med], rtol=1e-6
    )
    # bounds on ff and Hf are those on sf, transformed
    np.testing.assert_allclose(
        seals.cri(t, on="ff"), 1 - seals.cri(t)[:, ::-1]
    )
    np.testing.assert_allclose(
        seals.cri(t, on="Hf"), -np.log(seals.cri(t)[:, ::-1])
    )


@pytest.mark.parametrize("on", ["hf", "df"])
def test_hazard_and_density_intervals_are_consistent(seals, on):
    """audit B11: ``cri`` on the hazard and the density (outside the
    conformance interval sweep): a one-sided bound at level a is that end
    of the two-sided interval at 2a (all of a in one tail), a narrower
    level nests inside a wider one, the median lies between the ends,
    and every value is finite and non-negative."""
    x = [1000.0, 3000.0, 5000.0]
    both = seals.cri(x, on=on)
    narrow = seals.cri(x, on=on, alpha_ci=0.10)
    np.testing.assert_array_equal(
        seals.cri(x, on=on, alpha_ci=0.05, bound="lower"), narrow[..., 0]
    )
    np.testing.assert_array_equal(
        seals.cri(x, on=on, alpha_ci=0.05, bound="upper"), narrow[..., 1]
    )
    assert np.all(narrow[..., 0] >= both[..., 0])
    assert np.all(narrow[..., 1] <= both[..., 1])
    mid = seals.cri(x, on=on, point=True)
    assert np.all(both[..., 0] <= mid) and np.all(mid <= both[..., 1])
    for v in (both, narrow, mid):
        assert np.all(np.isfinite(v)) and np.all(v >= 0)


def test_predictive_b10_is_not_the_median_b10(seals):
    # the predictive (new unit) quantile is below the population B10's
    # posterior median when the uncertainty is wide
    assert seals.qf(0.1) < seals.quantile_cri(0.1, point=True)


def test_cb_refuses_and_points_to_cri(seals):
    for name in ("cb", "quantile_cb", "mean_cb", "param_cb"):
        with pytest.raises(ValueError, match="credible interval"):
            getattr(seals, name)(1000.0)


def test_summary_in_plain_words(seals):
    text = repr(seals)
    assert text.startswith(
        "Weibull: Bayesian fit, 40 items (3 failed, 37 running)"
    )
    assert "alpha: no prior given; flat on the log scale" in text
    assert "beta about 2 (90% between 1.5 and 3)" in text
    assert "B10 life" in text and "Mean life" in text
    assert "credible interval" in text
    assert "Compared with the maximum-likelihood fit" in text
    assert "integration error" in text


def test_compare(seals):
    table = seals.compare().table
    assert table["alpha"]["prior_interval"] is None
    assert table["beta"]["mle"] == pytest.approx(
        sp.Weibull.fit(X, C).params[1]
    )
    post = table["beta"]["posterior_interval"]
    mle = table["beta"]["mle_interval"]
    assert post[1] - post[0] < mle[1] - mle[0]
    assert "credible interval" in repr(seals.compare())


def test_draws_and_models(seals):
    d = seals.posterior.draws(500, random_state=1)
    assert d.shape == (500, 2)
    np.testing.assert_array_equal(d, seals.posterior.draws(500, 1))
    df = seals.posterior.to_dataframe(10, random_state=1)
    assert list(df.columns) == ["alpha", "beta"]
    models = seals.draw_models(3, random_state=1)
    assert all(m.dist is sp.Weibull for m in models)
    plug = seals.at_params()
    np.testing.assert_allclose(plug.params, seals.params)
    r = seals.random(1000, random_state=3)
    assert r.shape == (1000,)
    # the predictive's draws follow its sf
    assert np.mean(r > 5000) == pytest.approx(seals.sf(5000), abs=0.04)


def test_density_and_marginal(seals):
    a, b = seals.params
    dens = seals.posterior.density([[a, b], [a * 3, b]])
    assert dens[0] > dens[1] > 0
    v, m = seals.posterior.marginal("beta")
    assert np.trapezoid(m, v) == pytest.approx(1.0, abs=0.01)


def test_evidence_only_with_proper_priors(seals):
    assert np.isnan(seals.log_evidence)
    proper = sp.Weibull.fit_bayes(
        X,
        C,
        priors={**SHAPE, "alpha": sp.prior.around(1e4, between=(3e3, 3e4))},
    )
    assert np.isfinite(proper.log_evidence)


def test_laplace_is_labelled_approximate():
    model = sp.Weibull.fit_bayes(X, C, priors=SHAPE, method="laplace")
    assert model.diagnostics["approximate"]
    assert "Approximated by a normal" in repr(model)


def test_starved_grid_refuses():
    from surpyval.bayes.engines import grid

    with pytest.raises(ValueError, match="did not reach its accuracy"):
        bound = __import__("surpyval.bayes.fitting", fromlist=["x"])
        data = sp.SurpyvalData(x=X, c=C)
        b = bound.build_bound(
            sp.Weibull, data, SHAPE, False, False, False, None
        )
        ll = sp.bayes.log_likelihood_fn(sp.Weibull, data)

        def lp(phi):
            return b.log_prior(phi) + ll(b.full_theta(phi))

        mode, cov, _ = grid.find_mode(
            lp, [b.to_phi(np.atleast_2d(sp.Weibull.fit(X, C).params))[0]]
        )
        try:
            grid.run_grid(
                lp,
                mode,
                cov,
                bound.check_summary(sp.Weibull, b),
                points=5,
                max_refinements=0,
            )
        except grid.GridError as e:
            raise ValueError(str(e))


@pytest.mark.parametrize(
    "options",
    [
        dict(offset=True),
        dict(lfp=True),
        dict(zi=True),
    ],
)
def test_extra_parameters(options):
    x = np.r_[0.0, 0.0, sp.Weibull.random(30, 100, 2, random_state=4) + 20]
    c = np.zeros(len(x))
    if not options.get("zi"):
        x, c = x[2:], c[2:]
    if options.get("lfp"):
        c = np.where(x > 120, 1, 0)
        x = np.where(x > 120, 120, x)
    # the extras' priors: a Beta for a proportion, an interval for the
    # offset (flat, the data leave them all but undetermined)
    extra = {
        "lfp": {"lfp_p": sp.Beta.from_params([4.0, 1.0])},
        "zi": {"f0": sp.Beta.from_params([1.0, 10.0])},
        "offset": {"gamma": sp.prior.between(5.0, 19.0)},
    }[next(iter(options))]
    model = sp.Weibull.fit_bayes(x, c, priors={**SHAPE, **extra}, **options)
    name = {"offset": "gamma", "lfp": "lfp_p", "zi": "f0"}[next(iter(options))]
    assert name in model.posterior.free_names
    lo, hi = model.param_cri(name)
    assert lo < getattr(model, name) < hi
    t = np.array([30.0, 80.0, 200.0])
    np.testing.assert_allclose(model.sf(t) + model.ff(t), 1.0)


def test_lfp_mean_is_infinite():
    x = sp.Weibull.random(30, 100, 2, random_state=4)
    c = np.where(x > 120, 1, 0)
    model = sp.Weibull.fit_bayes(
        np.minimum(x, 120),
        c,
        lfp=True,
        priors={**SHAPE, "lfp_p": sp.Beta.from_params([4.0, 1.0])},
    )
    with pytest.warns(UserWarning, match="infinite"):
        assert model.mean() == np.inf


def test_one_free_parameter_by_grid_and_fixed():
    model = sp.Weibull.fit_bayes(X, C, priors={"beta": sp.prior.fixed(2.0)})
    assert model.posterior.free_names == ["alpha"]
    assert model.params[1] == 2.0
    assert model.param_cri("beta").tolist() == [2.0, 2.0]


def test_entry_points_agree(seals):
    data = sp.SurpyvalData(x=X, c=C)
    by_data = sp.Weibull.fit_bayes(data, priors=SHAPE)
    df = pd.DataFrame({"t": X, "running": C})
    by_df = sp.Weibull.fit_bayes_from_df(
        df, x_col="t", c_col="running", priors=SHAPE
    )
    for other in (by_data, by_df):
        np.testing.assert_allclose(other.params, seals.params, rtol=1e-12)
        np.testing.assert_allclose(
            other.cri([1000.0, 4000.0]),
            seals.cri([1000.0, 4000.0]),
            rtol=1e-12,
        )
    with pytest.raises(ValueError, match="leave out c"):
        sp.Weibull.fit_bayes(data, c=C)


def test_from_df_with_interval_columns_and_a_scalar_truncation():
    """audit B6: ``fit_bayes_from_df`` with ``xl_col``/``xr_col`` and a
    scalar ``tl_col`` is the same fit as ``fit_bayes`` on the arrays;
    ``x_col`` with ``xl_col`` and a ``tl_col`` that is neither a label
    nor a scalar are refused."""
    df = pd.DataFrame(
        {"lo": [1.0, 2, 3, 4, 5], "hi": [2.0, 3, 4, 5, 6], "c": [2] * 5}
    )
    by_df = sp.Weibull.fit_bayes_from_df(
        df, xl_col="lo", xr_col="hi", c_col="c", tl_col=0.5, priors=SHAPE
    )
    direct = sp.Weibull.fit_bayes(
        xl=df.lo, xr=df.hi, c=df.c, tl=0.5, priors=SHAPE
    )
    np.testing.assert_array_equal(by_df.params, direct.params)
    with pytest.raises(ValueError, match="x_col"):
        sp.Weibull.fit_bayes_from_df(
            df, x_col="lo", xl_col="lo", xr_col="hi", priors=SHAPE
        )
    with pytest.raises(ValueError, match="tl_col"):
        sp.Weibull.fit_bayes_from_df(
            df, xl_col="lo", xr_col="hi", tl_col=[0.5] * 5, priors=SHAPE
        )


def test_deterministic(seals):
    again = sp.Weibull.fit_bayes(X, C, priors=SHAPE)
    np.testing.assert_array_equal(
        again.posterior.weights, seals.posterior.weights
    )


def test_unit():
    model = sp.Weibull.fit_bayes(X, C, priors=SHAPE, unit="hours")
    assert model.unit == "hour"
    assert "Unit : hour" in repr(model)
    assert sp.from_dict(model.to_dict()).unit == "hour"


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
def test_update_equals_fitting_everything_at_once(seals):
    first = sp.Weibull.fit_bayes(X[:20], C[:20], priors=SHAPE)
    both = first.update(X[20:], C[20:])
    np.testing.assert_allclose(both.params, seals.params, rtol=1e-12)
    np.testing.assert_allclose(both.param_cri(), seals.param_cri(), rtol=1e-12)


def test_update_replace_supersedes_the_old_records():
    first = sp.Weibull.fit_bayes(X, C, priors=SHAPE)
    # a quarter later every unit has aged 500 h; one more has failed
    x2 = np.where(C == 1, X + 500, X)
    c2 = C.copy()
    c2[10] = 0
    later = first.update(x2, c2, replace=True)
    direct = sp.Weibull.fit_bayes(x2, c2, priors=SHAPE)
    np.testing.assert_allclose(later.params, direct.params, rtol=1e-12)
    assert int(np.sum(later.data.n)) == 40


def _within_grid_tolerance(a, b):
    """The same posterior computed twice (from different starts): every
    parameter's median and interval agree to the grids' accuracy."""
    tol = 1e-3
    for name in a.posterior.free_names:
        qa = a.posterior.quantile(name, [0.025, 0.5, 0.975])
        qb = b.posterior.quantile(name, [0.025, 0.5, 0.975])
        width = qb[2] - qb[0]
        assert np.all(np.abs(qa - qb) <= tol * width), (name, qa, qb)


def test_power_prior_at_weight_one_is_the_joint_fit():
    old = sp.Weibull.random(20, 9000, 2, random_state=4)
    a = sp.Weibull.fit_bayes(X, C, priors=sp.prior.from_data(old))
    b = sp.Weibull.fit_bayes(np.r_[X, old], np.r_[C, np.zeros(20)])
    _within_grid_tolerance(a, b)
    half = sp.Weibull.fit_bayes(
        X, C, priors=sp.prior.from_data(old, weight=0.5)
    )
    w_half = np.diff(half.param_cri("beta"))
    w_full = np.diff(a.param_cri("beta"))
    assert w_half > w_full


def test_from_model():
    old = sp.Weibull.fit(sp.Weibull.random(60, 9000, 1.8, random_state=2))
    prior = sp.prior.from_model(old, weight=0.5)
    assert prior.names == ["alpha", "beta"]
    cov = old.covariance()
    se_log = np.sqrt(np.diag(cov)) / old.params
    np.testing.assert_allclose(
        np.sqrt(np.diag(prior.cov)), se_log / np.sqrt(0.5), rtol=1e-12
    )
    one = sp.prior.from_model(old, param="beta", widen=2)
    assert one.scale == pytest.approx(2 * se_log[1])
    model = sp.Weibull.fit_bayes(X, C, priors={"beta": one})
    assert model.method == "grid"


def test_from_model_refuses_a_model_with_no_verified_maximum():
    given = sp.Weibull.from_params([9000, 2.0])
    with pytest.raises(ValueError, match="Pass the old data instead"):
        sp.prior.from_model(given)


def test_from_model_of_a_bayesian_fit_is_exact(seals):
    earlier = sp.Weibull.fit_bayes(X[2:], C[2:], priors=SHAPE)
    again = sp.Weibull.fit_bayes(
        X[:2], C[:2], priors=sp.prior.from_model(earlier)
    )
    _within_grid_tolerance(again, seals)


def test_from_params():
    p = sp.prior.from_params(sp.Weibull, [9000, 2.0], within=0.2)
    from surpyval.bayes.transforms import Transform

    t = Transform.for_bounds((0, None))
    from scipy import stats

    m, s = p["alpha"]._resolved(t)
    lo, hi = np.exp(stats.norm.ppf([0.05, 0.95], m, s))
    assert (lo, hi) == pytest.approx((9000 / 1.2, 9000 * 1.2))
    model = sp.Weibull.fit_bayes(X, C, priors=p)
    assert np.isfinite(model.log_evidence)


def test_on_quantities_reproduces_the_belief():
    belief = sp.prior.on_quantities(
        sp.Weibull,
        B10=sp.prior.around(5000, between=(3000, 8000)),
        beta=sp.prior.between(1.2, 4),
    )
    check = sp.prior.check(sp.Weibull, belief)
    med, lo, hi = check.table["B10"]
    assert (lo, hi) == pytest.approx((3000, 8000), rel=1e-3)
    assert check.table["beta"][1:] == pytest.approx((1.2, 4), rel=1e-3)
    assert "90% prior interval" in repr(check)


def test_on_quantities_reliability_at_a_time():
    belief = sp.prior.on_quantities(
        sp.Weibull,
        R_at={2000: sp.prior.between(0.9, 0.99)},
        beta=sp.prior.between(1.2, 4),
    )
    model = sp.Weibull.fit_bayes(X, C, priors=belief)
    check = model.prior_check()
    assert check.table["beta"][1:] == pytest.approx((1.2, 4), rel=1e-3)
    assert model.cri(2000).shape == (2,)


def test_on_quantities_needs_one_statement_per_parameter():
    belief = sp.prior.on_quantities(
        sp.Weibull, B10=sp.prior.around(5000, between=(3000, 8000))
    )
    with pytest.raises(ValueError, match="one statement per free"):
        sp.Weibull.fit_bayes(X, C, priors=belief)
    with pytest.raises(ValueError, match="between B0 and B100"):
        sp.prior.on_quantities(sp.Weibull, B200=sp.prior.flat())


def test_prior_check_refuses_flat():
    with pytest.raises(ValueError, match="proper prior"):
        sp.prior.check(sp.Weibull, SHAPE)


def test_compare_to(seals):
    new = sp.Exponential.fit_bayes([300.0, 450, 800], priors=RATE)
    old = sp.Exponential.fit_bayes([100.0, 250, 400], priors=RATE)
    p = new.compare_to(old, on="qf", p=0.1)
    q = old.compare_to(new, on="qf", p=0.1)
    assert new.compare_to(old, x=500).probability > 0.5
    assert p.probability + q.probability == pytest.approx(1.0, abs=1e-9)
    assert p.probability > 0.5
    assert "independent" in repr(p)
    mle = sp.Weibull.fit(sp.Weibull.random(60, 9000, 1.8, random_state=2))
    r = seals.compare_to(mle, on="qf", p=0.1)
    assert 0 < r.probability < 1
    assert "delta-method" in r.assumption
    assert 0 < seals.compare_to(mle, on="sf", x=3000).probability < 1
    with pytest.raises(ValueError, match="needs the time"):
        seals.compare_to(mle, on="sf")


@pytest.fixture(scope="module")
def halves():
    """Two Weibull fits on the two halves of one sample of 30."""
    xs = sp.Weibull.random(30, 100, 2, random_state=5)
    a = sp.Weibull.fit_bayes(xs[:15], priors=SHAPE)
    b = sp.Weibull.fit_bayes(xs[15:], priors=SHAPE)
    return xs, a, b


def test_compare_to_an_independent_copy_of_itself_is_even(halves):
    """audit B7 (a): a posterior against an independent copy of itself
    is greater with probability one half."""
    _, a, _ = halves
    assert a.compare_to(a, on="qf", p=0.1).probability == pytest.approx(
        0.5, abs=1e-9
    )


@pytest.mark.parametrize("t", [50.0, 100.0, 150.0])
def test_compare_to_on_ff_is_the_complement_of_on_sf(halves, t):
    """audit B7 (b): failure probability is 1 - reliability, so the
    probability that it is greater is the complement."""
    _, a, b = halves
    on_ff = a.compare_to(b, on="ff", x=t).probability
    on_sf = a.compare_to(b, on="sf", x=t).probability
    assert on_ff == pytest.approx(1 - on_sf, abs=1e-9)


@pytest.mark.parametrize("on", ["mean", "beta"])
def test_compare_to_a_maximum_likelihood_fit_on_mean_and_a_parameter(
    halves, on
):
    """audit B7 (c): against a maximum-likelihood fit, on the mean and on
    a parameter, the probability is strictly inside (0, 1) and the
    assumption names the delta method."""
    xs, a, _ = halves
    mle = sp.Weibull.fit(xs[:15])
    r = a.compare_to(mle, on=on)
    assert 0 < r.probability < 1
    assert "delta-method" in r.assumption


def test_compare_to_a_much_lower_maximum_likelihood_shape(halves):
    """audit B7 (c): the posterior's beta (median about 1.7) against a
    maximum-likelihood fit to data with shape 0.7 (estimate about 0.6):
    this beta is the greater with probability above one half."""
    _, a, _ = halves
    low = sp.Weibull.fit(sp.Weibull.random(30, 100, 0.7, random_state=5))
    assert a.params[1] > 2 * low.params[1]
    assert a.compare_to(low, on="beta").probability > 0.5


def test_compare_to_refuses_an_unknown_quantity(halves):
    """audit B7 (d): an unknown ``on`` is refused, listing the choices."""
    _, a, b = halves
    with pytest.raises(ValueError, match="on must be"):
        a.compare_to(b, on="median_life")


# ---------------------------------------------------------------------------
# Saving, draws from elsewhere, plots
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["grid", "conjugate"])
def test_round_trips(seals, which):
    model = (
        seals
        if which == "grid"
        else sp.Exponential.fit_bayes([100.0, 250, 400], priors=RATE)
    )
    text = json.dumps(model.to_dict(), allow_nan=False)
    again = sp.from_json(text)
    assert isinstance(again, BayesianParametric)
    t = np.array([100.0, 1000.0, 5000.0])
    np.testing.assert_array_equal(again.sf(t), model.sf(t))
    np.testing.assert_array_equal(again.cri(t), model.cri(t))
    np.testing.assert_array_equal(again.param_cri(), model.param_cri())
    back = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(back.sf(t), model.sf(t))
    if which == "grid":
        pt = [[model.params[0], model.params[1]]]
        np.testing.assert_allclose(
            again.posterior.density(pt), model.posterior.density(pt)
        )
        np.testing.assert_allclose(
            back.posterior.density(pt), model.posterior.density(pt)
        )


@pytest.fixture(scope="module")
def earlier():
    """A sample of 30, and a Bayesian and a maximum-likelihood fit to its
    first half, for priors that come from earlier work."""
    xs = sp.Weibull.random(30, 100, 2, random_state=5)
    old = sp.Weibull.fit_bayes(xs[:15], priors=SHAPE)
    return xs, old, sp.Weibull.fit(xs[:15])


PRIOR_KINDS = {
    "around": lambda xs, old, mle: SHAPE,
    "distribution": lambda xs, old, mle: {
        "beta": sp.Gamma.from_params([4, 2]),
        "alpha": sp.LogNormal.from_params([np.log(100), 0.5]),
    },
    "uniform": lambda xs, old, mle: {"beta": sp.prior.uniform(1, 4)},
    "fixed": lambda xs, old, mle: {"beta": sp.prior.fixed(2.0)},
    "from_params": lambda xs, old, mle: sp.prior.from_params(
        sp.Weibull, [100, 2], within=0.3
    ),
    "on_quantities": lambda xs, old, mle: sp.prior.on_quantities(
        sp.Weibull,
        B10=sp.prior.between(20, 60),
        beta=sp.prior.around(2, between=(1.5, 3)),
    ),
    "from_model": lambda xs, old, mle: sp.prior.from_model(mle, weight=0.5),
    "from_model_param": lambda xs, old, mle: {
        "beta": sp.prior.from_model(mle, param="beta", widen=2)
    },
    "power": lambda xs, old, mle: [
        sp.prior.from_data(xs[:15], weight=0.5),
        SHAPE,
    ],
    "as_prior_exact": lambda xs, old, mle: old.as_prior(),
}


@pytest.mark.parametrize("kind", list(PRIOR_KINDS))
def test_round_trips_for_every_prior_kind(earlier, kind):
    """audit B12: a fit with each kind of prior survives JSON and pickle,
    and the restored priors drive a refit (``update``) exactly as the
    originals do (regression protection for ``prior_from_dict``)."""
    xs, old, mle = earlier
    model = sp.Weibull.fit_bayes(
        xs[15:], priors=PRIOR_KINDS[kind](xs, old, mle)
    )
    q = [50.0, 120.0]
    d = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    again = sp.from_dict(d)
    np.testing.assert_array_equal(again.cri(q), model.cri(q))
    assert again.priors.to_dict() == model.priors.to_dict()
    back = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(back.cri(q), model.cri(q))
    np.testing.assert_allclose(
        again.update([90.0]).params, model.update([90.0]).params, rtol=1e-6
    )


def test_from_draws():
    rng = np.random.default_rng(1)
    draws = np.c_[
        rng.lognormal(np.log(100), 0.1, 4000),
        rng.lognormal(np.log(2), 0.1, 4000),
    ]
    model = BayesianParametric.from_draws(
        sp.Weibull,
        draws,
        ["alpha", "beta"],
        data=sp.SurpyvalData(x=[50.0, 80, 120]),
    )
    assert model.method == "draws"
    np.testing.assert_allclose(model.params, np.median(draws, axis=0))
    t = np.array([50.0, 100.0])
    want = np.mean([sp.Weibull.sf(t, a, b) for a, b in draws], axis=0)
    np.testing.assert_allclose(model.sf(t), want)
    np.testing.assert_allclose(
        model.param_cri("beta"),
        np.quantile(draws[:, 1], [0.025, 0.975]),
        rtol=1e-3,
    )
    again = sp.from_dict(model.to_dict())
    np.testing.assert_array_equal(again.sf(t), model.sf(t))
    # columns in any order, extras by name
    swapped = BayesianParametric.from_draws(
        sp.Weibull, draws[:, ::-1], ["beta", "alpha"]
    )
    np.testing.assert_allclose(swapped.params, model.params)
    off = BayesianParametric.from_draws(
        sp.Weibull,
        np.c_[draws, np.full(4000, 5.0)],
        ["alpha", "beta", "gamma"],
    )
    assert off.offset and off.gamma == 5.0
    with pytest.raises(ValueError, match="missing beta"):
        BayesianParametric.from_draws(sp.Weibull, draws[:, :1], ["alpha"])
    with pytest.raises(ValueError, match="Unknown parameter"):
        BayesianParametric.from_draws(
            sp.Weibull, np.c_[draws, draws[:, 0]], ["alpha", "beta", "k"]
        )


def _draws(seed=1, k=4000):
    rng = np.random.default_rng(seed)
    return np.c_[
        rng.lognormal(np.log(100), 0.1, k),
        rng.lognormal(np.log(2), 0.1, k),
    ]


def test_from_draws_refuses_malformed_draws_and_weights():
    """audit B21: ``from_draws`` refuses a column count that differs from
    the names, a missing or unknown parameter column, a non-finite draw,
    and weights of the wrong length or sign."""
    d = _draws(k=400)
    with pytest.raises(ValueError, match="columns"):
        BayesianParametric.from_draws(
            sp.Weibull, np.c_[d, d[:, 0]], ["alpha", "beta"]
        )
    # (a name in place of beta is first a missing beta ...)
    with pytest.raises(ValueError, match="missing beta"):
        BayesianParametric.from_draws(sp.Weibull, d, ["alpha", "nu"])
    # (... and beside it, an unknown column)
    with pytest.raises(ValueError, match="nu"):
        BayesianParametric.from_draws(
            sp.Weibull, np.c_[d, d[:, 0]], ["alpha", "beta", "nu"]
        )
    bad = d.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        BayesianParametric.from_draws(sp.Weibull, bad, ["alpha", "beta"])
    for w in (np.ones(3), -np.ones(400)):
        with pytest.raises(ValueError, match="weights"):
            BayesianParametric.from_draws(
                sp.Weibull, d, ["alpha", "beta"], weights=w
            )


def test_from_draws_takes_data_as_a_tuple_or_an_array():
    """audit B21: ``data=(x, c)`` and a bare array of times both become
    the model's SurpyvalData."""
    x = np.array([50.0, 80.0, 120.0])
    c = np.array([0, 0, 1])
    by_tuple = BayesianParametric.from_draws(
        sp.Weibull, _draws(k=400), ["alpha", "beta"], data=(x, c)
    )
    assert isinstance(by_tuple.data, sp.SurpyvalData)
    np.testing.assert_array_equal(by_tuple.data.x, x)
    np.testing.assert_array_equal(by_tuple.data.c, c)
    by_array = BayesianParametric.from_draws(
        sp.Weibull, _draws(k=400), ["alpha", "beta"], data=x
    )
    assert isinstance(by_array.data, sp.SurpyvalData)
    np.testing.assert_array_equal(by_array.data.x, x)


def test_from_draws_refuses_draws_outside_the_bounds():
    """audit B2: draws of alpha or beta at or below 0 are refused with a
    ValueError naming the parameter and counting the draws. With these
    draws only beta has some (49 of 500); alpha's are all above 0."""
    rng = np.random.default_rng(5)
    draws = np.c_[rng.normal(100, 30, 500), rng.normal(2, 1.5, 500)]
    assert not np.any(draws[:, 0] <= 0) and np.any(draws[:, 1] <= 0)
    with pytest.raises(ValueError, match="49 of the 500 draws of beta"):
        BayesianParametric.from_draws(sp.Weibull, draws, ["alpha", "beta"])
    draws[0, 0] = -1.0
    with pytest.raises(ValueError, match="1 of the 500 draws of alpha"):
        BayesianParametric.from_draws(sp.Weibull, draws, ["alpha", "beta"])


def test_from_draws_refuses_a_proportion_outside_0_and_1():
    """audit B2: draws of ``lfp_p`` outside (0, 1) are refused with a
    ValueError naming it."""
    rng = np.random.default_rng(5)
    draws = np.c_[
        rng.lognormal(np.log(100), 0.1, 500),
        rng.lognormal(np.log(2), 0.1, 500),
        rng.normal(0.8, 0.3, 500),
    ]
    assert np.any((draws[:, 2] <= 0) | (draws[:, 2] >= 1))
    with pytest.raises(ValueError, match="lfp_p"):
        BayesianParametric.from_draws(
            sp.Weibull, draws, ["alpha", "beta", "lfp_p"]
        )


def test_update_of_a_draws_model_without_data():
    """audit B9: a model from draws with no data has nothing to add to;
    ``replace=True`` refits the new data with the priors given to
    ``from_draws``, the same as ``fit_bayes`` on that data."""
    bare = BayesianParametric.from_draws(
        sp.Weibull, _draws(), ["alpha", "beta"]
    )
    with pytest.raises(ValueError, match="replace=True"):
        bare.update([50.0, 80.0])
    with_priors = BayesianParametric.from_draws(
        sp.Weibull, _draws(), ["alpha", "beta"], priors=SHAPE
    )
    refit = with_priors.update([50.0, 80.0], replace=True)
    assert refit.method == "grid"
    direct = sp.Weibull.fit_bayes([50.0, 80.0], priors=SHAPE)
    np.testing.assert_allclose(refit.params, direct.params, rtol=1e-6)


def test_summaries_of_a_draws_model_and_a_conjugate_fit():
    """audit B16: ``summary()`` and ``repr()`` of a model from draws with
    no data and of a conjugate fit are non-empty text; ``prior_check()``
    of a draws model with proper priors is a PriorCheck, and with none
    given it refuses (a flat prior says nothing to check)."""
    from surpyval.bayes.model import PriorCheck

    bare = BayesianParametric.from_draws(
        sp.Weibull, _draws(), ["alpha", "beta"]
    )
    conjugate = sp.Exponential.fit_bayes([100.0, 250, 400], priors=RATE)
    assert conjugate.method == "conjugate"
    for model in (bare, conjugate):
        assert isinstance(model.summary(), str) and model.summary()
        assert repr(model)
    proper = BayesianParametric.from_draws(
        sp.Weibull,
        _draws(),
        ["alpha", "beta"],
        priors={**SHAPE, "alpha": sp.prior.around(100, between=(60, 160))},
    )
    check = proper.prior_check()
    assert isinstance(check, PriorCheck)
    assert repr(check)
    with pytest.raises(ValueError, match="proper prior"):
        bare.prior_check()


def test_plots(seals):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = seals.plot()
    labels = [line.get_label() for line in ax.get_lines()]
    assert "maximum likelihood" in labels
    # drawn in probabilities, which the axes' scale transforms
    line = ax.get_lines()[labels.index("maximum likelihood")]
    y = np.asarray(line.get_ydata(), dtype=float)
    assert np.all((y[~np.isnan(y)] > 0) & (y[~np.isnan(y)] < 1))
    plt.close("all")
    fig = seals.plot_posterior()
    assert len(fig.axes) == 3
    plt.close("all")
    belief = sp.prior.on_quantities(
        sp.Weibull,
        B10=sp.prior.around(5000, between=(3000, 8000)),
        beta=sp.prior.between(1.2, 4),
    )
    ax = sp.prior.check(sp.Weibull, belief).plot()
    assert len(ax.get_lines()) == 50
    plt.close("all")


def test_a_curved_posterior_is_held():
    # Late entry (all 20 units seen from 59.4) and a wide prior: the
    # curvature at the peak understates the posterior's spread, and the
    # grid, laid along that curvature, widened past 60 of its standard
    # deviations and refused; it is now laid again along the spread it
    # has found (the calibration study's replicates 18, 84 and 94).
    rng = np.random.default_rng(8118)
    theta = np.exp(rng.normal([np.log(100.0), np.log(1.5)], [1.0, 0.5]))
    tl = sp.Weibull.qf(0.2, *theta)
    x = sp.Weibull.qf(rng.uniform(0.2, 1.0, 20), *theta)
    model = sp.Weibull.fit_bayes(
        x,
        tl=tl,
        priors={
            "alpha": sp.prior.NormalOnScale(
                loc=np.log(100.0), scale=1.0, kind="log"
            ),
            "beta": sp.prior.NormalOnScale(
                loc=np.log(1.5), scale=0.5, kind="log"
            ),
        },
    )
    assert model.diagnostics["integration_error"] <= 1e-3


def test_a_draws_model_says_it_is_from_draws_not_the_prior_alone():
    """audit B16: the printout of a model from draws made elsewhere says
    so, rather than calling it a fit of the prior alone on flat priors."""
    model = BayesianParametric.from_draws(
        sp.Weibull, _draws(k=400), ["alpha", "beta"]
    )
    text = repr(model)
    assert "from draws given" in text
    assert "prior alone" not in text
    assert "data alone identify" not in text
    assert "alpha: from the draws given (no prior recorded)" in text
