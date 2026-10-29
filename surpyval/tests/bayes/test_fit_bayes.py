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
