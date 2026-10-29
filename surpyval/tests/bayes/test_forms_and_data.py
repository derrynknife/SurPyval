"""The Bayesian fit with an offset, zero inflation, a power prior or data
it cannot take: the predictive functions held to the support, the
quantile's search, the variance, the prior check, the checks on flat
priors, and the data refused rather than dropped."""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import prior
from surpyval.bayes.prior import quantity_at

# Ten lives that start after a failure-free period of a few hundred hours
X_OFF = np.array([520.0, 610, 700, 760, 830, 905, 990, 1100, 1250, 1400])
OFF_PRIORS = {
    "gamma": prior.between(200, 500),
    "beta": prior.between(0.8, 4),
    "alpha": prior.between(100, 2000),
}
# Twelve units, four dead on arrival (a zero-inflated Weibull)
X_ZI = np.r_[np.zeros(4), [120.0, 200, 260, 330, 410, 520, 600, 700]]
ZI_PRIORS = {"f0": prior.between(0.1, 0.5), "beta": prior.between(0.8, 4)}


@pytest.fixture(scope="module")
def offset_fit():
    return sp.Weibull.fit_bayes(X_OFF, offset=True, priors=OFF_PRIORS)


@pytest.fixture(scope="module")
def zi_fit():
    return sp.Weibull.fit_bayes(X_ZI, zi=True, priors=ZI_PRIORS)


def _unclamped_weibull(monkeypatch):
    """The Weibull's sf and df as bare formulas, which are NaN below 0:
    the predictive functions must not lean on the base distribution to
    hold them to its support."""

    def sf(x, alpha, beta):
        return np.exp(-((x / alpha) ** beta))

    def df(x, alpha, beta):
        z = x / alpha
        return beta / alpha * z ** (beta - 1) * np.exp(-(z**beta))

    monkeypatch.setattr(sp.Weibull, "sf", sf)
    monkeypatch.setattr(sp.Weibull, "df", df)


# ---------------------------------------------------------------------------
# The predictive functions below a point's offset
# ---------------------------------------------------------------------------
def test_offset_predictive_is_held_to_each_points_support(
    offset_fit, monkeypatch
):
    m = offset_fit
    g, w = m.posterior.points[:, 0], m.posterior.weights
    # more than half the posterior has its offset above 450
    assert w[g > 450].sum() > 0.5
    q = np.array([300.0, 450.0, 480.0])
    want = {f: getattr(m, f)(q) for f in ("sf", "ff", "df", "hf", "Hf")}
    want_cri = m.cri(q, on="sf")
    _unclamped_weibull(monkeypatch)
    for f, v in want.items():
        np.testing.assert_allclose(getattr(m, f)(q), v, rtol=1e-12)
    np.testing.assert_allclose(m.cri(q, on="sf"), want_cri, rtol=1e-12)
    # below every offset nothing has failed
    assert m.sf(-1e6) == pytest.approx(1.0, abs=1e-12)
    assert m.df(-1e6) == 0.0


def test_reliability_quantity_is_held_to_the_support(monkeypatch):
    theta = {
        "alpha": np.array([500.0, 500.0]),
        "beta": np.array([1.5, 1.5]),
        "gamma": np.array([100.0, 400.0]),
    }
    _unclamped_weibull(monkeypatch)
    r = quantity_at(sp.Weibull, ("R", 300.0), theta)
    assert r[1] == 1.0
    assert r[0] == pytest.approx(np.exp(-((200 / 500) ** 1.5)))


# ---------------------------------------------------------------------------
# qf's search for the quantile
# ---------------------------------------------------------------------------
def test_offset_qf_finishes_where_the_base_is_nan_below_0(
    offset_fit, monkeypatch
):
    m = offset_fit
    want = m.qf([0.001, 0.01, 0.5])
    _unclamped_weibull(monkeypatch)
    calls = {"n": 0}
    predict = type(m)._predict

    def counted(self, fname, x):
        calls["n"] += 1
        if calls["n"] > 20_000:
            raise AssertionError("qf's bracket search does not end")
        return predict(self, fname, x)

    monkeypatch.setattr(type(m), "_predict", counted)
    np.testing.assert_allclose(m.qf([0.001, 0.01, 0.5]), want, rtol=1e-9)


def test_qf_search_is_capped_and_says_why(monkeypatch):
    m = sp.Normal.fit_bayes(
        [9.0, 10.0, 11.0, 12.0],
        priors={"sigma": prior.between(0.5, 5), "mu": prior.between(5, 15)},
    )
    calls = {"n": 0}

    def stuck(self, fname, x):
        # a predictive failure function that never falls below 0.5
        calls["n"] += 1
        if calls["n"] > 20_000:
            raise AssertionError("qf's bracket search does not end")
        return np.full(np.shape(x), 0.5)

    monkeypatch.setattr(type(m), "_predict", stuck)
    with pytest.warns(UserWarning, match=r"qf\(0.3\) is NaN"):
        out = m.qf([0.3])
    assert np.isnan(out[0])
    assert calls["n"] < 1000

    def undefined(self, fname, x):
        return np.where(np.asarray(x) == np.inf, 1.0, np.nan)

    monkeypatch.setattr(type(m), "_predict", undefined)
    with pytest.warns(UserWarning, match="undefined"):
        assert np.isnan(m.qf([0.3])[0])


def test_offset_qf_stops_at_the_support_edge(offset_fit):
    m = offset_fit
    low = m.qf([1e-12])[0]
    edge = float(np.min(m.posterior.points[:, 0]))
    assert np.isfinite(low) and low >= edge
    assert float(m.ff(low)) == pytest.approx(1e-12, rel=1e-3, abs=1e-15)


# ---------------------------------------------------------------------------
# Quantiles below the zero-inflation mass
# ---------------------------------------------------------------------------
def test_zero_inflated_quantile_below_f0_is_0(zi_fit):
    m = zi_fit
    f0 = m.posterior.points[:, m.posterior.parameter_names.index("f0")]
    q = quantity_at(m.dist, ("q", 0.05), m._theta())
    assert np.all(q[f0 >= 0.05] == 0.0)
    assert np.all(np.isfinite(q))
    np.testing.assert_array_equal(m.quantile_cri(0.05), [0.0, 0.0])
    assert m.quantile_cri(0.05, point=True) == 0.0
    assert m.qf(0.05) == 0.0
    # where the mass is below p the quantile is the base's
    assert np.all(q[f0 < 0.05] > 0)
    # and what the plug-in model says agrees
    pm = sp.Weibull.from_params([300.0, 1.5], f0=0.2)
    th = {
        "alpha": np.array([300.0]),
        "beta": np.array([1.5]),
        "f0": np.array([0.2]),
    }
    for p in (0.1, 0.2, 0.5):
        assert quantity_at(sp.Weibull, ("q", p), th)[0] == pytest.approx(
            float(pm.qf(p))
        )


def test_zero_inflated_reliability_before_0_is_1():
    th = {
        "alpha": np.array([300.0]),
        "beta": np.array([1.5]),
        "f0": np.array([0.2]),
    }
    assert quantity_at(sp.Weibull, ("R", -1.0), th)[0] == 1.0
    assert quantity_at(sp.Weibull, ("R", 0.0), th)[0] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# prior_check of an offset fit
# ---------------------------------------------------------------------------
def test_prior_check_keeps_the_offset(offset_fit):
    table = offset_fit.prior_check().table
    assert "gamma" in table
    # the offset is below the first failure, so is every prior B1 life
    lo, hi = table["gamma"][1:]
    assert 150 < lo < hi < X_OFF.min()
    assert table["B1"][1] > lo


def test_prior_check_with_the_offset_fixed():
    m = sp.Weibull.fit_bayes(
        X_OFF,
        offset=True,
        fixed={"gamma": 300.0},
        priors={
            "beta": prior.between(0.8, 4),
            "alpha": prior.between(100, 2000),
        },
    )
    table = m.prior_check().table
    # every prior life starts at the fixed offset
    assert table["B1"][1] > 300.0


def test_prior_check_refuses_a_flat_offset():
    m = sp.Weibull.fit_bayes(
        X_OFF,
        offset=True,
        method="laplace",
        priors={
            "beta": prior.between(0.8, 4),
            "alpha": prior.between(100, 2000),
        },
    )
    with pytest.raises(ValueError, match="gamma has none"):
        m.prior_check()


# ---------------------------------------------------------------------------
# var() with an offset or zero inflation
# ---------------------------------------------------------------------------
def test_var_of_an_offset_fit_is_the_shifted_variance():
    shape = {"beta": prior.between(1.5, 3)}
    plain = sp.Weibull.fit_bayes(X_OFF - 300, priors=shape)
    off = sp.Weibull.fit_bayes(
        X_OFF, offset=True, fixed={"gamma": 300.0}, priors=shape
    )
    assert off.mean() == pytest.approx(plain.mean() + 300, rel=1e-10)
    assert off.var() == pytest.approx(plain.var(), rel=1e-9)


def test_var_matches_the_predictive_survival_function():
    from scipy import integrate

    # (a shape prior away from 0: the predictive tails of a small shape
    # are too long to integrate numerically)
    shape = {"beta": prior.between(1.5, 3)}
    zi = sp.Weibull.fit_bayes(
        X_ZI, zi=True, priors={**shape, "f0": prior.between(0.1, 0.5)}
    )
    off = sp.Weibull.fit_bayes(
        X_OFF, offset=True, priors={**OFF_PRIORS, **shape}
    )
    for m in (zi, off):
        lo = min(float(m.qf(1e-10)), 0.0)
        e1 = (
            lo
            + integrate.quad(lambda t: float(m.sf(t)), lo, np.inf, limit=400)[
                0
            ]
        )
        e2 = (
            lo**2
            + integrate.quad(
                lambda t: 2 * t * float(m.sf(t)), lo, np.inf, limit=400
            )[0]
        )
        assert m.mean() == pytest.approx(e1, rel=1e-6)
        assert m.var() == pytest.approx(e2 - e1**2, rel=1e-5)


def test_var_with_no_zero_mass_is_the_plain_variance():
    shape = {"beta": prior.between(1.5, 3)}
    plain = sp.Weibull.fit_bayes(X_ZI[4:], priors=shape)
    zi = sp.Weibull.fit_bayes(
        X_ZI[4:], zi=True, fixed={"f0": 0.0}, priors=shape
    )
    assert zi.var() == pytest.approx(plain.var(), rel=1e-9)


def test_var_of_a_limited_failure_population_is_infinite():
    m = sp.Weibull.fit_bayes(
        [100.0, 200, 300, 400, 800, 800, 800, 800],
        c=[0, 0, 0, 0, 1, 1, 1, 1],
        lfp=True,
        priors={
            "lfp_p": prior.between(0.3, 0.9),
            "beta": prior.between(1, 4),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert m.var() == np.inf


# ---------------------------------------------------------------------------
# A power prior's information counts for the checks on flat priors
# ---------------------------------------------------------------------------
# Eight earlier failures, then a zero-failure demonstration: 20 new units
# all running at 600 hours
OLD = np.array([410.0, 620, 700, 880, 950, 1100, 1240, 1500])
NEW = np.full(20, 600.0)
NEW_C = np.ones(20)


@pytest.fixture(scope="module")
def together():
    return sp.Weibull.fit_bayes(
        np.r_[OLD, NEW], c=np.r_[np.zeros(OLD.size), NEW_C]
    )


def _same_posterior(a, b):
    for name in ("alpha", "beta"):
        np.testing.assert_allclose(
            a.posterior.quantile(name, [0.025, 0.5, 0.975]),
            b.posterior.quantile(name, [0.025, 0.5, 0.975]),
            rtol=1e-6,
        )
    np.testing.assert_allclose(
        a.sf([300.0, 600, 1200]), b.sf([300, 600, 1200])
    )


def test_zero_failure_update_with_old_data_as_power_prior(together):
    # the old failures as the prior, the new units (none failed) as data:
    # with weight 1, the same as fitting both together
    m = sp.Weibull.fit_bayes(NEW, c=NEW_C, priors=prior.from_data(OLD))
    _same_posterior(m, together)


def test_zero_failure_update_with_as_prior(together):
    old = sp.Weibull.fit_bayes(OLD)
    m = sp.Weibull.fit_bayes(NEW, c=NEW_C, priors=old.as_prior())
    _same_posterior(m, together)
    _same_posterior(old.update(NEW, c=NEW_C), together)


def test_zero_failure_update_with_a_discounted_power_prior(together):
    shape = {"beta": prior.between(1, 6)}
    full, half = (
        sp.Weibull.fit_bayes(
            NEW, c=NEW_C, priors=[prior.from_data(OLD, weight=w), shape]
        )
        for w in (1.0, 0.5)
    )
    assert half.ok
    # half the old evidence: wider than counting it all
    lo, hi = half.param_cri("alpha")
    lo1, hi1 = full.param_cri("alpha")
    assert hi - lo > hi1 - lo1


def test_flat_still_refused_with_no_failures_anywhere():
    with pytest.raises(ValueError, match="With no failures"):
        sp.Weibull.fit_bayes(
            NEW, c=NEW_C, priors=prior.from_data(NEW, c=NEW_C)
        )


# ---------------------------------------------------------------------------
# Explicit flat priors count as flat
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "priors",
    [
        None,
        {"alpha": prior.flat(), "beta": prior.flat()},
        {"alpha": prior.flat()},
        {"beta": prior.flat()},
    ],
    ids=["omitted", "explicit", "alpha-explicit", "beta-explicit"],
)
def test_every_parameter_flat_needs_a_verified_maximum(priors):
    with pytest.raises(ValueError, match="Every parameter has no prior"):
        sp.Weibull.fit_bayes([1.0, 2.0, 2.0], c=[0, 1, 1], priors=priors)


def test_explicit_flat_fits_where_the_data_determine_it():
    x = np.array([410.0, 620, 700, 880, 950, 1100, 1240, 1500])
    a = sp.Weibull.fit_bayes(x)
    b = sp.Weibull.fit_bayes(x, priors={"alpha": prior.flat()})
    np.testing.assert_array_equal(a.params, b.params)


# ---------------------------------------------------------------------------
# Bernoulli: the data checks Bernoulli.fit makes
# ---------------------------------------------------------------------------
BETA = {"p": sp.Beta.from_params([1.0, 1.0])}


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(x=[0, 1, 2, 1]), "must be either 0 or 1"),
        (dict(x=[0, 1, 0.5, 1]), "must be either 0 or 1"),
        (dict(x=[0, 1, 1], c=[0, 1, 0]), "censored"),
        (dict(x=[0, 1, 1], tl=-1), "truncation"),
        (dict(x=[0, 1, 1], t=[[-1, 2]] * 3), "truncation"),
        (dict(x=[0, 1, 1], zi=True), "no zi form"),
        (dict(x=[0, 1, 1], offset=True), "no offset form"),
    ],
)
def test_bernoulli_refuses_what_its_fit_refuses(kw, match):
    with pytest.raises(ValueError, match=match):
        sp.Bernoulli.fit_bayes(priors=BETA, **kw)


def test_bernoulli_counts_are_kept():
    m = sp.Bernoulli.fit_bayes([0, 1], n=[3, 5], priors=BETA)
    a, b = m.posterior.exact["params"]
    assert (a, b) == (6.0, 4.0)


# ---------------------------------------------------------------------------
# A SurpyvalData holding what the fit cannot take is refused
# ---------------------------------------------------------------------------
PB = {"beta": prior.between(1, 4), "alpha": prior.between(50, 2000)}
_ROWS = dict(x=[100.0, 200, 300, 400], c=[0, 0, 0, 1])


@pytest.fixture(
    params=["covariates", "event types", "time-varying covariates", "markers"]
)
def rich_data(request):
    if request.param == "covariates":
        return sp.SurpyvalData(Z=np.array([[1.0], [2], [3], [4]]), **_ROWS)
    if request.param == "event types":
        return sp.SurpyvalData(x=_ROWS["x"], e=[1, 2, 1, 2])
    d = sp.SurpyvalData(**_ROWS)
    if request.param == "markers":
        d.markers = {0: {50.0: 1.0}}
    else:
        d.time_varying_covariates = object()
    return d


def test_fit_bayes_refuses_data_it_would_drop(rich_data):
    with pytest.raises(ValueError, match="fit_bayes.*without"):
        sp.Weibull.fit_bayes(rich_data, priors=PB)


def test_from_data_refuses_data_it_would_drop(rich_data):
    with pytest.raises(ValueError, match="prior.from_data.*without"):
        prior.from_data(rich_data)


def test_update_refuses_data_it_would_drop(rich_data):
    m = sp.Weibull.fit_bayes([100.0, 200, 300], priors=PB)
    with pytest.raises(ValueError, match="update.*without"):
        m.update(rich_data)


def test_combining_data_keeps_the_unit_and_refuses_extras():
    from surpyval.bayes.model import _combine

    a = sp.SurpyvalData(x=[1.0, 2.0], unit="hours")
    b = sp.SurpyvalData(x=[3.0], unit="hours")
    assert _combine(a, b).unit == a.unit == "hour"
    with pytest.raises(ValueError, match="same unit"):
        _combine(a, sp.SurpyvalData(x=[3.0], unit="days"))
    with pytest.raises(ValueError, match="covariates"):
        _combine(a, sp.SurpyvalData(x=[3.0], Z=np.array([[1.0]])))


# ---------------------------------------------------------------------------
# as_prior keeps the parameters the fit held fixed
# ---------------------------------------------------------------------------
def test_exact_as_prior_keeps_fixed_parameters():
    old = sp.Weibull.fit_bayes(
        OLD, fixed={"beta": 2.0}, priors={"alpha": prior.between(300, 3000)}
    )
    spec = old.as_prior()
    assert isinstance(spec.single["beta"], prior.Fixed)
    assert spec.single["beta"].value == 2.0
    # two new units, both running: the old data and beta = 2 decide it
    m = sp.Weibull.fit_bayes([700.0, 900], c=[1, 1], priors=spec)
    assert m.fixed["beta"] == 2.0
    once = sp.Weibull.fit_bayes(
        np.r_[OLD, 700.0, 900],
        c=np.r_[np.zeros(OLD.size), 1, 1],
        fixed={"beta": 2.0},
        priors={"alpha": prior.between(300, 3000)},
    )
    np.testing.assert_allclose(
        m.posterior.quantile("alpha", [0.025, 0.5, 0.975]),
        once.posterior.quantile("alpha", [0.025, 0.5, 0.975]),
        rtol=1e-6,
    )


def test_an_offset_prior_the_data_agree_with_raises_no_conflict():
    # gamma's scale runs downwards from its upper bound; its prior
    # interval is still read low to high
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = sp.Weibull.fit_bayes(X_OFF, offset=True, priors=OFF_PRIORS)
    assert 200 < m.posterior.quantile("gamma", 0.5) < 500
