"""Bayesian competing risks on the joint (window) path: failures found in
a window whose cause was not recorded, a cause's joint prior, and WAIC
and leave-one-out with left truncation.

The references, fixed before the first run:

- a window from an unknown cause: the probability that the first
  failure of any cause fell in it, ``S(a) - S(b)`` (over ``S(tl)`` with
  an entry), against the sum over the causes of each one's sub-density
  integrated by adaptive quadrature in time (``scipy.integrate.quad``),
  to 1e-9 relative;
- the Bayesian target's log-likelihood equals the maximum-likelihood
  fit's own (``_interval_likelihood.neg_ll``), to 1e-10;
- with flat priors the posterior's peak is the maximum-likelihood
  estimate, to 2e-5 relative (as for windows with a cause);
- a seeded simulation (seed 2026, 400 units, every third window's cause
  not recorded): the 95% intervals cover the true values;
- a cause's joint prior: with exact and right-censored data the joint
  target gives the cause's own ``fit_bayes`` posterior (means to 1e-8
  relative, 95% intervals to 2e-3, as with priors a parameter at a
  time);
- each row's log-likelihood with its own entry term equals that row's
  log-likelihood as a data set of one row, to 1e-10; the rows' terms
  times their counts sum to the log-likelihood.
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad

import surpyval as sp
from surpyval.univariate.competing_risks import ParametricCompetingRisks
from surpyval.univariate.competing_risks.parametric import (
    _interval_likelihood as joint,
)
from surpyval.univariate.competing_risks.parametric import bayes
from surpyval.univariate.competing_risks.parametric import (
    parametric_competing_risks as pcr,
)
from surpyval.univariate.competing_risks.parametric._joint_bayes import (
    CauseJointPrior,
    JointCompetingRisksTarget,
)

# The windows fixture of the joint tests, with four rows found failed in
# a window whose cause was not recorded (two interval, two left
# censored).
X = [[2, 2], [2, 4], [4, 6], [5, 5], [6, 8], [7, 7], [8, 10], [9, 9]]
X += [[10, 10], [12, 12], [3, 6], [5, 5], [6, 9], [4, 4]]
E = ["a", "b", "a", "b", "a", "a", "b", None, "a", None]
E += [None, None, None, None]
C = [-1, 2, 2, 0, 2, 0, 2, 1, 0, 1, 2, -1, 2, -1]


def _quiet(f, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kw)


def _inspected(n, seed, every=20.0, end=120.0, unknown=3):
    """Two causes, Weibull(100, 2) and Exponential(rate 1/150); every
    other unit inspected every ``every`` hours (found failed in a
    window), the rest seen failing; all stopped at ``end``. Every
    ``unknown``-th window's cause is not recorded."""
    ta = sp.Weibull.random(n, 100, 2, random_state=seed)
    tb = sp.Exponential.random(n, 1 / 150, random_state=seed + 1)
    t = np.minimum(ta, tb)
    cause = np.where(ta < tb, "a", "b").astype(object)
    x, e, c = [], [], []
    windows = 0
    for i in range(n):
        if t[i] > end:
            x.append([end, end])
            e.append(None)
            c.append(1)
        elif i % 2 == 0:
            x.append([t[i], t[i]])
            e.append(cause[i])
            c.append(0)
        else:
            lo = np.floor(t[i] / every) * every
            if lo > 0:
                x.append([lo, lo + every])
                c.append(2)
            else:
                x.append([every, every])
                c.append(-1)
            windows += 1
            e.append(None if windows % unknown == 0 else cause[i])
    return np.array(x), np.array(e, dtype=object), np.array(c)


# -- failures in a window from an unknown cause -----------------------------


@pytest.mark.parametrize(
    "da, pa, db, pb, row, tl",
    [
        (sp.Weibull, [100.0, 2.0], sp.Exponential, [1 / 150], (40, 60), None),
        (sp.Weibull, [100.0, 0.7], sp.Weibull, [80.0, 3.0], (20, 20), None),
        (sp.LogNormal, [4.0, 0.5], sp.Weibull, [60.0, 1.5], (50, 90), 10.0),
        (sp.Weibull, [100.0, 2.0], sp.Exponential, [1 / 150], (30, 30), 5.0),
    ],
)
def test_an_unknown_window_is_any_failure_in_it(da, pa, db, pb, row, tl):
    """A unit found failed in (a, b] (from its entry, or the start, for
    c = -1), its cause not recorded: the sum over the causes of each
    one's sub-density over the window, by adaptive quadrature in time,
    over S(tl) with an entry."""
    flag = -1 if row[0] == row[1] else 2
    x, c, n, e, t = pcr._validate([list(row)], [flag], None, [None], tl)
    data = joint.prepare(["a", "b"], [da, db], x, c, n, e, t)
    theta = np.array(pa + pb, dtype=float)
    got = -float(joint.neg_ll(theta, data))

    def sub(d1, p1, d2, p2):
        return lambda u: float(d1.df(u, *p1) * d2.sf(u, *p2))

    a = (tl if tl is not None else 0.0) if flag == -1 else row[0]
    b = row[1]
    total = 0.0
    for f in (sub(da, pa, db, pb), sub(db, pb, da, pa)):
        v, _ = quad(f, a, b, epsabs=0, epsrel=1e-13, limit=500)
        total += v
    if tl is not None:
        total /= float(da.sf(tl, *pa) * db.sf(tl, *pb))
    assert got == pytest.approx(np.log(total), rel=1e-9)
    # the Bayesian target's term is the same
    target = JointCompetingRisksTarget(
        ["a", "b"], [da, db], x, c, n, e, t, None
    )
    assert target.log_lik(theta[None, :])[0] == pytest.approx(got, abs=1e-12)


@pytest.mark.parametrize(
    "dists, tl",
    [
        ((sp.Exponential, sp.Exponential), None),
        ((sp.Weibull, sp.Exponential), None),
        ((sp.Weibull, sp.LogNormal), 1.0),
    ],
)
def test_target_is_the_mle_likelihood_with_unknown_windows(dists, tl):
    x, c, n, e, t = pcr._validate(X, C, None, E, tl)
    mle = _quiet(
        ParametricCompetingRisks.fit,
        X,
        E,
        c=C,
        dist=dict(zip("ab", dists)),
        tl=tl,
    )
    target = JointCompetingRisksTarget(
        ["a", "b"], list(dists), x, c, n, e, t, mle
    )
    assert float(target.prepared.n_unknown.sum()) == 4.0
    theta0 = target.estimates()
    rng = np.random.default_rng(8)
    for _ in range(4):
        theta = theta0 * np.exp(rng.normal(0, 0.3, theta0.size))
        data = joint.prepare(["a", "b"], list(dists), x, c, n, e, t)
        data.adapt(theta)
        ref = -float(joint.neg_ll(theta, data))
        assert target.log_lik(theta[None, :])[0] == pytest.approx(
            ref, abs=1e-10
        )


def test_the_mle_takes_unknown_windows():
    """The joint fit reaches a verified maximum with them; they move the
    estimates (they say the units failed in the window), and dropping
    them is not the same fit."""
    m = _quiet(ParametricCompetingRisks.fit, X, E, c=C, dist=sp.Exponential)
    assert m.models["a"].maximum == "verified"
    keep = [i for i in range(len(X)) if not (E[i] is None and C[i] != 1)]
    dropped = _quiet(
        ParametricCompetingRisks.fit,
        [X[i] for i in keep],
        [E[i] for i in keep],
        c=[C[i] for i in keep],
        dist=sp.Exponential,
    )
    # four more failures of some cause: both rates rise
    assert np.all(m.params > dropped.params)


@pytest.mark.parametrize(
    "dist, method",
    [
        (sp.Exponential, "grid"),
        ({"a": sp.Weibull, "b": sp.Exponential}, "grid"),
        (sp.Weibull, "laplace"),
    ],
)
def test_flat_priors_peak_at_the_mle_with_unknown_windows(dist, method):
    x, e, c = _inspected(80, seed=3)
    assert any(v is None for v in e[c != 1])
    model = _quiet(ParametricCompetingRisks.fit_bayes, x, e, c=c, dist=dist)
    assert model.joint
    assert model.method == {"a": method, "b": method}
    mode = np.asarray(model.result.diagnostics["mode"])
    np.testing.assert_allclose(mode, model.mle.params, rtol=2e-5)


def test_seeded_recovery_with_unknown_causes():
    x, e, c = _inspected(400, seed=2026)
    model = _quiet(
        ParametricCompetingRisks.fit_bayes,
        x,
        e,
        c=c,
        dist={"a": sp.Weibull, "b": sp.Exponential},
    )
    for k, name, truth in (
        ("a", "alpha", 100.0),
        ("a", "beta", 2.0),
        ("b", "failure_rate", 1 / 150),
    ):
        lo, hi = model.param_cri(k, name)
        assert lo < truth < hi


@pytest.mark.parametrize("fit", ["fit", "fit_bayes"])
def test_no_recorded_cause_at_all_is_refused_saying_what_to_do(fit):
    x = [[2, 4], [5, 5], [6, 8], [9, 9]]
    e = [None, None, None, None]
    c = [2, -1, 2, 1]
    with pytest.raises(ValueError, match="No failure has a recorded cause"):
        getattr(ParametricCompetingRisks, fit)(x, e, c=c)


# -- a cause's joint prior -------------------------------------------------


def _exact_data():
    n, seed = 60, 1
    ta = sp.Weibull.random(n, 100, 2, random_state=seed)
    tb = sp.Weibull.random(n, 150, 1.2, random_state=seed + 1)
    cen = np.full(n, 120.0)
    x = np.minimum.reduce([ta, tb, cen])
    e = np.where(x == cen, None, np.where(ta < tb, "a", "b")).astype(object)
    c = (x == cen).astype(int)
    return x, e, c


def _cause_priors():
    old = _quiet(
        sp.Weibull.fit, sp.Weibull.random(40, 110, 2.2, random_state=11)
    )
    return {
        "on_quantities": sp.prior.on_quantities(
            sp.Weibull,
            B10=sp.prior.around(35, between=(20, 60)),
            beta=sp.prior.between(1.2, 4),
        ),
        "from_model": sp.prior.from_model(old, weight=0.5),
        "MultivariateNormal": sp.prior.MultivariateNormal(
            ["alpha", "beta"], np.log([100.0, 2.0]), [[0.04, 0.0], [0.0, 0.09]]
        ),
    }


@pytest.mark.parametrize(
    "which", ["on_quantities", "from_model", "MultivariateNormal"]
)
def test_a_causes_joint_prior_means_what_it_means_in_its_own_fit(which):
    """With exact and right-censored data the joint target's posterior is
    the factorised one: the cause's joint prior on the joint path gives
    the cause's own fit_bayes posterior."""
    x, e, c = _exact_data()
    priors = {"a": _cause_priors()[which]}
    dist = {"a": sp.Weibull, "b": sp.Exponential}
    alone = _quiet(
        ParametricCompetingRisks.fit_bayes, x, e, c=c, dist=dist, priors=priors
    )
    both = _quiet(
        bayes.fit_joint,
        ParametricCompetingRisks,
        pcr._validate(x, c, None, e, None),
        dist,
        None,
        priors=priors,
    )
    assert both.joint and both.method == {"a": "grid", "b": "grid"}
    joints = both.result.priors.joint
    assert len(joints) == 1 and isinstance(joints[0], CauseJointPrior)
    assert joints[0].names == ["a.alpha", "a.beta"]
    for k in ("a", "b"):
        for name in alone.models[k].posterior.parameter_names:
            assert both.models[k].posterior.mean(name) == pytest.approx(
                alone.models[k].posterior.mean(name), rel=1e-8
            )
            np.testing.assert_allclose(
                both.param_cri(k, name), alone.param_cri(k, name), rtol=2e-3
            )
    # saved and loaded, the cause keeps its prior
    back = _quiet(sp.from_dict, both.to_dict())
    assert len(back.models["a"].priors.joint) == 1


def test_a_causes_joint_prior_on_windows_pulls_its_parameters():
    x, e, c = _inspected(80, seed=3)
    dist = {"a": sp.Weibull, "b": sp.Exponential}
    tight = sp.prior.MultivariateNormal(
        ["alpha", "beta"], np.log([100.0, 2.0]), [[1e-4, 0.0], [0.0, 1e-4]]
    )
    flat = _quiet(ParametricCompetingRisks.fit_bayes, x, e, c=c, dist=dist)
    pulled = _quiet(
        ParametricCompetingRisks.fit_bayes,
        x,
        e,
        c=c,
        dist=dist,
        priors={"a": tight},
    )
    lo, hi = pulled.param_cri("a", "beta")
    flo, fhi = flat.param_cri("a", "beta")
    assert hi - lo < 0.2 * (fhi - flo)
    assert lo < 2.0 < hi


def test_a_causes_joint_prior_refusals_name_the_cause():
    x, e, c = _inspected(80, seed=3)
    wrong = sp.prior.MultivariateNormal(
        ["alpha", "shape"], [0.0, 0.0], np.eye(2)
    )
    with pytest.raises(ValueError, match="No parameter 'a.shape'"):
        ParametricCompetingRisks.fit_bayes(x, e, c=c, priors={"a": wrong})
    both = sp.prior.MultivariateNormal(
        ["alpha", "beta"], [4.6, 0.7], np.eye(2)
    )
    with pytest.raises(ValueError, match="Two priors given for a.beta"):
        ParametricCompetingRisks.fit_bayes(
            x,
            e,
            c=c,
            priors={"a": [both, {"beta": sp.prior.between(1, 3)}]},
        )


# -- WAIC and leave-one-out with left truncation --------------------------


def test_each_rows_entry_term_is_its_own():
    """Each prepared row's log-likelihood (with its own entry term) is
    that row's as a data set of one; times their counts they sum to the
    log-likelihood. WAIC and leave-one-out are then taken."""
    x, e, c = _inspected(60, seed=5)
    # entries at 5 and 15 where the unit was seen after them, else 0
    i = np.arange(len(x))
    tl = np.where((i % 3 == 0) & (x[:, 0] > 5.0), 5.0, 0.0)
    tl = np.where((i % 5 == 0) & (x[:, 0] > 15.0), 15.0, tl)
    dists = [sp.Weibull, sp.Exponential]
    rows = pcr._validate(x, c, None, e, tl)
    target = JointCompetingRisksTarget(["a", "b"], dists, *rows, None)
    theta = np.array([[100.0, 2.0, 1 / 150], [80.0, 1.5, 1 / 120]])
    units = target.unit_log_lik(theta)
    counts = target.unit_counts()
    np.testing.assert_allclose(
        units @ counts, target.log_lik(theta), rtol=1e-12
    )
    # every prepared row, on its own
    data = target.prepared
    order = [data.index_censored]
    for r in data.rows:
        order += [r.index_exact, r.index_window]
    order.append(data.index_unknown)
    order = np.concatenate(order)
    assert sorted(order.tolist()) == list(range(data.n_rows))
    # the prepared rows are the distinct (c, cause, xl, xr, tl): rebuild
    # each from the data
    xv, cv, nv, ev, tv = rows
    xl = xv[:, 0] if xv.ndim == 2 else xv
    xr = xv[:, 1] if xv.ndim == 2 else xv
    checked = 0
    seen = set()
    for i in range(len(xv)):
        key = (cv[i], ev[i], xl[i], xr[i], tv[i, 0])
        if key in seen:
            continue
        seen.add(key)
        one = JointCompetingRisksTarget(
            ["a", "b"],
            dists,
            xv[i : i + 1],
            cv[i : i + 1],
            np.ones(1),
            ev[i : i + 1],
            tv[i : i + 1],
            None,
        )
        ref = one.log_lik(theta)
        # the same row among the full target's units
        hits = [
            j
            for j in range(units.shape[1])
            if np.allclose(units[:, j], ref, rtol=0, atol=1e-10)
        ]
        assert hits, key
        checked += 1
    assert checked == data.n_rows
    model = _quiet(
        ParametricCompetingRisks.fit_bayes,
        x,
        e,
        c=c,
        dist={"a": sp.Weibull, "b": sp.Exponential},
        tl=tl,
    )
    w, loo = model.result.waic(), model.result.loo()
    assert np.isfinite(w.elpd) and np.isfinite(loo.elpd)
    assert abs(w.elpd - loo.elpd) < 1.0
