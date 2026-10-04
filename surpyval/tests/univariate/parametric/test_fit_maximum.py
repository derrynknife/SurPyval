"""A fit records whether it reached a verified maximum (follow-up to #492).

``Parametric.maximum`` says what a fit's log-likelihood stands on:
``"verified"`` (a zero gradient and a positive-definite Hessian, or an
exact closed form), ``"unverified"`` (the search stopped short; the fit
warned), ``"no finite maximum"`` (the likelihood has none; the fit warned
so), ``"not applicable"`` (not a maximum-likelihood fit) or
``"unknown"`` (restored from a dictionary saved before it existed). It
agrees with the fit's warnings, survives ``to_dict`` / ``from_dict``, and
is what ``fit_best`` reads to set a candidate aside -- not the text of its
warnings.
"""

import json
import warnings

import numpy as np
import pytest

import surpyval as sp
import surpyval as surv
from surpyval.tests.conformance.registry import CASE_BY_NAME
from surpyval.univariate.parametric.fitters import mle as mle_module
from surpyval.univariate.parametric.parametric import (
    MAXIMUM_STATES,
    Parametric,
)
from surpyval.utils.no_maximum import quiet_maximum_warnings

SEVEN = np.arange(1, 8.0)
WEIBULL_50 = np.random.default_rng(5).weibull(2, 50) * 100
EXPO_WEIBULL_100 = sp.ExpoWeibull.random(100, 10, 2, 3, random_state=1)
# The 3-parameter Weibull profile likelihood rises all the way to the
# first failure (#487)
OFFSET_X = [55, 60, 70, 80, 95, 120, 140]


def _fit(fit, *args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model = fit(*args, **kwargs)
    return model, [str(w.message) for w in rec]


def test_an_ordinary_fit_is_a_verified_maximum():
    model, messages = _fit(sp.Weibull.fit, WEIBULL_50)
    assert model.maximum == "verified"
    assert messages == []


@pytest.mark.parametrize(
    "dist, x",
    [
        (sp.Exponential, WEIBULL_50),  # closed form
        (sp.Normal, WEIBULL_50),  # closed form
        (sp.Uniform, WEIBULL_50),  # the extreme observations
        (sp.Gamma, WEIBULL_50),
        (sp.LogNormal, WEIBULL_50),
    ],
)
def test_exact_and_optimised_maxima_are_verified(dist, x):
    assert _fit(dist.fit, x)[0].maximum == "verified"


def test_the_exact_discrete_fits_are_verified():
    assert sp.Bernoulli.fit([0, 1, 1]).maximum == "verified"
    assert sp.Binomial.fit([2, 3, 1, 4], n_trials=5).maximum == "verified"


def test_a_beta4_with_no_finite_maximum_says_so():
    model, messages = _fit(sp.Beta4.fit, SEVEN)
    assert model.maximum == "no finite maximum"
    assert [m for m in messages if m.startswith("No finite maximum")]


def test_an_offset_run_onto_the_first_failure_has_no_finite_maximum():
    model, messages = _fit(sp.Weibull.fit, OFFSET_X, offset=True)
    assert model.maximum == "no finite maximum"
    assert len(messages) == 1
    assert messages[0].startswith("No finite maximum: the offset gamma")


def test_an_offset_exponential_has_a_genuine_maximum():
    # Its density is finite at the origin: a maximum at gamma = x(1)
    model, messages = _fit(sp.Exponential.fit, OFFSET_X, offset=True)
    assert model.maximum != "no finite maximum"
    assert not [m for m in messages if m.startswith("No finite maximum")]


# ---------------------------------------------------------------------------
# A search running off ends there, and says so (#584).
# ---------------------------------------------------------------------------


def _caught(fit, *args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model = fit(*args, **kwargs)
    return model, rec


def test_584_a_search_onto_an_unbounded_edge_stops_there():
    # The Beta4 on its conformance fixture: BFGS ran a onto the smallest
    # observation (alpha = 1.0014), where a shape below 1 makes the
    # likelihood infinite. The other four rungs then took 4 s to end
    # "unverified" at a worse point (log-likelihood 4.00 against 4.18).
    d = CASE_BY_NAME["Beta4"].data()
    model, rec = _caught(sp.Beta4.fit, **d)
    assert model.maximum == "no finite maximum"
    assert model.optimizer == "BFGS"
    assert len(rec) == 1 and rec[0].filename == __file__
    message = str(rec[0].message)
    assert message.startswith(
        "No finite maximum: the Beta4 likelihood is unbounded"
    )
    assert "a = 0.1 on the smallest observation 0.1" in message
    assert "how='MPS'" in message


def test_584_a_runaway_ends_the_search_with_one_warning():
    # The ExpoWeibull's profile log-likelihood rises with beta without end
    # on this sample (-253.47 at beta = 3, -248.64 at 1000) towards a
    # power law ending at the largest observation (-248.4433). The search
    # ran every rung of the ladder and ended "unverified" (4.0 s); it now
    # stops at the first, at a point where Newton's method cannot
    # converge along the profile of beta.
    model, rec = _caught(sp.ExpoWeibull.fit, WEIBULL_50)
    assert model.maximum == "no finite maximum"
    assert len(rec) == 1 and rec[0].filename == __file__
    message = str(rec[0].message)
    assert message.startswith(
        "No finite maximum: the ExpoWeibull likelihood keeps increasing as "
        "beta ("
    )
    assert "power law" in message
    assert -model._neg_ll > -248.7  # past the profile at beta = 468


def test_a_search_stalled_at_a_kink_is_not_a_runaway():
    # The Weibull-LogLogistic spline of the applications page, whose
    # cumulative hazard jumps at its knot: BFGS stopped after eight
    # iterations against the jump with the other parameters' Hessian
    # indefinite, and Newton's test along alpha_ll's profile read that as
    # a runaway. The search ended there, and the answer kept was a start
    # with the knot below every observation (log-likelihood -1750.76),
    # where the later rungs reach -1741.79 with the knot at half the cap.
    from autograd import numpy as anp

    from surpyval.datasets import load_boston_housing

    x, c, n, _ = surv.xcnt_handler(load_boston_housing()["medv"].values)
    c[-1] = 1

    def Hf(x, *params):
        x = anp.array(x)
        knot = 50 * params[0]
        w, ll = params[1:3], params[3:]
        below = surv.Weibull.Hf(x, *w)
        above = surv.Weibull.Hf(knot, *w) + surv.LogLogistic.Hf(x, *ll)
        return anp.where(x < knot, below, above)

    spline = surv.CustomDistribution(
        "Spline",
        Hf,
        ["knot_frac", "alpha_w", "beta_w", "alpha_ll", "beta_ll"],
        ((0, 1), (0, None), (0, None), (0, None), (0, None)),
        (0, anp.inf),
    )
    model, _ = _caught(spline.fit, x=x, c=c, n=n, lfp=True)
    assert model.maximum != "no finite maximum"
    assert 0.45 < model.params[0] < 0.55
    assert -model._neg_ll > -1742.0


def test_584_the_frechet_limit_is_named():
    # Interval censored and truncated rows on which mu runs off (to 8e19
    # by the time the whole ladder had run, 23 s): the likelihood tends to
    # that of a Frechet distribution, a GumbelLEV of log(x), whose own fit
    # has log-likelihood -73.38 (the ExpoWeibull's profile: -73.91 at
    # mu = 4000, -73.58 at 1e8).
    rng = np.random.default_rng(1)
    x = 5 + 3 * rng.weibull(2.0, 60)
    c = rng.choice([0, 1, -1], 60)
    xl, xr = np.r_[x[10:], x[:10]], np.r_[x[10:], x[:10] + 1.0]
    c = np.r_[c[10:], np.full(10, 2)]
    tl = np.r_[np.zeros(50), np.full(10, 1.0)]
    model, rec = _caught(sp.ExpoWeibull.fit, xl=xl, xr=xr, c=c, tl=tl)
    assert model.maximum == "no finite maximum"
    assert len(rec) == 1 and rec[0].filename == __file__
    message = str(rec[0].message)
    assert "as mu (" in message and "GumbelLEV" in message
    frechet = sp.GumbelLEV.fit(
        xl=np.log(xl), xr=np.log(xr), c=c, tl=np.where(tl > 0, 0.0, -np.inf)
    )
    exact_log_x = np.sum(np.log(xl[c == 0]))
    assert -model._neg_ll < -frechet._neg_ll - exact_log_x


def test_584_a_maximum_on_a_finite_bound_is_not_a_runaway():
    # The offset of an Exponential rises to the first failure, where its
    # likelihood is highest (a maximum on the edge of the space): its
    # search parameter runs off, but the parameter itself stops at a
    # finite bound.
    case = CASE_BY_NAME["Exponential[offset]"]
    model, rec = _caught(case.fit, case.data())
    assert model.maximum == "verified"
    assert not rec


def test_584_a_rung_that_failed_on_a_slope_is_not_a_runaway():
    # An offset Weibull with an observation at -1: BFGS stopped against
    # the wall where the likelihood is not defined, at a point with a
    # negative curvature along beta's profile and a slope far from flat
    # (log-likelihood -618930), and the later rungs reached the maximum
    # or not, depending on the CPU's arithmetic: with AVX-512 disabled the
    # fit ended "no finite maximum". From a start fitted to the data at its
    # own offset (#622) BFGS reaches the maximum on both paths.
    case = CASE_BY_NAME["Weibull[offset]"]
    d = case.data()
    x = np.array(d["x"], dtype=float)
    x[0] = -1.0
    model, rec = _caught(case.fit, {**d, "x": x})
    assert model.maximum == "verified"
    assert -model._neg_ll == pytest.approx(-41.5995068, abs=1e-6)
    assert not [w for w in rec if "finite maximum" in str(w.message)]


# ---------------------------------------------------------------------------
# A Normal running off a truncated sample (#594).
# ---------------------------------------------------------------------------

# Each row's window is bounded above, so as mu grows (and sigma with its
# square root) the Normal tends to an exponential tilt inside every window,
# and the likelihood to a supremum it never reaches.
RUNS_UP = dict(
    x=[8.0, 15.0], c=[0, 1], n=[1, 2], tl=[5.0, 14.0], tr=[9.5, 19.0]
)
RUNS_UP_TOO = dict(
    x=[2.5, 1.5, 7.5, 2.0, 14.5],
    c=[-1, 1, 0, 1, 1],
    n=[3, 3, 3, 2, 3],
    tl=[-np.inf, 0.5, -np.inf, 0.0, -np.inf],
    tr=[6.5, 5.0, 8.5, np.inf, 17.5],
)


@pytest.mark.parametrize("data", [RUNS_UP, RUNS_UP_TOO])
def test_594_a_normal_running_off_says_so(data):
    # These ran the whole ladder (11 s and 13 s), ending "unverified", the
    # second at a log-likelihood of -9.67 that was rounding: its windows'
    # probabilities had underflowed, and the supremum is -10.03. (Far
    # enough out such a likelihood is flat to the verification's tolerance:
    # the first one's BFGS point, at mu = 1.6e4, passes it, and is a
    # runaway all the same; so a verified answer is checked too.)
    model, rec = _caught(sp.Normal.fit, **data)
    assert model.maximum == "no finite maximum"
    assert len(rec) == 1 and rec[0].filename == __file__
    assert str(rec[0].message).startswith(
        "No finite maximum: the Normal likelihood keeps increasing as mu ("
    )
    assert model.params[0] > 1e3


def test_594_a_window_in_the_far_lower_tail_keeps_its_digits():
    # Windows below the smallest normal float: the plain difference of the
    # CDFs had lost its digits (log-likelihood -9.6723 at mu = 4456,
    # sigma = 115.213; -inf further out). In log space it is the profile's
    # smooth value; the reference is the same sum with each window from
    # scipy's log_ndtr.
    from scipy.special import log_ndtr

    data = sp.SurpyvalData(**RUNS_UP_TOO)
    mu, sigma = 4456.0, 115.213

    def log_window(lo, hi):
        a, b = log_ndtr((lo - mu) / sigma), log_ndtr((hi - mu) / sigma)
        return b + np.log1p(-np.exp(a - b))

    d = RUNS_UP_TOO
    rows = [
        log_window(-np.inf, 2.5) - log_window(-np.inf, 6.5),
        log_window(1.5, 5.0) - log_window(0.5, 5.0),
        -0.5 * ((7.5 - mu) / sigma) ** 2
        - np.log(sigma * np.sqrt(2 * np.pi))
        - log_window(-np.inf, 8.5),
        0.0,  # (S(2) / S(0) is 1 to rounding)
        log_window(14.5, 17.5) - log_window(-np.inf, 17.5),
    ]
    expected = float(np.dot(d["n"], rows))
    got = -float(sp.Normal._neg_ll_func(data, mu, sigma, 0.0, 0.0, 1.0))
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(-10.0393909, abs=1e-6)


@pytest.mark.parametrize(
    "dist, x",
    [
        (sp.Weibull, WEIBULL_50),
        (sp.Gamma, WEIBULL_50),
        (sp.ExpoWeibull, EXPO_WEIBULL_100),
    ],
)
def test_a_starved_search_is_unverified(monkeypatch, dist, x):
    # No point the search reaches passes the check: every start fails,
    # and the fit warns and records it. A two-parameter family was
    # silent: the exemption meant for the Uniform (whose parameters are
    # its support's ends) matched every family with k = 2. (Data on
    # which each has a maximum: on WEIBULL_50 the ExpoWeibull has none,
    # and its search finds that, #584.)
    monkeypatch.setattr(mle_module, "is_local_minimum", lambda *a, **k: False)
    model, messages = _fit(dist.fit, x)
    assert model.maximum == "unverified"
    assert len(messages) == 1
    assert "did not reach a verified maximum" in messages[0]


def test_the_flag_agrees_with_the_warnings_when_they_are_held_back():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with quiet_maximum_warnings():
            beta4 = sp.Beta4.fit(SEVEN)
            expo = sp.ExpoWeibull.fit(WEIBULL_50)
            offset = sp.Weibull.fit(OFFSET_X, offset=True)
    assert rec == []
    assert beta4.maximum == "no finite maximum"
    assert expo.maximum == "no finite maximum"  # (#584)
    assert offset.maximum == "no finite maximum"


@pytest.mark.parametrize("how", ["MPS", "MSE", "MPP", "MOM"])
def test_other_estimators_do_not_maximise_the_likelihood(how):
    model, _ = _fit(sp.Weibull.fit, WEIBULL_50, how=how)
    assert model.maximum == "not applicable"


def test_models_not_fitted_by_maximum_likelihood():
    assert sp.Weibull.from_params([10, 2]).maximum == "not applicable"
    assert (
        sp.Weibull.from_params([10, 2], lfp_p=0.9).maximum == "not applicable"
    )
    fitted = sp.Weibull.fit(WEIBULL_50)
    assert fitted.with_params([90, 2]).maximum == "not applicable"
    x = np.array([1.0, 2.0, 3.0, 4.0])
    F = np.array([0.1, 0.3, 0.6, 0.9])
    assert sp.Weibull.fit_from_ecdf(x, F).maximum == "not applicable"


@pytest.mark.parametrize(
    "fit",
    [
        lambda: sp.Weibull.fit(WEIBULL_50),
        lambda: sp.Beta4.fit(SEVEN),
        lambda: sp.ExpoWeibull.fit(WEIBULL_50),
        lambda: sp.Weibull.fit(WEIBULL_50, how="MPS"),
        lambda: sp.Weibull.from_params([10, 2]),
    ],
)
def test_the_flag_round_trips(fit):
    model, _ = _fit(fit)
    stored = json.loads(json.dumps(model.to_dict()))
    assert stored["maximum"] == model.maximum
    assert stored["schema"] == 1  # an older reader ignores it
    assert sp.from_dict(stored).maximum == model.maximum


def test_a_dictionary_saved_before_the_flag_reads_as_unknown():
    old = sp.Weibull.fit(WEIBULL_50).to_dict()
    del old["maximum"]
    assert sp.from_dict(old).maximum == "unknown"
    old = sp.Weibull.fit(WEIBULL_50, how="MPS").to_dict()
    del old["maximum"]
    assert sp.from_dict(old).maximum == "not applicable"


def test_a_bad_stored_flag_is_refused():
    stored = sp.Weibull.fit(WEIBULL_50).to_dict()
    stored["maximum"] = "maybe"
    with pytest.raises(ValueError, match="'maximum'"):
        Parametric.from_dict(stored)


def test_every_state_is_documented():
    doc = Parametric.__doc__
    for state in MAXIMUM_STATES:
        assert f'``"{state}"``' in doc


# ---------------------------------------------------------------------------
# fit_best reads the flag, not the warnings
# ---------------------------------------------------------------------------
def _with_flag(monkeypatch, dist, maximum, message=None):
    """``dist.fit`` returning its model with ``maximum`` set, and giving
    ``message`` as a warning if one is given."""
    fit = dist.fit

    def patched(*args, **kwargs):
        model = fit(*args, **kwargs)
        model.maximum = maximum
        if message is not None:
            warnings.warn(message)
        return model

    monkeypatch.setattr(dist, "fit", patched)


def test_fit_best_sets_aside_a_candidate_by_its_flag(monkeypatch):
    # The Weibull wins on this sample among these three (a Weibull with
    # shape 2); flagged, it is set aside although it warned nothing.
    include = ["Weibull", "Gamma", "LogNormal"]
    assert sp.fit_best(WEIBULL_50, include=include).dist.name == "Weibull"
    _with_flag(monkeypatch, sp.Weibull, "unverified")
    model, messages = _fit(sp.fit_best, WEIBULL_50, include=include)
    assert model.dist.name != "Weibull"
    aside = [m for m in messages if m.startswith("fit_best set aside")]
    assert len(aside) == 1
    assert "Weibull (its fit is not a verified maximum)" in aside[0]


def test_fit_best_reads_no_finite_maximum_from_the_flag(monkeypatch):
    _with_flag(monkeypatch, sp.Weibull, "no finite maximum")
    model, messages = _fit(
        sp.fit_best, WEIBULL_50, include=["Weibull", "Gamma"]
    )
    assert model.dist.name == "Gamma"
    assert [m for m in messages if "Weibull (its likelihood has no" in m]


def test_fit_best_ignores_the_text_of_a_verified_candidate(monkeypatch):
    # A warning that merely reads like a no-maximum one does not set a
    # verified candidate aside; it is passed on as any other warning is.
    text = "No finite maximum: said, but not so"
    _with_flag(monkeypatch, sp.Weibull, "verified", message=text)
    model, messages = _fit(
        sp.fit_best, WEIBULL_50, include=["Weibull", "Gamma", "LogNormal"]
    )
    assert model.dist.name == "Weibull"
    assert text in messages
    assert not [m for m in messages if m.startswith("fit_best set aside")]


def test_fit_best_holds_back_the_candidates_own_warnings():
    model, messages = _fit(
        sp.fit_best, WEIBULL_50, include=["ExpoWeibull", "Weibull", "Beta4"]
    )
    assert model.dist.name == "Weibull"
    assert not [m for m in messages if "did not reach a verified" in m]
    assert not [m for m in messages if m.startswith("No finite maximum")]
    assert len(messages) == 1 and messages[0].startswith("fit_best set")


# ---------------------------------------------------------------------------
# Every fit reports its optimizer.
# ---------------------------------------------------------------------------


W, E, G = surv.Weibull, surv.Exponential, surv.Geometric


@pytest.mark.parametrize("how", ["MPP", "MOM", "MSE", "MPS"])
def test_every_fit_reports_its_optimizer(how):
    np.random.seed(1)
    model = W.fit(W.random(30, 10, 3), how=how)
    assert isinstance(model.optimizer, str) and model.optimizer
