"""SurPyval's non-parametric estimators against stored R results (#379).

The references are R ``survival::survfit`` (Kaplan-Meier with Greenwood,
Nelson-Aalen, Fleming-Harrington, the restricted mean, Turnbull, and the
MCF of recurrent events) and ``npsurv::npsurv`` (the Turnbull NPMLE),
computed by ``scripts/reference/reference_r.R`` on the shared fixtures.

Tolerances: the Kaplan-Meier, Nelson-Aalen and MCF estimates and their
variances are closed-form sums over the same risk sets in both programs,
so they must agree to rounding (``rtol=1e-9``; the files keep 15
significant digits). The Turnbull NPMLE is iterative on both sides; see
the tolerances there.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

import surpyval as sp
from surpyval.recurrent import NonParametricCounting

from ._data import fixture, values

EXACT = dict(rtol=1e-9, atol=1e-12)


def _right_censored(name):
    """(x, c, tl) of each right-censored fixture a survfit reference was
    computed on."""
    if name == "lung":
        d = fixture("lung")
        return d["time"], d["c"], None
    if name == "aml_maintained":
        d = fixture("aml")
        keep = d["maintained"] == 1
        return d["time"][keep], 1 - d["status"][keep], None
    if name == "ties":
        d = fixture("ties")
        return d["x"], d["c"], None
    d = fixture("left_truncation")
    return d["x"], d["c"], d["tl"]


def _at(model, attribute, t):
    """A model's per-step array (e.g. its Greenwood sum) read as a right-
    continuous step function at ``t``."""
    idx = np.searchsorted(model.x, t, side="right") - 1
    return getattr(model, attribute)[idx]


KM_CASES = ["lung", "aml_maintained", "ties", "left_truncation"]


@pytest.mark.parametrize("name", KM_CASES)
def test_kaplan_meier_matches_survfit(name):
    x, c, tl = _right_censored(name)
    ref = values("r_survival", "km_" + name)
    model = sp.KaplanMeier.fit(x, c=c, tl=tl)
    t = ref["time"]
    assert_allclose(model.sf(t), ref["surv"], **EXACT)
    # survfit's std.err is the standard error of -log S: the square root
    # of Greenwood's sum.
    assert_allclose(
        np.sqrt(_at(model, "greenwood", t)), ref["std_err"], **EXACT
    )
    # The default bounds are survfit's conf.type = "log-log".
    bounds = model.cb(t)
    finite = ~np.isnan(ref["lower"])
    assert_allclose(bounds[finite, 0], ref["lower"][finite], **EXACT)
    assert_allclose(bounds[finite, 1], ref["upper"][finite], **EXACT)


@pytest.mark.parametrize("name", KM_CASES)
def test_kaplan_meier_median_and_interval_match_survfit(name):
    # survfit inverts the log-log pointwise interval (Brookmeyer-Crowley);
    # a bound the curve never crosses is NA in R and NaN in SurPyval.
    x, c, tl = _right_censored(name)
    ref = values("r_survival", "km_" + name)
    model = sp.KaplanMeier.fit(x, c=c, tl=tl)
    assert model.median == ref["median"]
    expected = [
        np.nan if v is None else v
        for v in (ref["median_lower"], ref["median_upper"])
    ]
    assert_allclose(model.quantile_cb(0.5)[0], expected, **EXACT)


@pytest.mark.parametrize("name", ["lung", "ties", "left_truncation"])
def test_nelson_aalen_matches_survfit(name):
    x, c, tl = _right_censored(name)
    ref = values("r_survival", "na_" + name)
    model = sp.NelsonAalen.fit(x, c=c, tl=tl)
    t = ref["time"]
    assert_allclose(model.Hf(t), ref["cumhaz"], **EXACT)
    # Aalen's (Poisson) variance, survfit's std.chaz with ctype = 1.
    assert_allclose(
        np.sqrt(_at(model, "greenwood", t)), ref["std_chaz"], **EXACT
    )


def test_fleming_harrington_matches_survfit_ctype2():
    # survfit's ctype = 2 is the Fleming-Harrington tie correction: d tied
    # events add 1/r + 1/(r - 1) + ... to the cumulative hazard.
    x, c, _ = _right_censored("ties")
    ref = values("r_survival", "fh_ties")
    model = sp.FlemingHarrington.fit(x, c=c)
    t = ref["time"]
    assert_allclose(model.Hf(t), ref["cumhaz"], **EXACT)
    assert_allclose(
        np.sqrt(_at(model, "greenwood", t)), ref["std_chaz"], **EXACT
    )


@pytest.mark.parametrize(
    "ref_id, name",
    [
        ("rmst_lung_500", "lung"),
        ("rmst_lung_1000", "lung"),
        ("rmst_ties_15", "ties"),
    ],
)
def test_rmst_and_median_match_survfit(ref_id, name):
    x, c, _ = _right_censored(name)
    ref = values("r_survival", ref_id)
    model = sp.KaplanMeier.fit(x, c=c)
    res = model.rmst(tau=ref["tau"])
    assert_allclose(res["rmst"], ref["rmean"], **EXACT)
    assert_allclose(res["se"], ref["se_rmean"], **EXACT)
    # R's median takes the midpoint when the curve sits at exactly 0.5 over
    # an interval; none of these curves does, so the medians must agree.
    assert model.median == ref["median"]


def _interval_bounds():
    d = fixture("interval")
    right = np.where(np.isnan(d["right"]), np.inf, d["right"])
    return d["left"], right


def _turnbull():
    left, right = _interval_bounds()
    # The Kaplan-Meier option is the NPMLE (the default Fleming-Harrington
    # one is not). The EM is run to convergence so the comparison with
    # npsurv's constrained Newton solution is about the estimator, not
    # the iteration count.
    return sp.Turnbull.fit(
        xl=left,
        xr=right,
        turnbull_estimator="Kaplan-Meier",
        tol=1e-12,
        max_iter=100_000,
    )


def test_turnbull_matches_npsurv_npmle():
    ref = values("r_npsurv", "npmle_interval")
    model = _turnbull()
    # The NPMLE is unique only outside the innermost (Turnbull)
    # intervals, where the mass sits: check it at their right ends and,
    # for intervals of positive width, just at their left ends.
    S_after = 1 - np.cumsum(ref["p"])
    S_before = np.concatenate([[1.0], S_after[:-1]])
    # npsurv stops at a gradient of 4e-11 and the EM at a change of 1e-12;
    # 1e-6 on a probability is far inside any meaningful difference.
    assert_allclose(model.sf(ref["right"]), S_after, atol=1e-6)
    wide = ref["right"] > ref["left"]
    assert_allclose(model.sf(ref["left"][wide]), S_before[wide], atol=1e-6)


def test_turnbull_matches_survfit_interval2_loosely():
    # survfit's own Turnbull EM stops early (its curve is ~1e-3 from the
    # npsurv NPMLE, which SurPyval matches to 1e-6 above) and reports the
    # curve at points inside the Turnbull intervals. Read survfit's step
    # function at the npsurv interval ends, where the NPMLE is unique.
    ref = values("r_survival", "turnbull_interval")
    npmle = values("r_npsurv", "npmle_interval")
    q = npmle["right"]
    idx = np.searchsorted(ref["time"], q, side="right") - 1
    survfit_at = np.where(idx >= 0, ref["surv"][np.maximum(idx, 0)], 1.0)
    assert_allclose(_turnbull().sf(q), survfit_at, atol=5e-3)


def test_mcf_matches_survfit_with_id():
    # With an id, survfit's Nelson-Aalen on the gap times is the MCF and
    # its std.chaz is the robust (infinitesimal jackknife) error, which is
    # the Lawless-Nadeau variance SurPyval attaches.
    d = fixture("mettas_zhao")
    ref = values("r_survival", "mcf_mettas_zhao")
    model = NonParametricCounting.fit(d["x"], i=d["i"], c=d["c"])
    t = ref["time"]
    assert_allclose(model.mcf(t), ref["cumhaz"], **EXACT)
    # Normal bounds are M +- z sd, so their half-width gives sd.
    bounds = model.mcf_cb(t, bound_type="normal")
    sd = (bounds[:, 1] - bounds[:, 0]) / (2 * norm.ppf(0.975))
    assert_allclose(sd, ref["std_chaz"], **EXACT)
