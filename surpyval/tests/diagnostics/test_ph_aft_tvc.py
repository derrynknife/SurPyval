"""``diagnose()`` of a parametric proportional hazards or accelerated
failure time model fitted to time-varying covariates (``fit_tvc``).

Three kinds of evidence:

* R (``scripts/reference/reference_r_ph_aft_tvc.R``, stored in
  ``tests/reference/data/r_ph_aft_tvc.json``): flexsurv's Weibull PH and
  exponential fits of the start-stop heart data, the per-unit Cox-Snell
  and martingale residuals at its estimate, and the step and quadratic
  likelihood-ratio statistics. The two fits agree to about 1e-6, so the
  residuals are compared to ``rtol=1e-4`` and the statistics to
  ``rtol=1e-5``.
* The identity: a ``fit_tvc`` fit of constant covariates (one row per
  unit, or each unit split into rows with the same covariates) has the
  ordinary fit's residuals, statistics and verdicts.
* Seeded simulations (the generators of
  ``scripts/studies/tvc_diagnose_power.py``) where a time-varying effect
  is misspecified, which the checks must flag.
"""

import json
import re
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.diagnostics import Diagnosis, ph_aft
from surpyval.diagnostics.registry import Context
from surpyval.tests.reference._data import fixture, values

AGAINST_R = dict(rtol=1e-5, atol=1e-8)
RESIDUALS_R = dict(rtol=1e-4, atol=1e-5)
SAME = dict(rtol=1e-6, atol=1e-8)


def _quiet(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(*args, **kwargs)


def _ctx(model, names):
    return Context(model, 0.05, names, list(range(len(names))))


def _stat(d, check, covariates=()):
    table = d.table[d.table["check"] == check]
    return table.loc[table["covariates"] == covariates, "statistic"].iloc[0]


# ---------------------------------------------------------------------------
# Against R
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def heart():
    return fixture("heart")


@pytest.mark.parametrize(
    "fitter, ref",
    [("WeibullPH", "heart_weibullPH"), ("ExponentialPH", "heart_exp")],
)
def test_unit_residuals_and_tests_match_flexsurv(heart, fitter, ref):
    r = values("r_ph_aft_tvc", ref)
    Z = np.column_stack([heart["age"], heart["transplant"]])
    model = _quiet(
        getattr(sp, fitter).fit_tvc,
        heart["id"],
        heart["start"],
        heart["stop"],
        heart["c"],
        Z,
    )
    if fitter == "WeibullPH":
        assert_allclose(model.params[:2], [r["alpha"], r["shape"]], 1e-5)
    else:
        assert_allclose(model.params[0], r["scale_m"], 1e-5)
    assert_allclose(model.params[-2:], r["coef"], 1e-4)

    ctx = _ctx(model, ["age", "transplant"])
    b = ph_aft._base(ctx)
    assert_allclose(b.units.labels, r["unit_ids"])
    res = ph_aft._residual_data(ctx, "PH")
    assert_allclose(res["lo"], r["cox_snell"], **RESIDUALS_R)
    assert np.all(res["c"] == np.where(b.units.event, 0, 1))
    assert_allclose(
        ph_aft._unit_martingale(ctx), r["martingale"], **RESIDUALS_R
    )

    d = model.diagnose()
    assert_allclose(
        _stat(d, "proportional_hazards", ("coef_0",)),
        r["step_age"],
        **AGAINST_R,
    )
    assert_allclose(
        _stat(d, "proportional_hazards", ("coef_1",)),
        r["step_transplant"],
        **AGAINST_R,
    )
    assert_allclose(
        _stat(d, "proportional_hazards", ("coef_0", "coef_1")),
        r["step_global"],
        **AGAINST_R,
    )
    assert_allclose(
        _stat(d, "functional_form", ("coef_0",)), r["square_age"], **AGAINST_R
    )


# ---------------------------------------------------------------------------
# The identity: constant covariates give the ordinary fit's report
# ---------------------------------------------------------------------------
def _constant_data(seed=3, n=80, entry=False):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    g = rng.integers(0, 2, n).astype(float)
    T = rng.weibull(1.5, n) * 5 * np.exp(-0.4 * z + 0.3 * g)
    tl = rng.uniform(0, 0.5, n) if entry else np.zeros(n)
    T = np.maximum(T, tl + 0.05)
    c = (T > 6).astype(int)
    return np.minimum(T, 6), c, np.column_stack([z, g]), tl


def _split(T, c, X, first, pieces):
    """Each unit's follow-up ``(first, T]`` as ``pieces`` rows with the
    same covariates."""
    n = T.size
    cuts = [
        first + (T - np.maximum(first, 0)) * k / pieces
        for k in range(pieces + 1)
    ]
    cuts[0] = first
    i = np.tile(np.arange(n), pieces)
    xl = np.concatenate(cuts[:-1])
    xr = np.concatenate(cuts[1:])
    last = np.r_[np.zeros(n * (pieces - 1)), np.ones(n)].astype(bool)
    cc = np.where(last, np.tile(c, pieces), 1)
    return i, xl, xr, cc, np.tile(X, (pieces, 1))


def _comparable(d):
    t = d.table[d.table["check"] != "calibration"]
    return (
        t["check"].tolist(),
        t["statistic"].to_numpy(float),
        t["verdict"].tolist(),
    )


@pytest.mark.parametrize(
    "fitter", ["WeibullPH", "LogNormalPH", "WeibullAFT", "LogNormalAFT"]
)
@pytest.mark.parametrize("pieces", [1, 3])
def test_constant_covariates_give_the_ordinary_report(fitter, pieces):
    T, c, X, _ = _constant_data()
    M = getattr(sp, fitter)
    ordinary = _quiet(M.fit, T, X, c=c)
    # Followed since -inf: no entry conditioning, for any candidate
    # baseline, as the ordinary fit has none.
    first = np.full(T.size, -np.inf)
    if pieces > 1:
        first = np.zeros(T.size)
        i, xl, xr, cc, Z = _split(T, c, X, first, pieces)
        xl[: T.size] = -np.inf
    else:
        i, xl, xr, cc, Z = np.arange(T.size), first, T, c, X
    tvc = _quiet(M.fit_tvc, i, xl, xr, cc, Z)
    do, dt = ordinary.diagnose(), tvc.diagnose()
    checks_o, stat_o, verdict_o = _comparable(do)
    checks_t, stat_t, verdict_t = _comparable(dt)
    assert checks_t == checks_o and verdict_t == verdict_o
    assert_allclose(stat_t, stat_o, rtol=1e-5, atol=1e-6)
    kind = "PH" if fitter.endswith("PH") else "AFT"
    res_o = ph_aft._residual_data(_ctx(ordinary, ["z", "g"]), kind)
    res_t = ph_aft._residual_data(_ctx(tvc, ["z", "g"]), kind)
    # The ordinary fit's rows are sorted; the units are in id order.
    order = np.asarray(ordinary.data._source_rows)
    unit_lo = np.empty(T.size)
    unit_lo[order] = res_o["lo"]
    assert_allclose(res_t["lo"], unit_lo, rtol=1e-5)
    # The same advice, naming fit_tvc.
    assert [a.features for a in dt.advice] == [
        tuple(re.sub(r"\.fit(?!_tvc)", ".fit_tvc", f) for f in a.features)
        for a in do.advice
        if "calibrat" not in a
    ]


def test_delayed_entry_gives_the_ordinary_ph_report():
    T, c, X, tl = _constant_data(seed=5, entry=True)
    ordinary = _quiet(sp.WeibullPH.fit, T, X, c=c, tl=tl)
    i, xl, xr, cc, Z = _split(T, c, X, tl, 2)
    tvc = _quiet(sp.WeibullPH.fit_tvc, i, xl, xr, cc, Z)
    checks_o, stat_o, verdict_o = _comparable(ordinary.diagnose())
    checks_t, stat_t, verdict_t = _comparable(tvc.diagnose())
    assert checks_t == checks_o and verdict_t == verdict_o
    assert_allclose(stat_t, stat_o, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# The accumulated-age likelihood and residuals of an AFT fit
# ---------------------------------------------------------------------------
def _switching(seed, n=150, misspecified=False):
    """The AFT generator of scripts/studies/tvc_diagnose_power.py: a group
    ``g`` and a covariate that switches on at ``s``."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n).astype(float)
    s = rng.uniform(0.3, 1.5, n)
    k = np.where(g == 1, 2.5, 1.2) if misspecified else np.full(n, 1.5)
    age = 2.0 * rng.weibull(k)
    before, after = np.exp(0.3 * g), np.exp(0.3 * g + 0.6)
    T = np.where(
        age <= before * s, age / before, s + (age - before * s) / after
    )
    C = rng.uniform(1.0, 5.0, n)
    x, c = np.minimum(T, C), (T > C).astype(int)
    one = x <= s
    i = np.r_[np.arange(n), np.flatnonzero(~one)]
    xl = np.r_[np.zeros(n), s[~one]]
    xr = np.r_[np.where(one, x, s), x[~one]]
    cc = np.r_[np.where(one, c, 1), c[~one]]
    Z = np.column_stack([g[i], np.r_[np.zeros(n), np.ones((~one).sum())]])
    return i, xl, xr, cc, Z


@pytest.mark.parametrize("center", [False, True])
def test_the_aft_likelihood_is_the_fits_own(center):
    i, xl, xr, c, Z = _switching(11)
    model = _quiet(sp.WeibullAFT.fit_tvc, i, xl, xr, c, Z + 3.0, center=center)
    ctx = _ctx(model, ["g", "z2"])
    null = ph_aft._fitted_null(ctx)
    assert null.ok
    assert_allclose(null.theta, model._eval_params(), rtol=1e-4)
    assert_allclose(null.value, model.neg_ll(), rtol=1e-9)


def test_aft_residuals_are_the_accumulated_ages():
    i, xl, xr, c, Z = _switching(12)
    model = _quiet(sp.WeibullAFT.fit_tvc, i, xl, xr, c, Z)
    beta = model.params[2:]
    psi = np.bincount(i, weights=np.exp(Z @ beta) * (xr - xl))
    res = ph_aft._residual_data(_ctx(model, ["g", "z2"]), "AFT")
    assert_allclose(res["lo"], psi, rtol=1e-12)
    # A row's share of its unit's hazard adds up to the unit's.
    ctx = _ctx(model, ["g", "z2"])
    units = ph_aft._base(ctx).units
    H = model.model.dist.Hf(psi, *model.params[:2])
    shares = np.bincount(units.index, weights=ph_aft._increments(ctx))
    assert_allclose(shares, H, rtol=1e-10)


# ---------------------------------------------------------------------------
# Seeded misspecification is flagged
# ---------------------------------------------------------------------------
def _ph_wearing_off(seed, n=300):
    """The PH generator of scripts/studies/tvc_diagnose_power.py with the
    misspecified effect: ``z2``'s coefficient is 1.5 before ``t = 1`` and
    0 after."""
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    s = rng.uniform(0.3, 1.5, n)
    E = rng.exponential(size=n)
    T = np.empty(n)
    for u in range(n):
        edges = [0.0] + sorted({s[u], 1.0}) + [np.inf]
        left = E[u]
        for a, b in zip(edges[:-1], edges[1:]):
            effect = (1.5 if a < 1.0 else 0.0) if a >= s[u] else 0.0
            r = np.exp(0.5 * z1[u] + effect)
            H0a = (a / 2.0) ** 1.5
            piece = np.inf if np.isinf(b) else r * ((b / 2.0) ** 1.5 - H0a)
            if left <= piece:
                T[u] = 2.0 * (H0a + left / r) ** (1 / 1.5)
                break
            left -= piece
    C = rng.uniform(1.0, 5.0, n)
    x, c = np.minimum(T, C), (T > C).astype(int)
    one = x <= s
    i = np.r_[np.arange(n), np.flatnonzero(~one)]
    xl = np.r_[np.zeros(n), s[~one]]
    xr = np.r_[np.where(one, x, s), x[~one]]
    cc = np.r_[np.where(one, c, 1), c[~one]]
    Z = np.column_stack([z1[i], np.r_[np.zeros(n), np.ones((~one).sum())]])
    return i, xl, xr, cc, Z


def test_an_effect_that_wears_off_fails_proportional_hazards():
    i, xl, xr, c, Z = _ph_wearing_off(20261105)
    model = _quiet(sp.WeibullPH.fit_tvc, i, xl, xr, c, Z)
    d = model.diagnose()
    ph = d.table[d.table["check"] == "proportional_hazards"]
    verdicts = dict(zip(ph["covariates"], ph["verdict"]))
    assert verdicts[("coef_1",)] == "fail"
    assert verdicts[("coef_0", "coef_1")] == "fail"
    assert verdicts[("coef_0",)] != "fail"
    (advice,) = (a for a in d.advice if "coef_1" in a)
    assert "split each row that spans" in advice
    assert advice.features == (
        "WeibullPH.fit_tvc",
        "LogNormalAFT.fit_tvc",
        "CoxPH.fit_tvc(strata=)",
    )


def test_a_shape_per_group_fails_common_shape():
    i, xl, xr, c, Z = _switching(20261105, n=300, misspecified=True)
    model = _quiet(sp.WeibullAFT.fit_tvc, i, xl, xr, c, Z)
    d = model.diagnose()
    row = d.table[d.table["check"] == "common_shape"].iloc[0]
    assert row["covariates"] == ("coef_0",) and row["verdict"] == "fail"
    (advice,) = (a for a in d.advice if "shape of the Weibull" in a)
    assert advice.features == (
        "WeibullAFT.fit_tvc",
        "CoxPH.fit_tvc(strata=)",
        "CoxPH.fit_tvc_from_df(strata_col=)",
    )


def test_a_factor_that_changes_within_a_unit_is_not_tested_for_shape():
    i, xl, xr, c, Z = _switching(13)
    model = _quiet(sp.WeibullAFT.fit_tvc, i, xl, xr, c, Z)
    d = model.diagnose()
    shape = d.table[d.table["check"] == "common_shape"]
    # g is constant within units; z2 switches on within them.
    assert shape["covariates"].tolist() == [("coef_0",)]
    assert any("coef_1: its level changes within a unit" in n for n in d.notes)


# ---------------------------------------------------------------------------
# Units by name: influence and outliers
# ---------------------------------------------------------------------------
def _with_odd_unit():
    T, c, X, _ = _constant_data(seed=8, n=60)
    T, c = T.copy(), c.copy()
    # One unit fails very early with the covariate that predicts a long
    # life: an outlier and an influential unit.
    k = int(np.argmin(X[:, 0]))
    T[k], c[k] = 0.01, 0
    i, xl, xr, cc, Z = _split(T, c, X, np.zeros(T.size), 2)
    ids = np.array([f"pump-{u:02d}" for u in range(T.size)])[i]
    return ids, xl, xr, cc, Z, f"pump-{k:02d}"


def test_influence_and_outliers_name_the_unit(monkeypatch):
    ids, xl, xr, c, Z, odd = _with_odd_unit()
    model = _quiet(sp.WeibullPH.fit_tvc, ids, xl, xr, c, Z)
    # The odd unit moves the coefficient of coef_1 by about 0.39 standard
    # errors; a lower limit makes the screen warn, to see its advice.
    monkeypatch.setattr(ph_aft, "INFLUENCE_LIMIT", 0.35)
    d = model.diagnose()
    outliers = d.table[d.table["check"] == "outliers"].iloc[0]
    assert outliers["verdict"] == "warn" and odd in outliers["detail"]
    assert any(f"unit {odd}" in a for a in d.advice)
    influence = d.table[d.table["check"] == "influence"]
    assert any(f"unit {odd} moves" in t for t in influence["detail"])
    (advice,) = (a for a in d.advice if f"Unit {odd}" in a)
    assert advice.features == ("WeibullPH.fit_tvc",)
    back = Diagnosis.from_dict(
        json.loads(json.dumps(d.to_dict(), allow_nan=False))
    )
    assert back.advice == d.advice


def test_dfbeta_of_a_unit_is_the_sum_of_its_rows_scores():
    T, c, X, _ = _constant_data(seed=9, n=50)
    ordinary = _quiet(sp.WeibullPH.fit, T, X, c=c)
    i, xl, xr, cc, Z = _split(T, c, X, np.zeros(T.size), 3)
    tvc = _quiet(sp.WeibullPH.fit_tvc, i, xl, xr, cc, Z)
    dfb_o, se_o = ph_aft._dfbeta(_ctx(ordinary, ["z", "g"]))
    dfb_t, se_t = ph_aft._dfbeta(_ctx(tvc, ["z", "g"]))
    by_unit = np.empty_like(dfb_o)
    by_unit[np.asarray(ordinary.data._source_rows)] = dfb_o
    assert_allclose(dfb_t, by_unit, rtol=1e-4, atol=1e-8)
    assert_allclose(se_t, se_o, rtol=1e-5)


# ---------------------------------------------------------------------------
# What is left out, and refusals
# ---------------------------------------------------------------------------
def test_calibration_is_not_made_and_says_why():
    i, xl, xr, c, Z = _switching(14)
    d = _quiet(sp.WeibullPH.fit_tvc, i, xl, xr, c, Z).diagnose()
    assert "calibration" not in set(d.table["check"])
    assert any("covariate path" in n for n in d.notes)


def test_a_weighted_fit_tvc_refuses_saying_what_to_do():
    i, xl, xr, c, Z = _switching(15)
    w = np.random.default_rng(0).uniform(0.5, 2.0, i.size)
    model = _quiet(sp.WeibullPH.fit_tvc, i, xl, xr, c, Z, w=w)
    with pytest.raises(ValueError, match="weights.*without w="):
        model.diagnose()


def test_a_fit_tvc_model_without_its_units_refuses():
    i, xl, xr, c, Z = _switching(16)
    model = _quiet(sp.WeibullPH.fit_tvc, i, xl, xr, c, Z)
    model.tvc_subject_ids = None
    with pytest.raises(ValueError, match="Refit it with fit_tvc"):
        model.diagnose()


def test_plot_draws_every_panel():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    i, xl, xr, c, Z = _switching(17)
    for fitter, first in (
        (sp.WeibullPH, "coef_0: p = "),
        (sp.LogNormalAFT, "standardised residuals (LogNormal paper)"),
    ):
        model = _quiet(fitter.fit_tvc, i, xl, xr, c, Z)
        fig = model.diagnose().plot()
        titles = [ax.get_title() for ax in fig.axes]
        assert titles[0].startswith(first)
        assert not any(t.startswith("calibration") for t in titles)
        plt.close(fig)
