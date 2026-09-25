"""Regression tests for the second round of regression-model bug fixes."""

import itertools
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from surpyval import (
    PO,
    CoxPH,
    Weibull,
    WeibullAH,
    WeibullFrailty,
    WeibullPH,
)
from surpyval.datasets import load_tires_data
from surpyval.univariate.regression import AcceleratedLife, Power
from surpyval.univariate.regression._fit_skeleton import finite_start
from surpyval.univariate.regression.parametric_regression_model import (
    ParametricRegressionModel,
)
from surpyval.univariate.regression.proportional_hazards.cox_ph import (
    CoxPH_,
)
from surpyval.utils import validate_coxph

# -- 1. Cox refuses left- and interval-censored rows ------------------------


def _cox_data() -> tuple:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    Z = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).reshape(-1, 1)
    return x, Z


def test_cox_rejects_left_censoring():
    # c = -1 used to be read as right-censored: the fit matched c[2] = 1.
    x, Z = _cox_data()
    with pytest.raises(ValueError, match="left-censored"):
        CoxPH.fit(x, Z, c=[0, 0, -1, 0, 0, 1])


def test_cox_rejects_interval_censoring():
    # Interval rows used to fail with an IndexError deep in the generator.
    x, Z = _cox_data()
    x2 = [[1, 2], 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="interval-censored"):
        CoxPH.fit(x2, Z, c=[2, 0, 0, 0, 0, 1])


@pytest.mark.parametrize("method", ["breslow", "efron", "exact", "kp"])
def test_cox_rejects_left_censoring_every_entry_point(method):
    x, Z = _cox_data()
    c = [0, 0, -1, 0, 0, 1]
    with pytest.raises(ValueError, match="parametric regression"):
        CoxPH.fit(x, Z, c=c, method=method, strata=[0, 0, 0, 1, 1, 1])
    df = pd.DataFrame({"x": x, "z": Z[:, 0], "c": c})
    with pytest.raises(ValueError, match="parametric regression"):
        CoxPH.fit_from_df(df, x_col="x", Z_cols="z", c_col="c", method=method)


def test_cox_accepts_two_column_exact_times():
    # A two-column x with xl == xr everywhere is exact / right-censored data
    # written as intervals, and fits as such.
    x, Z = _cox_data()
    c = [0, 0, 1, 0, 0, 1]
    two_col = CoxPH.fit(np.column_stack([x, x]), Z, c=c)
    assert np.allclose(two_col.params, CoxPH.fit(x, Z, c=c).params)


# -- 2. CoxPH.fit_from_df delayed entry ---------------------------------------


def test_cox_fit_from_df_tl_col_matches_fit():
    rng = np.random.default_rng(1)
    n = 120
    z = rng.normal(size=n)
    tl = rng.uniform(0, 1.5, size=n)
    x = tl + rng.exponential(np.exp(-0.7 * z))
    c = (rng.uniform(size=n) < 0.2).astype(int)
    df = pd.DataFrame({"x": x, "z": z, "c": c, "entry": tl})

    for method in ("breslow", "efron"):
        from_df = CoxPH.fit_from_df(
            df, x_col="x", Z_cols="z", c_col="c", tl_col="entry", method=method
        )
        direct = CoxPH.fit(x, z.reshape(-1, 1), c=c, tl=tl, method=method)
        assert np.allclose(from_df.params, direct.params)
        # ... and the entry ages change the answer.
        ignored = CoxPH.fit(x, z.reshape(-1, 1), c=c, method=method)
        assert not np.allclose(from_df.params, ignored.params)


def test_cox_fit_from_df_masks_tl_and_strata_with_missing_covariates():
    # Rows with a missing covariate are dropped; the entry ages and stratum
    # labels must drop with them (strata used to raise a length mismatch).
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8.0],
            "z": [0, 1, np.nan, 1, 0, 1, 0, 1],
            "s": [0, 0, 0, 0, 1, 1, 1, 1],
            "tl": [0, 0, 0, 1, 1, 2, 2, 0.0],
        }
    )
    kept = df.dropna()
    m = CoxPH.fit_from_df(df, "x", Z_cols="z", tl_col="tl", strata_col="s")
    direct = CoxPH.fit(
        kept.x.values,
        kept[["z"]].values,
        tl=kept.tl.values,
        strata=kept.s.values,
        method="efron",
    )
    assert np.allclose(m.params, direct.params)


# -- 3. Polynomial KP / exact tie handling -----------------------------------


def _brute_kp_neg_ll(beta, x, Z, c, tl):
    """Kalbfleisch-Prentice by listing every d-subset of each risk set."""
    ll = 0.0
    eta = Z @ beta
    for tau in np.unique(x[c == 0]):
        deaths = np.flatnonzero((x == tau) & (c == 0))
        risk = np.flatnonzero((tl < tau) & (x >= tau))
        subsets = [
            eta[list(s)].sum()
            for s in itertools.combinations(risk, len(deaths))
        ]
        ll += eta[deaths].sum() - logsumexp(subsets)
    return -ll


def _brute_exact_neg_ll(beta, x, Z, c, tl):
    """The exact partial likelihood by summing over every ordering."""
    ll = 0.0
    a = np.exp(Z @ beta)
    for tau in np.unique(x[c == 0]):
        deaths = np.flatnonzero((x == tau) & (c == 0))
        risk = np.flatnonzero((tl < tau) & (x >= tau))
        total = 0.0
        for order in itertools.permutations(deaths):
            remaining, prob = a[risk].sum(), 1.0
            for j in order:
                prob *= a[j] / remaining
                remaining -= a[j]
            total += prob
        ll += np.log(total)
    return -ll


def _small_tied_data(seed):
    rng = np.random.default_rng(seed)
    n = 18
    Z = rng.normal(size=(n, 2))
    x = rng.integers(1, 6, size=n).astype(float)
    c = (rng.uniform(size=n) < 0.25).astype(int)
    tl = np.where(rng.uniform(size=n) < 0.3, rng.uniform(0, 2, n), -np.inf)
    tl = np.minimum(tl, x - 0.5)
    return x, Z, c, tl


@pytest.mark.parametrize(
    "method, brute",
    [("kp", _brute_kp_neg_ll), ("exact", _brute_exact_neg_ll)],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tie_likelihoods_match_brute_force(method, brute, seed):
    x, Z, c, tl = _small_tied_data(seed)
    xv, cv, nv, tlv, Zv = validate_coxph(x, c, None, Z, tl, method)
    neg_ll, jac_hess = CoxPH_()._resolve_func_generator(method)(
        xv, Zv, cv, nv, tlv
    )
    eps = 1e-5
    eye = np.eye(2)
    for beta in (np.zeros(2), np.array([0.4, -0.9]), np.array([1.5, 2.0])):
        ref = brute(beta, x, Z, c, tl)
        assert neg_ll(beta) == pytest.approx(ref, rel=1e-11)

        score, hess = jac_hess(beta)
        score_fd = np.array(
            [
                (brute(beta + eps * e, x, Z, c, tl))
                - brute(beta - eps * e, x, Z, c, tl)
                for e in eye
            ]
        ) / (2 * eps)
        assert np.allclose(score, score_fd, atol=1e-6)
        hess_fd = np.column_stack(
            [
                (jac_hess(beta + eps * e)[0] - jac_hess(beta - eps * e)[0])
                / (2 * eps)
                for e in eye
            ]
        )
        assert np.allclose(hess, hess_fd, atol=1e-5)


def _heavy_ties(seed=0):
    # 18 distinct times, 107 failures tied at one of them: the old KP took
    # over nine minutes on data like this.
    rng = np.random.default_rng(seed)
    x = np.concatenate(
        [np.full(107, 5.0), rng.integers(1, 19, size=150).astype(float)]
    )
    Z = rng.normal(size=(x.size, 2))
    c = (rng.uniform(size=x.size) < 0.2).astype(int)
    return x, Z, c


@pytest.mark.parametrize("method", ["kp", "exact"])
def test_heavy_ties_fit_quickly(method):
    x, Z, c = _heavy_ties()
    start = time.perf_counter()
    model = CoxPH.fit(x, Z, c=c, method=method)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
    assert np.all(np.isfinite(model.beta))
    assert np.all(np.isfinite(model.p_values))


@pytest.mark.parametrize("method", ["kp", "exact"])
def test_heavy_tie_fit_is_the_likelihood_maximum(method):
    # Score zero and a positive-definite information at the fitted beta.
    x, Z, c = _heavy_ties(1)
    model = CoxPH.fit(x, Z, c=c, method=method)
    score, hess = model.jac(model.beta)
    assert np.allclose(score, 0.0, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(hess) > 0)


def test_exact_every_unit_tied():
    # One tie set of 75 used to be refused (a cap of 12 ties) and, below
    # the cap, took tens of seconds.
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(80, 1))
    x = np.ones(80)
    c = np.zeros(80, dtype=int)
    c[:5] = 1  # five survivors keep the risk set larger than the tie set
    start = time.perf_counter()
    model = CoxPH.fit(x=x, Z=Z, c=c, method="exact")
    assert time.perf_counter() - start < 5.0
    assert np.isfinite(model.beta[0])


# -- 4./5. random() shapes ---------------------------------------------------


def _weibull_binary(n=100, seed=1, effect=-0.5):
    np.random.seed(seed)
    Z = np.random.binomial(1, 0.5, n).reshape(-1, 1)
    x = Weibull.random(n, 10, 2) * np.exp(effect * Z[:, 0])
    return x, Z


def test_ah_random_matches_ph_shapes():
    x, Z = _weibull_binary()
    ah = WeibullAH.fit(x, Z)
    ph = WeibullPH.fit(x, Z)
    for Zq in ([[0.0], [1.0]], [[1.0]], [0.0]):
        ah_x, ah_Z = ah.random(3, Zq)
        ph_x, ph_Z = ph.random(3, Zq)
        assert ah_x.shape == ph_x.shape
        assert ah_Z.shape == ph_Z.shape
        assert ah_Z.ndim == 2
    np.random.seed(0)
    draws, rows = ah.random(4, [[0.0], [1.0]])
    assert np.array_equal(rows[:, 0], [0, 0, 0, 0, 1, 1, 1, 1])
    assert np.all(draws > 0)


def _al_model():
    np.random.seed(0)
    Z = np.repeat([1.0, 2.0, 3.0], 30)
    x = Weibull.random(90, 10, 2) * Z**-1.0
    return AcceleratedLife(Weibull, Power).fit(x, Z)


def test_accelerated_life_random_size_per_stress():
    model = _al_model()
    # A pair of stresses gives size draws at each: 2 * size, never size**2
    # (the old uniform (low, high) option drew size stresses and then size
    # draws at each).
    for Zq in ((1.0, 2.0), [1.0, 2.0], [[1.0], [2.0]]):
        draws, rows = model.model.random(3, Zq, *model.params)
        assert draws.shape == (6,)
        assert rows.shape == (6, 1)
    draws, rows = model.random(3, 2.0)
    assert draws.shape == (3,)
    assert np.all(rows == 2.0)


# -- 6. non-finite start -----------------------------------------------------


def _tires():
    tires = load_tires_data()
    x = tires["Survival"].values
    c = tires["Censoring"].values
    Z = tires[
        [
            "Wedge gauge",
            "Interbelt gauge",
            "Peel force",
            "Wedge gauge×peel force",
        ]
    ].values
    return x, Z, c


def test_po_non_finite_init_falls_back_to_default_start():
    x, Z, c = _tires()
    ph = WeibullPH.fit(x=x, Z=Z, c=c)
    init = np.concatenate([ph.params[:2], -ph.params[2:]])
    with pytest.warns(UserWarning, match="not finite at the supplied"):
        po = PO(Weibull).fit(x=x, Z=Z, c=c, init=init)
    default = PO(Weibull).fit(x=x, Z=Z, c=c)
    # Previously the init came back unchanged, with neg_ll() == inf.
    assert not np.allclose(po.params, init)
    assert np.isfinite(po.neg_ll())
    assert po.neg_ll() == pytest.approx(default.neg_ll(), abs=1e-6)


def test_finite_start_raises_when_no_finite_start():
    def never_finite(p):
        return np.inf

    with pytest.raises(ValueError, match="not finite"):
        finite_start(never_finite, np.zeros(2), lambda: np.ones(2))
    with pytest.raises(ValueError, match="not finite"):
        finite_start(never_finite, np.zeros(2), None)


# -- 7. additive hazards positivity boundary ---------------------------------


def test_additive_hazards_warns_on_positivity_boundary():
    np.random.seed(0)
    Z = np.random.binomial(1, 0.5, 300).reshape(-1, 1).astype(float)
    x = Weibull.random(300, 10, 2) * np.exp(1.5 * Z[:, 0])
    with pytest.warns(UserWarning, match="positivity boundary"):
        WeibullAH.fit(x, Z)


def test_additive_hazards_interior_fit_does_not_warn():
    x, Z = _weibull_binary()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        WeibullAH.fit(x, Z)


# -- 8. k counts estimated parameters only -----------------------------------


def test_accelerated_life_k_excludes_placeholder():
    model = _al_model()
    assert len(model.params) == 4
    assert model.k == 3
    assert model.aic() == pytest.approx(2 * 3 + 2 * model.neg_ll())


def test_fixed_parameters_not_counted_in_k():
    x, Z = _weibull_binary()
    free = WeibullPH.fit(x, Z)
    fixed = WeibullPH.fit(x, Z, fixed={"beta": 2.0})
    assert free.k == 3
    assert fixed.k == 2
    n = len(x)
    assert fixed.aic() == pytest.approx(2 * 2 + 2 * fixed.neg_ll())
    assert fixed.aic_c() == pytest.approx(
        fixed.aic() + (2 * 2**2 + 2 * 2) / (n - 2 - 1)
    )


def test_restored_model_k_counts_estimated_parameters():
    model = _al_model()
    d = model.to_dict()
    d["k"] = 4  # a dict written by a version that counted the placeholder
    restored = ParametricRegressionModel.from_dict(d)
    assert restored.k == 3
    assert restored.aic() == pytest.approx(model.aic())


# -- 9. frailty --------------------------------------------------------------


def _frailty_free():
    rng = np.random.default_rng(2)
    z = rng.normal(size=300)
    x = 20 * (-np.log(rng.uniform(size=300)) / np.exp(0.8 * z)) ** (1 / 1.8)
    g = np.repeat(np.arange(30), 10)
    return x, z.reshape(-1, 1), g


def test_frailty_param_cb_theta_at_boundary_is_warning_free():
    x, Z, g = _frailty_free()
    model = WeibullFrailty.fit(x=x, Z=Z, groups=g)
    assert model.theta < 1e-8
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lower, upper = model.param_cb("theta")
        model.standard_errors()
    assert lower == 0.0
    assert upper == np.inf


def test_frailty_information_criteria_compare_with_ph():
    x, Z, g = _frailty_free()
    frailty = WeibullFrailty.fit(x=x, Z=Z, groups=g)
    ph = WeibullPH.fit(x=x, Z=Z)
    # theta -> 0 is the PH model, so the likelihoods agree and the frailty
    # model pays for one more parameter.
    assert frailty.neg_ll() == pytest.approx(ph.neg_ll(), abs=1e-6)
    assert frailty.aic() == pytest.approx(ph.aic() + 2, abs=1e-5)
    assert frailty.bic() == pytest.approx(ph.bic() + np.log(300), abs=1e-5)
    restored = type(frailty).from_dict(frailty.to_dict())
    assert restored.aic() == pytest.approx(frailty.aic())
    assert restored.bic() == pytest.approx(frailty.bic())
