"""Fine-Gray subdistribution-hazard regression.

The model targets the cumulative incidence of a cause directly:
``F_k(t|Z) = 1 - exp(-Lambda0(t) exp(beta'Z))``. Independent right-censoring
is handled by inverse-probability-of-censoring weighting (IPCW), so these
tests exercise parameter recovery *under censoring* -- the regime where a
naive (unweighted) subdistribution risk set would be biased.
"""

import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from surpyval.tests._helpers import competing_risks_regression_data
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
    FineGray,
)
from surpyval.univariate.competing_risks.regression import fine_gray


def _simulate_fine_gray(N, seed, beta=(0.7, -0.4), p=0.5, cens_scale=3.0):
    """
    Simulate two-cause competing-risks data whose cause-1 cumulative incidence
    follows a Fine-Gray model with coefficients ``beta`` (Fine & Gray 1999
    construction): ``F_1(t|Z) = 1 - (1 - p(1 - e^{-t}))^{exp(beta'Z)}``.
    Cause 2 mops up the rest; exponential right-censoring is applied.
    """
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=float)
    Z = rng.uniform(-1, 1, size=(N, beta.size))
    phi = np.exp(Z @ beta)

    p1 = 1 - (1 - p) ** phi  # P(cause = 1 | Z)
    is1 = rng.uniform(size=N) < p1

    x = np.empty(N)
    e = np.empty(N, dtype=object)

    # Invert the conditional cause-1 CIF for the cause-1 event times.
    v = rng.uniform(size=N)
    w = 1 - (1 - v * p1) ** (1 / phi)  # = p (1 - e^{-t})
    t1 = -np.log(np.clip(1 - w / p, 1e-12, 1.0))
    x[is1] = t1[is1]
    e[is1] = 1

    # Cause-2 times are exponential.
    t2 = rng.exponential(1.0, size=N)
    x[~is1] = t2[~is1]
    e[~is1] = 2

    # Independent right-censoring.
    cens = rng.exponential(cens_scale, size=N)
    c = (x > cens).astype(int)
    x = np.minimum(x, cens)
    e[c == 1] = None
    return x, Z, e, c


# --- standalone FineGray --------------------------------------------------


def test_recovers_parameters_under_censoring():
    x, Z, e, c = _simulate_fine_gray(8000, 0)
    m = FineGray.fit(x, Z, e, c=c, event=1)
    assert np.allclose(m.beta, [0.7, -0.4], atol=0.07)
    # Roughly a quarter of observations are censored in this design.
    assert 0.15 < c.mean() < 0.35


def test_baseline_cif_matches_theory_at_Z_zero():
    # At Z = 0 the model reduces to the baseline CIF p(1 - e^{-t}).
    x, Z, e, c = _simulate_fine_gray(8000, 1)
    m = FineGray.fit(x, Z, e, c=c, event=1)
    t = np.array([0.5, 1.0, 2.0])
    expected = 0.5 * (1 - np.exp(-t))
    assert np.allclose(m.cif(t, [0.0, 0.0]), expected, atol=0.03)


def test_cif_is_monotone_and_bounded():
    x, Z, e, c = _simulate_fine_gray(4000, 2)
    m = FineGray.fit(x, Z, e, c=c, event=1)
    t = np.linspace(0.0, 5.0, 50)
    cif = m.cif(t, [0.3, -0.2])
    assert np.all(cif >= 0) and np.all(cif <= 1)
    assert np.all(np.diff(cif) >= -1e-12)


def test_sf_and_hf_identities():
    x, Z, e, c = _simulate_fine_gray(4000, 3)
    m = FineGray.fit(x, Z, e, c=c, event=1)
    t = np.array([0.5, 1.0, 2.0])
    Z0 = [0.1, -0.1]
    assert np.allclose(m.sf(t, Z0), 1 - m.cif(t, Z0))


def test_positive_coefficient_is_significant():
    # The strong cause-1 covariate should be clearly significant; all p-values
    # are well-defined probabilities.
    x, Z, e, c = _simulate_fine_gray(6000, 4)
    m = FineGray.fit(x, Z, e, c=c, event=1)
    assert np.all(np.isfinite(m.p_values))
    assert np.all((m.p_values >= 0) & (m.p_values <= 1))
    assert m.p_values[0] < 0.01  # coef_0 = 0.7


def test_counts_equivalent_to_repeated_rows():
    x, Z, e, c = _simulate_fine_gray(1500, 5)
    m_rep = FineGray.fit(
        np.repeat(x, 2),
        np.repeat(Z, 2, axis=0),
        np.repeat(e, 2),
        c=np.repeat(c, 2),
        event=1,
    )
    m_cnt = FineGray.fit(x, Z, e, c=c, n=np.full(x.size, 2), event=1)
    assert np.allclose(m_rep.beta, m_cnt.beta, atol=1e-4)


def test_ipcw_matters_versus_naive_no_censoring():
    # With no censoring the IPCW weights are all 1, so the fit is the plain
    # subdistribution model and still recovers the truth.
    x, Z, e, c = _simulate_fine_gray(8000, 6, cens_scale=1e6)
    assert c.mean() == 0.0
    m = FineGray.fit(x, Z, e, c=c, event=1)
    assert np.allclose(m.beta, [0.7, -0.4], atol=0.07)


# --- input handling -------------------------------------------------------


def test_cause_required_when_multiple_event_types():
    x, Z, e, c = _simulate_fine_gray(500, 7)
    with pytest.raises(ValueError, match="pass `event`"):
        FineGray.fit(x, Z, e, c=c)


def test_unknown_cause_rejected():
    x, Z, e, c = _simulate_fine_gray(500, 8)
    with pytest.raises(ValueError, match="Unknown cause 99"):
        FineGray.fit(x, Z, e, c=c, event=99)


# --- CompetingRisksProportionalHazards integration -----------------------


def test_crph_fine_gray_matches_standalone():
    x, Z, e, c = _simulate_fine_gray(5000, 9)
    crph = CompetingRisksProportionalHazards.fit(
        x, Z, e, c=c, model="Fine-Gray"
    )
    standalone = FineGray.fit(x, Z, e, c=c, event=1)
    i1 = crph.event_idx_map[1]
    assert np.allclose(crph.betas[i1], standalone.beta, atol=1e-6)
    t = np.array([0.5, 1.0, 2.0])
    assert np.allclose(
        crph.cif(t, [0.2, -0.1], 1), standalone.cif(t, [0.2, -0.1])
    )


def test_crph_fine_gray_cif_identities():
    x, Z, e, c = _simulate_fine_gray(4000, 10)
    crph = CompetingRisksProportionalHazards.fit(
        x, Z, e, c=c, model="Fine-Gray"
    )
    t = np.array([0.5, 1.0, 2.0])
    Z0 = [0.1, 0.1]
    assert np.allclose(crph.sf(t, Z0, 1) + crph.cif(t, Z0, 1), 1.0)
    assert np.allclose(crph.Hf(t, Z0, 1), -np.log(crph.sf(t, Z0, 1)))


def test_crph_fine_gray_hf_df_raise():
    x, Z, e, c = _simulate_fine_gray(2000, 11)
    crph = CompetingRisksProportionalHazards.fit(
        x, Z, e, c=c, model="Fine-Gray"
    )
    with pytest.raises(ValueError, match="no pointwise"):
        crph.hf([1.0], [0.0, 0.0], 1)
    with pytest.raises(ValueError, match="no pointwise"):
        crph.df([1.0], [0.0, 0.0], 1)


def test_crph_fine_gray_requires_event():
    x, Z, e, c = _simulate_fine_gray(2000, 12)
    crph = CompetingRisksProportionalHazards.fit(
        x, Z, e, c=c, model="Fine-Gray"
    )
    with pytest.raises(ValueError, match="pass `event`"):
        crph.sf([1.0], [0.0, 0.0])


def test_crph_cox_path_still_runs():
    # Regression guard: the cause-specific Cox path (a sibling of the same
    # public entry point) fits and predicts.
    x, Z, e, c = _simulate_fine_gray(3000, 13)
    crph = CompetingRisksProportionalHazards.fit(x, Z, e, c=c, model="Cox")
    cif = crph.cif(np.array([0.5, 1.0, 2.0]), [0.1, -0.1], 1)
    assert np.all(np.isfinite(cif))


# ---------------------------------------------------------------------------
# ``cif`` pairs covariate rows with the times.
# ---------------------------------------------------------------------------


def test_fine_gray_cif_pairs_covariate_rows():
    x, Z, e = competing_risks_regression_data()
    fg = FineGray.fit(x, Z, e, event=1)
    paired = fg.cif([1.0, 2.0], [[0, 0], [1, 1]])
    np.testing.assert_allclose(
        paired, [fg.cif([1.0], [0, 0])[0], fg.cif([2.0], [1, 1])[0]]
    )


# cmprsk 2.2-11, crr(x, code, cbind(z1, z2), failcode = k) on
# competing_risks_regression_data(): its loglik, the log pseudo-likelihood
# at the fit, and the events of cause k.
CRR_LOGLIK = {1: (-241.875318320354808, 61), 2: (-123.918829886760818, 30)}


@pytest.mark.parametrize("event", sorted(CRR_LOGLIK))
def test_604_fine_gray_log_likelihood_is_crrs(event):
    import surpyval as sp

    x, Z, e = competing_risks_regression_data()
    model = FineGray.fit(x, Z, e, event=event)
    loglik, events = CRR_LOGLIK[event]
    assert isinstance(model.log_likelihood, float)
    assert model.log_likelihood == pytest.approx(loglik, rel=1e-9)
    assert model.neg_ll() == -model.log_likelihood
    # k the coefficients; BIC's n the events of the cause of interest
    assert model.aic() == pytest.approx(4 - 2 * loglik, rel=1e-9)
    assert model.bic() == pytest.approx(
        2 * np.log(events) - 2 * loglik, rel=1e-9
    )
    restored = sp.from_dict(model.to_dict())
    assert restored.bic() == model.bic()
    assert restored.aic_c() == model.aic_c()


def test_604_fine_gray_competing_risks_model_has_no_likelihood():
    x, Z, e = competing_risks_regression_data()
    model = CompetingRisksProportionalHazards.fit(x, Z, e, model="Fine-Gray")
    for name in ("neg_ll", "aic", "aic_c", "bic"):
        with pytest.raises(ValueError, match="FineGray model"):
            getattr(model, name)()


@pytest.mark.parametrize("scale", [1e4, 1e6])
def test_606_large_scale_covariates_reach_the_same_maximum(scale):
    # exp(beta'Z) overflowed at the first BFGS step on covariates of order
    # 1e4, and the fit raised "SVD did not converge" in safe_inv. In the
    # covariates' units the fit is the same model: coefficients divided by
    # the scale, the same likelihood and incidence.
    x, Z, e = competing_risks_regression_data()
    model = FineGray.fit(x, Z, e, event=1)
    big = FineGray.fit(x, Z * scale, e, event=1)
    assert big.maximum == "verified"
    np.testing.assert_allclose(big.beta * scale, model.beta, rtol=1e-8)
    np.testing.assert_allclose(
        big.standard_errors() * scale, model.standard_errors(), rtol=1e-6
    )
    assert big.neg_ll() == pytest.approx(model.neg_ll(), rel=1e-12)
    np.testing.assert_allclose(
        big.cif([1.0, 5.0], Z[0] * scale), model.cif([1.0, 5.0], Z[0])
    )
    both = CompetingRisksProportionalHazards.fit(
        x, Z * scale, e, model="Fine-Gray"
    )
    np.testing.assert_allclose(both.betas[0] * scale, model.beta, rtol=1e-8)


def test_605_fine_gray_covariance_is_a_method_and_cov_gone():
    import surpyval as sp

    x, Z, e = competing_risks_regression_data()
    model = FineGray.fit(x, Z, e, event=1)
    cov = model.covariance()
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), model.standard_errors())
    # ``cov`` and ``se``, deprecated in v0.23, are gone
    assert not hasattr(model, "cov") and not hasattr(model, "se")
    d = model.to_dict()
    assert "covariance" in d and "cov" not in d and "_neg_ll" in d
    # A dict written before v0.23
    d["cov"], d["neg_ll"] = d.pop("covariance"), d.pop("_neg_ll")
    restored = sp.from_dict(d)
    np.testing.assert_array_equal(restored.covariance(), cov)
    assert restored.neg_ll() == model.neg_ll()


def test_656_fine_gray_names_params_and_summary():
    rng = np.random.default_rng(0)
    N = 200
    Z = np.c_[rng.binomial(1, 0.5, N), rng.normal(size=N)]
    ta = rng.exponential(1 / (0.1 * np.exp(0.7 * Z[:, 0])))
    tb = rng.exponential(1 / 0.05, N)
    x = np.minimum(ta, tb).round(2)
    e = np.where(ta < tb, "a", "b")
    frame = FineGray.fit(
        x, pd.DataFrame(Z, columns=["grp", "age"]), e, event="a"
    )
    assert frame.parameter_names == ["grp", "age"]
    np.testing.assert_array_equal(frame.params, frame.beta)
    table = frame.summary(alpha_ci=0.1)
    assert table.index.tolist() == ["grp", "age"]
    np.testing.assert_allclose(table["se(coef)"], frame.standard_errors())
    np.testing.assert_allclose(table["p"], frame.p_values)
    assert "coef lower 90%" in table.columns
    # An array Z names them coef_j, as the other regression models (#614)
    array = FineGray.fit(x, Z, e, event="a")
    assert array.parameter_names == ["coef_0", "coef_1"]


# Events of interest (cause "a") that 0.25 z0 - z1 separates from the rest
# of their subdistribution risk sets -- the competing failures, at risk to
# the end, and the censored row -- with neither column alone (#746)
_COMBINATION = (
    np.array([2, 1, 1.5, 0.5, 0.5, 2, 0.3, 0.3, 2.5]),
    np.array(
        [
            [0, 1.5],
            [0.5, 1],
            [-1.5, 1],
            [2, -1],
            [0, -1.5],
            [0, 1.5],
            [0, 2.0],
            [-2, 1.5],
            [0, 1.6],
        ]
    ),
    np.array(["a"] * 6 + ["b", "b", None], dtype=object),
)


@pytest.mark.parametrize("order", [np.arange(9), np.arange(9)[::-1]])
def test_746_fine_gray_finds_a_run_off_along_a_combination(order):
    # The judge of each coefficient's own profile saw none, and the fit
    # raised "SVD did not converge" on its nan Hessian; the data decide,
    # with CoxPH's exact test on the subdistribution risk sets
    x, Z, e = (a[order] for a in _COMBINATION)
    with pytest.warns(UserWarning, match="proportion 0.25 : -1") as w:
        model = FineGray.fit(x, Z, e, event="a", center=True)
    assert len(w) == 1
    assert model.maximum == "no finite maximum"
    assert np.isnan(model.standard_errors()).all()


def test_746_fine_gray_competing_failures_stay_at_risk():
    # A competing failure at 0.3 with 0.25 z0 - z1 above the events at 1,
    # 1.5 and 2: it stays in their subdistribution risk sets (it would not
    # in a cause-specific one), so the events are not separated and the
    # maximum is finite.
    x, Z, e = _COMBINATION
    Z = Z.copy()
    Z[6] = [0.0, -2.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = FineGray.fit(x, Z, e, event="a", center=True)
    assert model.maximum == "verified"


def test_746_an_ordinary_fine_gray_fit_does_not_look_for_a_run_off():
    # The data are asked only when the search gives cause, as for CoxPH
    x, Z, e, c = _simulate_fine_gray(400, 1)
    with mock.patch.object(
        fine_gray, "runoff_direction", side_effect=AssertionError
    ):
        model = FineGray.fit(x, Z, e, c=c, event=1)
    assert model.maximum == "verified"


def test_746_every_coefficient_aliased_is_fitted():
    # A constant column alone raised "need at least one array to stack"
    # (autograd's Hessian in no coefficients); it is aliased, and nan
    x = np.arange(1.0, 9)
    e = np.array(["a", "b", "a", "a", "b", "a", None, "a"], dtype=object)
    with pytest.warns(UserWarning, match="cannot be estimated"):
        model = FineGray.fit(x, np.ones((8, 1)), e, event="a")
    assert np.isnan(model.beta).all()
    assert np.isnan(model.standard_errors()).all()


def test_760_objective_far_along_a_run_off_is_quiet_and_right():
    # Far out along the run-off direction every later risk set's sum
    # underflowed, and the objective was log(0): -inf, the best point any
    # search could find. The sums are taken in logs there (#760): the
    # objective falls to its limit, and its derivatives are finite.
    from autograd import grad, hessian

    x, Z, e = _COMBINATION
    with pytest.warns(UserWarning, match="proportion 0.25 : -1"):
        model = FineGray.fit(x, Z, e, event="a", center=True)
    objective = model._objective
    d = np.array([0.25, -1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        values = [objective(t * d) for t in [10.0, 100.0, 1e3, 1e4, 1e6]]
        # (to the rounding of a linear predictor of 1e6, 1e-10)
        assert np.all(np.diff(values) <= 1e-8)
        np.testing.assert_allclose(values[2:], values[-1], rtol=0, atol=1e-8)
        # Off the direction it rises, without bound
        assert objective(np.array([1e4, 0.0])) > 3e4
        g, h = grad(objective)(1e4 * d), hessian(objective)(1e4 * d)
    assert np.all(np.isfinite(g)) and np.all(np.isfinite(h))
    # The run-off direction is flat; the other is not
    np.testing.assert_allclose(h @ d, 0.0, atol=1e-9)
    assert np.linalg.eigvalsh(h).max() > 1


def test_760_log_risk_set_sums_are_the_sums():
    # The sums in logs agree with the direct ones where both hold
    x, Z, e, c = _simulate_fine_gray(300, 2)
    model = FineGray.fit(x, Z, e, c=c, event=1)
    objective = model._objective
    sets, Zk = objective.args[1], objective.args[3]
    n_sorted = objective.args[0]
    eta = Zk @ np.array([0.7, -0.4])
    direct = np.log(fine_gray._risk_set_sums(n_sorted * np.exp(eta), sets))
    in_logs = fine_gray._log_risk_set_sums(eta, n_sorted, sets)
    np.testing.assert_allclose(in_logs, direct, rtol=1e-13, atol=1e-13)


def test_760_run_off_baseline_is_quiet():
    # The baseline's exp(beta'Z) overflowed at the run-off coefficients,
    # with numpy's warnings (#760)
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(40, 2))
    Z = Z[np.argsort(-(Z @ np.array([1.0, -0.6])))]
    e = np.array(["a"] * 15 + ["b"] * 10 + [None] * 15, dtype=object)
    with pytest.warns(UserWarning) as record:
        model = FineGray.fit(np.arange(1.0, 41), Z, e, event="a", center=True)
    assert [w.category for w in record] == [UserWarning]
    assert model.maximum == "no finite maximum"
    H = model._cumhaz
    assert np.all(np.isfinite(H)) and np.all(np.diff(H) >= 0)
