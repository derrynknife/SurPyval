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


# -- Fine and Gray's robust (sandwich) variance ---------------------------


class _ByDefinition:
    """Fine and Gray's (1999) quantities at ``beta`` by their definitions,
    each a loop over the units (as ``cmprsk::crr``'s Fortran takes them),
    in the numbers of ``arithmetic``. ``status`` is 0 censored, 1 the
    cause of interest, 2 a competing failure; ``exp(beta'Z)`` is taken
    with ``arithmetic`` only at ``beta = 0``. Sampling weights ``w`` weight
    every sum over the units (the risk sets, the events, the censoring
    Kaplan-Meier and its at-risk counts), and a unit's influence is its
    weight times the unweighted one's."""

    def __init__(self, x, Z, status, beta, arithmetic=float, w=None):
        self.x, self.Z, self.status = x, Z, status
        self.N, self.p = len(x), len(Z[0])
        self.number = arithmetic
        self.one, self.zero = arithmetic(1), arithmetic(0)
        self.w = [self.one] * self.N if w is None else list(w)
        if any(beta):
            self.risk = [float(np.exp(np.dot(row, beta))) for row in Z]
        else:
            self.risk = [self.one] * self.N
        self.units = range(self.N)
        # (values already worked out, kept: the loops ask for each often)
        self._G, self._weight, self._q = {}, {}, {}
        self.events = [i for i in self.units if status[i] == 1]
        self.censoring_times = sorted(
            {x[i] for i in self.units if status[i] == 0}
        )
        self.S0 = {i: self._sum(x[i]) for i in self.events}
        self.mean = {
            i: [self._sum(x[i], k) / self.S0[i] for k in range(self.p)]
            for i in self.events
        }

    def at_risk(self, u):
        return sum(
            (self.w[j] for j in self.units if self.x[j] >= u), self.zero
        )

    def censored_at(self, u):
        return sum(
            (
                self.w[j]
                for j in self.units
                if self.x[j] == u and self.status[j] == 0
            ),
            self.zero,
        )

    def G_before(self, t):
        # The censoring Kaplan-Meier just before t, everyone with x >= u
        # at risk at a censoring time u
        if t not in self._G:
            out = self.one
            for u in self.censoring_times:
                if u < t:
                    out *= self.one - self.censored_at(u) / self.at_risk(u)
            self._G[t] = out
        return self._G[t]

    def weight(self, j, t):
        # Unit j's weight in the subdistribution risk set at t
        if (j, t) not in self._weight:
            if self.x[j] >= t:
                out = self.one
            elif self.status[j] == 2:
                out = self.G_before(t) / self.G_before(self.x[j])
            else:
                out = self.zero
            self._weight[j, t] = out
        return self._weight[j, t]

    def _sum(self, t, k=None):
        # The risk set's sum of exp(beta'Z) at t, times Z_k where given
        return sum(
            self.w[j]
            * self.weight(j, t)
            * self.risk[j]
            * (self.one if k is None else self.Z[j][k])
            for j in self.units
        )

    def spread(self, j, i, k):
        # w_j(x_i) exp(beta'Z_j) (Z_jk - Zbar_k(x_i)) / S0(x_i)
        return (
            self.weight(j, self.x[i])
            * self.risk[j]
            * (self.Z[j][k] - self.mean[i][k])
            / self.S0[i]
        )

    def information(self):
        return [
            [
                sum(
                    self.w[i]
                    * self.w[j]
                    * self.spread(j, i, a)
                    * (self.Z[j][b] - self.mean[i][b])
                    for i in self.events
                    for j in self.units
                )
                for b in range(self.p)
            ]
            for a in range(self.p)
        ]

    def q(self, u, k):
        if (u, k) not in self._q:
            self._q[u, k] = sum(
                (
                    self.w[i] * self.w[j] * self.spread(j, i, k)
                    for i in self.events
                    if self.x[i] >= u
                    for j in self.units
                    if self.x[j] < u and self.status[j] == 2
                ),
                self.zero,
            )
        return self._q[u, k]

    def eta(self, i, k):
        out = -sum(
            (self.w[j] * self.spread(i, j, k) for j in self.events),
            self.zero,
        )
        if self.status[i] == 1:
            out += self.Z[i][k] - self.mean[i][k]
        return out

    def psi(self, i, k):
        out = self.zero
        for u in self.censoring_times:
            if u <= self.x[i]:
                jump = self.status[i] == 0 and self.x[i] == u
                out += self.q(u, k) * (
                    (self.one if jump else self.zero) / self.at_risk(u)
                    - self.censored_at(u) / self.at_risk(u) ** 2
                )
        return out

    def meat(self):
        unit = [
            [self.eta(i, k) + self.psi(i, k) for k in range(self.p)]
            for i in self.units
        ]
        return [
            [
                sum(wi**2 * u[a] * u[b] for wi, u in zip(self.w, unit))
                for b in range(self.p)
            ]
            for a in range(self.p)
        ]


def _fine_gray_meat(x, Z, status, beta, arithmetic=float, w=None):
    """The information ``A`` and Fine and Gray's ``sum_i w_i^2 (eta_i +
    psi_i)(eta_i + psi_i)'`` at ``beta``, by the definitions."""
    parts = _ByDefinition(x, Z, status, beta, arithmetic, w)
    return parts.information(), parts.meat()


def _sandwich_parts(x, Z, status, beta, n=None, w=None):
    """The fit's own information and meat at ``beta``: the model's
    ``_robust_covariance`` with an identity bread is the meat."""
    x = np.asarray(x, dtype=float)
    Z = np.asarray(Z, dtype=float)
    status = np.asarray(status)
    n = np.ones(x.size) if n is None else np.asarray(n, dtype=float)
    nw = n if w is None else n * np.asarray(w, dtype=float)
    g_times, g_vals = fine_gray.censoring_survival(x, status == 0, nw)
    G_x = fine_gray.step_left_limit(g_times, g_vals, x, before=1.0)
    sets = fine_gray._risk_sets(
        x, nw, status == 1, status == 2, G_x, g_times, g_vals
    )
    order = sets.order
    return fine_gray._robust_covariance(
        np.eye(Z.shape[1]),
        Z[order],
        np.asarray(beta, dtype=float),
        sets,
        n[order],
        None if w is None else np.asarray(w, dtype=float)[order],
        (status == 1)[order],
        (status == 0)[order],
        x[order],
    )


def test_robust_meat_by_hand_in_exact_arithmetic():
    """Four units and one covariate at ``beta = 0``, worked by hand.

    A competing failure at 1 (z = 1), a censoring at 2 (z = 0), and
    events of interest at 3 (z = 0) and 4 (z = 1). The censoring
    Kaplan-Meier drops to 2/3 at 2 (one of three at risk), so the
    competing failure stays in the later risk sets with weight 2/3.

    At 3 the risk set is the two events (weight 1) and the competing
    failure (2/3): ``S0 = 8/3``, mean 5/8. At 4 it is the last event and
    the competing failure, both z = 1: mean 1, no spread. The information
    is ``(25/64 + 9/64 + (2/3) 9/64) / (8/3) = 15/64``.

    Score residuals: the competing failure ``-(2/3)(3/8)/(8/3) = -3/32``;
    the censored unit 0; the event at 3 ``-5/8 + (5/8)/(8/3) = -25/64``;
    the event at 4 ``0 - (3/8)/(8/3) = -9/64``.

    The censoring term: ``q(2)`` is the competing failure's share of the
    score at the later events, ``(2/3)(3/8)/(8/3) = 3/32``, and each unit
    at risk at 2 adds ``q (dN/3 - 1/9)``: the censored unit ``+1/48``,
    the two events ``-1/96`` each (they sum to zero).

    The meat is then ``(18^2 + 4^2 + 77^2 + 29^2) / 192^2 = 395/2048``.
    """
    from fractions import Fraction

    x = [1, 2, 3, 4]
    status = [2, 0, 1, 1]
    z = [1, 0, 0, 1]
    info, meat = _fine_gray_meat(
        x, [[Fraction(v)] for v in z], status, [0], Fraction
    )
    assert info[0][0] == Fraction(15, 64)
    assert meat[0][0] == Fraction(395, 2048)
    got = _sandwich_parts(x, np.array(z, dtype=float)[:, None], status, [0.0])
    np.testing.assert_allclose(got, [[395 / 2048]], rtol=1e-13)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_robust_covariance_is_the_definition_on_tied_data(seed):
    """The O(N) sums against the definitions (loops over the units) on 60
    units with two covariates and times on a grid of 12, at a ``beta``
    away from 0."""
    rng = np.random.default_rng(seed)
    N = 60
    x = rng.integers(1, 13, N).astype(float)
    status = rng.choice([0, 1, 2], N, p=[0.35, 0.4, 0.25])
    Z = np.column_stack([rng.binomial(1, 0.5, N), rng.normal(size=N)])
    beta = np.array([0.4, -0.3])
    info, meat = _fine_gray_meat(
        list(x), [list(row) for row in Z], list(status), beta
    )
    got = _sandwich_parts(x, Z, status, beta)
    np.testing.assert_allclose(got, np.array(meat, dtype=float), rtol=1e-10)
    # and the bread: the model-based covariance is the inverse of the
    # definition's information at the fitted coefficients
    e = np.where(status == 0, None, status.astype(object))
    model = FineGray.fit(x, Z, e, event=1)
    info, meat = _fine_gray_meat(
        list(x), [list(row) for row in Z], list(status), model.beta
    )
    bread = np.linalg.inv(np.array(info, dtype=float))
    np.testing.assert_allclose(
        model.covariance(robust=False), bread, rtol=1e-7
    )
    np.testing.assert_allclose(
        model.covariance(),
        bread @ np.array(meat, dtype=float) @ bread,
        rtol=1e-7,
    )


def test_robust_covariance_counts_equal_repeated_rows():
    rng = np.random.default_rng(7)
    N = 80
    x = rng.integers(1, 10, N).astype(float)
    status = rng.choice([0, 1, 2], N, p=[0.3, 0.45, 0.25])
    Z = rng.normal(size=(N, 2))
    n = rng.integers(1, 4, N)
    e = np.where(status == 0, None, status.astype(object))
    counted = FineGray.fit(x, Z, e, n=n, event=1)
    repeated = FineGray.fit(
        np.repeat(x, n), np.repeat(Z, n, axis=0), np.repeat(e, n), event=1
    )
    for robust in (True, False):
        np.testing.assert_allclose(
            counted.covariance(robust), repeated.covariance(robust), rtol=1e-8
        )


@pytest.mark.parametrize("cens_scale", [1e9, 1.0], ids=["none", "heavy"])
def test_unit_weights_give_the_unweighted_robust_covariance(cens_scale):
    """``w = 1`` is the unweighted fit, its robust covariance included:
    the censoring term is carried into weighted fits, so no censoring and
    heavy censoring alike give the same matrix."""
    x, Z, e, c = _simulate_fine_gray(300, 3, cens_scale=cens_scale)
    assert c.any() == (cens_scale == 1.0)
    model = FineGray.fit(x, Z, e, event=1)
    weighted = FineGray.fit(x, Z, e, event=1, w=1.0)
    np.testing.assert_allclose(weighted.beta, model.beta, rtol=1e-12)
    np.testing.assert_allclose(
        weighted.covariance(), model.covariance(), rtol=1e-12, atol=0
    )
    ones = FineGray.fit(x, Z, e, event=1, w=np.ones(x.size))
    np.testing.assert_allclose(
        ones.covariance(), model.covariance(), rtol=1e-12, atol=0
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_weighted_robust_covariance_is_the_definition_on_tied_data(seed):
    """With sampling weights: the O(N) sums against the definitions,
    every sum over the units weighted (the censoring Kaplan-Meier and its
    at-risk counts too) and each unit's influence its weight times the
    unweighted one, on tied data at a ``beta`` away from 0, and the
    fitted model's covariance against the definition's sandwich."""
    rng = np.random.default_rng(10 + seed)
    N = 50
    x = rng.integers(1, 11, N).astype(float)
    status = rng.choice([0, 1, 2], N, p=[0.35, 0.4, 0.25])
    Z = np.column_stack([rng.binomial(1, 0.5, N), rng.normal(size=N)])
    w = rng.uniform(0.3, 3.0, N)
    beta = np.array([0.4, -0.3])
    info, meat = _fine_gray_meat(
        list(x), [list(row) for row in Z], list(status), beta, w=list(w)
    )
    got = _sandwich_parts(x, Z, status, beta, w=w)
    np.testing.assert_allclose(got, np.array(meat, dtype=float), rtol=1e-10)
    e = np.where(status == 0, None, status.astype(object))
    model = FineGray.fit(x, Z, e, event=1, w=w)
    info, meat = _fine_gray_meat(
        list(x),
        [list(row) for row in Z],
        list(status),
        model.beta,
        w=list(w),
    )
    bread = np.linalg.inv(np.array(info, dtype=float))
    np.testing.assert_allclose(
        model.covariance(),
        bread @ np.array(meat, dtype=float) @ bread,
        rtol=1e-7,
    )


def test_robust_is_the_default_everywhere_and_model_based_an_option():
    from scipy.stats import norm

    x, Z, e, c = _simulate_fine_gray(400, 11, cens_scale=1.0)
    model = FineGray.fit(x, Z, e, event=1)
    robust, plain = model.covariance(), model.covariance(robust=False)
    assert not np.allclose(robust, plain, rtol=1e-3)
    np.testing.assert_array_equal(model.covariance(robust=True), robust)
    np.testing.assert_allclose(
        model.standard_errors(), np.sqrt(np.diag(robust))
    )
    np.testing.assert_allclose(
        model.standard_errors(robust=False), np.sqrt(np.diag(plain))
    )
    for is_robust, cov in ((True, robust), (False, plain)):
        table = model.summary(robust=is_robust)
        se = np.sqrt(np.diag(cov))
        np.testing.assert_allclose(table["se(coef)"], se)
        np.testing.assert_allclose(
            table["p"], 2 * norm.sf(np.abs(model.beta / se))
        )
        np.testing.assert_allclose(
            table["coef lower 95%"], model.beta - norm.ppf(0.975) * se
        )
    np.testing.assert_array_equal(
        model.summary().to_numpy(), model.summary(robust=True).to_numpy()
    )
    np.testing.assert_allclose(model.p_values, model.summary()["p"])
    assert "robust (Fine and Gray's sandwich)" in repr(model)
    # The competing-risks model's Fine-Gray path: each cause's block
    crph = CompetingRisksProportionalHazards.fit(x, Z, e, model="Fine-Gray")
    np.testing.assert_allclose(crph.covariance()[:2, :2], robust, rtol=1e-12)
    np.testing.assert_allclose(
        crph.models[1].covariance(robust=False), plain, rtol=1e-12
    )
    np.testing.assert_allclose(
        crph.summary()["se(coef)"].to_numpy()[:2], np.sqrt(np.diag(robust))
    )


def test_both_covariances_are_saved_and_restored():
    import json

    x, Z, e, c = _simulate_fine_gray(200, 5, cens_scale=1.0)
    model = FineGray.fit(x, Z, e, event=1)
    restored = fine_gray.FineGrayModel.from_dict(
        json.loads(json.dumps(model.to_dict()))
    )
    for robust in (True, False):
        np.testing.assert_array_equal(
            restored.covariance(robust), model.covariance(robust)
        )
        np.testing.assert_array_equal(
            restored.summary(robust=robust).to_numpy(),
            model.summary(robust=robust).to_numpy(),
        )
    np.testing.assert_array_equal(restored.p_values, model.p_values)
    assert repr(restored) == repr(model)


def test_a_dict_saved_before_the_robust_covariance_keeps_its_meaning():
    """A dict written before v0.27 holds the model-based covariance under
    ``"covariance"``: it loads as that, and the robust one, which needs
    the data, is refused with what to do."""
    x, Z, e, c = _simulate_fine_gray(200, 5, cens_scale=1.0)
    model = FineGray.fit(x, Z, e, event=1)
    old = model.to_dict()
    old["covariance"] = old.pop("model_covariance")
    old["se"] = model.standard_errors(robust=False).tolist()
    old["p_values"] = model.summary(robust=False)["p"].tolist()
    restored = fine_gray.FineGrayModel.from_dict(old)
    np.testing.assert_allclose(
        restored.covariance(robust=False), model.covariance(robust=False)
    )
    np.testing.assert_allclose(
        restored.summary(robust=False).to_numpy(),
        model.summary(robust=False).to_numpy(),
    )
    for call in (
        restored.covariance,
        restored.standard_errors,
        restored.summary,
    ):
        with pytest.raises(ValueError, match="saved before v0.27.*robust"):
            call()
    assert np.isnan(restored.p_values).all()
    assert "model-based (saved before v0.27" in repr(restored)
    np.testing.assert_allclose(
        restored.cif([0.5, 1.0], [0.3, -0.2]),
        model.cif([0.5, 1.0], [0.3, -0.2]),
    )
    # and the competing-risks model built on such dicts
    crph = CompetingRisksProportionalHazards.fit(x, Z, e, model="Fine-Gray")
    old = crph.to_dict()
    for fg in old["fg_models"]:
        fg["covariance"] = fg.pop("model_covariance")
    restored = CompetingRisksProportionalHazards.from_dict(old)
    with pytest.raises(ValueError, match="saved before v0.27"):
        restored.covariance()
    assert "Fine-Gray" in repr(restored)


def test_a_weighted_fit_has_no_model_based_covariance():
    from surpyval.utils.weights import WeightedRefusal

    x, Z, e, c = _simulate_fine_gray(200, 5, cens_scale=1.0)
    rng = np.random.default_rng(0)
    model = FineGray.fit(x, Z, e, event=1, w=rng.uniform(0.5, 2.0, x.size))
    assert np.all(np.isfinite(model.covariance()))
    for call in (model.covariance, model.standard_errors):
        with pytest.raises(WeightedRefusal, match="robust=False"):
            call(robust=False)
    with pytest.raises(WeightedRefusal, match="robust=False"):
        model.summary(robust=False)
    restored = fine_gray.FineGrayModel.from_dict(model.to_dict())
    np.testing.assert_array_equal(restored.covariance(), model.covariance())
    with pytest.raises(WeightedRefusal):
        restored.covariance(robust=False)
