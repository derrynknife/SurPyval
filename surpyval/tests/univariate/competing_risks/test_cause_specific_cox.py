"""Cause-specific proportional-hazards competing-risks regression.

``CompetingRisksProportionalHazards.fit(model="Cox")`` fits one Cox model per
cause (other causes censored) and combines them into a cumulative-incidence
prediction. These tests pin the correctness of that combination against a
known analytic truth and exercise the DataFrame entry point.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval.tests._helpers import competing_risks_regression_data
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
)
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards as CRPH,
)
from surpyval.univariate.competing_risks import FineGray


def _exponential_cr_data(N, seed, lam=(0.5, 0.3), beta=None, cens=6.0):
    """
    Two-cause data with constant (exponential) cause-specific hazards
    ``h_e(t|Z) = lam_e exp(beta_e'Z)`` and *different* covariate effects per
    cause. The analytic cumulative incidence is

        F_e(t|Z) = lam_e(Z)/lam_tot(Z) * (1 - exp(-lam_tot(Z) t)).
    """
    if beta is None:
        beta = np.array([[0.8, -0.2], [-0.3, 0.6]])
    rng = np.random.default_rng(seed)
    lam = np.asarray(lam)
    Z = rng.uniform(-1, 1, size=(N, beta.shape[1]))
    rate = lam[None, :] * np.exp(Z @ beta.T)
    T = rng.exponential(1.0 / rate)
    x = T.min(axis=1)
    cause = T.argmin(axis=1) + 1
    C = rng.exponential(cens, size=N)
    c = (x > C).astype(int)
    x = np.minimum(x, C)
    e = np.array(
        [None if ci == 1 else int(ce) for ci, ce in zip(c, cause)],
        dtype=object,
    )
    return x, Z, e, c, lam, beta


def _analytic_cif(t, Z0, lam, beta, event):
    lz = lam * np.exp(Z0 @ beta.T)
    tot = lz.sum()
    return lz[event - 1] / tot * (1 - np.exp(-tot * t))


def test_cox_cif_matches_analytic_with_differing_covariate_effects():
    # The key correctness test: with different coefficients per cause, the
    # all-cause survival must combine each cause with its own coefficients
    # (not a summed coefficient), and the baseline must be cause-specific.
    x, Z, e, c, lam, beta = _exponential_cr_data(20000, 0)
    m = CRPH.fit(x, Z, e, c=c, model="Cox")
    Z0 = np.array([0.5, -0.3])
    ts = np.array([0.5, 1.0, 2.0, 4.0])
    for event in (1, 2):
        fitted = m.cif(ts, Z0, event)
        truth = _analytic_cif(ts, Z0, lam, beta, event)
        assert np.allclose(fitted, truth, atol=0.02)


def test_cox_recovers_per_cause_coefficients():
    x, Z, e, c, lam, beta = _exponential_cr_data(20000, 1)
    m = CRPH.fit(x, Z, e, c=c, model="Cox")
    assert np.allclose(m.betas[m.event_idx_map[1]], beta[0], atol=0.05)
    assert np.allclose(m.betas[m.event_idx_map[2]], beta[1], atol=0.05)


def test_cif_is_monotone_and_bounded():
    x, Z, e, c, lam, beta = _exponential_cr_data(4000, 2)
    m = CRPH.fit(x, Z, e, c=c, model="Cox")
    t = np.linspace(0.01, 8.0, 60)
    cif = m.cif(t, [0.2, -0.1], 1)
    assert np.all(cif >= 0) and np.all(cif <= 1)
    assert np.all(np.diff(cif) >= -1e-9)


def test_cifs_sum_below_one():
    # The competing CIFs plus the overall survival must sum to one.
    x, Z, e, c, lam, beta = _exponential_cr_data(6000, 3)
    m = CRPH.fit(x, Z, e, c=c, model="Cox")
    t = np.array([0.5, 1.0, 3.0])
    Z0 = [0.1, 0.2]
    total = m.cif(t, Z0, 1) + m.cif(t, Z0, 2) + m.sf(t, Z0)
    assert np.allclose(total, 1.0, atol=0.02)


def test_baseline_uses_cause_specific_events_only():
    # A cause with very few events must have a much smaller cumulative
    # incidence than a common cause -- a direct probe that the baseline is
    # built from cause-specific (not all-cause) event counts.
    rng = np.random.default_rng(4)
    N = 8000
    Z = rng.uniform(-1, 1, size=(N, 1))
    # Cause 1 is ~9x more frequent than cause 2.
    rate = np.array([0.9, 0.1])[None, :] * np.exp(
        Z @ np.array([[0.0], [0.0]]).T
    )
    T = rng.exponential(1.0 / rate)
    x = T.min(axis=1)
    cause = T.argmin(axis=1) + 1
    e = np.array([int(ce) for ce in cause], dtype=object)
    c = np.zeros(N, dtype=int)
    m = CRPH.fit(x, Z, e, c=c, model="Cox")
    f1 = m.cif([5.0], [0.0], 1)[0]
    f2 = m.cif([5.0], [0.0], 2)[0]
    assert f1 > 5 * f2  # cause 1 dominates, as its ~0.9 share implies


# --- fit_from_df ----------------------------------------------------------


def _frame(x, Z, e, c):
    return pd.DataFrame(
        {
            "t": x,
            "c": c,
            "cause": pd.array(e, dtype=object),
            "age": Z[:, 0],
            "dose": Z[:, 1],
        }
    )


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_fit_from_df_matches_array_fit(how):
    x, Z, e, c, lam, beta = _exponential_cr_data(4000, 5)
    df = _frame(x, Z, e, c)
    m_df = CRPH.fit_from_df(
        df,
        x_col="t",
        e_col="cause",
        Z_cols=["age", "dose"],
        c_col="c",
        model=how,
    )
    m_arr = CRPH.fit(x, Z, e, c=c, model=how)
    assert m_df.feature_names == ["age", "dose"]
    t = np.array([0.5, 1.0, 2.0])
    assert np.allclose(
        m_df.cif(t, [0.2, -0.1], 1), m_arr.cif(t, [0.2, -0.1], 1)
    )


def test_fit_from_df_accepts_nan_cause_for_censored():
    # A blank/NaN cause cell is treated as a censored observation.
    x, Z, e, c, lam, beta = _exponential_cr_data(3000, 6)
    df = _frame(x, Z, e, c)
    df_nan = df.copy()
    df_nan.loc[df_nan.c == 1, "cause"] = np.nan
    m_nan = CRPH.fit_from_df(
        df_nan, x_col="t", e_col="cause", Z_cols=["age", "dose"], c_col="c"
    )
    m_ref = CRPH.fit_from_df(
        df, x_col="t", e_col="cause", Z_cols=["age", "dose"], c_col="c"
    )
    t = np.array([1.0, 2.0])
    assert np.allclose(
        m_nan.cif(t, [0.1, 0.1], 1), m_ref.cif(t, [0.1, 0.1], 1)
    )


def test_fit_from_df_formula():
    x, Z, e, c, lam, beta = _exponential_cr_data(3000, 7)
    df = _frame(x, Z, e, c)
    m = CRPH.fit_from_df(
        df, x_col="t", e_col="cause", formula="age + dose", c_col="c"
    )
    assert "age" in m.feature_names and "dose" in m.feature_names
    assert np.all(np.isfinite(m.cif([1.0, 2.0], [0.2, -0.1], 1)))


def _two_cause_sample(seed, n=15, labels=("wear", "shock")):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 1))
    t1 = rng.exponential(1 / np.exp(0.8 * Z[:, 0]))
    t2 = rng.exponential(1 / np.exp(-0.5 * Z[:, 0]))
    x = np.minimum(t1, t2)
    e = np.where(t1 < t2, labels[0], labels[1]).astype(object)
    return x, Z, e


def test_cause_specific_incidences_sum_to_one_minus_survival():
    # #278 was fixed for the nonparametric CIF but not this path, where
    # the incidence weighted the hazard increments with exp(-H): total
    # incidence reached 1.07-1.18 in small samples, and far more where a
    # Breslow increment times a large multiplier exceeds 1.
    from surpyval.univariate.competing_risks import (
        CompetingRisksProportionalHazards,
    )

    for seed in range(20):
        x, Z, e = _two_cause_sample(seed)
        m = CompetingRisksProportionalHazards.fit(x, Z, e)
        for z in ([-1.5], [0.0], [1.5]):
            total = m.cif(m.x, z, "wear") + m.cif(m.x, z, "shock")
            assert total.max() <= 1.0 + 1e-12
            # 1 - sf, the model's own all-cause failure probability (#384).
            assert np.allclose(total, m.ff(m.x, z), rtol=1e-12, atol=1e-15)


def test_cause_order_is_sorted_and_reproducible():
    from surpyval.univariate.competing_risks import (
        CompetingRisksProportionalHazards,
    )

    x, Z, e = _two_cause_sample(0, n=60)
    m = CompetingRisksProportionalHazards.fit(x, Z, e)
    assert list(m.event_idx_map) == ["shock", "wear"]
    single = {}
    for cause in ("shock", "wear"):
        from surpyval import CoxPH

        c_e = np.where(e == cause, 0, 1)
        single[cause] = CoxPH.fit(x, Z, c_e, tie_method="efron").res.x
    assert np.allclose(m.betas[0], single["shock"])
    assert np.allclose(m.betas[1], single["wear"])


# ---------------------------------------------------------------------------
# ``cif`` with an unknown event, and covariate rows paired with
# the times; the number of covariate rows is checked.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
@pytest.mark.parametrize(
    "event, match", [(3, "Unknown cause 3"), (None, "pass `event`")]
)
def test_crph_cif_unknown_event_is_a_clear_error(how, event, match):
    x, Z, e = competing_risks_regression_data()
    model = CompetingRisksProportionalHazards.fit(x, Z, e, model=how)
    with pytest.raises(ValueError, match=match):
        model.cif([1.0], [0, 0], event)


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_crph_cif_pairs_covariate_rows_with_times(how):
    x, Z, e = competing_risks_regression_data()
    model = CompetingRisksProportionalHazards.fit(x, Z, e, model=how)
    paired = model.cif([1.0, 2.0], [[0, 0], [1, 1]], 1)
    single = [
        model.cif([1.0], [0, 0], 1)[0],
        model.cif([2.0], [1, 1], 1)[0],
    ]
    np.testing.assert_allclose(paired, single)
    with pytest.raises(ValueError, match="rows for 3 times"):
        model.cif([1.0, 2.0, 3.0], [[0, 0], [1, 1]], 1)


def test_wrong_number_of_covariate_rows_is_a_clear_error():
    x, Z, e = competing_risks_regression_data()
    with pytest.raises(ValueError, match="row"):
        CompetingRisksProportionalHazards.fit(x, Z[:-1], e)
    with pytest.raises(ValueError, match="row"):
        FineGray.fit(x, Z[:-1], e, event=1)
