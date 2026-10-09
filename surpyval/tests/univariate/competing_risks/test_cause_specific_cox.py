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


# ---------------------------------------------------------------------------
# #384: the incidences and the survival agree.
# ---------------------------------------------------------------------------


_CR = dict(
    x=np.array(
        [
            0.946,
            1.734,
            2.396,
            3.002,
            3.577,
            4.137,
            4.688,
            5.239,
            5.793,
            6.356,
            6.932,
            7.527,
            8.144,
            8.792,
            9.476,
            10.207,
            10.998,
            11.865,
            12.835,
            13.947,
            15.266,
            16.922,
            19.217,
            23.277,
        ]
    ),
    e=np.array([None, 1, 2, 1] * 6, dtype=object),
    n=np.array([1, 1, 2, 1, 1, 2] + [1] * 18),
    Z=np.array([[0.0], [1.0], [1.0]] * 8),
)


def test_cause_specific_incidences_sum_to_one_minus_sf():
    # They were built on the product-limit survival while sf is exp(-H):
    # on the conformance fixture at t = 30 they summed to 1.0 against
    # 1 - sf = 0.975.
    model = CompetingRisksProportionalHazards.fit(**_CR)
    t = np.linspace(0, 30, 61)
    for z in ([0.0], [1.0], [4.0]):
        total = model.cif(t, z, 1) + model.cif(t, z, 2)
        np.testing.assert_allclose(total, model.ff(t, z), rtol=1e-12)
        assert np.all(total <= 1.0)


def test_cause_specific_incidences_match_r_survival():
    # R 4 survival 3.5-8, the multi-state Cox model (Breslow ties, case
    # weights n):
    #   fit <- coxph(Surv(x, factor(e)) ~ z, id = id, weights = n,
    #                ties = "breslow")
    #   summary(survfit(fit, newdata = data.frame(z = c(0, 1))),
    #           times = c(5, 10, 20))$pstate
    model = CompetingRisksProportionalHazards.fit(**_CR, tie_method="breslow")
    np.testing.assert_allclose(
        model.betas.ravel(), [-0.1717714, -0.1262021], rtol=1e-6
    )
    t = np.array([5.0, 10.0, 20.0])
    expected = {
        0.0: [
            [0.69782892, 0.41618296, 0.08114849],
            [0.1756588, 0.3638424, 0.5851259],
            [0.1265123, 0.2199746, 0.3337256],
        ],
        1.0: [
            [0.7342261, 0.4716303, 0.1155839],
            [0.1514945, 0.3231542, 0.5495277],
            [0.1142794, 0.2052155, 0.3348884],
        ],
    }
    for z, (sf, cif1, cif2) in expected.items():
        np.testing.assert_allclose(model.sf(t, [z]), sf, rtol=1e-6)
        np.testing.assert_allclose(model.cif(t, [z], 1), cif1, rtol=1e-6)
        np.testing.assert_allclose(model.cif(t, [z], 2), cif2, rtol=1e-6)


def test_604_cause_specific_cox_comparison_values_are_r_survivals():
    # R survival 3.x, the multi-state coxph(Surv(x, factor(code)) ~ z1 +
    # z2) on competing_risks_regression_data(): its partial loglik, AIC and
    # BIC (k = 4 coefficients, BIC's n the 91 events of both causes).
    import surpyval as sp

    x, Z, e = competing_risks_regression_data()
    model = CompetingRisksProportionalHazards.fit(x, Z, e)
    assert isinstance(model.log_likelihood, float)
    assert model.log_likelihood == pytest.approx(-332.711461573959, rel=1e-9)
    assert model.aic() == pytest.approx(673.422923147917, rel=1e-9)
    assert model.bic() == pytest.approx(683.466361173984, rel=1e-9)
    restored = sp.from_dict(model.to_dict())
    for name in ("neg_ll", "aic", "aic_c", "bic"):
        assert getattr(restored, name)() == getattr(model, name)()


# -- inference, names, phi_e and beta (#656) -------------------------------


def _656_data():
    rng = np.random.default_rng(0)
    N = 200
    Z = np.c_[rng.binomial(1, 0.5, N), rng.normal(size=N)]
    ta = rng.exponential(1 / (0.1 * np.exp(0.7 * Z[:, 0])))
    tb = rng.exponential(1 / 0.05, N)
    tc = rng.uniform(0, 20, N)
    x = np.minimum(np.minimum(ta, tb), tc).round(2)
    first = np.where(ta < tb, "a", "b")
    e = np.where(tc < np.minimum(ta, tb), None, first).astype(object)
    return x, Z, e


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_656_per_cause_inference(how):
    import surpyval as sp

    x, Z, e = _656_data()
    model = CRPH.fit(x, Z, e, model=how)
    names = ["a: coef_0", "a: coef_1", "b: coef_0", "b: coef_1"]
    assert model.parameter_names == names
    np.testing.assert_array_equal(model.params, model.betas.ravel())
    # Each cause's block is that cause's own fit's
    for i, cause in enumerate(["a", "b"]):
        if how == "Cox":
            own = sp.CoxPH.fit(x, Z, (e != cause).astype(int))
        else:
            own = FineGray.fit(x, Z, e, event=cause)
        block = model.covariance()[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
        np.testing.assert_allclose(block, own.covariance(), rtol=1e-8)
    assert model.covariance()[0, 2] == 0.0
    se = model.standard_errors()
    np.testing.assert_allclose(se, np.sqrt(np.diag(model.covariance())))
    table = model.summary()
    assert table.index.tolist() == names
    np.testing.assert_allclose(table["se(coef)"], se)
    np.testing.assert_allclose(table["p"], model.p_values)
    text = repr(model)
    assert "a: coef_0" in text and "object at" not in text
    restored = sp.from_dict(model.to_dict())
    np.testing.assert_allclose(restored.covariance(), model.covariance())


def test_656_names_from_dataframe_columns():
    x, Z, e = _656_data()
    df = pd.DataFrame({"x": x, "e": e, "grp": Z[:, 0], "age": Z[:, 1]})
    model = CRPH.fit_from_df(df, "x", "e", ["grp", "age"])
    assert model.parameter_names == ["a: grp", "a: age", "b: grp", "b: age"]


def test_656_phi_e_takes_the_cause_label():
    x, Z, e = _656_data()
    model = CRPH.fit(x, Z, e)
    expected = np.exp(Z[:2] @ model.betas[model.event_idx_map["b"]])
    np.testing.assert_allclose(model.phi_e(Z[:2], "b"), expected)
    # The row index it took before is deprecated, not refused
    with pytest.warns(DeprecationWarning, match="'b'"):
        np.testing.assert_allclose(model.phi_e(Z[:2], 1), expected)
    with pytest.raises(ValueError, match="Unknown cause 'c'"):
        model.phi_e(Z[:2], "c")


def test_656_beta_is_deprecated():
    x, Z, e = _656_data()
    model = CRPH.fit(x, Z, e)
    with pytest.warns(DeprecationWarning, match="betas"):
        beta = model.beta
    np.testing.assert_allclose(beta, model.betas.sum(axis=0))


def test_656_fine_gray_log_likelihood_is_an_attribute_error():
    x, Z, e = _656_data()
    model = CRPH.fit(x, Z, e, model="Fine-Gray")
    assert not hasattr(model, "log_likelihood")
    assert getattr(model, "log_likelihood", None) is None
    with pytest.raises(AttributeError, match="no likelihood"):
        model.log_likelihood
    with pytest.raises(ValueError, match="no likelihood"):
        model.neg_ll()
    assert isinstance(CRPH.fit(x, Z, e).log_likelihood, float)


def test_714_a_separated_cause_says_so_whatever_the_row_order():
    # Four rows tied at 0.5: causes a, a, a, b with z 0, 1, 0, 0. Cause b's
    # one event has the smallest z of its risk set, so its partial
    # likelihood has no finite maximum, and its coefficient is wherever
    # Newton's method stopped (-36.4 or -37.7, by the row order). The fit
    # says so in either order, and cause a, which has a maximum, does not
    # move.
    x = np.full(4, 0.5)
    e = np.array(["a", "a", "a", "b"], dtype=object)
    Z = np.array([[0.0], [1.0], [0.0], [0.0]])
    fits = []
    for perm in ([0, 1, 2, 3], [1, 0, 2, 3]):
        with pytest.warns(UserWarning, match="No finite maximum"):
            fits.append(CRPH.fit(x[perm], Z[perm], e[perm]))
    for model in fits:
        assert model.maximum == "no finite maximum"
        assert model.betas[1, 0] < -30
    np.testing.assert_allclose(fits[0].betas[0], fits[1].betas[0])
    np.testing.assert_allclose(
        fits[0].cif([0.5, 1.0], [[0.5]], "a"),
        fits[1].cif([0.5, 1.0], [[0.5]], "a"),
    )
    for model in fits:
        assert model.cif([1.0], [[0.5]], "b")[0] < 1e-8


def test_746_fine_gray_Hf_before_the_first_time_is_plus_zero():
    # The Fine-Gray Hf is -log(1 - cif); -log(1) was -0.0 (#746).
    x, Z, e = competing_risks_regression_data()
    model = CRPH.fit(x, Z, e, model="Fine-Gray")
    H = model.Hf([0.0, np.min(x) / 2], np.zeros(2), event=1)
    assert np.all(H == 0) and not np.any(np.signbit(H))
