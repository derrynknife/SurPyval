"""Regression tests for the regression and competing-risks fixes of round
14: #376 (additive hazards survival above 1), #380 (Gray's test variance),
#384 (competing-risks Cox incidences against ``sf``), #388 (a missing
frailty group), #394 (an infinite event time), #409 (a Cox coefficient the
partial likelihood cannot determine) and #426 (Buckley-James with one
covariate row per time).
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.datasets import load_rossi_static
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    CompetingRisksProportionalHazards,
    FineGray,
)

# -- #394: an exactly observed time of inf --------------------------------

_Z5 = [[-1.0], [1.0], [0.0], [2.0], [1.0], [0.0]]
_X5 = [np.inf, 0.5, 1.0, 2.0, 3.0, 4.0]
_E5 = ["a", "b", "a", "b", "a", "a"]


@pytest.mark.parametrize(
    "fit",
    [
        lambda: sp.CoxPH.fit(x=_X5, Z=_Z5),
        lambda: sp.CoxPH.fit(x=_X5, Z=_Z5, strata=[1, 1, 1, 2, 2, 2]),
        lambda: sp.AdditiveHazards.fit(_X5, _Z5),
        lambda: sp.BuckleyJames.fit(_X5, _Z5),
        lambda: CompetingRisksProportionalHazards.fit(_X5, _Z5, _E5),
        lambda: FineGray.fit(_X5, _Z5, _E5, event="a"),
        lambda: CompetingRisks.fit(_X5, _E5),
    ],
    ids=[
        "CoxPH",
        "stratified",
        "LinYing",
        "BuckleyJames",
        "CR-Cox",
        "FG",
        "CR",
    ],
)
def test_semi_parametric_fitters_refuse_an_infinite_event_time(fit):
    # Each used to take it as an event at infinity: Cox gave beta 19.4 on
    # two rows, the Aalen-Johansen incidence counted it, and Lin-Ying
    # failed with a LinAlgError from an SVD.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=r"\(c=0\) must be finite"):
            fit()


def test_cox_accepts_an_infinite_censoring_time():
    model = sp.CoxPH.fit(
        [np.inf, 0.5, 1.0, 2.0], [[-1.0], [1.0], [0.0], [1.0]], c=[1, 0, 0, 0]
    )
    assert np.isfinite(model.beta).all()


# -- #426: Buckley-James with one covariate row per time ------------------


def _buckley_james():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(100, 2))
    t = np.exp(2.0 - 0.5 * Z[:, 0] + 0.2 * Z[:, 1] + rng.normal(0, 0.5, 100))
    c = (t > 12).astype(int)
    return sp.BuckleyJames.fit(np.minimum(t, 12), Z, c=c)


def test_buckley_james_pairs_one_row_per_time():
    # It used to raise numpy's bare matmul error.
    model = _buckley_james()
    x = np.array([5.0, 10.0, 7.0])
    Z = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.5]])
    alone = [model.sf(x[k : k + 1], Z[k])[0] for k in range(3)]
    np.testing.assert_allclose(model.sf(x, Z), alone, rtol=1e-15)
    np.testing.assert_allclose(model.ff(x, Z), 1 - np.array(alone))
    # One row is still used at every time.
    np.testing.assert_allclose(
        model.sf(x, Z[:1]), model.sf(x, Z[0]), rtol=1e-15
    )


def test_buckley_james_refuses_a_mismatched_z():
    model = _buckley_james()
    with pytest.raises(ValueError, match="3 covariate rows but there are 2"):
        model.sf([5.0, 10.0], np.zeros((3, 2)))
    with pytest.raises(ValueError, match="vector of length 2"):
        model.sf([5.0, 10.0], [0.0, 1.0, 2.0])


# -- #388: a missing frailty group ----------------------------------------


def _frailty_data():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(8), 6).astype(float)
    u = rng.gamma(2.0, 0.5, 8)[groups.astype(int)]
    Z = rng.binomial(1, 0.5, (48, 1)).astype(float)
    x = 10 * (rng.exponential(1, 48) / (u * np.exp(0.5 * Z[:, 0]))) ** 0.5
    return x, Z, groups


@pytest.mark.parametrize("missing", [np.nan, None])
def test_frailty_drops_a_row_with_a_missing_group(missing):
    # A NaN label used to be a group of its own (n_groups 9, not 8), and a
    # None label raised TypeError.
    x, Z, groups = _frailty_data()
    labels = list(groups)
    labels[3] = missing
    init = [8.0, 2.0, 0.0, 0.5]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sp.WeibullFrailty.fit(x, Z=Z, groups=labels, init=init)
    dropped = [str(w.message) for w in caught if "Dropped" in str(w.message)]
    assert dropped == ["Dropped 1 of 48 rows with a missing group label."]
    assert model.n_groups == 8
    keep = np.arange(48) != 3
    ref = sp.WeibullFrailty.fit(
        x[keep], Z=Z[keep], groups=groups[keep], init=init
    )
    np.testing.assert_allclose(model.sf(5.0, [1.0]), ref.sf(5.0, [1.0]))


def test_frailty_refuses_every_group_missing():
    x, Z, _ = _frailty_data()
    with pytest.raises(ValueError, match="Every group label is missing"):
        sp.WeibullFrailty.fit(x, Z=Z, groups=[None] * 48)


def test_frailty_predicts_nan_for_a_missing_group():
    x, Z, groups = _frailty_data()
    model = sp.WeibullFrailty.fit(x, Z=Z, groups=groups, init=[8, 2, 0, 0.5])
    assert np.isnan(model.sf([5.0, 6.0], [1.0], group=np.nan)).all()
    assert np.isfinite(model.sf(5.0, [1.0], group=groups[0]))


# -- #409: Cox coefficients the partial likelihood cannot determine -------

_SEPARATED = dict(
    x=np.array([6.5, 12.0, 2.0, 13.0]),
    Z=np.array(
        [[-2.0, 0.5, 1.0], [1.0, 1.5, 1.0], [2.0, 2.0, 1.0], [2.0, 1.0, 1.0]]
    ),
    c=np.array([0, 0, 0, 1]),
    n=np.array([3, 2, 2, 3]),
)


def _cox_data(seed=0, n=50):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    x = rng.exponential(size=n) * np.exp(-Z[:, 0])
    return x, Z


def test_cox_refuses_a_constant_column_on_separated_data():
    # It gave the constant column a coefficient of 3.1e14 and an all-NaN
    # baseline, with about ten raw numpy warnings.
    with pytest.raises(ValueError, match=r"column\(s\) \[2\] of Z"):
        sp.CoxPH.fit(**_SEPARATED)


@pytest.mark.parametrize("value", [1.0, 2000.0])
def test_cox_refuses_a_constant_column(value):
    # On data that do not separate it gave a spurious monotone-likelihood
    # warning and a NaN p-value.
    x, Z = _cox_data()
    Z = np.column_stack([Z[:, 0], np.full(len(x), value), Z[:, 1]])
    with pytest.raises(ValueError, match=r"column\(s\) \[1\] of Z"):
        sp.CoxPH.fit(x, Z)


def test_cox_warns_of_collinear_columns():
    # It returned p-values of 0 for all three, silently. A formula with no
    # intercept is fitted this way on purpose, so it warns: the
    # predictions are sound.
    x, Z = _cox_data()
    Z3 = np.column_stack([Z, Z[:, 0] - 2 * Z[:, 1]])
    with pytest.warns(UserWarning, match=r"columns \[0, 1, 2\] of Z are"):
        model = sp.CoxPH.fit(x, Z3)
    ref = sp.CoxPH.fit(x, Z)
    q = np.array([[0.5, -1.0, 2.5]])
    np.testing.assert_allclose(
        model.sf([0.5, 1.0], q), ref.sf([0.5, 1.0], q[:, :2]), rtol=1e-6
    )


def test_cox_refuses_a_column_constant_within_each_stratum():
    x, Z = _cox_data()
    strata = np.repeat([0, 1], 25)
    with pytest.raises(ValueError, match=r"column\(s\) \[2\] of Z"):
        sp.CoxPH.fit(x, np.column_stack([Z, strata]), strata=strata)
    # Across strata it is an ordinary covariate.
    assert np.isfinite(
        sp.CoxPH.fit(x, np.column_stack([Z, strata])).beta
    ).all()


def test_cox_still_fits_an_offset_covariate():
    # A covariate far from 0 but varying is identified.
    x, Z = _cox_data()
    rng = np.random.default_rng(1)
    Z = np.column_stack([Z, 10.0 + 0.5 * rng.normal(size=len(x))])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = sp.CoxPH.fit(x, Z)
    assert np.isfinite(model.p_values).all()


def test_cox_lets_a_column_of_zeros_through():
    # How a declared formula level with no rows arrives (#377): its
    # coefficient stays 0, and fit_from_df warns of the level.
    x, Z = _cox_data()
    model = sp.CoxPH.fit(x, np.column_stack([Z, np.zeros(len(x))]))
    ref = sp.CoxPH.fit(x, Z)
    assert model.beta[2] == 0.0
    np.testing.assert_allclose(model.beta[:2], ref.beta, rtol=1e-8)


def test_cox_on_separated_data_warns_once_without_the_constant_column():
    data = dict(_SEPARATED, Z=_SEPARATED["Z"][:, :2])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sp.CoxPH.fit(**data)
        sf = model.sf(np.array([5.0]), np.array([[0.5, 1.5]]))
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1 and messages[0].startswith("Monotone")
    assert np.isfinite(sf).all()


# -- #384: competing-risks Cox, incidences and survival -------------------

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


# -- #376: Lin-Ying survival stays a survival -----------------------------


def _lin_ying():
    rng = np.random.default_rng(11)
    Z = rng.normal(size=(120, 2))
    x = rng.exponential(1 / (0.2 + 0.05 * Z[:, 0] - 0.04 * Z[:, 1]).clip(0.01))
    c = (x > 8).astype(int)
    return sp.AdditiveHazards.fit(np.minimum(x, 8), Z, c=c)


@pytest.mark.parametrize("z", [[0.0, 0.0], [-3.0, 3.0], [2.0, -2.0]])
def test_lin_ying_survival_is_in_bounds_and_non_increasing(z):
    # Row (-3, 3) has a negative hazard; the survival used to climb above
    # 1 (the conformance fixture reached 57.8 at Z = (-2, 2) and 1.21 at
    # Z = 0, inside the data) and could be inf.
    model = _lin_ying()
    t = np.linspace(-1.0, 12.0, 2001)
    sf = model.sf(t, np.array(z))
    assert np.all((sf >= 0) & (sf <= 1))
    assert np.all(np.diff(sf) <= 0)
    assert np.all(sf[t <= 0] == 1.0)
    hf = model.hf(t, np.array(z))
    assert np.all(hf >= 0)


def test_lin_ying_hf_is_the_running_maximum_of_the_estimate():
    model = _lin_ying()
    t = np.sort(np.concatenate([np.linspace(0, 10, 4001), model.x]))
    for z in ([0.0, 0.0], [-3.0, 3.0], [1.0, -1.0]):
        bz = np.asarray(z) @ model.beta
        held = np.minimum(t, model.x[-1])
        H = model._baseline_H(held) + held * bz
        envelope = np.maximum(np.maximum.accumulate(H), 0.0)
        np.testing.assert_allclose(
            model.Hf(t, np.array(z)), envelope, rtol=1e-12, atol=1e-15
        )


def test_lin_ying_predictions_inside_the_data_are_the_estimate():
    # The docstring's Rossi prediction is where the estimate is at its
    # running maximum: unchanged.
    df = load_rossi_static()
    model = sp.AdditiveHazards.fit(
        df["week"].values,
        df[["fin", "age", "prio"]].values,
        c=df["arrest"].values,
    )
    np.testing.assert_allclose(
        model.sf([20, 52], [1, 25, 3]), [0.9269, 0.7725], atol=5e-5
    )


# -- #380: Gray's test is cmprsk's ----------------------------------------

_GRAY = dict(
    x=[2, 7, 6, 3, 1, 6, 3, 7, 3, 6, 2, 5, 1, 7, 1, 6, 7, 4, 7, 5, 7, 6, 8]
    + [1, 1, 6, 6, 6, 7, 4, 7, 5, 2, 2, 7, 6, 7, 8, 7, 3, 5, 4, 3, 8, 8],
    e=[1, 1, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 1, 2, 0, 2, 2, 0, 0, 0, 2, 1, 0]
    + [1, 1, 1, 2, 2, 1, 0, 0, 0, 2, 2, 0, 2, 1, 1, 1, 2, 2, 0, 2, 0, 2],
    group=[1, 1, 2, 0, 0, 1, 0, 2, 2, 1, 0, 2, 0, 2, 2, 0, 2, 2, 1, 2, 1, 1]
    + [1, 0, 1, 1, 0, 1, 0, 2, 0, 0, 1, 0, 1, 0, 2, 0, 2, 2, 2, 2, 1, 0, 1],
)


@pytest.mark.parametrize(
    "rho, event, expected",
    [
        (0, 1, 0.680791319807),
        (0, 2, 1.495728261106),
        (1, 1, 0.877130594184),
        (1, 2, 1.212189254797),
    ],
)
def test_gray_test_matches_cmprsk_on_three_tied_groups(rho, event, expected):
    # R cmprsk 2.2-11: cuminc(x, e, group, rho = rho)$Tests. The variance
    # was SurPyval's own linearisation (6.162 against cmprsk's 5.065 on
    # the reference suite's tied fixture), and the rho weight used a
    # different pooled incidence.
    e = [None if k == 0 else k for k in _GRAY["e"]]
    res = sp.gray_test(_GRAY["x"], e, _GRAY["group"], event=event, rho=rho)
    assert res.df == 2
    assert res.statistic == pytest.approx(expected, rel=1e-10)


def test_gray_test_counts_equal_repeated_rows():
    rng = np.random.default_rng(3)
    x = rng.integers(1, 6, 30).astype(float)
    e = rng.choice(np.array([None, "a", "b"], dtype=object), 30)
    g = rng.integers(0, 2, 30)
    n = rng.integers(1, 4, 30)
    counted = sp.gray_test(x, e, g, event="a", n=n, rho=0.5)
    rows = np.repeat(np.arange(30), n)
    repeated = sp.gray_test(x[rows], e[rows], g[rows], event="a", rho=0.5)
    assert counted.statistic == pytest.approx(repeated.statistic, rel=1e-12)
