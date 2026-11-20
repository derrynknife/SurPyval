"""The second wave of penalised (ridge, lasso, elastic-net) fits: the
Fine-Gray model (``FineGray.fit_penalised``), Lin and Ying's additive
hazards model (``AdditiveHazards.fit_penalised``), the semi-parametric
proportional odds model (``ProportionalOdds.fit_penalised``), the
parametric shared-frailty models (``WeibullFrailty.fit_penalised``, ...)
and the parametric fits with time-varying covariates
(``fit_tvc_penalised`` of the PH, PO and AFT families).

The references are R's glmnet 4.1.8 (R 4.3.3, survival 3.5.8, cmprsk
2.2.11; threshold 1e-30), printed by
``scripts/reference/reference_r_penalised_second_wave.R``:

- Fine-Gray: glmnet's ``family = "cox"`` on rows that carry the
  censoring weights (the subdistribution partial likelihood is a weighted
  Cox partial likelihood), on the PBC data at enrolment, death the cause
  of interest. The same rows give ``cmprsk::crr``'s fit at a negligible
  penalty, to 1e-9.
- Lin and Ying: the penalised fit minimises a quadratic, so it is
  glmnet's least-squares (``family = "gaussian"``) fit on its Cholesky
  factor, on the Rossi data; the ridge fit also has a closed form.

The others are checked by their optimality (Karush-Kuhn-Tucker)
conditions and by the identities they keep: no penalty is the
unpenalised fit; the frailty fit at ``theta = 0`` is the penalised
proportional hazards fit; splitting a time-varying subject's rows changes
nothing.
"""

import numpy as np
import pytest
from autograd import grad

import surpyval as sp
from surpyval import (
    AdditiveHazards,
    FineGray,
    ProportionalOdds,
    WeibullAFT,
    WeibullAH,
    WeibullFrailty,
    WeibullPH,
    WeibullPO,
    penalty_path,
)
from surpyval.datasets import load_pbc2, load_rossi_static

ROSSI = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
REFUSED = "fitted with a penalty"

# glmnet 4.1.8 on the weighted rows: (lambda, alpha) -> the coefficients of
# age, female, drug, log(bilirubin), albumin
GLMNET_FINE_GRAY = {
    (0.05, 0.0): [
        0.04315256487,
        -0.201978202,
        -0.1368857799,
        0.7870370006,
        -0.8352924392,
    ],
    (0.05, 0.5): [
        0.0412926934,
        -0.08780239756,
        -0.01554059534,
        0.7910509275,
        -0.7355352594,
    ],
    (0.02, 1.0): [
        0.04513442269,
        -0.06621543349,
        -0.05216760576,
        0.8564746526,
        -0.7595110527,
    ],
    (0.002, 1.0): [
        0.04859179308,
        -0.1266385379,
        -0.1586188562,
        0.8917112457,
        -0.8437280004,
    ],
}

# glmnet 4.1.8 least squares on the Cholesky factor: (lambda, alpha) -> the
# coefficients of ROSSI (excess hazards of arrest per week)
GLMNET_LIN_YING = {
    (0.05, 0.0): [
        -0.002079943786,
        -0.0002191343894,
        0.00168384658,
        -0.001182559487,
        -0.001860131242,
        -0.0003585529849,
        0.0006503625428,
    ],
    (0.05, 0.5): [
        -0.0009481099794,
        -0.0001698668191,
        0.0,
        -0.0008100634128,
        -0.0007451292934,
        0.0,
        0.0004652623158,
    ],
    (0.02, 1.0): [
        -0.001211864319,
        -0.0001840519198,
        0.0003466549828,
        -0.0008643202395,
        -0.001002142875,
        0.0,
        0.0005227127183,
    ],
    (0.005, 1.0): [
        -0.001941822193,
        -0.0002161985717,
        0.001435806734,
        -0.001070305854,
        -0.001717086381,
        -0.0001489892784,
        0.0006513738224,
    ],
}


def _pbc():
    df = load_pbc2().drop_duplicates("id")
    Z = np.column_stack(
        [
            df["age"],
            df["sex"] == "female",
            df["drug"] == "D-penicil",
            np.log(df["serBilir"]),
            df["albumin"],
        ]
    ).astype(float)
    e = np.where(df["status"] == "alive", None, df["status"]).astype(object)
    return df["years"].to_numpy(float), Z, e


def _rossi():
    df = load_rossi_static()
    return (
        df["week"].to_numpy(float),
        df[ROSSI].to_numpy(float),
        1 - df["arrest"].to_numpy(),
    )


def _refuses_inference(model, *calls):
    for call in (model.covariance, model.standard_errors, *calls):
        with pytest.raises(ValueError, match=REFUSED):
            call()


# -- Fine-Gray ---------------------------------------------------------------


@pytest.mark.parametrize("setting", sorted(GLMNET_FINE_GRAY))
def test_fine_gray_matches_glmnet_on_the_weighted_rows(setting):
    lam, alpha = setting
    x, Z, e = _pbc()
    model = FineGray.fit_penalised(
        x, Z, e, event="dead", penalizer=lam, l1_ratio=alpha
    )
    assert model.maximum == "verified"
    ref = np.asarray(GLMNET_FINE_GRAY[setting])
    np.testing.assert_array_equal(model.beta == 0, ref == 0)
    np.testing.assert_allclose(model.beta, ref, rtol=1e-6, atol=1e-10)


def test_fine_gray_no_penalty_is_the_fit():
    x, Z, e = _pbc()
    plain = FineGray.fit(x, Z, e, event="dead")
    zero = FineGray.fit_penalised(x, Z, e, event="dead", penalizer=0.0)
    np.testing.assert_allclose(zero.beta, plain.beta, rtol=1e-7)
    assert zero.neg_ll() == pytest.approx(plain.neg_ll(), rel=1e-12)
    np.testing.assert_allclose(
        zero.cif([2.0, 5.0], Z[:2]), plain.cif([2.0, 5.0], Z[:2]), rtol=1e-7
    )
    # The effective degrees of freedom of no penalty are the coefficients
    assert zero.aic() == pytest.approx(plain.aic(), rel=1e-12)


def test_fine_gray_stratified_no_penalty_is_the_stratified_fit():
    x, Z, e = _pbc()
    strata = (Z[:, 0] > 50).astype(int)
    plain = FineGray.fit(x, Z, e, event="dead", strata=strata)
    zero = FineGray.fit_penalised(
        x, Z, e, event="dead", strata=strata, penalizer=0.0
    )
    np.testing.assert_allclose(zero.beta, plain.beta, rtol=1e-7)
    np.testing.assert_allclose(
        zero.cif([5.0], Z[:1], stratum=1),
        plain.cif([5.0], Z[:1], stratum=1),
        rtol=1e-7,
    )
    lasso = FineGray.fit_penalised(
        x, Z, e, event="dead", strata=strata, penalizer=0.02, l1_ratio=1
    )
    assert lasso.maximum == "verified" and lasso.is_stratified


def test_fine_gray_refusals_saving_and_criteria():
    x, Z, e = _pbc()
    model = FineGray.fit_penalised(
        x, Z, e, event="dead", penalizer=0.02, l1_ratio=1
    )
    _refuses_inference(
        model,
        lambda: model.covariance(robust=False),
        lambda: model.standard_errors(robust=False),
        model.diagnose,
    )
    assert np.all(np.isnan(model.p_values))
    assert np.all(np.isnan(model.summary()["se(coef)"]))
    assert "Penalty" in repr(model)
    k = model.penalty.effective_df
    assert k == np.sum(model.beta != 0)
    assert model.aic() == pytest.approx(2 * k + 2 * model.neg_ll())
    restored = sp.from_dict(model.to_dict())
    assert restored.penalty.effective_df == k
    np.testing.assert_array_equal(restored.beta, model.beta)
    np.testing.assert_allclose(
        restored.cif([2.0, 5.0], Z[:2]), model.cif([2.0, 5.0], Z[:2])
    )
    with pytest.raises(ValueError, match=REFUSED):
        restored.covariance()
    with pytest.raises(ValueError, match="does not take weights"):
        FineGray.fit_penalised(x, Z, e, event="dead", w=np.ones(x.size))
    with pytest.raises(ValueError, match="FineGray"):
        FineGray.fit_penalised(x, Z, e, event="dead", tl=x / 2)


def test_fine_gray_penalty_path_chooses_by_bic():
    x, Z, e = _pbc()
    path = penalty_path(
        FineGray.fit_penalised,
        x,
        Z,
        e,
        event="dead",
        l1_ratio=1,
        n_penalties=5,
    )
    assert np.all(path.coefficients[0] == 0)
    assert np.isfinite(path.best_penalizer("bic"))


# -- Lin and Ying's additive hazards -----------------------------------------


@pytest.mark.parametrize("setting", sorted(GLMNET_LIN_YING))
def test_lin_ying_matches_glmnet_on_the_quadratic(setting):
    lam, alpha = setting
    x, Z, c = _rossi()
    model = AdditiveHazards.fit_penalised(
        x, Z, c=c, penalizer=lam, l1_ratio=alpha
    )
    assert model.maximum == "verified"
    ref = np.asarray(GLMNET_LIN_YING[setting])
    np.testing.assert_array_equal(model.beta == 0, ref == 0)
    np.testing.assert_allclose(model.beta, ref, rtol=1e-6, atol=1e-12)


def test_lin_ying_ridge_is_its_closed_form():
    # (A / (N tau s s') + lambda I) v = b / (N s), beta = v / (s tau)
    from surpyval.univariate.regression.additive_hazards.additive_hazards import (  # noqa: E501
        _estimating_equations,
        _validate,
    )

    x, Z, c = _rossi()
    model = AdditiveHazards.fit_penalised(x, Z, c=c, penalizer=0.3)
    sums = _estimating_equations(*_validate(x, Z, c, None))
    s, tau, N = model.penalty.scale, model.time_scale, x.size
    assert tau == pytest.approx(x.mean())
    Q = sums.A / (tau * np.outer(s, s)) / N
    v = np.linalg.solve(Q + 0.3 * np.eye(Z.shape[1]), sums.b / s / N)
    np.testing.assert_allclose(model.beta, v / (s * tau), rtol=1e-9)


def test_lin_ying_no_penalty_is_the_fit():
    x, Z, c = _rossi()
    plain = AdditiveHazards.fit(x, Z, c=c)
    zero = AdditiveHazards.fit_penalised(x, Z, c=c, penalizer=0.0)
    np.testing.assert_allclose(zero.beta, plain.beta, rtol=1e-9)
    np.testing.assert_allclose(zero.H0, plain.H0, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        zero.sf([10.0, 30.0], Z[:2]), plain.sf([10.0, 30.0], Z[:2])
    )


@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 1.0])
def test_lin_ying_penalty_does_not_depend_on_the_unit_of_time(l1_ratio):
    # Weeks to days: the coefficients, rates per unit of time, are divided
    # by 7 and the fit is otherwise the same, for every l1_ratio.
    x, Z, c = _rossi()
    weeks = AdditiveHazards.fit_penalised(
        x, Z, c=c, penalizer=0.02, l1_ratio=l1_ratio
    )
    days = AdditiveHazards.fit_penalised(
        7 * x, Z, c=c, penalizer=0.02, l1_ratio=l1_ratio
    )
    np.testing.assert_allclose(7 * days.beta, weeks.beta, rtol=1e-9)
    np.testing.assert_allclose(
        days.sf([70.0, 210.0], Z[:2]), weeks.sf([10.0, 30.0], Z[:2])
    )


def test_lin_ying_refusals_saving_and_path():
    x, Z, c = _rossi()
    model = AdditiveHazards.fit_penalised(
        x, Z, c=c, penalizer=0.02, l1_ratio=1
    )
    _refuses_inference(model, model.diagnose)
    assert np.all(np.isnan(model.summary()["se(coef)"]))
    assert "Penalty" in repr(model)
    with pytest.raises(AttributeError, match="no AIC"):
        model.aic()
    restored = sp.from_dict(model.to_dict())
    assert restored.time_scale == model.time_scale
    assert restored.penalty.effective_df == model.penalty.effective_df
    np.testing.assert_array_equal(
        restored.sf([10.0, 30.0], Z[:2]), model.sf([10.0, 30.0], Z[:2])
    )
    with pytest.raises(ValueError, match=REFUSED):
        restored.covariance()
    with pytest.raises(ValueError, match="does not take weights"):
        AdditiveHazards.fit_penalised(x, Z, c=c, w=np.ones(x.size))
    # A path has the coefficients, but nothing to choose by
    path = penalty_path(
        AdditiveHazards.fit_penalised, x, Z, c=c, l1_ratio=1, n_penalties=4
    )
    assert np.all(path.coefficients[0] == 0)
    assert np.all(np.isnan(path.bic)) and not path.likelihood
    assert "no likelihood" in repr(path)
    with pytest.raises(ValueError, match="needs a likelihood"):
        path.best("bic")


# -- semi-parametric proportional odds ---------------------------------------


def _po_optimality(model, x, Z, c, tl=None):
    """The optimality conditions of the penalised profile likelihood at
    the fit, on the covariates' own scale."""
    from surpyval.univariate.regression._fit_skeleton import covariate_center
    from surpyval.univariate.regression.proportional_odds.proportional_odds import (  # noqa: E501
        _inner,
        _POLikelihood,
    )

    n = np.ones(x.size)
    tl = np.full(x.size, -np.inf) if tl is None else tl
    lik = _POLikelihood(x, c, n, tl, Z - covariate_center(Z, n))
    beta = np.asarray(model.beta, dtype=float)
    _, der = _inner(lik, -beta, lik.start(x, n, tl), 1e-13)
    # The gradient of -pl / N in beta
    g = np.asarray(der["grad_gamma"]) / x.size
    pen, a = model.penalty.penalizer, model.penalty.l1_ratio
    s = model.penalty.scale
    g = g + pen * (1 - a) * s**2 * beta
    nonzero = beta != 0
    np.testing.assert_allclose(
        g[nonzero] + pen[nonzero] * a * s[nonzero] * np.sign(beta[nonzero]),
        0,
        atol=1e-8,
    )
    assert np.all(
        np.abs(g[~nonzero]) <= pen[~nonzero] * a * s[~nonzero] + 1e-8
    )
    return nonzero


@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 1.0])
def test_proportional_odds_optimality_conditions(l1_ratio):
    x, Z, c = _rossi()
    model = ProportionalOdds.fit_penalised(
        x, Z, c=c, penalizer=0.02, l1_ratio=l1_ratio
    )
    assert model.maximum == "verified"
    nonzero = _po_optimality(model, x, Z, c)
    if l1_ratio == 1.0:
        assert (~nonzero).any()


def test_proportional_odds_with_delayed_entry():
    x, Z, c = _rossi()
    tl = np.where(np.arange(x.size) % 3 == 0, np.minimum(4.0, x / 2), 0.0)
    model = ProportionalOdds.fit_penalised(
        x, Z, c=c, tl=tl, penalizer=0.01, l1_ratio=0.5
    )
    assert model.maximum == "verified"
    _po_optimality(model, x, Z, c, tl)


def test_proportional_odds_no_penalty_is_the_fit():
    x, Z, c = _rossi()
    plain = ProportionalOdds.fit(x, Z, c=c)
    zero = ProportionalOdds.fit_penalised(x, Z, c=c, penalizer=0.0)
    np.testing.assert_allclose(zero.beta, plain.beta, rtol=1e-7)
    assert zero.neg_ll() == pytest.approx(plain.neg_ll(), rel=1e-12)
    assert zero.aic() == pytest.approx(plain.aic(), rel=1e-12)
    np.testing.assert_allclose(
        zero.sf([10.0, 30.0], Z[:2]), plain.sf([10.0, 30.0], Z[:2]), rtol=1e-8
    )


def test_proportional_odds_refusals_and_saving():
    x, Z, c = _rossi()
    model = ProportionalOdds.fit_penalised(
        x, Z, c=c, penalizer=0.02, l1_ratio=1
    )
    _refuses_inference(
        model,
        lambda: model.param_cb("coef_0"),
        lambda: model.param_cb("coef_0", method="lr"),
        model.diagnose,
    )
    assert np.all(np.isnan(model.summary()["se(coef)"]))
    assert "Penalty" in repr(model)
    k = model.penalty.effective_df
    assert model.aic() == pytest.approx(2 * k + 2 * model.neg_ll())
    restored = sp.from_dict(model.to_dict())
    assert restored.penalty.effective_df == k
    np.testing.assert_array_equal(
        restored.sf([10.0, 30.0], Z[:2]), model.sf([10.0, 30.0], Z[:2])
    )
    with pytest.raises(ValueError, match=REFUSED):
        restored.covariance()
    with pytest.raises(ValueError, match="does not take weights"):
        ProportionalOdds.fit_penalised(x, Z, c=c, w=np.ones(x.size))


# -- parametric shared frailty -----------------------------------------------


def _frailty_data():
    """Thirty groups of six units sharing a gamma frailty of variance 0.5;
    one covariate acts, three do not."""
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(30), 6)
    u = rng.gamma(2.0, 0.5, 30)[groups]
    Z = rng.normal(size=(180, 4))
    H = rng.exponential(1, 180) / (u * np.exp(0.5 * Z[:, 0]))
    return 10 * H**0.5, Z, groups


def _no_frailty_data():
    """Groups less alike than chance: each group's six units are at the
    same six quantiles of the baseline (before their covariates act), so
    the groups' events are spread alike and the frailty variance's
    estimate is at 0 (the unpenalised fit's theta is 0 to rounding)."""
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(30), 6)
    Z = rng.normal(size=(180, 4))
    p = np.tile((np.arange(6) + 0.5) / 6, 30)
    H = -np.log1p(-p) / np.exp(0.5 * Z[:, 0])
    return 10 * H**0.5, Z, groups


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
@pytest.mark.parametrize("l1_ratio", [0.0, 1.0])
def test_frailty_optimality_conditions(family, l1_ratio):
    # The marginal likelihood's gradient over N: 0 in the baseline's
    # parameters and theta (free), and the penalty's subgradient in the
    # coefficients.
    x, Z, groups = _frailty_data()
    fitter = sp.Frailty(sp.Weibull, family=family)
    model = fitter.fit_penalised(
        x, Z, groups=groups, penalizer=0.05, l1_ratio=l1_ratio
    )
    assert model.maximum == "verified" and model.theta > 0
    data = model._fit_data
    from surpyval.univariate.regression.frailty.frailty_fitter import (
        _AUTOGRAD,
    )

    def neg_ll(v):
        return (
            fitter._neg_ll_natural(
                v,
                data["x"],
                data["c"],
                data["w"],
                data["Z"],
                data["inv"],
                4,
                _AUTOGRAD,
            )
            / x.size
        )

    g = np.asarray(grad(neg_ll)(np.asarray(model.params, dtype=float)))
    np.testing.assert_allclose(g[:2], 0, atol=1e-7)
    np.testing.assert_allclose(g[-1], 0, atol=1e-7)
    pen, s = model.penalty.penalizer, model.penalty.scale
    b = model.beta
    gb = g[2:-1] + pen * (1 - l1_ratio) * s**2 * b
    nonzero = b != 0
    np.testing.assert_allclose(
        gb[nonzero]
        + pen[nonzero] * l1_ratio * s[nonzero] * np.sign(b[nonzero]),
        0,
        atol=1e-7,
    )
    assert np.all(
        np.abs(gb[~nonzero]) <= pen[~nonzero] * l1_ratio * s[~nonzero] + 1e-8
    )


def test_frailty_no_penalty_is_the_fit():
    # The same maximum, to the unpenalised search's tolerance (the
    # penalised one's is tighter: its likelihood is at least as high).
    x, Z, groups = _frailty_data()
    plain = WeibullFrailty.fit(x, Z, groups=groups)
    zero = WeibullFrailty.fit_penalised(x, Z, groups=groups, penalizer=0.0)
    np.testing.assert_allclose(zero.params, plain.params, atol=1e-5)
    assert zero.neg_ll() <= plain.neg_ll() + 1e-9
    assert zero.neg_ll() == pytest.approx(plain.neg_ll(), rel=1e-9)
    assert zero.aic() == pytest.approx(plain.aic(), rel=1e-9)


@pytest.mark.parametrize("l1_ratio", [0.0, 1.0])
def test_frailty_at_theta_zero_is_the_penalised_ph_fit(l1_ratio):
    # No shared frailty in the data: the objective is smallest at theta =
    # 0, where the model is the proportional hazards model of the same
    # baseline, so the fit is WeibullPH.fit_penalised's.
    x, Z, groups = _no_frailty_data()
    assert WeibullFrailty.fit(x, Z, groups=groups).theta < 1e-12
    model = WeibullFrailty.fit_penalised(
        x, Z, groups=groups, penalizer=0.05, l1_ratio=l1_ratio
    )
    ph = WeibullPH.fit_penalised(x, Z, penalizer=0.05, l1_ratio=l1_ratio)
    assert model.theta == 0.0 and model.maximum == "verified"
    np.testing.assert_allclose(model.params[:-1], ph.params, rtol=1e-7)
    np.testing.assert_array_equal(model.beta == 0, ph.params[2:] == 0)
    # theta, at 0, counts one parameter, as in the unpenalised fit
    assert model.penalty.effective_df == pytest.approx(
        ph.penalty.effective_df + 1
    )
    np.testing.assert_allclose(
        model.sf([5.0, 10.0], Z[0]), ph.sf([5.0, 10.0], Z[0]), rtol=1e-7
    )


def test_frailty_refusals_and_saving():
    x, Z, groups = _frailty_data()
    model = WeibullFrailty.fit_penalised(
        x, Z, groups=groups, penalizer=0.05, l1_ratio=1
    )
    _refuses_inference(
        model,
        lambda: model.param_cb("coef_0"),
        lambda: model.param_cb("theta", method="lr"),
        model.diagnose,
    )
    assert np.all(np.isnan(model.p_values))
    assert np.all(np.isnan(model.summary()["se(coef)"]))
    assert "Penalty" in repr(model)
    k = model.penalty.effective_df
    assert model.aic() == pytest.approx(2 * k + 2 * model.neg_ll())
    restored = sp.from_dict(model.to_dict())
    assert restored.penalty.effective_df == k
    np.testing.assert_allclose(
        restored.sf([5.0, 10.0], Z[0]), model.sf([5.0, 10.0], Z[0])
    )
    with pytest.raises(ValueError, match=REFUSED):
        restored.covariance()
    with pytest.raises(ValueError, match="no Z was given"):
        WeibullFrailty.fit_penalised(x, None, groups=groups)
    with pytest.raises(ValueError, match="does not take weights"):
        WeibullFrailty.fit_penalised(x, Z, groups=groups, w=1.0)


# -- parametric fits with time-varying covariates ----------------------------


def _switch_data(seed=0, n=300):
    """Units whose hazard rises when a stress switches on, with two
    covariates that do nothing; start-stop rows."""
    rng = np.random.default_rng(seed)
    switch = rng.uniform(0.3, 1.5, n)
    t_low = rng.exponential(2.0, n)
    t_high = switch + rng.exponential(2.0 / np.e, n)
    T = np.where(t_low > switch, t_high, t_low)
    one = T <= switch
    i = np.r_[np.arange(n), np.flatnonzero(~one)]
    xl = np.r_[np.zeros(n), switch[~one]]
    xr = np.r_[np.where(one, T, switch), T[~one]]
    c = np.r_[np.where(one, 0, 1), np.zeros((~one).sum(), dtype=int)]
    noise = rng.normal(size=(n, 2))[i]
    Z = np.column_stack([np.r_[np.zeros(n), np.ones((~one).sum())], noise])
    return i, xl, xr, c, Z


def _aft_data(seed=1, n=200):
    """Each unit's stress steps up by 0.5 half way through its life; two
    covariates do nothing. The rows tile each life from 0."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    T = rng.weibull(2.0, n) * 5 * np.exp(-0.5 * z)
    other = rng.normal(size=(n, 2))
    i = np.repeat(np.arange(n), 2)
    xl = np.column_stack([0 * T, T / 2]).ravel()
    xr = np.column_stack([T / 2, T]).ravel()
    c = np.tile([1, 0], n)
    Z = np.column_stack([np.column_stack([z, z + 0.5]).ravel(), other[i]])
    return i, xl, xr, c, Z


def _split(i, xl, xr, c, Z, pieces=3):
    """Every row split into ``pieces`` rows of equal length with the same
    covariates, the status on the last."""
    rows = []
    for ii, a, b, cc, z in zip(i, xl, xr, c, Z):
        cuts = np.linspace(a, b, pieces + 1)
        for k in range(pieces):
            last = k == pieces - 1
            rows.append((ii, cuts[k], cuts[k + 1], cc if last else 1, z))
    i2, a2, b2, c2, Z2 = zip(*rows)
    return (
        np.array(i2),
        np.array(a2),
        np.array(b2),
        np.array(c2),
        np.array(Z2, dtype=float),
    )


TVC = [
    pytest.param(WeibullPH, _switch_data, id="WeibullPH"),
    pytest.param(WeibullPO, _switch_data, id="WeibullPO"),
    pytest.param(WeibullAFT, _aft_data, id="WeibullAFT"),
]


@pytest.mark.parametrize("fitter, data", TVC)
@pytest.mark.parametrize("l1_ratio", [0.0, 1.0])
def test_tvc_does_not_depend_on_how_rows_are_split(fitter, data, l1_ratio):
    # The likelihood is the same however a subject's follow-up is split
    # into rows with the same covariates, and so are N (the subjects) and
    # the scale (each subject's time-averaged covariates).
    rows = data()
    kw = dict(penalizer=0.03, l1_ratio=l1_ratio)
    whole = fitter.fit_tvc_penalised(*rows, **kw)
    parts = fitter.fit_tvc_penalised(*_split(*rows), **kw)
    assert whole.maximum == parts.maximum == "verified"
    assert whole.is_tvc and whole.penalty is not None
    np.testing.assert_allclose(parts.params, whole.params, rtol=1e-7)
    np.testing.assert_allclose(parts.penalty.scale, whole.penalty.scale)
    if l1_ratio == 1.0:
        assert np.any(whole.params[2:] == 0)


@pytest.mark.parametrize("fitter, data", TVC)
def test_tvc_no_penalty_is_the_fit_tvc(fitter, data):
    # The same maximum, to the unpenalised search's tolerance (the
    # penalised one's is tighter: its likelihood is at least as high).
    rows = data()
    plain = fitter.fit_tvc(*rows)
    zero = fitter.fit_tvc_penalised(*rows, penalizer=0.0)
    np.testing.assert_allclose(zero.params, plain.params, atol=1e-5)
    assert zero.neg_ll() <= plain.neg_ll() + 1e-9
    assert zero.neg_ll() == pytest.approx(plain.neg_ll(), rel=1e-10)
    assert zero.bic() == pytest.approx(plain.bic(), rel=1e-10)


@pytest.mark.parametrize("fitter, data", TVC)
def test_tvc_optimality_conditions(fitter, data):
    # The gradient of the model's own likelihood over the subjects N:
    # 0 in the baseline's parameters, the penalty's subgradient in the
    # coefficients.
    i, xl, xr, c, Z = data()
    model = fitter.fit_tvc_penalised(
        i, xl, xr, c, Z, penalizer=0.03, l1_ratio=0.5
    )
    N = np.unique(i).size
    params = np.asarray(model.params, dtype=float)
    g = np.asarray(
        grad(lambda v: model.model.neg_ll(model.data, *v) / N)(params)
    )
    np.testing.assert_allclose(g[:2], 0, atol=1e-7)
    pen, s, a = model.penalty.penalizer, model.penalty.scale, 0.5
    b = params[2:]
    gb = g[2:] + pen * (1 - a) * s**2 * b
    nonzero = b != 0
    np.testing.assert_allclose(
        gb[nonzero] + pen[nonzero] * a * s[nonzero] * np.sign(b[nonzero]),
        0,
        atol=1e-7,
    )
    assert np.all(
        np.abs(gb[~nonzero]) <= pen[~nonzero] * a * s[~nonzero] + 1e-8
    )


@pytest.mark.parametrize("fitter, data", TVC)
def test_tvc_timeline_is_the_start_stop_fit(fitter, data):
    from surpyval.univariate.regression.proportional_hazards.tvc import (
        handle_tvc,
    )

    i, xl, xr, c, Z = data()
    # The timeline of the same rows: each subject's change times, its
    # exit last with its status
    ident = np.asarray(i)
    rows_i, rows_x, rows_Z, rows_c = [], [], [], []
    for u in np.unique(ident):
        at = np.flatnonzero(ident == u)
        at = at[np.argsort(xl[at])]
        for k in at:
            rows_i.append(u)
            rows_x.append(xl[k])
            rows_Z.append(Z[k])
            rows_c.append(1)
        rows_i.append(u)
        rows_x.append(xr[at[-1]])
        rows_Z.append(Z[at[-1]])
        rows_c.append(c[at[-1]])
    kw = dict(penalizer=0.03, l1_ratio=1)
    timeline = fitter.fit_tvc_timeline_penalised(
        np.array(rows_i),
        np.array(rows_x),
        np.array(rows_Z),
        np.array(rows_c),
        **kw,
    )
    start_stop = fitter.fit_tvc_penalised(i, xl, xr, c, Z, **kw)
    assert handle_tvc is not None
    np.testing.assert_allclose(timeline.params, start_stop.params, rtol=1e-9)


@pytest.mark.parametrize("fitter, data", TVC)
def test_tvc_refusals_and_saving(fitter, data):
    i, xl, xr, c, Z = data()
    model = fitter.fit_tvc_penalised(i, xl, xr, c, Z, penalizer=0.03)
    _refuses_inference(model)
    k = model.penalty.effective_df
    assert model.aic() == pytest.approx(2 * k + 2 * model.neg_ll())
    restored = sp.from_dict(model.to_dict())
    assert restored.penalty.effective_df == k
    with pytest.raises(ValueError, match=REFUSED):
        restored.covariance()
    with pytest.raises(ValueError, match="does not take weights"):
        fitter.fit_tvc_penalised(i, xl, xr, c, Z, w=1.0)


def test_tvc_additive_hazards_is_refused_with_the_reason():
    i, xl, xr, c, Z = _switch_data()
    with pytest.raises(ValueError, match="depend on the unit of time"):
        WeibullAH.fit_tvc_penalised(i, xl, xr, c, Z)


def test_aft_tvc_refuses_a_gap_as_fit_tvc_does():
    i, xl, xr, c, Z = _aft_data()
    xl = xl.copy()
    xl[1] = xl[1] + 0.01
    with pytest.raises(ValueError, match="gap"):
        WeibullAFT.fit_tvc_penalised(i, xl, xr, c, Z)
