"""Penalised fits with sampling weights ``w``: ``CoxPH.fit_penalised``,
``CoxPH.fit_tvc_penalised`` and the parametric families'
``fit_penalised``.

The objective is ``-l_w(theta) / N_w`` plus the penalty, ``l_w`` the
weighted log-likelihood (partial likelihood for Cox), ``N_w = sum(n w)``,
with the covariates standardised by their ``n w``-weighted standard
deviations, divisor ``N_w``.

The Cox references are R 4.6.2, glmnet 5.2 and survival 3.8.12
(``scripts/reference/reference_r_weighted_penalised.R``):
``glmnet(X, Surv(...), family = "cox", weights = w, alpha, lambda = path,
cox.ties = ties, control = list(thresh = 1e-30))`` on a path of 60
penalties from 1 down to the target. glmnet's weighted Cox objective is
``-(1 / sum(w)) l_w + lambda * penalty`` with ``standardize = TRUE``
dividing each column by its ``w``-weighted sd with divisor ``sum(w)``
(glmnet scales the weights to sum to 1 first, which changes neither):
the same ``N`` and standardisation as here. Its weighted Efron rule is
survival's (the ridge fit agrees with ``coxph(..., weights = w, ties =
"efron")`` with a ``ridge()`` term to 1e-12, printed by the script), the
one ``CoxPH.fit(w=)`` uses. Tolerance: 1e-6 absolute on the coefficients
(the brief's), with the same coefficients exactly 0; they agree to about
1e-12.

The parametric fits have no R reference with this objective (glmnet has
no Weibull family); they are checked against the optimality (KKT)
conditions of a weighted likelihood written out here with numpy, against
repeated rows, and for weight-scale invariance.
"""

import warnings

import numpy as np
import pytest

from surpyval import CoxPH, WeibullAFT, WeibullPH, WeibullPO, penalty_path
from surpyval.tests._helpers import (
    rossi_with_censoring,
    weighted_reference_data,
)
from surpyval.univariate.regression import SemiParametricRegressionModel
from surpyval.utils.weights import WeightedRefusal

COLS = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
#: glmnet 5.2 on the Rossi data with w = 0.5 + ((row * 37) % 11) / 5:
#: (ties, lambda, alpha) -> the coefficients of COLS
GLMNET_ROSSI = {
    ("breslow", 0.05, 0.0): [
        -0.342255627726,
        -0.0368953449546,
        0.225134250097,
        -0.27458126197,
        -0.549171964368,
        -0.0676081265581,
        0.0732442089793,
    ],
    ("breslow", 0.05, 0.5): [
        -0.185558167405,
        -0.0257916546822,
        0,
        -0.223208108054,
        -0.305786843853,
        0,
        0.0557365169468,
    ],
    ("breslow", 0.02, 1.0): [
        -0.24152897433,
        -0.0317314626199,
        0.0235909424626,
        -0.229532342589,
        -0.403329530234,
        0,
        0.0651598125827,
    ],
    ("efron", 0.05, 0.0): [
        -0.342164522854,
        -0.0370179403284,
        0.22497752853,
        -0.274125268948,
        -0.550018201874,
        -0.06785795821,
        0.0734046110961,
    ],
    ("efron", 0.05, 0.5): [
        -0.185506753326,
        -0.0259239958929,
        0,
        -0.222886941476,
        -0.306936482241,
        0,
        0.0559173659707,
    ],
    ("efron", 0.02, 1.0): [
        -0.241414079582,
        -0.0318902131271,
        0.0235119326082,
        -0.228935941595,
        -0.404568092006,
        0,
        0.0653645827659,
    ],
}
#: glmnet 5.2, Surv(tl, x, d) on weighted_reference_data() with its
#: weights (delayed entry; N counts rows, as fit_penalised with tl does)
GLMNET_ENTRY = {
    ("breslow", 0.05, 0.0): [0.982398399962, 0.890907856091],
    ("breslow", 0.1, 1.0): [0.697569265515, 0.721326791808],
    ("efron", 0.05, 0.0): [0.974357863939, 0.892512662885],
    ("efron", 0.1, 1.0): [0.691281184525, 0.723745483953],
}


def _rossi():
    df = rossi_with_censoring()
    x = df["week"].to_numpy(float)
    w = 0.5 + ((np.arange(1, x.size + 1) * 37) % 11) / 5
    return x, df["censored"].to_numpy(), df[COLS].to_numpy(float), w


@pytest.fixture(scope="module")
def data():
    return weighted_reference_data()


# -- Cox ---------------------------------------------------------------------


@pytest.mark.parametrize("setting", sorted(GLMNET_ROSSI))
def test_weighted_cox_matches_glmnet(setting):
    ties, lam, alpha = setting
    x, c, Z, w = _rossi()
    model = CoxPH.fit_penalised(
        x, Z, c=c, w=w, penalizer=lam, l1_ratio=alpha, tie_method=ties
    )
    assert model.maximum == "verified"
    assert model.weighted
    ref = np.asarray(GLMNET_ROSSI[setting])
    np.testing.assert_array_equal(model.params == 0, ref == 0)
    np.testing.assert_allclose(model.params, ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("setting", sorted(GLMNET_ENTRY))
def test_weighted_cox_with_delayed_entry_matches_glmnet(setting, data):
    ties, lam, alpha = setting
    model = CoxPH.fit_penalised(
        data["x"],
        data["Z"],
        c=data["c"],
        tl=data["tl"],
        w=data["w"],
        penalizer=lam,
        l1_ratio=alpha,
        tie_method=ties,
    )
    assert model.maximum == "verified"
    np.testing.assert_allclose(
        model.params, GLMNET_ENTRY[setting], rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 1.0])
def test_weighted_cox_weight_scale_changes_nothing(ties, l1_ratio):
    x, c, Z, w = _rossi()
    kw = dict(c=c, penalizer=0.03, l1_ratio=l1_ratio, tie_method=ties)
    base = CoxPH.fit_penalised(x, Z, w=w, **kw)
    for scale in (0.013, 7.3, 1e4):
        other = CoxPH.fit_penalised(x, Z, w=scale * w, **kw)
        # The search is on the weights scaled to sum to the units, so the
        # fits agree to rounding, far inside the optimiser's tolerance.
        np.testing.assert_allclose(other.params, base.params, atol=1e-12)
        np.testing.assert_array_equal(other.params == 0, base.params == 0)
        np.testing.assert_allclose(
            other.penalty.scale, base.penalty.scale, rtol=1e-12
        )
        np.testing.assert_allclose(
            other.sf(10.0, Z[:3]), base.sf(10.0, Z[:3]), rtol=1e-10
        )


def test_weighted_cox_integer_weights_are_counts_under_breslow():
    # Breslow's partial likelihood treats w = k as k copies exactly.
    # Efron's does not: with weights, R's rule (and CoxPH.fit's) takes a
    # row of weight 3 as one death at weight 3, not three tied deaths
    # (DECISIONS.md, weighted Cox ties), so the comparison is Breslow's.
    x, c, Z, _ = _rossi()
    k = 1 + np.arange(x.size) % 3
    kw = dict(c=c, penalizer=0.05, l1_ratio=0.5, tie_method="breslow")
    weighted = CoxPH.fit_penalised(x, Z, w=k, **kw)
    counted = CoxPH.fit_penalised(x, Z, n=k, **kw)
    np.testing.assert_allclose(weighted.params, counted.params, atol=1e-12)
    np.testing.assert_allclose(weighted.H0, counted.H0, rtol=1e-10)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_weighted_cox_unit_weights_are_the_unweighted_fit(ties):
    x, c, Z, _ = _rossi()
    kw = dict(c=c, penalizer=0.05, l1_ratio=0.5, tie_method=ties)
    ones = CoxPH.fit_penalised(x, Z, w=np.ones(x.size), **kw)
    plain = CoxPH.fit_penalised(x, Z, **kw)
    np.testing.assert_allclose(ones.params, plain.params, atol=1e-12)
    np.testing.assert_allclose(ones.H0, plain.H0, rtol=1e-10)
    assert ones.neg_ll() == pytest.approx(plain.neg_ll(), rel=1e-12)


def test_weighted_cox_refuses_inference_and_criteria(data):
    model = CoxPH.fit_penalised(
        data["x"], data["Z"], c=data["c"], w=data["w"], penalizer=0.05
    )
    assert model.effective_n == pytest.approx(
        np.sum(data["w"]) ** 2 / np.sum(data["w"] ** 2)
    )
    # The printout gives the effective size, and no covariance
    assert "effective sample size" in repr(model)
    assert "robust covariance" not in repr(model)
    for call in (model.aic, model.bic):
        with pytest.raises(WeightedRefusal):
            call()
    for call in (model.standard_errors, model.covariance):
        with pytest.raises(ValueError, match="fitted with a penalty"):
            call()
    with pytest.raises(ValueError, match="fitted with a penalty"):
        model.robust_covariance()


def test_weighted_cox_round_trip(data):
    model = CoxPH.fit_penalised(
        data["x"],
        data["Z"],
        c=data["c"],
        w=data["w"],
        penalizer=0.05,
        l1_ratio=1.0,
    )
    d = model.to_dict()
    assert d["schema"] == 4 and d["weighted"] is True
    back = SemiParametricRegressionModel.from_dict(d)
    assert back.weighted and back.penalty is not None
    np.testing.assert_allclose(
        back.sf([1.0, 3.0], data["Z"][:2]), model.sf([1.0, 3.0], data["Z"][:2])
    )
    with pytest.raises(WeightedRefusal):
        back.aic()
    with pytest.raises(ValueError, match="fitted with a penalty"):
        back.standard_errors()


def test_weighted_cox_warns_of_few_effective_units(data):
    w = np.where(np.arange(data["x"].size) < 3, 100.0, 1.0)
    with pytest.warns(UserWarning, match="effective sample size"):
        CoxPH.fit_penalised(
            data["x"], data["Z"], c=data["c"], w=w, penalizer=0.05
        )


def test_weighted_cox_refusals(data):
    x, Z, c, w = data["x"], data["Z"], data["c"], data["w"]
    with pytest.raises(WeightedRefusal, match="stratified"):
        CoxPH.fit_penalised(x, Z, c=c, w=w, strata=Z[:, 0])
    with pytest.raises(WeightedRefusal, match="tie_method"):
        CoxPH.fit_penalised(x, Z, c=c, w=w, tie_method="exact")
    with pytest.raises(ValueError, match="positive"):
        CoxPH.fit_penalised(x, Z, c=c, w=-w)


def _split(data):
    """Each unit's follow-up split at its midpoint, the same covariates
    and weight on both episodes."""
    x, c, w, tl, Z = data["x"], data["c"], data["w"], data["tl"], data["Z"]
    mid = (tl + x) / 2
    k = x.size
    return dict(
        i=np.repeat(np.arange(k), 2),
        xl=np.ravel(np.column_stack([tl, mid])),
        xr=np.ravel(np.column_stack([mid, x])),
        c=np.ravel(np.column_stack([np.ones(k), c])).astype(int),
        Z=np.repeat(Z, 2, axis=0),
        w=np.repeat(w, 2),
    )


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_weighted_tvc_counts_subjects(ties, data):
    # Split rows of one subject leave the partial likelihood unchanged, and
    # N and the standardisation are per subject: the fit is the one of the
    # unsplit rows (one per subject, so rows are subjects there).
    s = _split(data)
    kw = dict(penalizer=0.05, l1_ratio=0.5, tie_method=ties)
    split = CoxPH.fit_tvc_penalised(
        s["i"], s["xl"], s["xr"], s["c"], s["Z"], w=s["w"], **kw
    )
    whole = CoxPH.fit_penalised(
        data["x"],
        data["Z"],
        c=data["c"],
        tl=data["tl"],
        w=data["w"],
        **kw,
    )
    assert split.is_tvc and split.weighted
    np.testing.assert_allclose(split.params, whole.params, atol=1e-9)
    np.testing.assert_allclose(
        split.penalty.scale, whole.penalty.scale, rtol=1e-12
    )
    scaled = CoxPH.fit_tvc_penalised(
        s["i"], s["xl"], s["xr"], s["c"], s["Z"], w=3.7 * s["w"], **kw
    )
    np.testing.assert_allclose(scaled.params, split.params, atol=1e-12)


def test_weighted_tvc_subject_weight_is_time_averaged(data):
    # A subject whose weight changes between its rows counts with its
    # weight averaged over its follow-up: the same fit as the scaled
    # objective built by hand (the subjects' N_w and standardisation).
    from surpyval.univariate.regression.proportional_hazards.cox_ph import (
        _subject_covariates,
    )

    s = _split(data)
    w = s["w"] * np.tile([1.0, 2.0], data["x"].size)
    length = s["xr"] - s["xl"]
    Z_units, n_units = _subject_covariates(
        s["xr"], s["xl"], s["Z"], np.ones(s["xr"].size), s["i"], w
    )
    expected = (length * w).reshape(-1, 2).sum(1) / length.reshape(-1, 2).sum(
        1
    )
    np.testing.assert_allclose(n_units, expected, rtol=1e-12)
    np.testing.assert_allclose(Z_units, data["Z"], rtol=1e-12)
    model = CoxPH.fit_tvc_penalised(
        s["i"], s["xl"], s["xr"], s["c"], s["Z"], w=w, penalizer=0.05
    )
    assert model.maximum == "verified"


def test_weighted_timeline_is_refused():
    with pytest.raises(WeightedRefusal, match="does not take weights"):
        CoxPH.fit_tvc_timeline_penalised(
            [0, 0, 1], [0, 2, 0], [[0], [1], [0]], [1, 0, 1], w=1.0
        )


def test_weighted_penalty_path_has_no_criteria():
    x, c, Z, w = _rossi()
    path = penalty_path(
        CoxPH.fit_penalised,
        x,
        Z[:, :3],
        c=c,
        w=w,
        l1_ratio=1.0,
        n_penalties=4,
    )
    assert path.weighted and np.all(np.isnan(path.aic))
    assert path.coefficients.shape == (4, 3)
    assert "no AIC or BIC" in repr(path)
    with pytest.raises(WeightedRefusal, match="bic"):
        path.best("bic")


# -- parametric --------------------------------------------------------------


def _truncated_weighted():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(300, 4))
    t = 10 * rng.weibull(1.5, 300) * np.exp(-0.5 * Z[:, 0] + 0.3 * Z[:, 1])
    tl = rng.uniform(0, 3, 300)
    tr = tl + rng.uniform(8, 30, 300)
    keep = (t > tl) & (t < tr)
    x, Z, tl, tr = t[keep], Z[keep], tl[keep], tr[keep]
    c = (x > 12).astype(int)
    w = np.round(rng.uniform(0.3, 3.0, x.size), 2)
    return np.minimum(x, 12.0), Z, c, tl, tr, w


def _weibull_aft_rows(theta, x, c, Z, tl, tr):
    """Each row's negative log-likelihood of a Weibull AFT model with
    right censoring and truncation to (tl, tr)."""
    alpha, beta, b = theta[0], theta[1], theta[2:]
    phi = np.exp(Z @ b)

    def sf(t):
        with np.errstate(invalid="ignore"):
            u = np.where(np.isfinite(t), phi * np.maximum(t, 0) / alpha, 0)
        out = np.exp(-(u**beta))
        return np.where(np.isposinf(t), 0.0, out)

    u = phi * x / alpha
    log_df = np.log(beta / alpha * phi) + (beta - 1) * np.log(u) - u**beta
    log_obs = np.where(c == 0, log_df, np.log(sf(x) - sf(tr)))
    return -(log_obs - np.log(sf(tl) - sf(tr)))


@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("standardise", [True, False])
def test_weighted_parametric_optimality_conditions(l1_ratio, standardise):
    # The KKT conditions of -l_w / N_w + penalty, in the standardised
    # coefficients b_s = b * s (s the n w-weighted sd, divisor N_w):
    # the baseline's gradient 0; for b_s != 0, the gradient of the smooth
    # part equals -l1 * sign(b_s); for b_s == 0 it is at most l1.
    x, Z, c, tl, tr, w = _truncated_weighted()
    lam = 0.04
    model = WeibullAFT.fit_penalised(
        x,
        Z,
        c=c,
        tl=tl,
        tr=tr,
        w=w,
        penalizer=lam,
        l1_ratio=l1_ratio,
        standardise=standardise,
    )
    assert model.maximum == "verified"
    theta = np.asarray(model.params, dtype=float)
    N_w = w.sum()
    mean = w @ Z / N_w
    s = np.sqrt(w @ (Z - mean) ** 2 / N_w) if standardise else np.ones(4)
    np.testing.assert_allclose(model.penalty.scale, s, rtol=1e-12)

    def f(th):
        return np.sum(w * _weibull_aft_rows(th, x, c, Z, tl, tr)) / N_w

    # The reported value is -l_w with the weights as given
    assert f(theta) * N_w == pytest.approx(model.neg_ll(), rel=1e-10)
    h = 1e-6 * np.maximum(np.abs(theta), 1.0)
    g = np.array(
        [
            (f(theta + h[j] * e) - f(theta - h[j] * e)) / (2 * h[j])
            for j, e in enumerate(np.eye(theta.size))
        ]
    )
    b_s = theta[2:] * s
    g_s = g[2:] / s + lam * (1 - l1_ratio) * b_s
    l1 = lam * l1_ratio
    np.testing.assert_allclose(g[:2], 0, atol=1e-6)
    nonzero = b_s != 0
    np.testing.assert_allclose(
        g_s[nonzero], -l1 * np.sign(b_s[nonzero]), atol=1e-6
    )
    assert np.all(np.abs(g_s[~nonzero]) <= l1 + 1e-6)
    if l1_ratio == 1.0:
        assert not nonzero.all()  # the lasso sets some to 0


@pytest.mark.parametrize("fitter", [WeibullPH, WeibullAFT, WeibullPO])
def test_weighted_parametric_scale_counts_and_unit_weights(fitter, data):
    x, Z, c, tl, w = data["x"], data["Z"], data["c"], data["tl"], data["w"]
    kw = dict(c=c, tl=tl, penalizer=0.05, l1_ratio=0.5)
    base = fitter.fit_penalised(x, Z, w=w, **kw)
    assert base.maximum == "verified" and base.weighted
    for scale in (0.01, 7.3):
        other = fitter.fit_penalised(x, Z, w=scale * w, **kw)
        np.testing.assert_allclose(other.params, base.params, atol=1e-9)
    k = 1 + np.arange(x.size) % 3
    weighted = fitter.fit_penalised(x, Z, w=k, **kw)
    counted = fitter.fit_penalised(x, Z, n=k, **kw)
    np.testing.assert_allclose(weighted.params, counted.params, atol=1e-9)
    ones = fitter.fit_penalised(x, Z, w=1.0, **kw)
    plain = fitter.fit_penalised(x, Z, **kw)
    np.testing.assert_allclose(ones.params, plain.params, atol=1e-9)
    with pytest.raises(WeightedRefusal):
        base.aic()
    with pytest.raises(ValueError, match="fitted with a penalty"):
        base.standard_errors()


def test_weighted_parametric_round_trip_and_data(data):
    import surpyval as sp
    from surpyval.univariate.regression import ParametricRegressionModel

    x, Z, c, w = data["x"], data["Z"], data["c"], data["w"]
    model = WeibullPH.fit_penalised(x, Z, c=c, w=w, penalizer=0.05)
    d = model.to_dict()
    assert d["schema"] == 4 and d["weighted"] is True
    back = ParametricRegressionModel.from_dict(d)
    np.testing.assert_allclose(back.params, model.params)
    with pytest.raises(WeightedRefusal):
        back.bic()
    held = sp.SurpyvalData(x=x, c=c, Z=Z, w=w)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from_data = WeibullPH.fit_penalised(held, penalizer=0.05)
    np.testing.assert_allclose(from_data.params, model.params, atol=1e-9)


def test_entry_at_zero_is_verified(data):
    # A delayed entry at 0 (the lower end of the Weibull's support) is no
    # truncation; its term's second derivatives used to be nan, which
    # left the fit "unverified".
    model = WeibullPH.fit_penalised(
        data["x"], data["Z"], c=data["c"], tl=data["tl"], penalizer=0.05
    )
    assert model.maximum == "verified"
