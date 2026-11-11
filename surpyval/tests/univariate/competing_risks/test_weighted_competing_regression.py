"""The competing-risks regressions with sampling weights ``w``: the
cause-specific Cox model (``CompetingRisksProportionalHazards``,
``model="Cox"``) and the Fine-Gray model (``FineGray`` and
``model="Fine-Gray"``).

The reference is R 4.6.2, ``survival`` 3.8.12 and ``cmprsk`` 2.2.12
(``scripts/reference/reference_r_weighted_competing_regression.R``), on
``_helpers.weighted_reference_data()`` without its late entries, a failure
of cause 2 where its row number from 0 is a multiple of 3 and of cause 1
otherwise:

- each cause's Cox model is ``coxph(Surv(x, cause == k) ~ z1 + z2,
  weights = w, robust = TRUE)``, both tie methods;
- ``cmprsk::crr`` takes no weights, so the Fine-Gray model is checked
  against ``coxph(Surv(start, stop, status) ~ z1 + z2, weights = wt,
  cluster = id, ties = "breslow")`` on its counting-process data, built
  with the censoring distribution ``G`` the weighted Kaplan-Meier
  (``survfit(..., weights = w)``); the script first checks that the same
  construction with ``w = 1`` gives ``crr``'s coefficients. ``coxph``'s
  robust standard errors hold ``G`` fixed; SurPyval's add the term for
  ``G`` (so ``w = 1`` is ``crr``'s variance), and are checked against
  ``coxph`` with that term taken out.

Tolerances: ``1e-8`` relative on the coefficients (both fits stop on
Newton steps of at most ``1e-10`` standard errors), ``1e-6`` relative on
the robust standard errors.
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import surpyval as sp
from surpyval.tests._helpers import weighted_reference_data
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
    FineGray,
)
from surpyval.univariate.competing_risks.regression.fine_gray_diagnostics import (  # noqa: E501
    fine_gray_zph_test,
)
from surpyval.utils.weights import WeightedRefusal

#: (coefficients, robust standard errors) of each cause, by tie method
R_COX = {
    "efron": {
        1: (
            [0.926516046488202, 0.929922544241924],
            [0.561865080693506, 0.281710203217765],
        ),
        2: (
            [1.53715698925759, 1.21364348675782],
            [0.776876006439384, 0.56156258987494],
        ),
    },
    "breslow": {
        1: (
            [0.929737352362786, 0.931388486795134],
            [0.561344543155616, 0.27884139722991],
        ),
        2: (
            [1.53715698925759, 1.21364348675782],
            [0.776876006439384, 0.56156258987494],
        ),
    },
}
R_FINE_GRAY = {
    1: (
        [0.529226126677845, 0.646138131151132],
        [0.533243953437484, 0.263885869443378],
    ),
    2: (
        [1.50642394786958, 0.715317567392448],
        [0.840965896821553, 0.460781576991221],
    ),
}
COEF = {"rtol": 1e-8}
SE = {"rtol": 1e-6}


@pytest.fixture(scope="module")
def data():
    d = weighted_reference_data()
    rows = np.arange(d["x"].size)
    e = np.array(
        [
            None if censored else (2 if i % 3 == 0 else 1)
            for i, censored in zip(rows, d["c"])
        ],
        dtype=object,
    )
    return {"x": d["x"], "Z": d["Z"], "e": e, "w": d["w"]}


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kwargs)


def _cr(data, w, model="Cox", **kwargs):
    return _quiet(
        CompetingRisksProportionalHazards.fit,
        data["x"],
        data["Z"],
        data["e"],
        w=w,
        model=model,
        **kwargs,
    )


# -- the R references -----------------------------------------------------


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_cause_specific_cox_matches_coxph(data, ties):
    model = _cr(data, data["w"], tie_method=ties)
    se = model.standard_errors().reshape(2, 2)
    for cause, (coef, ref_se) in R_COX[ties].items():
        row = model.event_idx_map[cause]
        assert_allclose(model.betas[row], coef, **COEF)
        assert_allclose(se[row], ref_se, **SE)


@pytest.mark.parametrize("cause", [1, 2])
def test_fine_gray_matches_coxph_on_its_counting_process_data(
    data, cause, monkeypatch
):
    """The coefficients are ``coxph``'s. Its robust standard errors hold
    the censoring distribution ``G`` fixed, so they are checked against
    the fit with the censoring term taken out; the term itself is checked
    against its definition with weights in ``test_fine_gray.py``."""
    from surpyval.univariate.competing_risks.regression import fine_gray

    coef, se = R_FINE_GRAY[cause]
    args = (data["x"], data["Z"], data["e"])
    model = _quiet(FineGray.fit, *args, event=cause, w=data["w"])
    assert_allclose(model.beta, coef, **COEF)
    # The same coefficients through the competing-risks model, and the
    # same covariance
    both = _cr(data, data["w"], model="Fine-Gray")
    row = both.event_idx_map[cause]
    assert_allclose(both.betas[row], coef, **COEF)
    assert_allclose(
        both.standard_errors().reshape(2, 2)[row],
        model.standard_errors(),
        rtol=1e-10,
    )
    # With G held fixed: coxph's
    monkeypatch.setattr(
        fine_gray,
        "_censoring_residuals",
        lambda Z, *rest: np.zeros(Z.shape),
    )
    fixed = _quiet(FineGray.fit, *args, event=cause, w=data["w"])
    assert_allclose(fixed.standard_errors(), se, **SE)
    # The term for G moves them, here by under 1%
    assert not np.allclose(
        model.standard_errors(), fixed.standard_errors(), rtol=1e-4
    )


# -- the properties of a weighted fit --------------------------------------


@pytest.mark.parametrize("model", ["Cox", "Fine-Gray"])
def test_unit_weights_give_the_unweighted_estimates(data, model):
    ones = _cr(data, np.ones(data["x"].size), model=model)
    plain = _cr(data, None, model=model)
    assert ones.weighted and not plain.weighted
    assert_allclose(ones.betas, plain.betas, rtol=1e-10)
    if model == "Fine-Gray":
        # Both are Fine and Gray's robust covariance, the censoring term
        # included
        assert_allclose(ones.covariance(), plain.covariance(), rtol=1e-12)
    else:
        # The covariance is the robust one, not the model-based one
        assert not np.allclose(ones.covariance(), plain.covariance())


@pytest.mark.parametrize("model", ["Cox", "Fine-Gray"])
@pytest.mark.parametrize("scale", [0.01, 7.3])
def test_weight_scale_changes_nothing(data, model, scale):
    base = _cr(data, data["w"], model=model)
    scaled = _cr(data, data["w"] * scale, model=model)
    assert_allclose(scaled.betas, base.betas, rtol=1e-9)
    assert_allclose(scaled.covariance(), base.covariance(), rtol=1e-7)
    assert_allclose(
        scaled.cif([2.0, 5.0], [0.5, 0.0], 1),
        base.cif([2.0, 5.0], [0.5, 0.0], 1),
        rtol=1e-9,
    )


def test_integer_weights_give_the_counted_fine_gray_exactly(data):
    """Whole-number weights count each row that many times, in the risk
    sets and in the censoring distribution ``G`` too, so the Fine-Gray
    estimates, baseline and pseudo-likelihood are the counted fit's to the
    last digit (R's ``finegray``, whose ``G`` ignores the weights, would
    not give this). The covariances differ by design: robust against
    model-based, one unit of weight 2 against two units."""
    k = data["x"].size
    w = np.where(np.arange(k) % 2 == 0, 2, 1)
    args = (data["x"], data["Z"], data["e"])
    weighted = _quiet(FineGray.fit, *args, event=1, w=w.astype(float))
    counted = _quiet(FineGray.fit, *args, n=w, event=1)
    assert_array_equal(weighted.beta, counted.beta)
    assert_array_equal(weighted._cumhaz, counted._cumhaz)
    assert weighted.neg_ll() == counted.neg_ll()


def test_integer_weights_give_the_counted_cause_specific_cox(data):
    """Under Breslow's ties (Efron's takes tied deaths at their mean
    weight, R's convention, which repeated rows do not)."""
    k = data["x"].size
    w = np.where(np.arange(k) % 2 == 0, 2, 1)
    args = (data["x"], data["Z"], data["e"])
    fit = CompetingRisksProportionalHazards.fit
    weighted = _quiet(fit, *args, w=w.astype(float), tie_method="breslow")
    counted = _quiet(fit, *args, n=w, tie_method="breslow")
    assert_allclose(weighted.betas, counted.betas, rtol=1e-8)
    assert_allclose(weighted.H0_e, counted.H0_e, rtol=1e-8)
    assert_allclose(weighted.neg_ll(), counted.neg_ll(), rtol=1e-12)


def test_robust_covariance_clusters_by_unit(data):
    """A count ``n = 2`` is two units, each its own cluster; the same row
    with weight 2 is one unit standing for two: the same estimates, and a
    robust variance of the weighted row's score twice as large in that
    unit's term. With every count 2 the meat doubles and the bread
    halves, so the counted fit's robust covariance is half the singly
    weighted one's."""
    k = data["x"].size
    args = (data["x"], data["Z"], data["e"])
    once = _quiet(FineGray.fit, *args, event=1, w=data["w"])
    twice = _quiet(FineGray.fit, *args, event=1, n=np.full(k, 2), w=data["w"])
    assert_allclose(twice.beta, once.beta, rtol=1e-10)
    assert_allclose(twice.covariance(), once.covariance() / 2, rtol=1e-8)


def test_row_order_does_not_matter(data):
    order = np.random.default_rng(5).permutation(data["x"].size)
    shuffled = {key: value[order] for key, value in data.items()}
    for model in ("Cox", "Fine-Gray"):
        base = _cr(data, data["w"], model=model)
        moved = _cr(shuffled, shuffled["w"], model=model)
        assert_allclose(moved.betas, base.betas, rtol=1e-9)
        assert_allclose(moved.covariance(), base.covariance(), rtol=1e-8)


def test_effective_sample_size_is_shown_and_warned(data):
    model = _cr(data, data["w"], model="Fine-Gray")
    w = data["w"]
    assert model.effective_n == pytest.approx(w.sum() ** 2 / (w**2).sum())
    assert "effective sample size" in repr(model)
    fg = _quiet(FineGray.fit, data["x"], data["Z"], data["e"], event=1, w=w)
    assert fg.effective_n == pytest.approx(model.effective_n)
    assert "effective sample size" in repr(fg)
    # One warning, not one per cause, for a few heavy units
    heavy = np.ones(data["x"].size)
    heavy[:2] = 60.0
    for model_kind in ("Cox", "Fine-Gray"):
        with pytest.warns(UserWarning, match="effective sample size") as rec:
            CompetingRisksProportionalHazards.fit(
                data["x"], data["Z"], data["e"], w=heavy, model=model_kind
            )
        said = [r for r in rec if "effective sample size" in str(r.message)]
        assert len(said) == 1


# -- refusals ---------------------------------------------------------------


@pytest.mark.parametrize("model", ["Cox", "Fine-Gray"])
def test_information_criteria_and_diagnostics_refuse(data, model):
    fitted = _cr(data, data["w"], model=model)
    for criterion in ("aic", "bic", "aic_c"):
        with pytest.raises(WeightedRefusal, match="pseudo-likelihood"):
            getattr(fitted, criterion)()
    with pytest.raises(WeightedRefusal, match="weights"):
        fitted.diagnose()


def test_fine_gray_refusals(data):
    model = _quiet(
        FineGray.fit, data["x"], data["Z"], data["e"], event=1, w=data["w"]
    )
    with pytest.raises(WeightedRefusal, match="pseudo-likelihood"):
        model.aic()
    for call in (model.diagnose, lambda: fine_gray_zph_test(model)):
        with pytest.raises(WeightedRefusal, match="Use the robust Wald"):
            call()


def test_exact_ties_and_penalised_fits_refuse_weights(data):
    args = (data["x"], data["Z"], data["e"])
    with pytest.raises(WeightedRefusal, match="tie_method='exact'"):
        CompetingRisksProportionalHazards.fit(
            *args, w=data["w"], tie_method="exact"
        )
    with pytest.raises(ValueError, match="does not take weights"):
        CompetingRisksProportionalHazards.fit_penalised(*args, w=data["w"])


# -- saving, DataFrames and SurpyvalData --------------------------------


@pytest.mark.parametrize("model", ["Cox", "Fine-Gray"])
def test_round_trip_is_schema_4_and_keeps_the_robust_covariance(data, model):
    fitted = _cr(data, data["w"], model=model)
    d = fitted.to_dict()
    assert d["schema"] == 4 and d["weighted"] is True
    restored = sp.from_dict(json.loads(json.dumps(d, allow_nan=False)))
    for again in (restored, pickle.loads(pickle.dumps(fitted))):
        assert again.weighted
        assert again.effective_n == pytest.approx(fitted.effective_n)
        assert_allclose(again.covariance(), fitted.covariance(), rtol=1e-12)
        with pytest.raises(WeightedRefusal):
            again.aic()
    # An unweighted model keeps its schema
    assert _cr(data, None, model=model).to_dict()["schema"] < 3


def test_fine_gray_round_trip(data):
    model = _quiet(
        FineGray.fit, data["x"], data["Z"], data["e"], event=2, w=data["w"]
    )
    d = model.to_dict()
    assert d["schema"] == 4
    restored = sp.from_dict(json.loads(json.dumps(d)))
    assert restored.weighted
    assert_allclose(restored.covariance(), model.covariance(), rtol=1e-12)
    with pytest.raises(WeightedRefusal):
        restored.bic()


def test_fit_from_df_takes_w_col_and_drops_weights_with_rows(data):
    df = pd.DataFrame(
        {
            "x": data["x"],
            "cause": data["e"],
            "z1": data["Z"][:, 0],
            "z2": data["Z"][:, 1],
            "w": data["w"],
        }
    )
    df.loc[3, "z2"] = np.nan  # dropped, with its weight
    keep = np.arange(len(df)) != 3
    for model in ("Cox", "Fine-Gray"):
        from_df = _quiet(
            CompetingRisksProportionalHazards.fit_from_df,
            df,
            "x",
            "cause",
            ["z1", "z2"],
            w_col="w",
            model=model,
        )
        arrays = _quiet(
            CompetingRisksProportionalHazards.fit,
            data["x"][keep],
            data["Z"][keep],
            data["e"][keep],
            w=data["w"][keep],
            model=model,
        )
        assert_array_equal(from_df.betas, arrays.betas)
        assert_array_equal(from_df.covariance(), arrays.covariance())
    fg = _quiet(
        FineGray.fit_from_df,
        df,
        "x",
        "cause",
        ["z1", "z2"],
        w_col="w",
        event=1,
    )
    fg_arrays = _quiet(
        FineGray.fit,
        data["x"][keep],
        data["Z"][keep],
        data["e"][keep],
        event=1,
        w=data["w"][keep],
    )
    assert_array_equal(fg.beta, fg_arrays.beta)
    assert_array_equal(fg.covariance(), fg_arrays.covariance())


def test_fit_takes_weighted_surpyval_data(data):
    c = np.array([1 if v is None else 0 for v in data["e"]])
    weighted = sp.SurpyvalData(
        data["x"], c, w=data["w"], Z=data["Z"], e=data["e"]
    )
    from_data = _quiet(CompetingRisksProportionalHazards.fit, weighted)
    arrays = _cr(data, data["w"])
    assert from_data.weighted
    assert_array_equal(from_data.betas, arrays.betas)
    fg = _quiet(FineGray.fit, weighted, event=1)
    assert fg.weighted
