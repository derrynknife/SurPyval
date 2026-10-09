"""CoxPH centres its covariates (#459); ``center=`` says where the baseline
is reported (#463).

The partial likelihood sees the covariates only through their
differences within a risk set, so adding a constant to a column changes
neither the coefficients nor any prediction (principle 6). The fit used
to work on the raw values, and a column far from 0 (a year, a date as a
day count) overflowed ``exp(beta'Z)``: a wrong coefficient, a NaN
survival, a false "monotone partial likelihood" warning and raw numpy
warnings (principle 22). The fit always runs on centred covariates now.
By default the baseline is reported at ``Z = 0`` (R's ``basehaz(fit,
centered = FALSE)``), as before #459, and the fit refuses covariates so
far from 0 that the baseline there over- or underflows; ``center=True``
keeps it at the covariate means, ``model.center`` (R's ``basehaz(fit)``).
"""

import json
import warnings

import numpy as np
import pytest

import surpyval
from surpyval import CoxPH
from surpyval.serialisation import required_schema
from surpyval.tests._helpers import no_warnings
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
)
from surpyval.univariate.regression import StepSchedule

OFFSETS = [0.0, 300.0, 2000.0, 1e5]
# Offsets at which the baseline at Z = 0 is representable (beta'center of
# about 220 at 300; 1600 at 2000 underflows).
AT_ZERO = [0.0, 300.0]
TOO_FAR = [2000.0, 1e5]
TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 15.0])


def _data(size=200, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=size)
    g = rng.integers(0, 2, size).astype(float)
    t = rng.exponential(1 / (0.1 * np.exp(0.8 * z - 0.5 * g)))
    cens = rng.uniform(0, 20, size)
    x = np.round(np.minimum(t, cens), 2)
    c = (cens < t).astype(int)
    return x, np.column_stack([z, g]), c


def _shift(offset):
    # The first column moved, the second (binary) left alone.
    return np.array([offset, 0.0])


def _same_predictions(model, ref, s, rtol=1e-7):
    for zq in ([0.3, 1.0], [-1.2, 0.0]):
        zq = np.array(zq)
        for fn in ("sf", "Hf", "hf", "ff"):
            got = getattr(model, fn)(TIMES, zq + s)
            want = getattr(ref, fn)(TIMES, zq)
            assert np.all(np.isfinite(got)), fn
            np.testing.assert_allclose(got, want, rtol=rtol, err_msg=fn)


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("tie_method", ["efron", "breslow"])
def test_offset_leaves_the_fit_and_predictions_unchanged(offset, tie_method):
    x, Z, c = _data()
    ref = CoxPH.fit(x, Z, c=c, tie_method=tie_method, center=True)
    s = _shift(offset)
    model = no_warnings(
        CoxPH.fit, x, Z + s, c=c, tie_method=tie_method, center=True
    )

    np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(model.p_values, ref.p_values, rtol=1e-6)
    np.testing.assert_allclose(model.center, ref.center + s, rtol=1e-12)
    # At the means, the baseline is the same whatever the offset.
    np.testing.assert_allclose(model.H0, ref.H0, rtol=1e-7)
    _same_predictions(model, ref, s)
    np.testing.assert_allclose(
        model.phi([[0.3, 1.0]] + s), ref.phi([[0.3, 1.0]]), rtol=1e-7
    )


@pytest.mark.parametrize("offset", AT_ZERO)
@pytest.mark.parametrize("tie_method", ["efron", "breslow"])
def test_by_default_the_baseline_is_at_zero(offset, tie_method):
    x, Z, c = _data()
    s = _shift(offset)
    centred = CoxPH.fit(x, Z + s, c=c, tie_method=tie_method, center=True)
    model = no_warnings(CoxPH.fit, x, Z + s, c=c, tie_method=tie_method)
    np.testing.assert_array_equal(model.center, [0.0, 0.0])
    np.testing.assert_array_equal(model.beta, centred.beta)
    # The baseline at the means moved to Z = 0, R's basehaz(centered=FALSE).
    factor = np.exp(-centred.center @ centred.beta)
    np.testing.assert_allclose(model.H0, centred.H0 * factor, rtol=1e-12)
    np.testing.assert_allclose(model.h0, centred.h0 * factor, rtol=1e-12)
    np.testing.assert_allclose(model.Hf(model.x, [0.0, 0.0]), model.H0)
    # phi is exp(beta'Z) again, and the predictions are the centred
    # model's (they are formed on the log scale, so a baseline of 1e-96
    # and a multiplier of 1e96 do not lose anything).
    np.testing.assert_allclose(
        model.phi([[0.3, 1.0]]), np.exp([0.3, 1.0] @ model.beta)
    )
    _same_predictions(model, centred, 0.0, rtol=1e-10)


@pytest.mark.parametrize("offset", TOO_FAR)
def test_by_default_a_baseline_that_cannot_be_represented_is_refused(offset):
    x, Z, c = _data()
    with pytest.raises(ValueError, match="center=True"):
        CoxPH.fit(x, Z + _shift(offset), c=c)
    with pytest.raises(ValueError, match="cannot be represented"):
        CoxPH.fit(x, Z + _shift(offset), c=c, strata=np.arange(x.size) % 2)


@pytest.mark.parametrize("tie_method", ["exact", "kp"])
def test_offset_with_the_exact_tie_methods(tie_method):
    x, Z, c = _data(size=60, seed=3)
    x = np.ceil(x)  # ties for the methods to handle
    ref = CoxPH.fit(x, Z, c=c, tie_method=tie_method)
    for offset, center in ((2000.0, True), (300.0, False)):
        s = _shift(offset)
        model = no_warnings(
            CoxPH.fit, x, Z + s, c=c, tie_method=tie_method, center=center
        )
        np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(
            model.sf(TIMES, np.array([0.3, 1.0]) + s),
            ref.sf(TIMES, [0.3, 1.0]),
            rtol=1e-7,
        )


def test_the_baseline_is_that_of_a_unit_at_the_center():
    x, Z, c = _data()
    n = np.random.default_rng(0).integers(1, 4, x.size)
    model = CoxPH.fit(x, Z + _shift(2000.0), c=c, n=n, center=True)
    # The n-weighted column means of the fitted rows.
    np.testing.assert_allclose(
        model.center, np.average(Z + _shift(2000.0), axis=0, weights=n)
    )
    np.testing.assert_allclose(model.Hf(model.x, model.center), model.H0)
    np.testing.assert_allclose(model.phi(model.center), 1.0)
    assert "Baseline at         : the covariate means" in repr(model)
    # For data where exp(beta'Z) is representable the baseline at Z = 0 is
    # recovered from it by the usual factor, and is the default fit's.
    ref = CoxPH.fit(x, Z, c=c, n=n, center=True)
    at_zero = CoxPH.fit(x, Z, c=c, n=n)
    np.testing.assert_allclose(
        ref.Hf(ref.x, [0.0, 0.0]),
        ref.H0 * np.exp(-ref.center @ ref.beta),
    )
    np.testing.assert_allclose(at_zero.H0, ref.Hf(ref.x, [0.0, 0.0]))
    assert "Baseline at" not in repr(at_zero)


@pytest.mark.parametrize("offset, center", [(2000.0, True), (300.0, False)])
def test_delayed_entry_and_residuals(offset, center):
    x, Z, c = _data()
    tl = 0.2 * x
    s = _shift(offset)
    ref = CoxPH.fit(x, Z, c=c, tl=tl)
    model = no_warnings(CoxPH.fit, x, Z + s, c=c, tl=tl, center=center)
    np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-8)
    np.testing.assert_allclose(
        model.sf(TIMES, [0.5, 1.0] + s), ref.sf(TIMES, [0.5, 1.0]), rtol=1e-7
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for kind in ("martingale", "deviance", "schoenfeld", "score"):
            np.testing.assert_allclose(
                model.compute_residuals(kind),
                ref.compute_residuals(kind),
                rtol=1e-6,
                atol=1e-9,
                err_msg=kind,
            )
        np.testing.assert_allclose(
            model.robust_covariance(), ref.robust_covariance(), rtol=1e-6
        )
        np.testing.assert_allclose(
            model.check_ph().loc["GLOBAL", "statistic"],
            ref.check_ph().loc["GLOBAL", "statistic"],
            rtol=1e-6,
        )


@pytest.mark.parametrize("offset, center", [(2000.0, True), (300.0, False)])
def test_strata(offset, center):
    x, Z, c = _data()
    strata = np.arange(x.size) % 3
    s = _shift(offset)
    ref = CoxPH.fit(x, Z, c=c, strata=strata)
    model = no_warnings(CoxPH.fit, x, Z + s, c=c, strata=strata, center=center)
    np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-8)
    # One centre, the mean over every stratum's rows.
    np.testing.assert_allclose(
        model.center, np.mean(Z + s, axis=0) if center else [0.0, 0.0]
    )
    for k in range(3):
        np.testing.assert_allclose(
            model.sf(TIMES, [0.5, 1.0] + s, stratum=k),
            ref.sf(TIMES, [0.5, 1.0], stratum=k),
            rtol=1e-7,
        )


@pytest.mark.parametrize("offset, center", [(2000.0, True), (30.0, False)])
def test_fit_tvc_and_its_predictions(offset, center):
    i = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 8]
    xl = [0, 2, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 4]
    xr = [2, 5, 3, 1, 4, 6, 3, 7, 2, 8, 5, 4, 9]
    c = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    Z = np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=float)
    ref = CoxPH.fit_tvc(i, xl, xr, c, Z)
    model = no_warnings(CoxPH.fit_tvc, i, xl, xr, c, Z + offset, center=center)
    np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-8)
    # R's agreg centres on the mean of the interval rows.
    np.testing.assert_allclose(
        model.center, [np.mean(Z) + offset] if center else [0.0]
    )

    path = StepSchedule.from_changepoints([0, 2], [[0], [1]])
    moved = StepSchedule.from_changepoints([0, 2], [[offset], [offset + 1.0]])
    t = [1, 3, 5, 7, 9]
    np.testing.assert_allclose(
        model.sf_tvc(t, moved), ref.sf_tvc(t, path), rtol=1e-8
    )
    np.testing.assert_allclose(
        model.Hf_tvc(t, moved), ref.Hf_tvc(t, path), rtol=1e-8
    )
    np.testing.assert_allclose(
        model.predict_tvc([0, 2], [2, 9], [[offset], [offset + 1.0]])[1],
        ref.predict_tvc([0, 2], [2, 9], [[0.0], [1.0]])[1],
        rtol=1e-8,
    )


def _cr_data():
    rng = np.random.default_rng(0)
    Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
    t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * Z[:, 0])))
    t_b = rng.exponential(1 / 0.05, 200)
    t_c = rng.uniform(0, 20, 200)
    x = np.minimum(np.minimum(t_a, t_b), t_c).round(3)
    e = np.where(
        t_c < np.minimum(t_a, t_b), None, np.where(t_a < t_b, "a", "b")
    )
    return x, Z, e


@pytest.mark.parametrize("offset, center", [(2000.0, True), (300.0, False)])
def test_competing_risks_cox_cif(offset, center):
    x, Z, e = _cr_data()
    ref = CompetingRisksProportionalHazards.fit(x, Z, e)
    model = no_warnings(
        CompetingRisksProportionalHazards.fit, x, Z + offset, e, center=center
    )
    np.testing.assert_allclose(model.betas, ref.betas, rtol=1e-8)
    np.testing.assert_allclose(
        model.center, [np.mean(Z) + offset] if center else [0.0]
    )
    for ev in ("a", "b"):
        np.testing.assert_allclose(
            model.cif(TIMES, [[offset + 1.0]], ev),
            ref.cif(TIMES, [[1.0]], ev),
            rtol=1e-8,
        )
    np.testing.assert_allclose(
        model.sf(TIMES, [[offset + 1.0]]), ref.sf(TIMES, [[1.0]]), rtol=1e-8
    )

    # Saved with its centre (schema 2) only when the baselines are there.
    d = json.loads(json.dumps(model.to_dict()))
    assert ("center" in d) == center
    assert d["schema"] == required_schema(d) == (2 if center else 1)
    back = surpyval.from_dict(d)
    np.testing.assert_allclose(
        back.cif(TIMES, [[offset + 1.0]], "a"),
        model.cif(TIMES, [[offset + 1.0]], "a"),
    )


def test_competing_risks_refuses_a_baseline_it_cannot_put_at_zero():
    x, Z, e = _cr_data()
    with pytest.raises(ValueError, match="center=True"):
        CompetingRisksProportionalHazards.fit(x, Z + 2000.0, e)


def test_save_and_load():
    x, Z, c = _data()
    model = CoxPH.fit(x, Z + _shift(2000.0), c=c, center=True)
    d = json.loads(json.dumps(model.to_dict()))
    # A reader that ignored the centre would mispredict, so it is refused
    # by schema-1 readers.
    assert d["schema"] == 2 == required_schema(d)
    back = surpyval.from_dict(d)
    np.testing.assert_array_equal(back.center, model.center)
    zq = np.array([0.3, 1.0]) + _shift(2000.0)
    np.testing.assert_array_equal(back.sf(TIMES, zq), model.sf(TIMES, zq))
    assert repr(back) == repr(model)


def test_the_default_dict_is_the_layout_of_before():
    x, Z, c = _data()
    model = CoxPH.fit(x, Z + _shift(300.0), c=c)
    d = json.loads(json.dumps(model.to_dict()))
    # No "center": the layout (and schema) SurPyval 0.20 and 0.21.0 read,
    # with the baseline at Z = 0.
    assert "center" not in d
    assert d["schema"] == 1 == required_schema(d)
    back = surpyval.from_dict(d)
    np.testing.assert_array_equal(back.center, [0.0, 0.0])
    zq = np.array([0.3, 1.0]) + _shift(300.0)
    np.testing.assert_array_equal(back.sf(TIMES, zq), model.sf(TIMES, zq))


def test_a_dict_written_by_0_21_0_with_a_center_still_loads():
    # SurPyval 0.21.0 (#459) always kept the baseline at the means and
    # saved "center"; such a file reads as the model it was.
    x, Z, c = _data()
    model = CoxPH.fit(x, Z, c=c, center=True)
    d = model.to_dict()
    assert np.any(d["center"])
    back = surpyval.from_dict(d)
    at_zero = CoxPH.fit(x, Z, c=c)
    for zq in ([0.3, 1.0], [-1.0, 0.0]):
        np.testing.assert_allclose(
            back.sf(TIMES, zq), at_zero.sf(TIMES, zq), rtol=1e-12
        )


def test_a_constant_column_far_from_zero_is_still_aliased():
    # Centred, a constant column is (nearly) zero; the identifiability
    # check still judges it on the raw values (#476).
    x, Z, c = _data()
    Z = np.column_stack([Z[:, 0], np.full(x.size, 2000.0)])
    with pytest.warns(UserWarning, match=r"column\(s\) 1 of Z cannot"):
        model = CoxPH.fit(x, Z, c=c)
    assert np.isnan(model.beta[1])
    assert model.beta[0] == pytest.approx(CoxPH.fit(x, Z[:, :1], c=c).beta[0])


def test_refusal_says_the_rows_overflow_not_the_move():
    # Covariates already centred on 0, whose coefficient runs off to
    # beta'Z of 850: the move to Z = 0 is by exp(0), and the refusal said
    # that it over- or underflowed. It is exp(beta'Z) on the rows (#777).
    z = np.linspace(-30.0, 30.0, 61)[::-1, None]
    x = np.arange(1.0, 62.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as info:
            CoxPH.fit(x, z)
    message = str(info.value)
    assert "exp(beta'Z) on the covariates as given over- or underflows" in (
        message
    )
    assert "times that at the means" not in message
    assert "center=True" in message
