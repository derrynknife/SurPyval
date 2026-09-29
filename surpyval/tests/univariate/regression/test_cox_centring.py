"""CoxPH centres its covariates (#459).

The partial likelihood sees the covariates only through their
differences within a risk set, so adding a constant to a column changes
neither the coefficients nor any prediction (principle 6). The fit used
to work on the raw values, and a column far from 0 (a year, a date as a
day count) overflowed ``exp(beta'Z)``: a wrong coefficient, a NaN
survival, a false "monotone partial likelihood" warning and raw numpy
warnings (principle 22).
"""

import json
import warnings

import numpy as np
import pytest

import surpyval
from surpyval import CoxPH
from surpyval.serialisation import required_schema
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
)
from surpyval.univariate.regression import StepSchedule

OFFSETS = [0.0, 300.0, 2000.0, 1e5]
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


def _fit_quietly(fit, *args, **kwargs):
    # No warning of any kind: before the fix a large offset gave a
    # monotone-likelihood warning and raw overflow warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fit(*args, **kwargs)


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("tie_method", ["efron", "breslow"])
def test_offset_leaves_the_fit_and_predictions_unchanged(offset, tie_method):
    x, Z, c = _data()
    ref = CoxPH.fit(x, Z, c=c, tie_method=tie_method)
    s = _shift(offset)
    model = _fit_quietly(CoxPH.fit, x, Z + s, c=c, tie_method=tie_method)

    np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(model.p_values, ref.p_values, rtol=1e-6)
    np.testing.assert_allclose(model.center, ref.center + s, rtol=1e-12)
    for zq in ([0.3, 1.0], [-1.2, 0.0]):
        zq = np.array(zq)
        for fn in ("sf", "Hf", "hf", "ff"):
            got = getattr(model, fn)(TIMES, zq + s)
            want = getattr(ref, fn)(TIMES, zq)
            assert np.all(np.isfinite(got)), fn
            np.testing.assert_allclose(got, want, rtol=1e-7, err_msg=fn)
    np.testing.assert_allclose(
        model.phi([[0.3, 1.0]] + s), ref.phi([[0.3, 1.0]]), rtol=1e-7
    )


@pytest.mark.parametrize("tie_method", ["exact", "kp"])
def test_offset_with_the_exact_tie_methods(tie_method):
    x, Z, c = _data(size=60, seed=3)
    x = np.ceil(x)  # ties for the methods to handle
    ref = CoxPH.fit(x, Z, c=c, tie_method=tie_method)
    s = _shift(2000.0)
    model = _fit_quietly(CoxPH.fit, x, Z + s, c=c, tie_method=tie_method)
    np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        model.sf(TIMES, np.array([0.3, 1.0]) + s),
        ref.sf(TIMES, [0.3, 1.0]),
        rtol=1e-7,
    )


def test_the_baseline_is_that_of_a_unit_at_the_center():
    x, Z, c = _data()
    n = np.random.default_rng(0).integers(1, 4, x.size)
    model = CoxPH.fit(x, Z + _shift(2000.0), c=c, n=n)
    # The n-weighted column means of the fitted rows.
    np.testing.assert_allclose(
        model.center, np.average(Z + _shift(2000.0), axis=0, weights=n)
    )
    np.testing.assert_allclose(model.Hf(model.x, model.center), model.H0)
    np.testing.assert_allclose(model.phi(model.center), 1.0)
    # For data where exp(beta'Z) is representable the baseline at Z = 0 is
    # recovered from it by the usual factor.
    ref = CoxPH.fit(x, Z, c=c, n=n)
    np.testing.assert_allclose(
        ref.Hf(ref.x, [0.0, 0.0]),
        ref.H0 * np.exp(-ref.center @ ref.beta),
    )


def test_delayed_entry_and_residuals():
    x, Z, c = _data()
    tl = 0.2 * x
    s = _shift(2000.0)
    ref = CoxPH.fit(x, Z, c=c, tl=tl)
    model = _fit_quietly(CoxPH.fit, x, Z + s, c=c, tl=tl)
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
            model.check_ph()["global"]["statistic"],
            ref.check_ph()["global"]["statistic"],
            rtol=1e-6,
        )


def test_strata():
    x, Z, c = _data()
    strata = np.arange(x.size) % 3
    s = _shift(2000.0)
    ref = CoxPH.fit(x, Z, c=c, strata=strata)
    model = _fit_quietly(CoxPH.fit, x, Z + s, c=c, strata=strata)
    np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-8)
    # One centre, the mean over every stratum's rows.
    np.testing.assert_allclose(model.center, np.mean(Z + s, axis=0))
    for k in range(3):
        np.testing.assert_allclose(
            model.sf(TIMES, [0.5, 1.0] + s, stratum=k),
            ref.sf(TIMES, [0.5, 1.0], stratum=k),
            rtol=1e-7,
        )


def test_fit_tvc_and_its_predictions():
    i = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 8]
    xl = [0, 2, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 4]
    xr = [2, 5, 3, 1, 4, 6, 3, 7, 2, 8, 5, 4, 9]
    c = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    Z = np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=float)
    ref = CoxPH.fit_tvc(i, xl, xr, c, Z)
    model = _fit_quietly(CoxPH.fit_tvc, i, xl, xr, c, Z + 2000.0)
    np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-8)
    # R's agreg centres on the mean of the interval rows.
    np.testing.assert_allclose(model.center, [np.mean(Z) + 2000.0])

    path = StepSchedule.from_changepoints([0, 2], [[0], [1]])
    moved = StepSchedule.from_changepoints([0, 2], [[2000.0], [2001.0]])
    t = [1, 3, 5, 7, 9]
    np.testing.assert_allclose(
        model.sf_tvc(t, moved), ref.sf_tvc(t, path), rtol=1e-8
    )
    np.testing.assert_allclose(
        model.Hf_tvc(t, moved), ref.Hf_tvc(t, path), rtol=1e-8
    )
    np.testing.assert_allclose(
        model.predict_tvc([0, 2], [2, 9], [[2000.0], [2001.0]])[1],
        ref.predict_tvc([0, 2], [2, 9], [[0.0], [1.0]])[1],
        rtol=1e-8,
    )


def test_competing_risks_cox_cif():
    rng = np.random.default_rng(0)
    Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
    t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * Z[:, 0])))
    t_b = rng.exponential(1 / 0.05, 200)
    t_c = rng.uniform(0, 20, 200)
    x = np.minimum(np.minimum(t_a, t_b), t_c).round(3)
    e = np.where(
        t_c < np.minimum(t_a, t_b), None, np.where(t_a < t_b, "a", "b")
    )
    ref = CompetingRisksProportionalHazards.fit(x, Z, e)
    model = _fit_quietly(CompetingRisksProportionalHazards.fit, x, Z + 2000, e)
    np.testing.assert_allclose(model.betas, ref.betas, rtol=1e-8)
    for ev in ("a", "b"):
        np.testing.assert_allclose(
            model.cif(TIMES, [[2001.0]], ev),
            ref.cif(TIMES, [[1.0]], ev),
            rtol=1e-8,
        )
    np.testing.assert_allclose(
        model.sf(TIMES, [[2001.0]]), ref.sf(TIMES, [[1.0]]), rtol=1e-8
    )

    # Saved with its centre (schema 2); a dict from before has none and
    # its baselines are at Z = 0.
    d = json.loads(json.dumps(model.to_dict()))
    assert d["schema"] == 2 == required_schema(d)
    back = surpyval.from_dict(d)
    np.testing.assert_allclose(
        back.cif(TIMES, [[2001.0]], "a"), model.cif(TIMES, [[2001.0]], "a")
    )
    old = ref.to_dict()
    del old["center"]
    old["h0_e"] = (
        np.asarray(ref.h0_e) * np.exp(-ref.betas @ ref.center)[:, None]
    ).tolist()
    np.testing.assert_allclose(
        surpyval.from_dict(old).cif(TIMES, [[1.0]], "a"),
        ref.cif(TIMES, [[1.0]], "a"),
        rtol=1e-12,
    )


def test_save_and_load():
    x, Z, c = _data()
    model = CoxPH.fit(x, Z + _shift(2000.0), c=c)
    d = json.loads(json.dumps(model.to_dict()))
    # A reader that ignored the centre would mispredict, so it is refused
    # by schema-1 readers.
    assert d["schema"] == 2 == required_schema(d)
    back = surpyval.from_dict(d)
    np.testing.assert_array_equal(back.center, model.center)
    zq = np.array([0.3, 1.0]) + _shift(2000.0)
    np.testing.assert_array_equal(back.sf(TIMES, zq), model.sf(TIMES, zq))


def test_a_dict_written_before_centring_loads_as_baseline_at_zero():
    x, Z, c = _data()
    model = CoxPH.fit(x, Z, c=c)
    old = model.to_dict()
    # What a SurPyval that did not centre wrote: the baseline at Z = 0 and
    # no "center".
    del old["center"]
    factor = np.exp(-model.center @ model.beta)
    old["h0"] = (np.asarray(model.h0) * factor).tolist()
    old["H0"] = (np.asarray(model.H0) * factor).tolist()
    old["schema"] = 1
    back = surpyval.from_dict(old)
    np.testing.assert_array_equal(back.center, [0.0, 0.0])
    for zq in ([0.3, 1.0], [-1.0, 0.0]):
        np.testing.assert_allclose(
            back.sf(TIMES, zq), model.sf(TIMES, zq), rtol=1e-12
        )


def test_a_zero_center_stays_schema_1():
    x, _, c = _data()
    Z = np.tile([-1.0, 1.0], x.size // 2)[:, None]  # mean exactly 0
    d = CoxPH.fit(x, Z, c=c).to_dict()
    assert d["center"] == [0.0]
    assert d["schema"] == 1


def test_a_constant_column_far_from_zero_is_still_refused():
    # Centred, a constant column is (nearly) zero; the identifiability
    # check still judges it on the raw values.
    x, Z, c = _data()
    Z = np.column_stack([Z[:, 0], np.full(x.size, 2000.0)])
    with pytest.raises(ValueError, match=r"column\(s\) \[1\]"):
        CoxPH.fit(x, Z, c=c)
