"""Fine-Gray centres its covariates (#463), as CoxPH does (#459).

The subdistribution partial likelihood sees the covariates only through
their differences within a (weighted) risk set, so adding a constant to a
column changes neither the coefficients nor any prediction. The fit used
to work on the raw values: a column far from 0 overflowed ``exp(beta'Z)``
and the fit failed with ``LinAlgError: SVD did not converge``, after raw
numpy overflow warnings (principle 22). It now always runs on centred
covariates; by default the baseline is reported at ``Z = 0``, as before,
and refused where that over- or underflows, and ``center=True`` keeps it
at the covariate means.
"""

import json
import warnings

import numpy as np
import pytest

import surpyval
from surpyval.serialisation import required_schema
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
    FineGray,
)

OFFSETS = [0.0, 300.0, 2000.0, 1e5]
TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 15.0])
QUERY = np.array([[0.3, 1.0], [-1.2, 0.0]])


def _data(size=200, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=size)
    g = rng.integers(0, 2, size).astype(float)
    t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * z - 0.4 * g)))
    t_b = rng.exponential(1 / 0.05, size)
    t_c = rng.uniform(0, 20, size)
    x = np.minimum(np.minimum(t_a, t_b), t_c).round(3)
    e = np.where(
        t_c < np.minimum(t_a, t_b), None, np.where(t_a < t_b, "a", "b")
    )
    return x, np.column_stack([z, g]), e


def _shift(offset):
    return np.array([offset, 0.0])


def _fit_quietly(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fit(*args, **kwargs)


def _same(model, ref, s):
    np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(model.se, ref.se, rtol=1e-6)
    np.testing.assert_allclose(model.p_values, ref.p_values, rtol=1e-6)
    np.testing.assert_allclose(model._neg_ll, ref._neg_ll, rtol=1e-9)
    for zq in QUERY:
        for fn in ("cif", "sf"):
            got = getattr(model, fn)(TIMES, zq + s)
            assert np.all(np.isfinite(got)), fn
            np.testing.assert_allclose(
                got, getattr(ref, fn)(TIMES, zq), rtol=1e-7, err_msg=fn
            )


@pytest.mark.parametrize("offset", OFFSETS)
def test_offset_leaves_the_fit_and_predictions_unchanged(offset):
    x, Z, e = _data()
    ref = FineGray.fit(x, Z, e, event="a", center=True)
    s = _shift(offset)
    model = _fit_quietly(FineGray.fit, x, Z + s, e, event="a", center=True)
    _same(model, ref, s)
    np.testing.assert_allclose(model.center, ref.center + s, rtol=1e-12)
    np.testing.assert_allclose(model.phi(QUERY + s), ref.phi(QUERY), rtol=1e-7)


@pytest.mark.parametrize("offset", [0.0, 300.0])
def test_by_default_the_baseline_is_at_zero(offset):
    x, Z, e = _data()
    s = _shift(offset)
    centred = FineGray.fit(x, Z + s, e, event="a", center=True)
    model = _fit_quietly(FineGray.fit, x, Z + s, e, event="a")
    np.testing.assert_array_equal(model.center, [0.0, 0.0])
    _same(model, centred, 0.0)
    np.testing.assert_allclose(
        model._cumhaz,
        centred._cumhaz * np.exp(-centred.center @ centred.beta),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        model.phi(QUERY), np.exp(QUERY @ model.beta), rtol=1e-12
    )


@pytest.mark.parametrize("offset", [2000.0, 1e5])
def test_by_default_a_baseline_that_cannot_be_represented_is_refused(offset):
    x, Z, e = _data()
    with pytest.raises(ValueError, match="center=True"):
        FineGray.fit(x, Z + _shift(offset), e, event="a")


def test_the_baseline_is_that_of_a_unit_at_the_center():
    x, Z, e = _data()
    n = np.random.default_rng(1).integers(1, 4, x.size)
    model = FineGray.fit(x, Z, e, n=n, event="b", center=True)
    np.testing.assert_allclose(model.center, np.average(Z, axis=0, weights=n))
    np.testing.assert_allclose(model.phi(model.center), 1.0)
    np.testing.assert_allclose(
        -np.log(model.sf(model._times, model.center)), model._cumhaz
    )


def test_save_and_load():
    x, Z, e = _data()
    s = _shift(2000.0)
    model = FineGray.fit(x, Z + s, e, event="a", center=True)
    d = json.loads(json.dumps(model.to_dict()))
    # A reader that ignored the centre would read the baseline as at 0.
    assert d["schema"] == 2 == required_schema(d)
    back = surpyval.from_dict(d)
    np.testing.assert_array_equal(back.center, model.center)
    np.testing.assert_array_equal(
        back.cif(TIMES, QUERY[0] + s), model.cif(TIMES, QUERY[0] + s)
    )


def test_the_default_dict_is_the_layout_of_before():
    x, Z, e = _data()
    s = _shift(300.0)
    model = FineGray.fit(x, Z + s, e, event="a")
    d = json.loads(json.dumps(model.to_dict()))
    assert "center" not in d
    assert d["schema"] == 1 == required_schema(d)
    back = surpyval.from_dict(d)
    np.testing.assert_array_equal(back.center, [0.0, 0.0])
    np.testing.assert_array_equal(
        back.cif(TIMES, QUERY[0] + s), model.cif(TIMES, QUERY[0] + s)
    )


def test_a_center_of_the_wrong_length_is_refused():
    x, Z, e = _data()
    d = FineGray.fit(x, Z, e, event="a", center=True).to_dict()
    d["center"] = [1.0]
    with pytest.raises(ValueError, match="'center' has 1 value"):
        surpyval.from_dict(d)


@pytest.mark.parametrize("offset, center", [(1e5, True), (300.0, False)])
def test_competing_risks_fine_gray(offset, center):
    x, Z, e = _data()
    s = _shift(offset)
    ref = CompetingRisksProportionalHazards.fit(x, Z, e, model="Fine-Gray")
    model = _fit_quietly(
        CompetingRisksProportionalHazards.fit,
        x,
        Z + s,
        e,
        model="Fine-Gray",
        center=center,
    )
    np.testing.assert_allclose(model.betas, ref.betas, rtol=1e-7)
    np.testing.assert_allclose(
        model.center, np.mean(Z + s, axis=0) if center else [0.0, 0.0]
    )
    for ev in ("a", "b"):
        for fn in ("cif", "sf", "ff", "Hf"):
            got = (
                model.cif(TIMES, QUERY[0] + s, ev)
                if fn == "cif"
                else getattr(model, fn)(TIMES, QUERY[0] + s, event=ev)
            )
            want = (
                ref.cif(TIMES, QUERY[0], ev)
                if fn == "cif"
                else getattr(ref, fn)(TIMES, QUERY[0], event=ev)
            )
            np.testing.assert_allclose(got, want, rtol=1e-7, err_msg=fn)

    # Saved with its centre, on the model and each cause's model, only
    # when the baselines are there.
    d = json.loads(json.dumps(model.to_dict()))
    assert d["schema"] == required_schema(d) == (2 if center else 1)
    back = surpyval.from_dict(d)
    np.testing.assert_array_equal(back.center, model.center)
    np.testing.assert_array_equal(
        back.cif(TIMES, QUERY[0] + s, "b"), model.cif(TIMES, QUERY[0] + s, "b")
    )


def test_a_competing_risks_dict_of_0_21_0_loads():
    # SurPyval 0.21.0 wrote a zero "center" for a Fine-Gray model, and no
    # "center" on the causes' models: the baselines at Z = 0, as now.
    x, Z, e = _data()
    model = CompetingRisksProportionalHazards.fit(x, Z, e, model="Fine-Gray")
    old = model.to_dict()
    assert "center" not in old
    for layout in (old, dict(old, center=[0.0, 0.0])):
        back = surpyval.from_dict(layout)
        np.testing.assert_array_equal(back.center, [0.0, 0.0])
        for ev in ("a", "b"):
            np.testing.assert_array_equal(
                back.cif(TIMES, QUERY[1], ev), model.cif(TIMES, QUERY[1], ev)
            )
