"""Aliased coefficients of the proportional-intensity regressions (#502).

A repeated covariate column, or a constant one (the baseline's scale --
the HPP's rate, Duane's ``b``, Crow-AMSAA's ``alpha``, Cox-Lewis's
``alpha`` -- is the intercept), leaves the likelihood flat along a
combination of the parameters. The fits split a repeated column's
coefficient wherever BFGS stopped (HPP: -0.2313 as -0.1156 and -0.1156)
and a constant column took part of the baseline (HPP rate 0.0807 ->
0.0764), silently, with standard errors from a singular information. As
every other regression does since #476, such a coefficient is now
aliased: ``nan``, listed in ``model.aliased``, one warning naming the
column, and the other parameters and the predictions those of the fit
without it.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.recurrent import (
    CoxLewis,
    CrowAMSAA,
    Duane,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)

X = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60, 5, 18, 30, 50, 60.0])
I = np.array([1] * 6 + [2] * 5 + [3] * 5)
C = np.zeros(16, int)
C[[5, 10, 15]] = 1
Z = np.array([0.0] * 6 + [1.0] * 5 + [0.5] * 5)[:, None]

FITS = [
    pytest.param(ProportionalIntensityHPP, {}, id="HPP"),
    pytest.param(ProportionalIntensityNHPP, {"baseline": Duane}, id="Duane"),
    pytest.param(
        ProportionalIntensityNHPP, {"baseline": CrowAMSAA}, id="CrowAMSAA"
    ),
    pytest.param(
        ProportionalIntensityNHPP, {"baseline": CoxLewis}, id="CoxLewis"
    ),
]


def _fit(F, Z_fit, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = F.fit(X, Z_fit, i=I, c=C, **kw)
    return model, caught


@pytest.mark.parametrize("extra", ["repeated", "constant"])
@pytest.mark.parametrize("F, kw", FITS)
def test_undetermined_column_is_aliased(F, kw, extra):
    ref = F.fit(X, Z, i=I, c=C, **kw)
    column = Z if extra == "repeated" else np.ones((16, 1))
    model, caught = _fit(F, np.c_[Z, column], **kw)
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1, messages
    assert messages[0].startswith(
        "Covariate column(s) 1 of Z cannot be estimated"
    )
    assert caught[0].filename == __file__
    np.testing.assert_array_equal(model.aliased, [1])
    assert np.isnan(model.coeffs[1])
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-6)
    np.testing.assert_allclose(model.coeffs[:1], ref.coeffs, rtol=1e-6)
    # Any value of the aliased column predicts as the fit without it.
    for q in ([0.5, 3.0], [1.0, -2.0]):
        np.testing.assert_allclose(
            model.cif([10.0, 50.0], q), ref.cif([10.0, 50.0], q[:1]), rtol=1e-6
        )
        np.testing.assert_allclose(
            model.iif([10.0, 50.0], q), ref.iif([10.0, 50.0], q[:1]), rtol=1e-6
        )
        np.testing.assert_allclose(
            model.cif_cb([10.0, 50.0], q),
            ref.cif_cb([10.0, 50.0], q[:1]),
            rtol=1e-5,
        )
    # Inference on the estimated parameters only: the aliased one is not
    # counted and has no variance.
    se = model.standard_errors()
    assert np.isnan(se[-1])
    np.testing.assert_allclose(se[:-1], ref.standard_errors(), rtol=1e-5)
    assert model.log_likelihood == pytest.approx(ref.log_likelihood)
    assert model.aic() == pytest.approx(ref.aic())
    assert model.bic() == pytest.approx(ref.bic())


def test_issue_example_no_longer_splits_the_coefficient():
    # The issue's table: the repeated column was split -0.1156 / -0.1156
    # (HPP) and the constant one moved the rate 0.08072 -> 0.07635.
    ref = ProportionalIntensityHPP.fit(X, Z, i=I, c=C)
    dup, _ = _fit(ProportionalIntensityHPP, np.c_[Z, Z])
    const, _ = _fit(ProportionalIntensityHPP, np.c_[Z, np.ones(16)])
    assert ref.params[0] == pytest.approx(0.08072, abs=1e-5)
    assert ref.coeffs[0] == pytest.approx(-0.2313, abs=1e-4)
    for model in (dup, const):
        assert model.params[0] == pytest.approx(ref.params[0], rel=1e-6)
        assert model.coeffs[0] == pytest.approx(ref.coeffs[0], rel=1e-6)


def test_constant_column_is_kept_without_a_baseline_scale():
    # A baseline with no scale to absorb it (has_scale False) has no
    # intercept: a constant column is then estimated, not aliased, while
    # a column of zeros still is.
    class Unscaled(type(Duane)):
        pass

    dist = Unscaled()
    dist.has_scale = False
    model, caught = _fit(
        ProportionalIntensityNHPP, np.c_[Z, np.ones(16)], baseline=dist
    )
    assert not [w for w in caught if "cannot be estimated" in str(w.message)]
    assert model.aliased.size == 0
    model, caught = _fit(
        ProportionalIntensityNHPP, np.c_[Z, np.zeros(16)], baseline=dist
    )
    assert len(caught) == 1 and "all zero" in str(caught[0].message)
    np.testing.assert_array_equal(model.aliased, [1])


def test_fit_from_df_names_the_aliased_column():
    df = pd.DataFrame(
        {"x": X, "i": I, "c": C, "load": Z[:, 0], "load2": 2 * Z[:, 0]}
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ProportionalIntensityHPP.fit_from_df(
            df, x_col="x", Z_cols=["load", "load2"], i_col="i", c_col="c"
        )
    assert len(caught) == 1
    assert str(caught[0].message).startswith(
        "Covariate column(s) 1 ('load2') of Z cannot be estimated"
    )
    np.testing.assert_array_equal(model.aliased, [1])


@pytest.mark.parametrize("F, kw", FITS[:2])
def test_aliased_model_round_trips_simulates_and_bootstraps_quietly(F, kw):
    model, _ = _fit(F, np.c_[Z, Z], **kw)
    restored = sp.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_array_equal(restored.aliased, [1])
    np.testing.assert_allclose(
        restored.cif([10.0, 50.0], [0.5, 9.0]),
        model.cif([10.0, 50.0], [0.5, 9.0]),
    )
    with warnings.catch_warnings():
        # The fit warned once; the bootstrap's refits and the simulation
        # do not warn again.
        warnings.simplefilter("error")
        sim = model.time_terminated_simulation(
            60.0, [0.5, 9.0], items=3, random_state=1
        )
        assert np.all(np.isfinite(sim.mcf([10.0])))
        gof = model.cramer_von_mises(n_boot=5, random_state=1)
        assert 0 <= gof.p_value <= 1
