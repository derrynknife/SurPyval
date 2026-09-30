"""The four-parameter Beta's likelihood is unbounded (#385).

With a shape below 1 the density is infinite at a support end, so a
maximum-likelihood fit can run that end onto the smallest (or largest)
observation, where the likelihood has no maximum; the answer then depends
on the data's units. Such a fit warns "No finite maximum" and recommends
maximum product of spacings, whose fit is finite and the same in any units.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.conformance.registry import CASE_BY_NAME

NO_MAXIMUM = "No finite maximum: the Beta4 likelihood is unbounded"


def _fixture(scale=1.0):
    d = CASE_BY_NAME["Beta4"].data()
    return {**d, "x": np.asarray(d["x"], dtype=float) * scale}


def _fit(**kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model = sp.Beta4.fit(**kwargs)
    return model, rec


@pytest.mark.parametrize("scale", [7.3, 1e4])
def test_a_fit_run_onto_an_observation_warns_once(scale):
    # On the fixture times 7.3 the fit ran a onto the smallest value with
    # alpha = 0.18: the likelihood keeps rising there without bound.
    model, rec = _fit(**_fixture(scale))
    found = [w for w in rec if str(w.message).startswith(NO_MAXIMUM)]
    assert len(found) == 1 and len(rec) == 1
    assert found[0].filename == __file__
    assert "how='MPS'" in str(found[0].message)
    assert model.params[0] < 1
    assert model.params[2] == pytest.approx(0.1 * scale, rel=1e-8)


def test_the_likelihood_is_unbounded_at_the_edge():
    # alpha = beta = 0.5 with the ends closing on the extremes: the
    # negative log-likelihood falls without bound.
    d = _fixture()
    data = sp.utils.surpyval_data.SurpyvalData(d["x"], d["c"], d["n"])
    lo, hi = d["x"].min(), d["x"].max()
    nll = [
        float(
            sp.Beta4._neg_ll_func(
                data, 0.5, 0.5, lo - e, hi + e, 0.0, 0.0, 1.0
            )
        )
        for e in (1e-2, 1e-4, 1e-8, 1e-12)
    ]
    assert np.all(np.diff(nll) < -2)


@pytest.mark.parametrize("scale", [1.0, 7.3, 1e-3, 1e4])
def test_mps_is_the_same_in_any_units(scale):
    ref, _ = _fit(**_fixture(), how="MPS")
    model, rec = _fit(**_fixture(scale), how="MPS")
    assert not [w for w in rec if "finite maximum" in str(w.message)]
    got = np.asarray(model.params, dtype=float)
    got[2:] /= scale
    # to the optimiser's tolerance (about 5e-6 here)
    np.testing.assert_allclose(got, ref.params, rtol=1e-4)
    assert ref.params[2] < _fixture()["x"].min()
    assert ref.params[3] > _fixture()["x"].max()


def test_an_interior_fit_does_not_warn_of_no_maximum():
    # Shapes well above 1 and a large sample: the ends stay outside the
    # data, where the density at the extremes is finite.
    rng = np.random.default_rng(1)
    x = 2.0 + 3.0 * rng.beta(4.0, 5.0, 400)
    model, rec = _fit(x=x)
    assert not [w for w in rec if "finite maximum" in str(w.message)]
    assert model.params[0] > 1 and model.params[1] > 1
    assert model.params[2] < x.min() and model.params[3] > x.max()
