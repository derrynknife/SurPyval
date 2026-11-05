"""``CompetingRisks.fit(..., w=)``: the weighted Aalen-Johansen estimate
and its robust (infinitesimal jackknife) variances. The values are checked
against R in ``reference/test_weighted_aalen_johansen.py``; these are the
properties every weighted fit keeps (docs/Weighted Data.rst)."""

import json
import pickle
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import surpyval as sp
from surpyval.tests._helpers import weighted_reference_data
from surpyval.univariate.competing_risks.aalen_johansen import (
    aalen_johansen_variance,
)
from surpyval.utils.weights import WeightedRefusal

Q = np.array([1.0, 2.2, 3.7, 6.1, 8.3])


@pytest.fixture(scope="module")
def data():
    d = weighted_reference_data()
    e = np.where(d["c"] == 0, np.where(d["Z"][:, 0] == 1, "a", "b"), None)
    return d["x"], d["c"], e, d["w"], d["Z"][:, 0]


def _fit(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.CompetingRisks.fit(*args, **kwargs)


@pytest.mark.parametrize("how", ["Nelson-Aalen", "Kaplan-Meier"])
def test_integer_weights_equal_counts(data, how):
    x, c, e, _, _ = data
    k = np.where(np.arange(len(x)) % 3 == 0, 3, 1)
    weighted = _fit(x, e, c, w=k.astype(float), how=how)
    counted = _fit(x, e, c, n=k, how=how)
    for name in ("x", "r", "d", "d_e", "S", "CIF"):
        assert_array_equal(getattr(weighted, name), getattr(counted, name))


def test_counts_are_copies_with_weights(data):
    """Counts still mean copies, weighted or not: the robust variance of
    a row counted twice is that of the row written twice."""
    x, c, e, w, _ = data
    n = np.where(np.arange(len(x)) < 5, 2, 1)
    rows = np.concatenate([np.arange(len(x)), np.arange(5)])
    counted = _fit(x, e, c, n=n, w=w)
    written = _fit(x[rows], e[rows], c[rows], w=w[rows])
    for name in ("CIF", "cif_var", "log_sf_var", "log_sf_var_km"):
        assert_allclose(
            getattr(counted, name), getattr(written, name), rtol=1e-12
        )


@pytest.mark.parametrize("scale", [0.01, 7.3])
def test_weight_scale_changes_nothing(data, scale):
    x, c, e, w, _ = data
    base = _fit(x, e, c, w=w)
    scaled = _fit(x, e, c, w=w * scale)
    assert_allclose(scaled.CIF, base.CIF, rtol=1e-12, atol=1e-15)
    for name in ("cif_var", "log_sf_var", "log_sf_var_km"):
        assert_allclose(
            getattr(scaled, name), getattr(base, name), rtol=1e-10, atol=0
        )
    assert_allclose(scaled.cb(Q, "a"), base.cb(Q, "a"), rtol=1e-10)
    assert_allclose(
        scaled.cb(Q, on="ff"), base.cb(Q, on="ff"), rtol=1e-10, atol=1e-15
    )
    assert scaled.effective_n == pytest.approx(base.effective_n, rel=1e-12)


def test_unit_weights_give_the_unweighted_estimate_with_robust_se(data):
    """``w = 1`` gives the unweighted cumulative incidences, but the
    variance is still the robust one, not Aalen's (which it is close to)."""
    x, c, e, _, _ = data
    plain = _fit(x, e, c)
    ones = _fit(x, e, c, w=np.ones(len(x)))
    assert ones.weighted and not plain.weighted
    assert_allclose(ones.CIF, plain.CIF, rtol=1e-14, atol=1e-16)
    assert_allclose(ones.r, plain.r, rtol=0)
    k = ones.event_idx_map["a"]
    aalen = aalen_johansen_variance(plain.r, plain.d, plain.d_e[k])
    assert not np.allclose(ones.cif_var[k], aalen, rtol=1e-6)
    assert_allclose(ones.cif_var[k], aalen, rtol=0.25)


def test_single_event_bounds_are_the_weighted_fits(data):
    x, c, e, w, _ = data
    model = _fit(x, e, c, w=w)
    all_cause = sp.NelsonAalen.fit(x, c, w=w)
    assert_allclose(model.cb(Q, on="sf"), all_cause.cb(Q), rtol=1e-12)
    km = _fit(x, e, c, w=w, how="Kaplan-Meier")
    net = sp.KaplanMeier.fit(x, np.where(e == "b", 0, 1), w=w)
    assert_allclose(
        km.cb(Q, "b", on="Hf"), net.cb(Q, on="Hf"), rtol=1e-12, atol=1e-15
    )


def test_effective_sample_size_and_its_warning(data):
    x, c, e, w, _ = data
    model = _fit(x, e, c, w=w)
    ess = np.sum(w) ** 2 / np.sum(w**2)
    assert model.effective_n == pytest.approx(ess)
    assert "effective sample size 33.37" in repr(model)
    with pytest.warns(UserWarning, match="effective sample size"):
        sp.CompetingRisks.fit(x[:12], e[:12], c[:12], w=w[:12])


def test_round_trip_is_schema_4_and_keeps_the_robust_bounds(data):
    x, c, e, w, _ = data
    model = _fit(x, e, c, w=w)
    d = model.to_dict()
    assert d["schema"] == 4 and d["weighted"] is True
    assert "weighted" not in _fit(x, e, c).to_dict()
    for again in (
        sp.from_dict(json.loads(json.dumps(d, allow_nan=False))),
        pickle.loads(pickle.dumps(model)),
    ):
        assert again.weighted
        assert again.effective_n == model.effective_n
        for on in ("cif", "sf", "Hf"):
            event = "a" if on == "cif" else None
            assert_allclose(
                again.cb(Q, event, on=on), model.cb(Q, event, on=on)
            )


def test_fit_from_df_and_surpyval_data_take_the_weights(data):
    import pandas as pd

    x, c, e, w, _ = data
    arrays = _fit(x, e, c, w=w)
    df = pd.DataFrame({"x": x, "c": c, "e": e, "w": w})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_df = sp.CompetingRisks.fit_from_df(
            df, "x", "e", c_col="c", w_col="w"
        )
    held = _fit(sp.SurpyvalData(x, c, e=e, w=w))
    for model in (from_df, held):
        assert model.weighted
        assert_array_equal(model.CIF, arrays.CIF)
        assert_array_equal(model.cif_var, arrays.cif_var)


def test_bad_weights_are_refused(data):
    x, c, e, w, _ = data
    with pytest.raises(ValueError, match="positive and finite"):
        sp.CompetingRisks.fit(x, e, c, w=np.where(w > 2, 0.0, w))
    with pytest.raises(ValueError, match="one weight per row"):
        sp.CompetingRisks.fit(x, e, c, w=w[:-1])


def test_plots_draw_with_weights(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, c, e, w, _ = data
    model = _fit(x, e, c, w=w)
    for stacked in (False, True):
        fig, ax = plt.subplots()
        model.plot(ax=ax, stacked=stacked)
        plt.close(fig)


def test_gray_test_refuses_weights(data):
    x, c, e, w, z1 = data
    with pytest.raises(WeightedRefusal, match="Gray's test") as info:
        sp.gray_test(x, e, z1, event="a", c=c, w=w)
    assert "logrank(..., w=w)" in str(info.value)
