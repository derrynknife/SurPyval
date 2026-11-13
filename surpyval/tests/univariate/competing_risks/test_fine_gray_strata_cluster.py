"""The stratified and the clustered Fine-Gray model: ``FineGray.fit(...,
strata=, strata_censoring=, cluster=)``. The checks against R's
``crrSC`` are in ``tests/reference/test_fine_gray_strata_cluster.py``."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from surpyval.univariate.competing_risks import FineGray
from surpyval.univariate.competing_risks.regression.fine_gray import (
    FineGrayModel,
)


def _data(seed=1, n=150):
    rng = np.random.default_rng(seed)
    Z = np.column_stack([rng.normal(size=n), rng.binomial(1, 0.5, n)])
    strata = rng.choice(np.array(["north", "south", "west"]), n)
    scale = np.where(strata == "north", 1.0, 2.0)
    t_a = rng.exponential(scale / (0.2 * np.exp(Z @ [0.5, -0.4])))
    t_b = rng.exponential(8.0, n)
    t_c = rng.uniform(0, 15, n)
    x = np.minimum.reduce([t_a, t_b, t_c]).round(3)
    e = np.where(x == t_c.round(3), None, np.where(t_a < t_b, "a", "b"))
    cluster = np.arange(n) // 3
    return x, Z, e.astype(object), strata, cluster


def test_one_stratum_and_singleton_clusters_are_the_plain_fit():
    x, Z, e, _, _ = _data()
    plain = FineGray.fit(x, Z, e, event="a")
    one = FineGray.fit(x, Z, e, event="a", strata=np.full(x.size, "all"))
    singles = FineGray.fit(x, Z, e, event="a", cluster=np.arange(x.size))
    for model in (one, singles):
        assert_allclose(model.beta, plain.beta, rtol=1e-10)
        assert_allclose(model.covariance(), plain.covariance(), rtol=1e-9)
    assert_allclose(
        one.cif([1, 4, 9], [0.2, 1], stratum="all"),
        plain.cif([1, 4, 9], [0.2, 1]),
        rtol=1e-10,
    )


def test_row_order_does_not_matter():
    x, Z, e, strata, cluster = _data()
    order = np.random.default_rng(3).permutation(x.size)
    for strata_censoring in ("within", "common"):
        kw = dict(event="a", strata_censoring=strata_censoring)
        a = FineGray.fit(x, Z, e, strata=strata, cluster=cluster, **kw)
        b = FineGray.fit(
            x[order],
            Z[order],
            e[order],
            strata=strata[order],
            cluster=cluster[order],
            **kw,
        )
        assert_allclose(a.beta, b.beta, rtol=1e-9)
        assert_allclose(a.covariance(), b.covariance(), rtol=1e-8)
        assert_allclose(
            a.cif([2, 5], [0, 1], stratum="west"),
            b.cif([2, 5], [0, 1], stratum="west"),
            rtol=1e-9,
        )


def test_strata_combine_with_clusters_and_weights():
    x, Z, e, strata, cluster = _data()
    stratified = FineGray.fit(x, Z, e, event="a", strata=strata)
    both = FineGray.fit(x, Z, e, event="a", strata=strata, cluster=cluster)
    assert_allclose(both.beta, stratified.beta, rtol=1e-12)
    assert not np.allclose(both.covariance(), stratified.covariance())
    # Whole-number weights are rows counted that many times
    w = np.where(np.arange(x.size) % 4 == 0, 2.0, 1.0)
    rep = np.repeat(np.arange(x.size), w.astype(int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weighted = FineGray.fit(x, Z, e, event="a", strata=strata, w=w)
    counted = FineGray.fit(
        x[rep], Z[rep], e[rep], event="a", strata=strata[rep]
    )
    assert_allclose(weighted.beta, counted.beta, rtol=1e-8)
    assert_allclose(
        weighted.cif(3, [0, 1], stratum="south"),
        counted.cif(3, [0, 1], stratum="south"),
        rtol=1e-8,
    )


def test_cif_needs_a_stratum_and_only_when_stratified():
    x, Z, e, strata, _ = _data()
    model = FineGray.fit(x, Z, e, event="a", strata=strata)
    with pytest.raises(ValueError, match="pass stratum="):
        model.cif(3, [0, 1])
    with pytest.raises(ValueError, match="Unknown stratum 'east'"):
        model.cif(3, [0, 1], stratum="east")
    assert_allclose(
        model.sf([1, 3], [0, 1], stratum="north"),
        1 - model.cif([1, 3], [0, 1], stratum="north"),
    )
    plain = FineGray.fit(x, Z, e, event="a")
    with pytest.raises(ValueError, match="not stratified"):
        plain.cif(3, [0, 1], stratum="north")
    # The strata's incidences differ (their baselines do)
    assert model.cif(5, [0, 0], stratum="north") > model.cif(
        5, [0, 0], stratum="south"
    )


def test_refusals_name_the_argument():
    x, Z, e, strata, cluster = _data()
    with pytest.raises(ValueError, match="'strata' must have one label"):
        FineGray.fit(x, Z, e, event="a", strata=strata[:-1])
    with pytest.raises(ValueError, match="'cluster' must have one label"):
        FineGray.fit(x, Z, e, event="a", cluster=cluster[:5])
    with pytest.raises(ValueError, match="strata_censoring"):
        FineGray.fit(x, Z, e, event="a", strata_censoring="common")
    with pytest.raises(ValueError, match="'strata_censoring' must be one of"):
        FineGray.fit(x, Z, e, event="a", strata=strata, strata_censoring="x")
    with pytest.raises(ValueError, match="Every 'strata' label is missing"):
        FineGray.fit(x, Z, e, event="a", strata=[None] * x.size)


def test_missing_labels_drop_their_rows_with_a_warning():
    x, Z, e, strata, cluster = _data()
    gappy = strata.astype(object)
    gappy[[0, 5]] = None
    with pytest.warns(UserWarning, match="Dropped 2 of 150 rows"):
        model = FineGray.fit(x, Z, e, event="a", strata=gappy)
    keep = np.ones(x.size, bool)
    keep[[0, 5]] = False
    direct = FineGray.fit(
        x[keep], Z[keep], e[keep], event="a", strata=strata[keep]
    )
    assert_allclose(model.beta, direct.beta, rtol=1e-12)


def test_a_stratum_without_events_of_the_cause():
    x, Z, e, strata, _ = _data()
    e = e.copy()
    e[(strata == "west") & (e == "a")] = "b"
    model = FineGray.fit(x, Z, e, event="a", strata=strata)
    assert np.all(model.cif([1, 5, 10], [0, 1], stratum="west") == 0)
    rest = strata != "west"
    without = FineGray.fit(
        x[rest], Z[rest], e[rest], event="a", strata=strata[rest]
    )
    assert_allclose(model.beta, without.beta, rtol=1e-10)


def test_round_trip_keeps_strata_and_clusters():
    x, Z, e, strata, cluster = _data()
    model = FineGray.fit(
        x,
        Z,
        e,
        event="a",
        strata=strata,
        cluster=cluster,
        strata_censoring="common",
    )
    d = json.loads(json.dumps(model.to_dict()))
    assert d["schema"] == 4 and "baseline_times" not in d
    back = FineGrayModel.from_dict(d)
    assert back.strata_labels == ["north", "south", "west"]
    assert back.n_clusters == model.n_clusters == 50
    assert back.strata_censoring == "common"
    for s in back.strata_labels:
        assert_allclose(
            back.cif([1, 4, 9], [0.3, 1], stratum=s),
            model.cif([1, 4, 9], [0.3, 1], stratum=s),
            rtol=1e-14,
        )
    assert_allclose(back.covariance(), model.covariance(), rtol=1e-14)
    text = repr(back)
    assert "Strata" in text and "50 (cluster-robust" in text


def test_diagnostics_and_partial_effects_refuse():
    x, Z, e, strata, cluster = _data()
    for kw in ({"strata": strata}, {"cluster": cluster}):
        model = FineGray.fit(x, Z, e, event="a", **kw)
        with pytest.raises(ValueError, match="strata or clusters"):
            model.diagnose()
    stratified = FineGray.fit(x, Z, e, event="a", strata=strata)
    with pytest.raises(ValueError, match="stratum=..."):
        stratified.plot_partial_effects(0, [0.0, 1.0], reference={1: 0})


def test_fit_from_df_takes_the_columns():
    x, Z, e, strata, cluster = _data()
    df = pd.DataFrame(
        {"x": x, "e": e, "z0": Z[:, 0], "z1": Z[:, 1], "s": strata}
    ).assign(g=cluster)
    model = FineGray.fit_from_df(
        df,
        x_col="x",
        e_col="e",
        Z_cols=["z0", "z1"],
        strata_col="s",
        cluster_col="g",
        event="a",
    )
    direct = FineGray.fit(x, Z, e, event="a", strata=strata, cluster=cluster)
    assert_allclose(model.beta, direct.beta, rtol=1e-12)
    assert_allclose(model.covariance(), direct.covariance(), rtol=1e-12)
    with pytest.raises(ValueError, match="strata_col='t' is not a column"):
        FineGray.fit_from_df(
            df, x_col="x", e_col="e", Z_cols="z0", strata_col="t", event="a"
        )
