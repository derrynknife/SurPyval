"""Delayed entry (left truncation, ``tl``) in the non-parametric
competing-risks estimate, ``CompetingRisks``: the identities it must keep.
The values against R are in
``surpyval/tests/reference/test_aalen_johansen_entry.py``."""

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import surpyval as sp
from surpyval.univariate.competing_risks.aalen_johansen import (
    aalen_johansen_variance,
)

matplotlib.use("Agg")


def _data(seed=3, size=60):
    """Times with ties, two causes and censoring, each unit entering at a
    time before its exit (many after other units' exits)."""
    rng = np.random.default_rng(seed)
    x = np.round(rng.weibull(1.5, size) * 10 + 1, 1)
    tl = np.round(rng.uniform(0, 1, size) * (x - 0.1), 1)
    e = rng.choice(np.array(["a", "b", None], dtype=object), size)
    return x, e, tl


def _flags(e):
    return np.array([1 if v is None else 0 for v in e])


def _aalen_se(model, cause):
    return np.sqrt(
        aalen_johansen_variance(
            model.r, model.d, model.d_e[model.event_idx_map[cause]]
        )
    )


def test_entry_before_every_time_changes_nothing():
    x, e, _ = _data()
    base = sp.CompetingRisks.fit(x, e)
    for tl in (0.0, np.full(len(x), -np.inf), np.zeros(len(x))):
        late = sp.CompetingRisks.fit(x, e, tl=tl)
        for name in ("x", "r", "d", "d_e", "CIF", "IIF", "H0_e"):
            assert_array_equal(getattr(late, name), getattr(base, name))
        assert_array_equal(late.cb(x, "a"), base.cb(x, "a"))


def test_risk_sets_count_only_the_units_that_have_entered():
    x = np.array([2.0, 3.0, 5.0, 6.0])
    e = np.array(["a", "b", "a", None], dtype=object)
    # The unit leaving at 6 enters at 3, exactly at the failure there: not
    # at risk for it, under the (entry, exit] rule.
    model = sp.CompetingRisks.fit(x, e, tl=[0, 0, 0, 3])
    assert_array_equal(model.r, [3, 2, 2, 1])
    # S(3) = 2/3 * 1/2, so the failure at 5 adds 1/3 * 1/2.
    assert_allclose(model.cif([2, 3, 5], "a"), [1 / 3, 1 / 3, 1 / 2])


@pytest.mark.parametrize("how", ["Kaplan-Meier", "Nelson-Aalen"])
def test_all_cause_estimate_is_the_single_event_one_with_entry(how):
    x, e, tl = _data()
    c = _flags(e)
    model = sp.CompetingRisks.fit(x, e, tl=tl, how=how)
    single = (sp.KaplanMeier if how == "Kaplan-Meier" else sp.NelsonAalen).fit(
        x, c, tl=tl
    )
    q = np.linspace(0, x.max(), 41)
    assert_allclose(model.sf(q), single.sf(q), rtol=1e-12)
    assert_allclose(
        model.cb(q, on="sf"), single.cb(q, on="sf"), rtol=1e-12, atol=1e-15
    )


def test_cumulative_incidences_add_up_to_the_kaplan_meier_failure():
    x, e, tl = _data()
    model = sp.CompetingRisks.fit(x, e, tl=tl, how="Kaplan-Meier")
    q = np.linspace(0, x.max(), 41)
    total = model.cif(q, "a") + model.cif(q, "b")
    assert_allclose(total, model.ff(q), rtol=1e-12, atol=1e-15)


def test_splitting_a_unit_where_it_reenters_changes_nothing():
    """A unit's (tl, x] as two rows, (tl, s] censored and (s, x] entering
    at s, leaves every risk set, and so every estimate and the Aalen
    variance, as it was."""
    x, e, tl = _data()
    s = (tl + x) / 2
    # Split the first ten units at s (not at any exit time, so the split
    # adds no time to the ladder).
    k = 10
    s = s[:k] + 0.01234
    xs = np.concatenate([s, x])
    es = np.concatenate([np.full(k, None, dtype=object), e])
    tls = np.concatenate([tl[:k], s, tl[k:]])
    whole = sp.CompetingRisks.fit(x, e, tl=tl)
    split = sp.CompetingRisks.fit(xs, es, tl=tls)
    q = np.linspace(0, x.max(), 41)
    for cause in ("a", "b"):
        assert_allclose(split.cif(q, cause), whole.cif(q, cause), rtol=1e-12)
        # The split adds censoring-only times; the variance is unchanged
        # at every time of the whole data.
        idx = np.searchsorted(split.x, whole.x)
        assert_allclose(
            _aalen_se(split, cause)[idx],
            _aalen_se(whole, cause),
            rtol=1e-12,
        )


def test_without_entry_the_estimate_is_biased():
    """Units that failed before they would have entered are never seen:
    fitting as if every unit were followed from 0 counts too few at risk
    early on, and the curve differs from the delayed-entry one."""
    x, e, tl = _data()
    late = sp.CompetingRisks.fit(x, e, tl=tl)
    naive = sp.CompetingRisks.fit(x, e)
    assert np.all(late.r <= naive.r) and np.any(late.r < naive.r)
    assert not np.allclose(late.cif(x, "a"), naive.cif(x, "a"))


def test_scalar_entry_is_every_units_entry():
    x, e, _ = _data()
    one = sp.CompetingRisks.fit(x, e, tl=0.5)
    each = sp.CompetingRisks.fit(x, e, tl=np.full(len(x), 0.5))
    assert_array_equal(one.CIF, each.CIF)
    with pytest.raises(ValueError, match="strictly less than"):
        sp.CompetingRisks.fit(x, e, tl=x.min())


def test_fit_from_df_takes_the_entry_column():
    x, e, tl = _data()
    df = pd.DataFrame({"x": x, "e": e, "tl": tl})
    from_df = sp.CompetingRisks.fit_from_df(
        df, x_col="x", e_col="e", tl_col="tl"
    )
    arrays = sp.CompetingRisks.fit(x, e, tl=tl)
    assert_array_equal(from_df.CIF, arrays.CIF)
    assert_array_equal(from_df.r, arrays.r)
    with pytest.raises(ValueError, match="tl_col='entry' is not a column"):
        sp.CompetingRisks.fit_from_df(df, x_col="x", e_col="e", tl_col="entry")
    with pytest.raises(TypeError, match="tl_col"):
        sp.CompetingRisks.fit_from_df(df, x_col="x", e_col="e", tl="tl")


def test_fit_takes_the_entry_times_of_surpyval_data():
    x, e, tl = _data()
    data = sp.SurpyvalData(x, _flags(e), tl=tl, e=e)
    assert_array_equal(
        sp.CompetingRisks.fit(data).CIF,
        sp.CompetingRisks.fit(x, e, tl=tl).CIF,
    )


def test_right_truncation_is_refused():
    x, e, tl = _data()
    data = sp.SurpyvalData(x, _flags(e), tl=tl, tr=np.full(len(x), 100.0), e=e)
    with pytest.raises(ValueError, match="takes no right truncation"):
        sp.CompetingRisks.fit(data)


def test_entry_at_or_after_the_exit_is_refused():
    with pytest.raises(ValueError, match="strictly less than"):
        sp.CompetingRisks.fit([1, 2, 3], ["a", "b", "a"], tl=[0, 2, 1])


def test_weighted_fit_with_entry_keeps_the_weighted_identities():
    x, e, tl = _data()
    w = 1 + (np.arange(len(x)) % 3) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.CompetingRisks.fit(x, e, tl=tl, w=w, how="Kaplan-Meier")
        single = sp.KaplanMeier.fit(x, _flags(e), tl=tl, w=w)
    q = np.linspace(0, x.max(), 41)
    assert_allclose(model.sf(q), single.sf(q), rtol=1e-12)
    assert_allclose(
        model.cb(q, on="sf"), single.cb(q, on="sf"), rtol=1e-10, atol=1e-15
    )
    # Integer weights: the same estimate as the rows repeated.
    w = 1 + np.arange(len(x)) % 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weighted = sp.CompetingRisks.fit(x, e, tl=tl, w=w)
    repeated = sp.CompetingRisks.fit(
        np.repeat(x, w), np.repeat(e, w), tl=np.repeat(tl, w)
    )
    assert_allclose(weighted.CIF, repeated.CIF, rtol=1e-12)


def test_serialised_model_with_entry_predicts_the_same():
    x, e, tl = _data()
    model = sp.CompetingRisks.fit(x, e, tl=tl)
    back = sp.CompetingRisks.from_dict(model.to_dict())
    q = np.linspace(0, x.max(), 41)
    assert_array_equal(back.cif(q, "a"), model.cif(q, "a"))
    assert_array_equal(back.cb(q, "b"), model.cb(q, "b"))


@pytest.mark.parametrize("stacked", [False, True])
def test_plot_with_entry(stacked):
    import matplotlib.pyplot as plt

    x, e, tl = _data()
    model = sp.CompetingRisks.fit(x, e, tl=tl)
    fig, ax = plt.subplots()
    assert model.plot(ax=ax, stacked=stacked) is ax
    plt.close(fig)
