"""Delayed entry (left truncation, ``tl``) in the competing-risks
regressions: the cause-specific Cox model takes it, each cause's ``CoxPH``
fit with the entry-adjusted risk sets; Fine-Gray and Gray's test refuse
it, naming what to use instead. The R references are in
``surpyval/tests/reference/test_competing_entry.py``."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import surpyval as sp
from surpyval import gray_test
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
    FineGray,
)

CRPH = CompetingRisksProportionalHazards


def _data(seed=7, size=150):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(size, 2))
    t_a = rng.exponential(1 / (0.1 * np.exp(0.5 * Z[:, 0])))
    t_b = rng.exponential(1 / (0.05 * np.exp(-0.4 * Z[:, 1])))
    t_c = rng.uniform(0, 25, size)
    x = np.minimum.reduce([t_a, t_b, t_c]).round(3) + 0.01
    first = np.where(t_a < t_b, "a", "b")
    e = np.where(t_c < np.minimum(t_a, t_b), None, first).astype(object)
    late = rng.random(size) < 0.4
    tl = np.where(late, (x * rng.random(size)).round(3), 0.0)
    return x, Z, e, tl


def _censored(e):
    return np.array([cause is None for cause in e], dtype=int)


def _same(one, other):
    assert_array_equal(one.betas, other.betas)
    assert_array_equal(one.x, other.x)
    assert_array_equal(one.H0_e, other.H0_e)
    assert_array_equal(one.covariance(), other.covariance())
    assert one.neg_ll() == other.neg_ll()


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_entries_before_the_first_time_change_nothing(ties):
    x, Z, e, _ = _data()
    base = CRPH.fit(x, Z, e, tie_method=ties)
    for tl in (np.zeros(x.size), np.full(x.size, -np.inf), -1.0):
        _same(CRPH.fit(x, Z, e, tl=tl, tie_method=ties), base)
    assert_array_equal(
        CRPH.fit(x, Z, e, tl=0.0).cif([2, 5, 9], [0.3, -1], "a"),
        base.cif([2, 5, 9], [0.3, -1], "a"),
    )


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_each_cause_is_its_cox_fit_with_entry(ties):
    x, Z, e, tl = _data()
    model = CRPH.fit(x, Z, e, tl=tl, tie_method=ties)
    assert model.models is not None
    for cause in ("a", "b"):
        c_k = np.where(e == cause, 0, 1)
        cox = sp.CoxPH.fit(x, Z, c_k, tl=tl, tie_method=ties)
        i = model.event_idx_map[cause]
        assert_array_equal(model.betas[i], cox.beta)
        assert_allclose(
            model.Hf(cox.x, np.zeros(2), event=cause), cox.H0, rtol=1e-12
        )
    # The entries move the estimates (the test above is not vacuous).
    assert not np.allclose(model.betas, CRPH.fit(x, Z, e).betas)


def test_the_incidences_sum_to_the_all_cause_failure_probability():
    x, Z, e, tl = _data()
    model = CRPH.fit(x, Z, e, tl=tl)
    t = np.linspace(0.5, 20, 9)
    z = [0.4, -0.2]
    total = model.cif(t, z, "a") + model.cif(t, z, "b")
    assert_allclose(total, model.ff(t, z), rtol=1e-12)
    assert np.all(np.diff(model.cif(t, z, "a")) >= 0)


def test_splitting_a_row_at_a_time_changes_nothing():
    # (tl, x] is (tl, s] survived then (s, x]: the same risk sets.
    x, Z, e, tl = _data()
    whole = CRPH.fit(x, Z, e, tl=tl)
    split = (x - tl) > 1
    s = tl[split] + 0.5
    x2 = np.concatenate([x[~split], s, x[split]])
    tl2 = np.concatenate([tl[~split], tl[split], s])
    Z2 = np.concatenate([Z[~split], Z[split], Z[split]])
    e2 = np.concatenate(
        [e[~split], np.full(split.sum(), None, dtype=object), e[split]]
    )
    parts = CRPH.fit(x2, Z2, e2, tl=tl2)
    assert_allclose(parts.betas, whole.betas, rtol=1e-8)
    assert_allclose(parts.covariance(), whole.covariance(), rtol=1e-7)
    t = [1.0, 4.0, 12.0]
    assert_allclose(
        parts.cif(t, [0, 0], "b"), whole.cif(t, [0, 0], "b"), rtol=1e-8
    )


def test_every_way_in_gives_the_same_fit():
    x, Z, e, tl = _data()
    model = CRPH.fit(x, Z, e, tl=tl)
    df = pd.DataFrame(
        {"x": x, "e": e, "z0": Z[:, 0], "z1": Z[:, 1], "entry": tl}
    )
    from_df = CRPH.fit_from_df(
        df, "x", "e", Z_cols=["z0", "z1"], tl_col="entry"
    )
    assert_array_equal(from_df.betas, model.betas)
    data = sp.SurpyvalData(x, c=_censored(e), tl=tl, Z=Z, e=e)
    assert_array_equal(CRPH.fit(data).betas, model.betas)
    restored = sp.from_dict(model.to_dict())
    assert_array_equal(
        restored.cif([3.0, 8.0], [1, 0], "a"),
        model.cif([3.0, 8.0], [1, 0], "a"),
    )


def test_a_row_with_a_missing_covariate_drops_its_entry_with_it():
    x, Z, e, tl = _data()
    Z = Z.copy()
    Z[4, 1] = np.nan
    keep = np.arange(x.size) != 4
    with pytest.warns(UserWarning):
        model = CRPH.fit(x, Z, e, tl=tl)
    assert_array_equal(
        model.betas, CRPH.fit(x[keep], Z[keep], e[keep], tl=tl[keep]).betas
    )


def test_the_penalised_fit_takes_entry_cause_by_cause():
    x, Z, e, tl = _data()
    model = CRPH.fit_penalised(x, Z, e, tl=tl, penalizer=0.05)
    c_a = np.where(e == "a", 0, 1)
    cox = sp.CoxPH.fit_penalised(x, Z, c_a, tl=tl, penalizer=0.05)
    assert_array_equal(model.betas[model.event_idx_map["a"]], cox.beta)


def test_the_diagnostics_run_on_the_entry_adjusted_fits():
    x, Z, e, tl = _data()
    report = CRPH.fit(x, Z, e, tl=tl).diagnose()
    assert set(report.table["cause"]) == {"a", "b"}


def test_right_truncation_is_refused_with_what_to_pass():
    x, Z, e, tl = _data()
    with pytest.raises(ValueError, match="one-dimensional `tl`"):
        CRPH.fit(x, Z, e, tl=np.column_stack([tl, x + 1]))


def _refused(call, who):
    with pytest.raises(ValueError, match="takes no delayed entry") as info:
        call()
    message = str(info.value)
    assert message.startswith(who)
    for name in ("CompetingRisks.fit", "tl=tl"):
        assert name in message
    return message


def test_fine_gray_refuses_entry_naming_the_alternatives():
    x, Z, e, tl = _data()
    message = _refused(
        lambda: FineGray.fit(x, Z, e, event="a", tl=tl), "FineGray"
    )
    assert "CompetingRisksProportionalHazards.fit" in message
    _refused(
        lambda: CRPH.fit(x, Z, e, tl=tl, model="Fine-Gray"),
        "The Fine-Gray model",
    )
    df = pd.DataFrame({"x": x, "e": e, "z0": Z[:, 0], "entry": tl})
    _refused(
        lambda: CRPH.fit_from_df(
            df, "x", "e", Z_cols="z0", tl_col="entry", model="Fine-Gray"
        ),
        "The Fine-Gray model",
    )
    _refused(
        lambda: FineGray.fit_from_df(
            df, "x", "e", Z_cols="z0", tl_col="entry", event="a"
        ),
        "FineGray",
    )
    data = sp.SurpyvalData(x, c=_censored(e), tl=tl, Z=Z, e=e)
    _refused(lambda: FineGray.fit(data, event="a"), "FineGray")


def test_fine_gray_takes_entries_before_the_first_time():
    x, Z, e, _ = _data()
    base = FineGray.fit(x, Z, e, event="a")
    assert_array_equal(
        FineGray.fit(x, Z, e, event="a", tl=np.zeros(x.size)).beta, base.beta
    )
    assert_array_equal(
        CRPH.fit(x, Z, e, tl=0.0, model="Fine-Gray").betas,
        CRPH.fit(x, Z, e, model="Fine-Gray").betas,
    )
    data = sp.SurpyvalData(x, c=_censored(e), Z=Z, e=e)
    assert_array_equal(FineGray.fit(data, event="a").beta, base.beta)


def test_gray_test_refuses_entry_naming_the_alternatives():
    x, Z, e, tl = _data()
    group = (Z[:, 0] > 0).astype(int)
    message = _refused(
        lambda: gray_test(x, e, group, "a", tl=tl), "Gray's test"
    )
    assert "logrank(" in message
    base = gray_test(x, e, group, "a")
    assert gray_test(x, e, group, "a", tl=0.0) == base
