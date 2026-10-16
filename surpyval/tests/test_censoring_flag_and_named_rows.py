"""A True/False censoring flag is refused (#920), and one item's
covariates given as a Series or a dict are read by name (#921)."""

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import EventHistoryData, ItemHistory, ItemLifetime, SurpyvalData

X = [1.0, 2.0, 3.0, 4.0, 5.0]
EVENT = np.array([True, False, True, False, False])


# #920: a boolean c


@pytest.mark.parametrize(
    "fit",
    [
        lambda c: sp.Weibull.fit(X, c=c),
        lambda c: sp.KaplanMeier.fit(X, c=c),
        lambda c: sp.CoxPH.fit(X, Z=[1.0, 0, 1, 0, 1], c=c),
        lambda c: sp.WeibullPH.fit(X, Z=[1.0, 0, 1, 0, 1], c=c),
        lambda c: sp.CompetingRisks.fit(
            X, e=["a", None, "b", None, None], c=c
        ),
        lambda c: sp.recurrent.CrowAMSAA.fit(X, i=[1] * 5, c=c),
        lambda c: sp.Weibull.fit_from_df(
            pd.DataFrame({"x": X, "c": c}), x_col="x", c_col="c"
        ),
        lambda c: SurpyvalData(X, c=c),
    ],
)
@pytest.mark.parametrize(
    "c",
    [
        EVENT,
        list(EVENT),
        pd.Series(EVENT),
        pd.Series(EVENT, dtype="boolean"),
    ],
)
def test_a_boolean_c_is_refused_everywhere(fit, c):
    with pytest.raises(ValueError, match="True/False 'c'.*c=1 - event"):
        fit(c)


def test_a_boolean_scalar_c_is_refused():
    with pytest.raises(ValueError, match="True/False 'c'"):
        SurpyvalData(X, c=True)


def test_numeric_flags_fit_as_before():
    as_int = sp.Weibull.fit(X, c=(~EVENT).astype(int))
    as_float = sp.Weibull.fit(X, c=(~EVENT).astype(float))
    assert np.allclose(as_int.params, as_float.params)
    assert SurpyvalData(X, c=1 - EVENT.astype(int)).c.tolist() == [
        0,
        1,
        0,
        1,
        1,
    ]


# #921: one item's covariates by name

ROW = {"age": 50.0, "arm": 1.0}


def _named_data(k=60, seed=3):
    rng = np.random.default_rng(seed)
    return SurpyvalData.from_items(
        [
            ItemLifetime(
                j,
                observed=float(rng.weibull(2) * 100 * np.exp(-0.5 * (j % 2))),
                covariates={"arm": float(j % 2), "age": rng.uniform(40, 70)},
            )
            for j in range(k)
        ]
    )


DATA = _named_data()


@pytest.mark.parametrize(
    "model",
    [
        sp.WeibullPH,
        sp.CoxPH,
        sp.WeibullAFT,
        sp.WeibullPO,
        sp.AdditiveHazards,
        sp.BuckleyJames,
    ],
    ids=lambda m: getattr(m, "name", type(m).__name__),
)
def test_series_and_dict_are_read_by_name(model):
    fitted = model.fit(DATA)
    by_frame = fitted.sf(50, pd.DataFrame([ROW]))
    assert np.allclose(fitted.sf(50, pd.Series(ROW)), by_frame)
    assert np.allclose(fitted.sf(50, ROW), by_frame)
    # The fitted order is arm, age
    assert np.allclose(fitted.sf(50, [[1.0, 50.0]]), by_frame)


def test_several_items_as_a_dict_of_columns():
    cox = sp.CoxPH.fit(DATA)
    both = cox.sf(50, {"age": [50.0, 60.0], "arm": [1.0, 0.0]})
    frame = cox.sf(50, pd.DataFrame({"age": [50.0, 60.0], "arm": [1.0, 0.0]}))
    assert np.allclose(both, frame)


def test_a_missing_name_is_refused():
    with pytest.raises(ValueError, match="arm"):
        sp.WeibullPH.fit(DATA).sf(50, {"age": 50.0})


def test_without_names_a_dict_is_refused_and_a_series_is_positional():
    plain = sp.WeibullPH.fit(DATA.x, DATA.Z, c=DATA.c)
    with pytest.raises(ValueError, match="have no names"):
        plain.sf(50, ROW)
    assert np.allclose(
        plain.sf(50, pd.Series([1.0, 50.0])), plain.sf(50, [1.0, 50.0])
    )


def test_a_series_named_after_a_covariate_is_that_column():
    one = sp.WeibullPH.fit(DATA.with_covariates(["age"]))
    assert np.allclose(
        one.sf(50, pd.Series([50.0, 60.0], name="age")),
        one.sf(50, [[50.0], [60.0]]),
    )


def test_fine_gray_reads_names():
    rng = np.random.default_rng(4)
    n = 120
    frame = pd.DataFrame(
        {
            "arm": (np.arange(n) % 2).astype(float),
            "age": rng.uniform(40, 70, n),
        }
    )
    t = rng.weibull(1.5, n) * 100 * np.exp(-0.5 * frame.arm)
    e = np.where(rng.random(n) < 0.5, "a", "b")
    fg = sp.FineGray.fit(t.values, frame, e=e, event="a")
    expected = fg.cif(30, [[1.0, 50.0]])
    for Z in (pd.DataFrame([ROW]), pd.Series(ROW), ROW):
        assert np.allclose(fg.cif(30, Z), expected)


def test_proportional_intensity_reads_names():
    rng = np.random.default_rng(1)
    histories = [
        ItemHistory(
            k,
            sorted(rng.uniform(0, 100, 2 + k % 3).tolist()),
            right_censored=100,
            covariates={"load": float(k % 3), "size": float(k % 2)},
        )
        for k in range(40)
    ]
    model = sp.ProportionalIntensityHPP.fit(
        EventHistoryData.from_items(histories)
    )
    row = {"size": 1.0, "load": 2.0}
    expected = model.cif(50, [2.0, 1.0])
    for Z in (pd.DataFrame([row]), pd.Series(row), row):
        assert np.allclose(model.cif(50, Z), expected)
    assert np.allclose(model.cif_cb(50, row), model.cif_cb(50, [2.0, 1.0]))
