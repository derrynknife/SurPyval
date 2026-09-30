"""The bundled dataset loaders.

The loaders read package-shipped CSVs with pandas' default (C) engine
(issue #207 dropped the redundant ``engine="python"``); every loader
must return a non-empty DataFrame.
"""

import pandas as pd
import pytest

import surpyval.datasets as datasets

LOADERS = [
    name
    for name, obj in vars(datasets).items()
    if callable(obj) and name.startswith("load_")
]


@pytest.mark.parametrize("name", LOADERS)
def test_loader_returns_nonempty_dataframe(name):
    df = getattr(datasets, name)()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0 and len(df.columns) > 0


@pytest.mark.parametrize("name", LOADERS)
def test_loader_has_no_saved_row_index(name):
    # #479: rossi.csv carried "Unnamed: 0.1" and "Unnamed: 0" (row indices
    # saved by pandas/R); heart, lung and rossi_tv carried "Unnamed: 0".
    df = getattr(datasets, name)()
    assert not [c for c in df.columns if str(c).startswith("Unnamed")]


def test_rossi_static_arrest_has_the_original_coding():
    # #479: ``arrest`` was stored inverted (1 = not arrested) as a float,
    # so ``c = 1 - arrest``, the natural call for a lifelines or R user,
    # silently fitted the complement. It is now 1 = arrested, as in R's
    # carData::Rossi and lifelines' load_rossi.
    df = datasets.load_rossi_static()
    assert list(df.columns) == [
        "week",
        "arrest",
        "fin",
        "age",
        "race",
        "wexp",
        "mar",
        "paro",
        "prio",
    ]
    assert df["arrest"].dtype.kind == "i"
    assert df["arrest"].sum() == 114 and len(df) == 432
    # every prisoner not arrested is followed to week 52
    assert (df.loc[df["arrest"] == 0, "week"] == 52).all()
    # the static copy agrees with the time-varying one
    tv = datasets.load_rossi_time_varying().groupby("id").first()
    assert (tv["arrest"].to_numpy() == df["arrest"].to_numpy()).all()
