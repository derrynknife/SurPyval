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
