"""Loaders for the stored reference results (#379).

The data files are written by the generators in ``scripts/reference``:
``fixtures.json`` holds the data every reference was computed on, and each
``r_<package>.json`` / ``py_<package>.json`` holds the values one piece of
software produced, keyed by an id, with the software's name and version,
the exact call and the settings recorded next to them. The tests compare
SurPyval with those values, so they need neither R nor the other Python
packages.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"


@lru_cache(maxsize=None)
def _load(name):
    with open(DATA / "{}.json".format(name)) as f:
        return json.load(f)


def _array(values):
    """A float array with NaN where the file has null."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def fixture(name):
    """The columns of fixture ``name`` as float arrays (null is NaN)."""
    columns = _load("fixtures")[name]["columns"]
    return {k: _array(v) for k, v in columns.items()}


def fixture_extra(name, key):
    """A non-column item of a fixture (e.g. a prediction matrix)."""
    return _load("fixtures")[name][key]


def reference(source, ref_id):
    """The full stored entry ``ref_id`` of ``source`` (e.g.
    ``"r_survival"``): software, version, call, settings and values."""
    return _load(source)["references"][ref_id]


def values(source, ref_id):
    """The values of a stored entry, with numeric lists as float arrays
    (null is NaN) and nested lists as 2-D arrays."""
    out = {}
    for key, value in reference(source, ref_id)["values"].items():
        if isinstance(value, list) and value and isinstance(value[0], list):
            out[key] = np.array([_array(row) for row in value], dtype=float)
        elif isinstance(value, list) and all(
            v is None or isinstance(v, (int, float)) for v in value
        ):
            out[key] = _array(value)
        else:
            out[key] = value
    return out


def sources():
    """The names of every stored reference file."""
    return sorted(
        p.stem for p in DATA.glob("*.json") if p.stem.startswith(("r_", "py_"))
    )
