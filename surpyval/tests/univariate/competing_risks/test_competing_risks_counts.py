"""``CompetingRisks.fit`` counts each cause without a per-row scan (#515).

The fit found each row's time with ``np.where(unique_x == x_i)`` in a
Python loop over the rows, O(n * m): 3.6 s of a 3.8 s fit at 1e5 rows. It
now bins the counts with ``searchsorted`` and ``np.add.at``, which adds them
in row order as the loop did, so the fit is bit-identical. The missing-cause
check that marks censored rows is vectorised as well (``missing_events``)
and must agree with ``is_missing_event`` on every kind of label.
"""

import decimal
import fractions
import importlib
import types

import numpy as np
import pandas as pd
import pytest

from surpyval.univariate.competing_risks import CompetingRisks
from surpyval.utils import is_missing_event, missing_events

cr_module = importlib.import_module(
    "surpyval.univariate.competing_risks.nonparametric.competing_risks"
)
# ``missing_events`` looks ``is_missing_event`` up in its own module.
data_formats = importlib.import_module("surpyval.utils.data_formats")


def _loop_d_e(model, x, c, n, e):
    """The per-row loop ``fit`` used before #515, as the reference."""
    unique_x = model.x
    d_e = np.zeros((model.n_event_types, len(unique_x)))
    for i, x_i in enumerate(x):
        if c[i] == 1:
            continue
        j = model.event_idx_map[e[i]]
        d_e[j, np.where(unique_x == x_i)] += n[i]
    return d_e


def _labels(kind, rng, size):
    pools = {
        "str": ["a", "b", "c"],
        "int": [1, 2, 10],
        "tuple": [("seal", 1), ("seal", 2), ("bearing", 1)],
        "mixed": ["a", 2, ("t", 1)],
        "zero": [0, 1],
    }
    pool = pools[kind]
    out = np.empty(size, dtype=object)
    for i, k in enumerate(rng.integers(0, len(pool), size)):
        out[i] = pool[k]
    return out


def _datasets():
    rng = np.random.default_rng(515)
    for kind in ["str", "int", "tuple", "mixed", "zero"]:
        for size in [1, 2, 25, 400]:
            x = np.round(rng.weibull(1.5, size) * 10, 1) + 0.1
            e = _labels(kind, rng, size)
            # Censored rows: None, NaN and pandas NA all mark them.
            censored = rng.uniform(size=size) < 0.3
            if size > 1:
                e[censored] = rng.choice(
                    np.array([None, np.nan, pd.NA], dtype=object),
                    int(censored.sum()),
                )
            n = rng.integers(1, 5, size) if size > 2 else None
            yield f"{kind} {size}", x, e, n


@pytest.mark.filterwarnings("ignore:Cause label 0")
@pytest.mark.parametrize("how", ["Nelson-Aalen", "Kaplan-Meier"])
def test_fit_bit_identical_to_the_row_loop(how):
    for label, x, e, n in _datasets():
        model = CompetingRisks.fit(x, list(e), n=n, how=how)
        c = np.array([1.0 if is_missing_event(v) else 0.0 for v in e])
        nn = np.ones(len(x)) if n is None else np.asarray(n, dtype=float)
        expected = _loop_d_e(model, x, c, nn, e)
        np.testing.assert_array_equal(model.d_e, expected, err_msg=label)


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("explicit c", dict(c=[0, 1, 0, 0, 1, 0])),
        ("counts", dict(n=[3, 1, 2, 5, 1, 4])),
    ],
)
def test_fit_with_c_and_n(label, kwargs):
    x = np.array([2.0, 3.0, 3.0, 5.0, 5.0, 8.0])
    e = np.array(["a", None, "b", "a", None, "b"], dtype=object)
    model = CompetingRisks.fit(x, e, **kwargs)
    c = np.asarray(kwargs.get("c", [0, 1, 0, 0, 1, 0]), dtype=float)
    n = np.asarray(kwargs.get("n", np.ones(6)), dtype=float)
    np.testing.assert_array_equal(model.d_e, _loop_d_e(model, x, c, n, e))


class _ArrayLike:
    """Hashable, but array-like: the scalar ``isna`` converts it first."""

    def __array__(self, *args, **kwargs):
        return np.array([np.nan])


LABELS = [
    None,
    np.nan,
    float("nan"),
    pd.NA,
    pd.NaT,
    np.datetime64("NaT"),
    np.timedelta64("NaT"),
    decimal.Decimal("NaN"),
    fractions.Fraction(1, 2),
    np.float32("nan"),
    complex("nan"),
    (np.nan,),
    (1, "a"),
    (),
    "a",
    "",
    1,
    0,
    True,
    np.int64(3),
    frozenset([1]),
    np.inf,
    b"x",
    np.str_("a"),
]


@pytest.mark.parametrize("extra", [[], [_ArrayLike()]])
def test_missing_events_matches_is_missing_event(extra):
    values = np.empty(len(LABELS) + len(extra), dtype=object)
    values[:] = LABELS + extra
    expected = np.array([is_missing_event(v) for v in values], dtype=bool)
    np.testing.assert_array_equal(missing_events(values), expected)


def test_fit_does_not_scan_the_times_per_row(monkeypatch):
    # The regression: no per-row ``np.where`` search, and no per-row
    # ``is_missing_event`` call.
    def refuse(*args, **kwargs):
        raise AssertionError("per-row call")

    numpy_without_where = types.SimpleNamespace(
        **{name: getattr(np, name) for name in dir(np) if name != "where"}
    )
    numpy_without_where.where = refuse
    monkeypatch.setattr(cr_module, "np", numpy_without_where)
    monkeypatch.setattr(data_formats, "is_missing_event", refuse)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    e = ["a", "b", "a", None, "a", "b", "a", None, "b", "a"]
    model = CompetingRisks.fit(x, e)
    assert model.d_e.sum() == 8
