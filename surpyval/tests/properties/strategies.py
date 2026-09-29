"""Hypothesis strategies for surpyval's data model (#379).

Every strategy draws a *valid* data set as the keyword arguments of a
fitter (a dict), so a test calls ``fitter.fit(**data)``. Times are drawn
from a coarse grid (multiples of :data:`STEP`) rather than from the real
line: that makes ties -- between events, and between an event and a
censoring -- the common case rather than a coincidence, and it makes a
shrunk counterexample readable. Hypothesis shrinks towards fewer rows,
smaller times, exact observations, counts of one and no truncation, so a
failure is reported in the simplest form that still fails.

- :func:`xcnt`: univariate ``x, c, n`` with any mix of exact (0), right
  (1), left (-1) and interval (2) censoring, and optional left / right
  truncation consistent with each row (``tl < x``, or ``tl <= xl`` for an
  interval; ``x <= tr``, strictly for a right censored row, and
  ``xr <= tr``). Tiny samples (one row) and degenerate ones (all
  censored, one distinct time) are in its range.
- :func:`regression`: ``x, Z, c, n`` with numeric covariates, optionally
  a constant column, and a categorical label ``g`` for the formula path.
- :func:`competing_risks`: ``x, e, n``, each row a cause or censored
  (``None``).
- :func:`xicn`: recurrent-event data, ``x, i, c`` -- per item, event
  times ending with its right censored end of observation.
- :func:`invalid_xcnt`: one invalid input of each kind the handlers must
  refuse with a ``ValueError``.
"""

import os

import numpy as np
from hypothesis import strategies as st

# Whether the thorough (nightly) profile is running (see conftest.py). The
# default profile draws smaller data sets and runs the slower properties
# on fewer models, to stay within a minute.
THOROUGH = os.environ.get("SURPYVAL_HYPOTHESIS_PROFILE") == "nightly"
MAX_ROWS = 10 if THOROUGH else 5

STEP = 0.5  # the grid of times
MAX_TICK = 30  # times are STEP * (1 .. MAX_TICK)

EXACT, RIGHT, LEFT, INTERVAL = 0, 1, -1, 2
# Ordered so that shrinking goes to the simplest kind first.
ALL_CENSORING = (EXACT, RIGHT, LEFT, INTERVAL)
RIGHT_CENSORING = (EXACT, RIGHT)


@st.composite
def _xcnt_row(draw, censoring, left_truncation, right_truncation, counts):
    """One row: ``(xl, xr, c, n, tl, tr)``, times on the grid (``tl`` and
    ``tr`` infinite when the row is not truncated). A failing example is
    printed as a list of these."""
    c = draw(st.sampled_from(censoring))
    lo = draw(st.integers(1, MAX_TICK))
    hi = lo + draw(st.integers(1, 8)) if c == INTERVAL else lo
    n = draw(st.integers(1, 3)) if counts else 1
    tl, tr = -np.inf, np.inf
    if left_truncation and draw(st.booleans()):
        # Strictly below a single value; an interval may start at tl.
        tl = draw(st.integers(0, lo if c == INTERVAL else lo - 1))
    if right_truncation and draw(st.booleans()):
        # At or above the value; strictly above a right censored one.
        tr = draw(st.integers(hi + (c == RIGHT), hi + 8))
    return lo * STEP, hi * STEP, c, n, tl * STEP, tr * STEP


def _assemble(rows, truncation_keys=True):
    lo, hi, c, n, tl, tr = (np.array(col) for col in zip(*rows))
    out = {
        "x": np.column_stack([lo, hi]) if (c == INTERVAL).any() else lo,
        "c": c.astype(int),
        "n": n.astype(int),
    }
    tl = tl.astype(float)
    tr = tr.astype(float)
    if truncation_keys and np.isfinite(tl).any():
        out["tl"] = tl
    if truncation_keys and np.isfinite(tr).any():
        out["tr"] = tr
    return out


def xcnt(
    censoring=ALL_CENSORING,
    left_truncation=True,
    right_truncation=True,
    counts=True,
    min_rows=1,
    max_rows=MAX_ROWS,
):
    """Univariate data: a dict with ``x``, ``c``, ``n`` and, when any row
    is truncated, ``tl`` and / or ``tr`` (infinite for the other rows).

    ``x`` has two columns (``[xl, xr]``, equal for a non-interval row)
    when some row is interval censored, and one otherwise.
    """
    row = _xcnt_row(censoring, left_truncation, right_truncation, counts)
    return st.lists(row, min_size=min_rows, max_size=max_rows).map(_assemble)


def has_distinct_failures(data, k=2):
    """Whether ``data`` has at least ``k`` distinct observed values that
    are not right censored (what a ``k``-parameter fit needs)."""
    x = np.asarray(data["x"], dtype=float)
    lo = x if x.ndim == 1 else x[:, 0]
    hi = x if x.ndim == 1 else x[:, 1]
    keep = np.asarray(data["c"]) != RIGHT
    return np.unique(np.r_[lo[keep], hi[keep]]).size >= k


def permutations(size):
    """A permutation of ``range(size)``."""
    return st.permutations(list(range(size))).map(np.array)


@st.composite
def with_permutation(draw, data_strategy, key="x"):
    """``(data, perm)``: a data set and a permutation of its rows."""
    data = draw(data_strategy)
    return data, draw(permutations(len(data[key])))


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
_COVARIATE = st.integers(-4, 4).map(lambda k: k * 0.5)


@st.composite
def regression(
    draw,
    min_rows=4,
    max_rows=2 * MAX_ROWS,
    columns=(1, 2),
    constant_column=True,
    categorical=False,
    counts=True,
):
    """``x, Z, c, n`` (exact or right censored) with numeric covariates.

    ``Z`` has ``columns`` numeric columns, and, if ``constant_column``
    allows it, a column of ones appended; with ``categorical``, ``g`` is
    a label ("a", "b" or "c") per row for the formula path. At least
    two distinct observed times are always present.
    """
    size = draw(st.integers(min_rows, max_rows))
    k = draw(st.sampled_from(columns))
    ticks = st.integers(1, MAX_TICK)
    x = np.array(draw(st.lists(ticks, min_size=size, max_size=size))) * STEP
    c = np.array(
        draw(st.lists(st.sampled_from((0, 1)), min_size=size, max_size=size))
    )
    # Two distinct observed times: the first two rows, forced apart.
    c[:2] = 0
    if x[0] == x[1]:
        x[1] = x[0] + STEP
    n = np.ones(size, int)
    if counts:
        n = np.array(
            draw(st.lists(st.integers(1, 3), min_size=size, max_size=size))
        )
    Z = np.array(
        draw(
            st.lists(
                st.lists(_COVARIATE, min_size=k, max_size=k),
                min_size=size,
                max_size=size,
            )
        ),
        dtype=float,
    )
    if constant_column and draw(st.booleans()):
        Z = np.column_stack([Z, np.ones(size)])
    out = {"x": x, "Z": Z, "c": c, "n": n}
    if categorical:
        out["g"] = np.array(
            draw(
                st.lists(st.sampled_from("abc"), min_size=size, max_size=size)
            ),
            dtype=object,
        )
    return out


def query_rows(columns, size=4):
    """``size`` covariate rows of ``columns`` values each."""
    return st.lists(
        st.lists(_COVARIATE, min_size=columns, max_size=columns),
        min_size=size,
        max_size=size,
    ).map(lambda rows: np.array(rows, dtype=float))


# ---------------------------------------------------------------------------
# Competing risks and recurrent events
# ---------------------------------------------------------------------------
@st.composite
def competing_risks(
    draw, min_rows=1, max_rows=MAX_ROWS + 2, causes=("a", "b")
):
    """``x, e, n``: each row fails of one of ``causes`` or is censored
    (``e`` is ``None``)."""
    marks = st.sampled_from(tuple(causes) + (None,))
    rows = draw(
        st.lists(
            st.tuples(st.integers(1, MAX_TICK), marks, st.integers(1, 3)),
            min_size=min_rows,
            max_size=max_rows,
        )
    )
    x, e, n = zip(*rows)
    return {
        "x": np.array(x, dtype=float) * STEP,
        "e": np.array(e, dtype=object),
        "n": np.array(n, dtype=int),
    }


@st.composite
def xicn(draw, max_items=4, max_events=5):
    """Recurrent events ``x, i, c``: per item, distinct event times
    (``c = 0``) and a right censored end of observation (``c = 1``) at or
    after the last of them."""
    items = draw(st.integers(1, max_items))
    x, i, c = [], [], []
    for item in range(items):
        events = sorted(
            draw(
                st.sets(
                    st.integers(1, MAX_TICK),
                    min_size=0 if item else 1,
                    max_size=max_events,
                )
            )
        )
        last = events[-1] if events else 1
        end = last + draw(st.integers(0, 6))
        x += events + [end]
        i += [item + 1] * (len(events) + 1)
        c += [0] * len(events) + [1]
    return {
        "x": np.array(x, dtype=float) * STEP,
        "i": np.array(i),
        "c": np.array(c),
    }


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------
INVALID_KINDS = (
    "negative count",
    "zero count",
    "fractional count",
    "tl above x",
    "tl equal to x",
    "tr below x",
    "short c",
    "short n",
    "short tl",
    "nan time",
    "inf exact time",
    "unknown flag",
)


@st.composite
def invalid_xcnt(draw, kinds=INVALID_KINDS):
    """``(kind, data)``: exact / right censored data with one defect.

    The defect is one of :data:`INVALID_KINDS`, applied to one row (or
    one array); every other row stays valid.
    """
    data = draw(
        xcnt(
            censoring=RIGHT_CENSORING,
            left_truncation=False,
            right_truncation=False,
            min_rows=2,
        )
    )
    kind = draw(st.sampled_from(kinds))
    size = len(data["x"])
    k = draw(st.integers(0, size - 1))
    x = np.asarray(data["x"], dtype=float).copy()
    c = np.asarray(data["c"]).copy()
    n = np.asarray(data["n"]).copy()
    if kind == "negative count":
        n[k] = -draw(st.integers(1, 3))
    elif kind == "zero count":
        n[k] = 0
    elif kind == "fractional count":
        n = n.astype(float)
        n[k] = n[k] + 0.5
    elif kind == "tl above x":
        tl = np.zeros(size)
        tl[k] = x[k] + STEP
        data["tl"] = tl
    elif kind == "tl equal to x":
        tl = np.zeros(size)
        tl[k] = x[k]
        data["tl"] = tl
    elif kind == "tr below x":
        tr = np.full(size, np.inf)
        tr[k] = x[k] - STEP
        data["tr"] = tr
    elif kind == "short c":
        c = c[:-1]
    elif kind == "short n":
        n = n[:-1]
    elif kind == "short tl":
        data["tl"] = np.zeros(size - 1)
    elif kind == "nan time":
        x[k] = np.nan
    elif kind == "inf exact time":
        x[k] = np.inf
        c[k] = EXACT
    elif kind == "unknown flag":
        c[k] = 3
    data.update(x=x, c=c, n=n)
    return kind, data
