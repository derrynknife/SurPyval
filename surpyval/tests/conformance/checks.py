"""Property checks shared by the conformance suite and the Hypothesis
properties (#379).

The conformance modules apply these to each registered case's fixed
fixture; ``surpyval/tests/properties`` applies the same functions to
generated data (a registered case with its ``data`` replaced), so a
property is written once whichever way its data arrive.

- :func:`rescaled`, :func:`permuted` and :func:`expanded` rewrite a
  case's data dict (a change of time unit, a permutation of the rows,
  counts replaced by repeated rows), and :func:`compare` compares the
  predictions of the refit with the original's;
- :func:`check_valid` checks one function's values against its rule in
  :data:`RULES` (range and direction in time);
- :func:`check_round_trip` checks the strict-JSON ``to_dict`` /
  ``from_dict`` round trip of a fitted model.
"""

import json

import numpy as np

import surpyval
from surpyval.serialisation import NON_FINITE_KEY, SCHEMA_VERSION
from surpyval.tests.conformance.registry import predictions, scramble

# ---------------------------------------------------------------------------
# Metamorphic rewrites of the data, and the comparison of two fits
# ---------------------------------------------------------------------------
# How each function scales with the time unit: value(K x) * K**power
# in the new unit equals value(x) in the old one.
_POWER = {"hf": 1, "df": 1, "iif": 1, "pdf": 2, "qf": -1}


def unit_power(case, key):
    """The power of the time unit that ``key``'s values carry."""
    fname = key.split("[")[0]
    if fname in case.jump_functions:
        return 0  # a jump is a probability, whatever the unit
    return _POWER.get(fname, 0)


def compare(case, got, ref, scale=None, rtol=None, atol=None):
    """Assert two :func:`~registry.predictions` dicts agree.

    ``scale`` is the unit change ``got`` was fitted in (see
    :func:`unit_power`); the tolerances default to the case's ``rtol``.
    """
    rtol = case.rtol if rtol is None else rtol
    atol = rtol * 1e-2 if atol is None else atol
    assert got.keys() == ref.keys()
    for key in ref:
        g = got[key]
        if scale is not None:
            g = g * scale ** unit_power(case, key)
        np.testing.assert_allclose(
            g, ref[key], rtol=rtol, atol=atol, err_msg=key
        )


def rescaled(case, data, k):
    """``data`` with the case's time entries multiplied by ``k``."""
    out = dict(data)
    for key in case.times:
        if key in data:
            out[key] = np.asarray(data[key], dtype=float) * k
    return out


def permuted(case, data, perm=None):
    """``data`` with the case's per-row entries permuted together.

    ``perm`` defaults to :func:`~registry.scramble` of the row count.
    """
    size = len(data[case.rows[0]])
    perm = scramble(size) if perm is None else np.asarray(perm)
    return {
        key: np.asarray(v)[perm] if key in case.rows else v
        for key, v in data.items()
    }


def expanded(case, data):
    """``data`` with each row repeated ``n`` times and ``n`` dropped."""
    n = np.asarray(data["n"])
    return {
        key: np.repeat(np.asarray(v), n, axis=0) if key in case.rows else v
        for key, v in data.items()
        if key != "n"
    }


# ---------------------------------------------------------------------------
# Valid values
# ---------------------------------------------------------------------------
TOL = 1e-12

# function -> (lower, upper, direction in time: +1, -1 or 0)
RULES = {
    "sf": (0.0, 1.0, -1),
    "ff": (0.0, 1.0, 1),
    "Hf": (0.0, np.inf, 1),
    "hf": (0.0, np.inf, 0),
    "df": (0.0, np.inf, 0),
    "cif": (0.0, 1.0, 1),
    "mcf": (0.0, np.inf, 1),
    "iif": (0.0, np.inf, 0),
    "qf": (-np.inf, np.inf, 1),
}
# For recurrent events ``cif`` is a cumulative intensity, not a probability.
COUNTING_RULES = dict(RULES, cif=(0.0, np.inf, 1))


def check_valid(name, values, rule):
    """Assert ``values`` (at increasing query points) obey ``rule``."""
    lower, upper, direction = rule
    values = np.asarray(values, dtype=float)
    if name == "qf":
        # A step estimate that never falls to 1 - q has no q-quantile;
        # that is NaN (as R's quantile gives NA), and the rest must rise.
        values = values[~np.isnan(values)]
    assert not np.any(np.isnan(values)), f"{name}: NaN at a valid time"
    assert np.all(values >= lower - TOL), f"{name} below {lower}: {values}"
    assert np.all(values <= upper + TOL), f"{name} above {upper}: {values}"
    if direction > 0:
        rising = values[1:] >= values[:-1] - 1e-10
        assert np.all(rising), f"{name} not non-decreasing: {values}"
    elif direction < 0:
        falling = values[1:] <= values[:-1] + 1e-10
        assert np.all(falling), f"{name} not non-increasing: {values}"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def _has_key(obj, key):
    if isinstance(obj, dict):
        return key in obj or any(_has_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_key(v, key) for v in obj)
    return False


def check_round_trip(case, model, x=None, Z=None):
    """Assert ``model`` survives a strict-JSON ``to_dict`` / ``from_dict``
    round trip: the document is strict JSON stamped with a schema that
    reads it, and the restored model is of the same class and predicts
    exactly what ``model`` does (at ``x`` and ``Z``, by default the
    case's query)."""
    d = model.to_dict()
    text = json.dumps(d, allow_nan=False)
    schema = d["schema"]
    assert isinstance(schema, int) and 1 <= schema <= SCHEMA_VERSION
    if _has_key(d, NON_FINITE_KEY):
        # A reader of an older schema would take the nulls for missing
        # entries, so such a document must be stamped with the newest.
        assert schema == SCHEMA_VERSION

    restored = surpyval.from_dict(json.loads(text))
    expected = model if isinstance(model, type) else type(model)
    got = restored if isinstance(restored, type) else type(restored)
    assert got is expected

    ref = predictions(case, model, x=x, Z=Z)
    new = predictions(case, restored, x=x, Z=Z)
    for key in ref:
        np.testing.assert_array_equal(new[key], ref[key], err_msg=key)
