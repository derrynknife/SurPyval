"""The one-pass paths of ``encode_non_finite`` / ``decode_non_finite``
(#515).

The encoder used to walk every element of a model's arrays with a
recursive call and a JSON Pointer string each; it now handles numeric
arrays and lists in one pass and builds pointers only for the non-finite
entries. The decoder returns number lists untouched and visits only the
items a record names. The documents must not change by a byte, and
neither must what the readers restore or the errors they raise: these
tests hold the new code to private copies of the old walks, on every
serialisable model in the conformance registry and on edge cases.
"""

import copy
import json
import math

import numpy as np
import pytest

import surpyval
import surpyval.serialisation as ser
from surpyval.serialisation import (
    NON_FINITE_KEY,
    decode_non_finite,
    encode_non_finite,
)
from surpyval.tests.conformance.registry import cases_for, fitted

# --- the element-by-element walks this replaced ---------------------------

_KINDS = {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}


def _old_encode(value, pointer, found):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {
            k: _old_encode(v, f"{pointer}/{ser._pointer_token(k)}", found)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = [
            _old_encode(v, f"{pointer}/{j}", found)
            for j, v in enumerate(value)
        ]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, float) and not math.isfinite(value):
        found[ser._non_finite_kind(value)].append(pointer)
        return None
    return value


def _old_encode_non_finite(model_dict):
    found = {kind: [] for kind in _KINDS}
    for key in list(model_dict):
        if key != NON_FINITE_KEY:
            model_dict[key] = _old_encode(
                model_dict[key], "/" + ser._pointer_token(key), found
            )
    if any(found.values()):
        record = dict(model_dict.get(NON_FINITE_KEY) or {})
        for kind, pointers in found.items():
            if pointers:
                record[kind] = list(record.get(kind, [])) + pointers
        model_dict[NON_FINITE_KEY] = record
    return model_dict


def _old_parse_record(record):
    corrupt = ser._corrupt_record
    if not isinstance(record, dict):
        raise corrupt("it is not a dictionary")
    trie = {}
    for kind, pointers in record.items():
        if kind not in _KINDS:
            raise corrupt(
                f"unknown kind {kind!r}, expected one of {list(_KINDS)}"
            )
        if not isinstance(pointers, list):
            raise corrupt(f"the {kind!r} entry is not a list")
        for pointer in pointers:
            if not isinstance(pointer, str) or not pointer.startswith("/"):
                raise corrupt(f"{pointer!r} is not a JSON Pointer")
            tokens = [
                t.replace("~1", "/").replace("~0", "~")
                for t in pointer[1:].split("/")
            ]
            node = trie
            for token in tokens[:-1]:
                node = node.setdefault(token, {})
                if not isinstance(node, dict):
                    raise corrupt(f"{pointer!r} overlaps a value")
            if tokens[-1] in node:
                raise corrupt(f"{pointer!r} is listed twice")
            node[tokens[-1]] = _KINDS[kind]
    return trie


def _old_decode(value, trie, where):
    corrupt = ser._corrupt_record
    if isinstance(trie, float):
        if value is not None:
            raise corrupt(f"it names {where}, which holds {value!r}, not null")
        return trie
    if isinstance(value, dict):
        if NON_FINITE_KEY in value:
            own = _old_parse_record(value[NON_FINITE_KEY])
            trie = own if trie is None else ser._merge_tries(trie, own)
        if trie is None:
            children = {
                k: _old_decode(v, None, f"{where}/{k}")
                for k, v in value.items()
            }
        else:
            children = {}
            for k, v in value.items():
                if k != NON_FINITE_KEY:
                    children[k] = _old_decode(
                        v, trie.get(str(k)), f"{where}/{k}"
                    )
            missing = set(trie) - {str(k) for k in value}
            if missing:
                raise corrupt(
                    f"it names {where}/{sorted(missing)[0]}, which does"
                    " not exist"
                )
        if children.keys() == value.keys() and all(
            children[k] is value[k] for k in value
        ):
            return value
        return children
    if isinstance(value, (list, tuple)):
        if trie is None:
            items = [_old_decode(v, None, where) for v in value]
        else:
            items = [
                _old_decode(v, trie.get(str(j)), f"{where}/{j}")
                for j, v in enumerate(value)
            ]
            missing = set(trie) - {str(j) for j in range(len(value))}
            if missing:
                raise corrupt(
                    f"it names {where}/{sorted(missing)[0]}, which does"
                    " not exist"
                )
        if all(a is b for a, b in zip(items, value)):
            return value
        return tuple(items) if isinstance(value, tuple) else items
    if trie is not None and value is not None:
        raise corrupt(f"it names a path inside the value at {where}")
    return value


def _old_decode_non_finite(model_dict):
    return _old_decode(model_dict, None, "")


# --- helpers ---------------------------------------------------------------


def _same(a, b):
    """Equal values of equal types all the way down, nan equal to nan."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return a == b


def _check_encoding(raw):
    """Old and new encoders give byte-identical JSON for ``raw``, and the
    old and new decoders restore identical values from it."""
    old = _old_encode_non_finite(copy.deepcopy(raw))
    new = encode_non_finite(copy.deepcopy(raw))
    text = json.dumps(old, allow_nan=False)
    assert json.dumps(new, allow_nan=False) == text
    assert _same(new, old)
    _check_decoding(json.loads(text))
    return text


def _check_decoding(document):
    """The old and new decoders agree on ``document``: the same values,
    the input returned as is exactly when the old one did, the same
    error."""
    try:
        old = _old_decode_non_finite(document)
    except ValueError as error:
        with pytest.raises(ValueError) as caught:
            decode_non_finite(document)
        assert str(caught.value) == str(error)
        return
    new = decode_non_finite(document)
    assert _same(new, old)
    assert (new is document) == (old is document)


def _as_arrays(value):
    """``value`` with its number lists as numpy arrays, as a model's
    ``to_dict`` holds them before it is encoded."""
    if isinstance(value, dict):
        return {k: _as_arrays(v) for k, v in value.items()}
    if isinstance(value, list) and value:
        try:
            array = np.array(value)
        except ValueError:
            array = None
        if array is not None and array.dtype.kind in "biuf":
            return array
        return [_as_arrays(v) for v in value]
    return value


# --- every serialisable model in the registry ------------------------------


@pytest.fixture
def capture_encoder_input(monkeypatch):
    """Record a deep copy of what each ``to_dict`` hands the encoder."""
    captured = []
    real = ser.encode_non_finite

    def capturing(model_dict, stamped=False):
        # A tree or forest takes its finished leaves or trees as they are
        # (stamped=True, #549); the old encoder walks them again, to the
        # same document.
        captured.append(copy.deepcopy(model_dict))
        return real(model_dict, stamped=stamped)

    monkeypatch.setattr(ser, "encode_non_finite", capturing)
    return captured


@pytest.mark.parametrize("case", cases_for("serialise"))
def test_registered_models_byte_identical(case, capture_encoder_input):
    model = fitted(case)
    try:
        document = model.to_dict(with_data=True)
    except TypeError:
        document = model.to_dict()
    # The outermost call comes last; nested models' calls come first.
    raw = capture_encoder_input[-1]
    old = _old_encode_non_finite(copy.deepcopy(raw))
    old["schema"] = ser.required_schema(old)
    text = json.dumps(old, allow_nan=False)
    assert json.dumps(document, allow_nan=False) == text
    _check_decoding(json.loads(text))
    # The same document with every number list as an array.
    raw_arrays = _as_arrays(_old_decode_non_finite(json.loads(text)))
    _check_encoding(raw_arrays)
    restored = surpyval.from_dict(json.loads(text))
    assert type(restored) is type(model) or restored is model


# --- edge cases ------------------------------------------------------------

_LONG = [0.5, 1.5, np.inf, -0.0, np.nan, 2.0, -np.inf, 3.0, 1e308, 4.0]


@pytest.mark.parametrize(
    "value",
    [
        np.array(_LONG),
        np.array(_LONG[:8], dtype=np.float32),
        np.array(_LONG).reshape(2, 5),
        np.array(_LONG * 3).reshape(3, 5, 2),
        np.array([[0.0, np.inf]] * 12),
        np.array([[-np.inf, np.inf]] * 12).tolist(),
        np.arange(12),
        np.arange(12).reshape(3, 4),
        np.array([True, False] * 5),
        np.array([]),
        np.zeros((0, 2)),
        np.zeros((2, 0)),
        np.array(3.5),
        np.array(np.nan),
        np.array(["a", "b", "c"] * 4, dtype=object),
        np.array(["pump", "valve", None] * 4, dtype=object),
        np.array([1, "x", 2.5, np.inf] * 3, dtype=object),
        np.array([0.5, np.nan], dtype=object),
        np.float64(np.inf),
        np.int64(3),
        _LONG,
        _LONG[:5],
        list(np.array(_LONG)),
        [float(v) for v in _LONG] + [1],
        list(range(20)),
        [True, False] * 6,
        ["a", None, "b", 1, 2.0] * 3,
        [10**400] * 9,
        [[0.5, np.inf]] * 10,
        [[0.5, np.inf], [0.5]] * 5,
        [[0.5, np.inf], (0.5, np.nan)] * 5,
        [[1, 2]] * 10,
        [[]] * 10,
        [[[0.5, np.inf]] * 3] * 9,
        tuple(_LONG),
        [_LONG, {"a": np.array([np.nan] * 9)}],
        {"deep": {"H": np.array(_LONG), "labels": ["x/y", "t~0"]}},
        [],
        None,
        "text",
    ],
)
def test_edge_values_byte_identical(value):
    _check_encoding({"v": value, "w": copy.deepcopy(value)})


def test_existing_record_is_extended():
    raw = {
        "inner": {"H": [0.5, None], NON_FINITE_KEY: {"inf": ["/H/1"]}},
        "H": np.array(_LONG),
        NON_FINITE_KEY: {"nan": ["/before/0"]},
    }
    _check_encoding(raw)


def test_keys_needing_escapes():
    _check_encoding({"a/b": np.array(_LONG), "c~d": [np.inf] * 9})


def test_decoder_returns_number_lists_as_they_are():
    x = [0.5, 1.5, 2.5] * 10
    document = {"x": x, "H": [0.5, None], NON_FINITE_KEY: {"inf": ["/H/1"]}}
    restored = decode_non_finite(document)
    assert restored["x"] is x
    assert restored["H"] == [0.5, math.inf]
    plain = {"x": x, "c": [0, 1] * 10}
    assert decode_non_finite(plain) is plain


_RECORDS = [
    {"inf": ["/v/3"]},
    {"inf": ["/v/0"]},
    {"inf": ["/v/30"]},
    {"inf": ["/v/03"]},
    {"inf": ["/v/-1"]},
    {"inf": ["/v/x"]},
    {"inf": ["/v/٣"]},
    {"inf": ["/v/3/0"]},
    {"inf": ["/v/0/0"]},
    {"inf": ["/v/3", "/v/0"], "nan": ["/v/99"]},
    {"inf": ["/v/99", "/v/0"]},
    {"inf": ["/v/2", "/v/0"]},
    {"inf": ["/v/2"], "nan": ["/v/0"]},
    {"inf": ["/rows/0/1", "/rows/0/0"]},
    {"-inf": ["/v/3"], "nan": ["/v/7"]},
    {"inf": ["/v"]},
    {"inf": ["/rows/2/1", "/rows/4/0"]},
    {"inf": ["/rows/2/1", "/rows/0/1"]},
    {"inf": ["/rows/2/5"]},
    {"inf": ["/rows/2"]},
    {"inf": ["/rows/2/1/0"]},
    {"inf": ["/rows/20/1"]},
    {"inf": ["/nested/0/a"]},
    {"inf": ["/t/1"]},
    {"inf": ["/missing"]},
]


@pytest.mark.parametrize("record", _RECORDS)
def test_decoder_errors_and_results_unchanged(record):
    values = [0.5, 1.5, 2.5, None, 3.5, 4.5, 5.5, None, 6.5]
    rows = [[0.5, 1.5], [2.5, None], [None, None], [0.5, 1.5], [None, 3.5]]
    document = {
        "v": values,
        "rows": rows,
        "nested": [{"a": None}, 1.0],
        "t": (0.5, None),
        NON_FINITE_KEY: record,
    }
    _check_decoding(document)


# --- the schema version from one walk (performance sweep) -----------------


def _old_required_schema(model_dict):
    # The four walks of every value, item by item, as they were
    def walk(value, test):
        if isinstance(value, dict):
            return test(value) or any(walk(v, test) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(walk(v, test) for v in value)
        return False

    def center(d):
        c = d.get("center")
        return isinstance(c, (list, tuple)) and any(v != 0 for v in c)

    def formula(d):
        meta = d.get("formula_meta")
        return isinstance(meta, dict) and "factor_levels" not in meta

    def support(d):
        return any(d.get(k) is not None for k in ("support", "band_n"))

    tests = (lambda d: NON_FINITE_KEY in d, formula, support, center)
    return 2 if any(walk(model_dict, t) for t in tests) else 1


@pytest.mark.parametrize("case", cases_for("serialise"))
def test_registered_models_schema_unchanged(case):
    model = fitted(case)
    try:
        document = model.to_dict(with_data=True)
    except TypeError:
        document = model.to_dict()
    assert ser.required_schema(document) == _old_required_schema(document)


@pytest.mark.parametrize(
    "document",
    [
        {"x": [1.0, 2.0]},
        {"a": [[1.0, {"center": [0.0, 3.0]}]]},
        {"a": ({"support": None, "band_n": [1]},)},
        {"m": [{"formula_meta": {"terms": []}}]},
        {"m": [{"formula_meta": {"factor_levels": {}}}]},
        {"t": [[1.0, None], [2.0, 3.0]], "center": [0.0]},
        {"deep": [[[{"non_finite": {}}]]]},
    ],
)
def test_edge_documents_schema_unchanged(document):
    assert ser.required_schema(document) == _old_required_schema(document)


def test_a_list_of_numbers_is_walked_once():
    # The four checks each walked a model's data value by value: 80% of
    # saving a Kaplan-Meier model of 1e5 rows with its data.
    passes = []

    class Counted(list):
        def __iter__(self):
            passes.append(1)
            return super().__iter__()

    ser.required_schema({"x": Counted([1.0] * 1000), "c": [0] * 1000})
    assert len(passes) <= 1


def test_a_cox_model_is_read_without_autograd_arrays(monkeypatch):
    # autograd's ``array`` inspects a list item by item for boxes: 90% of
    # reading a Cox model of 1e5 rows
    import autograd.numpy as anp

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(500, 2))
    x = rng.exponential(size=500) * np.exp(-Z[:, 0])
    model = surpyval.CoxPH.fit(x=x, Z=Z, c=rng.choice([0, 1], 500))
    text = model.to_json()
    array = anp.array

    def refusing(value, *args, **kwargs):
        assert not (isinstance(value, list) and len(value) > 100)
        return array(value, *args, **kwargs)

    monkeypatch.setattr(anp, "array", refusing)
    restored = surpyval.from_json(text)
    for name in ("x", "r", "d", "h0", "H0", "beta", "center"):
        np.testing.assert_array_equal(
            getattr(restored, name), getattr(model, name)
        )
