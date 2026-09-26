"""Package-level readers for serialised SurPyval models.

Every serialisable fitted SurPyval model writes ``to_dict`` /
``to_json``; the matching class-level ``from_dict`` / ``from_json``
readers require knowing up front which class wrote the file. The
package-level readers here dispatch on the serialised dictionary
itself, so one call restores a model of the right class no matter
which model wrote it:

.. code:: python

    import surpyval

    model = surpyval.from_json("model.json")  # any model's file
    model = surpyval.from_dict(model_dict)    # any model's dict

Dispatch uses the two conventions in the serialised dictionaries:
most classes write a ``"model"`` tag equal to their class name, and
the core univariate families are identified by their
``"parameterization"`` (``"parametric"``, ``"non-parametric"`` or
``"parametric-regression"``).

The dictionaries are strict JSON: a non-finite float (``inf``, ``-inf``
or ``nan``) is written as ``null``, and the dictionary that holds it
records its meaning under ``"non_finite"`` -- ``{"inf": [...], "-inf":
[...], "nan": [...]}``, each list holding the RFC 6901 JSON Pointers
(relative to that dictionary) of the values of that kind. Every reader
puts the original values back; see :func:`encode_non_finite`.
"""

import functools
import inspect
import json
import math
import numbers
import os
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

# ``"model"`` tag -> defining module. The tag every class writes into
# its dict is its own class name, so the registry only needs to find
# the defining module, lazily (importing everything eagerly here would
# be a heavy import and a cycle risk).
_TAGGED_MODELS: dict[str, str] = {
    "SemiParametricRegressionModel": (
        "surpyval.univariate.regression.semi_parametric_regression_model"
    ),
    "FrailtyModel": ("surpyval.univariate.regression.frailty.frailty_model"),
    "AdditiveHazardsModel": (
        "surpyval.univariate.regression.additive_hazards.additive_hazards"
    ),
    "BuckleyJamesModel": (
        "surpyval.univariate.regression.buckley_james.buckley_james"
    ),
    "MixtureModel": "surpyval.univariate.parametric.mixture_model",
    "RoystonParmarModel": ("surpyval.univariate.parametric.royston_parmar"),
    "FineGrayModel": (
        "surpyval.univariate.competing_risks.regression.fine_gray"
    ),
    "CompetingRisksProportionalHazards": (
        "surpyval.univariate.competing_risks.regression"
        ".competing_risks_proportional_hazard"
    ),
    "ParametricCompetingRisks": (
        "surpyval.univariate.competing_risks.parametric"
        ".parametric_competing_risks"
    ),
    "CompetingRisks": (
        "surpyval.univariate.competing_risks.nonparametric.competing_risks"
    ),
    "CauseSpecificMCF": (
        "surpyval.recurrent.competing_risks.nonparametric.cause_specific_mcf"
    ),
    "CauseSpecificNHPP": (
        "surpyval.recurrent.competing_risks.parametric.cause_specific_nhpp"
    ),
    "NonParametricCounting": "surpyval.recurrent.nonparametric.mcf",
    "ParametricRecurrenceModel": (
        "surpyval.recurrent.parametric.parametric_recurrence"
    ),
    "ProportionalIntensityModel": (
        "surpyval.recurrent.regression.proportional_intensity"
    ),
    "RenewalModel": "surpyval.recurrent.renewal.renewal_model",
    "DegradationModel": "surpyval.degradation.degradation_analysis",
    "InducedFailureDistribution": (
        "surpyval.degradation.degradation_analysis"
    ),
    "WienerProcessModel": "surpyval.degradation.process_models",
    "GammaProcessModel": "surpyval.degradation.process_models",
    "DestructiveDegradationModel": "surpyval.degradation.destructive",
    "SurvivalTree": "surpyval.beta.ml.forest.tree",
    "RandomSurvivalForest": "surpyval.beta.ml.forest.forest",
    # The degenerate distributions are stateless: the class is the model,
    # so they serialise by name alone.
    "NeverOccurs": ("surpyval.univariate.parametric.distributions.degenerate"),
    "InstantlyOccurs": (
        "surpyval.univariate.parametric.distributions.degenerate"
    ),
}

# ``"parameterization"`` value -> (defining module, class name), for
# the core univariate families, which carry no ``"model"`` class tag.
# (The non-parametric dicts do have a ``"model"`` key, but it holds the
# estimator name -- e.g. ``"Kaplan-Meier"`` -- not a class name.)
_PARAMETERIZATIONS: dict[str, tuple[str, str]] = {
    "parametric": (
        "surpyval.univariate.parametric.parametric",
        "Parametric",
    ),
    "non-parametric": (
        "surpyval.univariate.nonparametric.nonparametric",
        "NonParametric",
    ),
    "parametric-regression": (
        "surpyval.univariate.regression.parametric_regression_model",
        "ParametricRegressionModel",
    ),
    "copula": (
        "surpyval.multivariate.parametric.copula.copula_model",
        "CopulaModel",
    ),
}


# The version of the serialised-dictionary layout, stamped into every
# ``to_dict`` output as ``"schema"``. Bump it only when a dictionary's
# shape changes incompatibly; the readers use it to recognise (and
# refuse, with a clear error) documents written by a newer SurPyval,
# and to migrate older layouts where needed. Documents with no
# ``"schema"`` key predate versioning and read as schema 0.
#
# Schema 2 writes non-finite floats as ``null`` plus a ``"non_finite"``
# record (see :func:`encode_non_finite`); schema 0 and 1 documents wrote
# them as the non-standard ``NaN`` / ``Infinity`` / ``-Infinity``
# literals, which Python's ``json`` (and BSON) read back as floats, so
# they need no migration. The bump makes an older SurPyval refuse a
# schema-2 document rather than read its ``null`` values as missing.
SCHEMA_VERSION = 2

# The key under which a serialised dictionary records the meaning of the
# ``null`` values that stand in for its non-finite floats.
NON_FINITE_KEY = "non_finite"

# ``"non_finite"`` kind -> the float it stands for, in the order the
# kinds are written.
_NON_FINITE_KINDS: dict[str, float] = {
    "inf": math.inf,
    "-inf": -math.inf,
    "nan": math.nan,
}


def require_model_tag(model_dict: dict, tag: str, human: str) -> None:
    """Reject a ``from_dict`` dictionary whose ``model`` tag is not ``tag``.

    Every serialisable model's ``from_dict`` starts with this check;
    each used to write out the three-line guard itself. ``human`` is the
    phrase for the thing being built ("a renewal model", "an MCF
    estimate", ...); the raised message always contains ``tag`` so a
    caller can match on the model name.
    """
    if model_dict.get("model") != tag:
        article = "an" if tag[0] in "AEIOU" else "a"
        raise ValueError(
            "Must create {} from {} {} dict".format(human, article, tag)
        )


def _pointer_token(key: Any) -> str:
    # RFC 6901 escaping: "~" first, so the "~1" written for "/" is not
    # itself re-escaped.
    return str(key).replace("~", "~0").replace("/", "~1")


def _non_finite_kind(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def _encode(value: Any, pointer: str, found: dict[str, list[str]]) -> Any:
    """``value`` with native types and its non-finite floats as ``None``,
    recording each replaced float's pointer in ``found``."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {
            k: _encode(v, f"{pointer}/{_pointer_token(k)}", found)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = [
            _encode(v, f"{pointer}/{j}", found) for j, v in enumerate(value)
        ]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, float) and not math.isfinite(value):
        found[_non_finite_kind(value)].append(pointer)
        return None
    return value


def encode_non_finite(model_dict: dict) -> dict:
    """Make a serialised dictionary strict JSON, in place.

    ``json.dumps`` writes ``inf``, ``-inf`` and ``nan`` as the literals
    ``Infinity``, ``-Infinity`` and ``NaN``, which are not JSON: strict
    parsers (JavaScript's ``JSON.parse``, many databases) refuse the
    whole document. Yet the values are meaningful in a fitted model --
    an untruncated bound, the cumulative hazard after the last death, an
    undefined variance -- so they cannot simply be dropped.

    The convention: each non-finite float is written as ``null``, and
    the dictionary records what every such ``null`` stood for under
    ``"non_finite"``, as lists of RFC 6901 JSON Pointers relative to the
    dictionary, grouped by kind::

        {"H": [0.1, 0.4, null], "greenwood": [0.01, 0.05, null],
         "non_finite": {"inf": ["/H/2"], "nan": ["/greenwood/2"]}}

    A ``null`` that no pointer names is an ordinary ``None``. Readers put
    the floats back with :func:`decode_non_finite`, which every
    ``from_dict`` does; a consumer in another language sees ``null``
    where no number applies and can use the record to recover the exact
    values. The kinds with no values are omitted, as is the record
    itself when the dictionary holds no non-finite float.

    Numpy arrays and scalars are converted to native Python types on the
    way, so the dictionary is also BSON-native. A record the dictionary
    already carries (``to_dict`` output re-encoded) is extended, and the
    records of nested model dictionaries are left in place.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.serialisation import encode_non_finite
    >>> encode_non_finite({"H": [0.5, np.inf], "var": np.nan})
    {'H': [0.5, None], 'var': None, 'non_finite': {'inf': ['/H/1'], 'nan': ['/var']}}
    """  # noqa: E501
    found: dict[str, list[str]] = {kind: [] for kind in _NON_FINITE_KINDS}
    for key in list(model_dict):
        if key != NON_FINITE_KEY:
            model_dict[key] = _encode(
                model_dict[key], "/" + _pointer_token(key), found
            )
    if any(found.values()):
        record = dict(model_dict.get(NON_FINITE_KEY) or {})
        for kind, pointers in found.items():
            if pointers:
                record[kind] = list(record.get(kind, [])) + pointers
        model_dict[NON_FINITE_KEY] = record
    return model_dict


def _corrupt_record(detail: str) -> ValueError:
    return ValueError(
        f"The serialised model's {NON_FINITE_KEY!r} record is corrupt:"
        f" {detail}."
    )


def _parse_record(record: Any) -> dict[str, Any]:
    """A ``"non_finite"`` record as a trie: path token -> sub-trie, with
    the restored float at each leaf."""
    if not isinstance(record, dict):
        raise _corrupt_record("it is not a dictionary")
    trie: dict[str, Any] = {}
    for kind, pointers in record.items():
        if kind not in _NON_FINITE_KINDS:
            raise _corrupt_record(
                f"unknown kind {kind!r}, expected one of"
                f" {list(_NON_FINITE_KINDS)}"
            )
        if not isinstance(pointers, list):
            raise _corrupt_record(f"the {kind!r} entry is not a list")
        for pointer in pointers:
            if not isinstance(pointer, str) or not pointer.startswith("/"):
                raise _corrupt_record(f"{pointer!r} is not a JSON Pointer")
            tokens = [
                t.replace("~1", "/").replace("~0", "~")
                for t in pointer[1:].split("/")
            ]
            node = trie
            for token in tokens[:-1]:
                node = node.setdefault(token, {})
                if not isinstance(node, dict):
                    raise _corrupt_record(f"{pointer!r} overlaps a value")
            if tokens[-1] in node:
                raise _corrupt_record(f"{pointer!r} is listed twice")
            node[tokens[-1]] = _NON_FINITE_KINDS[kind]
    return trie


def _merge_tries(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for token, sub in b.items():
        if token not in out:
            out[token] = sub
        elif isinstance(out[token], dict) and isinstance(sub, dict):
            out[token] = _merge_tries(out[token], sub)
        else:
            raise _corrupt_record("two records name the same value")
    return out


def _decode(value: Any, trie: Any, where: str) -> Any:
    """``value`` with the ``null`` values ``trie`` names restored; the
    same object when nothing inside it changes."""
    if isinstance(trie, float):
        if value is not None:
            raise _corrupt_record(
                f"it names {where}, which holds {value!r}, not null"
            )
        return trie
    if isinstance(value, dict):
        if NON_FINITE_KEY in value:
            own = _parse_record(value[NON_FINITE_KEY])
            trie = own if trie is None else _merge_tries(trie, own)
        if trie is None:
            children = {
                k: _decode(v, None, f"{where}/{k}") for k, v in value.items()
            }
        else:
            children = {}
            for k, v in value.items():
                if k != NON_FINITE_KEY:
                    children[k] = _decode(v, trie.get(str(k)), f"{where}/{k}")
            missing = set(trie) - {str(k) for k in value}
            if missing:
                raise _corrupt_record(
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
            items = [_decode(v, None, where) for v in value]
        else:
            items = [
                _decode(v, trie.get(str(j)), f"{where}/{j}")
                for j, v in enumerate(value)
            ]
            missing = set(trie) - {str(j) for j in range(len(value))}
            if missing:
                raise _corrupt_record(
                    f"it names {where}/{sorted(missing)[0]}, which does"
                    " not exist"
                )
        if all(a is b for a, b in zip(items, value)):
            return value
        return tuple(items) if isinstance(value, tuple) else items
    # A record naming values inside a null is let through: the whole
    # array was nulled (a hand edit, or a legacy layout rebuilt from a
    # newer dict), so there is nothing left to restore.
    if trie is not None and value is not None:
        raise _corrupt_record(f"it names a path inside the value at {where}")
    return value


def decode_non_finite(model_dict: dict) -> dict:
    """Undo :func:`encode_non_finite`: the dictionary with every ``null``
    its ``"non_finite"`` records name put back to ``inf``, ``-inf`` or
    ``nan``, and the records removed.

    The records of nested model dictionaries are applied too. The input
    is not modified; a dictionary without records (including those
    written before schema 2, whose non-finite values were stored as
    floats) is returned unchanged. A record naming a missing entry or a
    value that is not ``null`` raises a ``ValueError``.

    Examples
    --------
    >>> from surpyval.serialisation import decode_non_finite
    >>> decode_non_finite(
    ...     {"H": [0.5, None], "non_finite": {"inf": ["/H/1"]}}
    ... )
    {'H': [0.5, inf]}
    """
    return _decode(model_dict, None, "")


def stamp_schema(model_dict: dict) -> dict:
    """Finish a ``to_dict`` output: make it strict JSON (non-finite floats
    as ``null``, see :func:`encode_non_finite`) and stamp the
    serialisation schema version. Every ``to_dict`` ends with it."""
    encode_non_finite(model_dict)
    model_dict["schema"] = SCHEMA_VERSION
    return model_dict


def check_schema(model_dict: dict) -> int:
    """Return the ``"schema"`` version of a serialised dictionary, refusing
    one this SurPyval cannot read.

    The version is an integer (``to_dict`` always writes one). Anything
    else is refused rather than guessed at: ``"2"`` and ``2.0`` used to
    slip past the version check, which only looked at ``int`` values, and
    were then read as if they were the current layout. A dictionary with
    no ``"schema"`` key predates versioning and reads as schema 0.
    """
    schema = model_dict.get("schema", 0)
    # ``bool`` is an ``int`` subclass, but ``True`` is not a version.
    if isinstance(schema, bool) or not isinstance(schema, numbers.Integral):
        raise ValueError(
            "The serialised model's 'schema' must be an integer version"
            f" number, got {schema!r}."
        )
    schema = int(schema)
    if schema < 0:
        raise ValueError(
            f"The serialised model's 'schema' version {schema} is invalid:"
            " versions are non-negative."
        )
    if schema > SCHEMA_VERSION:
        raise ValueError(
            f"This serialised model uses schema version {schema}, but "
            f"this version of SurPyval reads schema versions up to "
            f"{SCHEMA_VERSION}. Upgrade surpyval to load it."
        )
    return schema


def check_parameters(dist: Any, params: Any) -> None:
    """Refuse parameters a distribution cannot take, naming the culprit.

    A serialised dictionary is plain data, so a hand-edited or corrupted
    one could restore a model with, say, a negative Weibull scale, which
    then answered every query with NaN or nonsense. ``from_params`` has
    always refused such values; this applies the same bounds (the
    distribution's ``bounds``, ``None`` meaning unbounded) to restored
    parameters. A value *at* a bound is let through, since a fit may
    legitimately end there; NaN is always refused. Distributions without
    ``bounds`` are not checked.
    """
    values = np.atleast_1d(np.asarray(params, dtype=float))
    names = list(getattr(dist, "param_names", []) or [])
    if np.isnan(values).any():
        raise ValueError(
            f"The serialised parameters of '{getattr(dist, 'name', dist)}'"
            " contain NaN."
        )
    bounds = getattr(dist, "bounds", None)
    if bounds is None or values.ndim != 1 or len(bounds) != values.size:
        return
    for idx, ((low, high), value) in enumerate(zip(bounds, values)):
        if (low is not None and value < low) or (
            high is not None and value > high
        ):
            name = names[idx] if idx < len(names) else f"#{idx}"
            raise ValueError(
                f"The serialised parameter {name}={float(value)!r} of"
                f" '{getattr(dist, 'name', dist)}' is outside its bounds"
                f" {(low, high)}."
            )


def _check_restored_parametric(model: Any) -> None:
    """Bounds-check a restored univariate parametric model, including the
    limited-failure-population ``p`` and zero-inflation ``f0`` fractions,
    which are probabilities."""
    check_parameters(model.dist, model.params)
    for flag, attr in (("lfp", "p"), ("zi", "f0")):
        if getattr(model, flag, False):
            value = float(getattr(model, attr))
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"The serialised '{attr}'={value!r} is a proportion and"
                    " must be in [0, 1]."
                )
    if getattr(model, "offset", False) and not np.isfinite(
        float(getattr(model, "gamma", 0.0))
    ):
        raise ValueError("The serialised offset 'gamma' must be finite.")


# Marks a ``from_dict`` function already wrapped by ``checked_from_dict``.
_CHECKED_ATTR = "_surpyval_checked_reader"

_Reader = TypeVar("_Reader", bound=Callable[..., Any])


def _read_checked(
    reader: Any, raw: Callable[[Any, dict], Any], model_dict: Any
) -> Any:
    """``raw(reader, model_dict)`` with the checks every reader needs."""
    if not isinstance(model_dict, dict):
        raise ValueError(
            "Expected a serialised model dict, got "
            f"{type(model_dict).__name__}"
        )
    check_schema(model_dict)
    model_dict = decode_non_finite(model_dict)
    # The class readers index the dictionary directly, so a truncated or
    # hand-edited one surfaced as a bare ``KeyError: 'distribution'``
    # from deep inside a reader. Name the missing entry instead.
    try:
        model = raw(reader, model_dict)
    except KeyError as err:
        key = err.args[0] if err.args else None
        name = getattr(reader, "__name__", type(reader).__name__)
        raise ValueError(
            "The serialised model dictionary is incomplete or corrupt: it"
            f" has no {key!r} entry, which {name}.from_dict needs."
        ) from err
    # Only the core univariate parametric dictionary carries no "model"
    # tag; tagged models that also say "parametric" (mixtures, ...) keep
    # their parameters in other shapes.
    if (
        model_dict.get("parameterization") == "parametric"
        and "model" not in model_dict
    ):
        _check_restored_parametric(model)
    return model


def checked_from_dict(raw: _Reader) -> _Reader:
    """Give a class's ``from_dict`` the checks every reader applies.

    The wrapped reader refuses a non-dictionary and a ``"schema"`` it
    cannot read (:func:`check_schema`), restores the non-finite floats
    (:func:`decode_non_finite`), turns a ``KeyError`` for a missing entry
    into a ``ValueError`` naming it, and bounds-checks the parameters of
    a restored univariate parametric model -- exactly what
    ``surpyval.from_dict`` and ``from_json`` do. Classes using
    :class:`SerialisableMixin` get it automatically; others apply it
    under ``@classmethod``. Applying it twice is harmless.
    """
    if getattr(raw, _CHECKED_ATTR, False):
        return raw

    @functools.wraps(raw)
    def from_dict(cls: Any, model_dict: dict) -> Any:
        return _read_checked(cls, raw, model_dict)

    setattr(from_dict, _CHECKED_ATTR, True)
    return cast(_Reader, from_dict)


def read_model_dict(reader: Any, model_dict: dict) -> Any:
    """``reader.from_dict(model_dict)`` with the checks every reader needs.

    Shared by the package-level :func:`from_dict` and every class's
    ``from_json``, so a model read either way gets the same schema check,
    the same error for a missing entry and the same parameter check as a
    class's own ``from_dict`` (see :func:`checked_from_dict`).
    """
    from_dict_method = reader.from_dict
    # A bound method forwards attribute reads to its function.
    if getattr(from_dict_method, _CHECKED_ATTR, False):
        return from_dict_method(model_dict)
    return _read_checked(
        reader, lambda _reader, d: from_dict_method(d), model_dict
    )


def _resolve(module_name: str, class_name: str) -> Any:
    return getattr(import_module(module_name), class_name)


def to_native(value: Any) -> Any:
    """
    Convert numpy scalars and arrays (recursively, through lists and
    tuples) to native Python types.

    ``to_dict`` implementations use this so their dictionaries contain
    only native types: BSON encoders (e.g. MongoDB's) reject numpy
    scalars such as ``np.int64`` outright, unlike ``json.dumps``.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    return value


class SerialisableMixin:
    """Shared ``to_json`` / ``from_json`` plumbing for serialisable
    models: every class keeps only its ``to_dict`` / ``from_dict``
    pair (this used to be copy-pasted into ~20 classes).

    A ``from_dict`` a subclass defines is wrapped by
    :func:`checked_from_dict` when the class is created, so calling it
    directly checks the dictionary exactly as ``surpyval.from_dict``
    does; it used to skip the schema check and let a missing entry
    escape as a bare ``KeyError``."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        own = cls.__dict__.get("from_dict")
        if isinstance(own, classmethod):
            setattr(
                cls, "from_dict", classmethod(checked_from_dict(own.__func__))
            )

    if TYPE_CHECKING:
        # The contract every user of this mixin fulfils, declared for
        # the type checker but deliberately *not* defined. A stub
        # raising NotImplementedError would read better, but it would
        # also be inherited -- and ``copula_model`` decides whether a
        # margin is serialisable with ``hasattr(m, "to_dict")``, which
        # an inherited stub would answer True for every time.
        def to_dict(self) -> dict: ...

        @classmethod
        def from_dict(cls, model_dict: dict) -> Any: ...

    def to_json(self, fp: str | os.PathLike, with_data: bool = False) -> None:
        """Write :meth:`to_dict` to ``fp`` as strict JSON.

        Parameters
        ----------
        fp : str or os.PathLike
            The file to write.
        with_data : bool, optional
            Write ``to_dict(with_data=True)``, which also stores the fitted
            data, for the models whose ``to_dict`` takes ``with_data``
            (the univariate ``Parametric`` and ``NonParametric``); a
            ``TypeError`` for any other model. Defaults to :code:`False`.
        """
        # Typed loosely: most models' ``to_dict`` takes no arguments.
        to_dict: Any = self.to_dict
        if with_data:
            if "with_data" not in inspect.signature(to_dict).parameters:
                raise TypeError(
                    f"{type(self).__name__}.to_dict does not store the"
                    " fitted data, so to_json(with_data=True) is not"
                    " available for it."
                )
            model_dict = to_dict(with_data=True)
        else:
            model_dict = to_dict()
        # ``to_dict`` already wrote non-finite floats as null;
        # ``allow_nan=False`` guarantees the file is strict JSON.
        with open(fp, "w+") as f:
            json.dump(model_dict, f, allow_nan=False)

    @classmethod
    def from_json(cls, fp: str | os.PathLike) -> Any:
        """Load a model from a JSON file written by :meth:`to_json`."""
        with open(fp, "r") as f:
            model_dict = json.load(f)
        if not isinstance(model_dict, dict):
            raise ValueError(
                "Expected a serialised model dict, got "
                f"{type(model_dict).__name__}"
            )
        return read_model_dict(cls, model_dict)


def from_dict(model_dict: dict) -> Any:
    """
    Restore any serialised SurPyval model from its dictionary.

    Reads the dictionary written by any fitted model's ``to_dict`` and
    dispatches to the right class's ``from_dict``, so the caller does
    not need to know which class wrote it.

    Parameters
    ----------
    model_dict : dict
        A dictionary produced by a SurPyval model's ``to_dict``.

    Returns
    -------
    The restored model, of whichever class serialised the dictionary.

    Raises
    ------
    ValueError
        If the dictionary is not recognisable as a serialised SurPyval
        model, lacks an entry its reader needs (the message names it), has
        a ``"schema"`` that is not a non-negative integer or was written
        by a newer SurPyval (a higher ``"schema"`` version), holds
        parameters outside the distribution's bounds (for a univariate
        parametric model), or names a distribution the reader does not
        know -- which includes a ``CustomDistribution`` that has not been
        constructed again in this session (a dictionary stores only its
        name, since its cumulative hazard is a Python function). Also if
        its ``"non_finite"`` record (see :func:`encode_non_finite`) is
        corrupt. Each class's own ``from_dict`` raises the same errors.

    Notes
    -----
    What a restored model keeps differs by family: in general the
    parameters and whatever predictions need, but not the fitted data,
    so methods that need the data (``plot``, bootstrap and
    likelihood-ratio bounds, residuals) raise on the restored model.
    A fitted univariate parametric, regression or copula model keeps
    the likelihood and sample size of its information criteria, so
    ``aic`` and ``bic`` (and ``aic_c``, where the model has one) work on
    its restored copy.
    See "Saving and Loading Models" in the Conventions page.

    Examples
    --------
    >>> import surpyval
    >>> from surpyval import Weibull
    >>> model = Weibull.fit([3.0, 4.0, 5.0, 6.0, 7.0])
    >>> restored = surpyval.from_dict(model.to_dict())
    >>> restored.dist.name
    'Weibull'
    """
    if not isinstance(model_dict, dict):
        raise ValueError(
            "Expected a serialised model dict, got "
            f"{type(model_dict).__name__}"
        )

    # Before dispatch too: a model class added by a newer SurPyval is
    # unknown here, and "upgrade" is the useful answer for it.
    check_schema(model_dict)

    tag = model_dict.get("model")
    parameterization = model_dict.get("parameterization")
    if isinstance(tag, str) and tag in _TAGGED_MODELS:
        reader = _resolve(_TAGGED_MODELS[tag], tag)
    elif parameterization in _PARAMETERIZATIONS:
        reader = _resolve(*_PARAMETERIZATIONS[parameterization])
    else:
        reader = None

    if reader is not None:
        return read_model_dict(reader, model_dict)

    described = ", ".join(
        f"{k}={model_dict[k]!r}"
        for k in ("model", "parameterization")
        if k in model_dict
    )
    raise ValueError(
        "Not a recognisable serialised SurPyval model"
        + (f" ({described})" if described else "")
        + ": expected a 'model' class tag or a known 'parameterization'."
    )


def from_json(fp: str | Path) -> Any:
    """
    Restore any serialised SurPyval model from a JSON file.

    Reads a file written by any fitted model's ``to_json`` and
    dispatches to the right class's reader; see :func:`from_dict`.

    Parameters
    ----------
    fp : str | Path
        Path to a JSON file written by a SurPyval model's ``to_json``.

    Returns
    -------
    The restored model, of whichever class serialised the file.

    Examples
    --------
    >>> import os, tempfile
    >>> import surpyval
    >>> from surpyval import Weibull
    >>> model = Weibull.fit([3.0, 4.0, 5.0, 6.0, 7.0])
    >>> path = os.path.join(tempfile.mkdtemp(), "weibull.json")
    >>> model.to_json(path)
    >>> restored = surpyval.from_json(path)
    >>> restored.params
    array([5.53092634, 4.04187535])
    """
    with open(fp, "r") as f:
        return from_dict(json.load(f))
