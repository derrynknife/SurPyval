"""Equivalence harness for refactors: does a change alter anything?

``record OUT`` writes one JSON snapshot of what the package computes and
says; ``compare A B`` reports every difference between two of them. A
refactor records a snapshot before its change and another after it, and
shows ``compare`` clean (see "Proving a refactor changed nothing" in
``docs/Contributing.rst``)::

    python scripts/refactor/snapshot.py record /tmp/before.json
    # ... make the change ...
    python scripts/refactor/snapshot.py record /tmp/after.json
    python scripts/refactor/snapshot.py compare /tmp/before.json \\
        /tmp/after.json

What a snapshot holds
---------------------
``cases``
    Every case of the conformance registry
    (``surpyval/tests/conformance/registry.py``), fitted to its fixture:
    the fit's warnings (category, message and count) or exception; the
    fitted ``params``, ``neg_ll`` / ``log_likelihood``, ``aic`` and
    ``bic``; every function at the registry's query; every uncertainty
    method the case declares (its ``bounds``), at ``alpha_ci=0.05``,
    each side (``bound=``) and each ``on=`` it takes; ``to_dict()``;
    ``repr`` and ``summary()``; each alternate fit path (formula,
    DataFrame, ...) with its parameters and predictions; and a corpus of
    invalid inputs (a NaN or an infinity in each per-row array, a
    negative time, mismatched lengths, empty data, an invalid censoring
    code or count, an unknown keyword), each with the exception type and
    message it raises or, if it fits, the parameters and warnings.
``extras``
    What the registry does not reach: the time-varying-covariate fits
    (``fit_tvc``, ``fit_tvc_timeline`` and their DataFrame forms) of the
    PH, AFT, PO, AH and Cox families; seeded bootstraps
    (``bootstrap_cb``, Buckley-James ``bootstrap_ci``, the destructive
    degradation and degradation-path ``cb``); and the plain NHPP, HPP,
    proportional-intensity NHPP (with each ``CountingProcess`` baseline)
    and proportional-intensity HPP fits on one fixture mixing observed,
    left-censored, interval-censored, right-censored and right-truncated
    rows.
``api``
    Every public name in the namespaces the completeness check walks
    (``NAMESPACES`` in ``conformance/test_completeness.py``) and in
    ``surpyval.utils``: its kind, its signature with defaults, and for a
    class or fitter, its public methods' signatures.
``imports``
    The modules ``import surpyval`` loads, in a fresh interpreter.
``tests`` / ``doctests``
    The IDs ``pytest --collect-only`` collects under ``surpyval`` (the
    suite, and the ``--doctest-modules`` examples).

Floats are stored exactly (``float.hex``), so ``compare`` is bit-exact
by default; ``--rtol`` allows a relative difference instead, applied to
the numbers inside text (``repr``, messages) too. Object addresses
(``0x7f...``) and the repository path are masked in text.

Determinism: ``record`` re-runs itself with ``PYTHONHASHSEED=0`` and
one BLAS / OpenMP thread, seeds the global random streams before each
task, and runs every task in a fresh worker process (``--jobs``), so
the order tasks run in cannot matter. Snapshots depend on the numpy /
scipy / BLAS build, so compare snapshots recorded in one environment,
with the same version of this script (copy it aside if the change
under test edits it). The outputs are not committed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def _reexec_deterministic() -> None:
    """Re-run this script with a fixed hash seed and one BLAS thread."""
    if all(os.environ.get(k) == v for k, v in _ENV.items()):
        return
    env = {**os.environ, **_ENV}
    argv = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    os.execve(sys.executable, argv, env)


# The package under test is this checkout's, not an installed copy.
sys.path.insert(0, str(ROOT))

import gzip  # noqa: E402
import importlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402
from typing import Any  # noqa: E402

# ---------------------------------------------------------------------------
# Encoding: everything becomes JSON, floats exactly
# ---------------------------------------------------------------------------
_ADDRESS = re.compile(r"0x[0-9a-fA-F]{6,}")


def norm(text: str) -> str:
    """Text with object addresses and the checkout's path masked."""
    text = text.replace(str(ROOT), "<root>")
    return _ADDRESS.sub("0x?", text)


def _fhex(value: float) -> str:
    return float(value).hex()


def enc(value: Any, depth: int = 0) -> Any:
    """``value`` as JSON: a float is ``{"f": hex}``, an array
    ``{"a": dtype, "s": shape, "v": [...]}``, a list ``{"l": [...]}``, a
    dict ``{"d": [[key, value], ...]}`` and any other object
    ``{"o": type, "r": repr}``. Strings, ints, bools and None stay as they
    are."""
    import numpy as np

    if depth > 40:
        return {"o": "too-deep", "r": ""}
    if value is None or isinstance(value, (bool, str)):
        return norm(value) if isinstance(value, str) else value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return {"f": _fhex(value)}
    if isinstance(value, (complex, np.complexfloating)):
        return {"c": [_fhex(value.real), _fhex(value.imag)]}
    if isinstance(value, np.ndarray) or type(value).__name__ == "ArrayBox":
        arr = np.asarray(getattr(value, "_value", value))
        kind = arr.dtype.kind
        if kind == "f":
            flat: list = [_fhex(v) for v in arr.ravel().tolist()]
        elif kind in "biu":
            flat = [int(v) for v in arr.ravel().tolist()]
        else:
            flat = [enc(v, depth + 1) for v in arr.ravel().tolist()]
        return {"a": str(arr.dtype), "s": list(arr.shape), "v": flat}
    if isinstance(value, (list, tuple)):
        return {"l": [enc(v, depth + 1) for v in value]}
    if isinstance(value, dict):
        return {
            "d": [[norm(str(k)), enc(v, depth + 1)] for k, v in value.items()]
        }
    module = type(value).__module__
    if module.startswith("pandas"):
        if hasattr(value, "to_dict") and hasattr(value, "columns"):
            return {
                "pd": type(value).__name__,
                "v": enc(value.to_dict("split")),
            }
        if hasattr(value, "to_dict"):
            return {
                "pd": type(value).__name__,
                "v": enc(
                    {"index": list(value.index), "values": value.to_numpy()}
                ),
            }
    return {
        "o": f"{module}.{type(value).__qualname__}",
        "r": norm(repr(value)),
    }


def _warnings(caught: list) -> list:
    """[category, message, count] in order of first appearance."""
    out: dict[tuple[str, str], int] = {}
    for w in caught:
        key = (w.category.__qualname__, norm(str(w.message)))
        out[key] = out.get(key, 0) + 1
    return [[c, m, k] for (c, m), k in out.items()]


def run(fn: Callable[[], Any]) -> tuple[Any, dict]:
    """``fn()`` and its record: ``{"w": warnings}`` and, if it raised,
    ``{"exc": [type, message]}`` (the value is then None)."""
    import numpy as np

    rec: dict = {}
    errstate = np.geterr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = fn()
        except Exception as e:  # noqa: BLE001 -- recorded, not handled
            value = None
            rec["exc"] = [type(e).__qualname__, norm(str(e))]
        finally:
            # A call that leaves numpy's error state changed must not
            # change what the next one records.
            np.seterr(**errstate)
    if caught:
        rec["w"] = _warnings(caught)
    return value, rec


def probe(fn: Callable[[], Any]) -> dict:
    """``run(fn)`` with the encoded value in ``"v"``."""
    value, rec = run(fn)
    if "exc" not in rec:
        enc_value, enc_rec = run(lambda: enc(value))
        if "exc" in enc_rec:
            rec["v"] = {"o": "unencodable", "r": enc_rec["exc"][1]}
        else:
            rec["v"] = enc_value
    return rec


def _seed() -> None:
    import numpy as np

    np.random.seed(20261002)
    random.seed(20261002)


# ---------------------------------------------------------------------------
# Registry cases
# ---------------------------------------------------------------------------
_SCALARS = ("params", "neg_ll", "log_likelihood", "aic", "bic")


def _scalars(model: Any) -> dict:
    out = {}
    for name in _SCALARS:
        if name not in dir(type(model)) and name not in vars_of(model):
            continue
        out[name] = probe(lambda: _value(model, name))
    return out


def vars_of(obj: Any) -> dict:
    try:
        return vars(obj)
    except TypeError:
        return {}


def _value(model: Any, name: str) -> Any:
    v = getattr(model, name)
    if not callable(v):
        return v
    try:
        required = [
            p
            for p in inspect.signature(v).parameters.values()
            if p.default is p.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        required = []
    # A likelihood that is a function of the coefficients (Cox's
    # ``neg_ll(beta)``) is recorded at the fitted ones.
    return v(model.params) if required else v()


def _parameter_names(model: Any) -> list:
    """The names ``param_cb`` takes, as ``test_options.py`` sweeps them."""
    if hasattr(model, "_parameter_bounds"):
        life = getattr(model, "life_parameter", None)
        return [n for n in model.parameter_names if n != life]
    if hasattr(model, "_param_vector"):
        return list(model.parameter_names)
    names = list(model.dist.parameter_names)
    if model.lfp:
        names.append("lfp_p" if "p" in names else "p")
    if model.zi:
        names.append("f0")
    return names


def _bound_call(case, spec, model, side, fname, event):
    import numpy as np

    from surpyval.tests.conformance import registry as reg

    kw = dict(spec.kwargs)
    kw[spec.level] = 0.05
    if side is not None:
        kw["bound"] = side
    method = getattr(model, spec.method)
    if spec.kind == "function":
        if spec.on or spec.point != "qf":
            x = np.asarray(spec.query, float) if spec.query else case.x
        else:
            x = reg.Q_PROBS
        if spec.on:
            kw["on"] = fname
        args: list = [x]
        if case.interface in reg.WITH_COVARIATES:
            args.append(case.Z[: x.size])
        if spec.per_cause:
            args.append(event)
        return method(*args, **kw)
    if spec.kind == "param":
        return [method(n, **kw) for n in _parameter_names(model)]
    if spec.kind == "coef":
        return method(**kw)
    out = []
    for args in spec.query:
        res = method(*args, **kw)
        out.append(res.rul_interval if spec.kind == "rul" else res)
    return out


def _bounds(case, model, full: bool) -> dict:
    out: dict = {}
    for spec in case.bounds:
        if spec.nightly and not full:
            out[spec.name] = "skipped: nightly (record with --full)"
            continue
        sides = ("two-sided", "lower", "upper") if spec.sides else (None,)
        if spec.kind == "function":
            names = spec.on or (spec.point,)
            causes = case.events if spec.per_cause else (None,)
            curves = [(f, e) for f in names for e in causes]
        else:
            curves = [(None, None)]
        for side in sides:
            for fname, event in curves:
                key = f"{spec.name}|{side}|{fname}|{event}"
                out[key] = probe(
                    lambda: _bound_call(case, spec, model, side, fname, event)
                )
    return out


def _predictions(case, model) -> dict:
    from surpyval.tests.conformance import registry as reg

    out = {}
    for fname, event in reg.calls(case):
        x = reg.query(case, fname)
        key = fname if event is None else f"{fname}[{event}]"
        out[key] = probe(lambda: reg.call(case, model, fname, x, None, event))
    return out


def _model_record(case, model, full: bool, deep: bool = True) -> dict:
    out: dict = {"type": enc(type(model).__qualname__)}
    out.update(_scalars(model))
    out["predictions"] = _predictions(case, model)
    if not deep:
        return out
    out["bounds"] = _bounds(case, model, full)
    if hasattr(model, "to_dict"):
        out["to_dict"] = probe(model.to_dict)
    out["repr"] = probe(lambda: repr(model))
    if callable(getattr(model, "summary", None)):
        out["summary"] = probe(lambda: _text(model.summary()))
    return out


def _text(value: Any) -> Any:
    """A printout in full: a table's every row and column (pandas cuts
    them to the terminal by default), and the values themselves."""
    if hasattr(value, "to_string") and hasattr(value, "columns"):
        return {"text": value.to_string(), "table": value}
    return str(value)


def _copy(data: dict) -> dict:
    import numpy as np

    return {
        k: (np.array(v, copy=True) if isinstance(v, (list, np.ndarray)) else v)
        for k, v in data.items()
    }


def _set_first(data: dict, key: str, value: float) -> bool:
    """Put ``value`` in the first entry of ``data[key]``, as floats."""
    import numpy as np

    try:
        arr = np.array(data[key], dtype=float, copy=True)
    except (TypeError, ValueError):
        return False
    if arr.ndim == 0 or arr.size == 0:
        return False
    arr.flat[0] = value
    data[key] = arr
    return True


def _invalid_inputs(case) -> dict[str, dict]:
    """Mutations of the case's fixture that a fit must refuse or handle."""
    import numpy as np

    base = case.data()
    rows = [k for k in case.rows if k in base]
    out: dict[str, dict] = {}

    def add(label, mutate):
        d = _copy(base)
        if mutate(d) is not False:
            out[label] = d

    for key in rows:
        add(f"nan[{key}]", lambda d, k=key: _set_first(d, k, np.nan))
    for key in [k for k in case.times if k in base]:
        add(f"inf[{key}]", lambda d, k=key: _set_first(d, k, np.inf))
        add(f"negative[{key}]", lambda d, k=key: _set_first(d, k, -1.0))
    for key in rows[1:]:

        def short(d, k=key):
            arr = np.asarray(d[k])
            if arr.ndim == 0 or arr.shape[0] < 2:
                return False
            d[k] = arr[:-1]

        add(f"short[{key}]", short)

    def empty(d):
        if not rows:
            return False
        for k in rows:
            arr = np.asarray(d[k])
            if arr.ndim == 0:
                return False
            d[k] = arr[:0]

    add("empty", empty)
    if "c" in base:
        add("c=7", lambda d: _set_first(d, "c", 7.0))
    if "n" in base:
        add("n=-1", lambda d: _set_first(d, "n", -1.0))
        add("n=0.5", lambda d: _set_first(d, "n", 0.5))
    if rows:

        def strings(d):
            d[rows[0]] = np.array(["a"] * len(np.asarray(d[rows[0]])))

        add(f"strings[{rows[0]}]", strings)
    add("unknown-keyword", lambda d: d.update(not_an_option=1))
    return out


def record_case(name: str, full: bool) -> dict:
    from surpyval.tests.conformance import registry as reg

    _seed()
    case = reg.CASE_BY_NAME[name]
    out: dict = {}
    model, out["fit"] = run(lambda: case.fit(case.data()))
    if model is not None:
        out.update(_model_record(case, model, full))
    paths: dict = {}
    for pname, path in case.paths.items():
        _seed()
        m, rec = run(lambda: path(case.data()))
        if m is not None:
            rec.update(_model_record(case, m, full, deep=False))
        paths[pname] = rec
    out["paths"] = paths
    invalid: dict = {}
    for label, data in _invalid_inputs(case).items():
        _seed()
        m, rec = run(lambda: case.fit(data))
        if m is not None:
            rec.update(_scalars(m))
        invalid[label] = rec
    out["invalid"] = invalid
    return out


# ---------------------------------------------------------------------------
# Extras the registry does not reach
# ---------------------------------------------------------------------------
def _tvc_data():
    """Start-stop rows: units switch from Z=0 to Z=1 at a known time."""
    import numpy as np

    rng = np.random.default_rng(7)
    n = 40
    switch = rng.uniform(0.3, 1.5, n)
    t_low = rng.exponential(2.0, n)
    t_high = switch + rng.exponential(2.0 / np.e, n)
    T = np.where(t_low > switch, t_high, t_low)
    end = np.minimum(T, 3.0)  # administrative censoring at 3
    event = T <= 3.0
    one = end <= switch
    i = np.r_[np.arange(n), np.flatnonzero(~one)]
    xl = np.r_[np.zeros(n), switch[~one]]
    xr = np.r_[np.where(one, end, switch), end[~one]]
    c = np.r_[
        np.where(one & event, 0, 1),
        np.where(event[~one], 0, 1),
    ]
    Z = np.r_[np.zeros(n), np.ones((~one).sum())]
    order = np.lexsort((xl, i))
    ss = {
        "i": i[order],
        "xl": xl[order],
        "xr": xr[order],
        "c": c[order],
        "Z": Z[order],
    }
    # The same histories as a timeline: Z from x, the last row's status.
    ti, tx, tZ, tc = [], [], [], []
    for k in range(n):
        ti += [k]
        tx += [0.0]
        tZ += [0.0]
        tc += [1]
        if not one[k]:
            ti += [k]
            tx += [switch[k]]
            tZ += [1.0]
            tc += [1]
        ti += [k]
        tx += [end[k]]
        tZ += [tZ[-1]]
        tc += [0 if event[k] else 1]
    tl = {
        "i": np.array(ti),
        "x": np.array(tx),
        "Z": np.array(tZ),
        "c": np.array(tc),
    }
    return ss, tl


def _tvc_model_record(model) -> dict:
    out = _scalars(model)
    out["to_dict"] = probe(model.to_dict)
    out["repr"] = probe(lambda: repr(model))
    x = [0.5, 1.0, 2.0, 2.5]
    path = ([[0.0], [1.0]],)
    out["sf_tvc"] = probe(lambda: model.sf_tvc(x, *path, xl=[0.0, 1.0]))
    out["Hf_tvc"] = probe(lambda: model.Hf_tvc(x, *path, xl=[0.0, 1.0]))
    if hasattr(model, "cb_tvc"):
        out["cb_tvc"] = probe(lambda: model.cb_tvc(x, *path, xl=[0.0, 1.0]))
    return out


TVC_FITTERS = (
    "WeibullPH",
    "ExponentialPH",
    "WeibullAFT",
    "LogNormalAFT",
    "WeibullPO",
    "WeibullAH",
    "CoxPH",
)


def record_tvc(name: str) -> dict:
    import pandas as pd

    import surpyval as sp

    fitter = getattr(sp, name)
    ss, tl = _tvc_data()
    ss_df = pd.DataFrame({**ss, "z": ss["Z"]}).drop(columns="Z")
    ss_df["dose"] = ss_df["z"].map({0.0: "low", 1.0: "high"})
    tl_df = pd.DataFrame({**tl, "z": tl["Z"]}).drop(columns="Z")
    calls = {
        "fit_tvc": lambda: fitter.fit_tvc(
            ss["i"], ss["xl"], ss["xr"], ss["c"], ss["Z"]
        ),
        "fit_tvc_timeline": lambda: fitter.fit_tvc_timeline(
            tl["i"], tl["x"], tl["Z"], tl["c"]
        ),
        "fit_tvc_from_df[Z_cols]": lambda: fitter.fit_tvc_from_df(
            ss_df, "i", "xl", "xr", "c", Z_cols="z"
        ),
        "fit_tvc_from_df[formula]": lambda: fitter.fit_tvc_from_df(
            ss_df, "i", "xl", "xr", "c", formula="dose"
        ),
        "fit_tvc_timeline_from_df": lambda: fitter.fit_tvc_timeline_from_df(
            tl_df, "i", "x", "z", "c"
        ),
    }
    out: dict = {}
    for label, fit in calls.items():
        method = label.split("[")[0]
        if not hasattr(fitter, method):
            out[label] = "absent"
            continue
        _seed()
        model, rec = run(fit)
        if model is not None:
            rec.update(_tvc_model_record(model))
        out[label] = rec
    return out


def record_bootstrap(_: str) -> dict:
    import numpy as np

    from surpyval.tests.conformance import registry as reg

    out: dict = {}

    def fit(name):
        case = reg.CASE_BY_NAME[name]
        return case, run(lambda: case.fit(case.data()))[0]

    for name in ("KaplanMeier", "NelsonAalen", "Turnbull"):
        case, model = fit(name)
        for side in ("two-sided", "lower", "upper"):
            _seed()
            out[f"{name}.bootstrap_cb[{side}]"] = probe(
                lambda: model.bootstrap_cb(
                    case.x, bound=side, n_boot=60, random_state=11
                )
            )
    case, model = fit("BuckleyJames")
    _seed()
    out["BuckleyJames.bootstrap_ci"] = probe(
        lambda: model.bootstrap_ci(n_boot=30, random_state=5)
    )
    case, model = fit("DestructiveDegradation")
    for side in ("two-sided", "lower"):
        _seed()
        out[f"DestructiveDegradation.cb[{side}]"] = probe(
            lambda: model.cb(case.x, bound=side, n_boot=12, random_state=3)
        )
    case, model = fit("DegradationAnalysis[linear]")
    _seed()
    out["DegradationAnalysis[linear].cb[bootstrap]"] = probe(
        lambda: model.cb(case.x, method="bootstrap", n_boot=12, random_state=4)
    )
    _seed()
    out["DegradationAnalysis[linear].random"] = probe(
        lambda: model.random(5, random_state=np.random.default_rng(2))
    )
    return out


def _recurrent_data() -> dict:
    """Observed (c=0), left (-1), interval (2) and right-censored (1)
    rows, with right truncation (``tr``) on two of four items."""
    import numpy as np

    x = np.array(
        [
            [3.0, 3.0],
            [6.0, 6.0],
            [8.0, 12.0],
            [14.0, 14.0],
            [2.0, 2.0],
            [5.0, 9.0],
            [11.0, 11.0],
            [14.0, 14.0],
            [4.0, 4.0],
            [7.0, 7.0],
            [10.0, 13.0],
            [18.0, 18.0],
            [1.0, 1.0],
            [6.0, 8.0],
            [9.0, 9.0],
            [16.0, 16.0],
        ]
    )
    return {
        "x": x,
        "i": np.repeat([1, 2, 3, 4], 4),
        "c": np.array([-1, 0, 2, 0, -1, 2, 0, 1, 0, 0, 2, 0, -1, 2, 0, 1]),
        "n": np.array([2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1]),
        "tr": np.repeat([16.0, np.inf, 20.0, np.inf], 4),
        "Z": np.repeat([[0.0], [1.0], [0.5], [0.2]], 4, axis=0),
    }


def _recurrent_record(model) -> dict:
    out = _scalars(model)
    x = [2.0, 5.0, 10.0, 15.0]
    regression = hasattr(model, "coeffs") or "Z" in str(
        inspect.signature(model.cif)
    )
    args: tuple = (x, [0.5]) if regression else (x,)
    for fname in ("cif", "iif"):
        if hasattr(model, fname):
            out[fname] = probe(lambda: getattr(model, fname)(*args))
    if hasattr(model, "to_dict"):
        out["to_dict"] = probe(model.to_dict)
    out["repr"] = probe(lambda: repr(model))
    return out


def record_recurrent(_: str) -> dict:
    from surpyval import recurrent as rc

    d = _recurrent_data()
    plain = {k: d[k] for k in ("x", "i", "c", "n", "tr")}
    out: dict = {}
    processes = ("HPP", "CrowAMSAA", "Duane", "CoxLewis")
    for name in processes:
        _seed()
        model, rec = run(lambda: getattr(rc, name).fit(**plain))
        if model is not None:
            rec.update(_recurrent_record(model))
        out[name] = rec
    for name in processes:
        _seed()
        model, rec = run(
            lambda: rc.ProportionalIntensityNHPP.fit(
                d["x"],
                d["Z"],
                i=d["i"],
                c=d["c"],
                n=d["n"],
                tr=d["tr"],
                dist=getattr(rc, name),
            )
        )
        if model is not None:
            rec.update(_recurrent_record(model))
        out[f"ProportionalIntensityNHPP[{name}]"] = rec
    _seed()
    model, rec = run(
        lambda: rc.ProportionalIntensityHPP.fit(
            d["x"], d["Z"], i=d["i"], c=d["c"], n=d["n"], tr=d["tr"]
        )
    )
    if model is not None:
        rec.update(_recurrent_record(model))
    out["ProportionalIntensityHPP"] = rec
    return out


# ---------------------------------------------------------------------------
# API, imports and test IDs
# ---------------------------------------------------------------------------
def _signature(obj: Any) -> str:
    try:
        return norm(str(inspect.signature(obj)))
    except (TypeError, ValueError):
        return "<no signature>"


def _members(cls: type) -> dict:
    out = {}
    for name in sorted(dir(cls)):
        if name.startswith("_"):
            continue
        attr = inspect.getattr_static(cls, name, None)
        if isinstance(attr, property):
            out[name] = "property"
        elif isinstance(attr, (staticmethod, classmethod)):
            out[name] = f"{type(attr).__name__}{_signature(attr.__func__)}"
        elif inspect.isfunction(attr):
            out[name] = f"method{_signature(attr)}"
        elif inspect.isclass(attr):
            out[name] = "class"
        else:
            out[name] = f"attribute {type(attr).__name__}"
    return out


def _describe(obj: Any) -> dict:
    if inspect.ismodule(obj):
        return {"kind": "module", "name": obj.__name__}
    if inspect.isclass(obj):
        return {
            "kind": "class",
            "module": obj.__module__,
            "signature": _signature(obj),
            "members": _members(obj),
        }
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return {
            "kind": "function",
            "module": getattr(obj, "__module__", None),
            "signature": _signature(obj),
        }
    if hasattr(obj, "fit"):
        return {
            "kind": "instance",
            "type": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "members": _members(type(obj)),
        }
    if callable(obj):
        return {"kind": "callable", "signature": _signature(obj)}
    return {"kind": "value", "type": type(obj).__qualname__}


def _namespaces() -> list[str]:
    from surpyval.tests.conformance import test_completeness

    names = list(test_completeness.NAMESPACES)
    if "surpyval.utils" not in names:
        names.append("surpyval.utils")
    return names


def record_api(_: str) -> dict:
    out: dict = {}
    for namespace in _namespaces():
        try:
            module = importlib.import_module(namespace)
        except Exception as e:  # noqa: BLE001
            out[namespace] = {"exc": [type(e).__qualname__, norm(str(e))]}
            continue
        names = getattr(module, "__all__", None)
        listed = names is not None
        if names is None:
            names = [n for n in dir(module) if not n.startswith("_")]
        entry: dict = {"__all__": listed}
        for name in sorted(names):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # deprecated names
                try:
                    obj = getattr(module, name)
                except Exception as e:  # noqa: BLE001
                    entry[name] = {"exc": [type(e).__qualname__, str(e)]}
                    continue
            entry[name] = _describe(obj)
        out[namespace] = entry
    return out


def _subprocess(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **_ENV},
    )


_IMPORTS = (
    "import json, sys; import surpyval; "
    "print(json.dumps(sorted(sys.modules)))"
)


def _collect(extra: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        *extra,
    ]


def _ids(proc: subprocess.Popen) -> dict:
    stdout, stderr = proc.communicate()
    ids = sorted(line for line in stdout.splitlines() if "::" in line)
    out: dict = {"count": len(ids), "ids": ids}
    if proc.returncode not in (0, 5):
        out["error"] = norm(stderr[-2000:] or stdout[-2000:])
    return out


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------
RECORDERS: dict[str, Callable[[str], dict]] = {
    "tvc": record_tvc,
    "bootstrap": record_bootstrap,
    "recurrent": record_recurrent,
    "api": record_api,
}


def _task(task: tuple[str, str, bool]) -> tuple[str, str, Any, float]:
    section, name, full = task
    start = time.perf_counter()
    try:
        if section == "cases":
            value = record_case(name, full)
        else:
            value = RECORDERS[section](name)
    except Exception:  # noqa: BLE001 -- a harness failure is recorded
        value = {"harness-error": norm(traceback.format_exc())}
    foreign = _foreign_modules()
    if foreign:
        value = {"harness-error": f"imported from another checkout: {foreign}"}
    return section, name, value, time.perf_counter() - start


def _foreign_modules() -> list[str]:
    """SurPyval modules loaded from outside this checkout.

    An editable install of another checkout serves a module this one
    lacks (one a change deleted or moved), so a snapshot could silently
    record the other checkout's code.
    """
    root = str(ROOT) + os.sep
    return sorted(
        name
        for name, module in list(sys.modules.items())
        if name.split(".")[0] == "surpyval"
        and getattr(module, "__file__", None)
        and not os.path.abspath(module.__file__).startswith(root)
    )


def _tasks(full: bool, only: list[str] | None) -> list[tuple[str, str, bool]]:
    from surpyval.tests.conformance import registry as reg

    tasks = [("cases", c.name, full) for c in reg.CASES]
    if only:
        tasks = [t for t in tasks if t[1] in only]
        return tasks
    tasks += [("tvc", name, full) for name in TVC_FITTERS]
    tasks += [
        ("bootstrap", "bootstrap", full),
        ("recurrent", "recurrent", full),
        ("api", "api", full),
    ]
    return tasks


def record(args: argparse.Namespace) -> int:
    import numpy as np
    import scipy

    import surpyval

    # Import everything the workers use before they are forked, so each
    # fork starts from the same, fully imported state.
    import surpyval.tests.conformance.registry  # noqa: F401
    from surpyval.tests.conformance import test_completeness  # noqa: F401

    start = time.perf_counter()
    side = {}
    if not args.only:
        side["imports"] = _subprocess([sys.executable, "-c", _IMPORTS])
        if not args.no_tests:
            side["tests"] = _subprocess(_collect(["surpyval"]))
            side["doctests"] = _subprocess(
                _collect(
                    [
                        "--doctest-modules",
                        "surpyval",
                        "--ignore=surpyval/tests",
                    ]
                )
            )
    tasks = _tasks(args.full, args.only)
    snapshot: dict = {
        "meta": {
            "surpyval": surpyval.__file__.replace(str(ROOT), "<root>"),
            "version": surpyval.__version__,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "full": args.full,
            "only": args.only,
        },
        "cases": {},
        "extras": {},
    }
    timings: dict[str, float] = {}
    ctx = multiprocessing.get_context("fork")
    # A fresh process per task: no state (caches, random streams, the
    # warnings registry) carries from one task to the next.
    with ctx.Pool(args.jobs, maxtasksperchild=1) as pool:
        for k, (section, name, value, seconds) in enumerate(
            pool.imap_unordered(_task, tasks)
        ):
            target = snapshot["cases"] if section == "cases" else None
            if section == "api":
                snapshot["api"] = value
            elif target is not None:
                target[name] = value
            else:
                snapshot["extras"][f"{section}:{name}"] = value
            timings[f"{section}:{name}"] = seconds
            if args.verbose:
                print(
                    f"[{k + 1}/{len(tasks)}] {section}:{name} {seconds:.1f}s",
                    file=sys.stderr,
                )
    for key in ("cases", "extras"):
        snapshot[key] = dict(sorted(snapshot[key].items()))
    for key, proc in side.items():
        if key == "imports":
            stdout, stderr = proc.communicate()
            snapshot["imports"] = (
                json.loads(stdout) if proc.returncode == 0 else norm(stderr)
            )
        else:
            snapshot[key] = _ids(proc)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    opener: Any = gzip.open if out.suffix == ".gz" else open
    with opener(out, "wt") as fh:
        json.dump(snapshot, fh, sort_keys=False, separators=(",", ":"))
    if args.timings:
        Path(args.timings).write_text(
            json.dumps(
                dict(sorted(timings.items(), key=lambda kv: -kv[1])), indent=1
            )
        )
    errors = [
        k
        for k, v in {**snapshot["cases"], **snapshot["extras"]}.items()
        if isinstance(v, dict) and "harness-error" in v
    ]
    print(
        f"recorded {len(snapshot['cases'])} cases and "
        f"{len(snapshot['extras'])} extras to {out} in "
        f"{time.perf_counter() - start:.0f} s ({args.jobs} jobs)"
    )
    if errors:
        print(f"harness errors (see the snapshot): {errors}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
_NUMBER = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan"
)


def _close(a: float, b: float, rtol: float) -> bool:
    if a == b or (math.isnan(a) and math.isnan(b)):
        return True
    if rtol == 0 or math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def _text_close(a: str, b: str, rtol: float) -> bool:
    if a == b:
        return True
    if rtol == 0 or _NUMBER.sub("#", a) != _NUMBER.sub("#", b):
        return False
    na, nb = _NUMBER.findall(a), _NUMBER.findall(b)
    return len(na) == len(nb) and all(
        _close(float(x), float(y), rtol) for x, y in zip(na, nb)
    )


def _short(value: Any) -> str:
    text = json.dumps(value)
    return text if len(text) <= 160 else text[:157] + "..."


def _fromhex(v: str) -> float:
    return float.fromhex(v) if v.startswith(("0x", "-0x")) else float(v)


def diff(a: Any, b: Any, path: str, rtol: float) -> Iterator[str]:
    """Every difference between two encoded values, as text lines."""
    if isinstance(a, dict) and isinstance(b, dict):
        if "f" in a and "f" in b and len(a) == len(b) == 1:
            if not _close(_fromhex(a["f"]), _fromhex(b["f"]), rtol):
                x, y = _fromhex(a["f"]), _fromhex(b["f"])
                yield f"{path}: {x!r} != {y!r}"
            return
        if "a" in a and "a" in b and "v" in a and "v" in b:
            if a["a"] != b["a"] or a["s"] != b["s"]:
                yield (f"{path}: array {a['a']}{a['s']} != {b['a']}{b['s']}")
                return
            if a["a"].startswith("float"):
                bad = [
                    k
                    for k, (x, y) in enumerate(zip(a["v"], b["v"]))
                    if not _close(_fromhex(x), _fromhex(y), rtol)
                ]
                if bad:
                    k = bad[0]
                    x, y = _fromhex(a["v"][k]), _fromhex(b["v"][k])
                    yield (
                        f"{path}: {len(bad)} of {len(a['v'])} values differ;"
                        f" first [{k}]: {x!r} != {y!r}"
                    )
                return
            for k, (x, y) in enumerate(zip(a["v"], b["v"])):
                yield from diff(x, y, f"{path}[{k}]", rtol)
            return
        if "d" in a and "d" in b and len(a) == len(b) == 1:
            a, b = dict(map(tuple, a["d"])), dict(map(tuple, b["d"]))
        elif "l" in a and "l" in b and len(a) == len(b) == 1:
            a, b = a["l"], b["l"]
            if len(a) != len(b):
                yield f"{path}: length {len(a)} != {len(b)}"
                return
            for k, (x, y) in enumerate(zip(a, b)):
                yield from diff(x, y, f"{path}[{k}]", rtol)
            return
        for key in a.keys() | b.keys():
            sub = f"{path}.{key}" if path else str(key)
            if key not in b:
                yield f"{sub}: only in A: {_short(a[key])}"
            elif key not in a:
                yield f"{sub}: only in B: {_short(b[key])}"
            else:
                yield from diff(a[key], b[key], sub, rtol)
        return
    if isinstance(a, list) and isinstance(b, list):
        if all(isinstance(v, str) for v in a + b) and len(a) > 50:
            # A list of names (modules, test IDs): report the set change.
            gone, new = sorted(set(a) - set(b)), sorted(set(b) - set(a))
            for v in gone[:20]:
                yield f"{path}: removed {v}"
            for v in new[:20]:
                yield f"{path}: added {v}"
            more = (
                len(gone) + len(new) - min(len(gone), 20) - min(len(new), 20)
            )
            if more > 0:
                yield f"{path}: ... and {more} more"
            return
        if len(a) != len(b):
            yield f"{path}: length {len(a)} != {len(b)}"
            return
        for k, (x, y) in enumerate(zip(a, b)):
            yield from diff(x, y, f"{path}[{k}]", rtol)
        return
    if isinstance(a, str) and isinstance(b, str):
        if _text_close(a, b, rtol):
            return
        la, lb = a.splitlines(), b.splitlines()
        if len(la) > 1 or len(lb) > 1:
            # A printout: show its first differing line.
            for k, (sa, sb) in enumerate(zip(la, lb)):
                if not _text_close(sa, sb, rtol):
                    yield f"{path} line {k + 1}: {_short(sa)} != {_short(sb)}"
                    return
            yield f"{path}: {len(la)} lines != {len(lb)} lines"
            return
        yield f"{path}: {_short(a)} != {_short(b)}"
        return
    if a != b or type(a) is not type(b):
        yield f"{path}: {_short(a)} != {_short(b)}"


def _load(path: str) -> dict:
    opener: Any = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


SECTIONS = ("cases", "extras", "api", "imports", "tests", "doctests")


def compare(args: argparse.Namespace) -> int:
    a, b = _load(args.a), _load(args.b)
    sections = [s for s in SECTIONS if s not in args.skip]
    if a.get("meta", {}).get("numpy") != b.get("meta", {}).get("numpy"):
        print("note: recorded with different numpy versions")
    total = 0
    for section in sections:
        if section not in a and section not in b:
            continue
        if section not in a or section not in b:
            print(f"{section}: only in {'A' if section in a else 'B'}")
            total += 1
            continue
        lines = list(diff(a[section], b[section], section, args.rtol))
        total += len(lines)
        status = "identical" if not lines else f"{len(lines)} differences"
        print(f"{section}: {status}")
        for line in lines[: args.limit]:
            print(f"  {line}")
        if len(lines) > args.limit:
            print(f"  ... {len(lines) - args.limit} more (--limit)")
    tol = "bit-exact" if args.rtol == 0 else f"rtol={args.rtol:g}"
    print(f"{'no' if total == 0 else total} differences ({tol})")
    return 0 if total == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="record a snapshot to OUT")
    rec.add_argument("out", help="output file (.json, or .json.gz)")
    rec.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="worker processes (default: half the cores)",
    )
    rec.add_argument(
        "--full",
        action="store_true",
        help="also the bounds the registry runs nightly only (minutes)",
    )
    rec.add_argument(
        "--only",
        nargs="+",
        metavar="CASE",
        help="record only these registry cases (no extras, API or tests)",
    )
    rec.add_argument(
        "--no-tests",
        action="store_true",
        help="skip collecting the test and doctest IDs",
    )
    rec.add_argument("--timings", help="write each task's seconds here")
    rec.add_argument("-v", "--verbose", action="store_true")
    cmp = sub.add_parser("compare", help="compare snapshots A and B")
    cmp.add_argument("a")
    cmp.add_argument("b")
    cmp.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="relative tolerance for numbers (default 0: bit-exact)",
    )
    cmp.add_argument(
        "--skip",
        nargs="+",
        default=(),
        choices=SECTIONS,
        help="sections to leave out",
    )
    cmp.add_argument(
        "--limit", type=int, default=50, help="lines shown per section"
    )
    args = parser.parse_args(argv)
    if args.command == "record":
        return record(args)
    return compare(args)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        _reexec_deterministic()
    sys.exit(main())
