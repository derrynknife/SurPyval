"""Alternate fit paths agree (#379).

A model that can be reached several ways -- ``fit``, ``fit_from_df``
(with column names or a formula), ``from_params`` with the fitted
parameters, ``fit_from_surpyval_data`` / ``fit_from_recurrent_data``,
Cox's ``fit_tvc`` with one interval per subject -- gives the same
predictions from the same data and the default options of each.
"""

import importlib
import inspect
from dataclasses import replace

import numpy as np
import pytest

from surpyval.tests.conformance.registry import (
    CASES,
    NO_DF_PATH,
    cases_for,
    fitted,
    predictions,
    refit,
)


def _path_params():
    # A known failure of one path is listed as "fit_paths[<path>]".
    params = []
    for param in cases_for("fit_paths"):
        case = param.values[0]
        for path in case.paths:
            marks = list(param.marks)
            reason = case.xfail.get(f"fit_paths[{path}]")
            if reason:
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            params.append(
                pytest.param(case, path, id=f"{case.name}-{path}", marks=marks)
            )
    return params


@pytest.mark.parametrize("case, path", _path_params())
def test_fit_paths_agree(case, path):
    ref = predictions(case, fitted(case))
    other = refit(replace(case, fit=case.paths[path]), case.data())
    got = predictions(case, other)
    for key in ref:
        np.testing.assert_allclose(
            got[key],
            ref[key],
            rtol=case.rtol,
            atol=case.rtol * 1e-2,
            err_msg=key,
        )


def _fitters():
    names = sorted({name for case in CASES for name in case.fitters})
    out = []
    for name in names:
        module, _, attr = name.rpartition(".")
        obj = getattr(importlib.import_module(module), attr)
        if hasattr(obj, "fit"):
            out.append(pytest.param(obj, id=name))
    return out


@pytest.mark.parametrize("fitter", _fitters())
def test_every_fitter_reads_a_data_frame(fitter):
    # Principle 14 (#511): every public fitter with a fit has the
    # DataFrame entry point too.
    assert callable(getattr(fitter, "fit_from_df", None))


# The names of fit's data arrays: a DataFrame entry point never takes one
# of these bare, but names the column that fills it ``<name>_col``.
_DATA_NAMES = {"x", "c", "n", "t", "xl", "xr", "tl", "tr", "i", "e", "y", "Z"}


def _from_df_methods():
    out = []
    for param in _fitters():
        fitter = param.values[0]
        for attr in sorted(dir(fitter)):
            if attr.endswith("_from_df") and callable(getattr(fitter, attr)):
                out.append(pytest.param(fitter, attr, id=f"{param.id}.{attr}"))
    return out


@pytest.mark.parametrize("fitter, method", _from_df_methods())
def test_data_frame_columns_are_named_with_col(fitter, method):
    # Principle 21: every DataFrame entry point names a column argument
    # with a ``_col`` suffix (``_cols`` for a list of columns). The other
    # arguments are the frame, a ``formula``, and options its ``fit`` (or
    # ``fit_tvc`` / ``fit_tvc_timeline``) takes too and is passed.
    fit = getattr(fitter, method.removesuffix("_from_df"))
    fit_options = set(inspect.signature(fit).parameters) - _DATA_NAMES
    bad = [
        name
        for name, p in inspect.signature(
            getattr(fitter, method)
        ).parameters.items()
        if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        and not name.endswith(("_col", "_cols"))
        and name not in {"df", "formula"} | fit_options
    ]
    assert not bad, f"{method} names columns without _col: {bad}"


def _columns_for(arg, fit_args):
    """The DataFrame arguments, any one of which fills ``fit``'s data
    argument ``arg`` (a tuple names arguments needed together)."""
    if arg == "Z":
        return ["Z_cols", "formula"]
    if arg == "t":
        return [("tl_col", "tr_col"), ("tl_cols", "tr_cols")]
    if arg == "x" and "t" in fit_args and "i" not in fit_args:
        # A fit that takes the xcnt data model (``x``, ``c``, ``n`` and a
        # truncation ``t``; not an event log) takes intervals as a
        # two-column ``x``, which a DataFrame holds as two columns.
        return [
            ("x_col", "xl_col", "xr_col"),
            ("x_cols", "xl_cols", "xr_cols"),
        ]
    return [f"{arg}_col", f"{arg}_cols"]


@pytest.mark.parametrize("fitter, method", _from_df_methods())
def test_data_frame_reads_every_data_argument(fitter, method):
    # Principle 14 (#571): a DataFrame entry point expresses all the data
    # its fit does. Each data argument of ``fit`` has a column argument;
    # a fit that reads intervals in a two-column ``x`` has ``xl_col`` and
    # ``xr_col`` as well (WeibullAFT.fit_from_df had neither).
    fit = getattr(fitter, method.removesuffix("_from_df"))
    fit_args = [
        p for p in inspect.signature(fit).parameters if p in _DATA_NAMES
    ]
    columns = set(inspect.signature(getattr(fitter, method)).parameters)
    missing = [
        arg
        for arg in fit_args
        if not any(
            set((need,) if isinstance(need, str) else need) <= columns
            for need in _columns_for(arg, fit_args)
        )
    ]
    assert not missing, f"{method} cannot fill fit's {missing} from columns"


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_every_case_has_a_data_frame_path(case):
    # ... and test_fit_paths_agree compares it with fit, for every model
    # fitted from data (#511).
    if case.name in NO_DF_PATH:
        assert "fit_from_df" not in case.paths
        return
    assert "fit_from_df" in case.paths
