"""Fitted models pickle (#573; Conventions, "Saving and Loading Models").

``pickle`` is how Python sends an object to another process --
``multiprocessing``, ``concurrent.futures`` process pools, ``joblib``,
Dask, Ray -- and how ``pickle`` / ``joblib.dump`` cache it. A fitted
model that kept a closure or a lambda (its likelihood, a parameter
transform), or whose class pickle could not find by name (a fitter
singleton), could not be sent or cached.

These properties check that every registered model pickles and that the
unpickled model is the same model: of the same class, with the same
predictions at the registry's query, the same ``to_dict`` (where it can
be saved), the same model-comparison values and the same bounds from
each of its uncertainty methods -- bit for bit, as nothing is
recomputed but what the model rebuilds from the same values. The second
property does the same for the model of every alternate fit path.
"""

import json
import multiprocessing
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.conformance.registry import (
    CASE_BY_NAME,
    cases_for,
    fitted,
    predictions,
    refit,
    tvc_path,
)
from surpyval.tests.conformance.test_options import (
    _functions,
    _parameters,
    _raw,
)

COMPARISON = ("neg_ll", "aic", "aic_c", "bic")


def _round_trip(model):
    return pickle.loads(pickle.dumps(model))


def _comparison(model):
    """The model-comparison values the model gives (test_comparison.py),
    by name; those it does not give are left out."""
    out = {}
    for name in COMPARISON:
        method = getattr(model, name, None)
        if not callable(method):
            continue
        try:
            out[name] = method()
        except (ValueError, TypeError):
            continue
    return out


def _bounds(case, model):
    """Each uncertainty method's two-sided 95% bounds on the first thing
    it bounds, by name (the slow and nightly sweeps left out)."""
    out = {}
    for spec in case.bounds:
        if spec.slow or spec.nightly:
            continue
        fname, event = (None, None)
        if spec.kind == "function":
            fname, event = _functions(case, spec)[0]
        try:
            out[spec.name] = _raw(
                case, spec, model, "two-sided", 0.05, fname, event
            )
        except Exception as error:  # the same error both times
            out[spec.name] = repr(error)
    return out


def _assert_same(case, model, restored, saved=True):
    assert type(restored) is type(model)
    ref = predictions(case, model)
    new = predictions(case, restored)
    assert ref.keys() == new.keys()
    for key in ref:
        np.testing.assert_array_equal(new[key], ref[key], err_msg=key)
    if saved:
        assert json.dumps(restored.to_dict(), allow_nan=False) == json.dumps(
            model.to_dict(), allow_nan=False
        )


@pytest.mark.parametrize("case", cases_for("pickle"))
def test_fitted_model_round_trips_through_pickle(case):
    model = fitted(case)
    restored = _round_trip(model)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        _assert_same(case, model, restored, case.applies("serialise"))
        assert _comparison(restored) == _comparison(model)
        ref, new = _bounds(case, model), _bounds(case, restored)
    assert ref.keys() == new.keys()
    for key in ref:
        if isinstance(ref[key], str):
            assert new[key] == ref[key], key
        else:
            np.testing.assert_array_equal(new[key], ref[key], err_msg=key)


def _lr_spec(case):
    """The case's first likelihood-ratio bound (not a nightly sweep's),
    or ``None``."""
    for spec in case.bounds:
        if spec.kwargs.get("method") == "lr" and not spec.nightly:
            return spec
    return None


def _one_lr_bound(case, spec, model):
    """One two-sided 95% likelihood-ratio bound: on the first parameter,
    or at the first time of the sweep."""
    if spec.kind == "param":
        name = _parameters(model)[0][0]
        return np.asarray(model.param_cb(name, **spec.kwargs), float)
    fname, event = _functions(case, spec)[0]
    return _raw(case, spec, model, "two-sided", 0.05, fname, event, k=0)


def _lr_cases():
    out = []
    for param in cases_for("pickle", where=lambda c: bool(_lr_spec(c))):
        case = param.values[0]
        marks = list(param.marks)
        if _lr_spec(case).slow:
            marks.append(pytest.mark.slow)
        out.append(pytest.param(case, id=case.name, marks=marks))
    return out


@pytest.mark.parametrize("case", _lr_cases())
def test_617_a_model_after_a_likelihood_ratio_bound(case):
    """A model that has computed a likelihood-ratio bound pickles without
    its searches' caches (the likelihoods, walks, regions and bounds they
    found: 290 kB on a Weibull's 50-time band, #617), and the unpickled
    model rebuilds them to the same bound."""
    model = refit(case, case.data())
    spec = _lr_spec(case)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        bound = _one_lr_bound(case, spec, model)
        restored = _round_trip(model)
        kept = sorted(k for k in vars(restored) if k.startswith("_lr_"))
        assert not kept, kept
        np.testing.assert_array_equal(
            _one_lr_bound(case, spec, restored), bound
        )


def _paths(case):
    out = dict(case.paths)
    tvc = tvc_path(case)
    if tvc is not None and "fit_tvc" not in out:
        out["fit_tvc"] = tvc
    return out


@pytest.mark.parametrize(
    "case", cases_for("pickle_paths", where=lambda case: bool(_paths(case)))
)
def test_alternate_fit_paths_round_trip_through_pickle(case):
    data = case.data()
    for name, fit in _paths(case).items():
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            model = fit(data)
            restored = _round_trip(model)
            # (Predictions only: a path's own to_dict is the fit_paths
            # property's.)
            try:
                _assert_same(case, model, restored, saved=False)
            except AssertionError as error:
                raise AssertionError(f"{name}: {error}") from error


def _sf_in_worker(model, x):
    return np.asarray(model.sf(x), float)


def _cif_in_worker(model, x):
    return np.asarray(model.cif(x), float)


def test_573_fitted_models_work_in_a_process_pool():
    """The issue's use: fitted models sent to worker processes (as
    ``n_jobs`` and ``joblib`` do) predict there as they do here."""
    cases = [CASE_BY_NAME[name] for name in ("Weibull", "Gamma", "CoxPH")]
    models = [fitted(case) for case in cases]
    x = np.array([1.0, 2.0, 5.0])
    crow = sp.CrowAMSAA.fit(np.array([1.0, 3, 4, 8]))
    # Spawned, not forked: each worker imports surpyval afresh, as on
    # macOS and Windows, so the models reach it only through pickle (and
    # forking a multi-threaded process is deprecated).
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        parametric = [pool.submit(_sf_in_worker, m, x) for m in models[:2]]
        cox = pool.submit(models[2].sf, cases[2].x, cases[2].Z)
        recurrent = pool.submit(_cif_in_worker, crow, x)
        for model, future in zip(models[:2], parametric):
            np.testing.assert_array_equal(future.result(), model.sf(x))
        np.testing.assert_array_equal(
            cox.result(), models[2].sf(cases[2].x, cases[2].Z)
        )
        np.testing.assert_array_equal(recurrent.result(), crow.cif(x))


def test_573_fitter_singletons_unpickle_as_themselves():
    """A recurrence fitter is a singleton (``CrowAMSAA`` is an instance);
    it unpickles as the same object, so a model that holds it does too,
    and the model's own class is found through the ``<Name>_`` alias."""
    for fitter in (sp.HPP, sp.CrowAMSAA, sp.NonParametricCounting):
        assert _round_trip(fitter) is fitter
    model = sp.CrowAMSAA.fit(np.array([1.0, 3, 4, 8]))
    assert _round_trip(model).dist is sp.CrowAMSAA
    mcf = sp.NonParametricCounting.fit(np.array([1.0, 3, 4, 8]))
    restored = _round_trip(mcf)
    assert type(restored) is type(mcf) and restored is not mcf
    np.testing.assert_array_equal(restored.mcf_hat, mcf.mcf_hat)
