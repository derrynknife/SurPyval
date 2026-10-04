"""The missing-value rule (#375; Conventions, "Missing values").

Prediction: NaN in, NaN out, element by element -- a missing time,
probability or covariate makes exactly the outputs that depend on it
NaN, and leaves the others as they were. (The exceptions, methods whose
input describes one unit's history -- ``predict_rul``, ``sf_tvc``, a
proportional-intensity ``mcf`` given a unit's covariates -- are not among
the functions the registry calls.)

Fitting: a missing time or response always raises. A missing covariate
drops its row with one warning ("Dropped k of n rows ...") where each row
is an independent observation, and raises where a row is only part of
one (a recurrent-event row).
"""

import warnings
from importlib import import_module

import numpy as np
import pytest

from surpyval.tests.conformance.registry import (
    BIVARIATE,
    WITH_COVARIATES,
    call,
    calls,
    cases_for,
    fitted,
    predictions,
    query,
    refit,
)


def _check_nan_where(got, ref, where, name):
    got = np.asarray(got, float)
    assert np.all(np.isnan(got[where])), f"{name}: {got[where]} not NaN"
    keep = np.ones(len(got), bool)
    keep[where] = False
    np.testing.assert_allclose(
        got[keep], np.asarray(ref, float)[keep], rtol=1e-12, err_msg=name
    )


@pytest.mark.parametrize("case", cases_for("missing_query"))
def test_missing_query_value(case):
    model = fitted(case)
    for fname, event in calls(case):
        if fname in case.jump_functions:
            # A jump is a difference between neighbouring query points,
            # so a missing point also changes its neighbour's value.
            continue
        x = np.array(query(case, fname), dtype=float)
        k = len(x) // 2
        if case.interface == BIVARIATE:
            x[k, 1] = np.nan
        else:
            x[k] = np.nan
        Z = None if fname == "qf" else case.Z
        ref = call(case, model, fname, query(case, fname), Z, event)
        got = call(case, model, fname, x, Z, event)
        _check_nan_where(got, ref, [k], fname)


def _quantile_functions(case, model):
    """Every ``qf`` of a fitted model, each as a function of ``p`` alone:
    the model's own (with one covariate row for every probability where
    it takes covariates) and, for a parametric model, its distribution's
    at the fitted parameters."""
    out = {}
    if case.interface in WITH_COVARIATES:
        row = np.asarray(case.Z, dtype=float)[0]
        # With the case's own arguments (a stratified Cox model's stratum)
        out["qf"] = lambda p: model.qf(p, row, **case.call_kwargs)
        return out
    if type(model).__name__ == "MixtureModel":
        # ``params`` has a row per component: the mixture's qf alone
        out["qf"] = model.qf
        return out
    dist = getattr(model, "dist", None)
    if callable(getattr(dist, "qf", None)) and hasattr(model, "params"):
        out["dist.qf"] = lambda p: dist.qf(p, *model.params)
    elif dist is not None:
        # A parametric model of a distribution with no quantile function
        # (FixedEventProbability) has none either.
        return out
    out["qf"] = model.qf
    return out


def _has_qf(case):
    module, _, name = case.model_class.rpartition(".")
    return callable(getattr(getattr(import_module(module), name), "qf", None))


@pytest.mark.parametrize("case", cases_for("qf_outside", where=_has_qf))
def test_611_qf_outside_unit_interval(case):
    # One rule for every qf (#576, #611), as scipy's ppf gives: NaN where
    # p is outside [0, 1], the other quantiles unchanged, and one warning
    # pointing at the caller. The non-parametric qf raised, a process
    # model's returned 0 and inf, and the distributions' gave whatever
    # their formula did (a negative Exponential time, a Uniform point past
    # its end).
    model = fitted(case)
    p = np.array([-0.5, 0.5, 1.5])
    functions = _quantile_functions(case, model)
    for name, qf in functions.items():
        ref = np.asarray(qf(p[1:2]), dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = np.asarray(qf(p), dtype=float)
        mine = [w for w in caught if "outside [0, 1]" in str(w.message)]
        assert len(mine) == 1, (name, [str(w.message) for w in caught])
        assert mine[0].filename == __file__, (name, mine[0].filename)
        assert np.isnan(got[[0, 2]]).all(), (name, got)
        np.testing.assert_array_equal(got[1:2], ref, err_msg=name)


@pytest.mark.parametrize("case", cases_for("missing_covariate"))
def test_missing_query_covariate(case):
    model = fitted(case)
    Z = np.array(case.Z, dtype=float)
    k = 2
    Z[k, 0] = np.nan
    for fname, event in calls(case):
        if fname in case.jump_functions:
            continue
        ref = call(case, model, fname, case.x, case.Z, event)
        got = call(case, model, fname, case.x, Z, event)
        _check_nan_where(got, ref, [k], fname)


def _with_nan(data, key, k):
    out = dict(data)
    value = np.array(data[key], dtype=float)
    if value.ndim == 1:
        value[k] = np.nan
    else:
        value[k, 0] = np.nan
    out[key] = value
    return out


@pytest.mark.parametrize("case", cases_for("missing_fit"))
def test_missing_time_at_fit_raises(case):
    data = case.data()
    for key in ("x", "y"):
        if key not in data:
            continue
        with pytest.raises(ValueError):
            refit(case, _with_nan(data, key, 1))


def _missing_input_params():
    # (case, key): each covariate matrix and grouping label of a case.
    params = []
    for param in cases_for("missing_fit"):
        case = param.values[0]
        keys = ((case.covariates,) if case.covariates else ()) + case.labels
        for key in keys:
            marks = list(param.marks)
            # A known failure for one input is listed as "missing_fit[key]".
            reason = case.xfail.get(f"missing_fit[{key}]")
            if reason:
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            params.append(
                pytest.param(case, key, id=f"{case.name}-{key}", marks=marks)
            )
    return params


@pytest.mark.parametrize("case, key", _missing_input_params())
def test_missing_covariate_or_label_at_fit(case, key):
    data = case.data()
    k = 3
    bad = _with_nan(data, key, k)
    if key == case.covariates and not case.drops_missing_covariate:
        with pytest.raises(ValueError):
            refit(case, bad)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = case.fit(bad)
    dropped = [w for w in caught if "Dropped" in str(w.message)]
    assert len(dropped) == 1, [str(w.message) for w in caught]
    assert issubclass(dropped[0].category, UserWarning)
    keep = np.arange(len(data["x"])) != k
    clean = {
        name: np.asarray(v)[keep] if name in case.rows else v
        for name, v in data.items()
    }
    got = predictions(case, model)
    ref = predictions(case, refit(case, clean))
    for name in ref:
        np.testing.assert_allclose(
            got[name], ref[name], rtol=1e-9, atol=1e-12, err_msg=name
        )


def _semi_parametric_fits():
    from surpyval import (
        AdditiveHazards,
        BuckleyJames,
        CompetingRisksProportionalHazards,
        CoxPH,
        FineGray,
        ProportionalOdds,
    )

    # Each takes (x, Z, c, e) and returns the fitted coefficients; ``e``
    # (the causes) is read by the competing-risks models only.
    return [
        ("CoxPH", lambda x, Z, c, e: CoxPH.fit(x, Z, c=c).params),
        (
            "ProportionalOdds",
            lambda x, Z, c, e: ProportionalOdds.fit(x, Z, c=c).params,
        ),
        (
            "AdditiveHazards",
            lambda x, Z, c, e: AdditiveHazards.fit(x, Z, c=c).params,
        ),
        ("BuckleyJames", lambda x, Z, c, e: BuckleyJames.fit(x, Z, c=c).beta),
        (
            "FineGray",
            lambda x, Z, c, e: FineGray.fit(x, Z, e, c=c, event="a").beta,
        ),
        (
            "CompetingRisksProportionalHazards",
            lambda x, Z, c, e: CompetingRisksProportionalHazards.fit(
                x, Z, e, c=c
            ).betas,
        ),
    ]


@pytest.mark.parametrize(
    "name, fit",
    [pytest.param(*nf, id=nf[0]) for nf in _semi_parametric_fits()],
)
def test_bad_covariate_row_is_dropped_before_the_event_time_check(name, fit):
    # The semi-parametric fitters share one input check (consolidation
    # sweep, decision 4): a row with a missing covariate is dropped, with
    # the usual warning, before the exactly observed times are checked to
    # be finite, as Cox did (principle 3). The proportional odds, Lin-Ying
    # and Buckley-James fits raised "must be finite" on such a row.
    rng = np.random.default_rng(4)
    x = rng.exponential(5.0, 40) + 0.1
    Z = rng.normal(size=(40, 1))
    c = (rng.uniform(size=40) < 0.2).astype(int)
    c[:2] = 0
    # Two causes; a censored row has none.
    e = np.where(np.arange(40) % 2 == 0, "a", "b").astype(object)
    e[c == 1] = None
    x_bad, Z_bad, c_bad, e_bad = x.copy(), Z.copy(), c.copy(), e.copy()
    x_bad[5], Z_bad[5, 0], c_bad[5], e_bad[5] = np.inf, np.nan, 0, "b"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = fit(x_bad, Z_bad, c_bad, e_bad)
    dropped = [w for w in caught if "Dropped" in str(w.message)]
    assert len(dropped) == 1, [str(w.message) for w in caught]
    keep = np.arange(40) != 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = fit(x[keep], Z[keep], c[keep], e[keep])
    np.testing.assert_allclose(got, ref, rtol=1e-12, err_msg=name)
    # An infinite observed time on a row that is kept is still refused.
    x_bad[5], Z_bad[5, 0] = np.inf, 0.0
    with pytest.raises(ValueError, match="must be finite"):
        fit(x_bad, Z_bad, c_bad, e_bad)
