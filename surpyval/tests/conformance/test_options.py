"""Option combinations (#379, principles 18 and 21).

The rest of the suite calls each model with its default options; many
past bugs lived in the others. This module sweeps them:

- every uncertainty method a case declares (its ``bounds``, see
  :class:`~registry.Bound`) over ``on=``, ``bound=``, the levels in
  :data:`~registry.ALPHAS` and its variants (``method=``,
  ``bound_type=``, ``interp=``), checking that the bounds contain the
  estimate, stay in the function's range (or the parameter's support)
  with lower <= upper, that a one-sided bound at alpha is the matching
  end of the two-sided bound at 2 alpha, that a smaller alpha gives a
  wider interval around the narrower one, that a Wald interval closes
  onto the estimate as alpha -> 1, that the bounds on ``ff`` and ``Hf``
  are those on ``sf`` transformed, and the documented shapes;
- every ``interp=`` value a case declares: a valid curve, equal to the
  step estimate at its step times, and an unknown value refused;
- every estimation option a case declares (``how=``, Cox's ``method=``,
  the non-parametric estimator choices): each gives a valid model, and on
  a large sample from the model they agree;
- the names, meanings and defaults of the shared options across every
  registered model (principle 21), in ``test_option_convention``.

A case whose uncertainty methods are not all declared fails
``cb_declared``, so a new model's bounds join the sweep.
"""

import contextlib
import functools
import importlib
import inspect
import warnings
from dataclasses import replace

import numpy as np
import pytest

from surpyval.tests.conformance.checks import (
    COUNTING_RULES,
    RULES,
    check_valid,
)
from surpyval.tests.conformance.registry import (
    ALPHAS,
    CASES,
    KNOWN_INCONSISTENCIES,
    NON_STRICT,
    Q_PROBS,
    WITH_COVARIATES,
    call,
    cases_for,
    fitted,
    refit,
    skip_without_finite_maximum,
)

SIDES = ("two-sided", "lower", "upper")

# Public methods taking a level or bound= that are not swept, and why.
NOT_SWEPT = {
    "plot": "draws the bounds of a swept method",
    "get_plot_data": "collects the bounds of a swept method for plot",
    "R_cb": "the survival bounds behind cb in [upper, lower] order "
    "(documented); cb(on='sf') is the method to call",
    "life_parameter_covariance": "a covariance, not an interval",
    "summary": "a table of the parameters' Wald intervals (param_cb's for "
    "the baseline; checked in regression/test_summary.py, #484)",
    "trend_test": "a hypothesis test: alpha_ci is the level its trend is "
    "concluded at (#481), not an interval's",
    "repair_test": "a hypothesis test: alpha_ci is the level its "
    "conclusion about the repair is drawn at, not an interval's",
    "cb_tvc": "needs a covariate path; checked against cb along a constant "
    "path, for every bound and on=, in conformance/test_tvc.py (#172)",
}
# "confidence" was the recurrent models' level until v0.22; a method
# that took it again would be an unswept uncertainty method.
_LEVEL_NAMES = ("alpha_ci", "confidence")


# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------
def _bound_params(prop, where=None):
    """(case, bound) pairs ``prop`` applies to.

    A known failure of one method is keyed ``"<prop>[<bound name>]"``.
    """
    params = []
    for param in cases_for(prop):
        case = param.values[0]
        for spec in case.bounds:
            if where is not None and not where(case, spec):
                continue
            marks = list(param.marks)
            key = f"{prop}[{spec.name}]"
            reason = case.xfail.get(key)
            if reason:
                # Non-strict where the outcome depends on the build
                strict = key not in NON_STRICT.get(case.name, ())
                marks.append(pytest.mark.xfail(strict=strict, reason=reason))
            if spec.slow:
                marks.append(pytest.mark.slow)
            if spec.nightly:
                # Opt in with --run-calibration (the root conftest).
                marks.append(pytest.mark.calibration)
            params.append(
                pytest.param(
                    case, spec, id=f"{case.name}-{spec.name}", marks=marks
                )
            )
    return params


def _functions(case, spec):
    """(function, cause) pairs a "function" bound is swept over."""
    names = spec.on or (spec.point,)
    causes = case.events if spec.per_cause else (None,)
    return [(f, e) for f in names for e in causes]


# ---------------------------------------------------------------------------
# Calling an uncertainty method; results are cached, as every invariant
# reads the same sweep
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _query(case, spec):
    if spec.on or spec.point != "qf":
        return np.asarray(spec.query, float) if spec.query else case.x
    return Q_PROBS


def _level(spec, alpha):
    return {spec.level: alpha}


def _parameters(model):
    """(names, estimates, supports) of the parameters ``param_cb`` takes."""
    if hasattr(model, "_parameter_bounds"):
        names = list(model.parameter_names)
        values = getattr(model, "_mle", None)
        values = model.params if values is None else values
        supports = model._parameter_bounds()
        # An accelerated life model's life parameter is given by its life
        # model, not estimated, and param_cb refuses it (#489).
        keep = [
            k
            for k, n in enumerate(names)
            if n != getattr(model, "life_parameter", None)
        ]
        return (
            [names[k] for k in keep],
            np.asarray(values, float)[keep],
            [supports[k] for k in keep],
        )
    if hasattr(model, "_param_vector"):  # a frailty model
        names = list(model.parameter_names)
        # (no distribution for a Cox baseline)
        dist = [] if model.dist is None else list(model.dist.bounds)
        supports = [
            (
                (0, None)
                if n == "theta"
                else dist[k] if k < len(dist) else (None, None)
            )
            for k, n in enumerate(names)
        ]
        return names, np.asarray(model._param_vector(), float), supports
    names = list(model.dist.parameter_names)
    values = list(model.params)
    supports = list(model.dist.bounds)
    if model.lfp:
        names.append("lfp_p" if "p" in names else "p")
        values.append(model.p)
        supports.append((0, 1))
    if model.zi:
        names.append("f0")
        values.append(model.f0)
        supports.append((0, 1))
    return names, np.asarray(values, float), supports


def _raw(case, spec, model, side, alpha, fname=None, event=None, k=None):
    """One call of the method, as a (rows,) or (rows, 2) array; ``k``
    queries the ``k``-th time alone, as a scalar (a () or (2,) result)."""
    with _silenced():
        return _call(case, spec, model, side, alpha, fname, event, k)


@contextlib.contextmanager
def _silenced():
    # The sweeps check values. Warnings, leaked or deliberate, are
    # principle 22's (test_warnings.py): a bound is computed once and
    # cached, so a leak would fail whichever test happened to compute it
    # first, and the xfails here would depend on the order of the tests.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        yield


def _call(case, spec, model, side, alpha, fname, event, k):
    kw = dict(spec.kwargs)
    kw.update(_level(spec, alpha))
    if spec.sides:
        kw["bound"] = side
    method = getattr(model, spec.method)
    if spec.kind == "function":
        x = _query(case, spec)
        if spec.on:
            kw["on"] = fname
        args = [x if k is None else x[k]]
        if case.interface in WITH_COVARIATES:
            args.append(case.Z[: x.size] if k is None else case.Z[k])
        if spec.per_cause:
            args.append(event)
        return np.asarray(method(*args, **kw), float)
    if spec.kind == "param":
        names = _parameters(model)[0]
        rows = [np.asarray(method(n, **kw), float) for n in names]
        return np.vstack(rows) if side == "two-sided" else np.hstack(rows)
    if spec.kind == "coef":
        return np.asarray(method(**kw), float)
    rows = []
    for args in spec.query:
        out = method(*args, **kw)
        if spec.kind == "rul":
            rows.append(out.rul_interval)
        elif isinstance(out, dict):  # rmst
            rows.append((out["lower"], out["upper"]))
        else:
            rows.append(out)
    return np.asarray(rows, float)


def bounds(case, spec, side="two-sided", alpha=0.05, fname=None, event=None):
    """The method's bounds at the case's query (cached)."""
    key = (case.name, spec.name, side, alpha, fname, event)
    if key not in _CACHE:
        _CACHE[key] = _raw(case, spec, fitted(case), side, alpha, fname, event)
    return _CACHE[key]


def estimate(case, spec, fname=None, event=None):
    """The estimate the method's bounds are about, one per row."""
    with _silenced():
        return _estimate(case, spec, fname, event)


def _estimate(case, spec, fname, event):
    model = fitted(case)
    if spec.kind == "function":
        # The estimate interpolated as its bounds are.
        extra = {k: v for k, v in spec.kwargs.items() if k == "interp"}
        c = replace(case, call_kwargs={**case.call_kwargs, **extra})
        x = _query(case, spec)
        Z = case.Z[: x.size] if case.interface in WITH_COVARIATES else None
        return np.asarray(call(c, model, fname, x, Z, event), float)
    if spec.kind == "param":
        return _parameters(model)[1]
    if spec.kind == "coef":
        return np.asarray(model.coef, float)
    out = []
    for args in spec.query:
        if spec.kind == "rul":
            out.append(getattr(model, spec.method)(*args, **spec.kwargs).rul)
        elif spec.method == "rmst":
            out.append(model.rmst(*args)["rmst"])
        else:
            out.append(model.mean())
    return np.asarray(out, float)


def value_range(case, spec, fname):
    """(lower, upper) arrays of the valid values, one per row."""
    rows = len(estimate(case, spec, fname, _functions(case, spec)[0][1]))
    if spec.kind == "param":
        supports = _parameters(fitted(case))[2]
        lo = [-np.inf if s[0] is None else s[0] for s in supports]
        hi = [np.inf if s[1] is None else s[1] for s in supports]
        return np.asarray(lo, float), np.asarray(hi, float)
    if spec.kind == "function":
        counting = case.interface.startswith("counting")
        lower, upper, _ = (COUNTING_RULES if counting else RULES)[fname]
    elif spec.kind in ("rul", "summary"):
        lower, upper = 0.0, np.inf
    else:
        lower, upper = -np.inf, np.inf
    return np.full(rows, lower), np.full(rows, upper)


def _sweep(case, spec):
    """(label, function, cause) for each curve the method bounds."""
    if spec.kind != "function":
        return [(spec.kind, None, None)]
    return [
        (f if e is None else f"{f}[{e}]", f, e)
        for f, e in _functions(case, spec)
    ]


def _tol(spec, ref):
    return spec.rtol * np.maximum(np.abs(ref), 1.0) + 1e-12


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", cases_for("cb_declared"))
def test_every_uncertainty_method_is_swept(case):
    model = fitted(case)
    declared = {spec.method for spec in case.bounds}
    missing = []
    for name in dir(model):
        if name.startswith("_") or name in NOT_SWEPT or name in declared:
            continue
        try:
            params = inspect.signature(getattr(model, name)).parameters
        except (TypeError, ValueError, AttributeError):
            continue
        if any(p in params for p in _LEVEL_NAMES + ("bound",)):
            missing.append(name)
    assert not missing, (
        f"{case.name}: uncertainty methods not declared in its bounds "
        f"(registry.py, _bounds): {missing}"
    )


def _percentile(spec):
    return "n_boot" in spec.kwargs


# A percentile bootstrap interval need not contain the estimate (its ends
# are quantiles of the refits, which a biased estimator shifts), so the
# bootstraps are left out of containment and closure.
@pytest.mark.parametrize(
    "case, spec",
    _bound_params("cb_contains", where=lambda c, s: not _percentile(s)),
)
def test_bounds_contain_the_estimate(case, spec):
    skip_without_finite_maximum(case)
    for label, fname, event in _sweep(case, spec):
        p = estimate(case, spec, fname, event)
        lo_ok, hi_ok = value_range(case, spec, fname)
        # An estimate outside the function's range (an additive hazards
        # survival above 1, documented) cannot be inside a bound that
        # respects the range.
        check = np.isfinite(p) & (p >= lo_ok - 1e-12) & (p <= hi_ok + 1e-12)
        tol = _tol(spec, p)
        for alpha in ALPHAS:
            sides = SIDES if spec.sides else ("two-sided",)
            for side in sides:
                b = bounds(case, spec, side, alpha, fname, event)
                lo = b[..., 0] if side == "two-sided" else b
                hi = b[..., 1] if side == "two-sided" else b
                if side != "upper":
                    _assert_side(spec, label, alpha, side, p, lo, check, tol)
                    assert np.all((lo <= p + tol)[check & ~np.isnan(lo)]), (
                        f"{label} {side} alpha={alpha}: lower {lo} above "
                        f"the estimate {p}"
                    )
                if side != "lower":
                    _assert_side(spec, label, alpha, side, p, hi, check, tol)
                    assert np.all((hi >= p - tol)[check & ~np.isnan(hi)]), (
                        f"{label} {side} alpha={alpha}: upper {hi} below "
                        f"the estimate {p}"
                    )


def _assert_side(spec, label, alpha, side, p, b, check, tol):
    # A finite estimate has a bound, except where one is documented as
    # NaN (outside the range of the data).
    if not spec.nan_ok:
        missing = check & np.isnan(b)
        assert not missing.any(), (
            f"{label} {side} alpha={alpha}: NaN bound where the estimate "
            f"is {p[missing]}"
        )


@pytest.mark.parametrize("case, spec", _bound_params("cb_range"))
def test_bounds_are_ordered_and_in_range(case, spec):
    for label, fname, event in _sweep(case, spec):
        lower, upper = value_range(case, spec, fname)
        for alpha in ALPHAS:
            b = bounds(case, spec, "two-sided", alpha, fname, event)
            lo, hi = b[..., 0], b[..., 1]
            both = ~np.isnan(lo) & ~np.isnan(hi)
            assert np.all(
                lo[both] <= hi[both] + _tol(spec, hi[both])
            ), f"{label} alpha={alpha}: lower {lo} above upper {hi}"
            if not spec.in_range:
                continue
            ends = [lo, hi]
            if spec.sides:
                ends += [
                    bounds(case, spec, s, alpha, fname, event)
                    for s in ("lower", "upper")
                ]
            for v in ends:
                ok = np.isnan(v) | (
                    (v >= lower - 1e-12) & (v <= upper + 1e-12)
                )
                assert ok.all(), (
                    f"{label} alpha={alpha}: bound {v} outside "
                    f"[{lower}, {upper}]"
                )


@pytest.mark.parametrize(
    "case, spec", _bound_params("cb_sides", where=lambda c, s: s.sides)
)
def test_one_sided_is_an_end_of_two_sided(case, spec):
    for label, fname, event in _sweep(case, spec):
        for alpha in ALPHAS:
            two = bounds(case, spec, "two-sided", 2 * alpha, fname, event)
            for k, side in enumerate(("lower", "upper")):
                one = bounds(case, spec, side, alpha, fname, event)
                np.testing.assert_allclose(
                    one,
                    two[..., k],
                    rtol=spec.rtol,
                    atol=1e-12,
                    err_msg=f"{label}: {side} at {alpha} vs two-sided at "
                    f"{2 * alpha}",
                )


@pytest.mark.parametrize("case, spec", _bound_params("cb_nested"))
def test_smaller_alpha_is_wider(case, spec):
    for label, fname, event in _sweep(case, spec):
        levels = sorted(ALPHAS)
        for wide_a, narrow_a in zip(levels[:-1], levels[1:]):
            wide = bounds(case, spec, "two-sided", wide_a, fname, event)
            narrow = bounds(case, spec, "two-sided", narrow_a, fname, event)
            tol = _tol(spec, narrow)
            for k, sign in ((0, 1), (1, -1)):
                w, n = wide[..., k], narrow[..., k]
                both = ~np.isnan(w) & ~np.isnan(n)
                with np.errstate(invalid="ignore"):
                    ok = (w == n) | (sign * (n - w) >= -tol[..., k])
                assert ok[both].all(), (
                    f"{label}: the {1 - wide_a:.0%} interval "
                    f"{wide} does not contain the {1 - narrow_a:.0%} one "
                    f"{narrow}"
                )


@pytest.mark.parametrize(
    "case, spec", _bound_params("cb_centre", where=lambda c, s: s.wald)
)
def test_interval_closes_onto_the_estimate(case, spec):
    # At alpha_ci = 1 - 1e-6 a Wald interval is the estimate +- 1.25e-6
    # standard errors (on its transformed scale): the estimate, to 1e-5.
    for label, fname, event in _sweep(case, spec):
        p = estimate(case, spec, fname, event)
        b = bounds(case, spec, "two-sided", 1 - 1e-6, fname, event)
        lo, hi = value_range(case, spec, fname)
        # As for containment, an estimate outside the function's range
        # (an additive-hazards sf above 1) is left out; and an estimate on
        # the edge of a parameter's support has no Wald interval around
        # it (documented for a frailty's theta -> 0: [0, inf]).
        keep = np.isfinite(p) & (p >= lo - 1e-12) & (p <= hi + 1e-12)
        if spec.kind == "param":
            keep &= (p - lo > 1e-8 * np.abs(p) + 1e-12) & (hi - p > 1e-8)
        for k in (0, 1):
            keep = keep & ~np.isnan(b[..., k])
            np.testing.assert_allclose(
                b[..., k][keep],
                p[keep],
                rtol=max(spec.rtol, 1e-5),
                atol=1e-10,
                err_msg=f"{label}: the interval at alpha_ci -> 1 is not "
                "centred on the estimate",
            )


def _transforms(case, spec):
    return (
        spec.kind == "function"
        and "sf" in spec.on
        and ("ff" in spec.on or "Hf" in spec.on)
    )


@pytest.mark.parametrize(
    "case, spec", _bound_params("cb_transform", where=_transforms)
)
def test_ff_and_Hf_bounds_are_the_sf_bounds_transformed(case, spec):
    percentile = _percentile(spec)
    targets = {"ff": lambda s: 1.0 - s}
    if not percentile:
        # A percentile of -log(sf) interpolates between the draws on the
        # Hf scale, so it equals -log of the sf percentile only at a draw.
        targets["Hf"] = lambda s: -np.log(s)
    with np.errstate(all="ignore"):
        for fname, g in targets.items():
            if fname not in spec.on:
                continue
            for alpha in ALPHAS:
                sf = bounds(case, spec, "two-sided", alpha, "sf")
                got = bounds(case, spec, "two-sided", alpha, fname)
                np.testing.assert_allclose(
                    _underflow(got, g(sf[..., ::-1])),
                    g(sf[..., ::-1]),
                    rtol=max(spec.rtol, 1e-10),
                    atol=1e-12,
                    err_msg=f"{fname} two-sided at {alpha}",
                )
                if not spec.sides:
                    continue
                for side, other in (("lower", "upper"), ("upper", "lower")):
                    want = g(bounds(case, spec, other, alpha, "sf"))
                    np.testing.assert_allclose(
                        _underflow(
                            bounds(case, spec, side, alpha, fname), want
                        ),
                        want,
                        rtol=max(spec.rtol, 1e-10),
                        atol=1e-12,
                        err_msg=f"{fname} {side} at {alpha}",
                    )


def _underflow(got, want):
    """``got`` with inf where ``want`` is -log of an sf bound that
    underflowed to 0: an Hf bound computed on its own scale is finite
    there (#418), and must be past -log of the smallest double."""
    under = np.isposinf(want) & np.isfinite(got)
    assert np.all(got[under] > 744.0), got[under]
    return np.where(under, np.inf, got)


def _function_bound(case, spec):
    return spec.kind == "function"


@pytest.mark.parametrize(
    "case, spec", _bound_params("cb_shape", where=_function_bound)
)
def test_bound_shapes(case, spec):
    model = fitted(case)
    x = _query(case, spec)
    for label, fname, event in _sweep(case, spec):
        sides = SIDES if spec.sides else ("two-sided",)
        for side in sides:
            b = bounds(case, spec, side, 0.05, fname, event)
            want = (x.size, 2) if side == "two-sided" else (x.size,)
            assert b.shape == want, (label, side, b.shape)
            # A scalar query keeps its shape (principle 7): the pair
            # [lower, upper] two-sided, one number one-sided.
            k = x.size // 2
            one = _raw(case, spec, model, side, 0.05, fname, event, k)
            assert one.shape == want[1:], (label, side, one.shape)
            # (A search warm-starts from the previous time, so it agrees
            # to its own tolerance.)
            np.testing.assert_allclose(
                one,
                b[k],
                rtol=max(spec.rtol, 1e-10),
                atol=1e-12,
                err_msg=label,
            )


@pytest.mark.parametrize(
    "case, spec",
    _bound_params("cb_api", where=lambda c, s: s.sides or bool(s.on)),
)
def test_bound_and_on_values(case, spec):
    model = fitted(case)
    fname, event = (
        _functions(case, spec)[0]
        if spec.kind == "function"
        else (
            None,
            None,
        )
    )
    if spec.sides:
        with pytest.raises(ValueError):
            _raw(case, spec, model, "both", 0.05, fname, event)
    for alias, fname in (("R", "sf"), ("F", "ff")):
        if fname not in spec.on:
            continue
        for side in SIDES:
            np.testing.assert_array_equal(
                _raw(case, spec, model, side, 0.05, alias, event),
                bounds(case, spec, side, 0.05, fname, event),
                err_msg=f"on={alias!r} {side}",
            )


# ---------------------------------------------------------------------------
# interp=
# ---------------------------------------------------------------------------
def _interp_functions(case, model):
    """(function, cause) pairs of the case that take ``interp=``."""
    out = []
    for fname, event in [(f, None) for f in case.functions] + [
        (f, e) for f in case.event_functions for e in case.events
    ]:
        params = inspect.signature(getattr(model, fname)).parameters
        if "interp" in params:
            out.append((fname, event))
    return out


def _step_times(case, model, event):
    own = model.models[event] if event is not None else model
    return np.unique(np.asarray(own.x, float))


def _interp_params():
    params = []
    for param in cases_for("interp"):
        case = param.values[0]
        for value in case.interp:
            marks = list(param.marks)
            reason = case.xfail.get(f"interp[{value}]")
            if reason:
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            params.append(
                pytest.param(
                    case, value, id=f"{case.name}-{value}", marks=marks
                )
            )
    return params


@pytest.mark.parametrize("case, value", _interp_params())
def test_interp_gives_a_valid_curve(case, value):
    model = fitted(case)
    rules = COUNTING_RULES if case.interface.startswith("counting") else RULES
    functions = _interp_functions(case, model)
    assert functions, case.name
    for fname, event in functions:
        c = replace(case, call_kwargs={**case.call_kwargs, "interp": value})
        values = np.asarray(call(c, model, fname, case.x, event=event), float)
        # Documented: an interpolated curve is NaN outside the range of
        # the data, and the jumps are NaN before the first failure.
        check_valid(fname, values[~np.isnan(values)], rules[fname])
        if fname in case.jump_functions:
            continue
        # At its own step times every curve is the step estimate.
        knots = _step_times(case, model, event)
        Z = None
        if case.interface in WITH_COVARIATES:
            Z = case.Z[0]
        step = replace(
            case, call_kwargs={**case.call_kwargs, "interp": "step"}
        )
        np.testing.assert_allclose(
            np.asarray(call(c, model, fname, knots, Z, event), float),
            np.asarray(call(step, model, fname, knots, Z, event), float),
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"{fname}(interp={value!r}) at the step times",
        )


@pytest.mark.parametrize("case", cases_for("interp_refused"))
def test_unknown_interp_is_refused(case):
    model = fitted(case)
    for fname, event in _interp_functions(case, model):
        c = replace(case, call_kwargs={**case.call_kwargs, "interp": "bogus"})
        with pytest.raises(ValueError):
            call(c, model, fname, case.x, event=event)


# ---------------------------------------------------------------------------
# Estimation options
# ---------------------------------------------------------------------------
def _estimator_params():
    params = []
    for param in cases_for("estimators"):
        case = param.values[0]
        for option, values in case.estimators.items():
            for value in values:
                marks = list(param.marks)
                key = f"estimators[{option}={value}]"
                if key in case.xfail:
                    marks.append(
                        pytest.mark.xfail(strict=True, reason=case.xfail[key])
                    )
                params.append(
                    pytest.param(
                        case,
                        option,
                        value,
                        id=f"{case.name}-{option}={value}",
                        marks=marks,
                    )
                )
    return params


def _check_model(case, model):
    """test_bounds' checks: valid values, monotone in time, at each
    covariate row (as one vector for every time)."""
    if case.interface == "bivariate":
        X = case.x[:5]  # each coordinate rises along these
        check_valid("cdf", np.asarray(model.cdf(X), float), (0.0, 1.0, 1))
        check_valid("sf", np.asarray(model.sf(X), float), (0.0, 1.0, -1))
        return
    rules = COUNTING_RULES if case.interface.startswith("counting") else RULES
    rows = [None] if case.Z is None else list(case.Z)
    for z in rows:
        for fname, event in [(f, None) for f in case.functions] + [
            (f, e) for f in case.event_functions for e in case.events
        ]:
            x = Q_PROBS if fname == "qf" else case.x
            values = np.asarray(call(case, model, fname, x, z, event), float)
            if fname in case.jump_functions:
                values = values[~np.isnan(values)]
            check_valid(fname, values, rules[fname])


@pytest.mark.parametrize("case, option, value", _estimator_params())
def test_estimation_option_gives_a_valid_model(case, option, value):
    model = refit(case, {**case.data(), option: value})
    _check_model(case, model)


def _agreement_params():
    params = []
    for param in cases_for("estimators_agree"):
        case = param.values[0]
        for option in case.estimators:
            marks = list(param.marks)
            key = f"estimators_agree[{option}]"
            if key in case.xfail:
                marks.append(
                    pytest.mark.xfail(strict=True, reason=case.xfail[key])
                )
            params.append(
                pytest.param(
                    case, option, id=f"{case.name}-{option}", marks=marks
                )
            )
    return params


def _summary(case, model):
    """The functions the estimators are compared on: probabilities
    absolutely, a cumulative intensity relatively."""
    if case.interface == "bivariate":
        return {"cdf": np.asarray(model.cdf(case.x), float)}, False
    if "sf" in case.functions:
        return {
            "sf": np.asarray(call(case, model, "sf", case.x), float)
        }, False
    out = {}
    for fname, event in [(f, None) for f in case.functions] + [
        (f, e) for f in case.event_functions for e in case.events
    ]:
        if fname in ("cif", "mcf"):
            out[f"{fname}[{event}]"] = np.asarray(
                call(case, model, fname, case.x, event=event), float
            )
    return out, case.interface.startswith("counting")


@pytest.mark.parametrize("case, option", _agreement_params())
def test_estimation_options_agree_on_a_large_sample(case, option):
    data = case.large(fitted(case))
    values = case.estimators[option] + case.estimators_large.get(option, ())
    fits = {v: refit(case, {**data, option: v}) for v in values}
    ref_value = values[0]
    ref, relative = _summary(case, fits[ref_value])
    for value, model in fits.items():
        got, _ = _summary(case, model)
        for key in ref:
            if relative:
                np.testing.assert_allclose(
                    got[key],
                    ref[key],
                    rtol=0.05,
                    atol=0.05,
                    err_msg=f"{option}={value!r} vs {ref_value!r}: {key}",
                )
            else:
                np.testing.assert_allclose(
                    got[key],
                    ref[key],
                    rtol=0,
                    atol=case.estimators_atol,
                    err_msg=f"{option}={value!r} vs {ref_value!r}: {key}",
                )


# ---------------------------------------------------------------------------
# One name, meaning and default per option (principle 21)
# ---------------------------------------------------------------------------
def _resolve(dotted):
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _signatures(models_only=False):
    """(owner, method, parameters) for every public method of every
    registered model class and (unless ``models_only``) fitter; read
    from the classes, without fitting anything."""
    owners = {}
    for case in CASES:
        dotted = (case.model_class,)
        if not models_only:
            dotted += case.fitters
        for obj in map(_resolve, dotted):
            cls = obj if isinstance(obj, type) else type(obj)
            owners.setdefault(cls.__name__.rstrip("_"), obj)
    out = []
    for owner, obj in sorted(owners.items()):
        for name in dir(obj):
            if name.startswith("_"):
                continue
            method = getattr(obj, name, None)
            if not callable(method) or isinstance(method, type):
                continue
            try:
                params = list(inspect.signature(method).parameters.values())
            except (TypeError, ValueError):
                continue
            if params and params[0].name == "self":
                params = params[1:]  # a method read from its class
            out.append((owner, name, params))
    return out


_TIME_FUNCTIONS: tuple[str, ...] = (
    "sf",
    "ff",
    "df",
    "hf",
    "Hf",
    "cif",
    "iif",
    "mcf",
    "cb",
)
_TIME_FUNCTIONS += ("cif_cb", "mcf_cb", "bootstrap_cb", "band")
_TIME_FUNCTIONS += ("iif_cb", "mtbf", "mtbf_cb")


def _default_is(name, value):
    def check(sigs):
        return [
            f"{o}.{m}({name}={p.default!r})"
            for o, m, ps in sigs
            for p in ps
            if p.name == name and p.default != value
        ]

    return check


def _spelled(*names):
    """Offenders when more than one of ``names`` is used: every use of
    the less common spellings."""

    def check(sigs):
        uses = {n: [] for n in names}
        for o, m, ps in sigs:
            for p in ps:
                if p.name in uses:
                    uses[p.name].append(f"{o}.{m}")
        used = [n for n in names if uses[n]]
        if len(used) < 2:
            return []
        return [f"{n}: {sorted(set(uses[n]))}" for n in used]

    return check


def _level_name(sigs):
    # The level of an interval is alpha_ci (0.05), never a confidence.
    out = [
        f"{o}.{m}(confidence=...)"
        for o, m, ps in sigs
        for p in ps
        if p.name in ("confidence", "confidence_level", "conf_level", "level")
    ]
    return out + _default_is("alpha_ci", 0.05)(sigs)


def _first_argument(methods, name):
    def check(sigs):
        return [
            f"{o}.{m}({ps[0].name}, ...)"
            for o, m, ps in sigs
            if m in methods and ps and ps[0].name not in (name, "args")
        ]

    return check


def _covariates_second(sigs):
    # Covariates are Z, right after the times (or the size of a draw).
    out = []
    for o, m, ps in sigs:
        names = [p.name for p in ps]
        if m in _TIME_FUNCTIONS + ("random", "qf") and "Z" in names:
            if names.index("Z") != 1:
                out.append(f"{o}.{m}({', '.join(names[:3])}, ...)")
    return out


def _how(sigs):
    # how= is the estimation method, defaulting to maximum likelihood. A
    # copula's default is the two-stage inference-functions-for-margins
    # estimate (documented; MLE is the joint alternative), and the
    # non-parametric competing risks' is its Nelson-Aalen survival (it
    # has no likelihood).
    own = {"Copula": "IFM", "CompetingRisks": "Nelson-Aalen"}
    return [
        f"{o}.{m}(how={p.default!r})"
        for o, m, ps in sigs
        for p in ps
        if p.name == "how"
        and p.default != "MLE"
        and not any(o.endswith(k) and p.default == v for k, v in own.items())
    ]


CONVENTIONS = {
    "alpha_ci": ("the level of an interval is alpha_ci=0.05", _level_name),
    "bound": (
        "bound= defaults to 'two-sided'",
        _default_is("bound", "two-sided"),
    ),
    "on": ("on= defaults to 'sf'", _default_is("on", "sf")),
    "interp": ("interp= defaults to 'step'", _default_is("interp", "step")),
    "seed": (
        "the random-number argument has one spelling",
        _spelled("random_state", "seed"),
    ),
    "resamples": (
        "the number of bootstrap resamples has one spelling",
        _spelled("n_boot", "B"),
    ),
    "ties": (
        "Cox's tie-handling option has one spelling",
        _spelled("method", "tie_method"),
    ),
    "time": (
        "the times are the first argument, x",
        _first_argument(_TIME_FUNCTIONS, "x"),
    ),
    "quantile": (
        "the probabilities of qf are the first argument, p",
        _first_argument(("qf", "quantile_cb"), "p"),
    ),
    "cause": (
        "the cause of a per-cause prediction is event=",
        _spelled("event", "cause"),
    ),
    "Z": ("covariates are Z, the second argument", _covariates_second),
    "how": (
        "how= is the estimation method, default 'MLE' where there is a "
        "likelihood",
        _how,
    ),
    "id column": (
        "the item-id column of a data frame has one spelling",
        _spelled("i_col", "id_col"),
    ),
}


def _convention_params():
    params = []
    for key in CONVENTIONS:
        marks = []
        if key in KNOWN_INCONSISTENCIES:
            marks.append(
                pytest.mark.xfail(
                    strict=True, reason=KNOWN_INCONSISTENCIES[key]
                )
            )
        params.append(pytest.param(key, id=key, marks=marks))
    return params


# Conventions of the prediction API, read on the fitted models only: a
# fitter's own distribution functions take the parameters as arguments
# (Weibull.qf(u, alpha, beta)), a layer of its own.
_MODEL_API = ("time", "quantile", "Z")


@functools.cache
def _all_signatures(models_only):
    return _signatures(models_only)


@pytest.mark.parametrize("key", _convention_params())
def test_option_convention(key):
    what, check = CONVENTIONS[key]
    sigs = _all_signatures(key in _MODEL_API)
    if key == "ties":
        # "method" means other things elsewhere (a bound's or an
        # estimator's method); the tie option is Cox's.
        sigs = [
            s
            for s in sigs
            if s[0] in ("CoxPH", "CompetingRisksProportionalHazards")
        ]
    if key == "cause":
        sigs = [s for s in sigs if s[1] in _TIME_FUNCTIONS + ("mcf_cb",)]
    offenders = check(sigs)
    assert not offenders, f"{what}; not so in: {offenders}"
