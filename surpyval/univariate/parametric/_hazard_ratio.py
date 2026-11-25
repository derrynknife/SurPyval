"""The hazard ratio ``hr`` of a parametric model.

A model with a hazard ratio ``hr`` has ``hr`` times the hazard of the
same model without it at every age, the offset ``gamma``, limited-failure
proportion ``lfp_p`` and zero inflation ``f0`` included: its cumulative
hazard is ``hr * H0`` and its survival ``R0 ** hr``, where ``H0`` and
``R0`` are the model's without ``hr``. So ``R ** (1 - b) * R ** b = R``
for every model and every ``b``. A zero-inflation mass ``f0`` at 0
becomes ``1 - (1 - f0) ** hr``.

``Parametric`` keeps its own functions for ``hr = 1`` and hands every one
to this module otherwise. Each function here takes the model and works
from :func:`base`, the same model with ``hr = 1``, whose functions are
already precise in both tails (``Hf`` from ``ff`` where it is small).
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as onp

from surpyval.utils.aliases import canonical_name
from surpyval.utils.validation import warn_outside_unit_interval

#: The distributions whose hazard-scaled form is the same family at
#: other parameters, with the map ``(params, hr) -> params``: a Weibull's
#: scale becomes ``alpha * hr ** (-1 / beta)`` and an exponential's rate
#: ``hr * lambda``.
_SAME_FAMILY = {
    "Weibull": lambda params, hr: [
        params[0] * hr ** (-1.0 / params[1]),
        params[1],
    ],
    "Exponential": lambda params, hr: [params[0] * hr],
}


def checked_hr(hr: Any) -> float:
    """The hazard ratio given to ``from_params`` or ``fit``, as a float,
    refusing anything but a finite number above 0."""
    value = float(hr) if isinstance(hr, numbers.Real) else onp.nan
    if isinstance(hr, bool) or not (onp.isfinite(value) and value > 0):
        raise ValueError(
            "hr, the hazard ratio, must be a finite number above 0 (1 "
            f"leaves the model as it is); got {hr!r}."
        )
    return value


def refuse_unless_one(hr: Any, what: str, alternative: str) -> None:
    """Refuse a hazard ratio other than 1 given to a model that cannot
    carry one (``what``), saying what to use; ``None`` and 1 pass."""
    if hr is None:
        return
    if checked_hr(hr) != 1:
        raise ValueError(
            f"{what} does not take a hazard ratio (hr = {float(hr):g}): "
            f"{alternative}"
        )


def base(model: Any) -> Any:
    """The model without its hazard ratio: the same object's state with
    ``hr = 1`` (no cached mean)."""
    out = object.__new__(type(model))
    state = dict(model.__dict__)
    state.pop("_mean", None)
    state["hr"] = 1.0
    out.__dict__.update(state)
    return out


def at(model: Any, core: Any, p: float = 1.0, f0: float = 0.0) -> Any:
    """The model, its hazard ratio and offset kept, at the distribution
    parameters ``core``, limited-failure proportion ``p`` and zero
    inflation ``f0``: what a delta-method bound differentiates (``mean``,
    ``mrl`` and the restricted mean of a model with ``hr``)."""
    out = object.__new__(type(model))
    state = dict(model.__dict__)
    state.pop("_mean", None)
    state["params"] = onp.array(core, dtype=float)
    state["lfp_p"] = float(p)
    state["f0"] = float(f0)
    out.__dict__.update(state)
    return out


def same_family(model: Any) -> Any:
    """The model as one of its own family with ``hr = 1``, where the
    family is closed under a hazard ratio (Weibull, Exponential) and the
    model has no limited failure population (an offset and zero
    inflation carry across: ``f0`` becomes ``1 - (1 - f0) ** hr``);
    ``None`` otherwise."""
    if model.lfp_p != 1 or getattr(model.dist, "custom", False):
        return None
    rule = _SAME_FAMILY.get(canonical_name(model.dist))
    if rule is None:
        return None
    from surpyval.univariate.parametric.parametric import (
        is_custom_distribution,
    )

    if is_custom_distribution(model.dist):
        return None
    hr = float(model.hr)
    kwargs: dict[str, Any] = {}
    if model.offset:
        kwargs["gamma"] = model.gamma
    if model.zi:
        kwargs["f0"] = float(-onp.expm1(hr * onp.log1p(-float(model.f0))))
    params = rule([float(v) for v in onp.asarray(model.params)], hr)
    return model.dist.from_params(params, **kwargs)


def zero_mass(model: Any) -> float:
    """The probability of failing at 0 of a zero-inflated model with its
    hazard ratio, ``1 - (1 - f0) ** hr``."""
    return float(-onp.expm1(model.hr * onp.log1p(-float(model.f0))))


def Hf(model: Any, x: Any) -> Any:
    return model.hr * base(model).Hf(x)


def sf(model: Any, x: Any) -> Any:
    return onp.exp(-model.hr * base(model).Hf(x))


def ff(model: Any, x: Any) -> Any:
    return -onp.expm1(-model.hr * base(model).Hf(x))


def hf(model: Any, x: Any) -> Any:
    b = base(model)
    h0 = onp.asarray(b.hf(x), dtype=float)
    if model.dist.discrete:
        # The probability of failing at k given survival to it: one less
        # the chance of getting through k, (1 - h0) ** hr.
        with onp.errstate(divide="ignore"):
            out = -onp.expm1(model.hr * onp.log1p(-h0))
    else:
        out = model.hr * h0
        if model.f0 != 0:
            # The point mass at 0 is a discrete hazard there
            out = onp.where(onp.asarray(x) == 0, zero_mass(model), out)
    return onp.asarray(out)[()]


def df(model: Any, x: Any, continuous: bool = False) -> Any:
    x_arr = onp.asarray(x, dtype=float)
    if model.dist.discrete:
        # P(T = k) = R(k - 1) h(k)
        out = onp.asarray(model.sf(x_arr - 1.0)) * onp.asarray(model.hf(x))
        return onp.asarray(out, dtype=float)[()]
    b = base(model)
    h0 = onp.asarray(b.hf(x), dtype=float)
    if model.f0 != 0:
        # At 0 the continuous part's own hazard, not the mass's
        at0 = x_arr == 0
        if onp.any(at0):
            with onp.errstate(divide="ignore", invalid="ignore"):
                own = onp.asarray(b.df(x, continuous=True)) / onp.asarray(
                    b.sf(x)
                )
            h0 = onp.where(at0, own, h0)
    s = onp.asarray(model.sf(x), dtype=float)
    with onp.errstate(invalid="ignore"):
        out = onp.where(s == 0, 0.0, model.hr * h0 * s)
    if model.f0 != 0 and not continuous:
        out = onp.where(x_arr == 0, zero_mass(model), out)
    return onp.asarray(out, dtype=float)[()]


def isf(model: Any, s: Any) -> Any:
    if isinstance(s, (list, tuple)):
        s = onp.array(s, dtype=float)
    u = onp.asarray(s, dtype=float)
    scalar = u.ndim == 0
    u = onp.atleast_1d(u)
    bad = warn_outside_unit_interval(u)
    with onp.errstate(divide="ignore", invalid="ignore"):
        # log R0 = log(R) / hr
        log_r0 = onp.log(onp.where(bad, 0.5, u)) / model.hr
    q = _at_log_survival(model, log_r0)
    if model.dist.discrete:
        q = _first_point(model, q, onp.where(bad, 0.5, u))
    q = onp.where(bad | onp.isnan(u), onp.nan, q) + 0.0
    return q[0] if scalar else q


def _at_log_survival(model: Any, log_r0: Any) -> Any:
    """The time at which the model without its hazard ratio has the
    survival ``exp(log_r0)``: by its ``isf`` where that is below 1/2, and
    by its ``qf`` of ``-expm1(log_r0)`` above, which keeps the digits of
    a small failure probability (``1 - R0`` rounds to 0 below 1.1e-16,
    where an exponential's ``qf(1e-20)`` is not 0)."""
    log_r0 = onp.atleast_1d(onp.asarray(log_r0, dtype=float))
    b = base(model)
    upper = log_r0 < -onp.log(2.0)
    lower = ~upper & ~onp.isnan(log_r0)
    q = onp.full(log_r0.shape, onp.nan)
    with onp.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if onp.any(upper):
            q[upper] = b.isf(onp.exp(log_r0[upper]))
        if onp.any(lower):
            q[lower] = b.qf(-onp.expm1(log_r0[lower]))
    return q


def _first_point(model: Any, k: Any, s: Any) -> Any:
    """The first point ``k`` of a discrete model's support with ``sf(k)
    <= s``, by the model's own ``sf``: ``R0 ** (1 / hr)`` is rounded, so
    the point found from it can be one off where ``s`` is a value of
    ``sf`` itself."""
    with onp.errstate(invalid="ignore"):
        lo = float(model.dist.support[0])
        finite = onp.isfinite(k) & (s > 0) & (s < 1)
        kk = onp.where(finite, k, lo)
        back = finite & (kk - 1 >= lo)
        back &= onp.asarray(model.sf(kk - 1), dtype=float) <= s
        kk = onp.where(back, kk - 1, kk)
        ahead = finite & ~back & (onp.asarray(model.sf(kk), dtype=float) > s)
        kk = onp.where(ahead, kk + 1, kk)
    return onp.where(finite, kk, k)


def qf(model: Any, p: Any) -> Any:
    if isinstance(p, (list, tuple)):
        p = onp.array(p, dtype=float)
    u = onp.asarray(p, dtype=float)
    scalar = u.ndim == 0
    u = onp.atleast_1d(u)
    bad = warn_outside_unit_interval(u)
    with onp.errstate(divide="ignore", invalid="ignore"):
        # log R0 = log(1 - p) / hr, without rounding 1 - p
        log_r0 = onp.log1p(-onp.where(bad, 0.5, u)) / model.hr
    q = _at_log_survival(model, log_r0)
    if model.dist.discrete:
        q = _first_point(model, q, 1.0 - onp.where(bad, 0.5, u))
    q = onp.where(bad | onp.isnan(u), onp.nan, q) + 0.0
    return q[0] if scalar else q


def _refuse(model: Any, what: str, alternative: str) -> ValueError:
    return ValueError(
        f"{what} of a {model.dist.name} model with a hazard ratio "
        f"(hr = {model.hr:g}) is not available: {alternative}"
    )


#: The most points a discrete model's sums run over before refusing.
_MAX_POINTS = 2**24


def discrete_raw_moment(model: Any, n: int) -> float:
    r"""``E[T ** n]`` of a discrete model on the counts ``0, 1, 2, ...``
    with a hazard ratio and no limited failure population, from its
    survival function :math:`R(k) = P(T > k)`: :math:`\sum_{k \ge 0}
    ((k + 1)^n - k^n) R(k)` (the mean is :math:`\sum_k R(k)`), summed in
    blocks until a block adds nothing."""
    total = 0.0
    start, size = 0, 1024
    while start < _MAX_POINTS:
        k = onp.arange(start, start + size, dtype=float)
        s = onp.asarray(model.sf(k), dtype=float)
        part = float(onp.sum(((k + 1.0) ** n - k**n) * s))
        total += part
        if s[-1] == 0 or (s[-1] < 1e-17 and part <= 1e-17 * total):
            return total
        start += size
        size *= 2
    raise _refuse(
        model,
        f"Moment {n}",
        f"its sum over the counts has not settled after {_MAX_POINTS} "
        "points (a tail that falls as a power of k, whose moment may be "
        "infinite); the mean up to a count n is "
        "model.sf(numpy.arange(n)).sum().",
    )


def raw_moment(model: Any, n: int) -> float:
    r"""``E[T ** n]`` of a model with a hazard ratio and no limited
    failure population, by quadrature over its survival function:
    :math:`\int_0^\infty n t^{n-1} R(t)\,dt - \int_{-\infty}^0 n t^{n-1}
    F(t)\,dt`."""
    if model.dist.discrete:
        return discrete_raw_moment(model, n)
    from scipy.integrate import quad

    from surpyval.utils.warnings import ignoring_warnings

    def upper(t: float) -> float:
        return n * t ** (n - 1) * float(model.sf(t))

    def lower(t: float) -> float:
        return n * t ** (n - 1) * float(model.ff(t))

    lo = float(getattr(model, "support", (-onp.inf, onp.inf))[0])
    start = max(lo, 0.0) if onp.isfinite(lo) else 0.0
    total = 0.0
    with ignoring_warnings():
        if start > 0:
            # R = 1 below the support: int_0^start n t^(n-1) dt
            total += start**n
        value, _ = quad(upper, start, onp.inf, epsabs=0.0, epsrel=1e-11)
        total += value
        if lo < 0:
            neg, _ = quad(
                lower, max(lo, -onp.inf), 0.0, epsabs=0.0, epsrel=1e-11
            )
            total -= neg
    return float(total)


def mean(model: Any, defective: bool) -> float:
    if model.lfp_p < 1 and not defective:
        # (1 - lfp_p) ** hr never fails
        return onp.inf
    same = same_family(model)
    if same is not None:
        return same.mean(defective=defective)
    if model.lfp_p < 1:
        raise _refuse(
            model,
            "The defective mean",
            "it has no closed form with a limited failure population; "
            "model.mean(tau=...) gives the restricted mean.",
        )
    return raw_moment(model, 1)


def moment(model: Any, n: int, defective: bool) -> float:
    if model.lfp_p < 1 and n >= 1 and not defective:
        return onp.inf
    same = same_family(model)
    if same is not None:
        return same.moment(n, defective=defective)
    if n == 0:
        return 1.0
    if model.lfp_p < 1:
        raise _refuse(
            model,
            "A defective moment",
            "it has no closed form with a limited failure population.",
        )
    return raw_moment(model, n)


def var(model: Any, defective: bool) -> float:
    if model.lfp_p < 1 and not defective:
        return onp.inf
    same = same_family(model)
    if same is not None:
        return same.var(defective=defective)
    m1 = mean(model, defective)
    return float(moment(model, 2, defective) - m1**2)


def entropy(model: Any) -> float:
    r"""The differential entropy :math:`-\int f \log f\,dt`: the
    family's closed form for a Weibull or Exponential, and otherwise by
    quadrature, with :math:`\log f = \log hr + \log h_0 - hr\,H_0` from
    the model without its hazard ratio (finite far into the upper tail,
    where ``f`` underflows)."""
    same = same_family(model)
    if same is not None and not model.zi:
        return same.entropy()
    if model.lfp_p != 1 or model.f0 != 0:
        # The atoms at 0 or at infinity: the model's own refusal
        return base(model).entropy()
    if model.dist.discrete:
        raise _refuse(
            model,
            "The entropy",
            "SurPyval has no entropy for a discrete distribution; sum "
            "-p * log(p) over p = model.df(numpy.arange(k)).",
        )
    return _entropy_by_quadrature(model)


def _entropy_by_quadrature(model: Any) -> float:
    from scipy.integrate import quad

    from surpyval.utils.warnings import ignoring_warnings

    b = base(model)
    log_hr = float(onp.log(model.hr))

    def integrand(t: float) -> float:
        with onp.errstate(all="ignore"):
            h0 = float(b.hf(t))
            log_f = log_hr + onp.log(h0) - model.hr * float(b.Hf(t))
        if not onp.isfinite(log_f):
            # f = 0 (or the edge of the support): no contribution
            return 0.0
        return -float(onp.exp(log_f)) * log_f

    lo, hi = (float(v) for v in model.support)
    # Split at the quartiles, so each piece has the mass to find
    q1, q2, q3 = (float(v) for v in model.qf([0.25, 0.5, 0.75]))
    edges = [lo, q1, q2, q3, hi]
    total = 0.0
    with ignoring_warnings():
        for a, c in zip(edges[:-1], edges[1:]):
            if c > a:
                value, _ = quad(
                    integrand, a, c, epsabs=0.0, epsrel=1e-11, limit=200
                )
                total += value
    return float(total)


def mrl(model: Any, x: Any) -> Any:
    t = onp.atleast_1d(onp.asarray(x, dtype=float))
    if model.lfp_p < 1:
        return onp.where(onp.isnan(t), onp.nan, onp.inf)
    same = same_family(model)
    if same is not None:
        return onp.atleast_1d(onp.asarray(same.mrl(t), dtype=float))
    if model.dist.discrete:
        raise _refuse(
            model,
            "The mean residual life",
            "for a discrete distribution sum the survival beyond the age, "
            "as model.sf(numpy.arange(k, n)).sum() / model.sf(k) at age k.",
        )
    from scipy.integrate import quad

    from surpyval.utils.warnings import ignoring_warnings

    out = onp.full(t.shape, onp.nan)
    for i, ti in enumerate(t.ravel()):
        if onp.isnan(ti):
            continue
        h_at = float(model.Hf(ti))
        if not onp.isfinite(h_at):
            continue

        def cs(u: float) -> float:
            # R(ti + u) / R(ti), from the cumulative hazards
            return float(onp.exp(h_at - float(model.Hf(ti + u))))

        lo = float(getattr(model, "support", (-onp.inf, onp.inf))[0])
        area = 0.0
        start = 0.0
        if onp.isfinite(lo) and ti < lo:
            # Nothing fails before the support starts
            area += lo - ti
            start = lo - ti
        with ignoring_warnings():
            value, _ = quad(cs, start, onp.inf, epsabs=0.0, epsrel=1e-11)
        out.flat[i] = area + value
    return out


def rmst(model: Any, taus: Any) -> Any:
    """``int_0^tau R(t) dt`` at each ``tau``."""
    same = same_family(model)
    if same is not None:
        return same._rmst_values(taus)
    from scipy.integrate import quad

    from surpyval.utils.warnings import ignoring_warnings

    out = onp.full(len(taus), onp.nan)
    for i, tau in enumerate(onp.asarray(taus, dtype=float)):
        if onp.isnan(tau):
            continue
        with ignoring_warnings():
            value, _ = quad(
                lambda u: float(model.sf(u)),
                0.0,
                tau,
                epsabs=0.0,
                epsrel=1e-11,
                limit=200,
            )
        out[i] = value
    return out
