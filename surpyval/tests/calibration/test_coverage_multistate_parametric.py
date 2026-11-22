"""Coverage of the parametric multi-state models' bounds on the state
occupation probabilities (``cb``, at its default ``bound_type``), where a
state is rarely occupied.

A cut-down version of ``scripts/studies/multistate_occupancy`` (which has
eight designs, three sizes and 1000 replicates, and whose README gives
the results). Units start healthy at 0 and are followed to a uniform
(10, 20) end; the hazards are functions of the time since the start
(``clock="forward"``):

- an illness-death model with constant hazards (healthy -> ill 0.02,
  healthy -> dead 0.10, ill -> dead 0.5), fitted with ``Exponential``:
  1.5% to 2.5% of units are ill at any of the times asked about;
- a progressive model with Weibull hazards (healthy -> ill scale 30 and
  shape 1.2, ill -> dead scale 2 and shape 1.3), fitted with ``Weibull``:
  2.4% to 3% ill;
- the first with a binary covariate (half the units) that multiplies
  every hazard by ``exp(0.5)``, fitted with ``ExponentialPH`` and asked
  about at ``Z = 1``.

``N = 300`` units (600 with the covariate) and 400 replicates. Checked at
90% and 95%: the bound on the probability of being healthy, ill and dead
at 2, 5 and 10 from the start, and of being ill, and dead, at 3 and 5 for
a unit that is ill at 2 (which rests on the few ill -> dead spells
alone). The truth is exact but for one integral, done by Gauss-Legendre
quadrature. The seed, replicates, sizes, targets and slack (0.01, the
suite's rule) were fixed before any run.
"""

import os
import warnings

import numpy as np
import pytest

import surpyval
from surpyval.multistate import (
    ParametricMultiState,
    ParametricMultiStateRegression,
)
from surpyval.tests.calibration._montecarlo import (
    check_coverage,
    process_pool,
)

N = 300
REPS = 400
SEED = 34120
STATES = ("healthy", "ill", "dead")
# (state, time, from_state, start)
TARGETS = tuple(
    [(st, t, None, 0.0) for st in STATES for t in (2.0, 5.0, 10.0)]
    + [(st, t, "ill", 2.0) for st in ("ill", "dead") for t in (3.0, 5.0)]
)
PAIRS = {
    "01": ("healthy", "ill"),
    "02": ("healthy", "dead"),
    "12": ("ill", "dead"),
}
# {transition: (scale, shape)}: H(t) = (t / scale) ** shape
DESIGNS = {
    "illness-death, constant": (
        "Exponential",
        {"01": (1 / 0.02, 1.0), "02": (1 / 0.10, 1.0), "12": (1 / 0.5, 1.0)},
    ),
    "progressive, Weibull": (
        "Weibull",
        {"01": (30.0, 1.2), "02": None, "12": (2.0, 1.3)},
    ),
}
BETA = 0.5


def _H(par, t):
    return 0.0 if par is None else (t / par[0]) ** par[1]


def _truth(pars):
    """The targets' occupation probabilities: ``exp(-H01(t) - H02(t))``
    healthy; ill the integral over the time ``u`` of falling ill of
    ``exp(-H01(u) - H02(u)) h01(u) exp(-(H12(t) - H12(u)))``; dead the
    rest; and ``exp(-(H12(t) - H12(s)))`` still ill for a unit ill at
    ``s``."""
    v, w = np.polynomial.legendre.leggauss(400)
    v, w = (v + 1) / 2, w / 2
    a, b = pars["01"]
    out = []
    for state, t, frm, s in TARGETS:
        if frm == "ill":
            stay = np.exp(-(_H(pars["12"], t) - _H(pars["12"], s)))
            out.append(stay if state == "ill" else 1 - stay)
            continue
        p00 = np.exp(-_H(pars["01"], t) - _H(pars["02"], t))
        u = t * v
        inner = np.exp(
            -_H(pars["01"], u)
            - _H(pars["02"], u)
            - (_H(pars["12"], t) - _H(pars["12"], u))
        )
        p01 = t * np.sum(w * inner * (b / a) * (u / a) ** (b - 1))
        out.append({"healthy": p00, "ill": p01, "dead": 1 - p00 - p01}[state])
    return np.array(out)


def _times(par, rng, size, after=0.0):
    if par is None:
        return np.full(size, np.inf)
    a, b = par
    return a * (rng.exponential(size=size) + (after / a) ** b) ** (1 / b)


def _spells(pars, n, rng):
    t01 = _times(pars["01"], rng, n)
    t02 = _times(pars["02"], rng, n)
    ill = t01 < t02
    leave = np.minimum(t01, t02)
    death = np.where(
        ill, _times(pars["12"], rng, n, np.where(ill, leave, 0.0)), leave
    )
    end = rng.uniform(10.0, 20.0, size=n)
    x, e, state, tl, i = [], [], [], [], []

    def add(u, *row):
        for column, value in zip((x, e, state, tl), row):
            column.append(value)
        i.append(u)

    for u in range(n):
        if leave[u] >= end[u]:
            add(u, end[u], None, "healthy", 0.0)
        elif not ill[u]:
            add(u, leave[u], "dead", "healthy", 0.0)
        else:
            add(u, leave[u], "ill", "healthy", 0.0)
            if death[u] >= end[u]:
                add(u, end[u], None, "ill", leave[u])
            else:
                add(u, death[u], "dead", "ill", leave[u])
    return (
        np.array(x),
        np.array(e, dtype=object),
        np.array(state, dtype=object),
        np.array(tl),
        np.array(i),
    )


def _transitions(pars):
    return [PAIRS[k] for k, p in pars.items() if p is not None]


def _bands(cb):
    """``(level, target, 2)`` bounds from ``cb(t, state, s, from_state,
    alpha_ci)``."""
    out = np.empty((2, len(TARGETS), 2))
    for a, alpha in enumerate((0.10, 0.05)):
        for j, (st, t, frm, s) in enumerate(TARGETS):
            out[a, j] = np.ravel(cb(t, st, s, frm, alpha))
    return out


def _one(args):
    name, r = args
    dist, pars = DESIGNS[name]
    rng = np.random.default_rng([SEED, list(DESIGNS).index(name), r])
    x, e, state, tl, i = _spells(pars, N, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ParametricMultiState.fit(
            x,
            e,
            state,
            tl=tl,
            i=i,
            transitions=_transitions(pars),
            dist=getattr(surpyval, dist),
        )
        return _bands(
            lambda t, st, s, frm, alpha: model.cb(
                t, st, s=s, from_state=frm, alpha_ci=alpha
            )
        )


def _scaled(pars, factor):
    # (a constant hazard times ``factor``: the scale divided by it)
    return {k: (a / factor, b) for k, (a, b) in pars.items()}


def _one_regression(r):
    pars = DESIGNS["illness-death, constant"][1]
    rng = np.random.default_rng([SEED, 9, r])
    parts = []
    for z, p in ((0.0, pars), (1.0, _scaled(pars, np.exp(BETA)))):
        x, e, state, tl, i = _spells(p, N, rng)
        parts.append((x, e, state, tl, i + int(z) * N, np.full(x.size, z)))
    x, e, state, tl, i, Z = (
        np.concatenate([part[k] for part in parts]) for k in range(6)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ParametricMultiStateRegression.fit(
            x,
            Z[:, None],
            e,
            state,
            tl=tl,
            i=i,
            transitions=_transitions(pars),
            model=surpyval.ExponentialPH,
        )
        return _bands(
            lambda t, st, s, frm, alpha: model.cb(
                t, [1.0], st, s=s, from_state=frm, alpha_ci=alpha
            )
        )


def _check(bands, truth, label):
    for a, level in enumerate((0.90, 0.95)):
        check_coverage(
            bands[:, a, :, 0],
            bands[:, a, :, 1],
            truth,
            level,
            f"{label} occupancy at {level:.0%}",
        )


@pytest.mark.parametrize("name", list(DESIGNS))
def test_coverage_of_the_occupancy_bounds(name):
    with process_pool(max_workers=min(os.cpu_count() or 1, 4)) as pool:
        bands = np.array(
            list(pool.map(_one, [(name, r) for r in range(REPS)]))
        )
    _check(bands, _truth(DESIGNS[name][1]), f"ParametricMultiState ({name})")


def test_coverage_of_the_regression_occupancy_bounds():
    with process_pool(max_workers=min(os.cpu_count() or 1, 4)) as pool:
        bands = np.array(list(pool.map(_one_regression, range(REPS))))
    pars = _scaled(DESIGNS["illness-death, constant"][1], np.exp(BETA))
    _check(bands, _truth(pars), "ParametricMultiStateRegression at Z = 1")
