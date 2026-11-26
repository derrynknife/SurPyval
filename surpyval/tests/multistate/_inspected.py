"""Visits of a progressive illness-death model as the long form of the
parametric multi-state models, with the onset of illness interval
censored."""

import numpy as np

NAMES = {1: "healthy", 2: "ill", 3: "dead"}
TRANSITIONS = [("healthy", "ill"), ("healthy", "dead"), ("ill", "dead")]


def spells(ids, times, states, names=NAMES, before=None):
    """``(x, e, state, tl, i)``: each unit's visits (``times``, the
    ``states`` seen, an exactly timed death as state 3) as spells. A move
    seen between two visits is interval censored, ``x = [last visit in
    the old state, first visit in the new]``; a death is a move at its
    time from the state seen at the visit before it; a unit alive at its
    last visit is censored there (a spell of no length, a unit seen in
    its state only at its last visit, is left out: the move into it says
    the unit was there). With ``before`` (each row's state just before
    it, read at a death), a unit that fell ill unseen and died before the
    next visit has the onset interval censored between its last visit
    and its death, and the ill spell that ends in the death starts in
    that interval."""
    ids, times, states = (np.asarray(v) for v in (ids, times, states))
    known = before is not None
    before = np.asarray(states if before is None else before)
    rows = []
    for u in np.unique(ids):
        sel = ids == u
        t, s, pre = times[sel], states[sel], before[sel]
        order = np.argsort(t)
        t, s, pre = t[order], s[order], pre[order]
        cur, entry, last = s[0], t[0], t[0]
        for k in range(1, t.size):
            if s[k] == cur:
                last = t[k]
                continue
            if s[k] == 3 and known and pre[k] != cur:
                # (fell ill since the last visit, then died)
                rows.append((u, entry, last, t[k], cur, pre[k]))
                rows.append((u, last, t[k], t[k], pre[k], 3))
                cur = None
                break
            if s[k] == 3:
                rows.append((u, entry, t[k], t[k], cur, 3))
                cur = None
                break
            rows.append((u, entry, last, t[k], cur, s[k]))
            cur, entry, last = s[k], t[k], t[k]
        if cur is not None and last > entry:
            rows.append((u, entry, last, last, cur, None))
    x = np.array([[r[2], r[3]] for r in rows], dtype=float)
    e = np.array(
        [None if r[5] is None else names[int(r[5])] for r in rows],
        dtype=object,
    )
    state = np.array([names[int(r[4])] for r in rows], dtype=object)
    tl = np.array([r[1] for r in rows], dtype=float)
    i = np.array([r[0] for r in rows])
    return x, e, state, tl, i


def _H(par, t):
    a, b = par
    return (t / a) ** b


def _h(par, t):
    a, b = par
    return b / a * (t / a) ** (b - 1)


def simulate(n, seed, pars, clock="forward", gap=1.0):
    """``n`` units of the illness-death model with Weibull hazards
    ``pars`` (``{(from, to): (scale, shape)}``, on the time since the
    start, or with ``clock="reset"`` since entering the state), healthy at
    0, seen every ``gap`` until a uniform (5, 10) end, deaths timed
    exactly, as the long form (see :func:`spells`)."""
    ids, times, states, before = visits(n, seed, pars, clock, gap)
    return spells(ids, times, states, before=before)


def visits(n, seed, pars, clock="forward", gap=1.0):
    """The visits of :func:`simulate`: ``(ids, times, states, before)``,
    a death as state 3 at its exact time (``before`` the state just before
    each row)."""
    from scipy.optimize import brentq

    rng = np.random.default_rng(seed)
    ids, times, states, before = [], [], [], []
    for u in range(n):
        end = float(rng.uniform(5, 10))
        visits = list(np.arange(0.0, end, gap)) + [end]
        state, t, path = 1, 0.0, [(0.0, 1)]
        while state != 3:
            out = [
                (3 if b == "dead" else 2, (a, b))
                for (a, b) in pars
                if a == ("healthy" if state == 1 else "ill")
            ]
            entry = t if clock == "reset" else 0.0
            e = float(rng.exponential())

            def total(v, out=out, t0=t, entry=entry):
                return sum(
                    _H(pars[p], v - entry) - _H(pars[p], t0 - entry)
                    for _, p in out
                )

            if total(end) <= e:
                break
            t_new = brentq(lambda v: total(v) - e, t, end, xtol=1e-12)
            rates = np.array([_h(pars[p], t_new - entry) for _, p in out])
            k = rng.choice(len(out), p=rates / rates.sum())
            state, t = out[k][0], t_new
            path.append((t, state))
        death = path[-1][0] if path[-1][1] == 3 else None
        for v in visits:
            ids.append(u)
            if death is not None and v >= death:
                times.append(death)
                states.append(3)
                before.append(path[-2][1])
                break
            times.append(v)
            states.append([s for w, s in path if w <= v][-1])
            before.append(states[-1])
    return ids, times, states, before


def visits_covariates(n, seed, pars, betas, clock="forward", gap=1.0):
    """The visits of :func:`visits` with covariates: ``male`` (0 or 1,
    fixed for a unit, half the units) and ``w`` (0 or 1, drawn afresh at
    each visit, holding until the next), multiplying each transition's
    Weibull hazard by ``exp(b_male male + b_w w)`` (``betas``: ``{(from,
    to): (b_male, b_w)}``; proportional hazards on either clock). Returns
    ``(ids, times, states, Z)``, ``Z`` one row ``(male, w)`` per row, a
    death's row with the covariates of the visit before it."""
    from scipy.optimize import brentq

    rng = np.random.default_rng(seed)
    ids, times, states, Z = [], [], [], []
    for u in range(n):
        male = float(rng.uniform() < 0.5)
        end = float(rng.uniform(5, 10))
        visits = list(np.arange(0.0, end, gap)) + [end]
        w = [float(rng.uniform() < 0.5) for _ in visits]
        state, t, path = 1, 0.0, [(0.0, 1)]
        entry = 0.0
        g = 0
        e = float(rng.exponential())
        while state != 3 and g < len(visits) - 1:
            hi = visits[g + 1]
            out = [
                (
                    3 if b == "dead" else 2,
                    (a, b),
                    np.exp(betas[(a, b)][0] * male + betas[(a, b)][1] * w[g]),
                )
                for (a, b) in pars
                if a == ("healthy" if state == 1 else "ill")
            ]
            zero = entry if clock == "reset" else 0.0

            def total(v, out=out, t0=t, zero=zero):
                return sum(
                    m * (_H(pars[p], v - zero) - _H(pars[p], t0 - zero))
                    for _, p, m in out
                )

            if total(hi) <= e:
                e -= total(hi)
                t = hi
                g += 1
                continue
            t_new = brentq(lambda v: total(v) - e, t, hi, xtol=1e-12)
            rates = np.array(
                [m * _h(pars[p], t_new - zero) for _, p, m in out]
            )
            k = rng.choice(len(out), p=rates / rates.sum())
            state, t = out[k][0], t_new
            entry = t
            path.append((t, state))
            e = float(rng.exponential())
        death = path[-1][0] if path[-1][1] == 3 else None
        for j, v in enumerate(visits):
            ids.append(u)
            if death is not None and v >= death:
                times.append(death)
                states.append(3)
                Z.append((male, w[j - 1]))
                break
            times.append(v)
            states.append([s for x, s in path if x <= v][-1])
            Z.append((male, w[j]))
    return ids, times, states, np.array(Z, dtype=float)
