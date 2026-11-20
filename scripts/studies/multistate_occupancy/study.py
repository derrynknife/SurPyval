"""Coverage of the delta-method bounds on state occupation probabilities
of ``ParametricMultiState`` (``cb``), for common and rare states, against
a parametric-bootstrap band on the same data.

    python scripts/studies/multistate_occupancy/study.py run [n_jobs]
    python scripts/studies/multistate_occupancy/study.py truth
    python scripts/studies/multistate_occupancy/study.py summary

The designs, sample sizes, replicates, seeds, targets, methods and the
rule for reading the result are in ``README.md`` beside this script; they
were written before the first run. ``run`` appends one line per replicate
to ``results/<design>_n<n>.jsonl`` (a rerun skips the replicates already
there, so a killed run resumes); ``summary`` turns them into
``summary.json`` and ``table.txt``; ``truth`` prints each design's true
occupation probabilities three ways (the quadrature the study uses,
``scipy.integrate.quad`` and a 10^6-path simulation with seed 20261120).
Run from the repository root with the package importable.
"""

import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

#: Each transition is a Weibull hazard of the time since the start,
#: ``H(t) = (t / scale) ** shape`` (shape 1: a constant hazard). ``None``:
#: the transition does not exist (a progressive model).
DESIGNS = {
    # name: (fitted distribution, {transition: (scale, shape)})
    "id_exp_rare": (
        "Exponential",
        {"01": (1 / 0.02, 1.0), "02": (1 / 0.10, 1.0), "12": (1 / 0.5, 1.0)},
    ),
    "id_exp_common": (
        "Exponential",
        {"01": (1 / 0.15, 1.0), "02": (1 / 0.05, 1.0), "12": (1 / 0.10, 1.0)},
    ),
    "prog_exp_rare": (
        "Exponential",
        {"01": (1 / 0.03, 1.0), "02": None, "12": (1 / 0.6, 1.0)},
    ),
    "prog_exp_common": (
        "Exponential",
        {"01": (1 / 0.2, 1.0), "02": None, "12": (1 / 0.15, 1.0)},
    ),
    "id_wei_rare": (
        "Weibull",
        {"01": (35.0, 1.3), "02": (10.0, 1.5), "12": (2.0, 1.2)},
    ),
    "id_wei_common": (
        "Weibull",
        {"01": (6.0, 1.3), "02": (20.0, 1.5), "12": (12.0, 1.2)},
    ),
    "prog_wei_rare": (
        "Weibull",
        {"01": (30.0, 1.2), "02": None, "12": (2.0, 1.3)},
    ),
    "prog_wei_common": (
        "Weibull",
        {"01": (6.0, 1.3), "02": None, "12": (10.0, 1.2)},
    ),
}
SIZES = (100, 300, 1000)
SEEDS = range(1000, 2000)
N_BOOT = 500
LEVELS = (0.90, 0.95)
STATES = ("healthy", "ill", "dead")
#: The targets, in order: ``(state, time, from_state, start)``. The first
#: nine are from the start (every unit healthy at 0); the last four are
#: for a unit that is ill at 2.
TARGETS = tuple(
    [(st, t, None, 0.0) for st in STATES for t in (2.0, 5.0, 10.0)]
    + [(st, t, "ill", 2.0) for st in ("ill", "dead") for t in (3.0, 5.0)]
)
CENSOR = (10.0, 20.0)
METHODS = ("exp", "normal", "logit", "boot")

# -- the occupation probabilities as a formula ------------------------------

_M = 4  # u = t v^4: the integrand is smooth at 0 for a shape above 1/4


def _H(par, t):
    if par is None:
        return 0.0
    a, b = par
    return (t / a) ** b


def formula(pars, nodes=200):
    """The targets' occupation probabilities at transition parameters
    ``pars`` (``{"01": (scale, shape), ...}``, each a float or an array
    of draws): ``(draws, len(TARGETS))``.

    Healthy: ``exp(-H01(t) - H02(t))``. Ill: the integral over the time
    ``u`` of falling ill of ``exp(-H01(u) - H02(u)) h01(u) exp(-(H12(t) -
    H12(u)))``, by Gauss-Legendre on ``u = t v^4``. Dead: the rest. For
    a unit ill at ``s``: ``exp(-(H12(t) - H12(s)))`` still ill."""
    v, w = np.polynomial.legendre.leggauss(nodes)
    v, w = (v + 1) / 2, w / 2
    pars = {
        k: None if p is None else tuple(np.atleast_1d(q)[:, None] for q in p)
        for k, p in pars.items()
    }
    size = pars["01"][0].shape[0]
    out = np.empty((size, len(TARGETS)))
    a01, b01 = pars["01"]
    for j, (state, t, frm, s) in enumerate(TARGETS):
        if frm == "ill":
            stay = np.exp(-(_H(pars["12"], t) - _H(pars["12"], s)))[:, 0]
            out[:, j] = stay if state == "ill" else 1 - stay
            continue
        p00 = np.exp(-_H(pars["01"], t) - _H(pars["02"], t))[:, 0]
        u = t * v**_M
        # h01(u) du = (b/a) (u/a)^(b-1) t M v^(M-1) dv
        h_du = (
            (b01 / a01) * (t / a01) ** (b01 - 1) * t * _M * v ** (_M * b01 - 1)
        )
        inner = np.exp(
            -_H(pars["01"], u)
            - _H(pars["02"], u)
            - (_H(pars["12"], t) - _H(pars["12"], u))
        )
        p01 = (inner * h_du) @ w
        out[:, j] = {"healthy": p00, "ill": p01, "dead": 1 - p00 - p01}[state]
    return out


def truth(design):
    return formula(DESIGNS[design][1], nodes=2000)[0]


# -- the data ---------------------------------------------------------------


def _draw(par, rng, size, after=0.0):
    """Times of a transition with ``H(t) = (t / a) ** b`` for units at
    risk of it from ``after``."""
    if par is None:
        return np.full(size, np.inf)
    a, b = par
    e = rng.exponential(size=size)
    return a * (e + (after / a) ** b) ** (1 / b)


def paths(design, n, rng):
    """``(ill, exit, death, end)``: whether each unit falls ill, when it
    leaves 'healthy', when it dies and when its follow-up ends."""
    pars = DESIGNS[design][1]
    t01 = _draw(pars["01"], rng, n)
    t02 = _draw(pars["02"], rng, n)
    ill = t01 < t02
    leave = np.minimum(t01, t02)
    t12 = _draw(pars["12"], rng, n, after=np.where(ill, leave, 0.0))
    death = np.where(ill, t12, leave)
    end = rng.uniform(*CENSOR, size=n)
    return ill, leave, death, end


def sample(design, n, rng):
    """The spells of ``n`` units, all healthy at 0, each followed to its
    own uniform (10, 20) end."""
    ill, leave, death, end = paths(design, n, rng)
    x, e, state, tl, i = [], [], [], [], []
    for u in range(n):
        if leave[u] >= end[u]:
            rows = [(end[u], None, "healthy", 0.0)]
        elif not ill[u]:
            rows = [(leave[u], "dead", "healthy", 0.0)]
        else:
            rows = [(leave[u], "ill", "healthy", 0.0)]
            if death[u] >= end[u]:
                rows.append((end[u], None, "ill", leave[u]))
            else:
                rows.append((death[u], "dead", "ill", leave[u]))
        for row in rows:
            x.append(row[0])
            e.append(row[1])
            state.append(row[2])
            tl.append(row[3])
            i.append(u)
    return (
        np.array(x),
        np.array(e, dtype=object),
        np.array(state, dtype=object),
        np.array(tl),
        np.array(i),
    )


# -- one replicate ----------------------------------------------------------

_PAIRS = {
    "01": ("healthy", "ill"),
    "02": ("healthy", "dead"),
    "12": ("ill", "dead"),
}


def _transitions(design):
    return [_PAIRS[k] for k, p in DESIGNS[design][1].items() if p is not None]


def _scale_shape(dist, values):
    """A transition's ``(scale, shape)`` from its parameters (the
    Exponential's is its rate)."""
    if dist == "Exponential":
        return 1 / values[..., 0], np.ones_like(values[..., 0])
    return values[..., 0], values[..., 1]


def logit_bounds(p, se, level):
    from scipy.special import expit, logit
    from scipy.stats import norm

    z = norm.ppf(0.5 + level / 2)
    with np.errstate(all="ignore"):
        half = z * se / (p * (1 - p))
        return expit(logit(p) - half), expit(logit(p) + half)


def replicate(design, n, seed):
    import surpyval
    from surpyval.multistate import ParametricMultiState
    from surpyval.multistate._base import probability_bounds

    d = list(DESIGNS).index(design)
    dist, pars = DESIGNS[design]
    rng = np.random.default_rng([d, n, seed])
    x, e, state, tl, i = sample(design, n, rng)
    out = {"design": design, "n": n, "seed": seed}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ParametricMultiState.fit(
                x,
                e,
                state,
                tl=tl,
                i=i,
                transitions=_transitions(design),
                dist=getattr(surpyval, dist),
            )
            p = np.empty(len(TARGETS))
            se = np.empty(len(TARGETS))
            cb = {
                (kind, level): np.empty((len(TARGETS), 2))
                for kind in ("exp", "normal")
                for level in LEVELS
            }
            groups: dict = {}
            for j, (st, t, frm, s) in enumerate(TARGETS):
                groups.setdefault((st, frm, s), []).append(j)
            for (st, frm, s), js in groups.items():
                ts = [TARGETS[j][1] for j in js]
                p[js] = model.occupancy(ts, st, s=s, from_state=frm)
                se[js] = model.occupancy_se(ts, st, s=s, from_state=frm)
            # The bounds are the package's own function of (p, se), as
            # ``cb`` computes them; ``cb`` itself is called for one
            # target of each kind and must agree.
            for key in cb:
                cb[key] = probability_bounds(
                    p, se, 1 - key[1], "two-sided", key[0]
                )
            for j in (4, 9):
                st, t, frm, s = TARGETS[j]
                direct = model.cb(t, st, s=s, from_state=frm, alpha_ci=0.05)
                # (to rounding: ``cb`` at one time and at several can
                # differ in the last digit)
                assert np.allclose(
                    np.ravel(direct),
                    cb[("exp", 0.95)][j],
                    rtol=0,
                    atol=1e-12,
                    equal_nan=True,
                )
            # The parametric bootstrap: N_BOOT draws of every
            # transition's parameters from the normal approximation to
            # their estimates on the log scale, each transition
            # independent; the band is the draws' percentiles.
            brng = np.random.default_rng([d, n, seed, 1])
            draws = {}
            fitted = {}
            for k, par in pars.items():
                if par is None:
                    draws[k] = fitted[k] = None
                    continue
                m = model.models[_PAIRS[k]]
                est = np.ravel(np.asarray(m._estimates(), dtype=float))
                cov = np.atleast_2d(m.covariance()) / np.outer(est, est)
                z = brng.multivariate_normal(
                    np.log(est), cov, size=N_BOOT, method="svd"
                )
                draws[k] = _scale_shape(dist, np.exp(z))
                fitted[k] = _scale_shape(dist, est[None, :])
            check = float(np.max(np.abs(formula(fitted)[0] - p)))
            occ = formula(draws)
    except Exception as err:  # a fit that could not be made
        out["failed"] = f"{type(err).__name__}: {err}"[:300]
        return out
    out["p"] = p.tolist()
    out["se"] = se.tolist()
    out["formula_gap"] = check
    for (kind, level), b in cb.items():
        out[f"{kind}_{level}"] = b.tolist()
    for level in LEVELS:
        lo, hi = logit_bounds(p, se, level)
        out[f"logit_{level}"] = np.column_stack([lo, hi]).tolist()
        q = np.nanquantile(occ, [(1 - level) / 2, (1 + level) / 2], axis=0)
        out[f"boot_{level}"] = q.T.tolist()
    return out


def _cell(args):
    design, n, seeds = args
    path = os.path.join(OUT, f"{design}_n{n}.jsonl")
    with open(path, "a") as f:
        for seed in seeds:
            f.write(json.dumps(replicate(design, n, seed)) + "\n")
            f.flush()
    return len(seeds)


def run(n_jobs):
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for design in DESIGNS:
        for n in SIZES:
            path = os.path.join(OUT, f"{design}_n{n}.jsonl")
            done = set()
            if os.path.exists(path):
                with open(path) as f:
                    done = {json.loads(line)["seed"] for line in f}
            todo = [s for s in SEEDS if s not in done]
            # (chunks of 50, so the appends of a cell never interleave
            # mid-line and the pool stays busy)
            for k in range(0, len(todo), 50):
                jobs.append((design, n, todo[k : k + 50]))
    t0 = time.time()
    # one cell's chunks append to the same file: run a cell's chunks in
    # order by giving each worker whole lines (flush per line).
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        total = 0
        for count in pool.map(_cell, jobs):
            total += count
            print(f"{total} replicates, {time.time() - t0:.0f} s", flush=True)


# -- the summary ------------------------------------------------------------


def summarise():
    cells = []
    for d, design in enumerate(DESIGNS):
        tr = truth(design)
        for n in SIZES:
            path = os.path.join(OUT, f"{design}_n{n}.jsonl")
            rows = {}
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    rows[r["seed"]] = r
            assert sorted(rows) == list(SEEDS), (design, n, len(rows))
            ok = [rows[s] for s in SEEDS if "failed" not in rows[s]]
            failed = len(rows) - len(ok)
            p = np.array([r["p"] for r in ok])
            se = np.array([r["se"] for r in ok])
            cell = {
                "design": design,
                "n": n,
                "replicates": len(rows),
                "failed_fits": failed,
                "formula_gap_max": max(r["formula_gap"] for r in ok),
                "targets": [],
            }
            for j, (st, t, frm, s) in enumerate(TARGETS):
                tgt = {
                    "state": st,
                    "time": t,
                    "from_state": frm,
                    "start": s,
                    "truth": float(tr[j]),
                    "mean_estimate": float(np.mean(p[:, j])),
                    "sd_estimate": float(np.std(p[:, j], ddof=1)),
                    "mean_se": float(np.nanmean(se[:, j])),
                    "nan_se": int(np.sum(~np.isfinite(se[:, j]))),
                }
                for method in METHODS:
                    for level in LEVELS:
                        b = np.array(
                            [r[f"{method}_{level}"][j] for r in ok],
                            dtype=float,
                        )
                        lo, hi = b[:, 0], b[:, 1]
                        # (a nan bound is a miss, of neither side)
                        hit = (lo <= tr[j]) & (tr[j] <= hi)
                        tgt[f"{method}_{level}"] = {
                            "coverage": float(np.mean(hit)),
                            "miss_below": float(np.mean(hi < tr[j])),
                            "miss_above": float(np.mean(lo > tr[j])),
                            "mean_width": float(np.nanmean(hi - lo)),
                        }
                cell["targets"].append(tgt)
            cells.append(cell)
    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(cells, f, indent=1)
    lines = []
    for level in LEVELS:
        se_mc = (level * (1 - level) / len(SEEDS)) ** 0.5
        floor = level - 3 * se_mc
        lines.append(
            f"Coverage at {level:.0%} (Monte Carlo SE {se_mc:.4f}; '*' "
            f"below {floor:.4f}, nominal less 3 SE). 'miss' is exp's "
            "(the band below the truth / the band above it); 'w' a mean width."
        )
        lines.append(
            f"{'design':16s}{'n':>5s} {'target':22s}{'truth':>8s}"
            + "".join(f"{m:>9s}" for m in METHODS)
            + f"{'miss exp':>16s}{'w exp':>8s}{'w boot':>8s}"
        )
        for cell in cells:
            for tgt in cell["targets"]:
                frm = "" if tgt["from_state"] is None else " | ill at 2"
                name = f"{tgt['state']} at {tgt['time']:g}{frm}"
                row = f"{cell['design']:16s}{cell['n']:5d} {name:22s}"
                row += f"{tgt['truth']:8.4f}"
                for m in METHODS:
                    c = tgt[f"{m}_{level}"]["coverage"]
                    row += f"{c:8.3f}" + ("*" if c < floor else " ")
                e = tgt[f"exp_{level}"]
                row += f"   {e['miss_below']:.3f} / {e['miss_above']:.3f}"
                row += f"{e['mean_width']:8.4f}"
                row += f"{tgt[f'boot_{level}']['mean_width']:8.4f}"
                lines.append(row)
        lines.append("")
    lines.append("Fits that could not be made (of 1000), by cell:")
    for cell in cells:
        lines.append(
            f"{cell['design']:16s}{cell['n']:5d}  {cell['failed_fits']}"
            f"   (formula against the package, largest gap "
            f"{cell['formula_gap_max']:.1e})"
        )
    with open(os.path.join(HERE, "table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def show_truth():
    from scipy.integrate import quad

    for d, design in enumerate(DESIGNS):
        pars = DESIGNS[design][1]
        tr = truth(design)
        rng = np.random.default_rng(20261120 + d)
        ill, leave, death, _ = paths(design, 10**6, rng)
        print(design)
        for j, (st, t, frm, s) in enumerate(TARGETS):
            if frm is None:
                is_ill = ill & (leave <= t) & (death > t)
                sim = {
                    "healthy": np.mean(leave > t),
                    "ill": np.mean(is_ill),
                    "dead": np.mean(death <= t),
                }[st]
                a, b = pars["01"]

                def g(u):
                    h = (b / a) * (u / a) ** (b - 1)
                    return (
                        np.exp(
                            -_H(pars["01"], u)
                            - _H(pars["02"], u)
                            - _H(pars["12"], t)
                            + _H(pars["12"], u)
                        )
                        * h
                    )

                p01 = quad(g, 0, t, epsabs=1e-13, epsrel=1e-13)[0]
                p00 = np.exp(-_H(pars["01"], t) - _H(pars["02"], t))
                ref = {"healthy": p00, "ill": p01, "dead": 1 - p00 - p01}[st]
                print(
                    f"  {st:8s} at {t:4g}: {tr[j]:.6f}  quad {ref:.6f}  "
                    f"simulated {sim:.6f}"
                )
            else:
                print(f"  {st:8s} at {t:4g} | ill at {s:g}: {tr[j]:.6f}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    if what == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    elif what == "truth":
        show_truth()
    else:
        summarise()
