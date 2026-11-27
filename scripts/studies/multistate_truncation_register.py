"""A simulated register of deaths: the right-truncated multi-state fit
against the fit that ignores the truncation.

An illness-death model with Weibull hazards of the time since the start
(healthy -> ill: scale 6, shape 1.5; healthy -> dead: 25, 1.2; ill ->
dead: 8, 2), 1,200 units healthy at 0, each with a cut-off uniform on
(2, 12), and only the units dead by their cut-off kept. Seed 710, fixed
before the run. Prints each fit's estimates and their distances from the
truth in standard errors (the figures in the multi-state guide).

    python scripts/studies/multistate_truncation_register.py
"""

import warnings

import numpy as np

from surpyval.multistate import ParametricMultiState

PARS = {
    ("healthy", "ill"): (6.0, 1.5),
    ("healthy", "dead"): (25.0, 1.2),
    ("ill", "dead"): (8.0, 2.0),
}


def register(n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n):
        tau = rng.uniform(2, 12)
        (a1, b1), (a2, b2), (a3, b3) = PARS.values()
        t1, t2 = a1 * rng.weibull(b1), a2 * rng.weibull(b2)
        if t1 < t2:
            td = a3 * ((t1 / a3) ** b3 - np.log(rng.uniform())) ** (1 / b3)
            path = [("healthy", -np.inf, t1, "ill"), ("ill", t1, td, "dead")]
            death = td
        else:
            path = [("healthy", -np.inf, t2, "dead")]
            death = t2
        if death <= tau:
            rows += [(u, *p, tau) for p in path]
    return rows


def main() -> None:
    rows = register(1200, 710)
    i, state, tl, x, e, tau = (np.array(c, dtype=object) for c in zip(*rows))
    x, tl, tau = (np.asarray(v, dtype=float) for v in (x, tl, tau))
    print("units seen:", len(set(i.tolist())), "rows:", len(rows))
    kw = dict(tl=tl, i=i, transitions=list(PARS))
    truth = np.concatenate([PARS[p] for p in PARS])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = {
            "truncated": ParametricMultiState.fit(x, e, state, tr=tau, **kw),
            "ignored": ParametricMultiState.fit(x, e, state, **kw),
        }
    for name, model in fits.items():
        z = (model.params - truth) / model.standard_errors()
        print(name, model.params.round(3), "z", z.round(2), model.maximum)


if __name__ == "__main__":
    main()
