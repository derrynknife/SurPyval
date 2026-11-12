"""The conformance registry's fixtures (see ``registry.py``).

Tiny and built without a random number generator -- quantiles of a
known distribution, in a fixed order -- so a failure reproduces from
the case alone.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Fixtures (deterministic: quantiles of a known model, in a fixed order)
# ---------------------------------------------------------------------------
def quantiles(size, scale=10.0, shape=2.0):
    """``size`` Weibull(scale, shape) plotting-position quantiles."""
    u = (np.arange(1, size + 1) - 0.3) / (size + 0.4)
    return scale * (-np.log1p(-u)) ** (1.0 / shape)


def scramble(size):
    """A fixed, thoroughly unsorted order of ``range(size)``."""
    stride = next(s for s in (5, 7, 11, 13) if np.gcd(s, size) == 1)
    return (np.arange(size) * stride + 3) % size


def uni_data():
    """12 rows: exact and right-censored values, some with counts > 1."""
    x = np.round(quantiles(12), 3)
    c = np.zeros(12, int)
    c[[4, 9]] = 1
    n = np.ones(12, int)
    n[[2, 7]] = [2, 3]
    return {"x": x, "c": c, "n": n}


def uni_exact_data():
    """``uni_data`` with every value exactly observed: the Uniform's MLE
    refuses censored values (#460)."""
    data = uni_data()
    data["c"] = np.zeros_like(data["c"])
    return data


def xcnt_data():
    """Every censoring type, with left truncation: the full data model."""
    x = np.array(
        [
            [1.0, 1.0],
            [2.0, 3.5],
            [3.0, 3.0],
            [4.0, 7.0],
            [5.5, 5.5],
            [6.0, 6.0],
            [7.5, 12.0],
            [9.0, 9.0],
            [11.0, 11.0],
            [14.0, 14.0],
        ]
    )
    c = np.array([0, 2, 0, 2, 1, -1, 2, 0, 0, 1])
    n = np.array([1, 1, 2, 1, 1, 1, 1, 2, 1, 1])
    tl = np.array([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    return {"x": x, "c": c, "n": n, "tl": tl}


def offset_data():
    d = uni_data()
    d["x"] = d["x"] + 5.0
    return d


def lfp_data():
    """A population a third of which is still running long after the rest
    have failed: a limited failure population."""
    d = uni_data()
    return {
        "x": np.r_[d["x"], 40.0, 45.0],
        "c": np.r_[d["c"], 1, 1],
        "n": np.r_[d["n"], 4, 4],
    }


def lfp_count_data():
    """Monthly return counts of 1000 units, 3% of them defective (Weibull
    scale 2 months, shape 0.7) and the rest wearing out (scale 120 months,
    shape 3), observed for 24 months: interval-censored counts, as
    warranty data come. The default start of a Weibull with ``lfp=True``
    ran ``p`` to 1 and reported a verified maximum 0.84 below the one at
    ``p = 0.032`` (#579)."""
    months = np.array([1, 2, 3, 4, 5, 6, 8, 18, 20, 22, 24.0])
    return {
        "x": np.r_[np.column_stack([months - 1, months]), [[24.0, 24.0]]],
        "c": np.r_[np.full(months.size, 2), 1],
        "n": np.array([7, 5, 4, 2, 1, 2, 1, 1, 2, 1, 1, 973]),
    }


def zi_data():
    """Two dead-on-arrival zeros in front of the usual sample."""
    d = uni_data()
    return {
        "x": np.r_[0.0, d["x"]],
        "c": np.r_[0, d["c"]],
        "n": np.r_[2, d["n"]],
    }


def mixture_data():
    x = np.r_[np.linspace(2.0, 6.0, 10), np.linspace(20.0, 40.0, 10)]
    n = np.ones(20, int)
    n[[2, 15]] = 2
    return {"x": x, "c": np.zeros(20, int), "n": n}


def discrete_data(start=0):
    x = np.array([0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 8]) + start
    c = np.zeros(12, int)
    c[[5, 10]] = 1
    n = np.ones(12, int)
    n[[2, 7]] = [2, 3]
    return {"x": x, "c": c, "n": n}


def beta_geometric_data():
    """40 draws from BetaGeometric(3, 5), right censored above 8: more
    dispersed than a Geometric, so the fit has an interior maximum (a, b =
    1.72, 3.01). ``discrete_data`` is under-dispersed for it: its fit runs
    to the geometric limit, where the likelihood has no finite maximum
    (#392), and no bound can be tested there."""
    return {
        "x": np.array([1, 2, 3, 4, 6, 7, 8]),
        "c": np.array([0, 0, 0, 0, 0, 0, 1]),
        "n": np.array([14, 9, 4, 3, 4, 1, 5]),
    }


def binary_data():
    return {
        "x": np.array([0, 1, 1, 0, 1, 1, 0]),
        "n": np.array([1, 2, 1, 1, 1, 3, 1]),
    }


def exact_event_data():
    return {
        "x": np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
        "c": np.array([1, 1, -1, -1, -1]),
        "n": np.array([1, 2, 1, 1, 1]),
    }


def unit_interval_data():
    x = np.array([0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8])
    n = np.ones(10, int)
    n[[2, 6]] = 2
    return {"x": x, "c": np.zeros(10, int), "n": n}


N_REG = 30


def reg_data():
    """30 rows, a binary and a continuous covariate, 5 right censored."""
    z0 = np.tile([0.0, 1.0], N_REG // 2)
    z1 = np.round(np.linspace(-1.0, 1.0, N_REG), 3)
    life = quantiles(N_REG, 1.0, 2.0)[scramble(N_REG)]
    x = np.round(10.0 * np.exp(-0.5 * z0 + 0.3 * z1) * life, 3)
    c = np.zeros(N_REG, int)
    c[::7] = 1
    n = np.ones(N_REG, int)
    n[[3, 11]] = 2
    return {"x": x, "Z": np.column_stack([z0, z1]), "c": c, "n": n}


N_CURE = 60


def cure_data():
    """60 rows for the mixture-cure regressions: the regression
    fixture's covariates and lives, where 2 units in 5 with ``z0 = 0``
    and 3 in 5 with ``z0 = 1`` never fail and are still running at 60,
    far past the lives (a plateau)."""
    z0 = np.tile([0.0, 1.0], N_CURE // 2)
    z1 = np.round(np.linspace(-1.0, 1.0, N_CURE), 3)
    life = quantiles(N_CURE, 1.0, 2.0)[scramble(N_CURE)]
    x = np.round(10.0 * np.exp(-0.5 * z0 + 0.3 * z1) * life, 3)
    pair = np.arange(N_CURE) // 2
    cured = np.where(z0 == 0, pair % 5 < 2, pair % 5 < 3)
    x = np.where(cured, 60.0, x)
    n = np.ones(N_CURE, int)
    n[[3, 11]] = 2
    return {
        "x": x,
        "Z": np.column_stack([z0, z1]),
        "c": cured.astype(int),
        "n": n,
    }


N_EQUIVALENCE = 40


def equivalence_data():
    """40 units on two clocks for the equivalent-scale fits: ``x`` (hours)
    and ``x2`` (starts), each unit at its own constant start rate (0,
    0.1, 0.3 or 1 per hour), with one start worth 2 hours; every fifth
    unit still running at half its life, and two rows counted twice."""
    m = N_EQUIVALENCE
    rate = np.tile([0.0, 0.1, 0.3, 1.0], m // 4)
    life = quantiles(m, 10.0, 2.0)[scramble(m)]
    c = (np.arange(m) % 5 == 4).astype(int)
    life = np.where(c == 1, life / 2.0, life)
    x = np.round(life / (1.0 + 2.0 * rate), 3)
    n = np.ones(m, int)
    n[[3, 11]] = 2
    return {"x": x, "x2": np.round(rate * x, 3), "c": c, "n": n}


def zi_reg_data():
    """60 rows for the zero-inflated regressions: the regression
    fixture's covariates and lives, where 1 unit in 10 with ``z0 = 0``
    and 2 in 10 with ``z0 = 1`` fail at time 0 (dead on arrival), and
    every seventh of the rest is right censored."""
    z0 = np.tile([0.0, 1.0], N_CURE // 2)
    z1 = np.round(np.linspace(-1.0, 1.0, N_CURE), 3)
    life = quantiles(N_CURE, 1.0, 2.0)[scramble(N_CURE)]
    x = np.round(10.0 * np.exp(-0.5 * z0 + 0.3 * z1) * life, 3)
    pair = np.arange(N_CURE) // 2
    doa = np.where(z0 == 0, pair % 10 == 0, pair % 5 == 0)
    x = np.where(doa, 0.0, x)
    c = np.zeros(N_CURE, int)
    c[1::7] = 1
    c[doa] = 0
    n = np.ones(N_CURE, int)
    n[[3, 11]] = 2
    return {"x": x, "Z": np.column_stack([z0, z1]), "c": c, "n": n}


def ah_data():
    """The regression fixture with its covariates turned to add hazard,
    for the additive hazards cases (#828): ``1 - z1`` for ``z1``, so both
    covariates are non-negative and shorten life. On ``reg_data`` a
    protective row's ``h0(t) + beta'Z`` is negative near 0, where the
    baselines' hazards start at 0, and every AH fit but the Exponential's
    and the Gumbel's ends on the boundary of its support (``H = 0`` at an
    early observation), with a warning; here each is an ordinary maximum
    inside it."""
    d = reg_data()
    d["Z"] = np.column_stack([d["Z"][:, 0], 1.0 - d["Z"][:, 1]])
    return d


def stress_data(columns=1):
    """Accelerated life data: life falls with stress (1, 2 or 3)."""
    d = reg_data()
    s = np.repeat([1.0, 2.0, 3.0], N_REG // 3)
    life = quantiles(N_REG, 1.0, 2.0)[scramble(N_REG)]
    d["x"] = np.round(30.0 * s**-1.2 * life, 3)
    stress = s[:, None]
    if columns == 2:
        stress = np.column_stack([s, np.tile([1.0, 2.0], N_REG // 2)])
    d["Z"] = stress
    return d


def grouped_reg_data():
    d = reg_data()
    d["groups"] = np.repeat(np.arange(6), N_REG // 6)
    return d


def stratified_reg_data():
    d = reg_data()
    d["strata"] = np.repeat([0, 1], N_REG // 2)
    return d


X_UNI = np.array([0.5, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 17.0, 25.0])
X_DISC = np.arange(0.0, 11.0)
X_REG = np.array([0.5, 2.0, 4.0, 6.0, 8.0, 11.0, 15.0, 22.0])
Z_REG = np.array(
    [
        [0.0, -0.8],
        [1.0, -0.5],
        [0.0, 0.0],
        [1.0, 0.3],
        [0.0, 0.9],
        [1.0, 1.0],
        [0.0, 0.4],
        [1.0, -0.2],
    ]
)
#: The query rows of ``ah_data``: ``Z_REG`` turned as its covariates are.
Z_AH = np.column_stack([Z_REG[:, 0], 1.0 - Z_REG[:, 1]])
X_STRESS = np.array([1.0, 3.0, 6.0, 10.0, 15.0, 22.0])
Z_STRESS = np.array([[1.0], [1.5], [2.0], [2.5], [3.0], [1.2]])
Z_STRESS2 = np.array(
    [[1.0, 1.0], [1.5, 2.0], [2.0, 1.0], [2.5, 2.0], [3.0, 1.0], [1.2, 2.0]]
)


def cr_data(with_Z=False):
    """24 rows, causes "a" and "b" and censored (``None``) rows."""
    size = 24
    x = np.round(quantiles(size, 10.0, 1.5), 3)
    e = np.array(["a", "b", "a", None] * (size // 4), dtype=object)
    e = e[scramble(size)]
    n = np.ones(size, int)
    n[[2, 5]] = 2
    d = {"x": x, "e": e, "n": n}
    if with_Z:
        d["Z"] = np.tile([0.0, 1.0, 1.0], size // 3)[scramble(size)][:, None]
    return d


X_CR = np.array([0.5, 2.0, 4.0, 7.0, 10.0, 14.0, 20.0, 30.0])
Z_CR = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.5], [1.0]])


def recurrent_data(with_Z=False, with_e=False):
    """Three repairable items, each ending with a censoring row."""
    x = np.array(
        [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60, 5, 18, 30, 50, 60.0]
    )
    i = np.array([1] * 6 + [2] * 5 + [3] * 5)
    c = np.zeros(16, int)
    c[[5, 10, 15]] = 1
    d = {"x": x, "i": i, "c": c, "n": np.ones(16, int)}
    if with_Z:
        d["Z"] = np.array([0.0] * 6 + [1.0] * 5 + [0.5] * 5)[:, None]
    if with_e:
        d["e"] = np.array(
            ["a", "b", "a", "b", "a", None, "a", "b", "a", "b", None]
            + ["a", "b", "a", "a", None],
            dtype=object,
        )
    return d


X_REC = np.array([1.0, 5.0, 12.0, 25.0, 40.0, 55.0])
Z_REC = np.array([[0.0], [1.0], [0.5], [0.0], [1.0], [0.2]])


def path_data():
    """6 units, 10 readings each, degrading linearly towards 150."""
    x = np.tile(np.arange(100.0, 1100.0, 100.0), 6)
    i = np.repeat(np.arange(6), 10)
    slopes = np.repeat([0.31, 0.28, 0.44, 0.37, 0.33, 0.40], 10)
    start = np.repeat([8.0, 12.0, 10.0, 9.0, 13.0, 11.0], 10)
    y = start + slopes * x + 1.5 * np.sin(np.arange(60) * 1.7)
    return {"x": x, "y": y, "i": i}


def process_data():
    """5 units, 11 readings each, of a process with positive increments."""
    t = np.tile(np.arange(0.0, 110.0, 10.0), 5)
    i = np.repeat(np.arange(5), 11)
    steps = 5.0 + 3.0 * np.sin(np.arange(50.0)).reshape(5, 10)
    y = np.hstack([np.r_[0.0, np.cumsum(s)] for s in steps])
    return {"x": t, "y": y, "i": i}


def destructive_data():
    x = np.repeat([10.0, 20.0, 30.0, 40.0], 5)
    y = np.exp(4.0 - 0.02 * x + 0.1 * np.sin(np.arange(20.0)))
    return {"x": x, "y": y}


def joint_data():
    """40 items of a joint model (#891): a straight-line marker read every
    60 days with a level and rate of each item's own, a 0/1 covariate,
    and a Weibull PH lifetime in the line's current value, censored at
    730; some counted twice. Each item's line and lifetime come from
    quantiles in scrambled orders, its readings' noise from a sine."""
    from scipy.special import ndtri

    size = 40
    u = (scramble(size) + 0.5) / size
    # an order unrelated to u's (a reversed scramble is affine in it)
    v = np.argsort(np.argsort(np.sin(2.7 * np.arange(size) + 1.0))) + 0.5
    v = v / size
    level = 3.0 + 0.6 * ndtri(u)
    rate = (1.0 + 0.5 * ndtri(v)) / 365
    z = (np.arange(size) % 2).astype(float)
    # the lifetime: H(T) = E on a fine grid of ages
    order = np.argsort(np.argsort(np.sin(1.9 * np.arange(size) + 0.3)))
    E = -np.log1p(-(order + 0.5) / size)
    t = np.linspace(0.0, 4000.0, 8001)
    x, c, markers = [], [], []
    for k in range(size):
        h = (
            (2 / 800)
            * (t / 800)
            * np.exp(0.9 * (level[k] + rate[k] * t - 4.0) + 0.4 * z[k])
        )
        H = np.r_[0.0, np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(t))]
        life = float(np.interp(E[k], H, t))
        end = round(min(life, 730.0), 2)
        ages = np.arange(0.0, end, 60.0)
        y = level[k] + rate[k] * ages + 0.5 * np.sin(1.3 * k + 0.7 * ages)
        x.append(end)
        c.append(0 if life < 730.0 else 1)
        markers.append(dict(zip(ages.tolist(), np.round(y, 3).tolist())))
    n = np.ones(size, int)
    n[[3, 17, 30]] = 2
    return {
        "x": np.array(x),
        "c": np.array(c),
        "n": n,
        "Z": z[:, None],
        "markers": markers,
    }


#: The joint model's query: an item read four times, still running at
#: 400 (``JOINT_ITEM``), at ages before and after that.
X_JOINT = np.array([100.0, 300.0, 450.0, 600.0, 900.0, 1200.0])
JOINT_ITEM = {
    "markers": {0.0: 3.1, 120.0: 3.4, 240.0: 3.6, 360.0: 3.9},
    "given": 400.0,
}


X_PATH = np.array([150.0, 250.0, 320.0, 400.0, 480.0, 600.0])
X_PROC = np.array([50.0, 120.0, 160.0, 200.0, 260.0, 350.0])
X_DESTR = np.array([5.0, 20.0, 40.0, 60.0, 80.0, 120.0])


def copula_data():
    u = (np.arange(1, 21) - 0.3) / 20.4
    swap = np.arange(20) ^ 1  # neighbours swapped: positively dependent
    x1 = 10.0 * (-np.log1p(-u)) ** 0.5
    x2 = 5.0 * (-np.log1p(-u[swap])) ** (1 / 1.4)
    n = np.ones(20, int)
    n[[3, 8]] = 2
    return {"x": np.round(np.column_stack([x1, x2]), 3), "n": n}


X_COP = np.array(
    [[2.0, 1.0], [5.0, 2.0], [8.0, 3.0], [10.0, 5.0], [14.0, 7.0], [3.0, 6.0]]
)


# Multi-state: an illness-death model with recovery, as spells
_MS_UNITS = [
    # (spells as (tl, x, state, entered), count, covariate)
    ([(0, 2, "healthy", "ill"), (2, 5, "ill", "healthy"),
      (5, 9, "healthy", "dead")], 1, 0.0),
    ([(0, 3, "healthy", "dead")], 1, 1.0),
    ([(0, 1.5, "healthy", "ill"), (1.5, 4, "ill", "dead")], 1, 1.0),
    ([(0.5, 6, "healthy", None)], 1, 0.0),
    ([(0, 4.5, "healthy", "ill"), (4.5, 10, "ill", None)], 1, 1.0),
    ([(1, 3.5, "ill", "healthy"), (3.5, 8, "healthy", "ill"),
      (8, 12, "ill", "dead")], 1, 0.25),
    ([(0, 7, "healthy", "dead")], 2, 0.0),
    ([(2, 11, "healthy", None)], 1, 1.0),
    ([(0, 2.5, "healthy", "ill"), (2.5, 6.5, "ill", "dead")], 1, 0.5),
    ([(0, 5.5, "healthy", "ill"), (5.5, 7.5, "ill", "healthy"),
      (7.5, 12, "healthy", None)], 1, 1.0),
]  # fmt: skip


def ms_data(with_Z=False):
    """Ten units' spells in an illness-death model with recovery, some
    entering late (one in "ill"), one row counted twice."""
    rows = [
        (tl, x, st, e, n, z)
        for spells, n, z in _MS_UNITS
        for tl, x, st, e in spells
    ]
    d = {
        "x": np.array([r[1] for r in rows], dtype=float),
        "e": np.array([r[3] for r in rows], dtype=object),
        "state": np.array([r[2] for r in rows], dtype=object),
        "tl": np.array([r[0] for r in rows], dtype=float),
        "n": np.array([r[4] for r in rows], dtype=int),
        "transitions": [
            ("healthy", "ill"),
            ("healthy", "dead"),
            ("ill", "healthy"),
            ("ill", "dead"),
        ],
    }
    if with_Z:
        d["Z"] = np.array([[r[5]] for r in rows], dtype=float)
    return d


X_MS = np.array([0.5, 1.75, 3.0, 4.25, 6.0, 7.25, 8.5, 11.5])
# The Cox model's predictions: from 7.5 on, the one unit left at risk in
# "ill" makes the predicted hazards of leaving it add up to more than 1
# for the other covariate rows (a property of the fit to so few, which the
# model warns of).
X_MSR = np.array([0.5, 1.75, 3.0, 4.25, 5.0, 6.0, 6.75, 7.25])
Z_MS = np.array([[0.0], [1.0], [0.5], [0.0], [1.0], [0.0], [1.0], [0.5]])


def joint_frailty_data():
    """60 units of a joint frailty model: a gamma frailty (variance 0.5)
    from quantiles in a scrambled order acts on Weibull recurrent events
    (scale 10, shape 1.2, on the age) and, as ``u ** 1``, on a Weibull
    terminal event (scale 20, shape 1.5), each followed to 15; the
    terminal ages' and the event steps' exponential draws from fixed
    sequences of fractional parts. Covariate ``z`` 0 or 1 (effects 0.5
    and 0.3)."""
    from scipy.stats import gamma

    size = 60
    u = gamma.ppf((scramble(size) + 0.5) / size, 2.0, scale=0.5)
    p_end = ((np.arange(size) * 29 + 7) * 0.4142135623730951) % 1.0
    z = (np.arange(size) % 2).astype(float)
    x, i, c, term, Z = [], [], [], [], []
    for k in range(size):
        death = 20.0 * (-np.log(p_end[k]) / (u[k] * np.exp(0.3 * z[k]))) ** (
            1 / 1.5
        )
        stop = round(min(death, 15.0), 3)
        rate = u[k] * np.exp(0.5 * z[k])
        cum, j = 0.0, 0
        while True:
            q = ((k * 37 + j * 11 + 1) * 0.6180339887498949) % 1.0
            cum += -np.log(q) / rate
            t = round(10.0 * cum ** (1 / 1.2), 3)
            if t >= stop:
                break
            x.append(t), i.append(k), c.append(0), term.append(0)
            Z.append(z[k])
            j += 1
        x.append(stop), i.append(k), c.append(1)
        term.append(int(death <= 15.0)), Z.append(z[k])
    return {
        "x": np.array(x),
        "Z": np.array(Z)[:, None],
        "i": np.array(i),
        "c": np.array(c),
        "terminal": np.array(term),
    }


X_JF = np.array([1.0, 3.0, 6.0, 9.0, 12.0, 14.0])
Z_JF = np.array([[0.0], [1.0], [0.5], [0.0], [1.0], [0.2]])
