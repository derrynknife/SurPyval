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
