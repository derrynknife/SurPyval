"""Write the shared fixtures for the stored reference results (#379).

Every reference value under ``surpyval/tests/reference/data`` is computed
on one of the data sets written here, by R (``reference_r.R``) and by the
Python packages (``reference_python.py``) alike, and the tests fit
SurPyval to the same rows. The data are stored, not regenerated at test
time, so a change in numpy's random streams can never move a fixture
under a stored reference.

Two kinds of fixture:

* classic data sets that SurPyval ships and R's ``survival`` has too
  (``lung``, ``heart``) or that are small enough to carry here (``aml``,
  ``ovarian``, ``veteran``); ``reference_r.R`` checks each against R's
  own copy, and the PBC trial's baseline rows from ``load_pbc2``;
* small seeded synthetic sets aimed at one feature each: ties between
  event and censoring times, left truncation, interval censoring,
  competing risks with ties, a prediction matrix for the Brier score and
  AUC, continuous data for the additive hazards model, and a deliberate
  violation of proportional hazards for the Cox diagnostics.

Run from the repository root::

    python scripts/reference/make_fixtures.py

Re-running reproduces ``fixtures.json`` byte for byte.
"""

import json
import math
from pathlib import Path

import numpy as np

from surpyval.datasets import (
    load_heart_transplants,
    load_kidney,
    load_lung,
    load_mettas_and_zhao,
    load_pbc2,
)

OUT = (
    Path(__file__).resolve().parents[2]
    / "surpyval"
    / "tests"
    / "reference"
    / "data"
    / "fixtures.json"
)

# R's survival::aml and survival::ovarian, copied by hand;
# reference_r.R stops if they differ from R's copies.
AML = {
    "time": [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161]
    + [5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45],
    "status": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0]
    + [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    "maintained": [1] * 11 + [0] * 12,
}

OVARIAN = {
    "futime": [59, 115, 156, 421, 431, 448, 464, 475, 477, 563, 638, 744]
    + [769, 770, 803, 855, 1040, 1106, 1129, 1206, 1227, 268, 329, 353]
    + [365, 377],
    "fustat": [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    + [0, 0, 1, 1, 1, 1, 0],
    "age": [72.3315, 74.4932, 66.4658, 53.3644, 50.3397, 56.4301, 56.937]
    + [59.8548, 64.1753, 55.1781, 56.7562, 50.1096, 59.6301, 57.0521]
    + [39.2712, 43.1233, 38.8932, 44.6, 53.9068, 44.2055, 59.589]
    + [74.5041, 43.137, 63.2192, 64.4247, 58.3096],
    "resid_ds": [2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1]
    + [2, 1, 2, 2, 1, 2, 1],
    "rx": [1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2]
    + [1, 1, 2, 2, 2],
    "ecog_ps": [1, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1]
    + [1, 2, 2, 1, 2, 1, 1],
}

# R's survival::veteran (the Veterans' Administration lung cancer
# trial), copied by hand; celltype is R's factor code (1 squamous,
# 2 smallcell, 3 adeno, 4 large). reference_r.R stops if it differs.
VETERAN = {
    "trt": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    + [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    + [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    + [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "celltype": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    + [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    + [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    + [4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
    + [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    + [3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    "time": [72, 411, 228, 126, 118, 10, 82, 110, 314, 100, 42, 8, 144, 25]
    + [11, 30, 384, 4, 54, 13, 123, 97, 153, 59, 117, 16, 151, 22, 56, 21, 18]
    + [139, 20, 31, 52, 287, 18, 51, 122, 27, 54, 7, 63, 392, 10, 8, 92, 35]
    + [117, 132, 12, 162, 3, 95, 177, 162, 216, 553, 278, 12, 260, 200, 156]
    + [182, 143, 105, 103, 250, 100, 999, 112, 87, 231, 242, 991, 111, 1, 587]
    + [389, 33, 25, 357, 467, 201, 1, 30, 44, 283, 15, 25, 103, 21, 13, 87, 2]
    + [20, 7, 24, 99, 8, 99, 61, 25, 95, 80, 51, 29, 24, 18, 83, 31, 51, 90]
    + [52, 73, 8, 36, 48, 7, 140, 186, 84, 19, 45, 80, 52, 164, 19, 53, 15]
    + [43, 340, 133, 111, 231, 378, 49],
    "status": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0]
    + [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    + [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "karno": [60, 70, 60, 60, 70, 20, 40, 80, 50, 70, 60, 40, 30, 80, 70, 60]
    + [60, 40, 80, 60, 40, 60, 60, 30, 80, 30, 50, 60, 80, 40, 20, 80, 30, 75]
    + [70, 60, 30, 60, 80, 60, 70, 50, 50, 40, 40, 20, 70, 40, 80, 80, 50, 80]
    + [30, 80, 50, 80, 50, 70, 60, 40, 80, 80, 70, 90, 90, 80, 80, 70, 60, 90]
    + [80, 80, 50, 50, 70, 70, 20, 60, 90, 30, 20, 70, 90, 80, 50, 70, 60, 90]
    + [50, 30, 70, 20, 30, 60, 40, 30, 20, 60, 70, 80, 85, 70, 70, 70, 50, 30]
    + [40, 40, 40, 99, 80, 60, 60, 60, 60, 50, 70, 10, 40, 70, 90, 80, 50, 40]
    + [40, 60, 70, 30, 60, 30, 60, 80, 75, 60, 70, 80, 30],
    "diagtime": [7, 5, 3, 9, 11, 5, 10, 29, 18, 6, 4, 58, 4, 9, 11, 3, 9, 2]
    + [4, 4, 3, 5, 14, 2, 3, 4, 12, 4, 12, 2, 15, 2, 5, 3, 2, 25, 4, 1, 28, 8]
    + [1, 7, 11, 4, 23, 19, 10, 6, 2, 5, 4, 5, 3, 4, 16, 5, 15, 2, 12, 12, 5]
    + [12, 2, 2, 8, 11, 5, 8, 13, 12, 6, 3, 8, 1, 7, 3, 21, 3, 2, 6, 36, 13]
    + [2, 28, 7, 11, 13, 2, 13, 2, 22, 4, 2, 2, 36, 9, 11, 8, 3, 2, 4, 2, 2]
    + [1, 17, 87, 8, 2, 5, 3, 3, 5, 22, 3, 3, 5, 8, 4, 4, 3, 3, 4, 10, 3, 4]
    + [4, 15, 4, 12, 5, 11, 10, 1, 5, 18, 4, 3],
    "age": [69, 64, 38, 63, 65, 49, 69, 68, 43, 70, 81, 63, 63, 52, 48, 61]
    + [42, 35, 63, 56, 55, 67, 63, 65, 46, 53, 69, 68, 43, 55, 42, 64, 65, 65]
    + [55, 66, 60, 67, 53, 62, 67, 72, 48, 68, 67, 61, 60, 62, 38, 50, 63, 64]
    + [43, 34, 66, 62, 52, 47, 63, 68, 45, 41, 66, 62, 60, 66, 38, 53, 37, 54]
    + [60, 48, 52, 70, 50, 62, 65, 58, 62, 64, 63, 58, 64, 52, 35, 63, 70, 51]
    + [40, 69, 36, 71, 62, 60, 44, 54, 66, 49, 72, 68, 62, 71, 70, 61, 71, 59]
    + [67, 60, 69, 57, 39, 62, 50, 43, 70, 66, 61, 81, 58, 63, 60, 62, 42, 69]
    + [63, 45, 68, 39, 66, 63, 49, 64, 65, 64, 67, 65, 37],
    "prior": [0, 10, 0, 10, 10, 0, 10, 0, 0, 0, 0, 10, 0, 10, 10, 0, 0, 0, 10]
    + [0, 0, 0, 10, 0, 0, 10, 0, 0, 10, 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0]
    + [0, 0, 0, 10, 10, 0, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 0, 10, 0, 10, 0]
    + [0, 0, 0, 0, 10, 10, 10, 0, 0, 10, 0, 10, 0, 10, 0, 0, 0, 0, 0, 0, 10]
    + [0, 0, 10, 0, 10, 0, 10, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10]
    + [0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10, 10]
    + [0, 0, 10, 10, 0, 0, 10, 0, 0],
}


def _native(values):
    """A list of plain floats/ints with ``None`` for NaN or infinity, so
    the file is strict JSON."""
    out = []
    for v in np.asarray(values).tolist():
        if isinstance(v, float) and not math.isfinite(v):
            out.append(None)
        else:
            out.append(v)
    return out


def lung():
    df = load_lung()
    return {
        "source": "surpyval.datasets.load_lung; R survival::lung with "
        "status recoded to SurPyval's flag (c = 2 - status)",
        "columns": {
            "time": _native(df["time"]),
            # load_lung's status is 1 = death (#509)
            "c": _native(1 - df["status"].astype(int)),
            "age": _native(df["age"]),
            "sex": _native(df["sex"]),
            "ph_ecog": _native(df["ph.ecog"]),
            "inst": _native(df["inst"]),
        },
    }


def kidney():
    df = load_kidney()
    return {
        "source": "surpyval.datasets.load_kidney; R survival::kidney "
        "(female = sex == 2, c = 1 - status)",
        "columns": {
            "time": _native(df["time"]),
            "c": _native(1 - df["status"]),
            "id": _native(df["id"]),
            "age": _native(df["age"]),
            "female": _native((df["sex"] == 2).astype(int)),
        },
    }


def heart():
    df = load_heart_transplants()
    return {
        "source": "surpyval.datasets.load_heart_transplants; R "
        "survival::heart (start-stop form, c = 1 - event)",
        "columns": {
            "start": _native(df["start"]),
            "stop": _native(df["stop"]),
            "c": _native(1 - df["event"]),
            "age": _native(df["age"]),
            "year": _native(df["year"]),
            "surgery": _native(df["surgery"]),
            "transplant": _native(df["transplant"]),
            "id": _native(df["id"]),
        },
    }


def pbc():
    df = load_pbc2().groupby("id", sort=True).first()
    cause = df["status"].map({"alive": 0, "transplanted": 1, "dead": 2})
    return {
        "source": "surpyval.datasets.load_pbc2, first row per patient (the "
        "312 trial patients at baseline); cause 0 censored, 1 transplant, "
        "2 death",
        "columns": {
            "years": _native(df["years"]),
            "cause": _native(cause.astype(int)),
            "drug": _native((df["drug"] == "D-penicil").astype(int)),
            "age": _native(df["age"]),
            "female": _native((df["sex"] == "female").astype(int)),
            "log_bili": _native(np.log(df["serBilir"])),
        },
    }


def mettas_zhao():
    df = load_mettas_and_zhao()
    return {
        "source": "surpyval.datasets.load_mettas_and_zhao (recurrent "
        "events on six systems; c = 1 marks the end of observation)",
        "columns": {
            "x": _native(df["x"]),
            "i": _native(df["i"]),
            "c": _native(df["c"]),
        },
    }


def ties():
    """Integer times from 1 to 20 with many ties, including events tied
    with censorings, a binary and a continuous covariate."""
    rng = np.random.default_rng(3791)
    n = 40
    z1 = rng.binomial(1, 0.5, n)
    z2 = np.round(rng.normal(0, 1, n), 1)
    t = rng.exponential(1 / (0.1 * np.exp(0.6 * z1 + 0.4 * z2)))
    u = rng.uniform(0, 20, n)
    x = np.ceil(np.minimum(t, u)).astype(int)
    c = (u < t).astype(int)
    return {
        "source": "seeded synthetic, make_fixtures.ties (rng 3791)",
        "columns": {
            "x": _native(x),
            "c": _native(c),
            "z1": _native(z1),
            "z2": _native(z2),
        },
    }


def left_truncation():
    """Weibull(scale 5, shape 1.5) lifetimes observed only after a late
    entry ``tl``, then right censored."""
    rng = np.random.default_rng(3792)
    n = 60
    tl = np.round(rng.uniform(0, 3, n), 1)
    z = rng.binomial(1, 0.5, n)
    scale = 5 * np.exp(-0.5 * z)
    t = np.empty(n)
    for k in range(n):
        # Draw until the lifetime passes the entry time (truncation).
        while True:
            draw = scale[k] * rng.weibull(1.5)
            if draw > tl[k] + 0.05:
                t[k] = draw
                break
    cens = tl + rng.uniform(1, 10, n)
    x = np.round(np.minimum(t, cens), 2)
    c = (cens < t).astype(int)
    return {
        "source": "seeded synthetic, make_fixtures.left_truncation "
        "(rng 3792)",
        "columns": {
            "tl": _native(tl),
            "x": _native(x),
            "c": _native(c),
            "z": _native(z),
        },
    }


def interval():
    """Weibull(scale 12 exp(-0.5 z), shape 2) lifetimes seen at
    inspections every 2 units (jittered) up to 16: interval censored,
    left censored before the first inspection, right censored after the
    last, and a few exact."""
    rng = np.random.default_rng(3793)
    n = 60
    z = rng.binomial(1, 0.5, n)
    t = 12 * np.exp(-0.5 * z) * rng.weibull(2, n)
    left = []
    right = []
    for k in range(n):
        if rng.uniform() < 0.1:
            # Observed exactly.
            v = round(float(t[k]), 2)
            left.append(v)
            right.append(v)
            continue
        grid = np.round(
            np.arange(2, 17, 2) + rng.uniform(-0.5, 0.5, 8), 1
        ).tolist()
        before = [g for g in grid if g < t[k]]
        after = [g for g in grid if g >= t[k]]
        left.append(before[-1] if before else 0.0)
        right.append(after[0] if after else None)
    return {
        "source": "seeded synthetic, make_fixtures.interval (rng 3793); "
        "left == right is exact, left == 0 is left censored at right, "
        "right null is right censored at left",
        "columns": {
            "left": left,
            "right": right,
            "z": _native(z),
        },
    }


def competing():
    """Two causes and censoring on integer times, so causes tie with
    each other and with censorings; a binary group and a continuous
    covariate."""
    rng = np.random.default_rng(3794)
    n = 80
    g = rng.binomial(1, 0.5, n)
    z = np.round(rng.normal(0, 1, n), 1)
    t1 = rng.exponential(1 / (0.08 * np.exp(0.6 * g + 0.3 * z)))
    t2 = rng.exponential(1 / 0.05, n)
    u = rng.uniform(0, 25, n)
    t = np.minimum(t1, t2)
    x = np.ceil(np.minimum(t, u)).astype(int)
    cause = np.where(u < t, 0, np.where(t1 < t2, 1, 2))
    return {
        "source": "seeded synthetic, make_fixtures.competing (rng 3794); "
        "cause 0 is censored",
        "columns": {
            "x": _native(x),
            "cause": _native(cause),
            "group": _native(g),
            "z": _native(z),
        },
    }


def prediction(name, seed, digits):
    """An evaluation set and a matrix of predicted survival at four
    horizons, for the Brier score and the time-dependent AUC. With
    ``digits=0`` times are whole numbers, so events tie with censorings
    (where SurPyval follows pec, not scikit-survival, #365)."""
    rng = np.random.default_rng(seed)
    n = 60
    z = np.round(rng.normal(0, 1, n), 2)
    t = 10 * np.exp(-0.5 * z) * rng.weibull(1.3, n)
    u = rng.uniform(0, 20, n)
    x = np.minimum(t, u)
    if digits == 0:
        x = np.ceil(x)
    else:
        x = np.round(x, digits)
    c = (u < t).astype(int)
    times = [3.0, 6.0, 9.0, 12.0]
    # A deliberately imperfect model: right shape, attenuated effect.
    S = np.exp(-((np.outer(np.exp(0.4 * z), times) / 10) ** 1.3))
    return {
        "source": "seeded synthetic, make_fixtures.prediction "
        "(rng {}); predictions S(t | z) = exp(-(exp(0.4 z) t / 10)^1.3)"
        " at 'times'".format(seed),
        "columns": {"x": _native(x), "c": _native(c), "z": _native(z)},
        "times": times,
        "survival": [_native(row) for row in np.round(S, 6)],
    }


def additive():
    """Continuous (untied) times from an additive hazard
    0.05 + 0.04 z1 + 0.02 z2, for the Lin-Ying model."""
    rng = np.random.default_rng(3796)
    n = 100
    z1 = rng.binomial(1, 0.5, n)
    z2 = np.round(rng.uniform(0, 2, n), 2)
    t = rng.exponential(1 / (0.05 + 0.04 * z1 + 0.02 * z2))
    u = rng.uniform(0, 30, n)
    x = np.round(np.minimum(t, u), 4)
    c = (u < t).astype(int)
    return {
        "source": "seeded synthetic, make_fixtures.additive (rng 3796)",
        "columns": {
            "x": _native(x),
            "c": _native(c),
            "z1": _native(z1),
            "z2": _native(z2),
        },
    }


def ph_violation():
    """A deliberate violation of proportional hazards, for the Cox
    diagnostics (#868): a binary ``z1`` whose hazard ratio reverses at
    t = 1 (e^1.3 before, e^-1.3 after), a continuous ``z2`` with a
    proportional, linear effect, and a continuous ``z3`` that acts on the
    log hazard through its square (a wrong functional form for a linear
    fit). A third of the units enter late (``tl``); times are rounded up
    to quarters, so there are ties."""
    rng = np.random.default_rng(3797)
    n = 320
    z1 = rng.binomial(1, 0.5, n)
    z2 = np.round(rng.normal(0, 1, n), 2)
    z3 = np.round(rng.uniform(-2, 2, n), 2)
    rate = 0.2 * np.exp(0.5 * z2 + 0.4 * z3**2)
    early, late = rate * np.exp(1.3 * z1), rate * np.exp(-1.3 * z1)
    e = rng.exponential(size=n)
    t = np.where(e < early, e / early, 1 + (e - early) / late)
    tl = np.where(rng.uniform(size=n) < 1 / 3, rng.uniform(0, 1, n), 0.0)
    tl = np.round(tl, 2)
    cens = rng.uniform(0, 8, n)
    seen = np.minimum(t, cens)
    keep = seen > tl  # delayed entry: units gone before entry are unseen
    x = np.ceil(seen * 4) / 4
    c = (cens < t).astype(int)
    return {
        "source": "seeded synthetic, make_fixtures.ph_violation (rng 3797)",
        "columns": {
            "tl": _native(tl[keep]),
            "x": _native(x[keep]),
            "c": _native(c[keep]),
            "z1": _native(z1[keep]),
            "z2": _native(z2[keep]),
            "z3": _native(z3[keep]),
        },
    }


def build():
    return {
        "_about": "Shared fixtures for surpyval/tests/reference; written "
        "by scripts/reference/make_fixtures.py. Null is a missing value "
        "(or an unbounded interval end).",
        "aml": {
            "source": "R survival::aml (maintained = 1 for x == "
            "'Maintained'); status 1 is a death",
            "columns": AML,
        },
        "ovarian": {
            "source": "R survival::ovarian; fustat 1 is a death",
            "columns": OVARIAN,
        },
        "lung": lung(),
        "heart": heart(),
        "kidney": kidney(),
        "pbc": pbc(),
        "mettas_zhao": mettas_zhao(),
        "ties": ties(),
        "left_truncation": left_truncation(),
        "interval": interval(),
        "competing": competing(),
        "prediction_ties": prediction("prediction_ties", 3795, 0),
        "prediction_continuous": prediction("prediction_continuous", 3795, 3),
        "additive": additive(),
        "veteran": {
            "source": "R survival::veteran; status 1 is a death; "
            "celltype is R's factor code (1 squamous, 2 smallcell, "
            "3 adeno, 4 large)",
            "columns": VETERAN,
        },
        "ph_violation": ph_violation(),
    }


def main():
    fixtures = build()
    text = json.dumps(fixtures, indent=1, allow_nan=False, sort_keys=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
