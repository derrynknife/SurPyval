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
  ``ovarian``); ``reference_r.R`` checks each against R's own copy, and
  the PBC trial's baseline rows from ``load_pbc2``;
* small seeded synthetic sets aimed at one feature each: ties between
  event and censoring times, left truncation, interval censoring,
  competing risks with ties, a prediction matrix for the Brier score and
  AUC, and continuous data for the additive hazards model.

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
    }


def main():
    fixtures = build()
    text = json.dumps(fixtures, indent=1, allow_nan=False, sort_keys=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
