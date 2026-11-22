"""Laplace's approximation against the 7-point grid as the items'
readings thin out: where the two fits of one fleet part.

Two designs, each fleet fitted with ``integration="grid"`` and with
``integration="laplace"``:

- ``linear``: the recovery study's pumps (300 a fleet, a line with its
  own start and rate, two varying parameters), read every 30, 90, 180,
  240 and 365 days for up to two years: at most 25, 9, 5, 4 and 3
  readings a pump. Seeds 1000 to 1004.
- ``quad``: 150 items whose marker is a quadratic with every coefficient
  its own (three varying parameters; the fleets of
  ``surpyval/tests/univariate/joint/test_joint_integration.py``), read
  every 0.1, 0.5 and 1.0 years for up to three: at most 31, 7 and 4
  readings an item. Seeds 1000 to 1002.

The design and seeds were fixed before any fit was run. This first run
(2026-11-21, ``laplace_few_readings.json``) was exploratory: it fixed no
rule for the warning's share, and no fleet had between 9% and 36% of its
pumps read three times or fewer.

The second run, ``STUDY=confirm`` (``laplace_few_readings_confirm.json``),
sets the warning's share, ``_FEW_SHARE`` in
``surpyval/univariate/joint/joint_ph.py``. Its design, and the rule that
turns its results into the constant, were written here and committed
before any of its fits was run:

- the ``linear`` pumps (300 a fleet, two varying parameters), read
  every 100, 120, 140 and 160 days, seeds 2000 to 2004 (fresh seeds):
  20 fleets with 9% to 36% of the pumps read three times or fewer (the
  spacings were chosen from the counts of readings alone, with no fit);
- each fleet fitted with ``integration="grid"`` and
  ``integration="laplace"``; its error is the largest over the
  parameters of the difference of the estimates in the grid fit's
  standard errors. A fleet where either fit raised, did not reach a
  verified maximum or had no standard errors counts as over the limit;
- the limit is the plan's 0.1 standard errors, and the share is the
  largest of 0.10, 0.15, 0.20, 0.25, 0.30 and 0.35 such that no fleet
  whose share of pumps read three times or fewer is at most it is over
  the limit (0.10 if a fleet at or under 0.10 is over: the warning then
  comes earlier than the study can resolve, which is reported). The
  count of readings that is "few" stays three. ``threshold`` below is
  the rule; the run prints its answer.

    python scripts/studies/joint_recovery/laplace_few_readings.py
    STUDY=confirm python scripts/studies/joint_recovery/laplace_few_readings.py

writes ``laplace_few_readings.json`` beside this script and prints a row
a fleet: the share of items with at most three readings and with fewer
than five, the largest difference between the two fits' estimates in the
grid fit's standard errors (and the parameter it is in), the largest
relative difference of the standard errors, the difference of the
maximised log-likelihoods an item, and each fit's time.
"""

import json
import os
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LINEAR = [
    (every, seed)
    for every in (30, 90, 180, 240, 365)
    for seed in range(1000, 1005)
]
QUAD = [
    (every, seed) for every in (0.1, 0.5, 1.0) for seed in range(1000, 1003)
]
CONFIRM = [
    (every, seed)
    for every in (100, 120, 140, 160)
    for seed in range(2000, 2005)
]
LIMIT = 0.1
SHARES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
CONFIRMING = os.environ.get("STUDY") == "confirm"


def over(row):
    """Whether a fleet's Laplace fit is over the limit, or could not be
    compared with the grid's."""
    return (
        "error" in row
        or row["maxima"] != ["verified", "verified"]
        or not row["max_shift"] <= LIMIT
    )


def threshold(rows):
    """The warning's share by the rule fixed before the confirming run:
    the largest of ``SHARES`` with no fleet at or under it over the
    limit."""
    chosen = SHARES[0]
    for share in SHARES:
        if any(over(r) for r in rows if r["share_up_to_3"] <= share):
            break
        chosen = share
    return chosen


def fleets():
    """``(design, every, seed, fitter, fit arguments, readings an item)``
    of every fleet."""
    from surpyval import JointPH, SurpyvalData, Weibull
    from surpyval.degradation import LinearPath, QuadraticPath
    from surpyval.tests.univariate.joint import _fleet
    from surpyval.tests.univariate.joint import test_joint_integration as t

    for every, seed in CONFIRM if CONFIRMING else LINEAR:
        items, _ = _fleet.simulate(n=300, seed=seed, every=float(every))
        counts = np.array([len(item.markers) for item in items])
        yield (
            "linear",
            every,
            seed,
            JointPH(Weibull, LinearPath),
            ((SurpyvalData.from_items(items),), {"marker_reference": 4.0}),
            counts,
        )
    for every, seed in [] if CONFIRMING else QUAD:
        data = t.simulate(150, seed, q=3, every=every)
        counts = np.array([len(m) for m in data["markers"]])
        yield (
            "quad",
            every,
            seed,
            JointPH(Weibull, QuadraticPath),
            ((), dict(data, marker_reference=t.CENTRE)),
            counts,
        )


def compare(fitter, args, kwargs):
    out = {}
    for integration in ("grid", "laplace"):
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fitter.fit(*args, integration=integration, **kwargs)
        out[integration] = (model, time.time() - start)
    return out


if __name__ == "__main__":
    rows = []
    for design, every, seed, fitter, (args, kwargs), counts in fleets():
        try:
            fits = compare(fitter, args, kwargs)
        except Exception as error:  # recorded, not hidden
            rows.append(
                dict(
                    design=design,
                    every=every,
                    seed=seed,
                    share_up_to_3=float(np.mean(counts <= 3)),
                    error=repr(error),
                )
            )
            print(design, every, seed, "FAILED", repr(error), flush=True)
            continue
        (grid, t_grid), (laplace, t_laplace) = fits["grid"], fits["laplace"]
        row = dict(
            design=design,
            every=every,
            seed=seed,
            items=int(counts.size),
            share_up_to_3=float(np.mean(counts <= 3)),
            share_under_5=float(np.mean(counts < 5)),
            maxima=[grid.maximum, laplace.maximum],
            names=list(grid.parameter_names),
            time_grid=t_grid,
            time_laplace=t_laplace,
        )
        try:
            se = np.asarray(grid.standard_errors(), dtype=float)
            se_l = np.asarray(laplace.standard_errors(), dtype=float)
            shift = (laplace.params - grid.params) / se
            worst = int(np.argmax(np.abs(shift)))
            row.update(
                shift=shift.tolist(),
                max_shift=float(np.abs(shift[worst])),
                worst=grid.parameter_names[worst],
                max_se_change=float(np.max(np.abs(se_l / se - 1.0))),
                ll_gap_an_item=float(
                    (grid.neg_ll() - laplace.neg_ll()) / counts.size
                ),
            )
        except ValueError as error:
            row.update(error=repr(error))
        rows.append(row)
        print(
            design,
            every,
            seed,
            "share<=3 %.2f" % row["share_up_to_3"],
            "share<5 %.2f" % row["share_under_5"],
            row["maxima"],
            "max shift %.3f se (%s)"
            % (row.get("max_shift", np.nan), row.get("worst")),
            "se change %.3f" % row.get("max_se_change", np.nan),
            "ll gap %.2e" % row.get("ll_gap_an_item", np.nan),
            "times %.0f %.0f s" % (t_grid, t_laplace),
            flush=True,
        )
    name = "laplace_few_readings%s.json" % ("_confirm" if CONFIRMING else "")
    out = dict(date=time.strftime("%Y-%m-%d"), rows=rows)
    if CONFIRMING:
        out["share"] = threshold(rows)
        print("the warning's share by the rule:", out["share"], flush=True)
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(out, f, indent=1)
