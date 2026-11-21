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

The design and seeds were fixed before any fit was run.

    python scripts/studies/joint_recovery/laplace_few_readings.py

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


def fleets():
    """``(design, every, seed, fitter, fit arguments, readings an item)``
    of every fleet."""
    from surpyval import JointPH, SurpyvalData, Weibull
    from surpyval.degradation import LinearPath, QuadraticPath
    from surpyval.tests.univariate.joint import _fleet
    from surpyval.tests.univariate.joint import test_joint_integration as t

    for every, seed in LINEAR:
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
    for every, seed in QUAD:
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
                dict(design=design, every=every, seed=seed, error=repr(error))
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
    with open(os.path.join(HERE, "laplace_few_readings.json"), "w") as f:
        json.dump(dict(date=time.strftime("%Y-%m-%d"), rows=rows), f, indent=1)
