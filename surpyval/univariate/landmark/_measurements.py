"""Reading the measurements carried by a ``SurpyvalData`` (``data.m``)
for landmarking: each named measurement's readings per unit."""

from __future__ import annotations

from typing import Any

import numpy as np


def readings_by_unit(m: Any, ids: list) -> dict:
    """``{name: [(ages, values), ...]}``, one ``(ages, values)`` pair per
    unit of ``ids`` (in that order, sorted by age, missing readings
    dropped), from measurements ``m`` given as ``{name: {unit: {age:
    value}}}``."""
    if not isinstance(m, dict):
        raise TypeError(
            "The data's measurements (m) must be {{name: {{unit: {{age: "
            "value}}}}}}; got {}.".format(type(m).__name__)
        )
    known = set(ids)
    out = {}
    for name, per_unit in m.items():
        stray = sorted(repr(u) for u in per_unit if u not in known)
        if stray:
            raise ValueError(
                "The measurement {!r} has readings of units not in the "
                "data: {}.".format(name, ", ".join(stray[:5]))
            )
        series = []
        for u in ids:
            pairs = sorted(
                (float(a), float(v))
                for a, v in dict(per_unit.get(u, {})).items()
                if np.isfinite(float(a)) and np.isfinite(float(v))
            )
            ages = np.array([a for a, _ in pairs], dtype=float)
            values = np.array([v for _, v in pairs], dtype=float)
            series.append((ages, values))
        out[str(name)] = series
    return out
