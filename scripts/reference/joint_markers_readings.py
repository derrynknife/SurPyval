"""Write the temperature readings of the several-marker test fleet
(``_multi_fleet.simulate(n=80, seed=7)``) as a CSV of ``id, t, y`` for
``reference_r_joint_markers.R``.

    python scripts/reference/joint_markers_readings.py readings.csv
"""

import csv
import sys

from surpyval.tests.univariate.joint._multi_fleet import simulate


def main(path: str) -> None:
    _, _, markers, _ = simulate(n=80, seed=7)
    with open(path, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["id", "t", "y"])
        for i, readings in enumerate(markers):
            for age, value in readings["temperature"].items():
                out.writerow([i, repr(age), repr(value)])


if __name__ == "__main__":
    main(sys.argv[1])
