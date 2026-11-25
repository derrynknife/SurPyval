"""The joint model of two markers on the Mayo PBC trial, for comparison
with R's multivariate joint models (``markers_pbc.R``: ``JMbayes2::jm``
and ``joineRML::mjoint``).

The data are ``surpyval/datasets/pbc2.csv`` (the ``pbc2`` of R's JM): 312
patients, 140 deaths (``status2``; transplant or alive is censored); the
markers log serum bilirubin and albumin, each a straight line in years
since enrolment with a level and a rate of each patient's own, the four
correlated; the hazard Weibull proportional hazards in both markers'
current values, no covariates. Visits with either reading missing are
left out (as the R script does), leaving 1,945 visits.

    python scripts/studies/joint_recovery/markers_pbc.py

Writes ``markers_pbc_fit.json`` here (the estimates and standard errors);
``markers_pbc.json`` records the run quoted in the docs beside the R
numbers.
Run 2026-11-25: 1,204 s on the full grid (2,401 nodes a patient).
"""

import json
import os
import time

import numpy as np
import pandas as pd

from surpyval import JointPH, Weibull
from surpyval.degradation import LinearPath

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "..", "surpyval", "datasets", "pbc2.csv")


def main() -> None:
    pbc = pd.read_csv(DATA).dropna(subset=["albumin", "serBilir"])
    pbc["lbil"] = np.log(pbc["serBilir"])
    x, c, rows = [], [], []
    for _, d in pbc.groupby("id", sort=False):
        x.append(float(d["years"].iloc[0]))
        c.append(0 if d["status2"].iloc[0] == 1 else 1)
        rows.append(
            {
                "bilirubin": dict(zip(d["year"], d["lbil"])),
                "albumin": dict(zip(d["year"], d["albumin"])),
            }
        )
    start = time.time()
    model = JointPH(
        Weibull, {"bilirubin": LinearPath, "albumin": LinearPath}
    ).fit(x, c=c, markers=rows)
    out = {
        "seconds": time.time() - start,
        "maximum": model.maximum,
        "estimate": dict(zip(model.parameter_names, map(float, model.params))),
        "se": dict(
            zip(model.parameter_names, map(float, model.standard_errors()))
        ),
    }
    with open(os.path.join(HERE, "markers_pbc_fit.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    print(model)


if __name__ == "__main__":
    main()
