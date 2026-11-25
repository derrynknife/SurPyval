"""Coverage of the bounds on a discrete growth model's reliability:
``reliability_cb`` by Wald (logit) and likelihood ratio, two-sided 90%.

Designs, each a modest development programme (the bounds are
asymptotic, so this measures them where they are used):

- Lloyd-Lipow, R_k = 0.9 - 0.3 / k, 10 stages of 30 trials, the
  reliability at stage 10;
- Crow-AMSAA discrete, lambda = 0.6, beta = 0.7, 6 configurations of 25
  trials, the last configuration's reliability;
- Gompertz, R = 0.95 * 0.45 ** (0.6 ** k), 10 stages of 30 trials, the
  reliability at stage 10.

400 data sets each, seed 1126. A data set whose fit is refused, or whose
bound is nan, counts as a miss.
"""

import warnings

import numpy as np

from surpyval.recurrent import CrowAMSAADiscrete, GompertzGrowth, LloydLipow
from surpyval.tests.calibration._montecarlo import check_coverage

REPS = 400
NOMINAL = 0.9


def _amsaa(n):
    T = np.cumsum(n, dtype=float)
    prev = np.concatenate([[0.0], T[:-1]])
    return 1 - 0.6 * (T**0.7 - prev**0.7) / n


DESIGNS = {
    "lloyd-lipow": (
        LloydLipow,
        np.full(10, 30),
        lambda n: 0.9 - 0.3 / np.arange(1, 11),
        10,
    ),
    "crow-amsaa-discrete": (CrowAMSAADiscrete, np.full(6, 25), _amsaa, 6),
    "gompertz": (
        GompertzGrowth,
        np.full(10, 30),
        lambda n: 0.95 * 0.45 ** (0.6 ** np.arange(1, 11)),
        10,
    ),
}


def test_discrete_growth_reliability_coverage():
    rng = np.random.default_rng(1126)
    for name, (fitter, n, curve, stage) in DESIGNS.items():
        r = curve(n)
        truth = r[stage - 1]
        bounds = {m: np.full((REPS, 2), np.nan) for m in ("wald", "lr")}
        for rep in range(REPS):
            failures = rng.binomial(n, 1 - r)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = fitter.fit(failures, n)
                    for m in bounds:
                        bounds[m][rep] = model.reliability_cb(
                            stage, alpha_ci=1 - NOMINAL, method=m
                        )
            except ValueError:
                continue
        for m, b in bounds.items():
            check_coverage(b[:, 0], b[:, 1], truth, NOMINAL, f"{name} {m}")
