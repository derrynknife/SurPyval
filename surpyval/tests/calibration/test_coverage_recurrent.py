"""Coverage of the parametric recurrent-event bounds (``cif_cb``,
``iif_cb``, ``param_cb``, and Crow's demonstrated-MTBF bounds,
``mtbf_cb(method="crow")``, #578).

Data: 10 systems observed to T = 50 from a power-law (Crow-AMSAA) process
with alpha = 10 and beta = 1.5 (about 11 events a system), simulated
from the definition (``_montecarlo.simulate_nhpp``) rather than through the
package's own simulator, so a fault there cannot cancel one in the fit.
The log-linear (Cox-Lewis) process is checked the same way.

Slack 0.01.
"""

import warnings

import numpy as np
import pytest

from surpyval.recurrent import CoxLewis, CrowAMSAA
from surpyval.tests.calibration._montecarlo import (
    check_coverage,
    check_rate,
    simulate_nhpp,
)

SYSTEMS = 10
T_END = 50.0


# fitter, true params, the true cif and its inverse, seed
PROCESSES = {
    "CrowAMSAA": (
        CrowAMSAA,
        [10.0, 1.5],
        lambda t: (t / 10.0) ** 1.5,
        lambda n: 10.0 * n ** (1 / 1.5),
        401,
    ),
    # Cox-Lewis: iif = exp(alpha + beta t), cif = (e^{alpha + beta t}
    # - e^alpha) / beta.
    "CoxLewis": (
        CoxLewis,
        [-1.0, 0.02],
        lambda t: (np.exp(-1.0 + 0.02 * t) - np.exp(-1.0)) / 0.02,
        lambda n: (np.log(0.02 * n + np.exp(-1.0)) + 1.0) / 0.02,
        402,
    ),
}


@pytest.mark.parametrize("name", sorted(PROCESSES))
def test_nhpp_cif_cb_and_param_cb(name):
    fitter, params, cif, inv_cif, seed = PROCESSES[name]
    rng = np.random.default_rng(seed)
    t_eval = np.array([10.0, 25.0, 50.0])
    truth = cif(t_eval)
    reps = 1000
    true_iif = fitter.from_params(params).iif(t_eval)
    lo, hi = np.empty((reps, 3)), np.empty((reps, 3))
    ilo, ihi = np.empty((reps, 3)), np.empty((reps, 3))
    plo, phi = np.empty((reps, 2)), np.empty((reps, 2))
    for r in range(reps):
        x, i, c = simulate_nhpp(rng, cif, inv_cif, SYSTEMS, T_END)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fitter.fit(x, i, c=c)
            b = model.cif_cb(t_eval)
            lo[r], hi[r] = b[:, 0], b[:, 1]
            b = model.iif_cb(t_eval)
            ilo[r], ihi[r] = b[:, 0], b[:, 1]
            for j, p in enumerate(model.parameter_names):
                plo[r, j], phi[r, j] = model.param_cb(p)
    check_coverage(lo, hi, truth, 0.95, name + " cif_cb")
    check_coverage(ilo, ihi, true_iif, 0.95, name + " iif_cb")
    check_coverage(plo, phi, np.asarray(params), 0.95, name + " param_cb")


@pytest.mark.parametrize("design", ["time", "failure"])
def test_578_crow_demonstrated_mtbf_bounds(design):
    # A reliability growth test (beta = 0.6, a falling intensity), the
    # 90% lower and upper bounds on the demonstrated MTBF at the end of
    # the test. Failure terminated (one system to its 6th failure) the
    # pivot is exact, so each side covers 90%; time terminated (3 systems
    # to T = 50, about 21 failures in all) the bounds invert a discrete
    # conditional test and cover at least 90% (92% here).
    rng = np.random.default_rng(578 if design == "time" else 579)
    beta, scale = 0.6, 2.0
    model_true = CrowAMSAA.from_params([scale, beta])
    reps = 4000
    low_hits = up_hits = 0
    for _ in range(reps):
        if design == "time":
            counts = rng.poisson((T_END / scale) ** beta, 3)
            x, i, c = [], [], []
            for unit, k in enumerate(counts):
                u = np.sort(rng.uniform(size=k))
                x += [*(T_END * u ** (1 / beta)), T_END]
                i += [unit] * (k + 1)
                c += [0] * k + [1]
            if sum(counts) < 1:
                low_hits += 1  # no bound, counted as covered
                up_hits += 1
                continue
            end = T_END
        else:
            arrivals = np.cumsum(rng.exponential(size=6))
            x = scale * arrivals ** (1 / beta)
            i, c = [1] * 6, [0] * 6
            end = x[-1]
        truth = float(model_true.mtbf(end))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CrowAMSAA.fit(x, i, c=c)
            lower = model.mtbf_cb(end, 0.1, "lower", method="crow")
            upper = model.mtbf_cb(end, 0.1, "upper", method="crow")
        low_hits += lower <= truth
        up_hits += truth <= upper
    side = "both" if design == "failure" else "lower"
    for hits, label in ((low_hits, "lower"), (up_hits, "upper")):
        check_rate(
            hits,
            reps,
            0.9,
            "Crow {}-terminated {} bound".format(design, label),
            side=side,
        )
