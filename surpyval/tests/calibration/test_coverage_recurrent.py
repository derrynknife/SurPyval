"""Coverage of the parametric recurrent-event bounds (``cif_cb``,
``param_cb``).

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
    lo, hi = np.empty((reps, 3)), np.empty((reps, 3))
    plo, phi = np.empty((reps, 2)), np.empty((reps, 2))
    for r in range(reps):
        x, i, c = simulate_nhpp(rng, cif, inv_cif, SYSTEMS, T_END)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fitter.fit(x, i, c=c)
            b = model.cif_cb(t_eval)
            lo[r], hi[r] = b[:, 0], b[:, 1]
            for j, p in enumerate(model.parameter_names):
                plo[r, j], phi[r, j] = model.param_cb(p)
    check_coverage(lo, hi, truth, 0.95, name + " cif_cb")
    check_coverage(plo, phi, np.asarray(params), 0.95, name + " param_cb")
