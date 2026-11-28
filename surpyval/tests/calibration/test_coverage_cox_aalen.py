"""The Cox-Aalen model (``CoxAalen``) on data from a true Cox-Aalen model.

``h(t | z, a) = (0.2 + 0.3 a 1(t < 2)) exp(0.7 z)``, with ``z`` binary
(multiplicative) and ``a`` uniform on (0, 1) (additive, its effect
stopping at t = 2): the truth is ``beta = 0.7``, ``A_0(t) = 0.2 t`` and
``A_1(t) = 0.3 min(t, 2)``. Censored at a uniform time on (2, 10). n =
400; ``additive=[1]``.

- **Bias and standard errors** of ``beta`` (``check_bias`` defaults),
  for the model-based and the robust standard errors.
- **Coverage** of the 95% Wald intervals of ``param_cb`` (model-based
  and robust), and of the pointwise 95% intervals of
  ``cumulative_coefficients_cb`` at t = 1, 3 and 6. Slack 0.02: the
  intervals are large-sample.

The full study is 1000 replicates; the reduced one is the first 200 of
the same draws (the same seed). Seed 31337. The design, seed and rules
were fixed before the first run.
"""

import numpy as np

from surpyval import CoxAalen
from surpyval.tests.calibration._montecarlo import check_bias, check_coverage

N = 400
REPS = 1000
REDUCED = 200
SEED = 31337
BETA = 0.7
TIMES = np.array([1.0, 3.0, 6.0])


def _truth(t):
    return np.column_stack([0.2 * t, 0.3 * np.minimum(t, 2.0)])


def _simulate(rng):
    z = rng.binomial(1, 0.5, N).astype(float)
    a = rng.uniform(0, 1, N)
    e = rng.exponential(size=N) * np.exp(-BETA * z)
    early = 0.2 + 0.3 * a
    t = np.where(e < 2 * early, e / early, 2 + (e - 2 * early) / 0.2)
    end = rng.uniform(2, 10, N)
    return np.minimum(t, end), (t > end).astype(int), np.column_stack([z, a])


def _study(reps):
    rng = np.random.default_rng(SEED)
    est, se, se_r = (np.empty(reps) for _ in range(3))
    lo, hi, lo_r, hi_r = (np.empty(reps) for _ in range(4))
    A_lo = np.empty((reps, TIMES.size, 2))
    A_hi = np.empty_like(A_lo)
    for r in range(reps):
        x, c, Z = _simulate(rng)
        m = CoxAalen.fit(x, Z, c=c, additive=[1])
        est[r] = m.beta[0]
        se[r] = m.standard_errors()[0]
        se_r[r] = m.standard_errors(robust=True)[0]
        lo[r], hi[r] = m.param_cb("coef_0")
        lo_r[r], hi_r[r] = m.param_cb("coef_0", robust=True)
        band = m.cumulative_coefficients_cb(TIMES)
        A_lo[r], A_hi[r] = band[..., 0], band[..., 1]
    check_bias(est, BETA, "Cox-Aalen beta", standard_errors=se)
    check_bias(est, BETA, "Cox-Aalen beta (robust se)", standard_errors=se_r)
    check_coverage(lo, hi, BETA, 0.95, "Cox-Aalen beta 95%", slack=0.02)
    check_coverage(
        lo_r, hi_r, BETA, 0.95, "Cox-Aalen beta 95% (robust)", slack=0.02
    )
    truth = _truth(TIMES)
    for k, t in enumerate(TIMES):
        check_coverage(
            A_lo[:, k],
            A_hi[:, k],
            truth[k],
            0.95,
            f"Cox-Aalen A 95% pointwise at t = {t:g}",
            slack=0.02,
        )


def test_cox_aalen_bias_and_coverage_reduced():
    _study(REDUCED)


def test_cox_aalen_bias_and_coverage():
    _study(REPS)
