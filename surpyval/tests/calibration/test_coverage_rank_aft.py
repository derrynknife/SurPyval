"""The rank-based AFT model (``RankAFT``) on data from a true AFT model.

``log T = 1 + 0.5 z1 - 1.0 z2 + 0.5 W``, with ``z1`` binary, ``z2``
uniform on (-1, 1) and ``W`` a standard minimum extreme-value error (so
``T`` is Weibull with shape 2 given the covariates); censored at a uniform
time on (1, 15), about a quarter of the units. In surpyval's sign the
coefficients are ``(-0.5, 1.0)``. n = 200; each data set is fitted with
Gehan and with log-rank weights.

- **Bias and standard errors** (``check_bias`` defaults): the
  standardised bias, and the root mean square of the sandwich standard
  errors over the empirical standard deviation of the estimates.
- **Coverage** of the 95% Wald intervals of ``param_cb``. Slack 0.02:
  the intervals are large-sample, and the sandwich is estimated.

The full study is 1000 replicates; the reduced one is the first 200 of
the same draws (the same seed). Seed 52013. The design, seed and rules
were fixed before the first run.
"""

import numpy as np

from surpyval import RankAFT
from surpyval.tests.calibration._montecarlo import check_bias, check_coverage

N = 200
REPS = 1000
REDUCED = 200
SEED = 52013
TRUTH = np.array([-0.5, 1.0])


def _simulate(rng):
    Z = np.column_stack(
        [rng.binomial(1, 0.5, N), rng.uniform(-1, 1, N)]
    ).astype(float)
    T = np.exp(1.0 - Z @ TRUTH + 0.5 * np.log(rng.exponential(size=N)))
    C = rng.uniform(1.0, 15.0, N)
    return np.minimum(T, C), Z, (T > C).astype(int)


def _study(reps):
    rng = np.random.default_rng(SEED)
    out = {
        w: {k: np.empty((reps, 2)) for k in ("est", "se", "lo", "hi")}
        for w in ("gehan", "logrank")
    }
    for r in range(reps):
        x, Z, c = _simulate(rng)
        for w, res in out.items():
            m = RankAFT.fit(x, Z, c=c, rank_weights=w)
            band = m.param_cb()
            res["est"][r], res["se"][r] = m.beta, m.standard_errors()
            res["lo"][r], res["hi"][r] = band[:, 0], band[:, 1]
    for w, res in out.items():
        check_bias(
            res["est"], TRUTH, f"RankAFT[{w}]", standard_errors=res["se"]
        )
        check_coverage(
            res["lo"],
            res["hi"],
            TRUTH,
            0.95,
            f"RankAFT[{w}] 95% Wald interval",
            slack=0.02,
        )


def test_rank_aft_bias_and_coverage_reduced():
    _study(REDUCED)


def test_rank_aft_bias_and_coverage():
    _study(REPS)
