"""Size and power of the distributional regression checks of
``diagnose()``.

- **Size**: the likelihood-ratio test of a quadratic term on each
  parameter (``functional_form``) of a ``WeibullDistributional`` fit to
  data from the model it fits (a load on the scale and the shape, about
  25% right censored, n = 200): each test must reject a true null at 5%
  (slack 0.01).
- **Power**: the same test with the shape bending with the load
  (``log beta`` quadratic in it) must reject at least 80% of the time,
  and the AIC screen (``distribution``) must warn for log-normal data
  fitted with a Weibull at least 80% of the time.
"""

import warnings

import numpy as np

import surpyval as sp
from surpyval.diagnostics import distributional
from surpyval.diagnostics.registry import Context
from surpyval.tests.calibration._montecarlo import check_rate

ALPHA = 0.05
N = 200
REPS = 1000
POWER_REPS = 200


def _data(rng, log_shape):
    load = rng.uniform(0, 1, N)
    x = 10 * np.exp(-0.5 * load) * rng.weibull(np.exp(log_shape(load)))
    cens = rng.uniform(0, 30, N)
    return np.minimum(x, cens), load, (cens < x).astype(int)


def _form_p(x, load, c):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.WeibullDistributional.fit(x, load, c=c)
    ctx = Context(model, ALPHA, ["load"], [0])
    return {r["parameter"]: r["p"] for r in distributional._form_rows(ctx)}


def test_functional_form_size():
    rng = np.random.default_rng(2881)
    rejected = {"alpha": 0, "beta": 0}
    for _ in range(REPS):
        p = _form_p(*_data(rng, lambda z: np.log(1.2) + z))
        for name, value in p.items():
            rejected[name] += value <= ALPHA
    for name, hits in rejected.items():
        check_rate(hits, REPS, ALPHA, f"functional_form on {name} size")


def test_functional_form_power():
    rng = np.random.default_rng(2882)
    hits = 0
    for _ in range(POWER_REPS):
        p = _form_p(*_data(rng, lambda z: np.log(1.2) + 8 * (z - 0.5) ** 2))
        hits += p["beta"] <= ALPHA
    check_rate(hits, POWER_REPS, 0.8, "functional_form power", side="lower")


def test_distribution_screen_power():
    rng = np.random.default_rng(2883)
    hits = 0
    for _ in range(POWER_REPS):
        load = rng.uniform(0, 1, N)
        x = np.exp(2 - 0.5 * load + 0.8 * rng.normal(size=N))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sp.WeibullDistributional.fit(x, load)
        ctx = Context(model, ALPHA, ["load"], [0])
        rows = distributional._distribution_rows(ctx)
        hits += bool(rows) and rows[0]["verdict"] == "warn"
    check_rate(
        hits, POWER_REPS, 0.8, "distribution screen power", side="lower"
    )
