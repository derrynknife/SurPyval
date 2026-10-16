"""Coverage of the joint model's bounds (#891).

The truth is a fleet of pumps whose vibration (mm/s) rises along a
straight path at a pump-specific rate, read every 30 days with noise
0.6 mm/s, and whose hazard is a Weibull (shape 1.4) times ``exp(0.5
heavy + 0.9 (m(t) - 3.5))``, ``heavy`` a duty indicator for 40% of the
pumps; observation ends at two years and 15% of the pumps are withdrawn
earlier for unrelated reasons. Each replicate is 200 pumps, fitted by
``JointModel.fit``, ``reps`` of them with a fixed seed.

- **Parameters:** the two-sided 95% Wald interval of every parameter
  (``param_cb``) must cover its true value 95% of the time, and the mean
  estimate must be within the bias rule of ``check_bias`` (the model's own
  standard errors against the spread of the estimates too). The
  random-effect standard deviations and the noise are on the log scale,
  the correlation on the logit, as ``param_cb`` forms them. The fits take
  the true ``marker_centre``, at which ``alpha`` is defined.
- **Dynamic prediction:** a pump held out of every replicate, with five
  readings rising from 3.3 to 5.0 mm/s over 120 days, is predicted from
  each fit; the 95% Wald band of its conditional survival (``cb``) must
  cover the true conditional survival (the truth's own prediction, which
  ``tests/joint/test_dynamic_prediction.py`` checks against Monte Carlo)
  at 60, 180 and 365 days from now.

Slack: 0.02 for the coverages (the Wald intervals are asymptotic, and the
variance components' are the slowest to get there) and the default 0.2 of
a standard deviation for the bias.
"""

import warnings

import numpy as np

from surpyval.joint import JointModel
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
)

TRUTH = {
    "alpha": 900.0,
    "beta": 1.4,
    "heavy": 0.5,
    "association": 0.9,
    "path_intercept": 2.8,
    "path_slope": 1.6 / 365,
    "sd_intercept": 0.5,
    "sd_slope": 0.7 / 365,
    "corr": 0.0,
    "sigma": 0.6,
}
CENTRE = 3.5
UNITS = 200
REPS = 200
#: The held-out pump: its readings so far, and the ages to predict at.
HELD_OUT = (
    np.array([0.0, 30.0, 60.0, 90.0, 120.0]),
    np.array([3.3, 3.9, 4.2, 4.6, 5.0]),
)
HORIZONS = 120.0 + np.array([60.0, 180.0, 365.0])


def truth_model():
    return JointModel.from_params(TRUTH, marker_centre=CENTRE)


def fleet(truth, rng):
    """One replicate: 200 pumps, read monthly for two years, some
    withdrawn early (independent censoring)."""
    heavy = (rng.random(UNITS) < 0.4).astype(float)
    df = truth.simulate(UNITS, 30.0, 730.0, Z=heavy, random_state=rng)
    units = df.groupby("i").first()
    x = units["x"].to_numpy()
    c = units["c"].to_numpy()
    lost = rng.random(UNITS) < 0.15
    withdraw = rng.uniform(30.0, 730.0, UNITS)
    early = lost & (withdraw < x)
    x = np.where(early, withdraw, x)
    c = np.where(early, 1, c)
    keep = df["marker_x"].to_numpy() <= x[df["i"].to_numpy()]
    return {
        "x": x,
        "c": c,
        "Z": heavy,
        "i": units.index.to_numpy(),
        "marker_x": df["marker_x"].to_numpy()[keep],
        "marker_y": df["marker_y"].to_numpy()[keep],
        "marker_i": df["i"].to_numpy()[keep],
    }


def replicate(seed):
    """One replicate's estimates, their 95% intervals and standard
    errors, and the held-out pump's band."""
    rng = np.random.default_rng(seed)
    data = fleet(truth_model(), rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = JointModel.fit(**data, marker_centre=CENTRE)
        names = model.parameter_names
        bounds = np.array([model.param_cb(n) for n in names])
        band = model.predict(*HELD_OUT, Z=[1.0]).cb(HORIZONS)
    return {
        "estimates": np.asarray(model.params, dtype=float),
        "lower": bounds[:, 0],
        "upper": bounds[:, 1],
        "se": model.standard_errors(),
        "band": band,
        "maximum": model.maximum,
    }


def test_joint_model_coverage():
    truth = truth_model()
    target = np.array([TRUTH[n] for n in truth.parameter_names])
    true_sf = truth.predict(*HELD_OUT, Z=[1.0]).sf(HORIZONS)
    runs = [replicate(9100 + r) for r in range(REPS)]
    verified = np.mean([r["maximum"] == "verified" for r in runs])
    print("verified maxima: {:.3f}".format(verified))
    assert verified > 0.99
    est = np.array([r["estimates"] for r in runs])
    check_bias(
        est,
        target,
        "joint model parameters",
        standard_errors=np.array([r["se"] for r in runs]),
    )
    check_coverage(
        np.array([r["lower"] for r in runs]),
        np.array([r["upper"] for r in runs]),
        target,
        0.95,
        "joint model param_cb",
        slack=0.02,
    )
    band = np.array([r["band"] for r in runs])
    check_coverage(
        band[..., 0],
        band[..., 1],
        true_sf,
        0.95,
        "dynamic prediction cb(sf)",
        slack=0.02,
    )
