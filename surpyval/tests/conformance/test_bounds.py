"""Valid values (#379).

At every query time a fitted model gives numbers, not NaN, and they are
valid for what they are:

- ``sf`` and ``ff`` in [0, 1], ``sf`` non-increasing and ``ff``
  non-decreasing in time; ``Hf`` non-negative and non-decreasing; ``hf``
  and ``df`` non-negative;
- a cumulative incidence in [0, 1] and non-decreasing, and the causes'
  incidences sum to at most 1;
- a cumulative intensity or mean cumulative function non-negative and
  non-decreasing, an intensity non-negative;
- a joint (copula) distribution in [0, 1], monotone in each coordinate
  and within the Frechet-Hoeffding bounds of its margins; a density
  non-negative.
"""

import numpy as np
import pytest

from surpyval.tests.conformance.registry import (
    BIVARIATE,
    Q_PROBS,
    call,
    cases_for,
    fitted,
)

TOL = 1e-12

# function -> (lower, upper, direction in time: +1, -1 or 0)
RULES = {
    "sf": (0.0, 1.0, -1),
    "ff": (0.0, 1.0, 1),
    "Hf": (0.0, np.inf, 1),
    "hf": (0.0, np.inf, 0),
    "df": (0.0, np.inf, 0),
    "cif": (0.0, 1.0, 1),
    "mcf": (0.0, np.inf, 1),
    "iif": (0.0, np.inf, 0),
    "qf": (-np.inf, np.inf, 1),
}
# For recurrent events ``cif`` is a cumulative intensity, not a probability.
COUNTING_RULES = dict(RULES, cif=(0.0, np.inf, 1))


def _check(name, values, rule):
    lower, upper, direction = rule
    if name == "qf":
        # A step estimate that never falls to 1 - q has no q-quantile;
        # that is NaN (as R's quantile gives NA), and the rest must rise.
        values = values[~np.isnan(values)]
    assert not np.any(np.isnan(values)), f"{name}: NaN at a valid time"
    assert np.all(values >= lower - TOL), f"{name} below {lower}: {values}"
    assert np.all(values <= upper + TOL), f"{name} above {upper}: {values}"
    if direction > 0:
        rising = values[1:] >= values[:-1] - 1e-10
        assert np.all(rising), f"{name} not non-decreasing: {values}"
    elif direction < 0:
        falling = values[1:] <= values[:-1] + 1e-10
        assert np.all(falling), f"{name} not non-increasing: {values}"


def _z_rows(case):
    # Each query row, as one covariate vector for every time.
    if case.Z is None:
        return [None]
    return list(case.Z)


@pytest.mark.parametrize(
    "case", cases_for("bounds", where=lambda c: c.interface != BIVARIATE)
)
def test_values_are_valid(case):
    model = fitted(case)
    rules = COUNTING_RULES if case.interface.startswith("counting") else RULES
    for z in _z_rows(case):
        for fname in case.functions:
            x = Q_PROBS if fname == "qf" else case.x
            values = np.asarray(call(case, model, fname, x, z), float)
            if fname in case.jump_functions:
                # The jumps of a non-parametric estimate are documented as
                # NaN before the first failure (NonParametric.hf).
                values = values[~np.isnan(values)]
            _check(fname, values, rules[fname])
        total = 0.0
        for fname in case.event_functions:
            for e in case.events:
                values = np.asarray(
                    call(case, model, fname, case.x, z, event=e), float
                )
                _check(f"{fname}[{e}]", values, rules[fname])
                if fname == "cif" and rules is RULES:
                    total = total + values
        assert np.all(np.asarray(total) <= 1 + 1e-10), total


@pytest.mark.parametrize(
    "case", cases_for("bounds", where=lambda c: c.interface == BIVARIATE)
)
def test_joint_distribution_is_valid(case):
    model = fitted(case)
    # Each coordinate rises along the first five query points.
    X = case.x[:5]
    assert np.all(np.diff(X, axis=0) > 0)
    cdf = np.asarray(model.cdf(X), float)
    sf = np.asarray(model.sf(X), float)
    _check("cdf", cdf, (0.0, 1.0, 1))
    _check("sf", sf, (0.0, 1.0, -1))
    _check("pdf", np.asarray(model.pdf(case.x), float), (0.0, np.inf, 0))
    F1 = np.asarray(model.margins[0].ff(X[:, 0]), float)
    F2 = np.asarray(model.margins[1].ff(X[:, 1]), float)
    assert np.all(cdf >= np.maximum(F1 + F2 - 1, 0) - 1e-10)
    assert np.all(cdf <= np.minimum(F1, F2) + 1e-10)
