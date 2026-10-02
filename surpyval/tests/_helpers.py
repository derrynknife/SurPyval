"""Data makers and small helpers shared by several test modules.

A helper that a second test module needs moves here (with a public name)
rather than being copied; see "Where a test goes" in docs/Contributing.rst.
"""

import json
import warnings

import numpy as np

import surpyval as sp
from surpyval import AcceleratedLife, Exponential, Turnbull, Weibull
from surpyval.datasets import load_rossi_static
from surpyval.life_models import Power

# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


def no_warnings(func, *args, **kwargs):
    """``func(*args, **kwargs)``, failing on any warning it gives."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return func(*args, **kwargs)


def quietly(fit, *args, **kwargs):
    """``fit(*args, **kwargs)`` with every warning silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(*args, **kwargs)


def dropped_row_messages(record):
    """The "Dropped ..." messages among the recorded warnings."""
    return [str(w.message) for w in record if "Dropped" in str(w.message)]


def json_round_trip(d):
    """JSON round-trip a dict (proves it is JSON-serialisable)."""
    return json.loads(json.dumps(d))


def strict_json_model_round_trip(model):
    """``model`` through strict JSON (no NaN or Infinity literals)."""
    text = json.dumps(model.to_dict(), allow_nan=False)
    return sp.from_dict(json.loads(text))


def fresh_conformance_fit(name):
    """A fresh fit of a conformance fixture, not the shared cached one."""
    from surpyval.tests.conformance.registry import CASE_BY_NAME

    case = CASE_BY_NAME[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return case.fit(case.data())


def finite_difference_covariance(model, rel_step=1e-3):
    """A parametric regression model's covariance and an independent one.

    Returns ``(covariance, reference)`` over the free parameters: the
    model's own, and the inverse of a central-difference Hessian of its
    negative log-likelihood at the same point (that of
    ``_inference_state``), each step ``rel_step`` of the parameter (at
    least 1e-2 of it) and extrapolated (Richardson) from that step and
    half of it."""
    from surpyval.univariate.regression._fit_skeleton import centred_copy

    p_hat, center, cov = model._inference_state()
    p_hat = np.asarray(p_hat, dtype=float)
    free = [
        i
        for i, n in enumerate(model.parameter_names)
        if n not in model._held()
    ]
    data = model.data
    if center is not None and np.any(center):
        data = centred_copy(data, center)

    def f(v):
        full = p_hat.copy()
        full[free] = v
        return float(model.model.neg_ll(data, *full))

    h = rel_step * np.maximum(np.abs(p_hat[free]), 1e-2)
    with np.errstate(all="ignore"):
        H = richardson_hessian(f, p_hat[free], h)
    return cov[np.ix_(free, free)], np.linalg.inv(H)


def richardson_gradient(f, x, step):
    """The gradient of the scalar ``f`` at ``x`` by central differences
    with steps ``step`` (one per coordinate) and half of them,
    extrapolated (Richardson): error O(step**4) from truncation, plus
    about ``eps |f| / step`` from rounding."""
    x = np.asarray(x, dtype=float)
    step = np.broadcast_to(np.asarray(step, dtype=float), x.shape)

    def central(h):
        g = np.empty(x.size)
        for i in range(x.size):
            e = np.zeros(x.size)
            e[i] = h[i]
            g[i] = (float(f(x + e)) - float(f(x - e))) / (2 * h[i])
        return g

    return (4 * central(step / 2) - central(step)) / 3


def richardson_hessian(f, x, step):
    """The Hessian of the scalar ``f`` at ``x`` by central second
    differences with steps ``step`` and half of them, symmetrised and
    extrapolated (Richardson): error O(step**4) from truncation, plus
    about ``eps |f| / step**2`` from rounding."""
    x = np.asarray(x, dtype=float)
    step = np.broadcast_to(np.asarray(step, dtype=float), x.shape)
    k = x.size

    def central(h):
        H = np.empty((k, k))
        for i in range(k):
            for j in range(k):
                ei, ej = np.zeros(k), np.zeros(k)
                ei[i], ej[j] = h[i], h[j]
                H[i, j] = (
                    float(f(x + ei + ej))
                    - float(f(x + ei - ej))
                    - float(f(x - ei + ej))
                    + float(f(x - ei - ej))
                ) / (4 * h[i] * h[j])
        return 0.5 * (H + H.T)

    return (4 * central(step / 2) - central(step)) / 3


def richardson_jacobian(f, x, step):
    """The Jacobian ``d f_i / d x_j`` of the vector ``f`` at ``x``, as
    :func:`richardson_gradient` takes a gradient."""
    x = np.asarray(x, dtype=float)
    step = np.broadcast_to(np.asarray(step, dtype=float), x.shape)

    def central(h):
        cols = []
        for j in range(x.size):
            e = np.zeros(x.size)
            e[j] = h[j]
            up = np.asarray(f(x + e), dtype=float)
            down = np.asarray(f(x - e), dtype=float)
            cols.append((up - down) / (2 * h[j]))
        return np.stack(cols, axis=-1)

    return (4 * central(step / 2) - central(step)) / 3


def neg_ll_at(model, theta):
    """A univariate model's negative log-likelihood at ``theta``."""
    with np.errstate(all="ignore"):
        return float(
            model.dist._neg_ll_func(
                model.surv_data, *theta, model.gamma, model.f0, model.p
            )
        )


def tree_leaves(node):
    """The terminal nodes under ``node`` of a survival tree."""
    from surpyval.beta.ml.forest.node import TerminalNode

    if isinstance(node, TerminalNode):
        return [node]
    return tree_leaves(node.left_child) + tree_leaves(node.right_child)


# ---------------------------------------------------------------------------
# Univariate data
# ---------------------------------------------------------------------------


def sharp_drop_long_tail_data():
    """A sharp drop followed by a long flat tail, which makes an ordinary
    cubic spline through the Kaplan-Meier estimate overshoot below zero."""
    return np.array([1.0, 2.0, 3.0, 3.2, 3.4, 8.0, 12.0, 20.0])


def small_kaplan_meier():
    """A Kaplan-Meier fit to five values, two of them censored."""
    return sp.KaplanMeier.fit([1, 2, 3, 4, 5], c=[0, 1, 0, 0, 1])


# The 11-row mixed-censoring example from the docs review.
TURNBULL_MIXED_CENSORING = dict(
    x=[1, 2, [3, 6], 7, 8, 9, [5, 9], [4, 10], [7, 10], 11, 12],
    c=[1, 1, 2, 0, 0, 0, 2, 2, 2, -1, 0],
    n=[1, 2, 1, 3, 2, 2, 1, 1, 2, 1, 1],
)


def fit_turnbull_quietly(**kwargs):
    """``Turnbull.fit`` without its convergence warnings."""
    with warnings.catch_warnings():
        # Some of these fits are deliberately slow to converge; the
        # warning is not what is under test.
        warnings.simplefilter("ignore", UserWarning)
        return Turnbull.fit(**kwargs)


def random_right_censoring(t, rng, c_max):
    """``t`` censored by uniform times on ``(0, c_max)``: ``(x, c)``."""
    cens = rng.uniform(0, c_max, t.size)
    return np.minimum(t, cens), (cens < t).astype(int)


# ---------------------------------------------------------------------------
# Regression and competing-risks data
# ---------------------------------------------------------------------------


def weibull_ph_data(n: int = 200, seed: int = 0) -> tuple:
    """Weibull proportional-hazards times on one normal covariate."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 1))
    x = 10 * rng.weibull(1.5, n) * np.exp(-0.5 * Z[:, 0] / 1.5)
    return x, Z


def weibull_binary_covariate_data(n=100, seed=1, effect=-0.5):
    """Weibull times scaled by a binary covariate (global seed)."""
    np.random.seed(seed)
    Z = np.random.binomial(1, 0.5, n).reshape(-1, 1)
    x = Weibull.random(n, 10, 2) * np.exp(effect * Z[:, 0])
    return x, Z


def counted_regression_data() -> tuple:
    """Rounded exponential times with counts: ``(x, Z, n, c)``."""
    rng = np.random.default_rng(7)
    x = np.round(rng.exponential(5, 50)) + 1
    Z = rng.normal(size=(50, 1))
    n = rng.integers(1, 4, 50)
    c = (rng.uniform(size=50) < 0.2).astype(int)
    return x, Z, n, c


def fitted_accelerated_life_model():
    """A Weibull-Power accelerated life model at three stresses."""
    np.random.seed(0)
    Z = np.repeat([1.0, 2.0, 3.0], 30)
    x = Weibull.random(90, 10, 2) * Z**-1.0
    return AcceleratedLife(Weibull, Power).fit(x, Z)


def rossi_with_censoring():
    """The Rossi recidivism data with a ``censored`` column.

    ``arrest`` is 1 for an arrest (#479); the censoring flag is
    ``1 - arrest``.
    """
    df = load_rossi_static()
    return df.assign(censored=1 - df["arrest"])


def competing_risks_regression_data(seed=1, n=120):
    """Two causes with proportional hazards on two covariates, censored:
    ``(x, Z, e)`` with ``None`` for a censored row."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    t1 = rng.exponential(1 / (0.2 * np.exp(Z @ [0.5, -0.3])))
    t2 = rng.exponential(1 / (0.1 * np.exp(Z @ [-0.4, 0.2])))
    tc = rng.exponential(8, n)
    x = np.minimum.reduce([t1, t2, tc])
    e = np.where(tc < np.minimum(t1, t2), None, np.where(t1 < t2, 1, 2))
    return x, Z, e.astype(object)


# ---------------------------------------------------------------------------
# Recurrent events
# ---------------------------------------------------------------------------

# Two repairable items, each observed to 60 (a censoring row closes it).
REPAIR_FLEET_X = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
REPAIR_FLEET_I = [1] * 6 + [2] * 5
REPAIR_FLEET_C = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def exponential_event_times():
    """Forty event times of one item at a constant rate (global seed)."""
    np.random.seed(1)
    return Exponential.random(40, 1e-2).cumsum()


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


def linear_degradation_units(
    n: int = 6, seed: int = 0, noise: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rising linear units measured at t = 1..8, crossing 15 around t = 14."""
    rng = np.random.default_rng(seed)
    t = np.arange(1.0, 9.0)
    xs, ys, ids = [], [], []
    for u in range(n):
        b = rng.normal(1.0, 0.2)
        xs.append(t)
        ys.append(1 + b * t + rng.normal(0, noise, t.size))
        ids.append(np.full(t.size, u))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)


def linear_degradation_units_with_extremes() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """The linear units plus one already past 15 at t = 1 (unit 6) and one
    below it trending away (unit 7)."""
    x, y, i = linear_degradation_units()
    rng = np.random.default_rng(1)
    t = np.arange(1.0, 9.0)
    x = np.concatenate([x, t, t])
    y = np.concatenate(
        [
            y,
            16 + t + rng.normal(0, 0.3, t.size),
            5 - 0.5 * t + rng.normal(0, 0.3, t.size),
        ]
    )
    i = np.concatenate([i, np.full(t.size, 6), np.full(t.size, 7)])
    return x, y, i


def linear_degradation_paths(seed=0):
    """Four linear units measured every 100 hours to 1000."""
    rng = np.random.default_rng(seed)
    x = np.tile(np.arange(100, 1100, 100), 4).astype(float)
    slopes = np.repeat([0.31, 0.28, 0.44, 0.37], 10)
    i = np.repeat([1, 2, 3, 4], 10)
    y = 10 + slopes * x + rng.normal(0, 1, x.size)
    return x, y, i


# ---------------------------------------------------------------------------
# Copulas
# ---------------------------------------------------------------------------

WEIBULL_MARGINS = [
    sp.Weibull.from_params([10, 2]),
    sp.Weibull.from_params([20, 3]),
]
