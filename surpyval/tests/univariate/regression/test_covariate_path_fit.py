"""Fitting a regression on continuous covariate paths (``CovariatePath``),
by cutting each path into steps and halving them until the fit stops
moving.

The checks: ``CovariatePath.to_steps`` (a ramp, a flat path, a path made
of steps, a repeating one, its refusals); the cumulative hazard along
midpoint steps against its closed form under a linear ramp, the error
falling with the square of the step; the fit against the maximum of the
likelihood written with that closed form; a flat path against the
constant-covariate fit and a step-shaped path against ``fit_tvc``;
recovery of ramp-stress simulations (seeds 147 and 148, chosen before
the fits were seen) under proportional hazards, Cox and accelerated
failure time; convergence as the steps are refined; and what is refused
or warned of.
"""

import functools
import warnings

import numpy as np
import pytest
from scipy.optimize import minimize

import surpyval as sp
from surpyval import CovariatePath, ItemLifetime, StepSchedule, SurpyvalData
from surpyval.utils import surpyval_data

RATES = np.array([0.02, 0.05, 0.1])
END = 60.0


def _outcome(life, end=None):
    """An item's outcome: failed at ``life``, or still running at
    ``end``."""
    end = END if end is None else end
    if life > end:
        return {"right_censored": float(end)}
    return {"observed": float(life)}


def _ramp(rate):
    return CovariatePath.from_points([0.0, END], [0.0, END * rate])


# -- the steps ---------------------------------------------------------------


def test_a_ramp_is_cut_into_midpoint_steps():
    ramp = CovariatePath.from_points([0, 10], [0.0, 1.0])
    stepped = ramp.to_steps(25, steps=4)
    np.testing.assert_array_equal(
        stepped.edges, [0.0, 2.5, 5.0, 7.5, 10.0, np.inf]
    )
    np.testing.assert_array_equal(
        stepped.Z.ravel(), [0.125, 0.375, 0.625, 0.875, 1.0]
    )
    # a step's value is the ramp's average over it
    starts, ends, values = stepped.segments(10.0)
    np.testing.assert_allclose(
        values.ravel(), (ramp(starts).ravel() + ramp(ends).ravel()) / 2
    )


def test_a_flat_or_stepped_path_comes_back_as_its_steps():
    flat = CovariatePath.from_points([0, 5, 9], [0.3, 0.3, 0.3])
    assert flat.to_steps(20, steps=16) == StepSchedule.from_changepoints(
        [0], [0.3]
    )
    stepped = CovariatePath.from_points([0, 5, 5, 10], [1.0, 1.0, 2.0, 2.0])
    assert stepped.to_steps(20, steps=16) == StepSchedule.from_changepoints(
        [0, 5], [1.0, 2.0]
    )
    hold = CovariatePath.from_points([0, 4, 4, 8], [0.0, 1.0, 0.5, 0.5])
    np.testing.assert_array_equal(
        hold.to_steps(30, steps=2).edges, [0.0, 2.0, 4.0, np.inf]
    )


def test_a_repeating_path_is_stepped_period_by_period():
    cycle = CovariatePath.from_callable(lambda t: np.sin(np.pi * t), period=2)
    stepped = cycle.to_steps(3.0, steps=2)
    np.testing.assert_array_equal(stepped.edges, [0.0, 1.0, 2.0, 2.5, np.inf])
    np.testing.assert_allclose(
        stepped.Z.ravel(), np.sin(np.pi * np.array([0.5, 1.5, 2.25, 2.75]))
    )
    earlier = cycle.to_steps(1.0, steps=1, start=-2.0)
    np.testing.assert_array_equal(earlier.edges, [-2.0, 0.0, np.inf])
    two = CovariatePath.from_points([0, 10], [[0.0, 5.0], [1.0, 5.0]])
    assert two.to_steps(10, steps=2).Z.tolist() == [[0.25, 5.0], [0.75, 5.0]]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(t_max=10, steps=0), "'steps' is the number"),
        (dict(t_max=10, steps=2.5), "'steps' is the number"),
        (dict(t_max=10, steps=True), "'steps' is the number"),
        (dict(t_max=np.inf), "finite ages"),
        (dict(t_max=10, start=-1.0), "starts at 2, after 'start' = -1"),
        (dict(t_max=2.0), "must be after the schedule's start"),
    ],
)
def test_what_cannot_be_stepped_is_refused(kwargs, match):
    path = CovariatePath.from_points([2, 10], [0.0, 1.0])
    with pytest.raises(ValueError, match=match):
        path.to_steps(**kwargs)


# -- a linear ramp by hand ---------------------------------------------------


def _ramp_items(n, seed, lam=0.02, beta=1.2):
    """Units on ramps ``z(t) = r t`` (three rates in turn) with a constant
    baseline hazard ``lam``: ``H(t) = lam (exp(beta r t) - 1) / (beta r)``,
    inverted exactly; censored at ``END``."""
    rng = np.random.default_rng(seed)
    rate = RATES[np.arange(n) % 3]
    life = np.log1p(beta * rate * rng.exponential(size=n) / lam) / (
        beta * rate
    )
    x, c = np.minimum(life, END), (life > END).astype(int)
    items = [
        ItemLifetime(
            k,
            **_outcome(x[k] if not c[k] else np.inf, x[k]),
            time_varying_covariates=_ramp(rate[k]),
        )
        for k in range(n)
    ]
    return items, x, c, rate


@functools.cache
def _exponential_fit():
    items, x, c, rate = _ramp_items(300, 11)
    return sp.ExponentialPH.fit(SurpyvalData.from_items(items)), x, c, rate


def test_the_cumulative_hazard_along_steps_tends_to_the_closed_form():
    model, *_ = _exponential_fit()
    lam, beta = model.params
    rate, t = 0.05, np.array([10.0, 30.0, 55.0])
    exact = lam * np.expm1(beta * rate * t) / (beta * rate)
    # the path itself, by quadrature
    np.testing.assert_allclose(model.Hf_tvc(t, _ramp(rate)), exact, rtol=1e-9)
    errors = []
    for steps in (8, 16, 32, 64):
        stepped = _ramp(rate).to_steps(END, steps)
        # at ages on the edge of a step for each of these numbers of steps
        at = np.array([15.0, 30.0, 52.5])
        ref = lam * np.expm1(beta * rate * at) / (beta * rate)
        errors.append(np.max(np.abs(model.Hf_tvc(at, stepped) / ref - 1)))
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    # the square of the step: a quarter at each halving
    np.testing.assert_allclose(ratios, 4.0, rtol=0.05)
    # the midpoint rule's error for exp(g u) over steps of width w is
    # (g w)^2 / 24 of the integral, to leading order
    widths = END / np.array([8, 16, 32, 64])
    np.testing.assert_allclose(
        errors[1:], (beta * rate * widths[1:]) ** 2 / 24, rtol=0.1
    )


def test_the_fit_is_the_maximum_of_the_closed_form_likelihood():
    model, x, c, rate = _exponential_fit()
    failed = c == 0

    def neg_ll(v):
        lam, beta = np.exp(v[0]), v[1]
        H = lam * np.expm1(beta * rate * x) / (beta * rate)
        return np.sum(H) - np.sum(
            np.log(lam) + beta * rate[failed] * x[failed]
        )

    exact = minimize(
        neg_ll, [np.log(0.05), 0.5], method="BFGS", options={"gtol": 1e-10}
    )
    target = np.array([np.exp(exact.x[0]), exact.x[1]])
    report = model.path_steps
    assert report["converged"] and report["change"] <= report["tolerance"]
    assert report["tolerance"] == 0.01
    gap = np.abs(model.params - target) / model.standard_errors()
    # what is left is about a third of the last move
    assert np.all(gap < report["tolerance"]), gap
    # and the truth is recovered
    z = (model.params - [0.02, 1.2]) / model.standard_errors()
    assert np.all(np.abs(z) < 3), z


# -- flat and stepped paths are the fits there already were -------------------


def _plain(n=120, seed=3):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0, 1, n)
    x = 50 * rng.weibull(1.5, n) * np.exp(-0.8 * z / 1.5)
    c = (x > 70).astype(int)
    return np.minimum(x, 70.0), c, z


@pytest.mark.parametrize("name", ["WeibullPH", "WeibullAFT", "CoxPH"])
def test_a_flat_path_is_the_constant_covariate_fit(name):
    x, c, z = _plain()
    fitter = getattr(sp, name)
    items = [
        ItemLifetime(
            k,
            **_outcome(x[k] if not c[k] else np.inf, x[k]),
            time_varying_covariates=CovariatePath.from_points([0.0], [z[k]]),
        )
        for k in range(x.size)
    ]
    on_paths = fitter.fit(SurpyvalData.from_items(items))
    constant = fitter.fit(x, Z=z[:, None], c=c)
    np.testing.assert_allclose(on_paths.params, constant.params, rtol=1e-5)
    assert on_paths.path_steps["converged"]
    assert on_paths.path_steps["change"] == 0.0
    assert on_paths.path_steps["rows"] == x.size


@pytest.mark.parametrize("name", ["WeibullPH", "WeibullAFT", "CoxPH"])
def test_a_step_shaped_path_is_the_step_fit(name):
    x, c, z = _plain()
    fitter = getattr(sp, name)

    def items(schedule):
        return SurpyvalData.from_items(
            [
                ItemLifetime(
                    k,
                    **_outcome(x[k] if not c[k] else np.inf, x[k]),
                    time_varying_covariates=schedule(z[k]),
                )
                for k in range(x.size)
            ]
        )

    on_paths = fitter.fit(
        items(
            lambda v: CovariatePath.from_points(
                [0, 20, 20, 45, 45], [v, v, 1 - v, 1 - v, 0.5]
            )
        )
    )
    on_steps = fitter.fit(
        items(
            lambda v: StepSchedule.from_changepoints(
                [0, 20, 45], [v, 1 - v, 0.5]
            )
        )
    )
    np.testing.assert_array_equal(on_paths.params, on_steps.params)
    assert not hasattr(on_steps, "path_steps")


# -- recovery -----------------------------------------------------------------


@functools.cache
def _weibull_ramps(seed=147, n=600, alpha=40.0, k=1.8, beta=0.9):
    """Units on three ramps under Weibull proportional hazards: the
    lifetime by a trapezium rule on a fine grid of ages."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 400.0, 400001)
    items = []
    for j in range(n):
        rate = RATES[j % 3]
        h = (k / alpha) * (t / alpha) ** (k - 1) * np.exp(beta * rate * t)
        H = np.concatenate([[0.0], np.cumsum((h[1:] + h[:-1]) / 2 * 0.001)])
        life = float(np.interp(rng.exponential(), H, t))
        items.append(
            ItemLifetime(
                j,
                **_outcome(life),
                time_varying_covariates=_ramp(rate),
            )
        )
    return SurpyvalData.from_items(items)


def test_recovers_a_ramp_stress_test_under_proportional_hazards():
    model = sp.WeibullPH.fit(_weibull_ramps())
    assert model.path_steps["converged"]
    z = (model.params - [40.0, 1.8, 0.9]) / model.standard_errors()
    assert np.all(np.abs(z) < 3), z


def test_recovers_a_ramp_stress_test_under_cox():
    model = sp.CoxPH.fit(_weibull_ramps())
    assert model.path_steps["converged"]
    z = (model.params - 0.9) / model.standard_errors()
    assert np.all(np.abs(z) < 3), z


def test_recovers_a_ramp_stress_test_under_accelerated_failure_time():
    # cumulative exposure: the accelerated age along z = r t is
    # psi(t) = (exp(b r t) - 1) / (b r), and T = psi^-1(T0), T0 from the
    # baseline; b is the model's own coefficient of the age's rate
    rng = np.random.default_rng(148)
    n, alpha, k = 600, 60.0, 1.8
    x0, z0 = _aft_constant(rng)
    probe = sp.WeibullAFT.fit(x0, Z=z0)
    sign = np.sign(probe.params[-1])
    b = 0.9
    rate = RATES[np.arange(n) % 3]
    base = alpha * rng.weibull(k, n)
    life = np.log1p(b * rate * base) / (b * rate)
    items = [
        ItemLifetime(
            j,
            **_outcome(life[j]),
            time_varying_covariates=_ramp(rate[j]),
        )
        for j in range(n)
    ]
    model = sp.WeibullAFT.fit(SurpyvalData.from_items(items))
    assert model.path_steps["converged"]
    z = (model.params - [alpha, k, sign * b]) / model.standard_errors()
    assert np.all(np.abs(z) < 3), z


def _aft_constant(rng):
    """Lifetimes shortened by a constant covariate (``exp(0.9 z)`` times
    faster), to read the sign of the model's coefficient."""
    z = rng.uniform(0, 1, 400)
    return 60.0 * rng.weibull(1.8, 400) * np.exp(-0.9 * z), z[:, None]


# -- convergence as the steps are refined -------------------------------------


def test_the_fit_on_finer_steps_tends_to_the_fit_on_the_paths():
    items, _, _, rate = _ramp_items(300, 11)
    model, *_ = _exponential_fit()
    se = model.standard_errors()
    gaps = []
    for steps in (2, 8, 32, 128):
        stepped = SurpyvalData.from_items(
            [
                item.clone(
                    time_varying_covariates=_ramp(rate[k]).to_steps(
                        item.value, steps
                    )
                )
                for k, item in enumerate(items)
            ]
        )
        fit = sp.ExponentialPH.fit(stepped)
        gaps.append(np.max(np.abs(fit.params - model.params) / se))
    # steps of one's own hold a failure's covariate at the middle of its
    # step, so the gap falls with the step (a quarter for four times the
    # steps), not its square
    assert gaps[0] > gaps[1] > gaps[2] > gaps[3], gaps
    np.testing.assert_allclose(
        np.array(gaps[:-1]) / np.array(gaps[1:]), 4.0, rtol=0.25
    )
    assert gaps[-1] < gaps[0] / 40


# -- beside static covariates, counts and late entry --------------------------


def test_paths_beside_static_covariates_counts_and_late_entry():
    items, *_ = _ramp_items(150, 5)
    group = (np.arange(150) % 2).astype(float)
    with_static = [
        item.clone(covariates={"group": group[k]})
        for k, item in enumerate(items)
    ]
    model = sp.WeibullPH.fit(SurpyvalData.from_items(with_static, counts=2))
    assert model.params.size == 4
    assert model.feature_names[0] == "group"
    assert model.path_steps["converged"]
    once = sp.WeibullPH.fit(SurpyvalData.from_items(with_static))
    np.testing.assert_allclose(model.params, once.params, rtol=1e-4)
    late = [
        item.clone(left_truncation=1.0) for item in items if item.value > 1.0
    ]
    entered = sp.ExponentialPH.fit(SurpyvalData.from_items(late))
    assert entered.path_steps["converged"]
    assert np.all(np.isfinite(entered.params))


# -- refusals and the warning -------------------------------------------------


def test_a_fit_that_keeps_moving_warns_with_its_last_move(monkeypatch):
    items, *_ = _ramp_items(60, 7)
    monkeypatch.setattr(surpyval_data, "_PATH_ROUNDS", 1)
    with pytest.warns(RuntimeWarning, match="still moved.*16 a stretch"):
        model = sp.ExponentialPH.fit(SurpyvalData.from_items(items))
    assert not model.path_steps["converged"]
    assert model.path_steps["steps"] == 16
    assert model.path_steps["change"] > 0.01


def test_a_baseline_below_zero_needs_the_entry_age():
    rng = np.random.default_rng(9)
    items = [
        ItemLifetime(
            k,
            observed=float(rng.normal(20, 5)),
            time_varying_covariates=CovariatePath.from_callable(
                lambda t, k=k: 0.01 * (k % 3) * t
            ),
        )
        for k in range(60)
    ]
    with pytest.raises(ValueError, match="left_truncation"):
        sp.NormalPH.fit(SurpyvalData.from_items(items))
    entered = [item.clone(left_truncation=0.0) for item in items]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.NormalPH.fit(SurpyvalData.from_items(entered))
    assert "steps" in model.path_steps


def test_a_path_that_starts_after_the_item_is_refused():
    late = CovariatePath.from_points([5, 30], [0.0, 1.0])
    with pytest.raises(ValueError, match="start at age 5"):
        ItemLifetime(0, observed=10.0, time_varying_covariates=late)
