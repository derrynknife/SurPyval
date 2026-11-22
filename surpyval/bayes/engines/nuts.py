"""A No-U-Turn sampler for a log density with a gradient.

``sample(f, starts, ...)`` draws from the density ``exp(value)`` of a
function ``f(z) -> (value, gradient)`` on an unconstrained space, such as
:class:`surpyval.bayes._lean.LogPosterior`. It is the sampler of
``fit_bayes(method="nuts")``; it knows nothing about distributions or
data.

The method
----------
- One transition is the No-U-Turn sampler with multinomial sampling of
  the trajectory's states (Betancourt, 2017, "A conceptual introduction
  to Hamiltonian Monte Carlo", appendix A): a leapfrog trajectory is
  doubled forwards or backwards at random until it turns back on itself
  (the generalised criterion, checked on every subtree and across each
  pair of joined subtrees) or reaches :data:`MAX_TREE_DEPTH` doublings;
  the next state is drawn from the trajectory with probability
  proportional to ``exp(-H)``, each new half taken in preference to the
  old one where it weighs more (biased progressive sampling). There is
  no slice variable.
- A trajectory whose energy rises more than :data:`MAX_ENERGY_ERROR`
  above its start stops and is counted as a *divergence*: the step is
  too long for the curvature there, so the region is not being explored
  as the density asks.
- The mass matrix is diagonal. Warm-up adapts the step size by dual
  averaging (Nesterov, 2009; Hoffman and Gelman, 2014) to a mean
  acceptance statistic of :data:`TARGET_ACCEPT`, and the mass matrix
  from the draws of windows that double in length, between a first
  buffer for finding the typical set and a last one for settling the
  step size (the windows of Stan's sampler, :func:`windows`). The
  variances are shrunk a little towards 1e-3 (Stan's regularisation),
  and the step size is found afresh after each window.
- Each chain has its own random stream, spawned from the seed, so a
  chain's draws depend on the seed and the chain's number alone: the
  same whatever ``n_jobs`` is.

The draws are checked by :mod:`surpyval.bayes.convergence`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

#: The most doublings of a trajectory (``2 ** 10`` leapfrog steps).
MAX_TREE_DEPTH = 10
#: The rise in energy along a trajectory that is called a divergence.
MAX_ENERGY_ERROR = 1000.0
#: The mean acceptance statistic the step size is adapted to.
TARGET_ACCEPT = 0.8
#: Dual averaging's constants (Hoffman and Gelman, 2014; Stan's values).
_GAMMA, _T0, _KAPPA = 0.05, 10.0, 0.75
#: Warm-up's buffers and first window with 1,000 iterations (Stan's).
_INIT_BUFFER, _TERM_BUFFER, _BASE_WINDOW = 75, 50, 25
_LOG_08 = math.log(0.8)

LogDensity = Callable[[npt.NDArray], tuple[float, npt.NDArray]]


class SamplerError(ValueError):
    """The sampler could not start (no finite density, no step size)."""


def windows(warmup: int) -> tuple[int, list[int]]:
    """Warm-up's plan: the iteration at which the mass matrix's windows
    start, and the iteration each window ends at (its last one). With
    fewer than 20 iterations there are no windows (the step size alone
    is adapted); where the buffers do not fit, they are 15% and 10% of
    the warm-up.

    Examples
    --------
    >>> from surpyval.bayes.engines.nuts import windows
    >>> windows(1000)
    (75, [99, 149, 249, 449, 949])
    >>> windows(100)
    (15, [89])
    >>> windows(10)
    (10, [])
    """
    warmup = int(warmup)
    if warmup < 20:
        return warmup, []
    init, term, base = _INIT_BUFFER, _TERM_BUFFER, _BASE_WINDOW
    if init + base + term > warmup:
        init = int(0.15 * warmup)
        term = int(0.1 * warmup)
        base = warmup - init - term
    last = warmup - term - 1
    ends = []
    end, size = init + base - 1, base
    while True:
        if end + 2 * size > last:
            # the next window would not fit: this one runs to the end
            end = last
        ends.append(end)
        if end == last:
            return init, ends
        size *= 2
        end += size


class _StepSize:
    """Dual averaging of the log step size."""

    def __init__(self, eps: float) -> None:
        self.restart(eps)

    def restart(self, eps: float) -> None:
        self.mu = math.log(10.0 * eps)
        self.counter = 0
        self.s_bar = 0.0
        self.x_bar = 0.0

    def learn(self, accept: float) -> float:
        self.counter += 1
        accept = min(1.0, accept)
        eta = 1.0 / (self.counter + _T0)
        self.s_bar = (1.0 - eta) * self.s_bar + eta * (TARGET_ACCEPT - accept)
        x = self.mu - self.s_bar * math.sqrt(self.counter) / _GAMMA
        x_eta = self.counter**-_KAPPA
        self.x_bar = (1.0 - x_eta) * self.x_bar + x_eta * x
        return math.exp(x)

    def final(self) -> float:
        return math.exp(self.x_bar)


class _Variance:
    """A running mean and variance of the draws (Welford)."""

    def __init__(self, d: int) -> None:
        self.d = d
        self.restart()

    def restart(self) -> None:
        self.n = 0
        self.mean = np.zeros(self.d)
        self.m2 = np.zeros(self.d)

    def add(self, z: npt.NDArray) -> None:
        self.n += 1
        delta = z - self.mean
        self.mean = self.mean + delta / self.n
        self.m2 = self.m2 + delta * (z - self.mean)

    def regularised(self) -> npt.NDArray:
        n = self.n
        var = self.m2 / (n - 1.0)
        return (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))


class _Trajectory:
    """What one transition counts as it builds its tree."""

    __slots__ = ("f", "eps", "inv_m", "h0", "rng", "steps", "accept", "div")

    def __init__(
        self,
        f: LogDensity,
        eps: float,
        inv_m: npt.NDArray,
        h0: float,
        rng: np.random.Generator,
    ) -> None:
        self.f, self.eps, self.inv_m, self.h0, self.rng = (
            f,
            eps,
            inv_m,
            h0,
            rng,
        )
        self.steps = 0
        self.accept = 0.0
        self.div = False


def _leapfrog(
    f: LogDensity,
    z: npt.NDArray,
    r: npt.NDArray,
    g: npt.NDArray,
    eps: float,
    inv_m: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, float, npt.NDArray]:
    r = r + (0.5 * eps) * g
    z = z + eps * (inv_m * r)
    logp, g = f(z)
    r = r + (0.5 * eps) * g
    return z, r, logp, g


def _energy(logp: float, r: npt.NDArray, inv_m: npt.NDArray) -> float:
    """The energy; infinite wherever it is not a finite number (a point
    the density does not cover, or one where it overflows), so such a
    point is never moved to."""
    h = -logp + 0.5 * float(np.dot(r, inv_m * r))
    return h if math.isfinite(h) else math.inf


def _turning(a: npt.NDArray, b: npt.NDArray, rho: npt.NDArray) -> bool:
    """Whether a trajectory whose ends have velocities ``a`` and ``b``
    and whose momenta sum to ``rho`` has turned back on itself."""
    return not (float(np.dot(b, rho)) > 0 and float(np.dot(a, rho)) > 0)


# A subtree is a tuple:
#   (ok, end state (z, r, logp, g), velocity at its first and last
#    state, momentum at its first and last state, sum of momenta, its
#    proposal (z, logp, g), log of its summed weights)
def _build(t: _Trajectory, state: tuple, depth: int, sign: float) -> tuple:
    if depth == 0:
        z, r, logp, g = _leapfrog(
            t.f, state[0], state[1], state[3], sign * t.eps, t.inv_m
        )
        t.steps += 1
        h = _energy(logp, r, t.inv_m)
        if h - t.h0 > MAX_ENERGY_ERROR:
            t.div = True
            return (False,)
        log_w = t.h0 - h
        t.accept += 1.0 if log_w > 0 else math.exp(log_w)
        v = t.inv_m * r
        return (True, (z, r, logp, g), v, v, r, r, r, (z, logp, g), log_w)
    first = _build(t, state, depth - 1, sign)
    if not first[0]:
        return first
    second = _build(t, first[1], depth - 1, sign)
    if not second[0]:
        return second
    log_w = float(np.logaddexp(first[8], second[8]))
    # uniform progressive sampling within a subtree
    take = math.log(t.rng.random()) < second[8] - log_w
    proposal = second[7] if take else first[7]
    rho = first[6] + second[6]
    ok = not (
        _turning(first[2], second[3], rho)
        or _turning(first[2], second[2], first[6] + second[4])
        or _turning(first[3], second[3], second[6] + first[5])
    )
    return (
        ok,
        second[1],
        first[2],
        second[3],
        first[4],
        second[5],
        rho,
        proposal,
        log_w,
    )


def transition(
    f: LogDensity,
    z: npt.NDArray,
    logp: float,
    g: npt.NDArray,
    eps: float,
    inv_m: npt.NDArray,
    rng: np.random.Generator,
    max_depth: int = MAX_TREE_DEPTH,
) -> tuple[npt.NDArray, float, npt.NDArray, dict]:
    """One No-U-Turn transition from ``z``: the next state (its value and
    gradient with it) and what the trajectory did (``accept``, the mean
    acceptance statistic; ``depth``; ``steps``; ``divergent``)."""
    r = rng.standard_normal(len(z)) / np.sqrt(inv_m)
    h0 = _energy(logp, r, inv_m)
    t = _Trajectory(f, eps, inv_m, h0, rng)
    v = inv_m * r
    left = right = (z, r, logp, g)
    # velocities and momenta at the trajectory's two ends
    v_left, v_right, r_left, r_right = v, v, r, r
    rho = r
    sample = (z, logp, g)
    log_w = 0.0
    depth = 0
    while depth < max_depth:
        forward = rng.random() > 0.5
        if forward:
            tree = _build(t, right, depth, 1.0)
        else:
            tree = _build(t, left, depth, -1.0)
        if not tree[0]:
            break
        depth += 1
        # biased progressive sampling between the old tree and the new
        if tree[8] > log_w or math.log(rng.random()) < tree[8] - log_w:
            sample = tree[7]
        log_w = float(np.logaddexp(log_w, tree[8]))
        new_rho = rho + tree[6]
        if forward:
            # old tree: left .. right; new one: tree first .. tree last
            turned = (
                _turning(v_left, tree[3], new_rho)
                or _turning(v_left, tree[2], rho + tree[4])
                or _turning(v_right, tree[3], tree[6] + r_right)
            )
            right, v_right, r_right = tree[1], tree[3], tree[5]
        else:
            turned = (
                _turning(tree[3], v_right, new_rho)
                or _turning(tree[2], v_right, rho + tree[4])
                or _turning(tree[3], v_left, tree[6] + r_left)
            )
            left, v_left, r_left = tree[1], tree[3], tree[5]
        rho = new_rho
        if turned:
            break
    info = {
        "accept": t.accept / t.steps if t.steps else 0.0,
        "depth": depth,
        "steps": t.steps,
        "divergent": t.div,
    }
    return sample[0], sample[1], sample[2], info


def first_step_size(
    f: LogDensity,
    z: npt.NDArray,
    logp: float,
    g: npt.NDArray,
    eps: float,
    inv_m: npt.NDArray,
    rng: np.random.Generator,
) -> float:
    """A step size whose single leapfrog step changes the energy by
    about ``log 0.8`` (Hoffman and Gelman's heuristic, as Stan has it):
    ``eps`` doubled or halved until the change crosses that."""
    direction = 0
    for _ in range(200):
        r = rng.standard_normal(len(z)) / np.sqrt(inv_m)
        h0 = _energy(logp, r, inv_m)
        _, r1, logp1, _ = _leapfrog(f, z, r, g, eps, inv_m)
        delta = h0 - _energy(logp1, r1, inv_m)
        this = 1 if delta > _LOG_08 else -1
        if direction == 0:
            direction = this
        elif this != direction:
            return eps
        eps = 2.0 * eps if direction == 1 else 0.5 * eps
        if not 1e-300 < eps < 1e7:
            break
    raise SamplerError(
        "No step size makes a first move from the starting point: the "
        "log-posterior does not change smoothly there. Give priors that "
        "say where the parameters are, or fix some with fixed=."
    )


def run_chain(
    f: LogDensity,
    start: npt.ArrayLike,
    rng: np.random.Generator,
    draws: int = 1000,
    warmup: int = 1000,
    inv_metric: npt.ArrayLike | None = None,
    max_depth: int = MAX_TREE_DEPTH,
) -> dict:
    """One chain: ``warmup`` adapting iterations, then ``draws`` kept
    ones with the step size and mass matrix fixed.

    Returns a dict with ``draws`` (``(draws, d)``), ``logp``, and per
    kept iteration ``accept``, ``depth``, ``steps`` and ``divergent``;
    ``step_size`` and ``inv_metric`` as adapted; and
    ``warmup_divergences``.
    """
    with np.errstate(all="ignore"):
        return _run_chain(f, start, rng, draws, warmup, inv_metric, max_depth)


def _run_chain(
    f: LogDensity,
    start: npt.ArrayLike,
    rng: np.random.Generator,
    draws: int,
    warmup: int,
    inv_metric: npt.ArrayLike | None,
    max_depth: int,
) -> dict:
    z = np.array(start, dtype=float)
    d = len(z)
    inv_m = (
        np.ones(d) if inv_metric is None else np.array(inv_metric, dtype=float)
    )
    logp, g = f(z)
    if not math.isfinite(logp):
        raise SamplerError(
            "The log-posterior is not finite at the chain's start."
        )
    eps = first_step_size(f, z, logp, g, 1.0, inv_m, rng)
    step = _StepSize(eps)
    var = _Variance(d)
    start_windows, ends = windows(warmup)
    ends_left = list(ends)
    out = np.empty((draws, d))
    keep: dict[str, Any] = {
        "logp": np.empty(draws),
        "accept": np.empty(draws),
        "depth": np.empty(draws, dtype=int),
        "steps": np.empty(draws, dtype=int),
        "divergent": np.zeros(draws, dtype=bool),
    }
    warm_div = 0
    for i in range(warmup + draws):
        if i == warmup and warmup > 0:
            eps = step.final()
        z, logp, g, info = transition(
            f, z, logp, g, eps, inv_m, rng, max_depth
        )
        if i >= warmup:
            j = i - warmup
            out[j] = z
            keep["logp"][j] = logp
            for k in ("accept", "depth", "steps", "divergent"):
                keep[k][j] = info[k]
            continue
        warm_div += int(info["divergent"])
        eps = step.learn(info["accept"])
        if ends_left and i >= start_windows:
            var.add(z)
            if i == ends_left[0]:
                ends_left.pop(0)
                inv_m = var.regularised()
                var.restart()
                eps = first_step_size(f, z, logp, g, eps, inv_m, rng)
                step.restart(eps)
    return {
        "draws": out,
        **keep,
        "step_size": float(eps),
        "inv_metric": inv_m,
        "warmup_divergences": warm_div,
    }


def _whole(value: Any, name: str, least: int) -> int:
    ok = not isinstance(value, (bool, np.bool_)) and isinstance(
        value, (int, np.integer)
    )
    if not ok or int(value) < least:
        raise ValueError(
            f"{name} must be a whole number of at least {least}; got "
            f"{value!r}."
        )
    return int(value)


def check_settings(chains: Any, draws: Any, warmup: Any) -> tuple:
    """``chains``, ``draws`` and ``warmup`` as whole numbers (at least 1
    chain, 4 draws a chain, and no negative warm-up); refuses anything
    else, naming it."""
    return (
        _whole(chains, "chains", 1),
        _whole(draws, "draws", 4),
        _whole(warmup, "warmup", 0),
    )


def sample(
    f: LogDensity,
    starts: npt.ArrayLike,
    *,
    draws: int = 1000,
    warmup: int = 1000,
    random_state: Any = None,
    n_jobs: int | None = None,
    inv_metric: npt.ArrayLike | None = None,
    max_depth: int = MAX_TREE_DEPTH,
) -> dict:
    """
    Draw from the density ``exp(value)`` of ``f(z) -> (value, gradient)``
    with one No-U-Turn chain per row of ``starts``.

    Parameters
    ----------
    f : callable
        The log density and its gradient at one point.
    starts : 2-D array like
        Each chain's starting point, ``(chains, d)``.
    draws, warmup : int, optional
        Kept and adapting iterations per chain. Default 1000 each.
    random_state : optional
        The seed, by the package's one rule (``None``: numpy's global
        stream). Each chain's stream is spawned from it.
    n_jobs : int, optional
        Chains run at once (``joblib``); one by default. The draws do
        not depend on it.
    inv_metric : 1-D array like, optional
        The diagonal of the inverse mass matrix warm-up starts from
        (variances of the parameters); ones by default.
    max_depth : int, optional
        The most doublings of a trajectory.

    Returns
    -------
    dict
        ``draws`` (``(chains, draws, d)``), ``logp``, ``accept``,
        ``depth``, ``steps`` and ``divergent`` (each ``(chains, draws)``),
        ``step_size`` (one per chain), ``inv_metric`` (``(chains, d)``)
        and ``warmup_divergences`` (one per chain).

    Examples
    --------
    A normal with standard deviations 1 and 3:

    >>> import numpy as np
    >>> from surpyval.bayes.engines.nuts import sample
    >>> sd = np.array([1.0, 3.0])
    >>> def f(z):
    ...     return -0.5 * np.sum((z / sd) ** 2), -z / sd**2
    >>> out = sample(f, np.zeros((2, 2)), draws=500, warmup=500,
    ...              random_state=1)
    >>> out["draws"].shape
    (2, 500, 2)
    >>> bool(np.all(np.abs(out["draws"].std(axis=(0, 1)) / sd - 1) < 0.2))
    True
    """
    from surpyval.utils.rng import as_generator

    starts = np.atleast_2d(np.asarray(starts, dtype=float))
    chains, draws, warmup = check_settings(len(starts), draws, warmup)
    rngs = as_generator(random_state).spawn(chains)
    args = [
        (f, starts[k], rngs[k], draws, warmup, inv_metric, max_depth)
        for k in range(chains)
    ]
    jobs = 1 if n_jobs is None else n_jobs
    if jobs == 1 or chains == 1:
        results = [run_chain(*a) for a in args]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=jobs)(delayed(run_chain)(*a) for a in args)
    out: dict[str, Any] = {
        k: np.stack([r[k] for r in results])
        for k in (
            "draws",
            "logp",
            "accept",
            "depth",
            "steps",
            "divergent",
            "inv_metric",
        )
    }
    out["step_size"] = np.array([r["step_size"] for r in results])
    out["warmup_divergences"] = np.array(
        [r["warmup_divergences"] for r in results]
    )
    return out
