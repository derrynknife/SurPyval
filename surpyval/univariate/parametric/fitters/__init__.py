import warnings
from typing import Any, Callable, Sequence

import autograd.numpy as np
import numpy.typing as npt
from autograd import hessian, value_and_grad
from scipy.optimize import OptimizeResult, minimize


class Gradient:
    """The gradient of a scalar ``fun`` by autograd, taken with its value.

    A drop-in for ``autograd.jacobian(fun)``: ``Gradient(fun)(x, *args)``
    is the same gradient, to the bit. It also gives ``value_and_grad(x,
    *args)``, both from one autograd pass, and keeps the last point's
    pair, so asking again at the same point costs nothing.

    The optimisers used to be given ``fun`` and its gradient separately,
    and at each point evaluated the likelihood once for its value and
    again inside the gradient's pass, which computes the value anyway: a
    sixth of each step on a Weibull of 1,000 rows, a fifth at 100,000
    (#593). :func:`minimize_with_gradient` passes them as one callable
    (scipy's ``jac=True``). The value of the pass is the plain function's
    value, so an optimiser takes the same path to the same point.
    """

    def __init__(self, fun: Callable[..., Any]) -> None:
        self.fun = fun
        self._value_and_grad = value_and_grad(fun)
        self._kept: "tuple[bytes, tuple, Any, npt.NDArray] | None" = None

    def value_and_grad(self, x: npt.ArrayLike, *args: Any) -> tuple:
        """``(fun(x, *args), gradient)``, from one pass."""
        key = np.asarray(x, dtype=float).tobytes()
        kept = self._kept
        if (
            kept is None
            or kept[0] != key
            or len(kept[1]) != len(args)
            or any(a is not b for a, b in zip(kept[1], args))
        ):
            value, grad = self._value_and_grad(x, *args)
            kept = self._kept = (key, args, value, grad)
        # A copy, so a caller that changes it in place cannot change it
        # for the next one
        return kept[2], np.array(kept[3])

    def __call__(self, x: npt.ArrayLike, *args: Any) -> npt.NDArray:
        return self.value_and_grad(x, *args)[1]


def minimize_with_gradient(
    fun: Callable[..., Any],
    x0: npt.ArrayLike,
    args: tuple[Any, ...] = (),
    jac: Any = None,
    **kwargs: Any,
) -> Any:
    """``scipy.optimize.minimize(fun, x0, args, jac=jac, **kwargs)``, with
    the value and the gradient from one pass where ``jac`` is the
    :class:`Gradient` of ``fun`` (scipy's ``jac=True``, #593)."""
    if isinstance(jac, Gradient) and jac.fun is fun:
        return minimize(jac.value_and_grad, x0, args=args, jac=True, **kwargs)
    return minimize(fun, x0, args=args, jac=jac, **kwargs)


def fallback_minimize(
    fun: Callable[..., Any],
    init: npt.NDArray,
    args: tuple[Any, ...],
    jac: Callable[..., Any] | None,
    hess: Callable[..., Any] | None,
    newton_tol: float | None = None,
    floor: "float | npt.ArrayLike" = 1.0,
    give_up: "Callable[[Any], bool] | None" = None,
) -> Any:
    """
    Minimise ``fun`` with BFGS and the supplied jacobian, escalating to
    Newton-CG with the hessian and then to Nelder-Mead whenever a method
    fails or returns nan parameters.

    Newton-CG used to go first. It reaches the same answers, but it
    needs the hessian, and building one is disproportionately expensive
    for the distributions whose derivatives autograd cannot take
    analytically: the incomplete gamma is central-differenced (see
    ``autograd_gamma_compat``), so every second-order entry costs a
    difference of differences. An offset Gamma MSE fit at n=5000 spent
    8.2 of its 8.3 seconds there, and BFGS reached a marginally better
    optimum in half a second.

    Measured over 132 fits -- MSE and MPS, nine distributions, plain,
    right censored, left censored and offset -- reversing the order left
    129 objectives identical and improved three, none worse, for 3.9x
    less time.

    The hessian is still consulted before escalating. Some distributions
    have all-shape parameters whose autograd second derivatives are
    zero, and a zero hessian makes Newton-CG stop at the initial guess
    while reporting success, so there is nothing to escalate to and
    Nelder-Mead should take over instead.

    ``floor`` is passed through to ``preconditioned_bfgs``. ``give_up``,
    where given, is asked of a BFGS result that failed whether the
    objective has no finite optimum to escalate for; it is returned at
    once if so (an offset MPS fit running to its family's limit, #616).
    """
    assert jac is not None and hess is not None
    with np.errstate(all="ignore"):
        # BFGS through the same rescaling maximum likelihood uses (see
        # ``preconditioned_bfgs``). Plain BFGS stops on an absolute
        # gradient threshold, so MPS and MSE were not scale invariant:
        # a Weibull MPS fit to data in thousands stopped 1% short of the
        # optimum and reported success.
        res = preconditioned_bfgs(fun, init, args, jac, floor=floor)
        # Which rung produced the answer, reported as ``model.optimizer``
        res.optimizer = "BFGS"

        failed = (
            (res.success is False)
            or np.isnan(res.x).any()
            or (not np.isfinite(res.fun))
        )
        if failed and give_up is not None and give_up(res):
            return res
        if failed and np.any(hess(np.array(init, dtype=float), *args)):
            newton = minimize_with_gradient(
                fun,
                init,
                args,
                jac,
                method="Newton-CG",
                hess=hess,
                tol=newton_tol,
            )
            # Only an improvement replaces what BFGS found. BFGS often
            # reports "precision loss" *at* the optimum, and a Newton-CG
            # run from the cold start can then "succeed" at a worse point,
            # which used to be taken anyway (a Normal MSE fit to data in
            # thousandths landed 1% off that way).
            if (
                newton.success
                and np.isfinite(newton.fun)
                and not (_usable(res) and res.fun <= newton.fun)
            ):
                res = newton
                res.optimizer = "Newton-CG"

        # The last rung is derivative free, as described above. It used to
        # be scipy's default method, which with no jacobian passed is BFGS
        # on finite differences: the method that had just failed with an
        # exact gradient, retried with a worse one -- and no help at all
        # for the zero-hessian case, whose gradients are the problem. It
        # runs from the cold start, as it always has, and also from the
        # best point found so far, which it then polishes rather than
        # discards (Nelder-Mead never ends worse than its start); the
        # better answer is kept. Either start alone can end in the worse
        # of two optima.
        if (res.success is False) or (np.isnan(res.x).any()):
            starts = [init] + ([res.x] if _usable(res) else [])
            for x0 in starts:
                nm = minimize(fun, x0, method="Nelder-Mead", args=args)
                if not (_usable(res) and res.fun < nm.fun):
                    res = nm
                    res.optimizer = "Nelder-Mead"

    return res


def _usable(res: Any) -> bool:
    """A result with finite parameters and a finite objective."""
    return bool(np.all(np.isfinite(res.x)) and np.isfinite(res.fun))


# The largest scaled gradient (see ``is_local_minimum``) a point may have
# and still count as the optimum. BFGS stops at 1e-6 in the same units;
# the other rungs' absolute tolerances, and BFGS's own "precision loss"
# stops at the optimum, land within 1e-5. A point this far from
# stationary is at most ``n * gtol**2`` from the optimum in
# log-likelihood, far below what any prediction can show.
OPTIMUM_GTOL = 1e-4


def is_local_minimum(
    fun: Callable[..., Any],
    jac: Callable[..., Any] | None,
    hess: Callable[..., Any] | None,
    x: npt.ArrayLike,
    args: tuple[Any, ...] = (),
    floor: "float | npt.ArrayLike" = 1.0,
    obj_scale: float = 1.0,
    gtol: float = OPTIMUM_GTOL,
) -> bool:
    """Whether ``x`` is verifiably a local minimum of ``fun``: its
    gradient is ~0 and its Hessian positive definite.

    An optimiser's ``success`` does not say this. BFGS, TNC and Newton-CG
    each stop on an absolute test, and from a start far from the optimum
    they meet it where the objective first looks flat: a Weibull fitted
    from ``alpha = 1e7`` "converged" at ``beta = 0.099`` with a
    log-likelihood 40 below the maximum (#427). Nor does a failure say the
    opposite: BFGS often reports a loss of precision *at* the optimum.

    Both tests are made in the units ``preconditioned_bfgs`` searches in,
    each component scaled by ``max(|x|, floor)`` and the objective by
    ``obj_scale`` (the number of observations), so they mean the same
    thing whatever units the data are in. The gradient must be below
    ``gtol`` in every component; the Hessian must have a Cholesky
    factor. A point where either cannot be evaluated finitely is not
    verified.
    """
    if jac is None or hess is None:
        return False
    at: npt.NDArray = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(at)):
        return False
    scale = np.maximum(np.abs(at), np.asarray(floor, dtype=float))
    with np.errstate(all="ignore"):
        try:
            g = scale * np.asarray(jac(at, *args), dtype=float) / obj_scale
            if not (np.all(np.isfinite(g)) and np.max(np.abs(g)) < gtol):
                return False
            h = np.atleast_2d(np.asarray(hess(at, *args), dtype=float))
            if not np.all(np.isfinite(h)):
                # autograd's second derivative can be NaN where the
                # first is finite (a Weibull CDF at an interval's lower
                # end of 0): central differences of the gradient instead.
                h = _hessian_from_gradient(jac, at, args, scale)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return False
    h = np.outer(scale, scale) * h / obj_scale
    if not np.all(np.isfinite(h)):
        return False
    try:
        np.linalg.cholesky(0.5 * (h + h.T))
    except np.linalg.LinAlgError:
        return False
    return True


def verify_or_polish(
    fun: Callable[[npt.NDArray], Any],
    res: Any,
    n_obs: float,
    objective: "Callable[[npt.NDArray], Any] | None" = None,
    numerical: bool = False,
    floor: "float | npt.ArrayLike" = 1.0,
) -> tuple[Any, bool]:
    """``res``, a minimum of ``fun`` found some other way, and whether it
    is verifiably a minimum of ``objective`` (``fun`` by default; see
    ``is_local_minimum``).

    The regression fitters search with Nelder-Mead and then TNC, whose
    absolute tolerances stop short of the optimum from a poor start while
    reporting success (#428), or fail far from it without a word: an
    additive hazards Gamma baseline stopped at alpha ~ 1e-282 on a "linear
    search failed". An answer that is not verified is polished with BFGS
    in the units maximum likelihood searches in (see
    ``preconditioned_bfgs``), kept where that improves it, and checked
    again; the caller warns if it still is not a minimum.

    ``numerical=True`` is for an objective autograd cannot differentiate
    (one written in plain numpy): its derivatives are then central
    differences (:func:`numerical_derivatives`). ``floor`` is each
    component's least unit for the check and the polish, as in
    ``is_local_minimum`` (a regression coefficient's is its covariate's,
    ``coefficient_floor`` in ``univariate/regression/_fit_skeleton.py``).
    """
    objective = fun if objective is None else objective
    x0 = np.asarray(res.x, dtype=float)
    if numerical:
        jac, hess = numerical_derivatives(objective, x0, floor)
        polish_jac = numerical_derivatives(fun, x0, floor)[0]
    else:
        jac, hess = Gradient(objective), hessian(objective)
        polish_jac = jac if objective is fun else Gradient(fun)
    if is_local_minimum(
        objective, jac, hess, res.x, floor=floor, obj_scale=n_obs
    ):
        return res, True
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # A penalised objective is constant where the model is invalid,
        # and autograd says so for every gradient taken there
        warnings.filterwarnings("ignore", "Output seems independent")
        polish = preconditioned_bfgs(
            fun, res.x, (), polish_jac, floor=floor, obj_scale=n_obs
        )
    if _usable(polish) and polish.fun <= res.fun:
        res = polish
    return res, is_local_minimum(
        objective, jac, hess, res.x, floor=floor, obj_scale=n_obs
    )


def numerical_derivatives(
    fun: Callable[[npt.NDArray], Any],
    x: npt.ArrayLike,
    floor: "float | npt.ArrayLike" = 1.0,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """``(jac, hess)`` of ``fun`` by central differences, for an objective
    autograd cannot differentiate: steps of ``1e-5`` of each component of
    ``x`` (at least ``1e-5`` of its ``floor``, as in ``is_local_minimum``)
    for the Hessian, a hundredth of that for the gradient, fixed from ``x``
    so that both are the same function wherever they are evaluated.

    A step from a floor of 1 is too small for a component whose unit is
    large: a proportional-intensity coefficient of a covariate in
    millionths, polished from 0 to -2.3e5, had its gradient differenced in
    steps of 1e-7, where the likelihood changed below its rounding, and a
    point 1.7% short of the maximum passed as verified (#577)."""
    from surpyval.utils.linalg import numerical_gradient, numerical_hessian

    steps = 1e-5 * np.maximum(
        np.abs(np.asarray(x, dtype=float)), np.asarray(floor, dtype=float)
    )

    def jac(v: npt.NDArray, *args: Any) -> npt.NDArray:
        return numerical_gradient(lambda u: float(fun(u)), v, 1e-2 * steps)

    def hess(v: npt.NDArray, *args: Any) -> npt.NDArray:
        return numerical_hessian(lambda u: float(fun(u)), v, steps)

    return jac, hess


def at_boundary_maximum(
    fun: Callable[[npt.NDArray], Any],
    x: npt.ArrayLike,
    toward: npt.ArrayLike,
    away: npt.ArrayLike,
    step: float,
    n_obs: float,
) -> bool:
    """Whether a parameter is on a boundary of its space at ``x`` and the
    likelihood is at a maximum there in it -- the condition that replaces
    a zero gradient for a parameter on a boundary.

    A parameter searched in a transformed space whose boundary is at
    infinity (a variance as its log, a probability as its logit) reaches
    the boundary only in the limit: there the likelihood no longer depends
    on it, its gradient and curvature are zero (or rounding), and the
    Hessian is singular, so ``is_local_minimum`` cannot pass however well
    the other parameters are fitted. It is on the boundary when ``fun``,
    the negative log-likelihood, is the same to rounding at ``toward``
    (``x`` with that parameter moved further towards the boundary); and it
    is a maximum there when moving it off the boundary into the space, to
    ``away``, ``step`` from the boundary in the parameter's natural units,
    does not raise the likelihood: a slope per observation above
    ``-OPTIMUM_GTOL``. The caller then checks the other parameters with
    this one held out.
    """
    with np.errstate(all="ignore"):
        f = float(fun(np.asarray(x, dtype=float)))
        f_toward = float(fun(np.asarray(toward, dtype=float)))
        f_away = float(fun(np.asarray(away, dtype=float)))
    if not (np.isfinite(f) and np.isfinite(f_toward) and np.isfinite(f_away)):
        return False
    flat = abs(f_toward - f) <= 1e-12 * max(abs(f), 1.0)
    return bool(flat and (f_away - f) / step / n_obs > -OPTIMUM_GTOL)


def verified_maximum(
    neg_ll: Callable[[npt.NDArray], Any],
    mle: npt.ArrayLike,
    bounds: Sequence[tuple[float | None, float | None]],
    n_obs: float,
) -> bool:
    """Whether ``mle`` is a verified maximum of the likelihood whose
    negative log is ``neg_ll``, a function of the natural parameters with
    the given ``(lower, upper)`` ``bounds``, per observation (``n_obs``),
    for a fit whose likelihood is not written for autograd.

    A parameter on a bound of its space where the likelihood is highest
    -- an ARA repair efficiency of 1, a Kijima ``q`` of 0, a copula at its
    independence end -- is held out of the test (:func:`at_boundary_maximum`:
    the likelihood the same a millionth of the way closer to the bound,
    and not rising ``1e-6`` off it). The others must have a zero gradient
    and a positive-definite Hessian (:func:`is_local_minimum`), by central
    differences (:func:`numerical_derivatives`), with each searched as the
    log of its distance from a one-sided bound, the logit between two, or
    as it is.
    """
    x = np.asarray(mle, dtype=float)
    if not np.all(np.isfinite(x)):
        return False

    def natural(v: Any) -> float:
        return float(neg_ll(np.asarray(v, dtype=float)))

    held = []
    for j, (low, high) in enumerate(bounds):
        for bound, inward in ((low, 1.0), (high, -1.0)):
            if bound is None:
                continue
            toward, away = x.copy(), x.copy()
            toward[j] = bound + (x[j] - bound) * 1e-6
            away[j] = bound + inward * 1e-6
            if at_boundary_maximum(natural, x, toward, away, 1e-6, n_obs):
                held.append(j)
                break
    free = [j for j in range(x.size) if j not in held]
    if not free:
        return True
    lows = [bounds[j][0] for j in free]
    highs = [bounds[j][1] for j in free]

    def to_natural(u: npt.NDArray) -> npt.NDArray:
        out = np.array(u, dtype=float)
        for k, (low, high) in enumerate(zip(lows, highs)):
            if low is not None and high is not None:
                out[k] = low + (high - low) / (1.0 + np.exp(-u[k]))
            elif low is not None:
                out[k] = low + np.exp(u[k])
            elif high is not None:
                out[k] = high - np.exp(u[k])
        return out

    u0 = np.array(x[free], dtype=float)
    with np.errstate(all="ignore"):
        for k, (low, high) in enumerate(zip(lows, highs)):
            if low is not None and high is not None:
                f = (u0[k] - low) / (high - low)
                u0[k] = np.log(f) - np.log1p(-f)
            elif low is not None:
                u0[k] = np.log(u0[k] - low)
            elif high is not None:
                u0[k] = np.log(high - u0[k])
    if not np.all(np.isfinite(u0)):
        return False

    def search(u: npt.NDArray) -> float:
        full = x.copy()
        full[free] = to_natural(np.asarray(u, dtype=float))
        return natural(full)

    jac, hess = numerical_derivatives(search, u0)
    with np.errstate(all="ignore"):
        return is_local_minimum(search, jac, hess, u0, obj_scale=n_obs)


def _hessian_from_gradient(
    jac: Callable[..., Any],
    x: npt.NDArray,
    args: tuple[Any, ...],
    scale: npt.NDArray,
) -> npt.NDArray:
    """The Hessian at ``x`` by central differences of ``jac``, each
    component stepped by 1e-5 of its ``scale``."""
    steps = 1e-5 * scale
    columns = []
    for j, step in enumerate(steps):
        e = np.zeros_like(x)
        e[j] = step
        up = np.asarray(jac(x + e, *args), dtype=float)
        down = np.asarray(jac(x - e, *args), dtype=float)
        columns.append((up - down) / (2 * step))
    return np.array(columns).T


def search_floor(model: Any) -> npt.NDArray:
    """Per-component ``floor`` for ``preconditioned_bfgs`` on a fit.

    One entry per free parameter, in the transformed space the search
    runs in (see ``bounds_convert``):

    - A parameter with a bound is searched, within 1 of the bound, as the
      log of its distance from it (or as a scaled arctanh between two
      bounds); further out it is linear and ``|u0|`` sets the scale. The
      log's natural unit is 1 at every data scale, so the floor is 1, as
      it always was: at ``u0 = 0`` -- a Weibull shape of exactly 1, say --
      the scale must not collapse.
    - An unbounded parameter (a location, a Uniform or Beta4 endpoint,
      the LogNormal's ``mu``) is searched as itself. Its natural unit is
      its own magnitude, or the data's spread when it starts near zero
      -- a Normal fitted to data straddling the origin. The floor is
      that spread, capped at the old floor of 1 so that nothing changes
      for data of order 1 and up: in particular the LogNormal's ``mu`` is
      in log units, where 1 is already the natural unit, and must not
      get a floor of 1e5 from data in the hundred thousands.

    The spread is the standard deviation of the finite observed values
    (interval endpoints included), which scales exactly with the data.
    """
    bounds = model.bounds
    fixed_idx = set(model.fitting_info["fixed_idx"])
    x = np.asarray(model.data["x"], dtype=float)
    x = x[np.isfinite(x)]
    spread = float(np.std(x)) if x.size > 1 else 0.0
    unbounded_floor = min(spread, 1.0) if spread > 0 else 1.0
    return np.array(
        [
            unbounded_floor if (low is None and upp is None) else 1.0
            for i, (low, upp) in enumerate(bounds)
            if i not in fixed_idx
        ]
    )


def offset_step(x: npt.ArrayLike) -> float:
    """The data's own unit for an offset: the mean spacing of the sorted
    finite values, their range over ``n - 1``.

    An offset is found as a distance below the smallest value, and how
    far below is only meaningful relative to the data's scale. The
    offset searches used to measure it in absolute units -- the fitters
    started one unit below the data, and the probability plot searched
    ``exp(-u)`` below it from ``u = 0`` -- so the start, and with it the
    answer, depended on the units the data were recorded in. The mean
    spacing scales exactly with the data and is, like the gap between the
    offset and the first failure, a distance *between* observations
    rather than their overall size.

    A single value, or a sample whose values are all equal, has no
    spacing; its own magnitude stands in (and 1 for a sample of zeros).
    """
    finite = np.sort(np.asarray(x, dtype=float).ravel())
    finite = finite[np.isfinite(finite)]
    lo = float(finite[0])
    step = (float(finite[-1]) - lo) / max(finite.size - 1, 1)
    if not step > 0:
        step = abs(lo) if lo != 0 else 1.0
    return step


def preconditioned_bfgs(
    fun: Callable[..., Any],
    x0: npt.NDArray,
    args: tuple[Any, ...] = (),
    jac: Callable[..., Any] | None = None,
    options: dict[str, Any] | None = None,
    floor: "float | npt.ArrayLike" = 1.0,
    obj_scale: float | None = None,
    callback: "Callable[[npt.NDArray], None] | None" = None,
) -> Any:
    """BFGS on a diagonally rescaled copy of the search vector.

    scipy stops BFGS when ``max|grad| < gtol``, an absolute threshold on
    a quantity that is not scale free: a log-likelihood's gradient
    shrinks like ``1/theta``, so on data measured in tens of thousands
    the default 1e-5 is met well short of the optimum and BFGS reports
    success on its first check. Three reference fits on real data of
    that magnitude landed 1e-2 away in relative terms, at a likelihood
    2e-3 below the answer they were recorded from (#323). It had been
    invisible only because those fits used to end on Nelder-Mead, which
    is derivative free and so kept going.

    Tuning the threshold does not fix it. Scaling ``gtol`` by the
    gradient at the initial guess looks scale free but is not -- the
    initialiser scales with the data too, so that gradient is itself
    roughly scale invariant and the threshold barely moves. Nor is there
    a constant that serves every case: tight enough for a Weibull at
    data scale 1e6 is unreachable for an n=8 sample.

    Rescaling the search fixes the cause instead. With

        s = max(|u0|, floor),   v = u / s,   g(v) = f(s v)

    the starting point is order 1 in every component whatever units the
    data is in, and since ``dg/dv = s * df/du`` -- ``s`` growing like the
    scale exactly as ``df/du`` shrinks -- the gradient the optimiser
    tests is order 1 too. scipy's own default then means the same thing
    at every scale, so no tolerance is passed at all.

    The floor keeps a component that starts at or near zero from being
    scaled away to nothing, and its right value depends on the space the
    search runs in. The univariate fitters search transformed
    parameters (see ``bounds_convert``): a parameter with a bound is
    searched as the log of its distance from the bound when that is
    below 1 (or as an arctanh between two bounds), a coordinate whose
    natural unit is 1 whatever the data scale -- there ``u0 = 0`` just
    means "one unit from the bound", and a floor of 1 is right. An
    unbounded parameter is searched as itself, in the units of the
    data, and a fixed floor of 1 is right only for data of order 1 or
    larger. Below that the floor, not ``|u0|``, set the scale: a Beta4
    endpoint at 1e-3 was searched in steps a thousand times its own
    size, BFGS lost precision, and the fit ended on TNC, whose
    tolerances are absolute, 0.1% off. Those callers pass a
    per-component ``floor`` (see ``search_floor``); the default of 1
    keeps every other caller as it was.

    Dividing through by ``|f(x0)|`` does the same job for the other
    scale: the objective is a sum over observations, so its gradient
    grows like ``n`` even when the data magnitude is fixed. A caller
    that knows the count better passes it as ``obj_scale``. Maximum
    likelihood does: a negative log-likelihood is not itself scale free
    -- multiplying the data by ``k`` adds ``log k`` per failure to it --
    so ``|f(x0)|`` was 100 for a Beta4 sample of 100, 1250 for the same
    sample multiplied by 1e5, and the convergence test loosened
    twelvefold with it. Dividing by ``n`` gives the gradient per
    observation, which is the same at every scale.

    The mapping is linear, diagonal and fixed before the search begins,
    so it cannot move the optimum; it changes the route taken and the
    units of the convergence test, nothing else. ``res.x`` is mapped
    back before returning, so every caller -- including the covariance
    step, which builds its own hessian at the returned point -- sees
    exactly what it saw before. scipy's own ``res.hess_inv`` would be in
    scaled units, and is not used anywhere.

    With ``jac=None`` scipy differences the scaled objective, so the
    finite-difference step is relative to each component's scale too.
    With ``jac=Gradient(fun)`` each point's value and gradient come from
    one pass (:class:`Gradient`, #593).

    ``callback``, where given, is called with each iterate (unscaled),
    and may end the search by raising ``StopIteration``.
    """
    x0 = np.asarray(x0, dtype=float)
    scale = np.maximum(np.abs(x0), np.asarray(floor, dtype=float))

    if obj_scale is None:
        f0 = float(fun(x0, *args))
        divisor = max(abs(f0), 1.0) if np.isfinite(f0) else 1.0
    else:
        divisor = float(obj_scale)

    def scaled_fun(v: npt.NDArray, *inner: Any) -> Any:
        return fun(scale * v, *inner) / divisor

    def scaled_jac(v: npt.NDArray, *inner: Any) -> Any:
        assert jac is not None
        return (
            scale * np.asarray(jac(scale * v, *inner), dtype=float)
        ) / divisor

    opts = dict(options or {})
    opts["gtol"] = 1e-6
    extra: dict = {}
    if callback is not None:

        def unscaled(intermediate_result: OptimizeResult) -> None:
            callback(intermediate_result.x * scale)

        extra["callback"] = unscaled

    if isinstance(jac, Gradient) and jac.fun is fun:
        # The value and the gradient from one pass (#593)
        gradient = jac

        def scaled_value_and_grad(v: npt.NDArray, *inner: Any) -> tuple:
            value, grad = gradient.value_and_grad(scale * v, *inner)
            scaled = (scale * np.asarray(grad, dtype=float)) / divisor
            return value / divisor, scaled

        res = minimize(
            scaled_value_and_grad,
            x0 / scale,
            args=args,
            method="BFGS",
            jac=True,
            options=opts,
            **extra,
        )
    else:
        res = minimize(
            scaled_fun,
            x0 / scale,
            args=args,
            method="BFGS",
            jac=None if jac is None else scaled_jac,
            options=opts,
            **extra,
        )
    res.x = res.x * scale
    res.fun = res.fun * divisor
    return res


def _dead_branch_safe_exp(x: npt.NDArray) -> Any:
    """``exp(x)`` for the ``x < 0`` half, with the other half clamped.

    ``np.where`` picks the right value but autograd evaluates *both*
    branches, so ``exp(x)`` is taped even where ``x + 1`` is the one
    selected. Above x = 709.78 that overflows to inf, and the inf then
    poisons the derivative of the branch that *was* selected, turning
    the parameter transform's jacobian -- and with it every confidence
    bound -- into nan.

    Clamping the argument to the half this branch is responsible for
    leaves it untouched where it is used and bounded where it is not.
    """
    return np.exp(np.minimum(x, 0.0))


def adj_relu(x: npt.NDArray) -> Any:
    return np.where(x >= 0, x + 1, _dead_branch_safe_exp(x))


def inv_adj_relu(x: npt.NDArray) -> Any:
    # No clamp needed here, unlike ``adj_relu``: this dead branch is
    # ``x >= 1``, which is precisely where ``log`` is best behaved.
    return np.where(x >= 1, x - 1, np.log(x))


def rev_adj_relu(x: npt.NDArray) -> Any:
    return -np.where(x >= 0, x + 1, _dead_branch_safe_exp(x))


def inv_rev_adj_relu(x: npt.NDArray) -> Any:
    return np.where(x < -1, -x - 1, np.log(-x))


class ParameterMap:
    """One parameter's map to the unbounded search space and back.

    ``forward`` maps a parameter to the search space and ``inverse``
    back (see ``add_to_funcs`` for the maps). An object rather than a
    pair of closures so a fitted model, whose ``fitting_info`` keeps the
    maps, can be pickled (#573).
    """

    def __init__(
        self, low: float | None, upp: float | None, unit: float = 1.0
    ) -> None:
        self.low = low
        self.upp = upp
        self.unit = unit

    def forward(self, x: Any) -> Any:
        low, upp, unit = self.low, self.upp, self.unit
        if (low is None) and (upp is None):
            return x
        elif (low == 0) and (upp == 1):
            D = 10
            return D * np.arctanh((2 * x) - 1)
        elif (low is not None) and (upp is not None):
            D = 10
            lo, width = float(low), float(upp) - float(low)
            return D * np.arctanh((2 * (x - lo) / width) - 1)
        elif upp is None:
            return inv_adj_relu((x - np.copy(low)) / unit)
        else:
            return inv_rev_adj_relu((x - np.copy(upp)) / unit)

    def inverse(self, x: Any) -> Any:
        low, upp, unit = self.low, self.upp, self.unit
        if (low is None) and (upp is None):
            return x
        elif (low == 0) and (upp == 1):
            D = 10
            return (np.tanh(x / D) + 1) / 2
        elif (low is not None) and (upp is not None):
            D = 10
            lo, width = float(low), float(upp) - float(low)
            return lo + width * (np.tanh(x / D) + 1) / 2
        elif upp is None:
            return unit * adj_relu(x) + np.copy(low)
        else:
            return np.copy(upp) + unit * rev_adj_relu(x)


def add_to_funcs(
    low: float | None,
    upp: float | None,
    i: int,
    funcs: list[Callable[..., Any]],
    inv_f: list[Callable[..., Any]],
    unit: float = 1.0,
) -> None:
    """Append the map of one parameter to the unbounded search space, and
    its inverse.

    An unbounded parameter is searched as itself. One between 0 and 1 is
    searched as ``10 * arctanh(2x - 1)``, and one in any other finite
    interval by the same map on ``(x - low) / (upp - low)``. A parameter
    with one bound is searched as the log of its distance from the bound
    where that distance is below ``unit``, and linearly beyond it
    (``adj_relu``). ``unit`` is 1 unless the caller passes one: see
    ``bounds_convert``.
    """
    mapping = ParameterMap(low, upp, unit)
    funcs.append(mapping.forward)
    inv_f.append(mapping.inverse)


class EachParameter:
    """Apply one map per parameter: ``bounds_convert``'s transforms.

    A module-level callable rather than a closure, so it pickles
    (#573)."""

    def __init__(self, funcs: list[Callable[..., Any]]) -> None:
        self.funcs = funcs

    def __call__(self, params: npt.NDArray) -> Any:
        return np.array(
            [f(p) for p, f in zip(params, self.funcs, strict=True)]
        )


class HoldFixed:
    """Insert the fixed parameters, mapped to the search space, among the
    free ones: ``bounds_convert``'s ``const`` when some are fixed."""

    def __init__(
        self,
        n_params: int,
        fixed: dict[str, float],
        param_map: dict[str, int],
        forward: list[Callable[..., Any]],
        not_fixed: npt.NDArray,
    ) -> None:
        self.n_params = n_params
        self.fixed = fixed
        self.param_map = param_map
        self.forward = forward
        self.not_fixed = not_fixed

    def __call__(self, p: npt.NDArray) -> Any:
        params: list[Any] = [0] * (self.n_params)
        for k, v in self.fixed.items():
            params[self.param_map[k]] = self.forward[self.param_map[k]](v)
        for i, v in zip(self.not_fixed, p):
            params[i] = v
        return np.array(params)


def identity(x: Any) -> Any:
    """``x`` itself: the map of a parameter searched as it is, and
    ``bounds_convert``'s ``const`` when nothing is fixed."""
    return x


def bounds_convert(
    x: "npt.ArrayLike | None",
    bounds: Sequence[tuple[float | None, float | None]],
    fixed: dict[str, float] | None,
    param_map: dict[str, int],
    units: Sequence[float] | None = None,
) -> tuple[Any, ...]:
    """
    This function is used to transform the parameters from the bounded
    parameter space to the unbounded parameter space. This is an improvement
    over using the scipy.optimize.minimize function's bounds parameter as
    it allows us to avoid the use of the constrained optimization methods.

    ``units`` gives, per parameter, the distance from a one-sided bound
    at which its map switches from logarithmic to linear (1 for all of
    them by default; see ``add_to_funcs``). With a unit of 1 the switch
    is at a fixed value, so a parameter measured in the data's units --
    an offset's distance below the first observation, a scale -- is
    searched as a log for data in thousandths and linearly for data in
    thousands: a different search at every scale. The parametric fits
    pass each parameter's own starting distance from its bound instead
    (see ``_search_units`` in ``optimised_fit``), which makes the
    search the same whatever units the data is in.

    The maps returned are module-level objects, not closures, so a model
    that keeps them pickles (#573).
    """
    bounded_to_unbounded_transforms: list[Callable[..., Any]] = []
    unbounded_to_bounded_transforms: list[Callable[..., Any]] = []

    for i, (lower, upper) in enumerate(bounds):
        add_to_funcs(
            lower,
            upper,
            i,
            bounded_to_unbounded_transforms,
            unbounded_to_bounded_transforms,
            1.0 if units is None else float(units[i]),
        )

    transform_params_to_unbounded = EachParameter(
        bounded_to_unbounded_transforms
    )
    transform_unbounded_value_to_params = EachParameter(
        unbounded_to_bounded_transforms
    )

    n_params = len(param_map)

    const: Callable[..., Any]
    if fixed is not None:
        fixed_idx = [param_map[x] for x in fixed.keys()]
        not_fixed = [x for x in range(n_params) if x not in fixed_idx]
        not_fixed = np.array(not_fixed, dtype=int)
        const = HoldFixed(
            n_params,
            fixed,
            param_map,
            bounded_to_unbounded_transforms,
            not_fixed,
        )
    else:
        const = identity
        fixed_idx = []
        not_fixed = np.array([x for x in range(n_params)])

    return (
        transform_params_to_unbounded,
        transform_unbounded_value_to_params,
        const,
        fixed_idx,
        not_fixed,
    )
