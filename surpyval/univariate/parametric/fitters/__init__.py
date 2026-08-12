from typing import Any, Callable, Sequence
import numpy.typing as npt
from scipy.optimize import minimize

from surpyval import np


def fallback_minimize(
    fun: Callable[..., Any],
    init: npt.NDArray,
    args: tuple[Any, ...],
    jac: Callable[..., Any] | None,
    hess: Callable[..., Any] | None,
    newton_tol: float | None = None,
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
    """
    assert jac is not None and hess is not None
    with np.errstate(all="ignore"):
        res = minimize(fun, init, method="BFGS", jac=jac, args=args)

        failed = (
            (res.success is False)
            or np.isnan(res.x).any()
            or (not np.isfinite(res.fun))
        )
        if failed and np.any(hess(np.array(init, dtype=float), *args)):
            newton = minimize(
                fun,
                init,
                method="Newton-CG",
                jac=jac,
                hess=hess,
                tol=newton_tol,
                args=args,
            )
            if newton.success and np.isfinite(newton.fun):
                res = newton

        if (res.success is False) or (np.isnan(res.x).any()):
            res = minimize(fun, init, args=args)

    return res


def preconditioned_bfgs(
    fun: Callable[..., Any],
    x0: npt.NDArray,
    args: tuple[Any, ...] = (),
    jac: Callable[..., Any] | None = None,
    options: dict[str, Any] | None = None,
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

        s = max(|u0|, 1),   v = u / s,   g(v) = f(s v)

    the starting point is order 1 in every component whatever units the
    data is in, and since ``dg/dv = s * df/du`` -- ``s`` growing like the
    scale exactly as ``df/du`` shrinks -- the gradient the optimiser
    tests is order 1 too. scipy's own default then means the same thing
    at every scale, so no tolerance is passed at all.

    Dividing through by ``|f(x0)|`` does the same job for the other
    scale: the objective is a sum over observations, so its gradient
    grows like ``n`` even when the data magnitude is fixed.

    The mapping is linear, diagonal and fixed before the search begins,
    so it cannot move the optimum; it changes the route taken and the
    units of the convergence test, nothing else. ``res.x`` is mapped
    back before returning, so every caller -- including the covariance
    step, which builds its own hessian at the returned point -- sees
    exactly what it saw before. scipy's own ``res.hess_inv`` would be in
    scaled units, and is not used anywhere.
    """
    x0 = np.asarray(x0, dtype=float)
    scale = np.maximum(np.abs(x0), 1.0)

    f0 = float(fun(x0, *args))
    obj_scale = max(abs(f0), 1.0) if np.isfinite(f0) else 1.0

    def scaled_fun(v: npt.NDArray, *inner: Any) -> Any:
        return fun(scale * v, *inner) / obj_scale

    assert jac is not None

    def scaled_jac(v: npt.NDArray, *inner: Any) -> Any:
        return (
            scale * np.asarray(jac(scale * v, *inner), dtype=float)
        ) / obj_scale

    opts = dict(options or {})
    opts["gtol"] = 1e-6

    res = minimize(
        scaled_fun,
        x0 / scale,
        args=args,
        method="BFGS",
        jac=None if jac is None else scaled_jac,
        options=opts,
    )
    res.x = res.x * scale
    res.fun = res.fun * obj_scale
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


def add_to_funcs(
    low: float | None,
    upp: float | None,
    i: int,
    funcs: list[Callable[..., Any]],
    inv_f: list[Callable[..., Any]],
) -> None:
    if (low is None) and (upp is None):
        funcs.append(lambda x: x)
        inv_f.append(lambda x: x)
    elif (low == 0) and (upp == 1):
        D = 10
        funcs.append(lambda x: D * np.arctanh((2 * x) - 1))
        inv_f.append(lambda x: (np.tanh(x / D) + 1) / 2)
    elif upp is None:
        funcs.append(lambda x: (inv_adj_relu(x - np.copy(low))))
        inv_f.append(lambda x: (adj_relu(x) + np.copy(low)))
    elif low is None:
        funcs.append(lambda x: inv_rev_adj_relu(x - np.copy(upp)))
        inv_f.append(lambda x: np.copy(upp) + rev_adj_relu(x))
    else:
        funcs.append(lambda x: x)
        inv_f.append(lambda x: x)


def bounds_convert(
    x: npt.ArrayLike,
    bounds: Sequence[tuple[float | None, float | None]],
    fixed: dict[str, float] | None,
    param_map: dict[str, int],
) -> tuple[Any, ...]:
    """
    This function is used to transform the parameters from the bounded
    parameter space to the unbounded parameter space. This is an improvement
    over using the scipy.optimize.minimize function's bounds parameter as
    it allows us to avoid the use of the constrained optimization methods.
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
        )

    def transform_params_to_unbounded(params: npt.NDArray) -> Any:
        return np.array(
            [
                f(p)
                for p, f in zip(
                    params, bounded_to_unbounded_transforms, strict=True
                )
            ]
        )

    def transform_unbounded_value_to_params(params: npt.NDArray) -> Any:
        return np.array(
            [
                f(p)
                for p, f in zip(
                    params, unbounded_to_bounded_transforms, strict=True
                )
            ]
        )

    n_params = len(param_map)

    if fixed is not None:
        fixed_idx = [param_map[x] for x in fixed.keys()]
        not_fixed = [x for x in range(n_params) if x not in fixed_idx]
        not_fixed = np.array(not_fixed, dtype=int)

        def constraints(p: npt.NDArray) -> Any:
            params = [0] * (n_params)
            for k, v in fixed.items():
                params[param_map[k]] = bounded_to_unbounded_transforms[
                    param_map[k]
                ](v)
            for i, v in zip(not_fixed, p):
                params[i] = v
            return np.array(params)

        const: Callable[..., Any] = constraints
    else:

        def const(x: npt.NDArray) -> Any:
            return x

        fixed_idx = []
        not_fixed = np.array([x for x in range(n_params)])

    return (
        transform_params_to_unbounded,
        transform_unbounded_value_to_params,
        const,
        fixed_idx,
        not_fixed,
    )
