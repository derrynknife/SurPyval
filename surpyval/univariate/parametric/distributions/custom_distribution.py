import inspect
import itertools
import types
import warnings
from typing import Callable

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from autograd import elementwise_grad
from scipy.optimize import brentq

from surpyval.univariate.parametric._fit_inputs import _offset_start
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.deprecation import renamed_arguments
from surpyval.utils.numeric import solve_bracketed
from surpyval.utils.surpyval_data import SurpyvalData

# The quantiles at which CustomDistribution.moment splits its integrals,
# so each piece is on the distribution's own scale.
_MOMENT_BREAKS = onp.array(
    [1e-6, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1 - 1e-6]
)

# Every CustomDistribution constructed in this session, by name, so that a
# model saved with ``to_dict`` can be read back: its cumulative hazard is
# a user function, which a dictionary cannot hold. Constructing a
# distribution again under the same name replaces the entry.
_REGISTRY: "dict[str, CustomDistribution]" = {}


def registered_custom(name: str) -> "CustomDistribution | None":
    """The ``CustomDistribution`` most recently constructed under
    ``name`` in this session, or ``None``."""
    return _REGISTRY.get(name)


def _model_attribute_names() -> frozenset[str]:
    """
    Every attribute name a fitted ``Parametric`` model can carry.

    A fit exposes each parameter as an attribute of the model
    (``model.alpha``), so a parameter named after one of the model's own
    attributes overwrote it: a parameter called ``k`` became the model's
    parameter count (AIC 290.08 for 289.50), and ``dist``, ``data``,
    ``lfp``, ``zi`` or ``method`` broke ``sf``, ``bic`` and ``cb``
    (#437). The names are read off the class -- its methods and
    properties, the attributes it declares for the fitters to fill in, and
    what its constructor sets for every combination of offset,
    limited-failure and zero-inflation -- so a new attribute is covered
    without a list to keep in step. ``res`` and ``log_likelihood`` are the
    two a fitter sets that the class does not declare.

    ``p`` is not among them: a distribution may have its own ``p`` (the
    limited-failure proportion is then named ``lfp_p``, see
    ``Parametric.__init__``), and the fit leaves the attribute alone.
    """
    from surpyval.univariate.parametric.parametric import Parametric

    names = set(dir(Parametric))
    for klass in Parametric.__mro__:
        names.update(getattr(klass, "__annotations__", {}))
    stub = types.SimpleNamespace(k=0, bounds=(), param_map={})
    for flags in itertools.product((False, True), repeat=3):
        names.update(vars(Parametric(stub, "MLE", None, *flags)))
    names.update({"res", "log_likelihood"})
    names.discard("p")
    return frozenset(names)


def _check_signature(fun: Callable[..., Boxable], k: int) -> None:
    """
    ``fun`` must take the time first and then the parameters, either as a
    star-argument of any name, ``(x, *params)``, or as ``k`` named
    positional arguments, ``(x, lam, shape)``.
    """
    try:
        signature = inspect.signature(fun)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "The cumulative hazard function's signature cannot be read"
        ) from e
    kinds = inspect.Parameter
    positional = [
        p
        for p in signature.parameters.values()
        if p.kind in (kinds.POSITIONAL_ONLY, kinds.POSITIONAL_OR_KEYWORD)
    ]
    star = any(
        p.kind == kinds.VAR_POSITIONAL for p in signature.parameters.values()
    )
    required_keyword = [
        p.name
        for p in signature.parameters.values()
        if p.kind == kinds.KEYWORD_ONLY and p.default is p.empty
    ]
    if star:
        ok = len(positional) == 1
    else:
        required = [p for p in positional if p.default is p.empty]
        ok = len(required) <= 1 + k <= len(positional)
    if not ok or required_keyword:
        raise ValueError(
            "The cumulative hazard function must take the time and then "
            "the parameters, either as '(x, *params)' or as one named "
            f"argument per parameter ({k} here, e.g. '(x, "
            + ", ".join(f"p{i}" for i in range(k))
            + f")'); got '{signature}'"
        )


class CustomDistribution(OptimisedFitMixin, ParametricFitter):
    """
    Used to create a custom distribution using only the cumulative hazard
    function. The cumulative hazard function must be a function of x and
    the parameters. The parameters must be named in the parameter_names and
    the bounds must be specified in the bounds argument. The support
    argument is used to specify the support of the distribution.

    Parameters
    ----------

    name: str
        Name of the distribution. It is also the key under which a saved
        model finds the distribution again (see below), so a second
        ``CustomDistribution`` under a name already used in the session
        replaces the first there, with a ``UserWarning``.

    fun: callable
        Function that returns the cumulative hazard function. It takes the
        time and then the parameters, either as a star-argument of any
        name, ``fun(x, *params)``, or as one named argument per parameter,
        ``fun(x, nu, b)``; anything else raises a ``ValueError``.

    parameter_names: list
        List of parameter names (``param_names``, its name before v0.22,
        is accepted with a ``DeprecationWarning`` until v0.23). A fitted
        model exposes each parameter as an attribute, so ``gamma``,
        ``f0`` and the names of the model's own attributes (``k``,
        ``dist``, ``data``, ``method``, ``sf``, ...) are refused with a
        ``ValueError`` that lists them.

    bounds: list
        List of tuples containing the lower and upper bounds of the
        parameters

    support: tuple
        Tuple containing the lower and upper bounds of the support of the
        distribution

    Examples
    --------

    >>> from autograd import numpy as np
    >>> import surpyval as surv
    >>>
    >>> name = 'Gompertz'
    >>>
    >>> def Hf(x, *params):
    ...     # the Gompertz cumulative hazard nu (e^{b x} - 1), zero at x = 0
    ...     return params[0] * (np.exp(params[1] * x) - 1)
    ...
    >>> parameter_names = ['nu', 'b']
    >>> bounds = ((0, None), (0, None))
    >>> support = (0, np.inf)
    >>> Gompertz = surv.CustomDistribution(
    ...     name, Hf, parameter_names, bounds, support
    ... )
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> model = Gompertz.fit(x)

    A model of a custom distribution can be saved with ``to_dict`` like
    any other, but the dictionary holds only the distribution's *name*:
    the cumulative hazard is a Python function, which a dictionary cannot
    carry. Constructing a ``CustomDistribution`` registers it under its
    name for the rest of the session, and ``from_dict`` reads a saved
    model back through that registry -- so in a new session, construct
    the distribution again (same name, same function) before loading.
    Loading without it raises a ``ValueError`` that says so.

    >>> restored = surv.from_dict(model.to_dict())
    >>> restored.dist is Gompertz
    True
    """

    @renamed_arguments(param_names="parameter_names")
    def __init__(
        self,
        name: str,
        # Validated at runtime to have the signature (x, *params);
        # Callable[..., Boxable] is as close as the type system gets.
        fun: Callable[..., Boxable],
        parameter_names: list[str],
        bounds: tuple[tuple[int | float | None, int | float | None], ...],
        support: tuple[int | float, int | float],
    ) -> None:
        _check_signature(fun, len(parameter_names))

        if len(parameter_names) != len(bounds):
            raise ValueError(
                "parameter_names and bounds must have same length"
            )

        # 'p' is allowed: a limited-failure model of a distribution with
        # its own 'p' names the proportion 'lfp_p' instead (see
        # ``Parametric.__init__``), as for the Geometric.
        if "gamma" in parameter_names:
            detail = "'gamma' reserved parameter name for offset distributions"
            raise ValueError(detail)

        if "f0" in parameter_names:
            detail = (
                "'f0' reserved parameter name for zero"
                "inflated or hurdle models"
            )
            raise ValueError(detail)

        for p_name in parameter_names:
            if hasattr(self, p_name):
                detail = "Can't name a parameter after a function"
                raise ValueError(detail)

        reserved = _model_attribute_names()
        clashes = [p_name for p_name in parameter_names if p_name in reserved]
        if clashes:
            public = sorted(r for r in reserved if not r.startswith("_"))
            raise ValueError(
                f"Parameter name(s) {clashes} are attributes of a fitted "
                "model, which exposes each parameter by name; choose "
                "another name. Reserved: " + ", ".join(public) + " (and "
                "every name starting with an underscore that the model "
                "uses)."
            )

        super().__init__(
            name=name,
            k=len(parameter_names),
            bounds=bounds,
            support=support,
            parameter_names=parameter_names,
            param_map={v: i for i, v in enumerate(parameter_names)},
            plot_x_scale="linear",
            y_ticks=np.linspace(0, 1, 11),
        )
        # Stored, then exposed through real methods below. Assigning
        # over self.Hf and friends stopped being possible once
        # OptimisedFitMixin declared them for its own use: a subclass
        # inherits those declarations, and assigning to an inherited
        # method is an error. Delegating is equivalent -- the previous
        # ``self.Hf = fun`` was an unbound instance attribute, so
        # ``self.Hf(x, *params)`` called ``fun(x, *params)`` either way.
        self._fun = fun
        # A distribution known only through its cumulative hazard has no
        # linearising transform, so there is no probability-plot
        # regression to fit (the identity transforms below only serve to
        # draw ``plot()``). Advertising MPP support sent ``how='MPP'`` on
        # to an ``unpack_rr`` that does not exist, and it died with an
        # AttributeError instead of the usual refusal.
        self.supports_mpp = False
        previous = _REGISTRY.get(name)
        if previous is not None and not (
            previous._fun is fun
            and list(previous.parameter_names) == list(parameter_names)
            and tuple(previous.bounds) == tuple(bounds)
            and tuple(previous.support) == tuple(support)
        ):
            warnings.warn(
                f"A CustomDistribution named '{name}' already exists in this "
                "session; this one replaces it in the registry that "
                "from_dict and from_json use to restore saved models, so a "
                f"model saved from the earlier '{name}' is now restored "
                "with this distribution's cumulative hazard.",
                UserWarning,
                stacklevel=2,
            )
        _REGISTRY[name] = self

    def Hf(self, x: Numeric, *params: Boxable) -> Boxable:
        """
        Cumulative hazard: the user-supplied function ``fun(x, *params)``.
        """
        return self._fun(x, *params)

    def hf(self, x: Numeric, *params: Boxable) -> Boxable:
        """
        Hazard rate, :math:`h(x) = dH(x)/dx`, differentiated from ``Hf``
        with autograd.
        """
        return elementwise_grad(self.Hf)(x, *params)

    def sf(self, x: Numeric, *params: Boxable) -> Boxable:
        """
        Survival function, :math:`R(x) = e^{-H(x)}`.
        """
        return np.exp(-self.Hf(x, *params))

    def ff(self, x: Numeric, *params: Boxable) -> Boxable:
        """
        Failure (CDF) function, :math:`F(x) = 1 - e^{-H(x)}`.
        """
        return -np.expm1(-self.Hf(x, *params))

    def df(self, x: Numeric, *params: Boxable) -> Boxable:
        """
        Density, :math:`f(x) = dF(x)/dx`, differentiated from ``ff`` with
        autograd. Where the survival function is 0 it is 0: the chain
        rule's :math:`e^{-H} dH/dx` is ``0 * inf`` there once ``H`` is
        infinite (#561).
        """
        density = elementwise_grad(self.ff)
        gone = self.sf(x, *params) == 0
        if not np.any(gone):
            return density(x, *params)
        with np.errstate(invalid="ignore"):
            return np.where(gone, 0.0, density(x, *params))

    def _scalar_fn(
        self, fn: Callable[..., Boxable], params: "list[float]"
    ) -> Callable[[float], float]:
        """``fn`` at a single point, as a float. Evaluated at a length-one
        array, since a user's cumulative hazard may index or reduce its
        argument."""

        def f(t: float) -> float:
            value = fn(onp.array([t]), *params)
            return float(onp.asarray(value, dtype=float).ravel()[0])

        return f

    def qf(self, u: Numeric, *params: Boxable) -> Boxable:
        r"""
        Quantile function, found numerically: the ``x`` with
        :math:`H(x) = -\ln(1 - u)`, by bracketing and root finding on the
        cumulative hazard (which is increasing on the support).

        It makes ``random`` available (inverse-transform sampling), and
        gives :meth:`moment` the distribution's own scale. Outside
        :math:`[0, 1]` it is NaN, as for the built-in distributions.

        Every probability is solved at once when the cumulative hazard
        broadcasts: a function that indexes or reduces its argument does
        not, so ``Hf`` on an array must give the same values as on each
        point alone, at a probe of spread points before the solve and at
        the answers after it; otherwise, or if it raises on an array, each
        probability is solved alone, as before (#596).
        """
        theta = [float(p) for p in params]
        lo, hi = float(self.support[0]), float(self.support[1])
        u_arr = onp.asarray(u, dtype=float)
        out = onp.full(u_arr.shape, onp.nan)
        # below 0 the target -log1p(-u) is negative, which the inversion
        # read as the support's lower edge
        valid = (u_arr >= 0.0) & (u_arr <= 1.0)
        with onp.errstate(all="ignore"):
            target = -onp.log1p(-u_arr[valid])
            solved = self._invert_Hf_together(theta, target, lo, hi)
            if solved is None:
                H = self._scalar_fn(self.Hf, theta)
                solved = onp.array(
                    [self._invert_Hf(H, t, lo, hi) for t in target]
                )
        out[valid] = solved
        return out[()]

    def _broadcast_Hf(
        self, theta: "list[float]", x: npt.NDArray
    ) -> "npt.NDArray | None":
        """``Hf`` at the points ``x`` evaluated together, or ``None`` if
        that is not ``Hf`` at each point alone: it raised, or gave an
        array of another shape, or values that differ from the point's own
        (beyond the rounding a vectorised loop may differ by)."""
        H = self._scalar_fn(self.Hf, theta)
        try:
            together = onp.asarray(self.Hf(x, *theta), dtype=float)
        except Exception:
            return None
        if together.shape != x.shape:
            return None
        # Points spread across the array (all of a short one), alone.
        k = onp.unique(onp.linspace(0, x.size - 1, min(x.size, 9)).round())
        k = k.astype(int)
        alone = onp.array([H(t) for t in x[k]])
        if not onp.allclose(
            together[k], alone, rtol=1e-12, atol=0.0, equal_nan=True
        ):
            return None
        return together

    def _invert_Hf_together(
        self, theta: "list[float]", target: npt.NDArray, lo: float, hi: float
    ) -> "npt.NDArray | None":
        """``_invert_Hf`` for every ``target`` at once: the same brackets
        (doubled out from the support's finite edge, or from [-1, 1]),
        then ``solve_bracketed``; ``None`` where ``Hf`` does not broadcast
        (see ``_broadcast_Hf``)."""
        # A probe of distinct points across scales in the support.
        if onp.isfinite(lo) and onp.isfinite(hi):
            probe = lo + (hi - lo) * onp.linspace(0.05, 0.95, 9)
        else:
            steps = 2.0 ** onp.arange(-12.0, 15.0, 3.0)
            if onp.isfinite(lo):
                probe = lo + steps
            elif onp.isfinite(hi):
                probe = hi - steps
            else:
                probe = onp.concatenate([-steps[::-2], steps[::2]])
        if self._broadcast_Hf(theta, probe) is None:
            return None

        def excess(x: npt.NDArray, sel: npt.NDArray) -> npt.NDArray:
            # As in _invert_Hf: a non-finite H keeps its sign as a large
            # finite value.
            value = onp.asarray(self.Hf(x, *theta), dtype=float)
            return onp.nan_to_num(
                value - target[sel], nan=1e300, posinf=1e300, neginf=-1e300
            )

        out = onp.full(target.shape, onp.nan)
        out[target <= 0] = lo
        out[onp.isposinf(target)] = hi
        todo = onp.flatnonzero((target > 0) & onp.isfinite(target))
        if not todo.size:
            return out
        a = onp.full(todo.size, lo)
        b = onp.full(todo.size, hi)
        if not onp.isfinite(lo) or not onp.isfinite(hi):
            if onp.isfinite(lo):
                b = lo + self._doubled(lambda w, s: excess(lo + w, s), todo)
            elif onp.isfinite(hi):
                a = hi - self._doubled(lambda w, s: -excess(hi - w, s), todo)
            else:
                a = -self._doubled(lambda w, s: -excess(-w, s), todo)
                b = self._doubled(excess, todo)
        e_a, e_b = excess(a, todo), excess(b, todo)
        out[todo] = onp.where(e_a >= 0, a, b)
        open_ = (e_a < 0) & (e_b > 0)
        if open_.any():
            sel = todo[open_]
            out[sel] = solve_bracketed(
                lambda x, s: excess(x, sel[s]),
                a[open_],
                b[open_],
                e_a[open_],
                e_b[open_],
                xtol=1e-300,
                rtol=4 * onp.finfo(float).eps,
            )
        # The answers, together and alone: a cumulative hazard that
        # broadcast on the probe but not at this size is solved alone.
        if self._broadcast_Hf(theta, out[todo]) is None:
            return None
        return out

    @staticmethod
    def _doubled(
        below: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
        todo: npt.NDArray,
    ) -> npt.NDArray:
        """For each problem in ``todo``, the width ``w``, doubled from 1,
        at which ``below(w, problems)`` is no longer negative (or 1e300 is
        passed), as ``_invert_Hf``'s bracket loops step."""
        width = onp.ones(todo.size)
        active = onp.arange(todo.size)
        while active.size:
            more = below(width[active], todo[active]) < 0
            more &= width[active] < 1e300
            active = active[more]
            width[active] *= 2.0
        return width

    @staticmethod
    def _invert_Hf(
        H: Callable[[float], float], target: float, lo: float, hi: float
    ) -> float:
        """The point of ``[lo, hi]`` where the increasing ``H`` reaches
        ``target``, by doubling a bracket out from the support's finite
        edge (or from [-1, 1] on the whole line) and then ``brentq``."""
        if onp.isnan(target):
            return onp.nan
        if target <= 0:
            return lo
        if onp.isinf(target):
            return hi

        def excess(x: float) -> float:
            # A non-finite H (overflow past the tail, a log at an edge)
            # keeps its sign as a large finite value, which the bracket
            # tests and brentq's bisection steps can use.
            value = H(x) - target
            if onp.isnan(value) or value == onp.inf:
                return 1e300
            return -1e300 if value == -onp.inf else value

        # A finite edge anchors the bracket and the other end moves out
        # geometrically, so it reaches any scale in a few dozen steps;
        # brentq then closes in to relative precision.
        if onp.isfinite(lo) and onp.isfinite(hi):
            a, b = lo, hi
        elif onp.isfinite(lo):
            a, width = lo, 1.0
            while excess(lo + width) < 0 and width < 1e300:
                width *= 2.0
            b = lo + width
        elif onp.isfinite(hi):
            b, width = hi, 1.0
            while excess(hi - width) > 0 and width < 1e300:
                width *= 2.0
            a = hi - width
        else:
            a, b = -1.0, 1.0
            while excess(a) > 0 and a > -1e300:
                a *= 2.0
            while excess(b) < 0 and b < 1e300:
                b *= 2.0
        if excess(a) >= 0:
            return a
        if excess(b) <= 0:
            return b
        return float(
            brentq(excess, a, b, xtol=1e-300, rtol=4 * onp.finfo(float).eps)
        )

    def moment(self, m: int, *params: Boxable) -> Boxable:
        r"""
        The ``m``-th raw moment, integrated from the survival function:

        .. math::
            E[X^m] = \int_0^\infty m x^{m-1} R(x)\,dx
                   - \int_{-\infty}^0 m x^{m-1} F(x)\,dx ,

        restricted to the declared support (``R = 1`` below it and
        ``F = 1`` above it contribute in closed form).

        The generic fallback integrates ``x**m * df(x)`` instead, and
        ``df`` here is autograd's derivative of ``ff``: in the far tail the
        cumulative hazard overflows, the derivative of ``exp(-inf)`` is
        ``0 * inf``, and one nan made every moment -- and so ``var()`` --
        nan. ``R`` and ``F`` themselves underflow cleanly to 0 and 1, and
        need no derivative at all.

        The integrals are split at quantiles of the distribution (see
        :meth:`qf`), so the quadrature works on the distribution's own
        scale. Integrating from 0 to infinity in one piece missed the mass
        of a distribution far from unit scale: a Weibull-like cumulative
        hazard with a scale of 1e5 gave a negative mean.
        """
        from scipy.integrate import quad

        if m == 0:
            return 1.0
        theta = [float(p) for p in params]
        lo, hi = float(self.support[0]), float(self.support[1])
        sf = self._scalar_fn(self.sf, theta)
        ff = self._scalar_fn(self.ff, theta)
        q = onp.asarray(self.qf(_MOMENT_BREAKS, *theta), dtype=float)
        q = q[onp.isfinite(q)]
        # The spread fixes where an infinite tail is split further, and the
        # magnitude the absolute tolerance: quad's default of 1.5e-8 is
        # coarse for a distribution at a scale of 1e-4.
        spread = float(onp.ptp(q)) if q.size > 1 else 1.0
        spread = spread if spread > 0 else 1.0
        magnitude = max(float(onp.max(onp.abs(q))) if q.size else 1.0, spread)
        epsabs = 1e-13 * magnitude**m

        def pieces(a: float, b: float) -> "list[tuple[float, float]]":
            inner = {float(x) for x in q if a < x < b}
            # An infinite end beyond every quantile break is split at
            # growing multiples of the spread, so the tail too is
            # integrated on the distribution's scale.
            if onp.isinf(b):
                last = max(inner, default=a)
                inner |= {last + spread * 4.0**k for k in range(6)}
            if onp.isinf(a):
                first = min(inner, default=b)
                inner |= {first - spread * 4.0**k for k in range(6)}
            edges = [a, *sorted(inner), b]
            return list(zip(edges[:-1], edges[1:]))

        total = 0.0
        with onp.errstate(all="ignore"):
            if hi > 0:
                start = max(lo, 0.0)
                # R = 1 on (0, lo) for a support starting above zero
                total += start**m
                for a, b in pieces(start, hi):
                    total += quad(
                        lambda t: m * t ** (m - 1) * sf(t),
                        a,
                        b,
                        limit=200,
                        epsabs=epsabs,
                    )[0]
            if lo < 0:
                end = min(hi, 0.0)
                # F = 1 on (hi, 0) for a support ending below zero
                total += end**m if hi < 0 else 0.0
                for a, b in pieces(lo, end):
                    total -= quad(
                        lambda t: m * t ** (m - 1) * ff(t),
                        a,
                        b,
                        limit=200,
                        epsabs=epsabs,
                    )[0]
        return total

    def mean(self, *params: Boxable) -> Boxable:
        """The mean, the first raw moment (see :meth:`moment`)."""
        return self.moment(1, *params)

    # Returns a list, where Weibull returns a tuple and the discrete
    # distributions return an array. The base contract does not pin
    # this down; callers coerce whichever they get.
    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        """
        A starting point for the fit, chosen by likelihood from a coarse
        grid of magnitudes.

        A custom distribution knows nothing about its parameters beyond
        their bounds, and a fixed start (1 for a positive parameter) can sit
        so far from the data's scale that the likelihood is flat to machine
        precision there -- a mortality rate of 1e-4 started at 1 gives
        ``exp(1 * 70)`` terms -- and the optimiser stops at once, reporting
        success. So each parameter gets a handful of candidate values
        spanning many orders of magnitude (and the data's own scale), and
        the combination with the best log-likelihood is the start (the
        fixed default is tried as a second start, see
        :meth:`_alternative_base_starts`). With an offset the returned
        vector leads with the offset.
        """
        if offset:
            # The grid's best for the data shifted by the starting offset
            # (``_offset_seed``); it was the unshifted data's (#622)
            return self._offset_seed(data)
        x = np.asarray(data.x, dtype=float)
        finite = np.abs(x[np.isfinite(x)])
        positive = finite[finite > 0]
        scale = float(np.median(positive)) if positive.size else 1.0

        grids = [
            self._start_candidates(low, high, scale)
            for low, high in self.bounds
        ]
        default = [g[0] for g in grids]

        def neg_ll(params: "list[float]") -> float:
            with np.errstate(all="ignore"):
                try:
                    value = float(
                        self._neg_ll_func(data, *params, 0.0, 0.0, 1.0)
                    )
                except (ValueError, FloatingPointError, OverflowError):
                    return np.inf
            return value if np.isfinite(value) else np.inf

        n_combinations = int(np.prod([len(g) for g in grids]))
        best: "list[float]" = list(default)
        best_value = neg_ll(best)
        if n_combinations <= 512:
            for candidate in itertools.product(*grids):
                value = neg_ll(list(candidate))
                if value < best_value:
                    best, best_value = list(candidate), value
        else:
            # too many to enumerate: coordinate-wise sweeps from the default
            for _ in range(3):
                for k, grid in enumerate(grids):
                    for value_k in grid:
                        trial = list(best)
                        trial[k] = value_k
                        value = neg_ll(trial)
                        if value < best_value:
                            best, best_value = trial, value

        return np.array(best, dtype=float)

    def _alternative_base_starts(
        self, data: SurpyvalData, offset: bool = False
    ) -> "list[npt.NDArray]":
        """
        The fixed default start (1 above a lower bound, the midpoint of a
        finite interval, 0 when unbounded) is also tried: the grid's best
        starting likelihood can sit on a plateau -- a spline knot below
        every data point, say -- that the optimiser cannot leave.
        """
        fixed = np.array(
            [self._start_candidates(lo, hi, 1.0)[0] for lo, hi in self.bounds],
            dtype=float,
        )
        if offset:
            fixed = np.concatenate([[_offset_start(data.x)], fixed])
        return [fixed]

    @staticmethod
    def _start_candidates(
        low: "float | None", high: "float | None", scale: float
    ) -> "list[float]":
        """Candidate starting values for one parameter within its bounds;
        the first is the old fixed default (1 above a lower bound, the
        midpoint of a finite interval, 0 when unbounded)."""
        magnitudes = [1.0, 1e-6, 1e-4, 1e-2, 1e2, 1.0 / scale, scale]
        if low is None and high is None:
            return [0.0] + [m for m in (1.0, -1.0, scale, -scale)]
        if high is None:
            assert low is not None
            return [float(low) + m for m in magnitudes]
        if low is None:
            return [float(high) - m for m in magnitudes]
        lo, hi = float(low), float(high)
        return [lo + (hi - lo) * f for f in (0.5, 0.1, 0.9, 0.01, 0.99)]

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return y

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return y

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return x
