from __future__ import annotations

import functools
from math import comb
from typing import TYPE_CHECKING, Any, Callable

import autograd.numpy as np
import numpy.typing as npt
from autograd.numpy.numpy_boxes import ArrayBox

from surpyval.utils.dataframe import UnivariateDataFrameMixin
from surpyval.utils.deprecation import RenamedAttribute, renamed_arguments
from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.validation import _check_x_not_empty

# The estimation machinery lives in ``optimised_fit`` and ``_fit_inputs``;
# its public names are importable from here as they always were.
from ._fit_inputs import (  # noqa: F401
    PARA_METHODS,
    OutsideSupportError,
    normalise_how,
)
from .optimised_fit import METHOD_FUNC_DICT, OptimisedFitMixin  # noqa: F401
from .parametric import Parametric, uniform_draws

# The two types a distribution function deals in. They are separate
# because only one of them can be an autograd box.
#
# ``Numeric`` is what the function is evaluated *at* -- an array of
# times, or of probabilities for ``qf``, or a single one. It is always
# real data.
#
# ``Boxable`` is a parameter, or a value computed from one. The third
# member is the point of it: maximum likelihood differentiates these
# functions, and autograd substitutes its own ``ArrayBox`` for the
# parameters to carry the derivative through, so a parameter really is
# one of three things and not two. The runtime types were established by
# instrumenting a real fit, which is the only place they are visible --
# a fit passes float64 while evaluating the likelihood and ArrayBox
# while differentiating it.
#
# Naming the box rather than writing ``Any`` is what makes the parameter
# positions checkable at all; under ``Any`` they accept anything, which
# ``disallow_untyped_defs`` would then certify. It also rules out the
# two narrowings that look right and are not:
#
#   alpha: npt.ArrayLike        25 errors on ``(x / alpha) ** beta``,
#                               because array-like covers str and bytes.
#                               The fix that clears them, np.asarray,
#                               wraps the box in an object array: the
#                               value stays right and the derivative
#                               does not. A plain product then returns a
#                               zero gradient with no exception, which
#                               an optimiser reads as "this parameter
#                               does not affect the likelihood", so the
#                               fit leaves it at its initial guess and
#                               reports success.
#
#   alpha: npt.NDArray | float  No errors at all, and false. Nothing in
#                               the toolchain would ever say so, and
#                               py.typed publishes it to every caller.
#
# Neither is a reason to reach for ``np.asarray`` here. That convention
# belongs to the non-parametric packages, where the values are real
# data; in this one it destroys the thing being computed.
Numeric = npt.NDArray | float
Boxable = npt.NDArray | float | ArrayBox


def reject_structural_params(
    dist_name: str,
    gamma: Any = None,
    p: Any = None,
    f0: Any = None,
) -> None:
    """Raise for structural arguments a closed-form distribution has no
    meaning for.

    ``ParametricFitter.from_params`` takes ``gamma`` (an offset), ``p``
    (the proportion that never fails) and ``f0`` (the proportion failing
    at time zero). ``Bernoulli``, ``Binomial`` and ``ExactEventTime``
    support none of them, but they accept the arguments anyway so their
    signatures match the base -- a subclass that silently dropped them
    could not be called through a ``ParametricFitter`` reference, which
    is what the earlier narrower signatures got wrong.
    """
    for name, value in (("gamma", gamma), ("p", p), ("f0", f0)):
        if value is not None:
            raise ValueError(
                f"{dist_name} does not support '{name}'; it has a "
                f"closed-form estimator with no offset, limited failure "
                f"population or zero inflation."
            )


# What each distribution function is outside the support of a continuous
# distribution: (below its lower edge, above its upper edge). Nothing has
# failed before the support starts and everything has after it ends, so
# the density is 0 on both sides, the hazard 0 below and infinite above
# (its limit at a finite upper edge, where H is infinite too).
_OUTSIDE_SUPPORT: dict[str, tuple[float, float]] = {
    "sf": (1.0, 0.0),
    "ff": (0.0, 1.0),
    "df": (0.0, 0.0),
    "hf": (0.0, np.inf),
    "Hf": (0.0, np.inf),
    "log_df": (-np.inf, -np.inf),
    "log_sf": (0.0, -np.inf),
    "log_ff": (-np.inf, 0.0),
}


#: The smallest normal float: a window probability below it has lost its
#: digits to underflow (see ``ll_interval_or_truncated``).
_TINY = float(np.finfo(float).tiny)


def _raw(value: Any) -> Any:
    """``value`` with any autograd box removed (a plain float or array)."""
    while isinstance(value, ArrayBox):
        value = value._value
    return value


def _support_guarded(
    fn: Callable[..., Any], below: float, above: float
) -> Callable[..., Any]:
    """Wrap a continuous distribution's function so that it returns its
    limiting value outside the support rather than evaluating the formula
    there.

    The formulas do not know where their support is: a Weibull
    ``sf(-1)`` came back as 0.99, ``df(-1)`` complex, an Exponential
    ``sf(-1)`` above one and a Beta ``df(1.5)`` as 4.5. The points
    outside are evaluated at a point inside, then overwritten, so the
    discarded branch is finite and cannot put a nan into an autograd
    gradient. Data inside the support takes the unwrapped path untouched.
    """

    @functools.wraps(fn)
    def guarded(self: "ParametricFitter", x: Any, *params: Any) -> Any:
        # A boxed x is autograd differentiating with respect to it
        # (CustomDistribution's hf and df); the outer call is guarded.
        if self.discrete or isinstance(x, ArrayBox):
            return fn(self, x, *params)
        lo, hi = self._support_edges(*params)
        x_arr = np.asarray(x, dtype=float)
        is_below = x_arr < lo
        is_above = x_arr > hi
        if not (np.any(is_below) or np.any(is_above)):
            return fn(self, x, *params)
        if np.isfinite(lo) and np.isfinite(hi):
            inside = 0.5 * (lo + hi)
        elif np.isfinite(lo):
            inside = lo + 1.0
        else:
            inside = hi - 1.0
        out = fn(self, np.where(is_below | is_above, inside, x_arr), *params)
        out = np.where(is_below, below, np.where(is_above, above, out))
        return out[()] if isinstance(out, np.ndarray) else out

    guarded._support_guarded = True  # type: ignore[attr-defined]
    return guarded


def _as_array(value: Any) -> Any:
    """A list or tuple as a float array; anything else (a scalar, an array,
    an autograd box) as it is."""
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)
    return value


# The distribution functions of a time ``x`` (and ``qf`` of a probability)
# that ``_array_inputs`` wraps.
_QUERY_FUNCTIONS = tuple(_OUTSIDE_SUPPORT) + ("qf",)

# Each function's limit as x goes to infinity, where every distribution
# has the same one (its value past an upper support edge): all but the
# hazard, whose limit is the family's own.
_AT_INFINITY: dict[str, float] = {
    name: above
    for name, (_, above) in _OUTSIDE_SUPPORT.items()
    if name != "hf"
}


def _array_inputs(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a distribution function so that a list or tuple argument
    becomes an array, and a missing (NaN) query point gives NaN there.

    The formulas are written for arrays: given a Python list, ``list *
    int`` repeated the list before numpy saw it (``Gamma.sf([5, 10], 8,
    3)`` returned six values) and ``list / int`` raised (#424). And a
    missing query is answered as missing (principle 3): a constant
    hazard's ``hf(nan)`` was its rate, a Uniform's ``sf(nan)`` 1 and
    Bernoulli's ``sf(nan)`` raised (#382). The function is evaluated with
    the NaNs replaced by a point it accepts, then NaN is put back, so
    nothing else about the other points changes.

    An overflow or a division by zero inside a formula is the formula
    reaching the infinite limit it is evaluated toward, not something
    the caller needs to hear about (principle 22): far in a Weibull's tail
    ``(x / alpha) ** beta`` overflows to ``inf``, and ``exp(-inf)`` is the
    0 the survival function is, but numpy warned "overflow encountered in
    power" (#561). Those two are not warned about here; an invalid
    operation (``inf - inf``, ``0 * inf``, giving a NaN) still is, since
    its value is wrong.

    A discrete distribution on the integers up to infinity takes its
    limits at ``x = inf`` (``_AT_INFINITY``) rather than evaluating its
    formulas there, where the incomplete gamma and beta functions and
    ``q ** inf`` gave NaN: a Poisson's ``sf(inf)`` was NaN, not 0 (#561).
    The hazard's limit there is the family's own, and is computed.
    """
    at_infinity = _AT_INFINITY.get(fn.__name__)

    @functools.wraps(fn)
    def wrapped(self: "ParametricFitter", x: Any, *params: Any) -> Any:
        with np.errstate(over="ignore", divide="ignore"):
            return evaluate(self, x, *params)

    def evaluate(self: "ParametricFitter", x: Any, *params: Any) -> Any:
        x = _as_array(x)
        params = tuple(_as_array(p) for p in params)
        if isinstance(x, ArrayBox):
            return fn(self, x, *params)
        x_arr = np.asarray(x, dtype=float)
        missing = np.isnan(x_arr)
        top = None
        replaced = missing
        if (
            at_infinity is not None
            and self.discrete
            and self.support[1] == np.inf
        ):
            top = np.isposinf(x_arr)
            replaced = missing | top
        if not np.any(replaced):
            return fn(self, x, *params)
        # A point asked for alongside is one the function accepts; failing
        # that, the middle probability or the support's finite edge.
        known = x_arr[~replaced]
        if known.size:
            fill = float(known[0])
        elif fn.__name__ == "qf":
            fill = 0.5
        else:
            lo, hi = self._support_edges(*params)
            fill = lo if np.isfinite(lo) else (hi if np.isfinite(hi) else 0.0)
        out = fn(self, np.where(replaced, fill, x_arr), *params)
        if top is not None and np.any(top):
            out = np.where(top, at_infinity, out)
        out = np.where(missing, np.nan, out)
        return out[()] if isinstance(out, np.ndarray) else out

    wrapped._array_inputs = True  # type: ignore[attr-defined]
    return wrapped


def _log1mexp(d: Any) -> Any:
    """``log(1 - exp(-d))`` for ``d >= 0``, exact at both ends: through
    ``expm1`` below ``log 2`` and ``log1p`` above it (Maechler, 2012). Each
    branch sees only arguments it is finite on, for autograd's sake; ``d =
    inf`` gives 0."""
    small = d < np.log(2.0)
    d_small = np.where(small, d, 1.0)
    d_large = np.where(small, 1.0, d)
    return np.where(
        small, np.log(-np.expm1(-d_small)), np.log1p(-np.exp(-d_large))
    )


DEFAULT_Y_TICKS = [
    0.0001,
    0.0002,
    0.0003,
    0.001,
    0.002,
    0.003,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
    0.999,
    0.9999,
]


class ParametricFitter(UnivariateDataFrameMixin):
    """
    Base class for all parametric distributions.

    A distribution needs only ``hf`` and ``Hf`` (or ``sf``, ``ff`` and
    ``df``) plus a ``_parameter_initialiser`` with the signature
    ``(self, data: SurpyvalData, offset: bool = False)`` for fitting to
    work; ``log_df``, ``log_sf``, ``log_ff`` and ``random`` have generic
    implementations here that subclasses can override with closed forms.
    Probability plotting (the MPP fit method and ``Parametric.plot``)
    additionally requires ``mpp_x_transform``, ``mpp_y_transform(y,
    *params)`` and ``mpp_inv_y_transform(y, *params)``.

    A subclass can take over an entire estimation method by defining
    ``mpp(x, c, n, heuristic, rr, on_d_is_0, offset)``, returning a
    results dict with at least a ``params`` numpy array.

    A subclass with an exact analytic MLE may also define
    ``_closed_form_mle(data)``, returning the parameter vector, or
    ``None`` when the closed form does not apply to *that* data. It is
    consulted before any initial guess or optimisation, so an eligible
    fit skips both entirely; ``init`` has no effect on that path because
    the closed form is exact. Structural requests (an offset, limited
    failure population, zero inflation, or fixed parameters) bypass it.

    Implementations must do their math with ``surpyval.np``, which is
    ``autograd.numpy``: maximum likelihood estimation differentiates
    through these functions, and plain numpy silently breaks the
    gradients.

    Examples
    --------
    Every continuous distribution (``Weibull``, ``Gamma``, ``LogNormal``
    ...) is an instance of it. Its functions take the parameters
    explicitly; ``fit`` and ``from_params`` return a ``Parametric``
    model that holds them:

    >>> import numpy as np
    >>> from surpyval import Weibull
    >>> from surpyval.univariate.parametric import ParametricFitter
    >>> isinstance(Weibull, ParametricFitter)
    True
    >>> Weibull.sf(np.array([5, 10]), 10, 2).round(4)
    array([0.7788, 0.3679])
    >>> x = np.array([3.1, 4.7, 5.2, 6.8, 7.4, 8.9, 10.2, 12.5])
    >>> model = Weibull.fit(x)
    >>> model.params.round(4)
    array([8.2823, 2.779 ])
    """

    # Whether the distribution's mass sits on integers rather than a
    # continuum. ``DiscreteParametricFitter`` overrides this; fit-method
    # validation and callers branch on the trait.
    discrete = False

    # ``param_names``, the pre-0.22 name of ``parameter_names``, still
    # reads (and sets) it for one release, with a DeprecationWarning.
    param_names = RenamedAttribute("parameter_names")

    if TYPE_CHECKING:
        # The distribution functions every subclass supplies and this
        # base calls -- ``cs`` divides two ``sf``s, ``log_sf`` negates
        # ``Hf``, ``random`` inverts ``qf``, and the four ``ll_*``
        # methods are written in terms of ``hf``, ``Hf`` and the log
        # densities. The docstring above already states the contract
        # ("a distribution needs only hf and Hf, or sf, ff and df");
        # this is the same statement in a form the checker reads.
        #
        # Declared, not defined: a body here would give every
        # distribution a silently wrong inherited implementation
        # instead of the AttributeError that correctly reports a
        # distribution which forgot one. ``OptimisedFitMixin`` carries
        # the mirror image of this block for the estimation machinery.
        def sf(self, x: Any, *params: Any) -> Any: ...
        def ff(self, x: Any, *params: Any) -> Any: ...
        def df(self, x: Any, *params: Any) -> Any: ...
        def hf(self, x: Any, *params: Any) -> Any: ...
        def Hf(self, x: Any, *params: Any) -> Any: ...
        def qf(self, u: Any, *params: Any) -> Any: ...
        def moment(self, m: Any, *params: Any) -> Any: ...
        def mpp_x_transform(self, x: Any) -> Any: ...
        def mpp_y_transform(self, y: Any, *params: Any) -> Any: ...
        def mpp_inv_y_transform(self, y: Any, *params: Any) -> Any: ...

        def _parameter_initialiser(
            self, data: SurpyvalData, offset: bool = False
        ) -> npt.NDArray: ...

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Every continuous distribution's own sf, ff, df, hf, Hf and log
        # functions are guarded outside its support (see
        # ``_support_guarded``). The discrete ones guard their integer
        # supports themselves, as their docstrings describe.
        super().__init_subclass__(**kwargs)
        if not cls.discrete:
            for name, (below, above) in _OUTSIDE_SUPPORT.items():
                fn = cls.__dict__.get(name)
                if callable(fn) and not getattr(fn, "_support_guarded", False):
                    setattr(cls, name, _support_guarded(fn, below, above))
        # Then every distribution's functions take lists and NaNs (see
        # ``_array_inputs``), outermost, so the guard sees an array.
        for name in _QUERY_FUNCTIONS:
            fn = cls.__dict__.get(name)
            if callable(fn) and not getattr(fn, "_array_inputs", False):
                setattr(cls, name, _array_inputs(fn))

    def _support_edges(self, *params: Any) -> tuple[float, float]:
        """The support ``(lower, upper)`` at ``params``: the declared one,
        with a data-dependent (NaN) edge read from the parameter that
        ``support_param_index`` nominates (``a``/``b`` of the Uniform and
        the 4-parameter Beta)."""
        lo, hi = (float(v) for v in self.support)
        if np.isnan(lo):
            lo = float(_raw(params[self.support_param_index[0]]))
        if np.isnan(hi):
            hi = float(_raw(params[self.support_param_index[1]]))
        return lo, hi

    @renamed_arguments(param_names="parameter_names")
    def __init__(
        self,
        name: str,
        k: int,
        bounds: tuple[tuple[int | float | None, int | float | None], ...],
        support: tuple[int | float, int | float],
        parameter_names: list[str],
        param_map: dict[str, int],
        plot_x_scale: str,
        y_ticks: list[float] | None = None,
    ) -> None:
        self.name: str = name
        self.k = k
        self.bounds = bounds
        self.support = support
        self.parameter_names = parameter_names
        self.param_map = param_map
        self.plot_x_scale = plot_x_scale
        self.y_ticks = DEFAULT_Y_TICKS if y_ticks is None else y_ticks
        self.supports_mpp = True
        # For distributions whose support is data-dependent (declared as
        # NaN, e.g. the 4-parameter Beta), these give the indices of the
        # parameters that supply the left and right support bounds once
        # the model is fitted. The default ``(0, 1)`` matches the legacy
        # behaviour used by ``Uniform``.
        self.support_param_index = (0, 1)

    def random(
        self,
        size: int | tuple[int, ...],
        *params: Any,
        random_state: Any = None,
    ) -> Any:
        r"""

        Draws random samples from the distribution in shape `size`, using
        the inverse transform method with the distribution's quantile
        function.

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        params : numpy array or scalar
            The parameters of the distribution
        random_state : int or numpy.random.Generator, optional
            Seed or generator for a reproducible draw of its own, which
            neither depends on nor advances numpy's global stream (an int
            is ``np.random.default_rng(seed)``). ``None`` (the default)
            draws from numpy's global stream, so ``np.random.seed``
            reproduces it.

        Returns
        -------

        random : scalar or numpy array
            Random values drawn from the distribution in shape `size`

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> np.random.seed(1)
        >>> Weibull.random(5, 3, 4)
        array([2.57122697, 3.18730986, 0.31024877, 2.32381059, 1.89352939])
        >>> Weibull.random(3, 3, 4, random_state=1).round(4)
        array([2.7607, 3.9499, 1.8844])
        """
        U = uniform_draws(size, random_state)
        return self.qf(U, *params)

    def log_df(self, x: npt.NDArray, *params: Any) -> Any:
        r"""Log of the density, :math:`\ln f(x) = \ln h(x) - H(x)` (the
        log of the mass :math:`P(T = x)` for a discrete distribution).

        Used by the likelihood; many distributions override it with a
        closed form that stays finite where ``df`` itself underflows.

        Where the cumulative hazard is infinite (far in a tail whose
        hazard grows without bound) it is -inf, not ``inf - inf`` (#561).
        """
        H = self.Hf(x, *params)
        gone = H == np.inf
        if not np.any(gone):
            return np.log(self.hf(x, *params)) - H
        with np.errstate(invalid="ignore"):
            return np.where(gone, -np.inf, np.log(self.hf(x, *params)) - H)

    def log_sf(self, x: Numeric, *params: Any) -> Any:
        r"""Log of the survival function, :math:`\ln R(x) = -H(x)`."""
        return -self.Hf(x, *params)

    def log_ff(self, x: Numeric, *params: Any) -> Any:
        r"""Log of the CDF, :math:`\ln F(x) = \ln(1 - e^{-H(x)})`,
        computed with ``expm1`` so it stays accurate where :math:`F` is
        small."""
        return np.log(-np.expm1(-self.Hf(x, *params)))

    @renamed_arguments(X="given")
    def cs(self, x: Numeric, given: Numeric, *params: Any) -> Any:
        r"""

        Conditional survival function: the probability of surviving a
        further ``x`` given survival to ``given`` already.

        .. math::
            R(x, given) = \frac{R(x + given)}{R(given)}

        This is the definition for every distribution, so it lives here
        rather than being restated on each one. ``Exponential``
        overrides it because the exponential is memoryless and
        :math:`R(x, given) = R(x)`, which is both cheaper and free of the
        cancellation the ratio suffers in the far tail.

        .. versionchanged:: 0.22
           The time already survived is ``given`` (it was ``X``, which
           still works until v0.23 with a ``DeprecationWarning``), the
           name the regression models' ``sf_tvc(..., given=)`` uses.

        Parameters
        ----------

        x : numpy array or scalar
            The additional time to survive, measured from ``given``
        given : numpy array or scalar
            The time already survived
        *params : numpy array like or scalar
            The parameters of the distribution, in the order given by
            its ``parameter_names``

        Returns
        -------

        cs : scalar or numpy array
            The value(s) of the conditional survival function.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.cs(x, 5, 3, 4)
        array([2.52537548e-04, 3.00394073e-10, 2.45288508e-19, 1.48999440e-32,
               5.42544000e-51])
        """
        return self.sf(x + given, *params) / self.sf(given, *params)

    def _plot_x_bounds(self, x: npt.NDArray, params: Any) -> Any:
        """Return (x_scale_min, x_scale_max) for probability plots.

        Returns None to auto-compute the bounds from the data.
        """
        return None

    @_check_x_not_empty
    def ll_observed(self, x: npt.NDArray, n: npt.NDArray, *params: Any) -> Any:
        *dist_params, gamma, f0, p = params
        if f0 == 0:
            # Not zero-inflated; x == 0 is an ordinary observation.
            zero_weight = 0
            non_zero_mask = np.full(x.shape, True)
        else:
            # The zero-inflation mass sits at x == 0 in observed time,
            # so the mask must be taken before the offset shift
            n_zeros = np.sum(n[x == 0])
            zero_weight = n_zeros * np.log(f0) if n_zeros != 0 else 0
            non_zero_mask = x != 0
        x = x - gamma
        N = np.sum(n[non_zero_mask])
        return (
            (
                n[non_zero_mask] * self.log_df(x[non_zero_mask], *dist_params)
            ).sum()
            + zero_weight
            + N * np.log(p - f0)
        )

    @_check_x_not_empty
    def ll_right_censored(
        self, x: npt.NDArray, n: npt.NDArray, *params: Any
    ) -> Any:
        *dist_params, gamma, f0, p = params
        x = x - gamma
        if p == 1:
            return np.sum(n * (np.log1p(-f0) + self.log_sf(x, *dist_params)))
        else:
            F = self.ff(x, *dist_params)
            return np.sum(n * np.log(1 - f0 - (p - f0) * F))

    @_check_x_not_empty
    def ll_left_censored(
        self, x: npt.NDArray, n: npt.NDArray, *params: Any
    ) -> Any:
        *dist_params, gamma, f0, p = params
        x = x - gamma
        if f0 == 0:
            # No zero-inflation: F_mix = p * F, so the numerically stable
            # log_ff path applies (the branch was inverted as ``f0 == 1``,
            # which never occurs, #256).
            return np.sum(n * self.log_ff(x, *dist_params)) + n.sum() * np.log(
                p
            )
        else:
            return np.sum(n * np.log(f0 + (p - f0) * self.ff(x, *dist_params)))

    @_check_x_not_empty
    def ll_interval_or_truncated(
        self,
        xl: npt.NDArray,
        xr: npt.NDArray,
        n: npt.NDArray,
        *params: Any,
    ) -> Any:
        """
        Log probability of falling inside each window ``(xl, xr]``.

        An infinite bound is replaced by its analytic limit rather than
        handed to the CDF. ``np.where`` selects the *value* correctly but
        evaluates both branches, so ``ff(inf)`` was still taped by
        autograd, and its nan derivative then propagated through the
        selection regardless of which side was chosen. The objective was
        right and the gradient was nan.

        The consequence was not subtle. Every singly-truncated fit lost
        all three gradient optimisers -- BFGS and Newton-CG each failed
        after a single evaluation and TNC burned its whole 1000
        evaluation budget -- leaving Nelder-Mead to finish derivative
        free. A Weibull that fits in 0.014s took 1.36s, and a ``tl`` of 0
        (a no-op, since ``F(0) = 0``) cost exactly the same as a real
        truncation, which is what gives the cause away. Windows with
        *both* bounds finite were always fast, because no infinity ever
        reached the tape.

        The infinity is substituted out of the *argument* before the CDF
        sees it, so a single vectorised call covers every row whatever
        its pattern of bounds. The outer ``np.where`` then selects
        between two values that are both already finite, which is safe.

        The stand-in cannot be an arbitrary constant. Zero looks natural
        and is wrong: a Weibull with ``beta < 1`` has an unbounded
        density derivative at the origin, so ``ff(0)`` would swap one nan
        gradient for another. Reusing a bound that is genuinely present
        keeps the stand-in inside the support and at the data's own
        magnitude. Its value never reaches the result -- ``np.where``
        discards it -- only its derivative has to be finite.

        Probabilities come from the mixture CDF
        ``F_mix(t) = f0 + (p - f0) * F0(t)``: with no right bound the
        window probability is the mixture survival ``1 - F_mix(tl)``,
        which includes the never-failing mass ``1 - p``. The old
        ``(p - f0) * (1 - F0(tl))`` form made the LFP plus
        left-truncation likelihood unbounded (#269). For finite-bound
        intervals the ``f0`` terms cancel, so plain fits are unchanged.

        The work is split in two: ``_window_inputs``, which depends on
        the data alone (and on the support), and
        ``_window_log_likelihood``, which evaluates the functions. The
        likelihood-ratio searches keep the first and call the second with
        the functions unwrapped (#602).
        """
        *dist_params, gamma, f0, p = params
        if len(n) == 0:
            return 0.0
        windows = self._window_inputs(xl, xr, gamma, dist_params)
        return self._window_log_likelihood(
            windows,
            n,
            dist_params,
            (gamma, f0, p),
            (self.ff, self.log_sf, self.log_ff),
        )

    def _window_inputs(
        self, xl: npt.NDArray, xr: npt.NDArray, gamma: Any, dist_params: Any
    ) -> tuple:
        """The data's part of :meth:`ll_interval_or_truncated` for the
        windows ``(xl, xr]``: ``(xl, xr, lo_finite, hi_finite,
        lo_evaluated, xl_safe, xr_safe)``, which bounds are finite, which
        lower bounds are evaluated (above the support's lower edge), and
        the bounds with a stand-in where they are not. They depend on the
        parameters only through the support's edges (the Uniform's)."""
        lo_finite = np.isfinite(xl)
        hi_finite = np.isfinite(xr)
        # A lower bound at or below the support's lower edge (a ``tl`` of
        # 0 for a lifetime distribution) is a CDF of exactly 0, so it is
        # taken as 0 rather than evaluated: its value is the same, but
        # the formula's second derivative there is nan (``0 * log 0`` in
        # a Weibull's), which sent the fit's covariance to the numerical
        # Hessian, 45% of a left-truncated Weibull fit at 1e5 rows.
        lo_evaluated = lo_finite
        stand_in = 1.0
        if not self.discrete:
            edge, top = self._support_edges(*dist_params)
            if np.isfinite(edge):
                lo_evaluated = lo_finite & (xl - _raw(gamma) > edge)
                # Inside the support, should no bound be left to stand in
                stand_in = (
                    0.5 * (edge + top) if np.isfinite(top) else edge + 1.0
                ) + float(_raw(gamma))

        # ``xl`` and ``xr`` are data, never traced, so this substitution
        # is invisible to autograd -- it only changes what the CDF is
        # asked to evaluate.
        present = np.concatenate([xl[lo_evaluated], xr[hi_finite]])
        if present.size:
            stand_in = float(present[0])
        xl_safe = np.where(lo_evaluated, xl, stand_in)
        xr_safe = np.where(hi_finite, xr, stand_in)
        return xl, xr, lo_finite, hi_finite, lo_evaluated, xl_safe, xr_safe

    def _window_log_likelihood(
        self,
        windows: tuple,
        n: npt.NDArray,
        dist_params: Any,
        extra: tuple,
        fns: tuple[Callable[..., Any], ...],
    ) -> Any:
        """The log-likelihood of :meth:`ll_interval_or_truncated`, from
        its windows' inputs (``_window_inputs``), the counts ``n``, the
        distribution's parameters, its ``(gamma, f0, p)`` and the
        functions ``(ff, log_sf, log_ff)`` it evaluates, each called as
        ``f(x, *dist_params)``."""
        xl, xr, lo_finite, hi_finite, lo_evaluated, xl_safe, xr_safe = windows
        gamma, f0, p = extra
        ff, log_sf, log_ff = fns
        # The zero-inflation mass ``f0`` sits at 0 in observed time (see
        # ``ll_observed``), so ``F_mix`` includes it only from 0 on: below
        # 0 nothing has failed, and a window opening below 0 contains the
        # mass. Counting it at a ``tl`` of -1 made a no-op truncation
        # divide by ``1 - f0``, and f0 ran to 1 (#548). (With ``f0 = 0``
        # this is the same arithmetic as before.)
        upper = np.where(
            hi_finite,
            f0 * (xr >= 0) + (p - f0) * ff(xr_safe - gamma, *dist_params),
            1.0,
        )
        lower = np.where(
            lo_finite,
            f0 * (xl >= 0)
            + (p - f0)
            * np.where(lo_evaluated, ff(xl_safe - gamma, *dist_params), 0.0),
            0.0,
        )
        window = np.maximum(upper - lower, 0.0)

        # In the upper tail ``F(l) - F(r)`` is a difference of two numbers
        # near 1, and once ``F(l)`` rounds to 1 it is 0: a left-truncated
        # LogNormal at mu = -5 had a log-likelihood of +inf, where it is
        # -23.73 (#412). Where ``F(l) > 1/2`` the window is taken from the
        # survival function in log space instead,
        # ``log S(l) + log(1 - S(r) / S(l))``, exact however small S is.
        # Each form is evaluated only where it is used (a stand-in
        # elsewhere), so the other cannot put a NaN into the gradient.
        plain = self.discrete or _raw(f0) != 0 or _raw(p) != 1
        upper_tail = (
            np.zeros(len(n), dtype=bool)
            if plain
            else lo_finite & (_raw(lower) > 0.5)
        )
        # In the lower tail the difference keeps its digits, but the CDFs
        # themselves underflow: a window below the smallest normal float
        # has lost them, and at 0 its log is -inf (a truncation window
        # whose log is then -inf - -inf, NaN). There it is taken in log
        # space from ``log_ff``, ``log F(r) + log(1 - F(l) / F(r))``
        # (#594). Only such windows: elsewhere nothing changes.
        lower_tail = (
            np.zeros(len(n), dtype=bool)
            if plain
            else hi_finite & ~upper_tail & (_raw(window) < _TINY)
        )
        if np.any(lower_tail):
            return np.sum(
                n
                * self._log_windows(
                    xl_safe - gamma,
                    xr_safe - gamma,
                    (lo_evaluated, hi_finite, upper_tail, lower_tail),
                    window,
                    dist_params,
                    (log_sf, log_ff),
                )
            )
        if not np.any(upper_tail):
            return np.sum(n * np.log(window))
        in_tail = float(xl[upper_tail][0])
        log_sl = log_sf(
            np.where(upper_tail, xl_safe, in_tail) - gamma, *dist_params
        )
        log_sr = np.where(
            upper_tail & hi_finite,
            log_sf(
                np.where(upper_tail & hi_finite, xr_safe, in_tail) - gamma,
                *dist_params,
            ),
            -np.inf,
        )
        tail = log_sl + _log1mexp(np.maximum(log_sl - log_sr, 0.0))
        body = np.log(np.where(upper_tail, 1.0, window))
        return np.sum(n * np.where(upper_tail, tail, body))

    def _log_windows(
        self,
        xl: npt.NDArray,
        xr: npt.NDArray,
        masks: tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray],
        window: Any,
        dist_params: Any,
        fns: tuple[Callable[..., Any], Callable[..., Any]],
    ) -> Any:
        """The log of each window's probability for
        :meth:`ll_interval_or_truncated` where some are in the lower tail:
        ``xl`` and ``xr`` its bounds (stand-ins where absent, the offset
        taken off), ``masks`` its ``(lo_evaluated, hi_finite, upper_tail,
        lower_tail)``, ``window`` the plain differences and ``fns`` the
        ``(log_sf, log_ff)`` to evaluate. The upper tail is taken from
        ``log_sf`` as there, the lower from ``log_ff``, and a lower-tail
        window with no mass at all (``F(r) = 0``) stays ``log 0``."""
        lo_evaluated, hi_finite, upper_tail, lower_tail = masks
        log_sf, log_ff = fns
        # A stand-in for the rows outside the tail, taken off the traced
        # bounds: with an offset they carry ``gamma``, and ``float`` of
        # a traced value raised a TypeError mid-search (#622)
        in_low = float(_raw(xr)[lower_tail][0])
        log_fr = log_ff(np.where(lower_tail, xr, in_low), *dist_params)
        with_l = lower_tail & lo_evaluated
        log_fl = np.where(
            with_l,
            log_ff(np.where(with_l, xl, in_low), *dist_params),
            -np.inf,
        )
        low = lower_tail & np.isfinite(_raw(log_fr))
        d = np.where(low, log_fr - log_fl, 1.0)
        out = np.where(
            low,
            np.where(low, log_fr, 0.0) + _log1mexp(np.maximum(d, 0.0)),
            np.log(np.where(upper_tail | low, 1.0, window)),
        )
        if not np.any(upper_tail):
            return out
        in_tail = float(_raw(xl)[upper_tail][0])
        log_sl = log_sf(np.where(upper_tail, xl, in_tail), *dist_params)
        log_sr = np.where(
            upper_tail & hi_finite,
            log_sf(
                np.where(upper_tail & hi_finite, xr, in_tail), *dist_params
            ),
            -np.inf,
        )
        tail = log_sl + _log1mexp(np.maximum(log_sl - log_sr, 0.0))
        return np.where(upper_tail, tail, out)

    def _log_likelihood(self, data: SurpyvalData, *params: Any) -> Any:
        return (
            self.ll_observed(data.x_o, data.n_o, *params)
            + self.ll_right_censored(data.x_r, data.n_r, *params)
            + self.ll_left_censored(data.x_l, data.n_l, *params)
            + self.ll_interval_or_truncated(
                data.x_il, data.x_ir, data.n_i, *params
            )
            - self.ll_interval_or_truncated(
                data.tl_unique, data.tr_unique, data.n_t_unique, *params
            )
        )

    def _neg_ll_func(self, data: SurpyvalData, *params: Any) -> Any:
        return -self._log_likelihood(data, *params)

    def _moment(self, n: Any, *params: Any, offset: bool = False) -> Any:
        """The ``n``-th raw moment, used by the method-of-moments fit and
        by ``Parametric.var``.

        With an offset the moment of ``gamma + X`` is the binomial
        expansion of the un-offset raw moments,
        :math:`\\sum_k \\binom{n}{k} \\gamma^{n-k} E[X^k]` -- exactly as
        ``Parametric.moment`` computes it. This used to integrate
        ``x**n * df(x - gamma)`` from ``gamma`` to infinity by quadrature
        even for distributions with closed-form moments, which was both
        slower and, on some machines, tripped ``quad``'s roundoff warning
        (and with it the warnings-as-errors documentation build).
        """
        from scipy.integrate import quad

        if offset:
            gamma = params[0]
            params = params[1::]
            base = [1.0] + [
                float(self._moment(k, *params)) for k in range(1, n + 1)
            ]
            return sum(
                comb(n, k) * gamma ** (n - k) * base[k] for k in range(n + 1)
            )
        if hasattr(self, "moment"):
            return self.moment(n, *params)

        def fun(x: Numeric) -> Any:
            return x**n * self.df(x, *params)

        return quad(fun, *self.support)[0]

    def _set_support(self, model: Any, offset: bool) -> Any:
        """Resolve and assign the fitted model's support interval.

        For an offset model the left edge is the fitted ``gamma``;
        otherwise each edge comes from the distribution's declared
        support, except a data-dependent (NaN) edge, which is read from
        the fitted parameter the distribution nominates via
        ``support_param_index`` (``a``/``b`` for the uniform and the
        4-parameter Beta).
        """
        if offset:
            left = model.gamma
        elif np.isfinite(self.support[0]):
            left = self.support[0]
        elif self.support[0] == -np.inf:
            left = -np.inf
        elif np.isnan(self.support[0]):
            left = model.params[self.support_param_index[0]]

        if np.isfinite(self.support[1]):
            right = self.support[1]
        elif self.support[1] == np.inf:
            right = np.inf
        elif np.isnan(self.support[1]):
            right = model.params[self.support_param_index[1]]

        model.support = np.array([left, right])

    def _for_params(self, params: Any) -> "ParametricFitter":
        """The fitter instance that models ``params``.

        ``self`` for every distribution with a fixed number of
        parameters. A distribution whose parameter count is set by the
        parameters themselves (``Hypoexponential``: one rate per stage)
        overrides this to return an instance with the matching ``k``,
        ``parameter_names`` and ``bounds``, so a model built from a
        serialised dictionary reports the right parameter count.
        """
        return self

    def from_params(
        self, params: Any, gamma: Any = None, p: Any = None, f0: Any = None
    ) -> Any:
        r"""

        Creating a SurPyval Parametric class with provided parameters.

        Parameters
        ----------

        params : array like
            array of the parameters of the distribution.

        gamma : scalar, optional
            offset value for the distribution. If not provided will fit a
            regular, unshifted/not offset, distribution.

        p : scalar, optional
            The proportion of the population that is susceptible -- the
            proportion that will *ever* die or fail (a limited failure
            population); ``1 - p`` never fails. If used it must be a value
            between 0 and 1. If None will assume 1, i.e. every unit
            eventually fails.

        f0 : scalar, optional
            The proportion of the population that will die or fail at time 0.
            If used it must be a value between 0 and 1. If None will assume 0,
            i.e. no proportion of the population will die or fail at time 0.

        Returns
        -------

        Parametric
            A parametric model with the parameters provided.


        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 4])
        >>> print(model)
        Parametric SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : given parameters
        Parameters          :
             alpha: 10
              beta: 4
        >>> model = Weibull.from_params([10, 4], gamma=2)
        >>> print(model)
        Parametric SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : given parameters
        Offset (gamma)      : 2
        Parameters          :
             alpha: 10
              beta: 4
        """
        if self.k != len(params):
            detail = f"Must have {self.k} params for {self.name} distribution"
            raise ValueError(detail)

        # A proportion outside [0, 1], or a zero-inflation fraction at or
        # above the proportion that ever fails, is not a distribution:
        # p = 1.5 gave sf(100) = -0.5 and f0 = -0.1 gave ff(0) = -0.1.
        if p is not None and not (0 < p <= 1):
            raise ValueError(
                f"p, the proportion that ever fails, must be in (0, 1]; "
                f"got {p}"
            )
        if f0 is not None:
            if not (0 <= f0 < 1):
                raise ValueError(
                    f"f0, the proportion failing at time 0, must be in "
                    f"[0, 1); got {f0}"
                )
            if f0 >= (1 if p is None else p):
                raise ValueError(
                    f"f0 ({f0}) must be less than p ({p}): the proportion "
                    "failing at time 0 is part of the proportion that ever "
                    "fails"
                )
            # The same condition fit(zi=True) applies: the mass f0 sits at
            # 0, which must be where the support starts.
            if self.support[0] != 0:
                raise ValueError(
                    "zero-inflated models can only work with models "
                    "starting at 0"
                )

        # Offsetting only makes sense for a half-line support; a fully
        # unbounded support (Normal) or a data-dependent one whose bounds
        # are themselves estimated (Uniform/Beta4, declared NaN) cannot be
        # offset. This mirrors the ``offsettable`` check in ``fit``.
        if gamma is not None and (
            np.isinf(self.support).all() or np.isnan(self.support).any()
        ):
            detail = f"{self.name} distribution cannot be offset"
            raise ValueError(detail)

        if gamma is not None:
            offset = True
        else:
            offset = False
            gamma = 0

        if p is not None:
            lfp = True
        else:
            lfp = False
            p = 1

        if f0 is not None:
            zi = True
        else:
            zi = False
            f0 = 0

        model = Parametric(self, "given parameters", None, offset, lfp, zi)
        model.gamma = gamma
        model.p = p
        model.f0 = f0
        model.params = np.array(params)
        self._set_support(model, offset)

        for i, (low, upp) in enumerate(self.bounds):
            if low is None:
                lower_limit = -np.inf
            else:
                lower_limit = low
            if upp is None:
                upper_limit = np.inf
            else:
                upper_limit = upp

            if not (lower_limit < params[i] < upper_limit):
                names = ", ".join(self.parameter_names)
                detail = f"Params {names} must be in" f" bounds {self.bounds}"
                raise ValueError(detail)
        self._check_params(params)
        return model

    def _check_params(self, params: Any) -> None:
        """Raise for a parameter vector that is within every parameter's
        own bounds but still not a distribution (``a < b`` for the
        Uniform and the 4-parameter Beta). Nothing to check by default."""
        return None


# The generic log functions above are inherited by the distributions that
# have no closed form of their own, so they are guarded here, once.
for _name in ("log_df", "log_sf", "log_ff"):
    setattr(
        ParametricFitter,
        _name,
        _array_inputs(
            _support_guarded(
                ParametricFitter.__dict__[_name], *_OUTSIDE_SUPPORT[_name]
            )
        ),
    )
