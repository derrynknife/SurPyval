"""The hypoexponential (generalised Erlang) distribution.

The lifetime of a unit that passes through ``m`` independent
exponential *stages* in series -- with rates ``lambda_1, ...,
lambda_m`` -- is the sum of the stage durations, and that sum is
hypoexponential. With every rate equal it is the Erlang (a Gamma with
integer shape); with distinct rates the survival function is a signed
mixture of exponentials found by partial fractions,

.. math::
    R(x) = \\sum_{j=1}^{m} C_j e^{-\\lambda_j x}, \\qquad
    C_j = \\prod_{l \\neq j} \\frac{\\lambda_l}{\\lambda_l - \\lambda_j},

with :math:`\\sum_j C_j = 1` so that :math:`R(0) = 1`. Load-sharing
groups (the group rate changes as members fail), warm and hot standby
and any other "k stages, each memoryless" model produce this
distribution.

The number of stages is not fixed in advance, so the distribution has no
fixed parameter count: ``Hypoexponential.from_params([r1, r2, ...])``
accepts any number of rates and the fitted model carries that many. The
distribution functions can be called directly on the singleton with the
rates as the parameters, ``Hypoexponential.sf(x, r1, r2, ...)``, exactly
as for every other distribution. ``Hypoexponential.fit(x, stages=m)``
estimates the ``m`` rates from data by maximum likelihood (or any other
``how``), with every kind of censoring and truncation; the number of
stages is the caller's choice, compared between fits by their AIC.

Rates may repeat. The partial-fraction coefficients grow without bound
(with alternating signs) as two rates approach each other, so where they
are large -- rates close together, or equal -- the functions are taken
instead from the first row of the matrix exponential of the stages'
generator, computed with only sums and products of non-negative numbers
(``_first_row``), which keeps their relative precision whatever the
rates.
"""

from __future__ import annotations

from typing import Any

import autograd.numpy as np
import numpy.typing as npt
from autograd.extend import defvjp, primitive
from autograd.numpy.numpy_boxes import ArrayBox
from scipy.special import factorial, gammaln, xlogy

from surpyval.univariate.parametric.parametric import draw_state
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
    normalise_how,
)
from surpyval.utils.deprecation import fit_entry
from surpyval.utils.surpyval_data import (
    SurpyvalData,
    accepts_surpyval_data,
    refuses_what_fit_cannot_take,
)
from surpyval.utils.units import takes_unit

from ..parametric import Parametric

#: The partial-fraction form is used while the absolute values of its
#: coefficients sum to at most this: its rounding error is about this
#: many units in the last place. Beyond it (rates close together, or
#: equal) the matrix exponential (``_first_row``) is used: at a
#: relative gap of 1e-5 between four rates the partial fractions gave
#: twice the density.
PARTIAL_FRACTION_LIMIT = 100.0

# the smallest positive double, the floor of the quantile's bracket
_SMALLEST = float(np.nextafter(0.0, 1.0))


def _validate_rates(rates: Any) -> npt.NDArray:
    """The stage rates as a 1-D float array: finite and positive."""
    r = np.atleast_1d(np.asarray(rates, dtype=float))
    if r.ndim != 1 or r.size == 0:
        raise ValueError(
            "Hypoexponential needs a one-dimensional, non-empty vector of "
            "stage rates; got shape {}".format(r.shape)
        )
    if not np.isfinite(r).all() or (r <= 0).any():
        raise ValueError(
            "Hypoexponential stage rates must all be finite and strictly "
            "positive; got {}".format(r.tolist())
        )
    return r


def _coefficients(rates: npt.NDArray) -> npt.NDArray:
    """The partial-fraction coefficients ``C_j`` of the survival function,
    ``prod_{l != j} r_l / (r_l - r_j)``; they sum to one."""
    m = len(rates)
    index = np.arange(m)
    return np.array(
        [
            np.prod(rates[index != j] / (rates[index != j] - rates[j]))
            for j in range(m)
        ]
    )


def _partial_fractions_hold(rates: npt.NDArray) -> bool:
    """Whether the partial-fraction form keeps its precision at these
    rates: they are distinct and the coefficients' absolute values sum
    to at most ``PARTIAL_FRACTION_LIMIT``."""
    if len(rates) == 1:
        return True
    if len(np.unique(rates)) < len(rates):
        return False
    with np.errstate(all="ignore"):
        size = float(np.sum(np.abs(_coefficients(rates))))
    return bool(np.isfinite(size) and size <= PARTIAL_FRACTION_LIMIT)


#: Where ``lambda_max * x`` is at most this, the CDF and density are
#: summed as their power series (``_series``) rather than from the partial
#: fractions, which cancel there: two stages at ``x = 1e-15`` have
#: ``F = 1e-30`` as a difference of numbers of size ``1e-15``.
_SERIES_MAX = 1.0
#: Terms of that series; the n-th is below ``1 / n!`` of the first.
_SERIES_TERMS = 40


def _series(x: npt.NDArray, rates: npt.NDArray) -> tuple:
    r"""
    :math:`\ln F(x)` and :math:`\ln f(x)` from the power series

    .. math::
        F(x) = \prod_j (\lambda_j x) \sum_{n \ge 0}
               \frac{(-1)^n h_n(\lambda x)}{(n + m)!}, \qquad
        f(x) = \frac{\prod_j (\lambda_j x)}{x} \sum_{n \ge 0}
               \frac{(-1)^n h_n(\lambda x)}{(n + m - 1)!},

    with :math:`h_n` the complete homogeneous symmetric polynomial of
    degree :math:`n` (the Taylor series of the divided difference of
    :math:`e^{-\lambda x}` over the rates, equal rates included). For
    :math:`\lambda_j x \le 1` the terms fall in size and alternate in
    sign, so neither sum cancels, and the logs stay finite where :math:`F`
    and :math:`f` underflow (#442, #443). ``x`` must be positive.
    """
    m = len(rates)
    mu = x[..., None] * rates
    h = np.zeros(x.shape + (_SERIES_TERMS,))
    h[..., 0] = 1.0
    for j in range(m):
        for n in range(1, _SERIES_TERMS):
            h[..., n] += mu[..., j] * h[..., n - 1]
    n = np.arange(_SERIES_TERMS)
    signed = np.where(n % 2 == 0, 1.0, -1.0) * h
    sum_ff = np.sum(signed * np.exp(-gammaln(n + m + 1.0)), axis=-1)
    sum_df = np.sum(signed * np.exp(-gammaln(n + m)), axis=-1)
    log_prod = np.sum(np.log(rates)) + m * np.log(x)
    return log_prod + np.log(sum_ff), log_prod - np.log(x) + np.log(sum_df)


#: Terms of the Taylor series of each scaled step of ``_first_row``, whose
#: matrix has norm at most 1: the last is below 1 / 20!, 4e-19.
_TAYLOR_TERMS = 20


def _first_row(x: npt.NDArray, rates: npt.NDArray, absorbing: bool) -> Any:
    r"""
    The first row of a matrix exponential of the stages' generator at
    each (positive, finite) ``x``, the probabilities of the stage a unit
    started in the first is in at ``x``.

    With ``absorbing`` it is :math:`e^{Qx}`, ``Q`` the generator with the
    absorbing state (failure) last, so the row's last entry is
    :math:`F(x)`; without, it is :math:`e^{(T + \lambda_{\min} I) x}` for the
    stages alone (``T``), the probabilities of each stage times
    :math:`e^{\lambda_{\min} x}`, which stay representable in the tail where
    :math:`R(x)` underflows.

    Each is :math:`e^{-c x} e^{B x}` with ``B`` = ``Q + \lambda_{\max} I`` (or
    ``T + lambda_max I``) and ``c`` its shift, and ``B`` has no negative
    entry. So ``e^{B h}`` at ``h = x / 2^s`` (with ``lambda_max h <= 1/2``) is
    a Taylor sum of non-negative terms, and squaring it ``s`` times adds
    and multiplies non-negative numbers only: nothing cancels, and every
    entry keeps its relative precision, however close the rates.
    """
    m = len(rates)
    size = m + 1 if absorbing else m
    r_max = float(np.max(rates))
    shift = r_max if absorbing else r_max - float(np.min(rates))
    index = np.arange(m)
    B = np.zeros((size, size))
    B[index, index] = r_max - rates
    if absorbing:
        B[index, index + 1] = rates
        B[m, m] = r_max
    else:
        B[index[:-1], index[:-1] + 1] = rates[:-1]
    identity = np.eye(size)
    with np.errstate(divide="ignore"):
        steps = np.ceil(np.log2(2.0 * r_max * x))
    steps = np.maximum(steps, 0).astype(int)
    out = np.empty(x.shape + (size,))
    for s in np.unique(steps):
        chosen = steps == s
        h = x[chosen] / 2.0**s
        A = B * h[:, None, None]
        E = identity + A / _TAYLOR_TERMS
        for k in range(_TAYLOR_TERMS - 1, 0, -1):
            E = identity + (A @ E) / k
        E = E * np.exp(-shift * h)[:, None, None]
        for _ in range(s):
            E = E @ E
        out[chosen] = E[:, 0, :]
    return out


def _pieces(x: Numeric, rates: Any) -> dict[str, npt.NDArray]:
    """
    Every function at ``x``, each from the form that is exact where it is
    used:

    - near the origin (``lambda_max x <= 1``) the power series of
      ``_series`` for ``F`` and ``f``;
    - elsewhere, where the partial fractions keep their precision
      (``_partial_fractions_hold``), ``F`` from them in ``expm1`` form,
      and, with the slowest rate factored out,
      ``R = e^(-lambda_min x) sum_j C_j e^(-(lambda_j - lambda_min) x)``
      and ``f`` likewise, whose logs stay finite after ``R`` and ``f``
      underflow (#443); the survival function as ``1 - F`` (and its log
      as ``log1p(-F)``) while ``F < 1/2``, and the hazard as ``f / R`` from
      the scaled sums beyond, which is not 0 / 0 where both underflow
      (#444);
    - elsewhere, for rates close together or equal, ``R`` and ``f`` from the
      scaled stage probabilities of ``_first_row`` and ``F``, while it is below
      1/2, from its absorbing form.
    """
    r = _validate_rates(rates)
    x_arr = np.asarray(x, dtype=float)
    x_flat = x_arr.ravel()
    m, r_min, r_max = len(r), float(np.min(r)), float(np.max(r))
    # the support edges; a NaN stays NaN
    at_zero_df = float(r[0]) if m == 1 else 0.0
    edges = {
        "ff": (0.0, 1.0),
        "sf": (1.0, 0.0),
        "log_ff": (-np.inf, 0.0),
        "log_sf": (0.0, -np.inf),
        "log_df": (np.log(at_zero_df) if m == 1 else -np.inf, -np.inf),
        "df": (at_zero_df, 0.0),
        "hf": (at_zero_df, r_min),
    }
    out = {}
    for name, (at_zero, at_inf) in edges.items():
        value = np.full(x_flat.shape, np.nan)
        value[x_flat == 0] = at_zero
        value[x_flat == np.inf] = at_inf
        out[name] = value
    finite = (x_flat > 0) & (x_flat < np.inf)
    series = finite & (x_flat * r_max <= _SERIES_MAX)
    direct = finite & ~series
    if np.any(series):
        log_ff, log_df = _series(x_flat[series], r)
        ff = np.exp(log_ff)
        # F is at most 1 - 1/e here (one stage), so 1 - F does not
        # cancel
        log_sf = np.log1p(-ff)
        parts = {
            "ff": ff,
            "sf": 1.0 - ff,
            "log_ff": log_ff,
            "log_sf": log_sf,
            "log_df": log_df,
            "df": np.exp(log_df),
            "hf": np.exp(log_df - log_sf),
        }
        for name, value in parts.items():
            out[name][series] = value
    if np.any(direct):
        if _partial_fractions_hold(r):
            parts = _partial_fraction_parts(x_flat[direct], r)
        else:
            parts = _matrix_parts(x_flat[direct], r)
        for name, value in parts.items():
            out[name][direct] = value
    return {
        name: value.reshape(x_arr.shape)[()] for name, value in out.items()
    }


def _partial_fraction_parts(x_d: npt.NDArray, r: npt.NDArray) -> dict:
    """The functions at ``x_d`` (away from the origin) from the partial
    fractions (see ``_pieces``)."""
    r_min = float(np.min(r))
    coef = _coefficients(r)
    e = np.exp(-x_d[:, None] * (r - r_min))
    scaled_sf = np.sum(coef * e, axis=-1)
    scaled_df = np.sum(coef * r * e, axis=-1)
    ff = np.clip(
        -np.sum(coef * np.expm1(-x_d[:, None] * r), axis=-1), 0.0, 1.0
    )
    low = ff < 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        # the sums are clipped at 0 against the cancellation of their
        # terms, and their logs are then -inf
        log_sf_d = -r_min * x_d + np.log(np.maximum(scaled_sf, 0.0))
        log_df = -r_min * x_d + np.log(np.maximum(scaled_df, 0.0))
        log_sf = np.where(low, np.log1p(-np.minimum(ff, 0.5)), log_sf_d)
        log_ff = np.where(
            low,
            np.log(ff),
            np.log1p(-np.exp(np.minimum(log_sf_d, np.log(0.5)))),
        )
        hf = np.where(low, np.exp(log_df - log_sf), scaled_df / scaled_sf)
    return {
        "ff": ff,
        "sf": np.where(low, 1.0 - ff, np.minimum(np.exp(log_sf_d), 1.0)),
        "log_ff": log_ff,
        "log_sf": log_sf,
        "log_df": log_df,
        "df": np.exp(log_df),
        "hf": hf,
    }


def _matrix_parts(x_d: npt.NDArray, r: npt.NDArray) -> dict:
    """The functions at ``x_d`` (away from the origin) from the matrix
    exponential (``_first_row``), for rates close together or equal."""
    r_min = float(np.min(r))
    scaled = _first_row(x_d, r, absorbing=False)
    with np.errstate(divide="ignore"):
        log_sf = -r_min * x_d + np.log(np.sum(scaled, axis=-1))
        # the last stage is left at its own rate
        log_df = -r_min * x_d + np.log(r[-1] * scaled[:, -1])
    sf = np.exp(log_sf)
    low = sf > 0.5
    ff = -np.expm1(log_sf)
    if np.any(low):
        # F below 1/2 from the absorbing state's probability, which does
        # not cancel as 1 - R does
        ff[low] = _first_row(x_d[low], r, absorbing=True)[:, -1]
    with np.errstate(divide="ignore"):
        log_ff = np.where(low, np.log(ff), np.log1p(-np.minimum(sf, 0.5)))
    return {
        "ff": ff,
        "sf": np.where(low, 1.0 - ff, sf),
        "log_ff": log_ff,
        "log_sf": log_sf,
        "log_df": log_df,
        "df": np.exp(log_df),
        "hf": np.exp(log_df - log_sf),
    }


@primitive
def _log_function(x: Any, rates: Any, kind: str) -> Any:
    """``log_sf``, ``log_df`` or ``log_ff`` (``kind``) at ``x`` for the vector
    of stage rates, as a function autograd differentiates (see
    ``_rates_vjp`` and ``_x_vjp``): the fit's likelihood and every bound
    taken by the delta method go through it."""
    return _pieces(x, rates)[kind]


def _difference(a: Any, b: Any) -> Any:
    """``a - b``, taken as 0 where ``b`` is infinite (a function at the edge
    of its support, where its derivative is not used)."""
    held = np.isfinite(b)
    return np.where(held, a - np.where(held, b, 0.0), 0.0)


def _rates_vjp(ans: Any, x: Any, rates: Any, kind: str) -> Any:
    r"""
    The derivative in each rate. A stage's density
    :math:`\lambda e^{-\lambda t}` has derivative in its rate
    :math:`(g_1 - g_2) / \lambda`, :math:`g_1` itself and :math:`g_2` the
    Erlang of two such stages, so every function :math:`G` of the sum
    (``R``, ``f``, ``F``) has

    .. math::
        \frac{\partial \ln G}{\partial \lambda_j}
        = \frac{1}{\lambda_j}\left(1 - \frac{G(x; \lambda,
          \lambda_j)}{G(x; \lambda)}\right),

    the numerator with stage ``j`` repeated. That is the same function
    again, so its derivatives follow in turn, to any order.
    """

    def vjp(g: Any) -> Any:
        columns = []
        for j in range(len(rates)):
            repeated = np.concatenate([rates, rates[j : j + 1]])
            d = _difference(_log_function(x, repeated, kind), ans)
            columns.append(np.sum(g * -np.expm1(d)) / rates[j])
        return np.array(columns)

    return vjp


def _x_vjp(ans: Any, x: Any, rates: Any, kind: str) -> Any:
    r"""
    The derivative in ``x`` (an offset fit's): ``-h`` for ``log_sf``,
    ``f / F`` for ``log_ff``, and for ``log_df``
    :math:`\lambda_m (f_{-m}(x) / f(x) - 1)`, from
    :math:`f(x) = \lambda_m (F_{-m}(x) - F(x))`, :math:`F_{-m}` the CDF of
    the sum without its last stage.
    """
    if kind == "log_df":
        if len(rates) == 1:
            derivative = -rates[0] * np.ones(np.shape(ans))
        else:
            rest = _log_function(x, rates[:-1], kind)
            derivative = rates[-1] * np.expm1(_difference(rest, ans))
    else:
        density = _log_function(x, rates, "log_df")
        ratio = np.exp(_difference(density, ans))
        derivative = -ratio if kind == "log_sf" else ratio
    return lambda g: g * derivative


defvjp(_log_function, _x_vjp, _rates_vjp)


def _traced(x: Any, rates: tuple) -> bool:
    """Whether autograd is differentiating through ``x`` or a rate."""
    return isinstance(x, ArrayBox) or any(
        isinstance(r, ArrayBox) for r in rates
    )


def _traced_log(x: Any, rates: tuple, kind: str) -> Any:
    """``_log_function`` with the rates, each perhaps traced, as one
    vector."""
    return _log_function(x, np.array(list(rates)), kind)


def _moments(k: int, rates: npt.NDArray) -> Any:
    r"""
    The ``k``-th raw moment, :math:`k! \, \alpha (-T)^{-k} \mathbf{1}` for the
    stages' generator ``T``: each solve of the bidiagonal ``-T w = v`` is
    ``w_j = v_j / lambda_j + w_{j+1}``, a sum of positive terms, exact for
    equal rates too.
    """
    v = np.ones(len(rates))
    for _ in range(k):
        v = np.cumsum((v / rates)[::-1])[::-1]
    return factorial(k) * v[0]


def _stage_count(stages: Any) -> int:
    """``stages`` as a positive integer, or a ``ValueError`` that says what
    to pass."""
    if (
        isinstance(stages, bool)
        or not isinstance(stages, (int, np.integer))
        or stages < 1
    ):
        raise ValueError(
            "stages, the number of exponential stages in series, must be a "
            "positive integer, e.g. Hypoexponential.fit(x, stages=2); got "
            "{!r}".format(stages)
        )
    return int(stages)


def _slowest_first(model: Parametric) -> None:
    """Put a fitted model's rates in order, slowest stage first, with
    their rows and columns of the covariance: the stages' order does not
    change the distribution, so the search may return them in any."""
    params = np.asarray(model.params, dtype=float)
    k = len(params)
    order = np.argsort(params, kind="stable")
    if np.array_equal(order, np.arange(k)):
        return
    model.params = params[order]
    for name in ("_covariance", "hess_inv"):
        cov = getattr(model, name, None)
        if cov is None:
            continue
        cov = np.array(cov, dtype=float)
        index = np.arange(cov.shape[0])
        index[:k] = order
        setattr(model, name, cov[np.ix_(index, index)])
    for name, value in zip(model.dist.parameter_names, model.params):
        setattr(model, name, value)


class Hypoexponential_(OptimisedFitMixin, ParametricFitter):
    r"""

    Class used to generate the Hypoexponential class: the sum of
    independent Exponential stages.

    The singleton has no fixed parameter count -- ``from_params`` takes
    any number of rates, and ``fit`` the number of stages, and the model
    either returns has that many parameters, named ``lambda_1 ...
    lambda_m``. The distribution functions take the rates as the
    parameters: ``Hypoexponential.sf(x, 0.5, 1.5, 3.0)``.

    Examples
    --------
    Two stages in series, with mean times 2 and 2/3:

    >>> from surpyval import Hypoexponential
    >>> model = Hypoexponential.from_params([0.5, 1.5])
    >>> model.sf([1, 2, 4]).round(4)
    array([0.7982, 0.5269, 0.2018])
    >>> round(float(model.mean()), 4)
    2.6667

    Fitted to data, with the number of stages given:

    >>> x = Hypoexponential.random(500, 0.5, 2.0, random_state=3)
    >>> Hypoexponential.fit(x, stages=2).params.round(3)
    array([0.504, 1.827])
    """

    # The Parameters entries the method docstrings below repeat. Edit them
    # here and run scripts/sync_docstrings.py --write to update the copies.
    _param_docs = """
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``
        """

    def __init__(self, name: str, m: int = 0) -> None:
        # ``m == 0`` is the exported singleton, whose arity is only fixed
        # when ``from_params`` or ``fit`` is called; that call (and
        # deserialisation) builds an ``m``-stage instance with concrete
        # parameter names and bounds, so the resulting model reports the
        # right ``k``.
        super().__init__(
            name=name,
            k=m,
            bounds=((0, None),) * m,
            support=(0, np.inf),
            parameter_names=[f"lambda_{j + 1}" for j in range(m)],
            param_map={f"lambda_{j + 1}": j for j in range(m)},
            plot_x_scale="linear",
        )
        # No probability-plotting transform: the survival function is
        # a sum of exponentials with no single linearising scale.
        self.supports_mpp = False
        # Every stage's rate is per unit of time (rescale_time).
        self._time_roles = ("rate",) * m

    def _for_params(self, params: Any) -> Hypoexponential_:
        """The ``m``-stage fitter that models ``params``."""
        return self._for_stages(len(np.atleast_1d(np.asarray(params))))

    def _for_stages(self, m: int) -> Hypoexponential_:
        """The ``m``-stage fitter."""
        if m == self.k:
            return self
        return Hypoexponential_(self.name, m=m)

    @takes_unit
    def from_params(
        self,
        params: npt.ArrayLike,
        gamma: Boxable | None = None,
        lfp_p: Boxable | None = None,
        f0: Boxable | None = None,
        hr: Boxable | None = None,
    ) -> Parametric:
        r"""

        Create a hypoexponential model from its stage rates.

        Parameters
        ----------

        params : array like
            The stage rates ``lambda_1, ..., lambda_m``, all strictly
            positive; they may repeat. Any number of them.
        gamma : scalar, optional
            An offset (shift) of the distribution.
        lfp_p : scalar, optional
            The proportion of the population that ever fails
            (limited-failure population).
        f0 : scalar, optional
            The proportion of the population that fails at time zero
            (zero inflation).

        Returns
        -------

        Parametric
            A parametric model with ``len(params)`` parameters.

        Examples
        --------
        >>> from surpyval import Hypoexponential
        >>> model = Hypoexponential.from_params([0.5, 1.5, 3.0])
        >>> print(model)
        Parametric SurPyval Model
        =========================
        Distribution        : Hypoexponential
        Fitted by           : given parameters
        Parameters          :
          lambda_1: 0.5
          lambda_2: 1.5
          lambda_3: 3.0
        >>> model.mean()
        np.float64(3.0)

        Equal rates are the Erlang distribution, a Gamma with an integer
        shape:

        >>> from surpyval import Gamma
        >>> erlang = Hypoexponential.from_params([2.0, 2.0, 2.0])
        >>> round(float(erlang.sf(1.5)), 10)
        0.4231900811
        >>> round(float(Gamma.sf(1.5, 3, 2.0)), 10)
        0.4231900811
        """
        rates = _validate_rates(params)
        return ParametricFitter.from_params(
            self._for_params(rates), rates, gamma, lfp_p, f0, hr
        )

    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike | None = None,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        how: str = "MLE",
        offset: bool = False,
        zi: bool = False,
        lfp: bool = False,
        tl: npt.ArrayLike | float | None = None,
        tr: npt.ArrayLike | float | None = None,
        xl: npt.ArrayLike | None = None,
        xr: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
        heuristic: str = "Nelson-Aalen",
        init: npt.ArrayLike = [],
        rr: str = "y",
        on_d_is_0: bool = False,
        turnbull_estimator: str = "Fleming-Harrington",
        w: npt.ArrayLike | float | None = None,
        hr: float | None = None,
        *,
        stages: int | None = None,
    ) -> Parametric:
        """

        Fit a hypoexponential with a given number of stages to data.

        The ``stages`` rates are estimated together, by maximum likelihood
        by default, with every kind of censoring, truncation, counts,
        weights, an offset, a limited failure population and zero
        inflation, as for every other distribution (see ``Weibull.fit``).
        The rates are exchangeable -- the sum
        of the stages does not depend on their order -- and are reported
        from the slowest stage to the fastest. They may come out equal,
        an Erlang stretch of the chain. A likelihood that keeps rising as
        a rate runs to infinity says the data need fewer stages, and the
        fit warns so. Choose the number of stages by comparing the fits'
        AIC.

        Probability plotting (``how='MPP'``) is not available: the
        distribution has no straight-line probability plot.

        Parameters
        ----------

        x, c, n, t, how, offset, zi, lfp, tl, tr, xl, xr, fixed, \
heuristic, init, rr, on_d_is_0, turnbull_estimator, w, hr
            As for ``Weibull.fit``; a rate in ``fixed`` is named
            ``lambda_1`` to ``lambda_m``, and ``init`` has one rate per stage.
        stages : int
            The number of exponential stages in series, ``m``: one rate
            each. Required on ``Hypoexponential`` itself; a fitted model's
            ``dist`` refits with its own number by default.

        Returns
        -------

        Parametric
            A parametric model with ``stages`` parameters.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> x = Hypoexponential.random(300, 0.2, 1.0, random_state=7)
        >>> c = (x > 12).astype(int)
        >>> x = np.minimum(x, 12)
        >>> model = Hypoexponential.fit(x, c, stages=2)
        >>> print(model)
        Parametric SurPyval Model
        =========================
        Distribution        : Hypoexponential
        Fitted by           : MLE
        Data                : 300 units: 275 events at 275 unique times,
                              25 right censored
        Parameters          :
          lambda_1: 0.24498253825060956
          lambda_2: 0.5835650937092565
        """
        surv_data = SurpyvalData(
            x=x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr, w=w
        )
        return self._fit_from_surpyval_data(
            surv_data,
            how=how,
            offset=offset,
            zi=zi,
            lfp=lfp,
            fixed=fixed,
            heuristic=heuristic,
            init=init,
            rr=rr,
            on_d_is_0=on_d_is_0,
            turnbull_estimator=turnbull_estimator,
            hr=hr,
            stages=stages,
        )

    fit_from_surpyval_data = fit_entry("_fit_from_surpyval_data")

    @refuses_what_fit_cannot_take
    def _fit_from_surpyval_data(
        self,
        surv_data: SurpyvalData,
        how: str = "MLE",
        offset: bool = False,
        zi: bool = False,
        lfp: bool = False,
        fixed: dict[str, float] | None = None,
        heuristic: str = "Nelson-Aalen",
        init: npt.ArrayLike = [],
        rr: str = "y",
        on_d_is_0: bool = False,
        turnbull_estimator: str = "Fleming-Harrington",
        hr: float | None = None,
        *,
        stages: int | None = None,
    ) -> Parametric:
        """

        Fit a hypoexponential with a given number of stages to data
        already held in a :class:`~surpyval.utils.surpyval_data.SurpyvalData`
        (see :meth:`fit`).

        Parameters
        ----------

        surv_data : SurpyvalData
            Survival data in the SurpyvalData class.
        how, offset, zi, lfp, fixed, heuristic, init, rr, on_d_is_0, \
turnbull_estimator, hr, stages
            As for :meth:`fit`.

        Returns
        -------

        Parametric
            A parametric model with ``stages`` parameters.

        Examples
        --------
        >>> from surpyval import Hypoexponential, SurpyvalData
        >>> x = Hypoexponential.random(500, 0.5, 2.0, random_state=3)
        >>> model = Hypoexponential.fit(
        ...     SurpyvalData(x), stages=2)
        >>> model.params.round(3)
        array([0.504, 1.827])
        """
        if normalise_how(how) == "MPP":
            # refused as for every family without a straight-line plot,
            # before the number of stages is asked for
            self._refuse_mpp_without_plot()
        if stages is None:
            if self.k == 0:
                raise ValueError(
                    "Hypoexponential.fit needs stages, the number of "
                    "exponential stages in series (one rate each), e.g. "
                    "Hypoexponential.fit(x, stages=2); compare the AIC of "
                    "fits with different numbers to choose it"
                )
            stages = self.k
        fitter = self._for_stages(_stage_count(stages))
        model = OptimisedFitMixin._fit_from_surpyval_data(
            fitter,
            surv_data,
            how=how,
            offset=offset,
            zi=zi,
            lfp=lfp,
            fixed=fixed,
            heuristic=heuristic,
            init=init,
            rr=rr,
            on_d_is_0=on_d_is_0,
            turnbull_estimator=turnbull_estimator,
            hr=hr,
        )
        if not fixed:
            _slowest_first(model)
        return model

    def _runaway_advice(
        self, runaway: list[str], values: dict, offset: bool = False
    ) -> str:
        """A rate running to infinity is a stage that takes no time: the
        data are described as well by fewer stages."""
        if self.k > 1 and all(name.startswith("lambda_") for name in runaway):
            fewer = max(self.k - len(runaway), 1)
            return (
                "a rate running to infinity is a stage that takes no time: "
                f"fit with fewer stages (stages={fewer}) and compare the AIC"
            )
        return super()._runaway_advice(runaway, values, offset)

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        """A start for the ``m`` rates: stage means in the ratio 1 : 1/2 :
        1/4 ..., adding up to the mean of the exponential fitted to the
        data (time on test over the number of failures)."""
        return self._start(data, 2.0)

    def _alternative_base_starts(
        self, data: SurpyvalData, offset: bool
    ) -> list[npt.NDArray]:
        """Two more starts for maximum likelihood: stage means nearly
        equal (towards the Erlang) and far apart (one stage holding most
        of the mean)."""
        if offset or self.k < 2:
            return []
        return [self._start(data, 1.25), self._start(data, 10.0)]

    def _start(self, data: SurpyvalData, ratio: float) -> npt.NDArray:
        """Rates whose stage means fall by ``ratio`` from one stage to the
        next and add up to the exponential mean of the data, slowest
        first."""
        x = np.asarray(data.x, dtype=float)
        c = np.asarray(data.c)
        n = np.asarray(data.n, dtype=float)
        if x.ndim == 2:
            x = x.mean(axis=1)
        keep = np.isfinite(x) & (x > 0)
        failures = float(np.sum(n[keep & (c != 1)]))
        total = float(np.sum(n[keep] * x[keep]))
        if total <= 0:
            total = 1.0
        mean = total / max(failures, 1.0)
        shares = ratio ** -np.arange(float(self.k))
        return np.sort(shares.sum() / (mean * shares))

    def sf(self, x: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Survival (or reliability) function for the Hypoexponential
        Distribution:

        .. math::
            R(x) = \sum_{j=1}^{m} C_j e^{-\lambda_j x}, \qquad
            C_j = \prod_{l \neq j} \frac{\lambda_l}{\lambda_l - \lambda_j}

        Evaluated as ``1 - F(x)`` while ``F(x) < 1/2`` -- ``F`` is exact
        at the origin, so ``R(0) = 1`` exactly -- and beyond that as the
        signed sum above with the slowest stage's exponential factored
        out, which keeps its relative precision in the tail; the sum is
        clipped at 0 against the floating-point cancellation of its terms
        (see ``_pieces``).

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Hypoexponential.sf(x, 0.5, 1.5, 3.0)
        array([0.87858244, 0.61289168, 0.39054997, 0.24112599, 0.14719997])
        """
        if _traced(x, rates):
            return np.exp(_traced_log(x, rates, "log_sf"))
        return _pieces(x, rates)["sf"]

    def ff(self, x: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the Hypoexponential
        Distribution:

        .. math::
            F(x) = 1 - \sum_{j=1}^{m} C_j e^{-\lambda_j x}
                 = -\sum_{j=1}^{m} C_j \left(e^{-\lambda_j x} - 1\right)

        Near the origin (``lambda_max x <= 1``), where the partial
        fractions cancel, it is summed as its power series, which keeps
        its relative precision down to underflow; beyond that it is
        evaluated in the second form (``expm1``).

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Hypoexponential.ff(x, 0.5, 1.5, 3.0)
        array([0.12141756, 0.38710832, 0.60945003, 0.75887401, 0.85280003])
        """
        if _traced(x, rates):
            return np.exp(_traced_log(x, rates, "log_ff"))
        return _pieces(x, rates)["ff"]

    def df(self, x: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Density function for the Hypoexponential Distribution:

        .. math::
            f(x) = \sum_{j=1}^{m} C_j \lambda_j e^{-\lambda_j x}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Hypoexponential.df(x, 0.5, 1.5, 3.0)
        array([0.24105459, 0.25789815, 0.1842277 , 0.11808731, 0.07304706])
        """
        if _traced(x, rates):
            return np.exp(_traced_log(x, rates, "log_df"))
        return _pieces(x, rates)["df"]

    def hf(self, x: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Instantaneous hazard rate for the Hypoexponential Distribution:

        .. math::
            h(x) = \frac{f(x)}{R(x)}

        It rises from ``0`` at the origin (two or more stages must all
        complete) towards the smallest stage rate, which dominates the
        tail.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Hypoexponential.hf(x, 0.5, 1.5, 3.0)
        array([0.27436764, 0.42078911, 0.4717135 , 0.48973284, 0.49624367])
        """
        if _traced(x, rates):
            return np.exp(
                _traced_log(x, rates, "log_df")
                - _traced_log(x, rates, "log_sf")
            )
        return _pieces(x, rates)["hf"]

    def Hf(self, x: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Cumulative hazard rate for the Hypoexponential Distribution:

        .. math::
            H(x) = -\ln R(x)

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Hypoexponential.Hf(x, 0.5, 1.5, 3.0)
        array([0.12944553, 0.48956707, 0.94019934, 1.42243572, 1.91596325])
        """
        if _traced(x, rates):
            return -_traced_log(x, rates, "log_sf")
        return 0.0 - _pieces(x, rates)["log_sf"]

    def log_sf(self, x: Numeric, *rates: Boxable) -> Boxable:
        """Log of the survival function (see ``_pieces``)."""
        if _traced(x, rates):
            return _traced_log(x, rates, "log_sf")
        return _pieces(x, rates)["log_sf"]

    def log_ff(self, x: Numeric, *rates: Boxable) -> Boxable:
        """Log of the CDF (see ``_pieces``)."""
        if _traced(x, rates):
            return _traced_log(x, rates, "log_ff")
        return _pieces(x, rates)["log_ff"]

    def log_df(self, x: Numeric, *rates: Boxable) -> Boxable:
        """Log of the density (see ``_pieces``)."""
        if _traced(x, rates):
            return _traced_log(x, rates, "log_df")
        return _pieces(x, rates)["log_df"]

    def qf(self, u: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Quantile function for the Hypoexponential distribution, the
        inverse of ``ff``. There is no closed form; the failure function
        is inverted by bisection -- on ``log F`` against ``log u`` below
        ``u = 1/2`` and on ``log R`` against ``log(1 - u)`` above it, so a
        tiny probability keeps its digits -- between the bounds

        .. math::
            \frac{-\ln(1 - u)}{\lambda_{\min}} \le q(u) \le
            \frac{m}{\lambda_{\min}} \ln \frac{m}{1 - u},

        the left from the slowest stage alone (the sum exceeds it) and
        the right from a union bound over the stages. ``u = 0`` gives
        ``0`` and ``u = 1`` gives ``inf``.

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Hypoexponential distribution at each
            value u

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> u = np.array([.1, .2, .3, .4, .5])
        >>> Hypoexponential.qf(u, 0.5, 1.5, 3.0)
        array([0.90854377, 1.3054212 , 1.67232635, 2.05027738, 2.46566683])
        """
        r = _validate_rates(rates)
        u_arr = np.asarray(u, dtype=float)
        scalar = u_arr.ndim == 0
        u_flat = np.atleast_1d(u_arr)
        out = np.full(u_flat.shape, np.nan)
        out[u_flat == 0.0] = 0.0
        out[u_flat == 1.0] = np.inf
        inside = (u_flat > 0.0) & (u_flat < 1.0)
        if inside.any():
            u_in = u_flat[inside]
            # u enters as log u below 1/2 and as log(1 - u) above it, and
            # is compared with log F or log R, so a small u keeps its
            # digits: 1 - u rounds to 1 below 1e-16, and the quantile
            # came out at the bisection's floor of about 1e-60 (#447).
            low = u_in <= 0.5
            log_u = np.log(u_in)
            log_1mu = np.log1p(-u_in)
            m, r_min = len(r), float(np.min(r))
            lo = np.maximum(-log_1mu / r_min, _SMALLEST)
            hi = (m / r_min) * (np.log(m) - log_1mu)
            for _ in range(400):
                # a geometric mid-point until the bracket is within a
                # factor of 2, so a quantile of 1e-150 takes a few dozen
                # steps, not a few hundred
                wide = hi > 2.0 * lo
                mid = np.where(
                    wide,
                    np.exp(0.5 * (np.log(lo) + np.log(hi))),
                    lo + 0.5 * (hi - lo),
                )
                p = _pieces(mid, r)
                too_small = np.where(
                    low, p["log_ff"] < log_u, p["log_sf"] > log_1mu
                )
                lo = np.where(too_small, mid, lo)
                hi = np.where(too_small, hi, mid)
                if np.all(hi - lo <= 4.0 * np.finfo(float).eps * hi):
                    break
            out[inside] = lo + 0.5 * (hi - lo)
        return out[0] if scalar else out

    def mean(self, *rates: Boxable) -> Boxable:
        r"""

        Mean of the Hypoexponential distribution: the stage means add.

        .. math::
            E = \sum_{j=1}^{m} \frac{1}{\lambda_j}

        Parameters
        ----------

        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        mean : scalar
            The mean of the Hypoexponential distribution

        Examples
        --------
        >>> from surpyval import Hypoexponential
        >>> Hypoexponential.mean(0.5, 1.5, 3.0)
        np.float64(3.0)
        """
        return np.sum(1.0 / _validate_rates(rates))

    def moment(self, m: int, *rates: Boxable) -> Boxable:
        r"""

        m-th raw moment of the Hypoexponential distribution, that of a
        phase-type distribution,

        .. math::
            M(m) = m! \, \alpha (-T)^{-m} \mathbf{1},

        with :math:`T` the stages' generator and :math:`\alpha` the start in
        the first stage. Each solve with the bidiagonal :math:`-T` adds
        positive terms only, so it is exact for equal rates too.
        ``moment(1)`` is ``mean``, and the variance
        ``moment(2) - moment(1)**2`` is :math:`\sum_j \lambda_j^{-2}`, the
        stage variances adding.

        Parameters
        ----------

        m : integer
            The ordinal of the moment to calculate
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        moment : scalar
            The moment of the Hypoexponential distribution

        Examples
        --------
        >>> from surpyval import Hypoexponential
        >>> Hypoexponential.moment(2, 0.5, 1.5, 3.0)
        np.float64(13.555555555555555)
        """
        return _moments(m, _validate_rates(rates))

    def entropy(self, *rates: Boxable) -> Boxable:
        r"""

        Entropy of the Hypoexponential distribution. There is no closed
        form, so it is computed by numerical integration of

        .. math::
            S = -\int_{0}^{\infty} f(x) \ln f(x) \, dx

        Parameters
        ----------

        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        entropy : scalar
            The entropy of the Hypoexponential distribution

        Examples
        --------
        >>> from surpyval import Hypoexponential
        >>> Hypoexponential.entropy(0.5, 1.5, 3.0)
        1.9455443600389024
        """
        from scipy import integrate

        r = _validate_rates(rates)

        def func(x: float) -> float:
            f = self.df(x, *r)
            return float(xlogy(f, f))

        return -integrate.quad(func, 0, np.inf)[0]

    def random(
        self,
        size: int | tuple[int, ...],
        *rates: Boxable,
        random_state: Any = None,
    ) -> npt.NDArray:
        r"""

        Draws random samples from the Hypoexponential distribution: the
        sum of one Exponential draw per stage, which is exact for any
        rates. (A fitted model's ``random`` goes through ``qf`` instead,
        like every other distribution's.)

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``
        random_state : int or numpy.random.Generator, optional
            Seed or generator for a draw of its own; ``None`` (the
            default) draws from numpy's global stream (see
            :meth:`ParametricFitter.random`).

        Returns
        -------

        random : numpy array
            Random values drawn from the distribution in shape ``size``

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> np.random.seed(1)
        >>> Hypoexponential.random(5, 0.5, 1.5, 3.0)
        array([1.92866664, 0.85812653, 0.86336444, 2.29543903, 1.86983688])
        """
        r = _validate_rates(rates)
        shape = (size,) if isinstance(size, int) else tuple(size)
        state = draw_state(random_state)
        source = np.random if state is None else state
        stages = source.exponential(1.0 / r, size=shape + (len(r),))
        return np.sum(stages, axis=-1)


Hypoexponential: Hypoexponential_ = Hypoexponential_("Hypoexponential")
