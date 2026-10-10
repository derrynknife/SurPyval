from __future__ import annotations

import inspect
from typing import Any, Callable

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from autograd.tracer import getval

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
)
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils.numeric import solve_bracketed
from surpyval.utils.rng import as_generator
from surpyval.utils.surpyval_data import SurpyvalData

from .._covariate_link import CovariateLink
from .._fit_skeleton import (
    HazardIdentitiesMixin,
    LogLinearPhi,
    MirroredDistributionAttrs,
    fit_log_linear,
    mirror_distribution,
    optimise_ph,
    uniform_draws,
)
from .._kinds import PROPORTIONAL_HAZARD
from .._likelihood import regression_neg_ll
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import DataFrameRegressionMixin, truncation_window
from ..tvc_fit import TVCFitMixin

# The name the PH covariate link had before it became the shared
# ``CovariateLink``, kept so a model pickled then still loads.
Phi = CovariateLink


def _zero_init(Z: npt.NDArray) -> npt.NDArray:
    """The log-linear fitters' start: every coefficient 0 (a module-level
    function rather than a lambda, so a fitted model pickles, #573)."""
    return np.zeros(Z.shape[1])


def _times_phi(phi: Boxable, base: Boxable) -> Boxable:
    """``phi * base``, the baseline's (cumulative) hazard scaled by the
    covariate function. A risk score ``exp(beta'Z)`` that overflowed to
    ``inf`` (a coefficient running off, a covariate far beyond the data)
    against a baseline of exactly 0 (before the hazard starts, or where it
    underflowed) is 0, and one that underflowed to 0 against an infinite
    baseline is ``inf``: the baseline's 0 or ``inf`` stands, as in Cox's
    predictions, rather than ``inf * 0`` with a raw warning (#714)."""
    with np.errstate(over="ignore", invalid="ignore"):
        out = phi * base
    lost = np.isnan(out) & ~np.isnan(phi) & ~np.isnan(base)
    if not np.any(lost):
        return out
    return np.where(lost, base, out)


#: A baseline hazard or cumulative hazard below this is scaled by
#: ``e^(beta'Z)`` on the log scale (``ProportionalHazardsFitter._scaled``):
#: its square, in the second derivatives of its log, underflows long before
#: it does, and it underflows to 0 itself while the hazard it is multiplied
#: into is finite (#728).
_SMALL_BASELINE = 1e-50
#: The log of the smallest normal float: a scaled hazard below it is taken
#: as the product, as it was.
_LOG_TINY = float(onp.log(onp.finfo(float).tiny))


def _stand_in(x: Any, rows: Any) -> Any:
    """``x`` at ``rows``, and the first of those elsewhere: a function
    evaluated there for ``rows`` alone gets a value from the other rows
    that cannot poison its gradient (autograd differentiates both
    branches of a ``np.where``)."""
    if onp.ndim(rows) == 0:
        return x
    raw = onp.broadcast_to(onp.asarray(getval(x), dtype=float), rows.shape)
    return np.where(rows, x, raw[rows][0])


def _small_on_log_scale(
    base: Boxable, log_base: Callable, x: Numeric
) -> "tuple[Any, Boxable] | None":
    """``(rows, log_values)``: the rows where the baseline (cumulative)
    hazard ``base`` at ``x`` is below ``_SMALL_BASELINE`` (0 included,
    where it underflowed) and its log, ``log_base(x)``, is finite, with
    that log there (and a stand-in elsewhere, :func:`_stand_in`); ``None``
    where there are none. A baseline that is 0 itself (``x = 0``) is
    left as it is."""
    with onp.errstate(invalid="ignore"):
        rows = onp.asarray(base < _SMALL_BASELINE)
    for _ in range(2):
        if not rows.any():
            return None
        logs = log_base(_stand_in(x, rows))
        finite = onp.asarray(np.isfinite(logs)) & rows
        if (finite == rows).all():
            return rows, logs
        rows = finite
    return None


class ProportionalHazardsFitter(
    MirroredDistributionAttrs,
    HazardIdentitiesMixin,
    TVCFitMixin,
    DataFrameRegressionMixin,
):
    """
    Parametric proportional hazards fitter: a parametric baseline hazard
    multiplied by a covariate function,

    .. math::
        h(x \\mid Z) = \\phi(Z)\\, h_0(x), \\qquad
        H(x \\mid Z) = \\phi(Z)\\, H_0(x),

    with :math:`\\phi(Z) = e^{\\beta' Z}` for the pre-built instances
    (``WeibullPH``, ``ExponentialPH``, ...) and for ``PH(dist)``. A positive
    coefficient raises the hazard (shortens life).

    Use the pre-built instances or the ``PH`` factory rather than building
    this class directly; the constructor exists for a custom ``phi(Z,
    *params)`` with its own bounds and parameter names. ``fit`` returns a
    :class:`~surpyval.univariate.regression.parametric_regression_model.ParametricRegressionModel`.

    Parameters
    ----------
    name : str
        The fitter's name (e.g. ``"WeibullLinearRR"``).
    dist : ParametricFitter
        The baseline distribution (e.g. ``Weibull``).
    phi : callable
        The covariate function, with the signature ``phi(Z, *params)``,
        written with ``autograd.numpy`` so the likelihood can be
        differentiated; it must be positive at the fitted parameters.
    phi_name : str
        A display name for ``phi``, shown in the fitted model's ``repr``.
    phi_bounds : tuple or callable
        The ``(lower, upper)`` bounds of each ``phi`` parameter (``None``
        for unbounded), or a function of the covariate matrix returning
        them. The bounds are how ``phi`` is kept positive.
    phi_param_map : dict or callable
        ``{name: position}`` of the ``phi`` parameters, or a function of the
        covariate matrix returning it.
    phi_init : callable, optional
        A function of the covariate matrix returning starting values for
        the ``phi`` parameters. Defaults to zeros.

    A model with a custom ``phi`` predicts, and gives bounds, like the
    pre-built ones, but cannot be serialised (``phi`` cannot be rebuilt
    from a name).

    Examples
    --------
    An excess-relative-risk model, :math:`\\phi(z) = 1 + \\beta z` with
    :math:`\\beta > 0`:

    >>> import numpy as np
    >>> import autograd.numpy as anp
    >>> from surpyval import ProportionalHazardsFitter, Weibull
    >>> def linear_rr(Z, *params):
    ...     return 1.0 + anp.dot(Z, anp.array(params))
    >>> WeibullLinearRR = ProportionalHazardsFitter(
    ...     "WeibullLinearRR", Weibull, linear_rr, "Linear [1 + beta'Z]",
    ...     phi_bounds=lambda Z: ((0, None),) * Z.shape[1],
    ...     phi_param_map=lambda Z: {
    ...         f"beta_{i}": i for i in range(Z.shape[1])
    ...     },
    ...     phi_init=lambda Z: np.full(Z.shape[1], 0.5),
    ... )
    >>> rng = np.random.default_rng(0)
    >>> dose = rng.uniform(0, 4, 400)
    >>> x = 10 * (-np.log(rng.uniform(size=400)) / (1 + 0.5 * dose)) ** 0.5
    >>> WeibullLinearRR.fit(x=x, Z=dose).params.round(3)
    array([10.15 ,  2.081,  0.604])
    """

    #: The ``repr`` (#614)
    fitter_kind = "proportional hazards fitter"
    name_suffix = "PH"

    def __init__(
        self,
        name: str,
        dist: Any,
        phi: Callable,
        phi_name: str,
        phi_bounds: "Callable[[npt.NDArray], tuple] | tuple",
        phi_param_map: "Callable[[npt.NDArray], dict] | dict",
        phi_init: "Callable[[npt.NDArray], npt.NDArray] | None" = None,
    ) -> None:
        # Compare names and kinds rather than the signature's string
        # form so an annotated phi (e.g. ``LogLinearPhi.phi``) passes.
        phi_sig = list(inspect.signature(phi).parameters.values())
        if not (
            len(phi_sig) == 2
            and phi_sig[0].name == "Z"
            and phi_sig[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            and phi_sig[1].name == "params"
            and phi_sig[1].kind is inspect.Parameter.VAR_POSITIONAL
        ):
            raise ValueError(
                "PH function must have the signature '(Z, *params)'"
            )

        self.name = name
        mirror_distribution(self, dist)
        self.phi = phi
        self.phi_name = phi_name
        self.Hf_dist = self.dist.Hf
        self.hf_dist = self.dist.hf
        self.sf_dist = self.dist.sf
        self.ff_dist = self.dist.ff
        self.df_dist = self.dist.df
        self.phi_init = phi_init
        self.phi_bounds = phi_bounds
        self.phi_param_map = phi_param_map

    def Hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Cumulative hazard :math:`\\phi(Z) H_0(x)` at ``x`` for covariates
        ``Z``; ``params`` are the distribution parameters followed by the
        covariate coefficients.

        Where :math:`H_0(x)` is very small (below 1e-50, or 0 where it
        underflowed), it is :math:`e^{\\log \\phi(Z) + \\log H_0(x)}`,
        with :math:`\\log H_0` from the baseline's ``log_ff``: finite, and
        with finite derivatives, where the product is not.
        """
        dist_params = np.array(params[0 : self.k_dist])
        H0 = self.Hf_dist(x, *dist_params)
        return self._scaled(
            H0, lambda at: self.dist.log_ff(at, *dist_params), x, Z, params
        )

    def hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Hazard rate :math:`\\phi(Z) h_0(x)` at ``x`` for covariates ``Z``;
        ``params`` as for :meth:`Hf`, and on the log scale where
        :math:`h_0(x)` is very small as :meth:`Hf` is, with
        :math:`\\log h_0 = \\log f_0 - \\log S_0`.
        """
        dist_params = np.array(params[0 : self.k_dist])
        h0 = self.hf_dist(x, *dist_params)
        return self._scaled(h0, self._log_h0(dist_params), x, Z, params)

    # -- on the log scale where the baseline is very small (#728) ---------
    #
    # At a LogNormal baseline running towards the Weibull limit (sigma
    # large, #583's draws 93 and 117) the search ends where H0 = 1e-306,
    # h0 is subnormal and e^(beta'Z) = 1e304: the products are finite, but
    # the second derivatives of log h0 need h0^2, which underflows, and
    # the Hessian was inf; a Weibull baseline with a large shape has H0
    # underflow to 0 where e^(beta'Z) H0 is not small at all, and so has a
    # baseline moved to Z = 0 from covariates far from it. Where the
    # baseline is that small, log H and log h are beta'Z + log H0 and
    # beta'Z + log h0, with log H0 = log F0 (exact to F0 / 2) and
    # log h0 = log f0 - log S0, from the baseline's log functions, which
    # stay finite with their derivatives; elsewhere they are what they
    # were.

    def _log_h0(self, dist_params: Any) -> Callable:
        """The baseline's log hazard, ``log f0 - log S0``, as a function
        of ``x``."""

        def log_h0(at: Numeric) -> Boxable:
            return self.dist.log_df(at, *dist_params) - self.dist.log_sf(
                at, *dist_params
            )

        return log_h0

    def _log_phi(self, Z: Numeric, phi_params: Any) -> Boxable:
        """``log phi(Z)``: ``beta'Z`` itself for the log-linear link,
        which does not overflow where ``phi`` does."""
        if self.phi is LogLinearPhi.phi:
            return np.dot(Z, phi_params)
        with np.errstate(divide="ignore"):
            return np.log(self.phi(Z, *phi_params))

    def _phi_off(self, Z: Numeric, phi_params: Any, rows: Any) -> Any:
        """``phi(Z)`` for the rows not taken on the log scale: for the
        log-linear link, 1 at ``rows``, where ``e^(beta'Z)`` can be at its
        largest and its derivatives overflow."""
        if self.phi is not LogLinearPhi.phi:
            return self.phi(Z, *phi_params)
        with np.errstate(over="ignore"):
            return np.exp(np.where(rows, 0.0, self._log_phi(Z, phi_params)))

    def _scaled(
        self,
        base: Boxable,
        log_base: Callable,
        x: Numeric,
        Z: Numeric,
        params: tuple,
    ) -> Boxable:
        """``phi(Z) base``, the baseline's (cumulative) hazard ``base`` at
        ``x`` scaled, as ``exp(log phi + log_base(x))`` where ``base`` is
        very small (see above) and the product does not underflow even
        so: below the smallest normal float it is the product as it was,
        whose derivatives in a log H0 of -1e160 (a LogNormal sigma at
        1e-80) are no better."""
        phi_params = np.array(params[self.k_dist :])
        found = _small_on_log_scale(base, log_base, x)
        if found is not None:
            rows, logs = found
            log_value = np.where(rows, self._log_phi(Z, phi_params), 0.0)
            log_value = log_value + logs
            on = rows & onp.asarray(log_value > _LOG_TINY)
            if on.any():
                off = _times_phi(
                    self._phi_off(Z, phi_params, on), np.where(on, 0.0, base)
                )
                # Past exp(709) the product is the infinite H it tends to
                # (sf 0), as on a fit run off along a coefficient (#801).
                with np.errstate(over="ignore"):
                    on_log = np.exp(np.where(on, log_value, 0.0))
                return np.where(on, on_log, off)
        return _times_phi(self.phi(Z, *phi_params), base)

    def log_sf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        return -self.Hf(x, Z, *params)

    def log_df(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """The log density, ``log h - H``, with ``log h`` taken as
        ``log phi + log h0`` where ``h0`` is very small (see above)."""
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])
        h0 = self.hf_dist(x, *dist_params)
        found = _small_on_log_scale(h0, self._log_h0(dist_params), x)
        if found is None:
            log_h = np.log(_times_phi(self.phi(Z, *phi_params), h0))
        else:
            rows, log_small = found
            log_phi = np.where(rows, self._log_phi(Z, phi_params), 0.0)
            phi = self._phi_off(Z, phi_params, rows)
            with np.errstate(divide="ignore"):
                off = np.log(_times_phi(phi, np.where(rows, 1.0, h0)))
            log_h = np.where(rows, log_phi + log_small, off)
        return log_h - self.Hf(x, Z, *params)

    def mpp_inv_y_transform(self, y: Numeric, *params: Boxable) -> Numeric:
        return y

    def mpp_y_transform(self, y: Numeric, *params: Boxable) -> Numeric:
        return y

    def random(
        self,
        size: int,
        Z: npt.ArrayLike,
        *params: float,
        random_state: Any = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Draw ``size`` samples for each covariate row of ``Z``.

        Returns the draws and a 2-D array of the covariate row each was
        drawn at, row by row. ``random_state`` seeds the draw: ``None``
        (the default) draws from numpy's global generator, so
        ``np.random.seed`` reproduces it; an int or a
        ``numpy.random.Generator`` gives a stream of its own.
        """
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])
        Z_arr = np.atleast_2d(np.asarray(Z, dtype=float))
        # One stream for every row: a seed given as an int would otherwise
        # restart, and give every row the same uniforms.
        rng = None if random_state is None else as_generator(random_state)
        x = []
        Z_out = []
        for row in Z_arr:
            phi = self.phi(row, *phi_params)
            U = uniform_draws(size, rng)
            x.append(
                self._invert_cumulative_hazard(-np.log(U) / phi, dist_params)
            )
            Z_out.append(np.tile(row, (size, 1)))
        return np.concatenate(x), np.vstack(Z_out)

    def _invert_cumulative_hazard(
        self, h: npt.NDArray, dist_params: npt.NDArray
    ) -> npt.NDArray:
        """
        The times at which the baseline cumulative hazard reaches ``h``.

        ``S(x|Z) = S0(x)^phi``, so a draw ``U`` of the survival is reached
        where ``H0(x) = -log(U) / phi``. Through the quantile function that
        is ``qf(1 - exp(-h))``; for a very small hazard multiplier ``h`` is
        so large that ``1 - exp(-h)`` rounds to 1 and ``qf`` returns
        ``inf``, although the time is finite. Those draws are solved on
        ``log H0(x) = log h`` directly, all at once (#585: a ``brentq``
        per draw took 2.4 s for 2000 draws), to ``rtol=1e-12``. A draw
        whose time is beyond the largest float (the bracket doubles to
        ``inf``) is ``inf``, as ``qf`` gives; the search raised there.
        """
        h = np.asarray(h, dtype=float)
        with onp.errstate(divide="ignore", over="ignore", invalid="ignore"):
            out = np.asarray(
                self.dist.qf(-np.expm1(-h), *dist_params), dtype=float
            )
        lost = np.flatnonzero(~np.isfinite(out) & np.isfinite(h))
        if lost.size:
            out = out.copy()
            out[lost] = self._solve_log_cumulative_hazard(
                onp.log(h[lost]), dist_params
            )
        return out

    def _solve_log_cumulative_hazard(
        self, target: npt.NDArray, dist_params: npt.NDArray
    ) -> npt.NDArray:
        """The times at which ``log H0(x)`` reaches each ``target``: each
        upper bracket doubles from the baseline median (or 1) until it is
        reached, the lower end half of it; then all are solved together."""

        def gap(t: npt.NDArray, sel: Any) -> npt.NDArray:
            with onp.errstate(all="ignore"):
                H = onp.asarray(self.dist.Hf(t, *dist_params), dtype=float)
                return onp.log(H) - target[sel]

        start = float(self.dist.qf(0.5, *dist_params))
        upper = onp.full(target.shape, max(start, 1.0))
        reached = onp.zeros(target.shape, dtype=bool)
        active = onp.arange(target.size)
        for _ in range(2000):
            if not active.size:
                break
            hit = gap(upper[active], active) >= 0
            reached[active[hit]] = True
            active = active[~hit]
            with onp.errstate(over="ignore"):
                upper[active] *= 2.0
            active = active[onp.isfinite(upper[active])]
        out = onp.full(target.shape, onp.inf)
        k = onp.flatnonzero(reached)
        hi = upper[k]
        lo = onp.where(hi > start, hi / 2.0, 0.0)
        g_lo, g_hi = gap(lo, k), gap(hi, k)
        out[k] = onp.where(g_lo == 0, lo, hi)
        open_ = (g_lo < 0) & (g_hi > 0)
        if open_.any():
            sel = k[open_]
            out[sel] = solve_bracketed(
                lambda t, s: gap(t, sel[s]),
                lo[open_],
                hi[open_],
                g_lo[open_],
                g_hi[open_],
                xtol=1e-300,
                rtol=1e-12,
            )
        return out

    def neg_ll(self, data: SurpyvalData, *params: Boxable) -> Boxable:
        return regression_neg_ll(self, data, *params)

    @staticmethod
    def create(distribution: Any) -> "ProportionalHazardsFitter":
        """
        Create a Proportional Hazards fitter for the given distribution using
        exp(beta'Z) as the hazard multiplier.

        Parameters
        ----------
        distribution : ParametricFitter
            A surpyval parametric distribution (e.g. ``Weibull``,
            ``Exponential``).

        Returns
        -------
        ProportionalHazardsFitter
            A configured fitter with a ``.fit(x, Z, ...)`` method.
        """
        return ProportionalHazardsFitter.create_general_log_linear_fitter(
            f"{distribution.name}PH", distribution
        )

    @classmethod
    def create_general_log_linear_fitter(
        cls, name: str, distribution: Any
    ) -> "ProportionalHazardsFitter":
        return cls(
            name,
            distribution,
            LogLinearPhi.phi,
            LogLinearPhi.NAME_E,
            LogLinearPhi.phi_bounds,
            phi_param_map=LogLinearPhi.make_param_map,
            phi_init=_zero_init,
        )

    @dataframe_covariates
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        init: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
        center: bool = False,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
    ) -> ParametricRegressionModel:
        """
        Fit the proportional hazards model to the data.

        Parameters
        ----------

        x : array_like
            The observed event times.
        Z : array_like
            The covariates to fit the model to, one row per observation.
            Rows with a missing or infinite covariate are dropped, with a
            warning.
        c : array_like, optional
            The censoring indicators.
        n : array_like, optional
            The number of observations at each time.
        t : array_like, optional
            Truncation bounds: an (N, 2) array of the left and right
            truncation times of each observation.
        tl, tr : array_like or float, optional
            The left / right truncation times of each observation (or one
            for every observation), the columns of ``t``, which they
            replace (#662).
        init : array_like, optional
            The initial values for the parameters: the distribution
            parameters followed by the covariate coefficients.
        fixed : dict, optional
            A dictionary of parameters to fix to a specific value, by name
            (a distribution parameter such as ``"beta"``, or a coefficient:
            its covariate's column name, else ``"coef_0"``, ``"coef_1"``,
            ...).
        center : bool, optional
            ``False`` (the default) reports the baseline at ``Z = 0``.
            ``True`` reports the baseline at the covariate means (stored as
            ``model.center``) instead: the fit runs on ``Z - center``, and
            ``init`` and ``fixed`` are read there too. Use it for
            covariates far from 0 (a year, a date), where the baseline at
            ``Z = 0`` cannot be represented or fitted, which the default
            fit refuses with a ``ValueError`` saying so.

        Returns
        -------

        ParametricRegressionModel
            The fitted model.

        Examples
        --------

        >>> from surpyval import WeibullPH
        >>> from surpyval.datasets import load_tires_data
        >>> data = load_tires_data()
        >>> x = data['Survival'].values
        >>> c = data['Censoring'].values
        >>> Z = data[[
        ...     'Wedge gauge', 'Interbelt gauge', 'Peel force',
        ...     'Wedge gauge×peel force'
        ... ]].values
        >>> model = WeibullPH.fit(x=x, Z=Z, c=c)
        >>> model.summary()[["coef", "se(coef)", "p"]].round(4)
                                coef  se(coef)       p
        part         name
        baseline     alpha    0.2426    0.0814     NaN
                     beta    16.0578    3.9506     NaN
        coefficients coef_0  -9.1651    3.7237  0.0138
                     coef_1  -7.9986    2.8119  0.0044
                     coef_2 -27.5032    9.5366  0.0039
                     coef_3  18.3854    6.4222  0.0042
        >>> model = WeibullPH.fit(x=x, Z=Z, c=c, fixed={"beta": 15})
        >>> model.params.round(4)
        array([  0.2377,  15.    ,  -8.6283,  -7.6175, -25.9524,  17.2701])
        """
        t = truncation_window(x, t, tl, tr)
        return fit_log_linear(
            self,
            x,
            Z,
            c,
            n,
            t,
            init,
            fixed,
            center,
            kind=PROPORTIONAL_HAZARD,
            optimiser=optimise_ph,
            reg_model=self._reg_model,
            phi_bounds=self.phi_bounds,
            phi_param_map=self.phi_param_map,
            phi_init=self.phi_init,
            # Only the log-linear multiplier can be centred and reported at
            # Z = 0 (#463); a custom phi is centred only with center=True.
            log_linear=self.phi is LogLinearPhi.phi,
        )

    def _reg_model(self, pmap: dict[str, int]) -> CovariateLink:
        # Keep this fitter's possibly-custom phi (and its historical
        # serialisation name) rather than assuming log-linear.
        return CovariateLink(self.phi_name, pmap, self.phi)
