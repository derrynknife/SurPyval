"""The fitted joint model from ``Copula.fit`` / ``Copula.from_params``."""

from __future__ import annotations

from typing import Any

import numpy as onp
import numpy.typing as npt

from surpyval.distribution import MultivariateDistribution
from surpyval.serialisation import SerialisableMixin, stamp_schema
from surpyval.univariate.information_criteria import (
    corrected_aic,
    ic_sample_size,
)
from surpyval.utils.linalg import (
    sf_link_bound,
    wald_bound_on_support,
    warn_wald_undefined,
)
from surpyval.utils.no_maximum import maximum_entry, restored_maximum
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import (
    BOUNDS,
    alpha_ci_error,
    check_option,
    no_covariance_error,
)
from surpyval.utils.warnings import warn_no_covariance

from ._inference import ParameterLayout, jacobian, joint_covariance

# Margin probabilities are kept strictly inside (0, 1), as in copula.py.
_U_CLIP = 1e-10
# The joint functions ``cb`` bounds ('R' and 'F' are the aliases of 'sf'
# and 'ff', as for the univariate models).
_CB_ON = ("sf", "R", "ff", "F")
# A Wald bound's methods: the delta method alone, for now.
_CB_METHODS = ("wald",)


class CopulaModel(SerialisableMixin, MultivariateDistribution):
    """A fitted bivariate copula glued to two univariate margins.

    Attributes
    ----------
    copula : Copula
        The copula family.
    params : numpy.ndarray
        The fitted copula parameter(s) (empty for the independence copula).
    margins : list
        The fitted margin models (each exposes ``ff``/``df``/``qf``).
    k : int or None
        The number of parameters the fit estimated: the copula's plus those
        of every margin the fit estimated (all of them for ``how="MLE"``;
        under ``how="IFM"`` those passed as distributions, not a margin
        passed already fitted). ``None`` for ``from_params``.

    Examples
    --------
    ``Copula.fit`` and ``Copula.from_params`` return one:

    >>> from surpyval import Weibull
    >>> from surpyval.multivariate import Clayton
    >>> margins = [
    ...     Weibull.from_params([10, 2]),
    ...     Weibull.from_params([20, 3]),
    ... ]
    >>> model = Clayton.from_params([2.0], margins)
    >>> model
    Copula SurPyval Model
    =====================
    Copula    : Clayton
    Parameters: theta=2
    Margins   : Weibull, Weibull
    Fitted by : given
    >>> model.sf([[5, 15], [10, 20]]).round(4)
    array([0.624 , 0.2354])
    >>> round(float(model.kendall_tau()), 3)
    0.5
    """

    def __init__(
        self,
        copula: Any,
        params: npt.ArrayLike,
        margins: Any,
        data: Any = None,
        how: str = "given",
        k: "int | None" = None,
    ) -> None:
        self.copula = copula
        self.params = onp.atleast_1d(onp.asarray(params, dtype=float))
        self.margins = list(margins)
        self.data = data
        self.method = how
        self.k = k
        # The fitted negative log-likelihood and BIC's sample size, computed
        # on first use from ``data`` (or restored by ``from_dict``, which has
        # no data).
        self._neg_ll: "float | None" = None
        self._n_obs: "float | None" = None
        # What the fit reached, one of ``MAXIMUM_STATES``
        # (``surpyval.utils.no_maximum``), as its warnings say: set by the
        # fit; "not applicable" for a model built from its parameters,
        # "unknown" for one restored from a dict saved without it.
        self.maximum = "not applicable"
        # Which margins the fit estimated (all of them for ``how="MLE"``;
        # under IFM those passed as distributions, not already fitted):
        # the others are known, with no variance. Set by the fit.
        self._margins_estimated: tuple = (how == "MLE",) * len(self.margins)
        # The covariance of every parameter, copula's and margins' (see
        # ``covariance``): computed on first use, or restored by
        # ``from_dict``.
        self._covariance: "npt.NDArray | None" = None

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``, entry by entry: the copula family's
        ``parameter_names`` (``["theta"]``, ``["rho"]``, or ``[]`` for the
        independence copula). The margins' parameters are on the margins.
        """
        return list(self.copula.parameter_names)

    # -- internal ---------------------------------------------------------
    def _uv(
        self, x: npt.ArrayLike
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        x = onp.atleast_2d(onp.asarray(x, dtype=float))
        if x.shape[1] != 2:
            raise ValueError("x must have two columns (one per dimension)")
        u = onp.clip(self.margins[0].ff(x[:, 0]), _U_CLIP, 1 - _U_CLIP)
        v = onp.clip(self.margins[1].ff(x[:, 1]), _U_CLIP, 1 - _U_CLIP)
        return x, u, v

    def _copula_cdf(self, u: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
        if u.size == 0:
            # No points (scipy's multivariate normal refuses zero rows).
            return onp.empty(0)
        return onp.asarray(self.copula.cdf(u, v, *self.params))

    # -- joint survival interface ----------------------------------------
    @keeps_query_shape(point_ndim=1)
    def cdf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Joint CDF ``P(X_1 <= x_1, X_2 <= x_2)``."""
        _, u, v = self._uv(x)
        return self._copula_cdf(u, v)

    @keeps_query_shape(point_ndim=1)
    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Joint survival ``P(X_1 > x_1, X_2 > x_2)``: the copula's upper
        quadrant, evaluated directly rather than as ``1 - u - v + C(u, v)``,
        which lost every digit where it is small (#619)."""
        _, u, v = self._uv(x)
        if u.size == 0:
            return onp.empty(0)
        return onp.asarray(self.copula._survival(u, v, *self.params))

    @keeps_query_shape(point_ndim=1)
    def pdf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Joint density ``c(F_1, F_2) f_1 f_2``."""
        x, u, v = self._uv(x)
        c = onp.asarray(self.copula.pdf(u, v, *self.params))
        f1 = onp.asarray(self.margins[0].df(x[:, 0]))
        f2 = onp.asarray(self.margins[1].df(x[:, 1]))
        return c * f1 * f2

    @keeps_query_shape(point_ndim=1)
    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        """Alias of :meth:`cdf` for consistency with surpyval naming."""
        return self.cdf(x)

    @keeps_query_shape(point_ndim=1)
    def conditional_cdf(
        self, x: npt.ArrayLike, given_dim: int = 0
    ) -> npt.NDArray:
        """``P(X_other <= x_other | X_d = x_d)`` -- the copula h-function.

        ``given_dim=0`` conditions on the first series, giving
        :math:`P(X_2 \\le x_2 \\mid X_1 = x_1)`; ``given_dim=1``
        conditions on the second. Any other value raises ``ValueError``
        (it used to be read silently as ``1``).
        """
        if given_dim not in (0, 1):
            raise ValueError(
                f"given_dim must be 0 or 1 (the series conditioned on), got "
                f"{given_dim!r}."
            )
        x, u, v = self._uv(x)
        if given_dim == 0:
            return onp.asarray(self.copula.du(u, v, *self.params))
        return onp.asarray(self.copula.dv(u, v, *self.params))

    # -- sampling ---------------------------------------------------------
    def random(
        self,
        size: "int | tuple[int, ...]",
        random_state: "int | None" = None,
    ) -> npt.NDArray:
        """Draw correlated samples: an array of shape ``(size, 2)`` for an
        integer ``size``, one row per draw, or ``(*size, 2)`` for a tuple
        (the two series on the last axis)."""
        shape: tuple[int, ...] = (
            (int(size),)
            if isinstance(size, (int, onp.integer))
            else tuple(size)
        )
        count = int(onp.prod(shape))
        # Draw flat, then shape: the margins' quantile functions and
        # ``column_stack`` treat a 2-D draw as extra columns, so a (2, 3)
        # request came back as (2, 6).
        u, v = self.copula.sample_uv(count, self.params, random_state)
        x1 = onp.asarray(self.margins[0].qf(u), dtype=float).ravel()
        x2 = onp.asarray(self.margins[1].qf(v), dtype=float).ravel()
        return onp.column_stack([x1, x2]).reshape(shape + (2,))

    # -- dependence summaries --------------------------------------------
    def kendall_tau(self) -> float:
        """Kendall's rank correlation implied by the fitted copula."""
        return self.copula.kendall_tau(*self.params)

    def spearman_rho(self) -> float:
        """Spearman's rank correlation implied by the fitted copula."""
        return self.copula.spearman_rho(*self.params)

    def tail_dependence(self) -> tuple:
        """The lower and upper tail-dependence coefficients
        ``(lambda_L, lambda_U)`` of the fitted copula."""
        return self.copula.tail_dependence(*self.params)

    # -- likelihood and information criteria ------------------------------
    def _has_likelihood(self) -> bool:
        # Fitted to data, or restored from a dict that stored the fit's
        # likelihood; not built by ``from_params``.
        return self.k is not None and (
            self.data is not None or self._neg_ll is not None
        )

    def _fitted_k(self) -> int:
        """The estimated-parameter count; raises for a model with no
        likelihood."""
        if self.k is None or not self._has_likelihood():
            raise ValueError(
                "The log-likelihood is only available for a model fitted to "
                "data with `fit`, not one built with `from_params`."
            )
        return self.k

    def neg_ll(self) -> float:
        """
        The negative log-likelihood of the fitted model: the full joint
        likelihood of the data it was fitted to, with each row's censoring
        (right, left, interval, per series), its truncation and its count
        ``n``, and the margins' densities for the observed entries. Raises
        ``ValueError`` for a ``from_params`` model.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Clayton
        >>> margins = [
        ...     Weibull.from_params([10, 2]),
        ...     Weibull.from_params([20, 3]),
        ... ]
        >>> X = Clayton.from_params([2.0], margins).random(300, random_state=0)
        >>> model = Clayton.fit(X, margins=[Weibull, Weibull])
        >>> round(model.neg_ll(), 3)
        1729.371
        >>> model.k, round(model.aic(), 3)
        (5, 3468.741)
        """
        return self._fit_stats()[0]

    def _fit_stats(self) -> tuple[float, float]:
        """``(neg_ll, n_obs)``: the negative log-likelihood and BIC's sample
        size (see :meth:`bic`), computed from the data once (or restored by
        :meth:`from_dict`)."""
        self._fitted_k()
        if self._neg_ll is None or self._n_obs is None:
            dims = [
                self.copula._prepare_dim(
                    self.margins[d], *self.data.dimension(d)
                )
                for d in range(self.data.D)
            ]
            self._neg_ll = self.copula.neg_ll(self.params, dims, self.data.n)
            # The shared rule of every SurPyval BIC; a joint row counts
            # when any of its series failed. It was every row here.
            self._n_obs = ic_sample_size(self.data.c, self.data.n)
        return float(self._neg_ll), float(self._n_obs)

    @property
    def log_likelihood(self) -> float:
        """The maximised log-likelihood, ``-neg_ll()``."""
        return -self.neg_ll()

    def aic(self) -> float:
        """
        Akaike's information criterion, :math:`2k - 2\\ln L`, with ``k`` the
        number of estimated parameters (see the class docstring). Lower is
        better.
        """
        return 2.0 * self._fitted_k() + 2.0 * self.neg_ll()

    def bic(self) -> float:
        """
        The Bayesian information criterion, :math:`k \\ln N - 2\\ln L`, with
        ``N`` the number of joint observations (rows, weighted by ``n``) in
        which at least one series failed -- was observed exactly, or left-
        or interval-censored -- or, when no row has a failure, the number
        of rows. It is the sample size of every SurPyval BIC: a unit
        right-censored in every series adds nothing. Lower is better.
        """
        neg_ll, n_obs = self._fit_stats()
        return float(self._fitted_k() * onp.log(n_obs) + 2.0 * neg_ll)

    def aic_c(self) -> float:
        """
        The small-sample corrected AIC, ``aic() + (2k^2 + 2k) / (N - k -
        1)``, with the ``k`` of :meth:`aic` and the ``N`` of :meth:`bic`;
        ``nan`` where ``N <= k + 1`` (``corrected_aic``), as on every
        other model (#605).
        """
        _, n_obs = self._fit_stats()
        return corrected_aic(self.aic(), self._fitted_k(), n_obs)

    # -- uncertainty (#540) -----------------------------------------------
    def _parameter_layout(self) -> ParameterLayout:
        return ParameterLayout(self, self._margins_estimated)

    def _parameter_bounds(self) -> list:
        """``(lower, upper)`` of each copula parameter, in the order of
        ``parameter_names`` (as the regression and recurrent models give
        theirs): the space :meth:`param_cb`'s bounds stay inside."""
        return list(self.copula.bounds)

    def _joint_covariance(self) -> npt.NDArray:
        """The covariance of every parameter (see :meth:`covariance`),
        computed once; raises ``ValueError`` where the model has none."""
        if self._covariance is not None:
            return self._covariance
        if self.k is None:
            raise no_covariance_error(
                "it was built from its parameters with `from_params`, not "
                "fitted to data"
            )
        if self.data is None:
            raise no_covariance_error(
                "it was restored from a dictionary saved without one, and "
                "a restored model keeps no data to compute it from: refit "
                "the model for its standard errors and bounds"
            )
        if any(not hasattr(m, "dist") for m in self.margins):
            raise no_covariance_error(
                "a non-parametric margin makes the fit the semi-parametric "
                "pseudo-likelihood estimator, whose rank-based variance "
                "(Genest, Ghoudi and Rivest 1995) SurPyval does not "
                "compute; fit parametric margins for Wald bounds"
            )
        cov, _ = joint_covariance(
            self._parameter_layout(), self.data, self.method
        )
        self._covariance = cov
        return cov

    def covariance(self, margins: bool = False) -> npt.NDArray:
        """
        The covariance of the fitted copula parameters, in the order of
        ``parameter_names``; with ``margins=True``, of every parameter of
        the model: the copula's, then each margin's in the order of that
        margin's own ``covariance()`` (the distribution's parameters,
        then a limited-failure ``p`` and a zero-inflation ``f0``).

        It is the inverse of the observed information, in the form the
        fit calls for:

        - ``how="MLE"``: the inverse of the Hessian of the joint negative
          log-likelihood in every parameter, the copula's and the margins'
          together.
        - ``how="IFM"``: the Godambe (sandwich) covariance of the two
          stages (Joe 2005), :math:`H^{-1} J H^{-T}`, where, with
          :math:`\\psi_i` the stacked score of row :math:`i` -- each
          margin's score in its own parameters, then the copula stage's
          score in the copula's -- :math:`H = -\\sum_i n_i \\partial
          \\psi_i / \\partial \\theta` (block triangular: a margin's score
          does not depend on the copula or the other margin) and
          :math:`J = \\sum_i n_i \\psi_i \\psi_i^T`. The copula stage's
          Hessian alone would treat the margins as known and understate
          the copula's variance (by about a fifth in standard error for
          a Clayton of Kendall's tau 0.5 with Weibull margins, 300 rows);
          the blocks of :math:`H` of the copula's score in the margins'
          parameters carry their uncertainty into it.

        The derivatives are central finite differences (the copula
        likelihood is not written for autograd). A margin's offset
        ``gamma``, a parameter fixed at fit time and every parameter of a
        margin passed to an IFM fit already fitted are known: zero row and
        column. A parameter on a bound of its space (an AMH ``theta`` of
        1) has no Wald variance: ``nan`` row and column. Where the
        information cannot be evaluated or inverted the estimated block
        is ``nan``, with a warning.

        Raises a ``ValueError`` where the model has none: one built with
        ``from_params``, one restored from a dictionary saved without it,
        or one fitted with a non-parametric margin.

        Parameters
        ----------
        margins : bool, optional
            Include the margins' parameters. Default ``False``.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Clayton
        >>> margins = [
        ...     Weibull.from_params([10, 2]),
        ...     Weibull.from_params([20, 3]),
        ... ]
        >>> X = Clayton.from_params([2.0], margins).random(300, random_state=0)
        >>> model = Clayton.fit(X, margins=[Weibull, Weibull])
        >>> model.covariance().round(4)
        array([[0.0496]])
        >>> model.covariance(margins=True).shape
        (5, 5)
        """
        cov = self._joint_covariance()
        # (A parameter on a bound has no variance by design.)
        if onp.isnan(
            onp.diag(cov)[~self._parameter_layout().on_bound()]
        ).any():
            warn_no_covariance()
        n_cop = len(self.parameter_names)
        return onp.array(cov if margins else cov[:n_cop, :n_cop])

    def standard_errors(self, margins: bool = False) -> npt.NDArray:
        """
        Standard errors of the fitted copula parameters (with
        ``margins=True``, of every parameter of the model): the square
        roots of the diagonal of :meth:`covariance`, in its order.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Clayton
        >>> margins = [
        ...     Weibull.from_params([10, 2]),
        ...     Weibull.from_params([20, 3]),
        ... ]
        >>> X = Clayton.from_params([2.0], margins).random(300, random_state=0)
        >>> model = Clayton.fit(X, margins=[Weibull, Weibull], how="MLE")
        >>> model.standard_errors().round(4)
        array([0.2215])
        """
        with onp.errstate(invalid="ignore"):
            return onp.sqrt(onp.diag(self.covariance(margins=margins)))

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> npt.NDArray:
        """
        Confidence bound(s) on a copula parameter.

        The Wald bound from :meth:`covariance` (the joint information for
        ``how="MLE"``, the Godambe information for ``how="IFM"``), formed
        on a scale on which the parameter's space is the whole line, as
        the univariate and regression models' are, so the bound stays
        inside it: the log of the distance from the bound for a parameter
        with one (a Clayton's ``theta > 0``, a Gumbel's ``theta >= 1``, a
        Student-t's ``nu``), the generalised logit for one in an interval
        (on ``(-1, 1)`` it is ``2 artanh(rho)``, Fisher's z, for a Gaussian
        or Student-t ``rho``), and the parameter itself for a Frank
        ``theta``. Where the bound does not exist -- the parameter is on a
        bound of its space, or its variance is not positive -- it is
        ``nan``, with a warning.

        Parameters
        ----------
        name : str
            The parameter, one of ``parameter_names``.
        alpha_ci : float, optional
            The significance level: 0.05 (the default) gives a 95% bound.
        bound : str, optional
            ``"two-sided"`` (the default), ``"upper"`` or ``"lower"``.
        method : str, optional
            ``"wald"``, the only method (the default).

        Returns
        -------
        numpy array
            ``[lower, upper]`` for a two-sided bound, else the one bound.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Clayton
        >>> margins = [
        ...     Weibull.from_params([10, 2]),
        ...     Weibull.from_params([20, 3]),
        ... ]
        >>> X = Clayton.from_params([2.0], margins).random(300, random_state=0)
        >>> model = Clayton.fit(X, margins=[Weibull, Weibull])
        >>> model.param_cb("theta").round(3)
        array([1.896, 2.774])
        """
        check_option("method", method, _CB_METHODS)
        check_option("bound", bound, BOUNDS)
        _check_alpha_ci(alpha_ci)
        names = self.parameter_names
        if name not in names:
            raise ValueError(
                "Unknown parameter {!r}; expected one of {}".format(
                    name, names
                )
            )
        i = names.index(name)
        var = float(self._joint_covariance()[i, i])
        if onp.isnan(var) and self._parameter_layout().on_bound()[i]:
            # Its place on the bound, not a variance, is why there is no
            # bound: the warning says so.
            var = 0.0
        lower, upper = self.copula.bounds[i]
        return wald_bound_on_support(
            float(self.params[i]), var, lower, upper, alpha_ci, bound, name
        )

    @keeps_query_shape(point_ndim=1)
    def cb(
        self,
        x: npt.ArrayLike,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> npt.NDArray:
        """
        Confidence bounds on the joint survival ``sf`` or the joint CDF
        ``ff`` at the points ``x``.

        The delta method with the covariance of every parameter of the
        model (:meth:`covariance` with ``margins=True``), so the margins'
        uncertainty is in the bound as well as the copula's, on the logit
        scale of the probability (as the univariate models' bounds are
        for a family with no probability-plot scale): it stays in
        ``(0, 1)``, and is formed from the probability and its complement
        each where it is the smaller, so it is accurate in both tails.
        Where the delta-method variance is not positive (the covariance is
        not positive definite, or a parameter is on a bound of its space)
        the bound is ``nan``, with a warning.

        Parameters
        ----------
        x : array like
            The points ``(x_1, x_2)``: a pair, or one pair per row.
        on : {'sf', 'ff'}, optional
            The joint survival :math:`P(X_1 > x_1, X_2 > x_2)` (``'sf'``,
            the default, also ``'R'``) or the joint CDF :math:`P(X_1 \\le
            x_1, X_2 \\le x_2)` (``'ff'``, also ``'F'``). (The two are not
            complements of each other.)
        alpha_ci : float, optional
            The significance level: 0.05 (the default) gives a 95% bound.
        bound : {'two-sided', 'upper', 'lower'}, optional
            Defaults to two-sided.
        method : {'wald'}, optional
            The delta method, the only method (the default).

        Returns
        -------
        numpy array
            One row per point holding ``[lower, upper]`` for a two-sided
            bound, else the one bound per point.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Clayton
        >>> margins = [
        ...     Weibull.from_params([10, 2]),
        ...     Weibull.from_params([20, 3]),
        ... ]
        >>> X = Clayton.from_params([2.0], margins).random(300, random_state=0)
        >>> model = Clayton.fit(X, margins=[Weibull, Weibull])
        >>> model.sf([[5, 15], [10, 20]]).round(4)
        array([0.6649, 0.2912])
        >>> model.cb([[5, 15], [10, 20]]).round(4)
        array([[0.6183, 0.7085],
               [0.2544, 0.331 ]])
        """
        check_option("on", on, _CB_ON)
        check_option("bound", bound, BOUNDS)
        check_option("method", method, _CB_METHODS)
        _check_alpha_ci(alpha_ci)
        x = onp.atleast_2d(onp.asarray(x, dtype=float))
        if x.shape[1] != 2:
            raise ValueError("x must have two columns (one per dimension)")
        if x.shape[0] == 0:
            return onp.empty((0, 2) if bound == "two-sided" else (0,))
        cov = self._joint_covariance()
        layout = self._parameter_layout()
        joint = "sf" if on in ("sf", "R") else "ff"

        def value(vector: npt.NDArray) -> npt.NDArray:
            parts = layout.build(vector)
            if parts is None:
                return onp.full(x.shape[0], onp.nan)
            return self._joint_at(x, joint, *parts)[0]

        p_hat, complement = self._joint_at(
            x, joint, self.params, self.margins, complement=True
        )
        # Every entry that varies (the known ones have a zero variance)
        entries = onp.flatnonzero(onp.diag(cov) != 0)
        var = onp.zeros(x.shape[0])
        if entries.size:
            with onp.errstate(all="ignore"):
                J = jacobian(value, layout, entries)
            sub = cov[onp.ix_(entries, entries)]
            var = onp.einsum("ij,jk,ik->i", J, sub, J)
            # Rounding can leave a zero variance a hair below zero.
            scale = onp.einsum("ij,jk,ik->i", abs(J), abs(sub), abs(J))
            var = onp.where((var < 0) & (var >= -1e-10 * scale), 0.0, var)
        bad = ~(var >= 0) & onp.isfinite(x).all(axis=1)
        if bad.any():
            warn_wald_undefined(
                f"{joint} at x = {x[bad].tolist()}",
                "its delta-method variance is negative or not finite, so "
                "the parameter covariance is not positive definite (a "
                "parameter is at or near a boundary of its space, or the "
                "likelihood is not regular there)",
                # cb -> the query-shape wrapper -> the caller
                stacklevel=3,
            )
        with onp.errstate(invalid="ignore"):
            se = onp.sqrt(onp.where(var >= 0, var, onp.nan))
        assert complement is not None  # (asked for)
        if joint == "sf":
            return sf_link_bound(
                p_hat, se, alpha_ci, bound, "logit", complement, on="sf"
            )
        return sf_link_bound(
            complement, se, alpha_ci, bound, "logit", p_hat, on="ff"
        )

    def _joint_at(
        self,
        x: npt.NDArray,
        joint: str,
        params: Any,
        margins: list,
        complement: bool = False,
    ) -> "tuple[npt.NDArray, npt.NDArray | None]":
        """``(value, complement)`` of the joint ``sf`` or ``ff`` at the
        points ``x`` for the copula ``params`` and ``margins``. The
        complement (``None`` unless asked for) is from the margins, ``u +
        v - C`` for ``sf`` and ``s_1 + s_2 - S`` for ``ff``, so that it is
        accurate where it is small."""
        copula = self.copula
        with onp.errstate(all="ignore"):
            u = onp.clip(margins[0].ff(x[:, 0]), _U_CLIP, 1 - _U_CLIP)
            v = onp.clip(margins[1].ff(x[:, 1]), _U_CLIP, 1 - _U_CLIP)
            if joint == "sf":
                value = onp.asarray(copula._survival(u, v, *params), float)
                if not complement:
                    return value, None
                cdf = onp.asarray(copula.cdf(u, v, *params), dtype=float)
                return value, u + v - cdf
            value = onp.asarray(copula.cdf(u, v, *params), dtype=float)
            if not complement:
                return value, None
            s1 = onp.clip(margins[0].sf(x[:, 0]), _U_CLIP, 1 - _U_CLIP)
            s2 = onp.clip(margins[1].sf(x[:, 1]), _U_CLIP, 1 - _U_CLIP)
            survival = onp.asarray(copula._survival(u, v, *params), float)
            return value, s1 + s2 - survival

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        """
        Serialise to a plain dictionary: the copula family, its
        parameter(s), the fit method and each margin's own ``to_dict``.
        The data is not stored, but for a fitted model the negative
        log-likelihood, parameter count and BIC's sample size are, so the
        restored model still reports ``neg_ll``/``aic``/``bic``. Restore
        with :meth:`from_dict` or ``surpyval.from_dict``.
        """
        margins = []
        for m in self.margins:
            margins.append(m.to_dict() if hasattr(m, "to_dict") else None)
        out: dict = {
            "parameterization": "copula",
            "copula": self.copula.name,
            "params": self.params.tolist(),
            "how": self.method,
            "margins": margins,
            **maximum_entry(self.maximum),
        }
        if self.copula.rotation:
            out["rotation"] = int(self.copula.rotation)
        if self._has_likelihood():
            # "_neg_ll", the key every model's dict stores it under (#605)
            out["_neg_ll"], out["ic_n"] = self._fit_stats()
            out["k"] = self._fitted_k()
        covariance = self._saved_covariance()
        if covariance is not None:
            # The key every model's dict stores it under (#605): the
            # restored model keeps its standard errors and bounds.
            out["covariance"] = covariance.tolist()
        return stamp_schema(out)

    def _saved_covariance(self) -> "npt.NDArray | None":
        """The covariance :meth:`to_dict` stores: the model's, where it
        has one (computed now if need be), else ``None``."""
        try:
            return self._joint_covariance()
        except ValueError:
            return None

    @classmethod
    def from_dict(cls, model_dict: dict) -> "CopulaModel":
        """Rebuild a copula model from a :meth:`to_dict` dictionary."""
        import surpyval

        from .archimedean import AMH, Clayton, Frank, Gumbel, Independence, Joe
        from .elliptical import Gaussian, StudentT

        families = {
            c.name: c
            for c in (
                Independence,
                Clayton,
                Gumbel,
                Frank,
                Gaussian,
                Joe,
                AMH,
                StudentT,
            )
        }
        copula_name = model_dict["copula"]
        if copula_name not in families:
            raise ValueError(
                f"Unknown copula family {copula_name!r}; expected one of "
                f"{sorted(families)}. A custom copula cannot be rebuilt "
                "from its name alone."
            )
        margins = []
        for i, m in enumerate(model_dict["margins"]):
            if m is None:
                raise ValueError(
                    f"Margin {i} was not serialisable (it has no `to_dict`), "
                    "so this copula model cannot be rebuilt."
                )
            margins.append(surpyval.from_dict(m))
        family = families[copula_name].rotated(model_dict.get("rotation", 0))
        model = cls(
            family,
            model_dict["params"],
            margins,
            data=None,
            how=model_dict.get("how", "given"),
            k=model_dict.get("k"),
        )
        model.maximum = restored_maximum(model_dict)
        if "covariance" in model_dict:
            size = len(model_dict["covariance"])
            model._covariance = onp.array(
                model_dict["covariance"], dtype=float
            ).reshape(size, size)
        # Dicts written before the likelihood was stored have none; such a
        # model (like a ``from_params`` one) has no likelihood to report.
        # "neg_ll" is the key of a dict written before v0.23.
        key = "_neg_ll" if "_neg_ll" in model_dict else "neg_ll"
        if key in model_dict:
            model._neg_ll = float(model_dict[key])
            # Dicts written before "ic_n" stored the weighted row count,
            # the sample size BIC then used.
            model._n_obs = float(
                model_dict["ic_n"]
                if "ic_n" in model_dict
                else model_dict["n_obs"]
            )
        return model

    def __repr__(self) -> str:
        param_str = ", ".join(
            f"{n}={p:.4g}" for n, p in zip(self.parameter_names, self.params)
        )
        margin_names = [
            getattr(getattr(m, "dist", m), "name", "?") for m in self.margins
        ]
        family = self.copula.name
        if self.copula.rotation:
            family += f" (rotated {self.copula.rotation} degrees)"
        return (
            "Copula SurPyval Model"
            "\n====================="
            f"\nCopula    : {family}"
            f"\nParameters: {param_str if param_str else '(none)'}"
            f"\nMargins   : {', '.join(margin_names)}"
            f"\nFitted by : {self.method}"
        )


def _check_alpha_ci(alpha_ci: float) -> None:
    """Refuse a significance level outside (0, 1)."""
    if not 0.0 < alpha_ci < 1.0:
        raise alpha_ci_error(alpha_ci)
