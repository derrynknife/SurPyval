"""Base copula and the censoring/truncation-aware joint likelihood.

A bivariate copula ``C(u, v; theta)`` links two uniform margins. With
real margins ``F_1, F_2`` and ``u = F_1(x_1)``, ``v = F_2(x_2)`` the joint
CDF is ``H(x_1, x_2) = C(F_1(x_1), F_2(x_2))``. Every censoring/truncation
type reduces to evaluating ``C`` and its partial derivatives at the
margin-transformed bounds -- so the whole likelihood is assembled from just
four primitives::

    C(u, v)              the copula CDF
    du = dC/du           the h-function P(V <= v | U = u)
    dv = dC/dv
    c  = d2C/du dv        the copula density

Each observed dimension contributes ``d/du`` (and its margin density);
each right/left/interval-censored dimension contributes a difference of
``C`` evaluated at its bounds. The per-dimension operators below make that
bookkeeping uniform across all 16 bivariate censoring combinations.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as onp
import numpy.typing as npt
from autograd import elementwise_grad
from scipy.optimize import minimize

from surpyval import np
from surpyval.utils.dataframe import (
    call_fit,
    frame_column,
    frame_columns,
    require_frame,
)
from surpyval.utils.deprecation import (
    RenamedAttribute,
    renamed_class_attribute,
)
from surpyval.utils.no_maximum import warn_no_maximum
from surpyval.utils.rng import as_generator

# Margin probabilities are kept strictly inside (0, 1): the Archimedean
# generators blow up at the boundary and the optimiser only ever needs
# interior values.
_EPS = 1e-10
_TINY = 1e-300


class Copula:
    """Bivariate copula family.

    Subclasses define :meth:`cdf` (and, for speed/stability, may override the
    partial derivatives, dependence measures and sampler). The fitting,
    likelihood and default autograd-based derivatives live here.
    """

    name: str = "Copula"
    # Parameter bounds in the same ``(low, high)`` form the univariate
    # fitters use, so ``bounds_convert`` can map them to unbounded space.
    bounds: tuple = ((0, None),)
    parameter_names: list[str] = ["theta"]
    # ``param_names``, the pre-0.22 name of ``parameter_names``, reads it
    # for one release, with a DeprecationWarning.
    param_names = RenamedAttribute("parameter_names")
    #: The Frechet bounds the family reaches only as its parameter runs to
    #: a limit: ``+1`` the comonotone copula (perfect positive dependence),
    #: ``-1`` the countermonotone one, each mapped to that limit as the
    #: fit's warning names it (see :meth:`_warn_if_perfectly_dependent`).
    #: Empty for a family that is not known to reach either.
    dependence_limits: dict = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # A family written against the pre-0.22 name, ``param_names``,
        # still works until v0.23, with a DeprecationWarning.
        super().__init_subclass__(**kwargs)
        renamed_class_attribute(cls, "param_names", "parameter_names")

    # -- the four copula primitives ---------------------------------------
    def cdf(self, u: Any, v: Any, *params: Any) -> Any:
        """The copula :math:`C(u, v)`; defined by each family."""
        raise NotImplementedError

    def du(self, u: Any, v: Any, *params: Any) -> Any:
        """``dC/du`` -- the h-function :math:`P(V \\le v \\mid U = u)`.

        The default differentiates :meth:`cdf` with autograd; override it
        when a closed form is known. ``u`` and ``v`` broadcast against each
        other, and the result has their common shape.
        """
        # autograd's elementwise gradient is the gradient of the *sum* of
        # the outputs, so it is only elementwise when each output depends on
        # its own input alone: broadcast first, or a scalar ``u`` with an
        # array ``v`` returned one summed derivative.
        u_b, v_b = _broadcast_pair(u, v)
        return elementwise_grad(lambda a: self.cdf(a, v_b, *params))(
            onp.asarray(u_b, dtype=float)
        )

    def dv(self, u: Any, v: Any, *params: Any) -> Any:
        """``dC/dv``, the h-function :math:`P(U \\le u \\mid V = v)`;
        ``u`` and ``v`` broadcast as for :meth:`du`."""
        u_b, v_b = _broadcast_pair(u, v)
        return elementwise_grad(lambda b: self.cdf(u_b, b, *params))(
            onp.asarray(v_b, dtype=float)
        )

    def pdf(self, u: Any, v: Any, *params: Any) -> Any:
        """``d2C/du dv`` -- the copula density. The default differentiates
        :meth:`du` with autograd; ``u`` and ``v`` broadcast as for
        :meth:`du`."""
        u_b, v_b = _broadcast_pair(u, v)
        return elementwise_grad(lambda b: self.du(u_b, b, *params))(
            onp.asarray(v_b, dtype=float)
        )

    # -- dependence measures (closed-form overrides preferred) ------------
    def kendall_tau(self, *params: float) -> float:
        """Kendall's tau.

        The default integrates :math:`\\tau = 1 - 4 \\int_0^1 \\int_0^1
        \\frac{\\partial C}{\\partial u} \\frac{\\partial C}{\\partial v}
        \\, du \\, dv` by Gauss-Legendre quadrature (400 nodes per margin:
        accurate to about 1e-10 at a tau of 0.5 and 1e-8 at 0.8, the
        integrand sharpening along the diagonal as the dependence grows);
        families with a closed form override it. It used to estimate tau
        from 50 000 simulated pairs, with an error near 1e-3.
        """
        u, v, w = _quadrature_grid()
        du = onp.asarray(self.du(u, v, *params), dtype=float)
        dv = onp.asarray(self.dv(u, v, *params), dtype=float)
        return float(1.0 - 4.0 * onp.sum(w * du * dv))

    def spearman_rho(self, *params: float) -> float:
        """Spearman's rho.

        The default integrates :math:`\\rho_S = 12 \\int_0^1 \\int_0^1
        C(u, v) \\, du \\, dv - 3` by Gauss-Legendre quadrature (400 nodes
        per margin, accurate to about 1e-11 for the built-in families);
        families with a closed form override it. It used to estimate rho
        from 50 000 simulated pairs, which was up to 5e-3 off (Clayton,
        Gumbel).
        """
        u, v, w = _quadrature_grid()
        C = onp.asarray(self.cdf(u, v, *params), dtype=float)
        return float(12.0 * onp.sum(w * C) - 3.0)

    def tail_dependence(self, *params: float) -> tuple:
        """Lower/upper tail-dependence coefficients ``(lambda_L, lambda_U)``.

        Default ``(0.0, 0.0)`` (no tail dependence); families override.
        """
        return (0.0, 0.0)

    # -- sampling ---------------------------------------------------------
    def sample_uv(
        self,
        size: int,
        params: Any,
        random_state: "int | None" = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Draw ``(u, v)`` pairs by conditional inversion of the h-function.

        ``u`` is uniform; given ``u`` and a uniform ``w``, ``v`` solves
        ``dC/du(u, v) = w`` (a CDF in ``v``, hence monotone) by bisection.
        Override for families with a direct sampler (e.g. Gaussian).
        """
        rng = as_generator(random_state)
        u = rng.uniform(_EPS, 1 - _EPS, size=size)
        w = rng.uniform(_EPS, 1 - _EPS, size=size)
        v = self._invert_du(u, w, params)
        return u, v

    def _invert_du(
        self,
        u: npt.NDArray,
        w: npt.NDArray,
        params: Any,
        iters: int = 60,
    ) -> npt.NDArray:
        lo = onp.full_like(onp.asarray(u, dtype=float), _EPS)
        hi = onp.full_like(lo, 1 - _EPS)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            over = onp.asarray(self.du(u, mid, *params)) > w
            hi = onp.where(over, mid, hi)
            lo = onp.where(over, lo, mid)
        return 0.5 * (lo + hi)

    # -- likelihood primitives -------------------------------------------
    def _eval(
        self,
        u: Any,
        v: Any,
        diff_u: bool,
        diff_v: bool,
        params: Any,
    ) -> Any:
        # The upper end of a right-censored slot is exactly 1, where every
        # copula has C(1, v) = v and dC/dv(1, v) = 1 (and symmetrically):
        # those are used as they are, rather than the family's formula at
        # 1 - 1e-10, which was off by up to 1e-10 and for a family whose
        # CDF is an integral (the Student-t) cost three quarters of a
        # doubly right-censored row.
        u_one = not diff_u and bool(onp.all(onp.asarray(u) == 1.0))
        v_one = not diff_v and bool(onp.all(onp.asarray(v) == 1.0))
        if u_one or v_one:
            shape = onp.broadcast_shapes(onp.shape(u), onp.shape(v))
            if diff_u or diff_v:
                return onp.ones(shape)
            other = onp.ones(shape) if u_one and v_one else (v if u_one else u)
            return onp.broadcast_to(onp.asarray(other, dtype=float), shape)
        u = np.clip(u, _EPS, 1 - _EPS)
        v = np.clip(v, _EPS, 1 - _EPS)
        if diff_u and diff_v:
            return self.pdf(u, v, *params)
        if diff_u:
            return self.du(u, v, *params)
        if diff_v:
            return self.dv(u, v, *params)
        return self.cdf(u, v, *params)

    @staticmethod
    def _op_terms(code: int, upoint: Any, ulo: Any, uhi: Any) -> list:
        """Per-dimension operator: list of ``(coef, u_value, differentiate)``.

        Applying the tensor product of the two dimensions' operators to ``C``
        yields the row's likelihood (densities for observed dims are added
        separately by the caller).
        """
        if code == 0:  # observed -> differentiate this slot
            return [(1.0, upoint, True)]
        if code == -1:  # left censored -> value at u
            return [(1.0, upoint, False)]
        if code == 1:  # right censored -> C(.,1) - C(.,u)
            return [(1.0, onp.ones_like(upoint), False), (-1.0, upoint, False)]
        # interval censored -> C(.,uhi) - C(.,ulo)
        return [(1.0, uhi, False), (-1.0, ulo, False)]

    def _pair_loglik(self, params: Any, d0: Any, d1: Any) -> Any:
        """Per-row log-likelihood for two prepared dimensions ``d0, d1``."""
        c0, c1 = d0["c"], d1["c"]
        N = len(c0)
        ll = onp.zeros(N)

        for a in onp.unique(c0):
            for b in onp.unique(c1):
                mask = (c0 == a) & (c1 == b)
                if not mask.any():
                    continue
                t0 = self._op_terms(
                    a, d0["u"][mask], d0["ulo"][mask], d0["uhi"][mask]
                )
                t1 = self._op_terms(
                    b, d1["u"][mask], d1["ulo"][mask], d1["uhi"][mask]
                )
                L = onp.zeros(int(mask.sum()))
                for coef0, u0, du0 in t0:
                    for coef1, u1, du1 in t1:
                        L = L + coef0 * coef1 * onp.asarray(
                            self._eval(u0, u1, du0, du1, params)
                        )
                logL = onp.log(onp.clip(L, _TINY, None))
                if a == 0:
                    logL = logL + d0["logf"][mask]
                if b == 0:
                    logL = logL + d1["logf"][mask]
                ll[mask] = logL

        if d0["has_trunc"] or d1["has_trunc"]:
            ll = ll - self._trunc_logmass(params, d0, d1)
        return ll

    def _boundary_cdf(self, u: Any, v: Any, params: Any) -> Any:
        """``C(u, v)`` with the boundary values every copula shares.

        ``C(0, v) = C(u, 0) = 0``, ``C(u, 1) = u`` and ``C(1, v) = v``.
        The truncation window's untruncated sides sit exactly on that
        boundary, where a family's formula can divide by zero (Clayton's
        ``u ** -theta`` at ``u = 0``); only interior points reach it.
        """
        u = onp.asarray(u, dtype=float)
        v = onp.asarray(v, dtype=float)
        u, v = onp.broadcast_arrays(u, v)
        zero = (u <= 0) | (v <= 0)
        u_one = u >= 1
        v_one = v >= 1
        interior = ~(zero | u_one | v_one)
        out = onp.where(v_one, u, onp.where(u_one, v, 0.0))
        out = onp.where(zero, 0.0, out)
        if interior.any():
            out = out.copy()
            out[interior] = onp.asarray(
                self.cdf(u[interior], v[interior], *params)
            )
        return out

    def _trunc_logmass(self, params: Any, d0: Any, d1: Any) -> Any:
        """Log copula mass over the per-row truncation rectangle."""
        ul0, ur0 = d0["ul"], d0["ur"]
        ul1, ur1 = d1["ul"], d1["ur"]
        mass = (
            self._boundary_cdf(ur0, ur1, params)
            - self._boundary_cdf(ul0, ur1, params)
            - self._boundary_cdf(ur0, ul1, params)
            + self._boundary_cdf(ul0, ul1, params)
        )
        return onp.log(onp.clip(mass, _TINY, None))

    # -- fitting ----------------------------------------------------------
    def _prepare_dim(
        self,
        margin: Any,
        x: Any,
        c: Any,
        xl: Any,
        xr: Any,
        tl: Any,
        tr: Any,
    ) -> dict:
        """Transform one dimension's data into copula (u-space) arrays.

        A non-parametric margin (e.g. a fitted ``KaplanMeier``) gives the
        semi-parametric pseudo-likelihood of Genest, Ghoudi and Rivest
        (1995): its step CDF is rescaled by ``N/(N+1)`` so the largest
        values stay inside the unit square, and it contributes no density
        term -- a step function has none, and the term would not depend
        on the copula parameter anyway.
        """
        semiparametric = _is_nonparametric(margin)
        scale = 1.0
        if semiparametric:
            n_units = float(onp.max(getattr(margin, "r", [len(x)])))
            scale = n_units / (n_units + 1.0)
        u = onp.clip(
            scale * onp.asarray(margin.ff(x), dtype=float), _EPS, 1 - _EPS
        )
        # An interval may start at the edge of a margin's support (0 for a
        # LogNormal, whose ff takes log(0) = -inf on the way to the correct
        # value 0); that is not an error, so it is not reported as one.
        with onp.errstate(divide="ignore"):
            ulo = scale * onp.asarray(margin.ff(xl), dtype=float)
            uhi = scale * onp.asarray(margin.ff(xr), dtype=float)
        ulo = onp.clip(ulo, _EPS, 1 - _EPS)
        uhi = onp.clip(uhi, _EPS, 1 - _EPS)
        if semiparametric:
            logf = onp.zeros(onp.shape(u))
        else:
            with onp.errstate(divide="ignore"):
                logf = onp.log(
                    onp.clip(onp.asarray(margin.df(x)), _TINY, None)
                )
        has_trunc = bool(onp.isfinite(tl).any() or onp.isfinite(tr).any())
        ul = scale * _ff_where_finite(margin, tl, 0.0)
        ur = onp.where(
            onp.isfinite(onp.asarray(tr, dtype=float)),
            scale * _ff_where_finite(margin, tr, 1.0),
            1.0,
        )
        return {
            "c": onp.asarray(c, dtype=int),
            "u": u,
            "ulo": ulo,
            "uhi": uhi,
            "logf": logf,
            "ul": onp.clip(ul, 0.0, 1.0),
            "ur": onp.clip(ur, 0.0, 1.0),
            "has_trunc": has_trunc,
        }

    def neg_ll(self, params: Any, dims: list, weights: npt.NDArray) -> float:
        """The copula-stage negative log-likelihood (used by the fit)."""
        ll = self._pair_loglik(params, dims[0], dims[1])
        return -float(onp.sum(weights * ll))

    def fit_from_df(
        self,
        df: Any,
        x_cols: "list[str]",
        c_cols: "list[str] | None" = None,
        n_col: "str | None" = None,
        xl_cols: "list[str] | None" = None,
        xr_cols: "list[str] | None" = None,
        tl_cols: "list[str] | None" = None,
        tr_cols: "list[str] | None" = None,
        **fit_options: Any,
    ) -> Any:
        """Fit the copula and its margins to the columns of a
        :class:`pandas.DataFrame`.

        Each argument names, per dimension, the columns :meth:`fit` takes
        as arrays: ``x_cols=["a", "b"]`` reads the two series from columns
        ``a`` and ``b``. The names are those of the univariate
        ``fit_from_df`` (``Weibull.fit_from_df(df, x_col=..., c_col=...)``)
        with ``_cols`` for a list of columns, one per dimension
        (principle 21); every other :meth:`fit` option (``margins``,
        ``how``, ``init``) is passed to it unchanged.

        Parameters
        ----------
        df : pandas.DataFrame
            The data, one row per unit.
        x_cols : list of str
            The column of each dimension's values.
        c_cols : list of str, optional
            The column of each dimension's censoring flags. Defaults to
            every value observed.
        n_col : str, optional
            The column of row counts.
        xl_cols, xr_cols : list of str, optional
            The columns of each dimension's interval ends, where the
            censoring flag is 2.
        tl_cols, tr_cols : list of str, optional
            The columns of each dimension's left / right truncation.
        **fit_options
            Every other option of :meth:`fit`.

        Returns
        -------
        CopulaModel
            The model :meth:`fit` returns for the same arrays.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Clayton
        >>> margins = [
        ...     Weibull.from_params([10, 2]),
        ...     Weibull.from_params([20, 3]),
        ... ]
        >>> X = Clayton.from_params([2.0], margins).random(300, random_state=0)
        >>> df = pd.DataFrame(X, columns=["pump", "motor"])
        >>> model = Clayton.fit_from_df(
        ...     df, x_cols=["pump", "motor"], margins=[Weibull, Weibull]
        ... )
        >>> model.params.round(3)
        array([2.293])
        """
        df = require_frame(df)
        arrays: dict[str, Any] = {
            "x": frame_columns(df, x_cols, "x_cols", time=True).astype(float)
        }
        for key, cols in (("c", c_cols), ("xl", xl_cols), ("xr", xr_cols)):
            if cols is not None:
                arrays[key] = frame_columns(
                    df, cols, f"{key}_cols", time=key != "c"
                )
        if n_col is not None:
            arrays["n"] = frame_column(df, n_col, "n_col")
        if tl_cols is not None or tr_cols is not None:
            shape = arrays["x"].shape
            lower = (
                onp.full(shape, -onp.inf)
                if tl_cols is None
                else frame_columns(df, tl_cols, "tl_cols", time=True).astype(
                    float
                )
            )
            upper = (
                onp.full(shape, onp.inf)
                if tr_cols is None
                else frame_columns(df, tr_cols, "tr_cols", time=True).astype(
                    float
                )
            )
            arrays["t"] = onp.stack([lower, upper], axis=-1)
        names = {k: f"{k}_cols" for k in ("x", "c", "xl", "xr")}
        names["n"] = "n_col"
        names["t"] = "tl_cols` / `tr_cols"
        return call_fit(self, arrays, names, fit_options)

    def fit(
        self,
        x: npt.ArrayLike,
        c: "npt.ArrayLike | None" = None,
        n: "npt.ArrayLike | None" = None,
        t: "npt.ArrayLike | None" = None,
        margins: Any = None,
        how: str = "IFM",
        xl: "npt.ArrayLike | None" = None,
        xr: "npt.ArrayLike | None" = None,
        init: "npt.ArrayLike | None" = None,
    ) -> Any:
        """Fit the copula and its margins to multivariate survival data.

        Parameters
        ----------
        x, c, n, t, xl, xr
            Multivariate survival data; see
            :class:`MultivariateSurpyvalData` for the accepted shapes.
        margins : sequence of length D
            Either surpyval distribution classes (e.g. ``surpyval.Weibull``)
            to be fitted, or already-fitted models exposing ``ff``/``df``.
            Under ``"IFM"`` a fitted model is used as it is. Under ``"MLE"``
            a fitted parametric model supplies the starting values and its
            configuration: it is re-estimated jointly with the copula with
            the same offset, limited-failure or zero-inflated option, and
            any parameters it was fitted with ``fixed`` stay at their
            values. A non-parametric margin (a class such as
            ``surpyval.KaplanMeier``, or a fitted non-parametric model) can
            only be used with ``"IFM"``: it gives the semi-parametric
            pseudo-likelihood estimator, and the likelihood and criteria
            then compare copula families with the same margins only.
        how : {"IFM", "MLE"}
            ``"IFM"`` (default) fits each margin independently then the
            single copula parameter (robust two-stage estimation).
            ``"MLE"`` jointly optimises copula parameter + margin parameters.
        init : array like, optional
            Starting value of the copula parameter(s) for the search, one per
            entry of ``parameter_names``, each strictly inside the family's
            ``bounds``. By default each family starts from its own guess
            (the built-in families match the empirical Kendall's tau).

        Returns
        -------
        CopulaModel
            The fitted model: the copula parameter(s) ``params`` and the
            fitted ``margins``, with the joint ``sf``/``cdf``/``pdf``,
            sampling, dependence measures and the likelihood-based
            ``log_likelihood``/``neg_ll``/``aic``/``bic``.

        Warns
        -----
        UserWarning
            "No finite maximum" when the rows observed in both dimensions
            are perfectly dependent (Kendall's tau of +-1) and the family
            reaches that dependence only as its parameter runs to a limit:
            the returned parameter is then meaningless.

        Examples
        --------
        Simulate from a Clayton copula with Weibull margins, then recover
        it:

        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Clayton
        >>> margins = [
        ...     Weibull.from_params([10, 2]),
        ...     Weibull.from_params([20, 3]),
        ... ]
        >>> X = Clayton.from_params([2.0], margins).random(300, random_state=0)
        >>> model = Clayton.fit(X, margins=[Weibull, Weibull])
        >>> model.params.round(3)
        array([2.293])
        >>> round(float(model.kendall_tau()), 3)
        0.534
        """
        from surpyval.multivariate.parametric.copula.copula_model import (
            CopulaModel,
        )
        from surpyval.multivariate.parametric.data import (
            MultivariateSurpyvalData,
        )

        data = MultivariateSurpyvalData(x, c=c, n=n, t=t, xl=xl, xr=xr)
        if data.D != 2:
            raise NotImplementedError("only bivariate copulas are supported")
        if margins is None:
            raise ValueError("margins must be provided (one per dimension)")
        if len(margins) != data.D:
            raise ValueError("need one margin per dimension")

        if how not in ("IFM", "MLE"):
            raise ValueError("how must be 'IFM' or 'MLE'")
        if init is not None:
            init = self._check_init(init)

        margin_models = self._fit_margins(margins, data)
        if how == "IFM":
            theta = self._fit_theta(margin_models, data, init)
            # A margin passed already fitted is used as it is, so only the
            # margins fitted here count as estimated parameters.
            # A non-parametric margin has no parameter vector: the
            # criteria then count the copula's parameters only (and are a
            # pseudo-likelihood's, for comparing copula families with the
            # same margins).
            k = len(self.parameter_names) + sum(
                len(getattr(m, "params", ()))
                for m, given in zip(margin_models, margins)
                if hasattr(given, "fit") and not _is_nonparametric(m)
            )
        else:
            theta, margin_models = self._fit_joint(
                margins, margin_models, data, init
            )
            # The joint search re-estimates every free margin parameter,
            # including an offset, cure or zero-inflation proportion.
            k = len(self.parameter_names) + sum(
                _JointMargin.n_free_of(m) for m in margin_models
            )

        # One warning per fit: perfect dependence explains any runaway.
        if not self._warn_if_perfectly_dependent(data, theta):
            self._warn_if_no_maximum(margin_models, data, theta)
        return CopulaModel(self, theta, margin_models, data=data, how=how, k=k)

    def _warn_if_no_maximum(
        self, margin_models: list, data: Any, theta: npt.NDArray
    ) -> None:
        """A family whose likelihood can lack a finite maximum on data
        that are not perfectly dependent checks for it here (the Student-t
        copula's degrees of freedom); by default there is nothing to
        check."""

    def _warn_if_perfectly_dependent(
        self, data: Any, theta: npt.NDArray
    ) -> bool:
        """Warn when the data sit at a Frechet bound the family reaches
        only in the limit of its parameter (#392).

        The criterion is on the data, not on the estimate: the rows
        observed in both dimensions are perfectly concordant (Kendall's
        tau of 1: one coordinate is an increasing function of the other,
        the comonotone copula) or perfectly discordant (-1). No member of
        a family such as the Clayton, with a finite parameter, has that
        dependence. With margins that map one coordinate exactly onto the
        other (the non-parametric margins, or parametric ones on data
        such as ``x2 = x1 / 2``) the likelihood keeps increasing towards
        the limit and the search stops wherever it gives up: Clayton
        theta 3.2e6, Frank 1.2e7, Gumbel 105.5 with a log-likelihood of
        inf, Gaussian rho at its cap of 0.9999. Otherwise the peak is set
        only by how far the fitted margins are from that map (Clayton 60
        on ``x2 = log(x1)``). Either way the estimate says nothing about
        the dependence. Data with even one discordant pair (or a tie in
        one coordinate only) are never flagged, so a fit to data drawn
        from any member of the family is silent unless the sample itself
        is perfectly dependent. Returns whether it warned.
        """
        sign, rows = _perfect_dependence(data)
        limit = self.dependence_limits.get(sign)
        if limit is None:
            return False
        bound, kind, relation = (
            ("1", "comonotone", "increasing")
            if sign > 0
            else ("-1", "countermonotone", "decreasing")
        )
        params = ", ".join(
            f"{name} = {value:.4g}"
            for name, value in zip(self.parameter_names, theta)
        )
        warn_no_maximum(
            f"the {rows} rows observed in both dimensions are perfectly "
            f"{'concordant' if sign > 0 else 'discordant'} (Kendall's tau "
            f"= {bound}), the {kind} copula (a Frechet bound), which the "
            f"{self.name} family reaches only as {limit}; the likelihood "
            "keeps increasing towards it, or peaks only where the fitted "
            "margins stop mapping one coordinate exactly onto the other",
            f"The reported {params} and the dependence measures derived "
            "from it are meaningless",
            f"the data are perfectly dependent (one variable is an "
            f"{relation} function of the other): model that relationship "
            "directly rather than with a copula",
        )
        return True

    def from_params(self, params: Any, margins: Any) -> Any:
        """
        Build a
        :class:`~surpyval.multivariate.parametric.copula.copula_model.CopulaModel`
        from a known parameter and margins, without fitting.

        Parameters
        ----------
        params : array like
            The copula parameter(s), e.g. ``[theta]`` (empty for the
            independence copula): one per entry of ``parameter_names``, each
            strictly inside the family's ``bounds`` (Gumbel's ``theta = 1``,
            the independence copula, is allowed too). Anything else raises
            ``ValueError``.
        margins : sequence of length 2
            Fitted (or ``from_params``) univariate models, one per
            dimension, each exposing ``ff`` and ``df``.

        Returns
        -------
        CopulaModel
            The model, for evaluation and simulation.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Clayton
        >>> margins = [
        ...     Weibull.from_params([10, 2]),
        ...     Weibull.from_params([20, 3]),
        ... ]
        >>> model = Clayton.from_params([2.0], margins)
        >>> round(float(model.kendall_tau()), 3)
        0.5
        """
        from surpyval.multivariate.parametric.copula.copula_model import (
            CopulaModel,
        )

        params = self._check_params(params)
        margins = list(margins)
        if len(margins) != 2:
            raise ValueError(
                f"A bivariate copula needs 2 margins, one per dimension; "
                f"got {len(margins)}."
            )
        for d, m in enumerate(margins):
            if not all(callable(getattr(m, a, None)) for a in ("ff", "df")):
                raise ValueError(
                    f"Margin {d} is not a fitted univariate model (it needs "
                    "`ff` and `df`); build one with e.g. "
                    "`Weibull.from_params`."
                )
        return CopulaModel(self, params, margins, data=None, how="given")

    #: Names of parameters whose finite bounds are themselves valid values
    #: (Gumbel's ``theta = 1`` is the independence copula). The fitter
    #: never reaches a bound, but ``from_params`` may be given one.
    closed_bounds: tuple = ()

    def _check_params(self, params: npt.ArrayLike) -> npt.NDArray:
        """Validate copula parameters given directly: one finite value per
        entry of ``parameter_names``, inside ``bounds``. An unchecked value
        silently gave a non-copula (a Clayton ``theta = -2`` has a negative
        density) or was clipped (a Gaussian ``rho = 1.5``)."""
        params = onp.atleast_1d(onp.asarray(params, dtype=float))
        if params.shape != (len(self.parameter_names),):
            raise ValueError(
                f"The {self.name} copula takes {len(self.parameter_names)} "
                f"parameter(s) {self.parameter_names}, got {params.tolist()}."
            )
        for name, value, (low, high) in zip(
            self.parameter_names, params, self.bounds
        ):
            closed = name in self.closed_bounds
            above = low is None or value > low or (closed and value == low)
            below = high is None or value < high or (closed and value == high)
            if not (onp.isfinite(value) and above and below):
                raise ValueError(
                    f"{self.name} copula parameter {name} = {value} is "
                    f"outside its bounds ({low}, {high})."
                )
        return params

    def _fit_margins(self, margins: Any, data: Any) -> list:
        models = []
        for d, margin in enumerate(margins):
            if not hasattr(margin, "fit"):
                models.append(margin)  # already a fitted model
                continue
            # Reuse the univariate fitter, honouring each margin's own
            # censoring. Interval entries (c == 2) are passed via xl/xr,
            # exactly as the univariate API expects.
            # Each margin is fitted with the rows' counts and its own
            # dimension's truncation window (a margin fitted without them
            # is biased, and the copula stage inherits the bias).
            xd, cd, xld, xrd, tld, trd = data.dimension(d)
            kwargs: dict = {"c": cd, "n": data.n}
            if onp.isfinite(tld).any():
                kwargs["tl"] = tld
            if onp.isfinite(trd).any():
                kwargs["tr"] = trd
            if (cd == 2).any():
                # surpyval mixes interval and point data in one 2-column x:
                # point rows have equal columns, interval rows carry [xl, xr].
                x2 = onp.column_stack(
                    [onp.where(cd == 2, xld, xd), onp.where(cd == 2, xrd, xd)]
                )
                models.append(margin.fit(x=x2, **kwargs))
            else:
                models.append(margin.fit(x=xd, **kwargs))
        return models

    def _bounds_transforms(self) -> tuple:
        from surpyval.univariate.parametric.fitters import bounds_convert

        param_map = {n: i for i, n in enumerate(self.parameter_names)}
        to_unbounded, to_bounded, const, _, _ = bounds_convert(
            None, self.bounds, None, param_map
        )
        return to_unbounded, to_bounded

    def _fit_theta(
        self,
        margin_models: list,
        data: Any,
        init: "npt.NDArray | None" = None,
    ) -> npt.NDArray:
        dims = [
            self._prepare_dim(margin_models[d], *data.dimension(d))
            for d in range(data.D)
        ]
        to_unbounded, to_bounded = self._bounds_transforms()

        def obj(phi: npt.NDArray) -> float:
            params = to_bounded(phi)
            return self.neg_ll(params, dims, data.n)

        if init is None:
            init = self._init_theta(dims)
        # A start on (or outside) a bound maps to +-inf or NaN in the
        # unbounded space, from which Nelder-Mead never moves: the fit would
        # silently return the starting value. Refuse it instead (the
        # transform's own warning would only restate the error).
        with onp.errstate(divide="ignore", invalid="ignore"):
            start = onp.asarray(to_unbounded(init), dtype=float)
        if not onp.all(onp.isfinite(start)):
            raise ValueError(
                f"The starting value {onp.asarray(init).tolist()} of the "
                f"{self.name} copula is not strictly inside its bounds "
                f"{self.bounds}; pass `init` to fit."
            )
        # Towards a limit of the family (see
        # ``_warn_if_perfectly_dependent``) the likelihood overflows; the
        # search reads inf and nan correctly, so numpy's warnings about
        # them are noise (they were 230 raw warnings from a Gumbel fit).
        with onp.errstate(all="ignore"):
            res = minimize(obj, start, method="Nelder-Mead")
        return onp.asarray(to_bounded(res.x), dtype=float)

    def _fit_joint(
        self,
        margins: Any,
        margin_models: list,
        data: Any,
        init: "npt.NDArray | None" = None,
    ) -> tuple:
        # Start from the IFM solution, then refine copula + margin params
        # jointly. Each margin is rebuilt from its parameter vector at every
        # step with the configuration it was fitted with (offset,
        # limited-failure or zero-inflated, and any fixed parameters kept
        # at their values); rebuilding a plain distribution of its family
        # silently dropped all of that.
        joint = [
            _JointMargin(margin_models[d], data, d) for d in range(data.D)
        ]
        theta0 = self._fit_theta(margin_models, data, init)
        n_cop = len(self.parameter_names)
        splits = onp.cumsum([j.n_free for j in joint])[:-1]
        to_unbounded, to_bounded = self._bounds_transforms()

        def unpack(phi: npt.NDArray) -> tuple:
            theta = to_bounded(phi[:n_cop])
            parts = onp.split(phi[n_cop:], splits)
            return theta, [j.build(part) for j, part in zip(joint, parts)]

        def obj(phi: npt.NDArray) -> float:
            theta, models = unpack(phi)
            if any(m is None for m in models):
                return onp.inf  # a margin parameter left its bounds
            dims = [
                self._prepare_dim(models[d], *data.dimension(d))
                for d in range(data.D)
            ]
            return self.neg_ll(theta, dims, data.n)

        start = onp.concatenate(
            [to_unbounded(theta0)] + [j.start for j in joint]
        )
        with onp.errstate(all="ignore"):  # as in ``_fit_theta``
            res = minimize(
                obj,
                start,
                method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-6},
            )
        theta, models = unpack(res.x)
        return onp.asarray(theta, dtype=float), models

    def _init_theta(self, dims: list) -> npt.NDArray:
        """Initial parameter guess, strictly inside ``bounds``.

        Per parameter: 1 when that is strictly inside its bounds (the
        historical default), otherwise the midpoint of a finite interval or
        one unit inside a single finite bound. A fixed 1 sat on the bound of
        a family such as ``(-1, 1)``, where the bounds transform gives an
        infinite start and the search never moved. The built-in families
        override this with a data-driven guess; a user can pass ``init`` to
        :meth:`fit`.
        """
        starts = []
        for low, high in self.bounds:
            if (low is None or low < 1.0) and (high is None or 1.0 < high):
                starts.append(1.0)
            elif low is not None and high is not None:
                starts.append(0.5 * (low + high))
            elif low is not None:
                starts.append(low + 1.0)
            else:
                starts.append(high - 1.0)
        return onp.asarray(starts, dtype=float)

    def _check_init(self, init: npt.ArrayLike) -> npt.NDArray:
        """Validate a user's ``init``: one value per parameter, each
        strictly inside its bounds (a start on a bound cannot move)."""
        init = onp.atleast_1d(onp.asarray(init, dtype=float))
        if init.shape != (len(self.parameter_names),):
            raise ValueError(
                f"init must have one value per copula parameter "
                f"{self.parameter_names}, got {init.tolist()}"
            )
        for name, value, (low, high) in zip(
            self.parameter_names, init, self.bounds
        ):
            inside = (
                bool(onp.isfinite(value))
                and (low is None or value > low)
                and (high is None or value < high)
            )
            if not inside:
                raise ValueError(
                    f"init for {name} must be strictly inside its bounds "
                    f"({low}, {high}), got {value}"
                )
        return init

    @staticmethod
    def _emp_tau(dims: list) -> float:
        """Empirical Kendall's tau over rows where both dims are observed."""
        from scipy.stats import kendalltau

        both = (dims[0]["c"] == 0) & (dims[1]["c"] == 0)
        if both.sum() < 3:
            return 0.0
        tau = kendalltau(dims[0]["u"][both], dims[1]["u"][both]).statistic
        return 0.0 if not onp.isfinite(tau) else float(tau)


class _JointMargin:
    """One margin's parameters in the joint (``how="MLE"``) search.

    The margin is re-estimated with the configuration it was fitted with:
    its full parameter vector is laid out as the model's ``param_map`` --
    the offset ``gamma`` (if any), the distribution's parameters, the
    limited-failure ``p`` and the zero-inflation ``f0`` -- and the
    parameters fixed at fit time stay at their values. ``gamma`` is kept
    below the dimension's smallest time, and ``p`` and ``f0`` inside
    (0, 1), through the same unbounded transforms the univariate fitters
    use. The distribution's own parameters are searched as they are (a
    step out of their bounds scores ``inf``), as they always were.
    """

    def __init__(self, model: Any, data: Any, d: int) -> None:
        from surpyval.univariate.parametric.fitters import bounds_convert
        from surpyval.univariate.parametric.parametric import Parametric

        if not isinstance(model, Parametric):
            raise ValueError(
                f"how='MLE' re-estimates every margin, so margin {d} must "
                "be a parametric distribution (e.g. surpyval.Weibull) or a "
                f"model fitted with one; got {type(model).__name__}. Use "
                "how='IFM' to keep this margin as it is."
            )
        self.dist = model.dist
        self.offset = bool(model.offset)
        self.lfp = bool(model.lfp)
        self.zi = bool(model.zi)
        self.k = len(model.params)
        full: list = []
        bounds: list = []
        if self.offset:
            upper = self._gamma_upper(data, d)
            gamma = float(model.gamma)
            if gamma >= upper:
                # Fitted to other data: start just inside this data.
                gamma = upper - 1e-6 * max(abs(upper), 1.0)
            full.append(gamma)
            bounds.append((None, upper))
        full.extend(onp.asarray(model.params, dtype=float).tolist())
        bounds.extend([(None, None)] * self.k)
        if self.lfp:
            full.append(float(model.p))
            bounds.append((0, 1))
        if self.zi:
            full.append(float(model.f0))
            bounds.append((0, 1))
        self.fixed = sorted(int(i) for i in model._user_fixed_idx())
        self.free = [i for i in range(len(full)) if i not in self.fixed]
        self.n_free = len(self.free)
        to_unb, self._to_bounded = bounds_convert(
            None, bounds, None, {str(i): i for i in range(len(full))}
        )[:2]
        self._full = onp.asarray(full, dtype=float)
        self._unbounded = onp.asarray(to_unb(self._full), dtype=float)
        self.start = self._unbounded[self.free]

    @staticmethod
    def _gamma_upper(data: Any, d: int) -> float:
        """The offset must stay below every time of the dimension (the
        univariate fitter's bound), ignoring exact zeros for a
        zero-inflated margin."""
        x, c, xl, _, _, _ = data.dimension(d)
        times = onp.where(c == 2, xl, x)
        times = times[onp.isfinite(times)]
        positive = times[times != 0]
        return float(onp.min(positive if positive.size else times))

    @staticmethod
    def n_free_of(model: Any) -> int:
        """Parameters of a margin the joint search estimated: all of its
        parameters (with ``gamma``, ``p``, ``f0``) but the fixed ones."""
        fixed: set = getattr(model, "_user_fixed_idx", lambda: set())()
        return int(model.k) - len(fixed)

    def build(self, phi: npt.NDArray) -> Any:
        """The margin for the free parameters ``phi`` (unbounded space), or
        ``None`` if a distribution parameter is outside its bounds."""
        unbounded = self._unbounded.copy()
        unbounded[self.free] = phi
        full = onp.asarray(self._to_bounded(unbounded), dtype=float)
        full[self.fixed] = self._full[self.fixed]  # exactly as fixed
        i = 0
        gamma = p = f0 = None
        if self.offset:
            gamma, i = full[0], 1
        end = i + self.k
        params = full[i:end]
        i = end
        if self.lfp:
            p, i = full[i], i + 1
        if self.zi:
            f0 = full[i]
        try:
            model = self.dist.from_params(params, gamma=gamma, p=p, f0=f0)
        except ValueError:
            return None
        if self.fixed:
            # Kept so the margin reports its fixed parameters (and its own
            # parameter count) as the fitted margin did.
            model.fitting_info = {"fixed_idx": list(self.fixed)}
        return model


@functools.lru_cache(maxsize=1)
def _quadrature_grid(nodes: int = 400) -> tuple:
    """Tensor Gauss-Legendre rule on the unit square: flat ``u``, ``v`` and
    weights ``w`` (summing to 1), for the dependence-measure integrals."""
    x, w = onp.polynomial.legendre.leggauss(nodes)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    u, v = onp.meshgrid(x, x, indexing="ij")
    return u.ravel(), v.ravel(), onp.outer(w, w).ravel()


def _broadcast_pair(u: Any, v: Any) -> tuple:
    """``u`` and ``v`` broadcast to their common shape.

    Adding zeros keeps an autograd-traced argument traced (``pdf``
    differentiates ``du`` through its ``v``), which
    ``numpy.broadcast_arrays`` would not.
    """
    zeros = onp.zeros(onp.broadcast_shapes(onp.shape(u), onp.shape(v)))
    return u + zeros, v + zeros


def _perfect_dependence(data: Any) -> tuple[int, int]:
    """Whether the rows observed in every dimension are perfectly
    dependent: ``(1, rows)`` if perfectly concordant, ``(-1, rows)`` if
    perfectly discordant, else ``(0, rows)``, with ``rows`` their count.

    This is Kendall's tau-b of those rows at +-1, decided exactly rather
    than in floating point: among the distinct pairs, no value of either
    coordinate repeats (a tie in one coordinate only lowers tau-b; a
    repeated row is tied in both and does not count) and the second
    coordinate, in the order of the first, only rises (or only falls).
    It takes two distinct pairs.
    """
    both = onp.all(data.c == 0, axis=1) & (data.n > 0)
    rows = int(onp.sum(data.n[both]))
    # Sorted by the first coordinate (then the second)
    pairs = onp.unique(data.x[both], axis=0)
    m = pairs.shape[0]
    if m < 2 or any(onp.unique(pairs[:, d]).size < m for d in (0, 1)):
        return 0, rows
    step = onp.diff(pairs[:, 1])
    if onp.all(step > 0):
        return 1, rows
    if onp.all(step < 0):
        return -1, rows
    return 0, rows


def _is_nonparametric(margin: Any) -> bool:
    """True for a fitted non-parametric margin (a step CDF, no params)."""
    from surpyval.distribution import NonParametricDistribution

    return isinstance(margin, NonParametricDistribution)


def _ff_where_finite(margin: Any, t: Any, fill: float) -> npt.NDArray:
    """``margin.ff(t)`` at finite ``t``, ``fill`` at the infinite defaults.

    Evaluating a margin at +-inf warns (``log(-inf)`` for a LogNormal), so
    the infinite entries -- "no truncation on this side" -- are never
    passed to it.
    """
    t = onp.asarray(t, dtype=float)
    finite = onp.isfinite(t)
    out = onp.full(t.shape, fill, dtype=float)
    if finite.any():
        out[finite] = onp.asarray(margin.ff(t[finite]), dtype=float)
    return out
