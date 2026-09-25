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

from typing import Any

import numpy as onp
import numpy.typing as npt
from autograd import elementwise_grad
from scipy.optimize import minimize

from surpyval import np

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
    param_names: tuple = ("theta",)

    # -- the four copula primitives ---------------------------------------
    def cdf(self, u: Any, v: Any, *params: Any) -> Any:
        """The copula :math:`C(u, v)`; defined by each family."""
        raise NotImplementedError

    def du(self, u: Any, v: Any, *params: Any) -> Any:
        """``dC/du`` -- the h-function :math:`P(V \\le v \\mid U = u)`.

        The default differentiates :meth:`cdf` with autograd; override it
        when a closed form is known. ``u`` and ``v`` must have the same
        shape: the automatic derivative sums over any broadcast axis, so a
        scalar ``u`` with an array ``v`` returns one summed number.
        """
        return elementwise_grad(lambda a: self.cdf(a, v, *params))(
            onp.asarray(u, dtype=float)
        )

    def dv(self, u: Any, v: Any, *params: Any) -> Any:
        """``dC/dv``, the h-function :math:`P(U \\le u \\mid V = v)`.
        As for :meth:`du`, ``u`` and ``v`` must have the same shape."""
        return elementwise_grad(lambda b: self.cdf(u, b, *params))(
            onp.asarray(v, dtype=float)
        )

    def pdf(self, u: Any, v: Any, *params: Any) -> Any:
        """``d2C/du dv`` -- the copula density. The default differentiates
        :meth:`du` with autograd; ``u`` and ``v`` must have the same shape."""
        return elementwise_grad(lambda b: self.du(u, b, *params))(
            onp.asarray(v, dtype=float)
        )

    # -- dependence measures (closed-form overrides preferred) ------------
    def kendall_tau(self, *params: float) -> float:
        """Kendall's tau. Default: empirical estimate from a large sample."""
        from scipy.stats import kendalltau

        u, v = self.sample_uv(50_000, params, random_state=0)
        return float(kendalltau(u, v).statistic)

    def spearman_rho(self, *params: float) -> float:
        """Spearman's rho. Default: empirical estimate from a large sample."""
        from scipy.stats import spearmanr

        u, v = self.sample_uv(50_000, params, random_state=0)
        return float(spearmanr(u, v).statistic)

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
        rng = onp.random.default_rng(random_state)
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
        """Transform one dimension's data into copula (u-space) arrays."""
        u = onp.clip(onp.asarray(margin.ff(x), dtype=float), _EPS, 1 - _EPS)
        # An interval may start at the edge of a margin's support (0 for a
        # LogNormal, whose ff takes log(0) = -inf on the way to the correct
        # value 0); that is not an error, so it is not reported as one.
        with onp.errstate(divide="ignore"):
            ulo = onp.asarray(margin.ff(xl), dtype=float)
            uhi = onp.asarray(margin.ff(xr), dtype=float)
        ulo = onp.clip(ulo, _EPS, 1 - _EPS)
        uhi = onp.clip(uhi, _EPS, 1 - _EPS)
        with onp.errstate(divide="ignore"):
            logf = onp.log(onp.clip(onp.asarray(margin.df(x)), _TINY, None))
        has_trunc = bool(onp.isfinite(tl).any() or onp.isfinite(tr).any())
        ul = _ff_where_finite(margin, tl, 0.0)
        ur = _ff_where_finite(margin, tr, 1.0)
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
            it only supplies starting values: it is re-estimated as a plain
            distribution of its family (``model.dist``), so an offset,
            limited-failure or zero-inflated option it was fitted with is not
            kept.
        how : {"IFM", "MLE"}
            ``"IFM"`` (default) fits each margin independently then the
            single copula parameter (robust two-stage estimation).
            ``"MLE"`` jointly optimises copula parameter + margin parameters.
        init : array like, optional
            Starting value of the copula parameter(s) for the search, one per
            entry of ``param_names``, each strictly inside the family's
            ``bounds``. By default each family starts from its own guess
            (the built-in families match the empirical Kendall's tau).

        Returns
        -------
        CopulaModel
            The fitted model: the copula parameter(s) ``params`` and the
            fitted ``margins``, with the joint ``sf``/``cdf``/``pdf``,
            sampling, dependence measures and the likelihood-based
            ``log_likelihood``/``neg_ll``/``aic``/``bic``.

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
            fitted = [hasattr(m, "fit") for m in margins]
        else:
            theta, margin_models = self._fit_joint(
                margins, margin_models, data, init
            )
            # The joint search re-estimates every margin parameter.
            fitted = [True] * len(margin_models)
        k = len(self.param_names) + sum(
            len(m.params) for m, f in zip(margin_models, fitted) if f
        )

        return CopulaModel(self, theta, margin_models, data=data, how=how, k=k)

    def from_params(self, params: Any, margins: Any) -> Any:
        """
        Build a
        :class:`~surpyval.multivariate.parametric.copula.copula_model.CopulaModel`
        from a known parameter and margins, without fitting.

        Parameters
        ----------
        params : array like
            The copula parameter(s), e.g. ``[theta]`` (empty for the
            independence copula), inside the family's ``bounds``; the value
            is not checked.
        margins : sequence of length 2
            Fitted (or ``from_params``) univariate models, one per
            dimension.

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

        params = onp.atleast_1d(onp.asarray(params, dtype=float))
        return CopulaModel(self, params, list(margins), data=None, how="given")

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

        param_map = {n: i for i, n in enumerate(self.param_names)}
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
        # jointly. Margins are re-evaluated from their parameter vectors at
        # each step via ``from_params``.
        theta0 = self._fit_theta(margin_models, data, init)
        dist_classes = [m.dist for m in margin_models]
        splits = onp.cumsum([len(m.params) for m in margin_models])[:-1]
        to_unbounded, to_bounded = self._bounds_transforms()

        def unpack(phi: npt.NDArray) -> tuple:
            theta = to_bounded(phi[: len(self.param_names)])
            rest = phi[len(self.param_names) :]
            parts = onp.split(rest, splits)
            models = [
                dist_classes[d].from_params(parts[d]) for d in range(data.D)
            ]
            return theta, models

        def obj(phi: npt.NDArray) -> float:
            theta, models = unpack(phi)
            dims = [
                self._prepare_dim(models[d], *data.dimension(d))
                for d in range(data.D)
            ]
            return self.neg_ll(theta, dims, data.n)

        init = onp.concatenate(
            [to_unbounded(theta0)] + [m.params for m in margin_models]
        )
        res = minimize(
            obj,
            init,
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
        if init.shape != (len(self.param_names),):
            raise ValueError(
                f"init must have one value per copula parameter "
                f"{self.param_names}, got {init.tolist()}"
            )
        for name, value, (low, high) in zip(
            self.param_names, init, self.bounds
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
