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

import numpy as onp
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
    def cdf(self, u, v, *params):
        raise NotImplementedError

    def du(self, u, v, *params):
        """``dC/du`` -- the h-function. autograd default; override if known."""
        return elementwise_grad(lambda a: self.cdf(a, v, *params))(
            onp.asarray(u, dtype=float)
        )

    def dv(self, u, v, *params):
        """``dC/dv``. autograd default; override if known."""
        return elementwise_grad(lambda b: self.cdf(u, b, *params))(
            onp.asarray(v, dtype=float)
        )

    def pdf(self, u, v, *params):
        """``d2C/du dv`` -- the copula density. autograd default."""
        return elementwise_grad(lambda b: self.du(u, b, *params))(
            onp.asarray(v, dtype=float)
        )

    # -- dependence measures (closed-form overrides preferred) ------------
    def kendall_tau(self, *params):
        """Kendall's tau. Default: empirical estimate from a large sample."""
        from scipy.stats import kendalltau

        u, v = self.sample_uv(50_000, params, random_state=0)
        return float(kendalltau(u, v).statistic)

    def spearman_rho(self, *params):
        """Spearman's rho. Default: empirical estimate from a large sample."""
        from scipy.stats import spearmanr

        u, v = self.sample_uv(50_000, params, random_state=0)
        return float(spearmanr(u, v).statistic)

    def tail_dependence(self, *params):
        """Lower/upper tail-dependence coefficients ``(lambda_L, lambda_U)``.

        Default ``(0.0, 0.0)`` (no tail dependence); families override.
        """
        return (0.0, 0.0)

    # -- sampling ---------------------------------------------------------
    def sample_uv(self, size, params, random_state=None):
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

    def _invert_du(self, u, w, params, iters=60):
        lo = onp.full_like(onp.asarray(u, dtype=float), _EPS)
        hi = onp.full_like(lo, 1 - _EPS)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            over = onp.asarray(self.du(u, mid, *params)) > w
            hi = onp.where(over, mid, hi)
            lo = onp.where(over, lo, mid)
        return 0.5 * (lo + hi)

    # -- likelihood primitives -------------------------------------------
    def _eval(self, u, v, diff_u, diff_v, params):
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
    def _op_terms(code, upoint, ulo, uhi):
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

    def _pair_loglik(self, params, d0, d1):
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

    def _trunc_logmass(self, params, d0, d1):
        """Log copula mass over the per-row truncation rectangle."""
        ul0, ur0 = d0["ul"], d0["ur"]
        ul1, ur1 = d1["ul"], d1["ur"]
        mass = (
            onp.asarray(self.cdf(ur0, ur1, *params))
            - onp.asarray(self.cdf(ul0, ur1, *params))
            - onp.asarray(self.cdf(ur0, ul1, *params))
            + onp.asarray(self.cdf(ul0, ul1, *params))
        )
        return onp.log(onp.clip(mass, _TINY, None))

    # -- fitting ----------------------------------------------------------
    def _prepare_dim(self, margin, x, c, xl, xr, tl, tr):
        """Transform one dimension's data into copula (u-space) arrays."""
        u = onp.clip(onp.asarray(margin.ff(x), dtype=float), _EPS, 1 - _EPS)
        ulo = onp.clip(onp.asarray(margin.ff(xl), dtype=float), _EPS, 1 - _EPS)
        uhi = onp.clip(onp.asarray(margin.ff(xr), dtype=float), _EPS, 1 - _EPS)
        with onp.errstate(divide="ignore"):
            logf = onp.log(onp.clip(onp.asarray(margin.df(x)), _TINY, None))
        has_trunc = bool(onp.isfinite(tl).any() or onp.isfinite(tr).any())
        ul = onp.where(onp.isfinite(tl), onp.asarray(margin.ff(tl)), 0.0)
        ur = onp.where(onp.isfinite(tr), onp.asarray(margin.ff(tr)), 1.0)
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

    def neg_ll(self, params, dims, weights):
        ll = self._pair_loglik(params, dims[0], dims[1])
        return -float(onp.sum(weights * ll))

    def fit(
        self,
        x,
        c=None,
        n=None,
        t=None,
        margins=None,
        how="IFM",
        xl=None,
        xr=None,
    ):
        """Fit the copula and its margins to multivariate survival data.

        Parameters
        ----------
        x, c, n, t, xl, xr
            Multivariate survival data; see
            :class:`MultivariateSurpyvalData` for the accepted shapes.
        margins : sequence of length D
            Either surpyval distribution classes (e.g. ``surpyval.Weibull``)
            to be fitted, or already-fitted models exposing ``ff``/``df``.
        how : {"IFM", "MLE"}
            ``"IFM"`` (default) fits each margin independently then the
            single copula parameter (robust two-stage estimation).
            ``"MLE"`` jointly optimises copula parameter + margin parameters.
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

        margin_models = self._fit_margins(margins, data)
        if how == "IFM":
            theta = self._fit_theta(margin_models, data)
        elif how == "MLE":
            theta, margin_models = self._fit_joint(
                margins, margin_models, data
            )
        else:
            raise ValueError("how must be 'IFM' or 'MLE'")

        return CopulaModel(self, theta, margin_models, data=data, how=how)

    def from_params(self, params, margins):
        """Build a :class:`CopulaModel` from a known parameter and margins."""
        from surpyval.multivariate.parametric.copula.copula_model import (
            CopulaModel,
        )

        params = onp.atleast_1d(onp.asarray(params, dtype=float))
        return CopulaModel(self, params, list(margins), data=None, how="given")

    def _fit_margins(self, margins, data):
        models = []
        for d, margin in enumerate(margins):
            if not hasattr(margin, "fit"):
                models.append(margin)  # already a fitted model
                continue
            # Reuse the univariate fitter, honouring each margin's own
            # censoring. Interval entries (c == 2) are passed via xl/xr,
            # exactly as the univariate API expects.
            xd, cd, xld, xrd, _, _ = data.dimension(d)
            if (cd == 2).any():
                # surpyval mixes interval and point data in one 2-column x:
                # point rows have equal columns, interval rows carry [xl, xr].
                x2 = onp.column_stack(
                    [onp.where(cd == 2, xld, xd), onp.where(cd == 2, xrd, xd)]
                )
                models.append(margin.fit(x=x2, c=cd))
            else:
                models.append(margin.fit(x=xd, c=cd))
        return models

    def _bounds_transforms(self):
        from surpyval.univariate.parametric.fitters import bounds_convert

        param_map = {n: i for i, n in enumerate(self.param_names)}
        to_unbounded, to_bounded, const, _, _ = bounds_convert(
            None, self.bounds, None, param_map
        )
        return to_unbounded, to_bounded

    def _fit_theta(self, margin_models, data):
        dims = [
            self._prepare_dim(margin_models[d], *data.dimension(d))
            for d in range(data.D)
        ]
        to_unbounded, to_bounded = self._bounds_transforms()

        def obj(phi):
            params = to_bounded(phi)
            return self.neg_ll(params, dims, data.n)

        init = to_unbounded(self._init_theta(dims))
        res = minimize(obj, init, method="Nelder-Mead")
        return onp.asarray(to_bounded(res.x), dtype=float)

    def _fit_joint(self, margins, margin_models, data):
        # Start from the IFM solution, then refine copula + margin params
        # jointly. Margins are re-evaluated from their parameter vectors at
        # each step via ``from_params``.
        theta0 = self._fit_theta(margin_models, data)
        dist_classes = [m.dist for m in margin_models]
        splits = onp.cumsum([len(m.params) for m in margin_models])[:-1]
        to_unbounded, to_bounded = self._bounds_transforms()

        def unpack(phi):
            theta = to_bounded(phi[: len(self.param_names)])
            rest = phi[len(self.param_names) :]
            parts = onp.split(rest, splits)
            models = [
                dist_classes[d].from_params(parts[d]) for d in range(data.D)
            ]
            return theta, models

        def obj(phi):
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

    def _init_theta(self, dims):
        """Initial parameter guess. Override per family for robustness."""
        return onp.asarray([1.0])

    @staticmethod
    def _emp_tau(dims):
        """Empirical Kendall's tau over rows where both dims are observed."""
        from scipy.stats import kendalltau

        both = (dims[0]["c"] == 0) & (dims[1]["c"] == 0)
        if both.sum() < 3:
            return 0.0
        tau = kendalltau(dims[0]["u"][both], dims[1]["u"][both]).statistic
        return 0.0 if not onp.isfinite(tau) else float(tau)
