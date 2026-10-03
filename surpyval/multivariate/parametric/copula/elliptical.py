"""Elliptical copulas: the Gaussian and Student-t copulas.

The Gaussian copula's CDF is the bivariate-normal CDF, which has no
``autograd`` path, so every primitive here is supplied in closed form using
``scipy``. The single parameter is the correlation ``rho in (-1, 1)``; it is
optimised on a ``tanh`` reparameterisation so the optimiser stays in range.

The Student-t copula adds the degrees of freedom ``nu > 0`` (optimised on
the log scale). Its h-function, density and conditional inverse are closed
forms in the univariate t distribution; its CDF, the bivariate t CDF, is
the integral of the h-function, evaluated by tanh-sinh quadrature (see
:meth:`StudentTCopula.cdf`).
"""

from typing import Any

import numpy as onp
import numpy.typing as npt
from scipy.special import ndtr, ndtri, poch, stdtr, stdtrit
from scipy.stats import multivariate_normal

from surpyval.multivariate.parametric.copula.copula import _U_CLIP, Copula
from surpyval.utils.no_maximum import warn_no_maximum
from surpyval.utils.rng import as_generator

# The fit's start is kept this far inside (-1, 1): a start of rho = +-1
# (from a Kendall's tau of +-1) has no tanh-scale value to search from.
_RHO_START = 0.9999


def _one_minus_rho2(rho: float) -> float:
    """``1 - rho**2`` as ``(1 - |rho|)(1 + |rho|)``, accurate however near
    ``|rho|`` is to 1 (``1 - |rho|`` is exact for ``|rho| >= 1/2``), where
    ``1 - rho**2`` loses the digits of ``rho**2`` that round away."""
    r = abs(rho)
    return (1.0 - r) * (1.0 + r)


def _toward(rho: float) -> float:
    """The sign of the dependence, +1 for ``rho >= 0``. As ``|rho|`` nears
    1 the two quantiles of a likely point near each other (or each
    other's negative), so the quadratic forms below are written in their
    difference ``a - s b``, which then cancels nothing."""
    return 1.0 if rho >= 0 else -1.0


def _neg_ll_inside(copula: Copula, params: Any, dims: list, w: Any) -> float:
    """The negative log-likelihood, ``inf`` at ``|rho| = 1`` (where the
    search's tanh rounds to 1): no copula of the family is there, and its
    formulas divide by ``1 - rho**2``."""
    if not abs(float(params[0])) < 1.0:
        return onp.inf
    return Copula.neg_ll(copula, params, dims, w)


def _warn_if_rho_at_bound(
    copula: Copula, margin_models: list, data: Any, theta: npt.NDArray
) -> bool:
    """Warn when the search ran ``rho`` to +-1 on data that are not
    perfectly dependent (those are told so first; see
    ``Copula._warn_if_perfectly_dependent``), as data censored in one
    dimension can do. Returns whether it warned.

    The criterion: the bound is at least as likely as the ``rho`` reached
    (the likelihood at the last double before it, ``1 - 2**-53`` from
    it, is no lower), and the ``rho`` reached is more likely than
    ``rho = 0`` (the data say something about ``rho``). The likelihood
    then keeps increasing to the bound, or rises to a plateau there (rows
    whose likelihood tends to 1, such as a censored row whose bound lies
    below the comonotone value, reach 1 in floating point before ``rho``
    does). At an interior maximum the bound is far less likely, and a
    likelihood flat in ``rho`` is as high at 0, so neither warns. Any
    other parameter (the t copula's ``nu``) is kept at its value.
    """
    rho = float(theta[0])
    s = _toward(rho)
    edge = onp.array(theta, dtype=float)
    edge[0] = s * onp.nextafter(1.0, 0.0)
    independent = onp.array(theta, dtype=float)
    independent[0] = 0.0
    dims = [
        copula._prepare_dim(margin_models[d], *data.dimension(d))
        for d in range(data.D)
    ]
    nll = copula.neg_ll(theta, dims, data.n)
    nll_edge = copula.neg_ll(edge, dims, data.n)
    nll_independent = copula.neg_ll(independent, dims, data.n)
    if not nll_edge <= nll < nll_independent:
        return False
    bound, kind, relation = (
        ("1", "comonotone", "increasing")
        if s > 0
        else ("-1", "countermonotone", "decreasing")
    )
    warn_no_maximum(
        f"rho runs to {bound}: the likelihood keeps increasing towards "
        f"the {kind} copula (a Frechet bound), which the {copula.name} "
        f"family reaches only as rho tends to {bound}",
        f"The reported rho = {rho!r} and the dependence measures "
        "derived from it are meaningless",
        f"the data are consistent with one variable being an {relation} "
        f"function of the other (the {kind} model): model that "
        "relationship directly rather than with a copula",
    )
    return True


class GaussianCopula(Copula):
    """Gaussian copula, ``rho in (-1, 1)`` (no tail dependence).

    Every function is accurate for any ``|rho| < 1``: the formulas are
    written in ``1 - |rho|`` and in the difference of the two normal
    quantiles, and the CDF is scipy's bivariate normal CDF (Genz's
    algorithm), within 1e-14 of a 40-digit integration up to ``rho = 1 -
    2**-52``.
    """

    name = "Gaussian"
    bounds = ((-1, 1),)
    parameter_names = ["rho"]
    dependence_limits = {1: "rho tends to 1", -1: "rho tends to -1"}

    def neg_ll(self, params: Any, dims: list, weights: npt.NDArray) -> float:
        return _neg_ll_inside(self, params, dims, weights)

    def _warn_if_no_maximum(
        self, margin_models: list, data: Any, theta: npt.NDArray
    ) -> bool:
        return _warn_if_rho_at_bound(self, margin_models, data, theta)

    def cdf(self, u: Any, v: Any, rho: Any) -> Any:
        rho = float(rho)
        a = ndtri(onp.clip(onp.asarray(u, dtype=float), 1e-12, 1 - 1e-12))
        b = ndtri(onp.clip(onp.asarray(v, dtype=float), 1e-12, 1 - 1e-12))
        a, b = onp.broadcast_arrays(a, b)
        # A missing coordinate gives a missing value (scipy's bivariate
        # normal CDF read NaN as a point far below, 0; #382); the point is
        # evaluated at 0 and overwritten.
        missing = onp.isnan(a) | onp.isnan(b)
        pts = onp.stack(
            [
                onp.ravel(onp.where(missing, 0.0, a)),
                onp.ravel(onp.where(missing, 0.0, b)),
            ],
            axis=-1,
        )
        cov = [[1.0, rho], [rho, 1.0]]
        # Within 1e-10 of |rho| = 1 scipy's check of the covariance calls
        # it singular (it is only at 1); its CDF is accurate there.
        out = multivariate_normal.cdf(
            pts, mean=[0.0, 0.0], cov=cov, allow_singular=True
        )
        out = onp.asarray(out).reshape(a.shape)
        return onp.where(missing, onp.nan, out)

    def du(self, u: Any, v: Any, rho: Any) -> Any:
        rho = float(rho)
        a = ndtri(onp.clip(onp.asarray(u, dtype=float), 1e-12, 1 - 1e-12))
        b = ndtri(onp.clip(onp.asarray(v, dtype=float), 1e-12, 1 - 1e-12))
        # b - rho a, from b - s a and 1 - |rho|
        s = _toward(rho)
        num = (b - s * a) + s * (1.0 - abs(rho)) * a
        return ndtr(num / onp.sqrt(_one_minus_rho2(rho)))

    def dv(self, u: Any, v: Any, rho: Any) -> Any:
        return self.du(v, u, rho)

    def pdf(self, u: Any, v: Any, rho: Any) -> Any:
        rho = float(rho)
        a = ndtri(onp.clip(onp.asarray(u, dtype=float), 1e-12, 1 - 1e-12))
        b = ndtri(onp.clip(onp.asarray(v, dtype=float), 1e-12, 1 - 1e-12))
        denom = _one_minus_rho2(rho)
        # (rho^2 (a^2 + b^2) - 2 rho a b) / (2 (1 - rho^2)), its numerator
        # as rho^2 (a - s b)^2 - 2 rho a b (1 - |rho|): no difference of
        # large terms as |rho| nears 1.
        s = _toward(rho)
        quad = rho**2 * (a - s * b) ** 2 / (2.0 * denom) - rho * a * b / (
            1.0 + abs(rho)
        )
        return onp.exp(-quad) / onp.sqrt(denom)

    def kendall_tau(self, rho: float) -> float:  # type: ignore[override]
        return 2.0 / onp.pi * onp.arcsin(float(rho))

    def spearman_rho(self, rho: float) -> float:  # type: ignore[override]
        return 6.0 / onp.pi * onp.arcsin(float(rho) / 2.0)

    def sample_uv(
        self,
        size: int,
        params: Any,
        random_state: "int | None" = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        rho = float(params[0])
        rng = as_generator(random_state)
        z1 = rng.standard_normal(size)
        z2 = rng.standard_normal(size)
        z2 = rho * z1 + onp.sqrt(_one_minus_rho2(rho)) * z2
        return ndtr(z1), ndtr(z2)

    def _bounds_transforms(self) -> tuple:
        # tanh keeps rho inside (-1, 1) during optimisation (but for its
        # rounding to +-1, which ``neg_ll`` rejects).
        def to_unbounded(params: npt.NDArray) -> npt.NDArray:
            return onp.arctanh(onp.asarray(params, dtype=float))

        def to_bounded(phi: npt.NDArray) -> npt.NDArray:
            return onp.tanh(onp.asarray(phi, dtype=float))

        return to_unbounded, to_bounded

    def _init_theta(self, dims: list) -> npt.NDArray:
        rho = onp.sin(onp.pi / 2.0 * self._emp_tau(dims))
        return onp.asarray([onp.clip(rho, -_RHO_START, _RHO_START)])


def _tanh_sinh(step: float = 1.0 / 16.0, reach: float = 3.25) -> tuple:
    """Tanh-sinh (double exponential) rule on ``(0, 1)``: the nodes as
    ``(z, 1 - z)`` (both exact, so the nodes crowding either end keep
    their distance to it) and the weights.

    The rule converges exponentially even for an integrand with algebraic
    singularities at the ends of the interval, which is what the
    h-function of the t copula has (see :meth:`StudentTCopula.cdf`).
    """
    t = onp.arange(-reach, reach + step / 2, step)
    s = 0.5 * onp.pi * onp.sinh(t)
    # z = (1 + tanh(s)) / 2 = expit(2 s), its complement expit(-2 s)
    z = 1.0 / (1.0 + onp.exp(-2.0 * s))
    zc = 1.0 / (1.0 + onp.exp(2.0 * s))
    w = step * onp.pi * onp.cosh(t) * z * zc
    return z, zc, w


_TS_Z, _TS_ZC, _TS_W = _tanh_sinh()


def _log_t_constant(nu: float) -> float:
    """The normalising constant of the t copula density, :math:`\\log
    \\frac{\\Gamma(\\nu/2 + 1)\\Gamma(\\nu/2)}{\\Gamma((\\nu + 1)/2)^2}
    = \\log\\frac{\\nu}{2} - 2 \\log \\frac{\\Gamma(\\nu/2 + 1/2)}
    {\\Gamma(\\nu/2)}`, which tends to 0 like :math:`1 / (2\\nu)`.

    The ratio is scipy's Pochhammer symbol; three ``gammaln`` of about
    ``nu log nu`` each cancelled to an error of 5e-7 at ``nu = 1.3e8``,
    which summed over the rows put the t copula 1.4e-4 above the
    Gaussian copula, its limit, in log-likelihood.
    """
    return float(onp.log(nu / 2.0) - 2.0 * onp.log(poch(nu / 2.0, 0.5)))


class StudentTCopula(Copula):
    """Student-t copula: correlation ``rho in (-1, 1)`` and degrees of
    freedom ``nu > 0`` (symmetric tail dependence).

    The copula of the bivariate t distribution with correlation ``rho``
    and ``nu`` degrees of freedom, the parameterisation of R's
    ``copula::tCopula`` (``param = rho``, ``df = nu``) and of
    ``VineCopula`` family 2 (``par``, ``par2``). Unlike the Gaussian
    copula (its limit as ``nu`` grows) it has tail dependence, equal in
    both tails,

    .. math::
        \\lambda_L = \\lambda_U = 2\\, T_{\\nu + 1}\\left(-\\sqrt{
        \\frac{(\\nu + 1)(1 - \\rho)}{1 + \\rho}}\\right),

    so it is the elliptical copula for joint extremes: failures that come
    together early, or survivals that come together late. Its Kendall's
    tau is the Gaussian's, :math:`\\frac{2}{\\pi}\\arcsin\\rho`, whatever
    ``nu``. The fit starts ``rho`` from the data's Kendall's tau and
    ``nu`` from the best of 1, 2, 4, ..., 64.

    When the data show no more tail dependence than a Gaussian copula,
    the likelihood keeps increasing with ``nu`` (towards that limit) and
    has no finite maximum; the fit then warns and recommends the Gaussian
    copula. The criterion is that the Gaussian copula, with its own best
    ``rho`` and the same margins, is at least as likely (to the
    optimiser's tolerance, 1e-4) as the t copula the search reached.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.multivariate import StudentT
    >>> margins = [
    ...     Weibull.from_params([10, 2]),
    ...     Weibull.from_params([20, 3]),
    ... ]
    >>> model = StudentT.from_params([0.7, 4.0], margins)
    >>> model
    Copula SurPyval Model
    =====================
    Copula    : StudentT
    Parameters: rho=0.7, nu=4
    Margins   : Weibull, Weibull
    Fitted by : given
    >>> round(model.kendall_tau(), 4)
    0.4936
    >>> [round(x, 4) for x in model.tail_dependence()]
    [0.3907, 0.3907]
    """

    name = "StudentT"
    bounds = ((-1, 1), (0, None))
    parameter_names = ["rho", "nu"]
    dependence_limits = {1: "rho tends to 1", -1: "rho tends to -1"}

    def neg_ll(self, params: Any, dims: list, weights: npt.NDArray) -> float:
        return _neg_ll_inside(self, params, dims, weights)

    @staticmethod
    def _args(u: Any, v: Any, rho: Any, nu: Any) -> tuple:
        u, v = onp.broadcast_arrays(
            onp.asarray(u, dtype=float), onp.asarray(v, dtype=float)
        )
        return u, v, float(rho), float(nu)

    @staticmethod
    def _quantile(nu: float, p: Any, q: Any) -> Any:
        """``T_nu^{-1}(p)`` given ``p`` and its complement ``q = 1 - p``,
        each used where it is the smaller (so that ``p`` near 1 keeps its
        accuracy)."""
        return onp.where(
            p <= 0.5,
            stdtrit(nu, onp.minimum(p, 0.5)),
            -stdtrit(nu, onp.minimum(q, 0.5)),
        )

    @staticmethod
    def _h(y: Any, x: Any, rho: float, nu: float) -> Any:
        """The h-function in the t scale: :math:`P(Y \\le y \\mid X = x)
        = T_{\\nu+1}\\left((y - \\rho x) \\big/ \\sqrt{(\\nu + x^2)(1 -
        \\rho^2) / (\\nu + 1)}\\right)`, with ``y - rho x`` formed from
        ``y - s x`` and ``1 - |rho|`` (as the Gaussian's)."""
        scale = onp.sqrt((nu + x**2) * _one_minus_rho2(rho) / (nu + 1.0))
        s = _toward(rho)
        num = (y - s * x) + s * (1.0 - abs(rho)) * x
        return stdtr(nu + 1.0, num / scale)

    def du(self, u: Any, v: Any, rho: Any, nu: Any) -> Any:
        u, v, rho, nu = self._args(u, v, rho, nu)
        x = self._quantile(nu, u, 1.0 - u)
        y = self._quantile(nu, v, 1.0 - v)
        return self._h(y, x, rho, nu)

    def dv(self, u: Any, v: Any, rho: Any, nu: Any) -> Any:
        return self.du(v, u, rho, nu)

    def pdf(self, u: Any, v: Any, rho: Any, nu: Any) -> Any:
        """The density, the bivariate t density over the product of its
        margins' densities, in log space."""
        u, v, rho, nu = self._args(u, v, rho, nu)
        x = self._quantile(nu, u, 1.0 - u)
        y = self._quantile(nu, v, 1.0 - v)
        one_m_r2 = _one_minus_rho2(rho)
        # x^2 + y^2 - 2 rho x y as (x - s y)^2 + 2 s (1 - |rho|) x y
        s = _toward(rho)
        quad = ((x - s * y) ** 2 + 2.0 * s * (1.0 - abs(rho)) * x * y) / (
            nu * one_m_r2
        )
        log_joint = (
            _log_t_constant(nu)
            - 0.5 * onp.log(one_m_r2)
            - (nu + 2.0) / 2.0 * onp.log1p(quad)
            + (nu + 1.0) / 2.0 * (onp.log1p(x**2 / nu) + onp.log1p(y**2 / nu))
        )
        return onp.exp(log_joint)

    def cdf(self, u: Any, v: Any, rho: Any, nu: Any) -> Any:
        """The copula, the bivariate t CDF at the margins' t quantiles.

        It is the integral of the h-function over the first margin,

        .. math::
            C(u, v) = \\int_0^u P(V \\le v \\mid U = s) \\, ds
            = v - \\int_u^1 P(V \\le v \\mid U = s) \\, ds

        (the shorter of the two is used), evaluated by tanh-sinh
        quadrature (105 nodes per piece), split where the integrand
        changes fastest -- where the conditional median of ``Y`` passes
        ``y``, at :math:`s^* = T_\\nu(y / \\rho)` -- so that a sharp step
        at strong dependence sits at the end of a piece, where the rule's
        nodes crowd. The rule also absorbs the algebraic singularities of
        the integrand at ``s = 0`` and ``1`` (it approaches the
        tail-dependence limits as a power of ``s``). It agrees with the
        exact bivariate t CDF of Genz (2004; R's
        ``mvtnorm::pmvt(algorithm = TVPACK())``, integer ``nu`` only) to
        about 1e-11 (7e-11 at ``rho = 0.999``), and with mpmath's
        30-digit integration at non-integer ``nu`` to 1e-15. scipy's
        ``multivariate_t.cdf`` is a randomised quasi-Monte Carlo
        integration: about 1e-4 off at its default tolerances and
        different on every call, which an optimiser cannot use;
        vinecopulib interpolates linearly between the integers either side
        of a non-integer ``nu`` (1.4e-4 off at ``nu = 2.5``).
        """
        u, v, rho, nu = self._args(u, v, rho, nu)
        shape = u.shape
        u, v = u.ravel(), v.ravel()
        if rho == 0.0:
            return (u * v).reshape(shape)
        y = self._quantile(nu, v, 1.0 - v)
        # The split point, kept inside [0, u]: a piece of zero length
        # contributes nothing.
        # Above u = 1/2 the shorter integral is taken, over (u, 1):
        # C(u, v) = v - int_u^1 P(V <= v | U = s) ds.
        upper = u > 0.5
        a = onp.where(upper, u, 0.0)
        b = onp.where(upper, 1.0, u)
        # The split point, kept inside [a, b]: a piece of zero length
        # contributes nothing.
        split = onp.clip(stdtr(nu, y / rho), a, b)
        total = onp.zeros_like(u)
        for lo, hi in ((a, split), (split, b)):
            width = (hi - lo)[:, None]
            # node s and 1 - s, each accurate at its own end
            s = lo[:, None] + width * _TS_Z[None, :]
            s_c = (1.0 - hi)[:, None] + width * _TS_ZC[None, :]
            s = onp.clip(s, 1e-300, None)
            x = self._quantile(nu, s, s_c)
            h = self._h(y[:, None], x, rho, nu)
            total += onp.sum(width * _TS_W[None, :] * h, axis=1)
        total = onp.where(upper, v - total, total)
        return onp.clip(total, 0.0, onp.minimum(u, v)).reshape(shape)

    def kendall_tau(  # type: ignore[override]
        self, rho: float, nu: float
    ) -> float:
        """Kendall's tau, :math:`\\frac{2}{\\pi}\\arcsin\\rho` (the same
        for every elliptical copula, so independent of ``nu``)."""
        return float(2.0 / onp.pi * onp.arcsin(float(rho)))

    def spearman_rho(  # type: ignore[override]
        self, rho: float, nu: float
    ) -> float:
        """Spearman's rho, :math:`12\\,E[UV] - 3`, which has no closed form
        for the t copula.

        :math:`E[UV] = \\int_0^1 u\\, E[V \\mid U = u] \\, du`, and given
        ``U = u`` (``X = x``) the second coordinate is ``Y = rho x +
        sigma(x) Z`` with ``Z`` a t variate with ``nu + 1`` degrees of
        freedom, so :math:`E[V \\mid U = u] = \\int_0^1 T_\\nu(\\rho x +
        \\sigma(x) T_{\\nu+1}^{-1}(q)) \\, dq`. Both integrals are taken by
        tanh-sinh quadrature; the result agrees with the integral of the
        copula's CDF to about 1e-10.
        """
        rho = float(rho)
        nu = float(nu)
        x = self._quantile(nu, _TS_Z, _TS_ZC)[:, None]
        sigma = onp.sqrt((nu + x**2) * _one_minus_rho2(rho) / (nu + 1.0))
        # The q-integral is split where the argument of T_nu crosses 0:
        # far in the tails (x large) the integrand steps there from 0 to 1.
        cut = stdtr(nu + 1.0, -rho * x / sigma)
        cond_mean = onp.zeros(len(_TS_Z))
        for lo, hi in ((onp.zeros_like(cut), cut), (cut, onp.ones_like(cut))):
            width = hi - lo
            q = lo + width * _TS_Z[None, :]
            q_c = (1.0 - hi) + width * _TS_ZC[None, :]
            z = self._quantile(nu + 1.0, onp.clip(q, 1e-300, None), q_c)
            cond_mean += width[:, 0] * (stdtr(nu, rho * x + sigma * z) @ _TS_W)
        return float(12.0 * onp.sum(_TS_W * _TS_Z * cond_mean) - 3.0)

    def tail_dependence(  # type: ignore[override]
        self, rho: float, nu: float
    ) -> tuple:
        rho = float(rho)
        lam = 2.0 * stdtr(
            nu + 1.0, -onp.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
        )
        return (float(lam), float(lam))

    def sample_uv(
        self,
        size: int,
        params: Any,
        random_state: "int | None" = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Draw ``(u, v)`` pairs by conditional inversion in closed form:
        given ``u`` (``x``) and a uniform ``w``,
        :math:`v = T_\\nu(\\rho x + \\sigma(x) T_{\\nu+1}^{-1}(w))`."""
        rho = float(params[0])
        nu = float(params[1])
        rng = as_generator(random_state)
        u = rng.uniform(_U_CLIP, 1 - _U_CLIP, size=size)
        w = rng.uniform(_U_CLIP, 1 - _U_CLIP, size=size)
        x = self._quantile(nu, u, 1.0 - u)
        z = self._quantile(nu + 1.0, w, 1.0 - w)
        sigma = onp.sqrt((nu + x**2) * _one_minus_rho2(rho) / (nu + 1.0))
        y = rho * x + sigma * z
        # T_nu(y), formed from the smaller tail for accuracy near 1
        v = onp.where(y <= 0, stdtr(nu, y), 1.0 - stdtr(nu, -y))
        return u, onp.clip(v, _U_CLIP, 1 - _U_CLIP)

    def _bounds_transforms(self) -> tuple:
        # tanh for rho (as the Gaussian's) and log for nu.
        def to_unbounded(params: npt.NDArray) -> npt.NDArray:
            params = onp.asarray(params, dtype=float)
            return onp.array(
                [
                    onp.arctanh(params[0]),
                    onp.log(params[1]),
                ]
            )

        def to_bounded(phi: npt.NDArray) -> npt.NDArray:
            phi = onp.asarray(phi, dtype=float)
            return onp.array([onp.tanh(phi[0]), onp.exp(phi[1])])

        return to_unbounded, to_bounded

    # The degrees of freedom the starting search compares
    _NU_GRID = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)

    def _init_theta(self, dims: list) -> npt.NDArray:
        rho = float(onp.sin(onp.pi / 2.0 * self._emp_tau(dims)))
        rho = float(onp.clip(rho, -0.99, 0.99))
        weights = onp.ones(len(dims[0]["c"]))
        nll = [self.neg_ll([rho, nu], dims, weights) for nu in self._NU_GRID]
        return onp.asarray([rho, self._NU_GRID[int(onp.nanargmin(nll))]])

    def _warn_if_no_maximum(
        self, margin_models: list, data: Any, theta: npt.NDArray
    ) -> bool:
        """Warn when the Gaussian copula, the limit ``nu -> inf``, is at
        least as likely as the t copula the search reached (see the class
        docstring), or when ``rho`` ran to +-1 (as for the Gaussian
        copula; one warning). Returns whether it warned."""
        if _warn_if_rho_at_bound(self, margin_models, data, theta):
            return True
        dims = [
            self._prepare_dim(margin_models[d], *data.dimension(d))
            for d in range(data.D)
        ]
        nll_t = self.neg_ll(theta, dims, data.n)
        rho = Gaussian._fit_theta(margin_models, data, init=theta[:1])
        nll_gauss = Gaussian.neg_ll(rho, dims, data.n)
        if not nll_gauss <= nll_t + 1e-4:
            return False
        warn_no_maximum(
            "nu grows without bound: the data show no more tail "
            "dependence than the Gaussian copula, the limit of the t "
            f"copula as nu grows (log-likelihood {-nll_gauss:.6g} with "
            f"rho = {float(rho[0]):.4g}, against {-nll_t:.6g} for the t "
            "copula reached)",
            f"The reported nu = {theta[1]:.4g} and the tail dependence "
            "derived from it are meaningless",
            "fit the Gaussian copula instead",
        )
        return True


Gaussian = GaussianCopula()
StudentT = StudentTCopula()
