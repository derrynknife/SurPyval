"""The Gaussian and Student-t copulas of three or more series.

``Gaussian.fit`` and ``StudentT.fit`` given data with more than two axes
(and their ``from_params`` given more than two margins) use the families
here: the copula of a multivariate normal, or t, vector with a
correlation matrix.

The correlation matrix is ``structure="unstructured"`` (a correlation per
pair of series, ``rho_i_j``) or ``"exchangeable"`` (one ``rho`` for every
pair). It is searched in coordinates that keep it positive definite: the
unstructured matrix through its canonical partial correlations (the
Cholesky factor's rows as successive partial correlations, Lewandowski,
Kurowicka and Joe 2009), each on Fisher's z scale, and the exchangeable
``rho`` on the logit scale of its range ``(-1 / (D - 1), 1)``.

A row's likelihood is the copula density of its observed coordinates
times the probability, given them, that the censored coordinates lie in
their rectangle (a conditional normal, or t, vector), from
``_rectangle.py``: closed forms for one or two censored coordinates and a
deterministic quadrature for three. Rows with more censored coordinates
are refused (the integral is then not accurate enough to fit with), as is
truncation.
"""

from __future__ import annotations

from typing import Any

import numpy as onp
import numpy.typing as npt
from scipy.special import ndtr, ndtri, poch, stdtr

from surpyval.utils.no_maximum import warn_no_maximum
from surpyval.utils.rng import as_generator
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import check_option

from ._rectangle import MAX_LIKELIHOOD_AXES, rectangle
from .copula import _LOG_FLOOR, _U_CLIP, Copula, _check_margins
from .elliptical import StudentT, StudentTCopula

#: The forms of the correlation matrix.
STRUCTURES = ("unstructured", "exchangeable")
# A start is kept this far inside the positive-definite matrices
_MIN_EIGENVALUE = 0.02


def _pair_view(data: Any, i: int, j: int) -> Any:
    """The two series ``i`` and ``j`` of ``data`` (its values, censoring
    codes and counts), for the bivariate checks."""

    class _Pair:
        x = data.x[:, [i, j]]
        c = data.c[:, [i, j]]
        n = data.n

    return _Pair


class EllipticalCopulaND(Copula):
    """The Gaussian or Student-t copula of ``axes`` series (three or
    more), made by ``Gaussian.fit`` and ``StudentT.fit`` on data with that
    many axes, and by their ``from_params``.

    Its parameters are the correlations, ``rho_i_j`` for each pair of
    series ``i < j`` in the order ``(0, 1), (0, 2), ..., (1, 2), ...``
    (``structure="unstructured"``) or one ``rho`` shared by every pair
    (``"exchangeable"``), then the Student-t copula's degrees of freedom
    ``nu``.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.multivariate import Gaussian
    >>> margins = [Weibull.from_params([10 * (d + 1), 2]) for d in range(3)]
    >>> model = Gaussian.from_params([0.5, 0.2, 0.4], margins)
    >>> model.copula
    Gaussian copula of 3 series (unstructured)
    >>> model.parameter_names
    ['rho_0_1', 'rho_0_2', 'rho_1_2']
    >>> model.correlation
    array([[1. , 0.5, 0.2],
           [0.5, 1. , 0.4],
           [0.2, 0.4, 1. ]])
    """

    student = False
    exchangeable = False

    def __init__(self, name: str, axes: int, structure: str) -> None:
        self.name = name
        self.student = name == "StudentT"
        self.n_axes = int(axes)
        self.structure = structure
        self.pairs = [
            (i, j) for i in range(self.n_axes) for j in range(i + 1, axes)
        ]
        if structure == "unstructured":
            names = [f"rho_{i}_{j}" for i, j in self.pairs]
            bounds: list = [(-1, 1)] * len(names)
        else:
            names = ["rho"]
            bounds = [(-1.0 / (self.n_axes - 1), 1)]
        self.n_corr = len(names)
        if self.student:
            names.append("nu")
            bounds.append((0, None))
        self.parameter_names = names
        self.bounds = tuple(bounds)
        self.dependence_limits = {}

    def __repr__(self) -> str:
        return (
            f"{self.name} copula of {self.n_axes} series ({self.structure})"
        )

    def _model_cls(self) -> Any:
        from ._nd_model import MultivariateCopulaModel

        return MultivariateCopulaModel

    def _saved(self) -> dict:
        """What a model's dictionary records of the family."""
        return {
            "copula": f"Multivariate{self.name}",
            "axes": self.n_axes,
            "structure": self.structure,
        }

    # -- the parameters ----------------------------------------------------
    def correlation(self, *params: Any) -> npt.NDArray:
        """The correlation matrix of the parameters.

        Examples
        --------
        >>> from surpyval.multivariate import Gaussian
        >>> from surpyval import Weibull
        >>> margins = [Weibull.from_params([10, 2])] * 3
        >>> model = Gaussian.from_params(
        ...     [0.3], margins, structure="exchangeable"
        ... )
        >>> model.copula.correlation(0.3)
        array([[1. , 0.3, 0.3],
               [0.3, 1. , 0.3],
               [0.3, 0.3, 1. ]])
        """
        values = onp.asarray(params[: self.n_corr], dtype=float)
        R = onp.eye(self.n_axes)
        for k, (i, j) in enumerate(self.pairs):
            R[i, j] = R[j, i] = values[k if self.n_corr > 1 else 0]
        return R

    def _split(self, params: Any) -> tuple:
        """``(R, nu)``; ``nu`` is ``None`` for the Gaussian copula."""
        params = onp.asarray(params, dtype=float)
        nu = float(params[-1]) if self.student else None
        return self.correlation(*params), nu

    def _cholesky(self, params: Any) -> npt.NDArray | None:
        """The Cholesky factor of the correlation matrix, or ``None``
        where the parameters are not a copula's (a matrix that is not
        positive definite, or ``nu <= 0``)."""
        params = onp.asarray(params, dtype=float)
        if not onp.all(onp.isfinite(params)):
            return None
        if self.student and not params[-1] > 0:
            return None
        try:
            return onp.linalg.cholesky(self.correlation(*params))
        except onp.linalg.LinAlgError:
            return None

    def _check_params(self, params: npt.ArrayLike) -> npt.NDArray:
        params = self._flat(params)
        params = super()._check_params(params)
        if self._cholesky(params) is None:
            raise ValueError(
                f"{self.name} copula: the correlations "
                f"{params[: self.n_corr].tolist()} do not make a positive-"
                "definite correlation matrix (each pair's correlation is "
                "limited by the others'); give correlations that do."
            )
        return params

    def _check_init(self, init: npt.ArrayLike) -> npt.NDArray:
        init = super()._check_init(self._flat(init))
        if self._cholesky(init) is None:
            raise ValueError(
                f"init: the correlations {init[: self.n_corr].tolist()} do "
                "not make a positive-definite correlation matrix; give "
                "correlations that do."
            )
        return init

    def _flat(self, params: Any) -> Any:
        """``params`` with a correlation matrix written out as the
        pairs' correlations: a ``(D, D)`` matrix for the Gaussian copula,
        ``(matrix, nu)`` for the Student-t."""
        matrix, rest = params, []
        if (
            self.student
            and isinstance(params, (list, tuple))
            and len(params) == 2
            and onp.ndim(params[0]) == 2
        ):
            matrix, rest = params[0], [float(params[1])]
        if onp.ndim(matrix) != 2:
            return params
        matrix = onp.asarray(matrix, dtype=float)
        D = self.n_axes
        if (
            matrix.shape != (D, D)
            or not onp.allclose(matrix, matrix.T)
            or not onp.allclose(onp.diag(matrix), 1.0)
        ):
            raise ValueError(
                f"{self.name} copula: a correlation matrix must be "
                f"symmetric, {D} by {D}, with a diagonal of 1; got shape "
                f"{matrix.shape}."
            )
        values = [matrix[i, j] for i, j in self.pairs]
        if self.structure == "exchangeable":
            if not onp.allclose(values, values[0]):
                raise ValueError(
                    "An exchangeable copula has one correlation for "
                    "every pair; the matrix given has several. Use "
                    "structure='unstructured'."
                )
            values = values[:1]
        return onp.asarray(values + rest, dtype=float)

    def _bounds_transforms(self) -> tuple:
        D, n_corr, student = self.n_axes, self.n_corr, self.student
        low = -1.0 / (D - 1)
        rows = [(i, j) for i in range(1, D) for j in range(i)]

        def to_unbounded(params: npt.NDArray) -> npt.NDArray:
            params = onp.asarray(params, dtype=float)
            if self.structure == "exchangeable":
                rho = params[0]
                phi = [onp.log(rho - low) - onp.log1p(-rho)]
            else:
                L = self._cholesky(params)
                if L is None:
                    return onp.full(params.shape, onp.nan)
                phi = []
                for i in range(1, D):
                    rest = 1.0
                    for j in range(i):
                        z = L[i, j] / onp.sqrt(rest)
                        phi.append(onp.arctanh(z))
                        rest = rest * (1.0 - z) * (1.0 + z)
            if student:
                phi.append(onp.log(params[-1]))
            return onp.asarray(phi, dtype=float)

        def to_bounded(phi: npt.NDArray) -> npt.NDArray:
            phi = onp.asarray(phi, dtype=float)
            if self.structure == "exchangeable":
                rho = low + (1.0 - low) / (1.0 + onp.exp(-phi[0]))
                out = [rho]
            else:
                z = onp.tanh(phi[:n_corr])
                L = onp.zeros((D, D))
                L[0, 0] = 1.0
                rest = onp.ones(D)
                for k, (i, j) in enumerate(rows):
                    L[i, j] = z[k] * onp.sqrt(rest[i])
                    rest[i] = rest[i] * (1.0 - z[k]) * (1.0 + z[k])
                for i in range(1, D):
                    L[i, i] = onp.sqrt(rest[i])
                R = L @ L.T
                out = [R[i, j] for i, j in self.pairs]
            if student:
                out.append(onp.exp(phi[-1]))
            return onp.asarray(out, dtype=float)

        return to_unbounded, to_bounded

    def _just_inside(self, theta: npt.NDArray) -> npt.NDArray:
        """``theta`` with its correlation matrix shrunk towards the
        identity until it is well inside the positive-definite ones."""
        out = onp.array(theta, dtype=float)
        R = self.correlation(*out)
        off = R - onp.eye(self.n_axes)
        for _ in range(60):
            if onp.linalg.eigvalsh(onp.eye(self.n_axes) + off)[0] > (
                _MIN_EIGENVALUE
            ):
                break
            off = 0.9 * off
        R = onp.eye(self.n_axes) + off
        values = [R[i, j] for i, j in self.pairs]
        out[: self.n_corr] = (
            values if self.n_corr > 1 else float(onp.mean(values))
        )
        if self.n_corr == 1:
            low = self.bounds[0][0]
            out[0] = onp.clip(out[0], 0.98 * low, 0.98)
        return out

    def _init_theta(self, dims: list) -> npt.NDArray:
        taus = [self._emp_tau([dims[i], dims[j]]) for i, j in self.pairs]
        rhos = onp.clip(onp.sin(onp.pi / 2.0 * onp.asarray(taus)), -0.98, 0.98)
        if self.n_corr == 1:
            start = [float(onp.mean(rhos))]
        else:
            start = list(rhos)
        if self.student:
            start.append(4.0)
        theta = self._just_inside(onp.asarray(start, dtype=float))
        if self.student:
            weights = onp.ones(len(dims[0]["c"]))
            grid = StudentTCopula._NU_GRID
            nll = []
            for nu in grid:
                theta[-1] = nu
                nll.append(self.neg_ll(theta, dims, weights))
            theta[-1] = grid[int(onp.nanargmin(nll))]
        return theta

    # -- the likelihood ----------------------------------------------------
    def _z(self, u: Any, nu: float | None) -> Any:
        """The normal (or t) quantiles of the probabilities ``u``."""
        if nu is None:
            return ndtri(u)
        return StudentT._quantile(nu, u, 1.0 - u)

    def _log_density(self, z: Any, R: Any, nu: float | None) -> Any:
        """The log copula density of the quantiles ``z`` (one row per
        point) for the correlation matrix ``R``."""
        p = z.shape[1]
        if p < 2:
            return onp.zeros(z.shape[0])
        _, logdet = onp.linalg.slogdet(R)
        q = onp.einsum("ij,ij->i", z @ onp.linalg.inv(R), z)
        if nu is None:
            return -0.5 * logdet - 0.5 * (q - onp.sum(z * z, axis=1))
        constant = onp.log(poch(nu / 2.0, p / 2.0)) - p * onp.log(
            poch(nu / 2.0, 0.5)
        )
        return (
            constant
            - 0.5 * logdet
            - (nu + p) / 2.0 * onp.log1p(q / nu)
            + (nu + 1.0) / 2.0 * onp.sum(onp.log1p(z * z / nu), axis=1)
        )

    def _conditional(
        self, z: Any, R: Any, nu: float | None, O: Any, C: Any
    ) -> tuple:
        """The coordinates ``C`` given the coordinates ``O`` at ``z``:
        ``(mean, scale, corr, nu)``, each row's conditional mean and
        scale per coordinate, the correlation matrix and the degrees of
        freedom."""
        m = z.shape[0]
        if len(O) == 0:
            return (
                onp.zeros((m, len(C))),
                onp.ones((m, len(C))),
                R[onp.ix_(C, C)],
                nu,
            )
        inv = onp.linalg.inv(R[onp.ix_(O, O)])
        B = R[onp.ix_(C, O)] @ inv
        S = R[onp.ix_(C, C)] - B @ R[onp.ix_(O, C)]
        sd = onp.sqrt(onp.diag(S))
        scale = onp.broadcast_to(sd, (m, len(C)))
        if nu is not None:
            q = onp.einsum("ij,ij->i", z @ inv, z)
            scale = scale * onp.sqrt((nu + q) / (nu + len(O)))[:, None]
            nu = nu + len(O)
        return z @ B.T, scale, S / onp.outer(sd, sd), nu

    def _rows_loglik(self, params: Any, dims: list) -> Any:
        R, nu = self._split(params)
        codes = onp.column_stack([d["c"] for d in dims])
        U = onp.column_stack([d["u"] for d in dims])
        lower = onp.column_stack([d["ulo"] for d in dims])
        upper = onp.column_stack([d["uhi"] for d in dims])
        logf = onp.column_stack([d["logf"] for d in dims])
        observed = codes == 0
        ll = onp.zeros(codes.shape[0])
        patterns, which = onp.unique(observed, axis=0, return_inverse=True)
        which = onp.ravel(which)
        for p, pattern in enumerate(patterns):
            rows = onp.flatnonzero(which == p)
            O, C = onp.flatnonzero(pattern), onp.flatnonzero(~pattern)
            if C.size > MAX_LIKELIHOOD_AXES:
                raise ValueError(_too_censored(self, rows.size, C.size))
            z = self._z(U[onp.ix_(rows, O)], nu)
            out = self._log_density(z, R[onp.ix_(O, O)], nu)
            out = out + onp.sum(logf[onp.ix_(rows, O)], axis=1)
            if C.size:
                c = codes[onp.ix_(rows, C)]
                u = U[onp.ix_(rows, C)]
                lo_u = onp.where(c == 2, lower[onp.ix_(rows, C)], u)
                hi_u = onp.where(c == 2, upper[onp.ix_(rows, C)], u)
                lo = onp.where(c == -1, -onp.inf, self._z(lo_u, nu))
                hi = onp.where(c == 1, onp.inf, self._z(hi_u, nu))
                mean, scale, corr, df = self._conditional(z, R, nu, O, C)
                mass = rectangle(
                    (lo - mean) / scale, (hi - mean) / scale, corr, df
                )
                out = out + onp.log(onp.clip(mass, _LOG_FLOOR, None))
            ll[rows] = out
        return ll

    def neg_ll(self, params: Any, dims: list, weights: npt.NDArray) -> float:
        """The negative log-likelihood; ``inf`` where the parameters are
        not a copula's (the correlation matrix is not positive
        definite)."""
        if self._cholesky(params) is None:
            return onp.inf
        ll = self._rows_loglik(params, dims)
        return -float(onp.sum(weights * ll))

    # -- the copula's functions --------------------------------------------
    def _quantiles(self, U: Any, nu: float | None) -> Any:
        U = onp.clip(onp.asarray(U, dtype=float), _U_CLIP, 1 - _U_CLIP)
        return self._z(U, nu)

    def cdf(self, U: Any, *params: Any) -> Any:  # type: ignore[override]
        """The copula at the points ``U`` (one row per point, a column
        per series)."""
        R, nu = self._split(params)
        z = self._quantiles(U, nu)
        return _where_complete(
            z, lambda v: rectangle(onp.full(v.shape, -onp.inf), v, R, nu)
        )

    def _survival(self, U: Any, *params: Any) -> Any:  # type: ignore[override]
        """:math:`P(U_1 > u_1, \\dots, U_D > u_D)`."""
        R, nu = self._split(params)
        z = self._quantiles(U, nu)
        return _where_complete(
            z, lambda v: rectangle(v, onp.full(v.shape, onp.inf), R, nu)
        )

    def pdf(self, U: Any, *params: Any) -> Any:  # type: ignore[override]
        """The copula's density at the points ``U``."""
        R, nu = self._split(params)
        return onp.exp(self._log_density(self._quantiles(U, nu), R, nu))

    def conditional_cdf(
        self, U: Any, of: int, given: Any, params: Any
    ) -> npt.NDArray:
        """:math:`P(U_{of} \\le u_{of} \\mid U_j = u_j, j \\in given)`."""
        R, nu = self._split(params)
        z = self._quantiles(U, nu)
        O = onp.asarray(list(given), dtype=int)
        mean, scale, _, df = self._conditional(
            z[:, O], R, nu, O, onp.asarray([of])
        )
        t = (z[:, of] - mean[:, 0]) / scale[:, 0]
        return ndtr(t) if df is None else stdtr(df, t)

    def sample(
        self, size: int, params: Any, random_state: Any = None
    ) -> npt.NDArray:
        """``size`` draws from the copula: one row per draw."""
        R, nu = self._split(params)
        rng = as_generator(random_state)
        z = rng.standard_normal((size, self.n_axes)) @ onp.linalg.cholesky(R).T
        if nu is None:
            u = ndtr(z)
        else:
            w = rng.chisquare(nu, size=size) / nu
            t = z / onp.sqrt(w)[:, None]
            u = onp.where(t <= 0, stdtr(nu, t), 1.0 - stdtr(nu, -t))
        return onp.clip(u, _U_CLIP, 1 - _U_CLIP)

    # -- dependence measures (a matrix: an entry per pair) -------------------
    def _pairwise(self, value: Any, params: Any) -> npt.NDArray:
        R, nu = self._split(params)
        out = onp.eye(self.n_axes)
        for i, j in self.pairs:
            out[i, j] = out[j, i] = value(R[i, j], nu)
        return out

    def kendall_tau(self, *params: float) -> Any:
        """Kendall's tau of each pair of series, a matrix:
        :math:`\\frac{2}{\\pi}\\arcsin\\rho_{ij}`."""
        return self._pairwise(
            lambda r, nu: 2.0 / onp.pi * onp.arcsin(r), params
        )

    def spearman_rho(self, *params: float) -> Any:
        """Spearman's rho of each pair of series, a matrix (the
        bivariate copula's of each pair)."""

        def one(r: float, nu: float | None) -> float:
            if nu is None:
                return float(6.0 / onp.pi * onp.arcsin(r / 2.0))
            return float(StudentT.spearman_rho(r, nu))

        return self._pairwise(one, params)

    def tail_dependence(self, *params: float) -> tuple:
        """``(lambda_L, lambda_U)``, each a matrix with a coefficient per
        pair of series (zeros for the Gaussian copula; the diagonal is
        1)."""

        def one(r: float, nu: float | None) -> float:
            return 0.0 if nu is None else StudentT.tail_dependence(r, nu)[0]

        lam = self._pairwise(one, params)
        return lam, lam.copy()

    # -- fitting -------------------------------------------------------------
    @takes_unit
    def fit(  # type: ignore[override]
        self,
        x: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        margins: Any = None,
        how: str = "IFM",
        xl: npt.ArrayLike | None = None,
        xr: npt.ArrayLike | None = None,
        init: npt.ArrayLike | None = None,
        rotation: int = 0,
    ) -> Any:
        """Fit the copula and its margins; see ``Gaussian.fit`` and
        ``StudentT.fit``, which call it for data with more than two
        axes.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Gaussian
        >>> margins = [Weibull.from_params([10 * (d + 1), 2]) for d in range(3)]
        >>> truth = Gaussian.from_params([0.5, 0.2, 0.4], margins)
        >>> X = truth.random(300, random_state=0)
        >>> model = truth.copula.fit(X, margins=[Weibull] * 3)
        >>> model.params.round(2)
        array([0.5 , 0.2 , 0.42])
        """
        from surpyval.multivariate.parametric.data import (
            MultivariateSurpyvalData,
        )

        if rotation:
            raise ValueError(
                f"The {self.name} copula is not rotated (it is radially "
                "symmetric): leave out rotation."
            )
        data = x
        if not isinstance(x, MultivariateSurpyvalData):
            data = MultivariateSurpyvalData(x, c=c, n=n, t=t, xl=xl, xr=xr)
            c = n = t = xl = xr = None
        _check_rows(self, data)
        return Copula.fit(
            self,
            data,
            c=c,
            n=n,
            t=t,
            margins=margins,
            how=how,
            xl=xl,
            xr=xr,
            init=init,
        )

    @takes_unit
    def from_params(  # type: ignore[override]
        self, params: Any, margins: Any, rotation: int = 0
    ) -> Any:
        """The model of known parameters and margins; see
        ``Gaussian.from_params``.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import StudentT
        >>> margins = [Weibull.from_params([10 * (d + 1), 2]) for d in range(3)]
        >>> model = StudentT.from_params([0.5, 0.2, 0.4, 6.0], margins)
        >>> model.copula.from_params([0.1, 0.1, 0.1, 3.0], margins).params
        array([0.1, 0.1, 0.1, 3. ])
        """
        if rotation:
            raise ValueError(
                f"The {self.name} copula is not rotated (it is radially "
                "symmetric): leave out rotation."
            )
        params = self._check_params(params)
        margins = _check_margins(margins, self.n_axes, fitted_only=True)
        return self._model_cls()(
            self, params, margins, data=None, how="given"
        )

    def _warn_if_perfectly_dependent(
        self, data: Any, theta: npt.NDArray
    ) -> bool:
        """Warn when two of the series are perfectly dependent in the
        rows that observe both (see the bivariate families): no
        correlation matrix with that pair's correlation short of +-1 has
        that dependence."""
        from .copula import _perfect_dependence

        for i, j in self.pairs:
            sign, rows = _perfect_dependence(_pair_view(data, i, j))
            if not sign:
                continue
            bound, word, relation = (
                ("1", "concordant", "increasing")
                if sign > 0
                else ("-1", "discordant", "decreasing")
            )
            warn_no_maximum(
                f"the {rows} rows observed in both series {i} and {j} are "
                f"perfectly {word} (Kendall's tau = {bound}), which the "
                f"{self.name} family reaches only as their correlation "
                f"tends to {bound}; the likelihood keeps increasing "
                "towards it, or peaks only where the fitted margins stop "
                "mapping one series exactly onto the other",
                "The reported correlations and the dependence measures "
                "derived from them are meaningless",
                f"series {j} is an {relation} function of series {i}: "
                "model that relationship directly, and fit the copula to "
                "the other series",
            )
            return True
        return False

    def _warn_if_no_maximum(
        self, margin_models: list, data: Any, theta: npt.NDArray
    ) -> bool:
        """Warn when the correlation matrix ran to a singular one, or
        (the Student-t copula) when the Gaussian copula, the limit as
        ``nu`` grows, is at least as likely as the fit reached."""
        R, nu = self._split(theta)
        smallest = float(onp.linalg.eigvalsh(R)[0])
        if smallest < 1e-6:
            warn_no_maximum(
                "the correlation matrix runs to a singular one (its "
                f"smallest eigenvalue is {smallest:.3g}): the likelihood "
                "keeps increasing towards a combination of the series "
                "that is fixed by the others",
                "The reported correlations and the dependence measures "
                "derived from them are meaningless",
                "one series is (nearly) a function of the others: model "
                "that relationship directly, and fit the copula to the "
                "other series",
            )
            return True
        if not self.student:
            return False
        dims = [
            self._prepare_dim(margin_models[d], *data.dimension(d))
            for d in range(data.D)
        ]
        nll_t = self.neg_ll(theta, dims, data.n)
        gaussian = EllipticalCopulaND("Gaussian", self.n_axes, self.structure)
        start = gaussian._just_inside(onp.asarray(theta[:-1], dtype=float))
        rho = gaussian._fit_theta(margin_models, data, init=start)
        nll_gauss = gaussian.neg_ll(rho, dims, data.n)
        if not nll_gauss <= nll_t + 1e-4:
            return False
        warn_no_maximum(
            "nu grows without bound: the data show no more tail "
            "dependence than the Gaussian copula, the limit of the t "
            f"copula as nu grows (log-likelihood {-nll_gauss:.6g}, "
            f"against {-nll_t:.6g} for the t copula reached)",
            f"The reported nu = {theta[-1]:.4g} and the tail dependence "
            "derived from it are meaningless",
            "fit the Gaussian copula instead",
        )
        return True


def _where_complete(z: Any, f: Any) -> npt.NDArray:
    """``f`` of the rows of ``z`` with no missing value; ``nan`` for the
    others."""
    out = onp.full(z.shape[0], onp.nan)
    whole = ~onp.isnan(z).any(axis=1)
    if whole.any():
        out[whole] = f(z[whole])
    return out


def _too_censored(copula: Any, rows: int, censored: int) -> str:
    return (
        f"A {copula.name} copula of {copula.n_axes} series is fitted to "
        f"rows with up to {MAX_LIKELIHOOD_AXES} censored series; {rows} "
        f"row(s) have {censored}. Their likelihood is an integral over "
        "the censored series that is not computed accurately enough to "
        f"fit with beyond {MAX_LIKELIHOOD_AXES}. Fit the copula to "
        f"{MAX_LIKELIHOOD_AXES} of the series at a time, or leave those "
        "rows out and say so with the result."
    )


def _check_rows(copula: Any, data: Any) -> None:
    """Refuse what the fit of more than two series does not take:
    truncation, and rows with too many censored series."""
    if onp.isfinite(data.t).any():
        rows = int(onp.sum(onp.isfinite(data.t).any(axis=(1, 2))))
        raise ValueError(
            f"A {copula.name} copula of {copula.n_axes} series does not "
            f"take truncated data (t; {rows} row(s) have a truncation "
            "window): the truncated likelihood divides by the copula's "
            "mass over the window, which is not computed for more than "
            "two series. A copula of two series takes truncation: fit "
            "the series in pairs."
        )
    censored = onp.sum(data.c != 0, axis=1)
    over = censored > MAX_LIKELIHOOD_AXES
    if over.any():
        raise ValueError(
            _too_censored(copula, int(over.sum()), int(censored.max()))
        )


def for_axes(name: str, axes: int, structure: str) -> EllipticalCopulaND:
    """The ``name`` (``"Gaussian"`` or ``"StudentT"``) copula of ``axes``
    series with the correlation ``structure``."""
    check_option("structure", structure, STRUCTURES)
    return EllipticalCopulaND(name, axes, structure)
