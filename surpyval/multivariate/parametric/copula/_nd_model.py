"""The fitted model of a copula of three or more series."""

from __future__ import annotations

from typing import Any

import numpy as onp
import numpy.typing as npt

from surpyval.utils.shapes import keeps_query_shape

from .copula_model import _U_CLIP, CopulaModel


class MultivariateCopulaModel(CopulaModel):
    """A fitted copula of three or more series with their margins:
    what ``Gaussian.fit`` and ``StudentT.fit`` return for data with more
    than two axes.

    It is a :class:`CopulaModel` -- the same parameters, likelihood,
    information criteria, covariance, bounds and saved form -- whose
    points have a value per series: ``sf``, ``cdf`` and ``pdf`` take
    ``(x_1, ..., x_D)`` rows, ``random`` returns them, and the dependence
    measures are matrices with an entry per pair of series.

    Attributes
    ----------
    copula : Copula
        The family, for this number of series.
    params : numpy.ndarray
        The copula's parameters, named by ``parameter_names``.
    margins : list
        The margin models, one per series.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.multivariate import Gaussian
    >>> margins = [Weibull.from_params([10 * (d + 1), 2]) for d in range(3)]
    >>> model = Gaussian.from_params([0.5, 0.2, 0.4], margins)
    >>> model
    Copula SurPyval Model
    =====================
    Copula    : Gaussian
    Parameters: rho_0_1=0.5, rho_0_2=0.2, rho_1_2=0.4
    Margins   : Weibull, Weibull, Weibull
    Fitted by : given
    >>> model.sf([[5, 10, 15], [10, 20, 30]]).round(4)
    array([0.5513, 0.1157])
    >>> model.kendall_tau().round(3)
    array([[1.   , 0.333, 0.128],
           [0.333, 1.   , 0.262],
           [0.128, 0.262, 1.   ]])
    """

    @property
    def n_axes(self) -> int:
        """The number of series."""
        return len(self.margins)

    @property
    def correlation(self) -> npt.NDArray:
        """The correlation matrix of an elliptical (Gaussian or
        Student-t) copula."""
        if not hasattr(self.copula, "correlation"):
            raise ValueError(
                f"The {self.copula.name} copula has no correlation matrix; "
                "kendall_tau() gives each pair's rank correlation."
            )
        return self.copula.correlation(*self.params)

    @property
    def edges(self) -> list:
        """A vine's pair-copulas, edge by edge (tree by tree): each a
        dictionary of its ``tree``, the ``pair`` of series it joins, the
        series it is ``given``, its ``family`` and ``rotation``, its
        ``params`` and its ``kendall_tau``."""
        if not hasattr(self.copula, "describe"):
            raise ValueError(
                f"The {self.copula.name} copula is not a vine, so it has no "
                "edges; correlation gives its correlation matrix."
            )
        return self.copula.describe(self.params)

    # -- internal ---------------------------------------------------------
    def _columns(self, x: npt.ArrayLike) -> npt.NDArray:
        x = onp.atleast_2d(onp.asarray(x, dtype=float))
        if x.shape[1] != self.n_axes:
            raise ValueError(
                f"x must have {self.n_axes} columns (one per series), got "
                f"{x.shape[1]}."
            )
        return x

    def _probabilities(self, x: npt.NDArray, margins: Any = None) -> Any:
        margins = self.margins if margins is None else margins
        with onp.errstate(all="ignore"):
            return onp.column_stack(
                [
                    onp.clip(m.ff(x[:, d]), _U_CLIP, 1 - _U_CLIP)
                    for d, m in enumerate(margins)
                ]
            )

    # -- joint survival interface ----------------------------------------
    @keeps_query_shape(point_ndim=1)
    def cdf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Joint CDF :math:`P(X_1 \\le x_1, \\dots, X_D \\le x_D)`."""
        x = self._columns(x)
        if x.shape[0] == 0:
            return onp.empty(0)
        U = self._probabilities(x)
        return onp.asarray(self.copula.cdf(U, *self.params))

    @keeps_query_shape(point_ndim=1)
    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Joint survival :math:`P(X_1 > x_1, \\dots, X_D > x_D)`: the
        chance that every series outlasts its time."""
        x = self._columns(x)
        if x.shape[0] == 0:
            return onp.empty(0)
        U = self._probabilities(x)
        return onp.asarray(self.copula._survival(U, *self.params))

    @keeps_query_shape(point_ndim=1)
    def pdf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Joint density: the copula's density times the margins'."""
        x = self._columns(x)
        if x.shape[0] == 0:
            return onp.empty(0)
        out = onp.asarray(
            self.copula.pdf(self._probabilities(x), *self.params)
        )
        for d, m in enumerate(self.margins):
            out = out * onp.asarray(m.df(x[:, d]))
        return out

    @keeps_query_shape(point_ndim=1)
    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        """Alias of :meth:`cdf`."""
        return self.cdf(x)

    @keeps_query_shape(point_ndim=1)
    def conditional_cdf(
        self, x: npt.ArrayLike, of: int = 0, given: Any = None
    ) -> npt.NDArray:
        """:math:`P(X_{of} \\le x_{of} \\mid X_j = x_j, j \\in given)`:
        the distribution of one series given the values of others.

        Parameters
        ----------
        x : array like
            The points, a value per series (the values of the series
            that are neither ``of`` nor in ``given`` are not used).
        of : int, optional
            The series whose conditional distribution is returned.
            Default 0.
        given : sequence of int, optional
            The series conditioned on. Default: all the others.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Gaussian
        >>> margins = [Weibull.from_params([10, 2])] * 3
        >>> model = Gaussian.from_params([0.5, 0.2, 0.4], margins)
        >>> float(model.conditional_cdf([8.0, 12.0, 12.0], of=0).round(4))
        0.3112
        >>> x = [8.0, 12.0, 12.0]
        >>> float(model.conditional_cdf(x, of=0, given=[2]).round(4))
        0.4145
        """
        D = self.n_axes
        if isinstance(of, bool) or of not in range(D):
            raise ValueError(f"of must be a series, 0 to {D - 1}, got {of!r}.")
        others = [d for d in range(D) if d != of]
        given = others if given is None else list(onp.atleast_1d(given))
        if (
            not given
            or len(set(given)) != len(given)
            or any(g not in others for g in given)
        ):
            raise ValueError(
                f"given must be distinct series other than of={of} (from "
                f"{others}), got {given!r}."
            )
        x = self._columns(x)
        if x.shape[0] == 0:
            return onp.empty(0)
        U = self._probabilities(x)
        return onp.asarray(
            self.copula.conditional_cdf(
                U, int(of), [int(g) for g in given], self.params
            )
        )

    # -- sampling ---------------------------------------------------------
    def random(
        self,
        size: int | tuple[int, ...],
        random_state: int | None = None,
    ) -> npt.NDArray:
        """Draw correlated samples: an array of shape ``(size, D)`` for an
        integer ``size``, one row per draw, or ``(*size, D)`` for a
        tuple (the series on the last axis)."""
        shape: tuple[int, ...] = (
            (int(size),)
            if isinstance(size, (int, onp.integer))
            else tuple(size)
        )
        count = int(onp.prod(shape))
        U = self.copula.sample(count, self.params, random_state)
        X = onp.column_stack(
            [
                onp.asarray(m.qf(U[:, d]), dtype=float).ravel()
                for d, m in enumerate(self.margins)
            ]
        )
        return X.reshape(shape + (self.n_axes,))

    # -- bounds -----------------------------------------------------------
    def cb(self, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        """Confidence bounds on the joint ``sf`` or ``ff``, as
        :meth:`CopulaModel.cb`. A vine has neither, so it refuses, saying
        what to use, whatever else the model has."""
        if hasattr(self.copula, "_no_joint_probability"):
            raise self.copula._no_joint_probability(
                "its joint survival function or CDF, so no bound on them"
            )
        return super().cb(x, *args, **kwargs)

    def _joint_at(
        self,
        x: npt.NDArray,
        joint: str,
        params: Any,
        margins: list,
        complement: bool = False,
    ) -> tuple[npt.NDArray, npt.NDArray | None]:
        """``(value, complement)`` of the joint ``sf`` or ``ff`` at the
        points ``x``; the complement is ``1 - value``."""
        U = self._probabilities(x, margins)
        with onp.errstate(all="ignore"):
            f = self.copula._survival if joint == "sf" else self.copula.cdf
            value = onp.asarray(f(U, *params), dtype=float)
        value = onp.clip(value, _U_CLIP, 1 - _U_CLIP)
        return value, (1.0 - value if complement else None)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        """
        Serialise to a plain dictionary, as :meth:`CopulaModel.to_dict`,
        with the number of series (and the family's structure). The
        family is saved under a name of its own, which a SurPyval older
        than v0.29 does not know and refuses.
        """
        return super().to_dict()

    def _saved_family(self) -> dict:
        # (before the schema is stamped, so the document is the one the
        # stamp read)
        return dict(self.copula._saved())


def restored_family(model_dict: dict) -> Any:
    """The family of a saved copula of more than two series, or ``None``
    for any other saved copula."""
    name = model_dict.get("copula")
    if name in ("MultivariateGaussian", "MultivariateStudentT"):
        from .elliptical_nd import for_axes

        return for_axes(
            name[len("Multivariate") :],
            int(model_dict["axes"]),
            model_dict.get("structure", "unstructured"),
        )
    if name == "Vine":
        from .vine import restored

        return restored(model_dict)
    return None
