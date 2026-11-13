"""The life of a unit at fixed covariates: a regression as a plain life
model.

``model.at(Z)`` binds one covariate row to a fitted regression and returns
a :class:`LifeAtCovariates`, whose functions take times (or probabilities)
only, as a univariate model's do: ``sf(x)`` is ``model.sf(x, Z)``. It is
how a regression's results are most often used downstream ("the life of
this pump at this load"), and it can be handed to code that expects a
plain life model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.shapes import covariate_rows


def _covariate_width(regression: Any) -> int:
    """The number of covariate columns a row of ``regression`` has."""
    for name in ("_n_covariates", "_covariate_width"):
        width = getattr(regression, name, None)
        if callable(width):
            return int(width())
    return int(np.asarray(regression.beta).shape[0])


def _one_row(regression: Any, Z: Any) -> Any:
    """``Z`` as the one covariate row the regression's functions take,
    refused, with what to do instead, where it is several rows."""
    if type(Z).__name__ == "DataFrame":
        n_rows = len(Z)
        row = Z
    else:
        arr = np.asarray(Z, dtype=float)
        if arr.ndim > 2:
            raise ValueError(
                "at() takes one covariate row; got an array with {} "
                "dimensions.".format(arr.ndim)
            )
        rows = covariate_rows(arr, _covariate_width(regression))
        n_rows = rows.shape[0]
        row = rows[:1]
    if n_rows != 1:
        raise ValueError(
            "at() takes one covariate row and gives the life of a unit "
            "with those covariates; got {} rows. Call it once per row, "
            "[model.at(row) for row in Z], or pass every row to the "
            "model's own functions, model.sf(x, Z).".format(n_rows)
        )
    return row


class LifeAtCovariates:
    """
    The lifetime distribution of a unit with fixed covariates: a fitted
    regression with one covariate row bound, as a plain life model.

    ``regression.at(Z)`` returns one. Its functions take times (or
    probabilities) only, as a univariate model's do, and give exactly what
    the regression gives at that row: ``at(Z).sf(x)`` is
    ``regression.sf(x, Z)``, and so on for ``ff``, ``df``, ``hf``, ``Hf``
    and ``qf``. ``random(size)`` is ``qf(U)`` for uniform ``U``, as
    ``Parametric.random`` is, so the same ``random_state`` gives common
    random numbers across models. ``mean``, ``rmst`` and the confidence
    bounds ``cb`` / ``quantile_cb`` are the regression's at the row, where
    the regression has them (they refuse, saying why, where it does not).

    It is a view, not a copy: it holds the regression and the row, and
    every call goes to the regression, so it has no ``to_dict`` of its
    own; save the regression (``regression.to_dict()``) and the row.

    Attributes
    ----------
    regression : object
        The fitted regression.
    Z : array or DataFrame
        The covariate row, a ``(1, p)`` array (or the one-row DataFrame
        given, for a model fitted with ``fit_from_df``).
    stratum : object
        The stratum of a stratified Cox model, else ``None``.

    Examples
    --------
    The life of a unit at a load of 1, from a Weibull proportional
    hazards fit:

    >>> import numpy as np
    >>> from surpyval import Weibull, WeibullPH
    >>> np.random.seed(1)
    >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
    >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
    >>> model = WeibullPH.fit(x, Z)
    >>> life = model.at([1])
    >>> life.sf([1, 2, 3]).round(4)
    array([0.9575, 0.8641, 0.7429])
    >>> bool(np.all(life.sf([1, 2, 3]) == model.sf([1, 2, 3], [1])))
    True
    >>> round(float(life.qf(0.1)), 4)  # its B10 life
    1.6593
    """

    def __init__(self, regression: Any, Z: Any, stratum: Any = None) -> None:
        self.regression = regression
        self.Z = _one_row(regression, Z)
        self.stratum = stratum

    def _kwargs(self) -> dict:
        return {} if self.stratum is None else {"stratum": self.stratum}

    def _call(self, name: str, value: Any, **kwargs: Any) -> Any:
        method = getattr(self.regression, name, None)
        if method is None:
            raise AttributeError(
                "{} has no {}(), so its life at fixed covariates has none "
                "either.".format(type(self.regression).__name__, name)
            )
        return method(value, self.Z, **self._kwargs(), **kwargs)

    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        """The survival function at times ``x``:
        ``regression.sf(x, Z)``."""
        return self._call("sf", x)

    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        """The failure probability (CDF) at times ``x``:
        ``regression.ff(x, Z)``."""
        return self._call("ff", x)

    def df(self, x: npt.ArrayLike) -> npt.NDArray:
        """The density at times ``x``: ``regression.df(x, Z)`` (a jump,
        not a density, for a regression with a step baseline)."""
        return self._call("df", x)

    def hf(self, x: npt.ArrayLike) -> npt.NDArray:
        """The hazard at times ``x``: ``regression.hf(x, Z)``."""
        return self._call("hf", x)

    def Hf(self, x: npt.ArrayLike) -> npt.NDArray:
        """The cumulative hazard at times ``x``:
        ``regression.Hf(x, Z)``."""
        return self._call("Hf", x)

    def qf(self, p: npt.ArrayLike) -> npt.NDArray:
        """The quantile function at probabilities ``p``, the time by which
        a proportion ``p`` have failed: ``regression.qf(p, Z)``."""
        return self._call("qf", p)

    def random(self, size: int, random_state: Any = None) -> npt.NDArray:
        """
        Draw ``size`` lifetimes of a unit with these covariates.

        Each draw is ``qf(u)`` for one uniform ``u``, drawn as
        ``Parametric.random`` draws them: so ``random(size)`` is
        ``qf(U)`` for the same ``U``, and the same ``random_state`` gives
        common random numbers across models. A regression whose curve
        stops above 0 at its last observed time (a step baseline, as
        ``CoxPH``'s) never reaches some ``u``: those draws are ``inf``, the
        unit outliving the data, as its own ``random`` gives them.

        Parameters
        ----------
        size : int
            The number of draws.
        random_state : None, int or numpy.random.Generator, optional
            ``None`` (the default) draws from numpy's global generator, so
            ``np.random.seed`` reproduces the draw; an int or a
            ``Generator`` gives a stream of its own.

        Returns
        -------
        numpy array
            ``size`` lifetimes.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullAFT
        >>> model = WeibullAFT.fit([1.0, 2, 3, 4, 5, 6], [0.0, 1, 0, 1, 0, 1])
        >>> life = model.at([1.0])
        >>> u = np.random.default_rng(3).uniform(size=3)
        >>> bool(np.allclose(life.random(3, random_state=3), life.qf(u)))
        True
        """
        from surpyval.univariate.parametric.parametric import uniform_draws

        u = np.asarray(uniform_draws(size, random_state), dtype=float)
        out = np.asarray(self.qf(u), dtype=float)
        # A probability the curve never reaches: beyond the data.
        return np.where(np.isnan(out) & ~np.isnan(u), np.inf, out)

    def mean(self) -> float:
        """The mean life, ``regression.mean(Z)``, where the regression has
        one: a regression with a step baseline that stops at the last
        observed time has no mean; use :meth:`rmst` or :meth:`qf`."""
        method = getattr(self.regression, "mean", None)
        if method is None:
            raise ValueError(
                "{} has no mean life: its baseline is a step function that "
                "stops at the last observed time, so the area under the "
                "survival curve is not known beyond it. Use the median, "
                "qf(0.5), or the restricted mean, rmst(tau).".format(
                    type(self.regression).__name__
                )
            )
        return float(method(self.Z))

    def rmst(self, tau: float | None, **kwargs: Any) -> Any:
        """The restricted mean survival time up to ``tau``, with its
        interval: ``regression.rmst(tau, Z, **kwargs)``."""
        return self._call("rmst", tau, **kwargs)

    def cb(self, x: npt.ArrayLike, **kwargs: Any) -> npt.NDArray:
        """Confidence bounds on a function at times ``x``:
        ``regression.cb(x, Z, **kwargs)`` (``on="sf"`` by default, and
        the regression's ``alpha_ci``, ``bound``, ``method`` and the
        rest)."""
        method = getattr(self.regression, "cb", None)
        if method is None:
            raise ValueError(
                "{} has no confidence bounds on its predicted curves; use "
                "the bounds on its coefficients (param_cb, summary), or a "
                "parametric regression, whose cb bounds sf at Z.".format(
                    type(self.regression).__name__
                )
            )
        return method(x, self.Z, **self._kwargs(), **kwargs)

    def quantile_cb(self, p: npt.ArrayLike, **kwargs: Any) -> npt.NDArray:
        """Confidence bounds on the quantile ``qf(p)``:
        ``regression.quantile_cb(p, Z, **kwargs)``."""
        return self._call("quantile_cb", p, **kwargs)

    def __repr__(self) -> str:
        if type(self.Z).__name__ == "DataFrame":
            row = self.Z.iloc[0].to_dict()
        else:
            row = np.asarray(self.Z, dtype=float).ravel().tolist()
        extra = "" if self.stratum is None else f", stratum={self.stratum!r}"
        return "LifeAtCovariates({}, Z={}{})".format(
            type(self.regression).__name__, row, extra
        )


def checked_params(params: npt.ArrayLike, names: list[str]) -> npt.NDArray:
    """``params`` as a float vector of one entry per name, refused with
    the expected length and the names otherwise (``with_params``)."""
    values = np.array(params, dtype=float)
    if values.shape != (len(names),):
        raise ValueError(
            "with_params takes {} parameters, in the order of "
            "covariance_names {}; got shape {}.".format(
                len(names), names, np.shape(params)
            )
        )
    return values


def rebuilt(model: Any, drop: tuple[str, ...]) -> Any:
    """A new model of ``model``'s class with its attributes but those in
    ``drop`` (which fall back to the class defaults), for
    ``with_params``."""
    cls: Any = type(model)
    out = cls.__new__(cls)
    out.__dict__.update(
        {k: v for k, v in model.__dict__.items() if k not in drop}
    )
    return out


def draws_by_quantile(
    model: Any, size: int, rows: npt.NDArray, random_state: Any
) -> tuple[npt.NDArray, npt.NDArray]:
    """``size`` draws at each covariate row of ``rows`` by inverting the
    model's ``qf``: a uniform ``U`` (the survival the draw is to have)
    gives ``qf(1 - U, row)``, ``inf`` where the curve, stopping at the
    last observed time, never reaches ``1 - U``. The draws and the row
    each was drawn at, as every regression's ``random`` returns them."""
    from ._fit_skeleton import draws_by_row

    def invert(U: npt.NDArray, row: npt.NDArray) -> npt.NDArray:
        out = np.asarray(model.qf(1.0 - U, row[None, :]), dtype=float)
        return np.where(np.isnan(out), np.inf, out)

    return draws_by_row(size, rows, random_state, invert)


class AtCovariatesMixin:
    """``at(Z)``: the regression's lifetime distribution at one covariate
    row, as a :class:`LifeAtCovariates`."""

    def at(self, Z: Any) -> LifeAtCovariates:
        """
        The life of a unit with the covariates ``Z``, as a plain life
        model: an object whose ``sf``, ``ff``, ``df``, ``hf``, ``Hf``,
        ``qf``, ``random`` and ``mean`` take times (or probabilities)
        only, each the model's own at ``Z`` (``at(Z).sf(x)`` is
        ``sf(x, Z)``). Hand it to code that expects a univariate model.

        Parameters
        ----------
        Z : array like or DataFrame
            One covariate row (a scalar for a one-covariate model; a
            one-row DataFrame for a model fitted with ``fit_from_df``).
            Several rows are refused: call it once per row.

        Returns
        -------
        LifeAtCovariates
            A view on this model at ``Z``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPH
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> round(float(model.at([0]).mean()), 3)
        8.575
        """
        return LifeAtCovariates(self, Z)
