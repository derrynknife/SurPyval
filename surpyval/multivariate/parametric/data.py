"""Aligned multivariate survival data for copula models.

A joint observation is a *row* across ``D`` correlated series. Every series
of a row carries its own censoring code (using the same convention as the
univariate :class:`~surpyval.utils.surpyval_data.SurpyvalData`)::

    c ==  0  observed (exact)
    c ==  1  right censored   (the true value is > x)
    c == -1  left censored    (the true value is < x)
    c ==  2  interval censored (the true value is in [xl, xr])

so that every censoring/truncation type the univariate library supports is
available per-dimension in the joint likelihood. A weight (count) ``n``
applies to the whole row; the truncation window ``t`` is given per row and
per dimension.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class MultivariateSurpyvalData:
    """Normalise and hold row-aligned multivariate survival data.

    Parameters
    ----------
    x : array-like, shape (N, D) or sequence of D length-N arrays
        Point values per dimension. For an interval-censored entry
        (``c == 2``) the point value is ignored and ``xl``/``xr`` are used.
    c : array-like, shape (N, D) or (D,), optional
        Per-dimension censoring codes in ``{0, 1, -1, 2}``. A single row of
        ``D`` codes applies to every row. Defaults to all observed.
    n : array-like, shape (N,), optional
        Integer weight (count) of each row. Defaults to all ones.
    t : array-like, shape (N, D, 2), optional
        Per-dimension truncation window ``[tl, tr]``. Defaults to
        ``(-inf, inf)`` (no truncation).
    xl, xr : array-like, shape (N, D), optional
        Interval-censoring bounds, required where ``c == 2``.

    Raises
    ------
    ValueError
        If an array has the wrong shape, a censoring code is not one of
        the four, or a series (with the counts and its own truncation
        window) is not valid univariate data: a NaN value, a count that
        is not a positive whole number, an interval with ``xl >= xr``, or
        a value outside its truncation window. The message names the
        series (``"Series 0: ..."``).

    Examples
    --------
    A list is read as one sequence per series. Here three rows of two
    series, the second series right censored in every row:

    >>> from surpyval.multivariate import MultivariateSurpyvalData
    >>> data = MultivariateSurpyvalData(
    ...     [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], c=[0, 1]
    ... )
    >>> data.N, data.D
    (3, 2)
    >>> data.c
    array([[0, 1],
           [0, 1],
           [0, 1]])
    >>> data.dimension(1)[:2]
    (array([4., 5., 6.]), array([1, 1, 1]))
    """

    def __init__(
        self,
        x: npt.ArrayLike,
        c: "npt.ArrayLike | None" = None,
        n: "npt.ArrayLike | None" = None,
        t: "npt.ArrayLike | None" = None,
        xl: "npt.ArrayLike | None" = None,
        xr: "npt.ArrayLike | None" = None,
    ) -> None:
        x = self._as_2d(x)
        N, D = x.shape

        if c is None:
            c = np.zeros((N, D), dtype=int)
        else:
            raw = np.asarray(c)
            if D > 1 and raw.shape in ((D,), (1, D)):
                # one code per dimension, shared by every row
                c = np.broadcast_to(raw.reshape(1, D), (N, D)).astype(int)
            else:
                c = self._as_2d(c).astype(int)
            if c.shape != (N, D):
                raise ValueError(f"c must have shape {(N, D)}, got {c.shape}")
            if not np.isin(c, (0, 1, -1, 2)).all():
                raise ValueError("c values must be in {0, 1, -1, 2}")

        if n is None:
            n = np.ones(N, dtype=int)
        else:
            n = np.asarray(n)
            if n.shape != (N,):
                raise ValueError(f"n must have shape {(N,)}, got {n.shape}")

        # Interval bounds: fall back to the point value where not given so the
        # arrays are always well shaped; only the c == 2 entries are read.
        # Without bounds an interval entry would be the zero-width interval
        # [x, x] and contribute zero likelihood, so that is an error.
        if (c == 2).any() and (xl is None or xr is None):
            raise ValueError("interval-censored rows (c == 2) need xl and xr")
        xl = x.copy() if xl is None else self._as_2d(xl)
        xr = x.copy() if xr is None else self._as_2d(xr)
        for name, bound in (("xl", xl), ("xr", xr)):
            if bound.shape != (N, D):
                raise ValueError(
                    f"{name} must have shape {(N, D)}, got {bound.shape}"
                )

        if t is None:
            t = np.empty((N, D, 2))
            t[..., 0] = -np.inf
            t[..., 1] = np.inf
        else:
            t = np.asarray(t, dtype=float)
            if t.shape != (N, D, 2):
                raise ValueError(
                    f"t must have shape {(N, D, 2)}, got {t.shape}"
                )

        self.x = x.astype(float)
        self.c = c
        self.n = n
        self.t = t
        self.xl = xl.astype(float)
        self.xr = xr.astype(float)
        self.N = N
        self.D = D
        for d in range(D):
            self._check_series(d)

    def _check_series(self, d: int) -> None:
        """Hold series ``d`` to the rules of univariate data.

        Each series, with the rows' counts and its own truncation window,
        must be valid data for a univariate fit (``xcnt_handler``): no
        NaN, counts that are positive whole numbers, ``xl <= xr`` and
        ``xl < xr`` where interval censored, and every value inside its
        truncation window. A margin fitted here checked its series on
        the way, but a margin passed already fitted did not, and a NaN
        returned the copula parameter's starting value with a NaN
        likelihood, a negative count was a negative weight and an
        interval with ``xl > xr`` a negative probability (clipped).
        """
        from surpyval.utils import xcnt_handler

        xd, cd, xld, xrd, tld, trd = self.dimension(d)
        if (cd == 2).any():
            # An interval row carries [xl, xr], a point row [x, x], as the
            # margin fit is given them.
            xd = np.column_stack(
                [np.where(cd == 2, xld, xd), np.where(cd == 2, xrd, xd)]
            )
        try:
            xcnt_handler(
                x=xd,
                c=cd,
                n=self.n,
                t=np.column_stack([tld, trd]),
                group_and_sort=False,
            )
        except ValueError as error:
            raise ValueError(f"Series {d}: {error}") from error

    @staticmethod
    def _as_2d(x: npt.ArrayLike) -> npt.NDArray:
        # A list/tuple is read as a sequence of per-dimension (column)
        # vectors; an ndarray is taken as already row-by-dimension.
        if isinstance(x, (list, tuple)):
            series = [np.asarray(xi, dtype=float) for xi in x]
            lengths = [len(np.atleast_1d(xi)) for xi in series]
            if len(set(lengths)) > 1:
                # numpy's "all the input array dimensions ... must match"
                raise ValueError(
                    f"The series have different lengths {lengths}: the "
                    "values are paired by position (row i of each series "
                    "is unit i), so each series needs one value per unit."
                )
            x = np.column_stack(series)
        else:
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
        return x

    def dimension(self, d: int) -> tuple:
        """Return ``(x, c, xl, xr, tl, tr)`` arrays for series ``d``."""
        return (
            self.x[:, d],
            self.c[:, d],
            self.xl[:, d],
            self.xr[:, d],
            self.t[:, d, 0],
            self.t[:, d, 1],
        )
