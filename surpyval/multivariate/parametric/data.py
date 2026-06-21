"""Aligned multivariate survival data for copula models.

A joint observation is a *row* across ``D`` correlated series. Every series
of a row carries its own censoring code (using the same convention as the
univariate :class:`~surpyval.utils.surpyval_data.SurpyvalData`)::

    c ==  0  observed (exact)
    c ==  1  right censored   (the true value is > x)
    c == -1  left censored    (the true value is < x)
    c ==  2  interval censored (the true value is in [xl, xr])

so that every censoring/truncation type the univariate library supports is
available per-dimension in the joint likelihood. Weights ``n`` and the
truncation window ``t`` apply to the whole row.
"""

import numpy as np


class MultivariateSurpyvalData:
    """Normalise and hold row-aligned multivariate survival data.

    Parameters
    ----------
    x : array-like, shape (N, D) or sequence of D length-N arrays
        Point values per dimension. For an interval-censored entry
        (``c == 2``) the point value is ignored and ``xl``/``xr`` are used.
    c : array-like, shape (N, D), optional
        Per-dimension censoring codes in ``{0, 1, -1, 2}``. Defaults to all
        observed.
    n : array-like, shape (N,), optional
        Integer weight (count) of each row. Defaults to all ones.
    t : array-like, shape (N, D, 2), optional
        Per-dimension truncation window ``[tl, tr]``. Defaults to
        ``(-inf, inf)`` (no truncation).
    xl, xr : array-like, shape (N, D), optional
        Interval-censoring bounds, required where ``c == 2``.
    """

    def __init__(self, x, c=None, n=None, t=None, xl=None, xr=None):
        x = self._as_2d(x)
        N, D = x.shape

        if c is None:
            c = np.zeros((N, D), dtype=int)
        else:
            c = self._as_2d(c).astype(int)
            if c.shape == (D,):
                c = np.broadcast_to(c, (N, D)).copy()
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
        xl = x.copy() if xl is None else self._as_2d(xl)
        xr = x.copy() if xr is None else self._as_2d(xr)
        if (c == 2).any() and (xl is None or xr is None):
            raise ValueError("interval-censored rows (c == 2) need xl and xr")

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

    @staticmethod
    def _as_2d(x):
        # A list/tuple is read as a sequence of per-dimension (column)
        # vectors; an ndarray is taken as already row-by-dimension.
        if isinstance(x, (list, tuple)):
            x = np.column_stack([np.asarray(xi, dtype=float) for xi in x])
        else:
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
        return x

    def dimension(self, d):
        """Return ``(x, c, xl, xr, tl, tr)`` arrays for series ``d``."""
        return (
            self.x[:, d],
            self.c[:, d],
            self.xl[:, d],
            self.xr[:, d],
            self.t[:, d, 0],
            self.t[:, d, 1],
        )
