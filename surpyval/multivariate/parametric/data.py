"""Aligned multivariate survival data: one row, two or more values.

A joint observation is a *row* across ``D >= 2`` axes (series, ...).
The data says nothing about what the axes mean; the model that is fitted
to it does: a copula reads them as two correlated series. (One life on a
main clock with a subordinate second clock is univariate data: the
two-clock fits, ``Equivalence``, take ``x`` and ``x2`` /
``SecondaryClock``.)
Data with one value per row is a
:class:`~surpyval.utils.surpyval_data.SurpyvalData`. Every series
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

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def check_axes(model: str, data: Any, n_axes: int, meaning: str) -> None:
    """Refuse ``data`` unless it has exactly ``n_axes`` axes.

    Each fitter that takes a :class:`MultivariateSurpyvalData` says how
    many axes it takes (its ``n_axes``) and what they mean (``meaning``,
    ``model`` the subject of the message, e.g. ``"A Clayton copula"``;
    e.g. ``"two series"``), and refuses data with any other number with a
    ``ValueError`` saying so.
    """
    if data.D != n_axes:
        names = "" if data.names is None else f" ({', '.join(data.names)})"
        raise ValueError(
            f"{model} takes exactly {n_axes} axes ({meaning}); the data "
            f"has {data.D}{names}. Build the data from the {n_axes} "
            "columns this model is for."
        )


class MultivariateSurpyvalData:
    """Normalise and hold row-aligned multivariate survival data.

    Each of ``x``, ``xl``, ``xr`` and ``c`` is ``N`` rows by ``D`` axes,
    ``t`` is ``N`` by ``D`` by 2 and ``n`` is one count per row, for any
    ``D`` of 2 or more. What the axes mean is the model's business: a
    copula takes exactly two correlated series.

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
    names : sequence of D str, optional
        A name for each axis, such as ``("hours", "landings")``. Used
        only in messages and in the results of a model fitted to the
        data; nothing is computed from them.

    Raises
    ------
    ValueError
        If there are fewer than two axes (use ``SurpyvalData``), an array
        has the wrong shape, a censoring code is not one of the four,
        ``names`` is not ``D`` distinct names, or a series (with the
        counts and its own truncation window) is not valid univariate
        data: a NaN value, a count that is not a positive whole number,
        an interval with ``xl >= xr``, or a value outside its truncation
        window. The message names the axis (``"Series 0: ..."``, or
        ``"Series 0 (hours): ..."`` with names).

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

    Any number of axes of 2 or more, optionally named:

    >>> three = MultivariateSurpyvalData(
    ...     [[100.0, 250.0], [3.0, 9.0], [40.0, 80.0]],
    ...     c=[[0, 1], [0, 1], [0, 1]],
    ...     names=("hours", "landings", "cycles"),
    ... )
    >>> three.D, three.names
    (3, ('hours', 'landings', 'cycles'))
    >>> three.dimension("landings")[0]
    array([3., 9.])
    """

    def __init__(
        self,
        x: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        xl: npt.ArrayLike | None = None,
        xr: npt.ArrayLike | None = None,
        names: Any = None,
    ) -> None:
        x = self._as_2d(x)
        N, D = x.shape
        if D < 2:
            raise ValueError(
                f"MultivariateSurpyvalData holds two or more values per row "
                f"(axes); got {D}. Data with one value per row is a "
                "SurpyvalData (surpyval.SurpyvalData(x, c, n, t)), which "
                "every univariate fit takes."
            )
        self.names = self._check_names(names, D)

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
            bad = ~np.isin(c, (0, 1, -1, 2))
            if bad.any():
                d = int(np.flatnonzero(bad.any(axis=0))[0])
                raise ValueError(
                    f"c values must be in {{0, 1, -1, 2}}; "
                    f"{self._label(d)} has "
                    f"{sorted(set(c[bad[:, d], d].tolist()))}."
                )

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
            raise ValueError(f"{self._label(d)}: {error}") from error

    @staticmethod
    def _check_names(names: Any, D: int) -> tuple[str, ...] | None:
        if names is None:
            return None
        if isinstance(names, str):
            names = [names]
        try:
            out = tuple(str(name) for name in names)
        except TypeError:
            raise ValueError(
                f"names gives one name per axis, e.g. ('hours', "
                f"'landings'); got {names!r}."
            ) from None
        if len(out) != D:
            raise ValueError(
                f"names gives one name per axis: the data has {D} axes, "
                f"and names {len(out)} ({list(out)})."
            )
        if len(set(out)) != D:
            raise ValueError(
                f"names must be distinct, one per axis; got {list(out)}."
            )
        return out

    def _label(self, d: int) -> str:
        """``"Series d"``, with the axis's name where it has one."""
        names = getattr(self, "names", None)
        if names is None:
            return f"Series {d}"
        return f"Series {d} ({names[d]})"

    def _index(self, d: Any) -> int:
        """The column of axis ``d``: its position or its name."""
        if isinstance(d, str):
            if self.names is None or d not in self.names:
                known = (
                    "the data has no axis names"
                    if self.names is None
                    else f"its axes are {list(self.names)}"
                )
                raise ValueError(f"No axis is named {d!r}: {known}.")
            return self.names.index(d)
        try:
            index = int(d)
        except (TypeError, ValueError):
            raise ValueError(
                f"An axis is given by its position (0 to {self.D - 1}) or "
                f"its name; got {d!r}."
            ) from None
        if not -self.D <= index < self.D:
            raise ValueError(
                f"The data has {self.D} axes (0 to {self.D - 1}); got "
                f"axis {index}."
            )
        return index % self.D

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

    def dimension(self, d: int | str) -> tuple:
        """Return ``(x, c, xl, xr, tl, tr)`` arrays for axis ``d``, given
        by its position (any of ``0`` to ``D - 1``) or its name."""
        d = self._index(d)
        return (
            self.x[:, d],
            self.c[:, d],
            self.xl[:, d],
            self.xr[:, d],
            self.t[:, d, 0],
            self.t[:, d, 1],
        )

    def __repr__(self) -> str:
        axes = "" if self.names is None else f", names={list(self.names)}"
        return f"MultivariateSurpyvalData(N={self.N}, D={self.D}{axes})"

    def to_dict(self) -> dict:
        """
        The data as a dictionary of plain Python values, strict JSON when
        dumped: its ``"class"``, the ``x``, ``c``, ``n``, ``t``, ``xl``
        and ``xr`` arrays as lists, and the axis ``names`` where it has
        them. The infinite truncation bounds are written as ``null`` and
        recorded under ``"non_finite"``, as for ``SurpyvalData``.
        ``surpyval.from_dict`` reads it too.

        Examples
        --------
        >>> from surpyval.multivariate import MultivariateSurpyvalData
        >>> data = MultivariateSurpyvalData(
        ...     [[1.0, 2.0], [3.0, 5.0]], names=("hours", "starts")
        ... )
        >>> again = MultivariateSurpyvalData.from_dict(data.to_dict())
        >>> again.names, again.x.tolist()
        (('hours', 'starts'), [[1.0, 3.0], [2.0, 5.0]])
        """
        from surpyval.serialisation import encode_non_finite

        data: dict[str, Any] = {
            "class": "MultivariateSurpyvalData",
            "x": self.x.tolist(),
            "c": self.c.tolist(),
            "n": self.n.tolist(),
            "t": self.t.tolist(),
            "xl": self.xl.tolist(),
            "xr": self.xr.tolist(),
        }
        if self.names is not None:
            data["names"] = list(self.names)
        return encode_non_finite(data)

    @classmethod
    def from_dict(cls, data_dict: dict) -> MultivariateSurpyvalData:
        """The data back from its :meth:`to_dict` dictionary (checked
        again, as when it was built)."""
        from surpyval.serialisation import check_schema, decode_non_finite

        if not isinstance(data_dict, dict):
            raise ValueError(
                "Expected a serialised MultivariateSurpyvalData dict, got "
                f"{type(data_dict).__name__}."
            )
        tag = data_dict.get("class")
        if tag is not None and tag != "MultivariateSurpyvalData":
            raise ValueError(
                f"This dictionary was written by {tag}, not "
                f"MultivariateSurpyvalData: read it with {tag}.from_dict, "
                "or surpyval.from_dict."
            )
        for key in ("x", "c", "n", "t"):
            if key not in data_dict:
                raise ValueError(
                    "A serialised MultivariateSurpyvalData needs "
                    f"{key!r}; this dictionary has no such entry."
                )
        check_schema(data_dict)
        data = decode_non_finite(dict(data_dict))
        x = np.array(data["x"], dtype=float)
        c = np.array(data["c"], dtype=int)
        xl = np.array(data["xl"], dtype=float) if "xl" in data else None
        xr = np.array(data["xr"], dtype=float) if "xr" in data else None
        return cls(
            x,
            c=c,
            n=np.array(data["n"]),
            t=np.array(data["t"], dtype=float),
            xl=xl,
            xr=xr,
            names=data.get("names"),
        )

    def to_json(self, filepath: str | Path | None = None) -> str | None:
        """
        The data as strict JSON text (:meth:`to_dict` dumped), or written
        to the file ``filepath`` (then None is returned).

        Examples
        --------
        >>> from surpyval.multivariate import MultivariateSurpyvalData
        >>> data = MultivariateSurpyvalData([[1.0, 2.0], [3.0, 5.0]])
        >>> MultivariateSurpyvalData.from_json(data.to_json()).D
        2
        """
        from surpyval.utils.data_json import dump

        return dump(self.to_dict(), filepath)

    @classmethod
    def from_json(cls, source: str | Path) -> MultivariateSurpyvalData:
        """The data back from JSON text, or from the file holding it
        (``surpyval.from_json`` reads it too)."""
        from surpyval.utils.data_json import load

        return cls.from_dict(load(source))
