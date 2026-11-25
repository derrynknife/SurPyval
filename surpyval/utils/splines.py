"""
Natural cubic splines: one basis for every model that uses one.

A natural (or restricted) cubic spline is a curve made of cubic pieces
joined at knots so that its value, slope and curvature are continuous,
and that is a straight line before the first knot and after the last.
With ``K + 2`` knots (two boundary knots and ``K`` between them) every
such curve is ``a + b x`` plus a weighted sum of ``K`` basis functions;
this module gives those functions, and their derivatives and
antiderivative, in the two forms SurPyval's models use:

``"durrleman-simon"``
    Durrleman and Simon's (1989) form, as Royston and Parmar (2002) and
    Stata's ``stpm2`` write it: for each interior knot :math:`k_j`, with
    :math:`\\lambda_j = (k_{max} - k_j) / (k_{max} - k_{min})`,

    .. math::

        v_j(x) = (x - k_j)_+^3 - \\lambda_j (x - k_{min})_+^3
                 - (1 - \\lambda_j) (x - k_{max})_+^3 .

    Every :math:`v_j` starts at the first boundary knot. The
    Royston-Parmar models (``RoystonParmar``,
    ``RoystonParmarRegression``) use it, on log time.

``"esl"``
    The truncated-power form of Hastie, Tibshirani and Friedman's
    *Elements of Statistical Learning* (2009, eq. 5.4 and 5.5), on the
    scaled age :math:`u = (x - k_{min}) / (k_{max} - k_{min})` with the
    knots :math:`\\xi_1 = 0 < \\dots < \\xi_{K+2} = 1` on that scale:

    .. math::

        d_k(u) = \\frac{(u - \\xi_k)_+^3 - (u - 1)_+^3}{1 - \\xi_k},
        \\qquad N_k = d_k - d_{K+1},\\quad k = 1, \\dots, K .

    :math:`N_k` is 0 up to its own knot :math:`\\xi_k` (the first
    boundary knot for :math:`N_1`), so its coefficient bends the curve
    only after that knot. ``SplinePath`` uses it.

The two forms span the same curves (tests check both against SciPy's
``CubicSpline(bc_type="natural")`` and R's ``splines::ns``); they differ
only in what each coefficient means.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: The forms :func:`natural_cubic_basis` takes.
FORMS = ("durrleman-simon", "esl")

#: For each derivative order, the power of the truncated term and the
#: factor in front of it: d/dx (x - k)_+^3 = 3 (x - k)_+^2, and so on;
#: -1 is the antiderivative (x - k)_+^4 / 4, which is 0 up to the knot.
_ORDERS = {-1: (4, 0.25), 0: (3, 1.0), 1: (2, 3.0), 2: (1, 6.0)}


def natural_cubic_basis(
    x: npt.ArrayLike,
    knots: npt.ArrayLike,
    deriv: int = 0,
    form: str = "durrleman-simon",
) -> npt.NDArray:
    """
    The natural cubic spline basis at ``x``, or its derivative.

    The columns are ``[1, x, b_1(x), ..., b_K(x)]``, one ``b`` for each
    interior knot, so any natural cubic spline with these knots is
    ``natural_cubic_basis(x, knots) @ coefficients``.

    Parameters
    ----------
    x : array_like
        Where to evaluate it (1-D).
    knots : array_like
        Every knot, ascending, the two boundary knots first and last.
    deriv : int, optional
        0 (default) the basis itself; 1 and 2 its first and second
        derivatives in ``x``; -1 its antiderivative, which is ``x`` and
        ``x**2 / 2`` for the first two columns and 0 up to the first
        knot for the others (so the area from ``a`` to ``b`` is the
        difference of the rows at ``b`` and ``a``).
    form : str, optional
        ``"durrleman-simon"`` (default; Royston-Parmar's) or ``"esl"``
        (``SplinePath``'s): see the module's notes.

    Returns
    -------
    numpy array
        ``(len(x), len(knots))``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.utils.splines import natural_cubic_basis
    >>> natural_cubic_basis([0.0, 1.0, 3.0], [0.0, 1.0, 2.0])
    array([[ 1. ,  0. ,  0. ],
           [ 1. ,  1. , -0.5],
           [ 1. ,  3. , -6. ]])
    >>> natural_cubic_basis([0.0, 5.0, 10.0], [0.0, 4.0, 10.0],
    ...                     form="esl").round(4)
    array([[ 1.    ,  0.    ,  0.    ],
           [ 1.    ,  5.    ,  0.1233],
           [ 1.    , 10.    ,  0.64  ]])
    """
    x = np.asarray(x, dtype=float)
    bends = natural_cubic_bends(x, knots, deriv, form)
    return np.column_stack(_line(x, deriv) + bends)


def natural_cubic_bends(
    x: npt.ArrayLike,
    knots: npt.ArrayLike,
    deriv: int = 0,
    form: str = "durrleman-simon",
) -> list:
    """
    The basis functions past the straight line, one for each interior
    knot (the columns of :func:`natural_cubic_basis` after ``1`` and
    ``x``), each an array of ``x``'s shape, whatever that shape is.

    Examples
    --------
    >>> from surpyval.utils.splines import natural_cubic_bends
    >>> [b.round(4) for b in natural_cubic_bends(
    ...     [[5.0, 10.0]], [0.0, 4.0, 10.0], form="esl")]
    [array([[0.1233, 0.64  ]])]
    """
    if deriv not in _ORDERS:
        raise ValueError(
            "deriv is the order of the derivative: -1 (the "
            f"antiderivative), 0, 1 or 2; got {deriv!r}."
        )
    if form not in FORMS:
        raise ValueError(
            f"form is one of {FORMS}: 'durrleman-simon' (Royston-"
            f"Parmar's) or 'esl' (SplinePath's); got {form!r}."
        )
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    if form == "esl":
        return _esl(x, k, deriv)
    return _durrleman_simon(x, k, deriv)


def _line(x: npt.NDArray, deriv: int) -> list:
    """The columns ``1`` and ``x`` (or their derivatives)."""
    if deriv == -1:
        return [x, x**2 / 2.0]
    if deriv == 0:
        return [np.ones_like(x), x]
    if deriv == 1:
        return [np.zeros_like(x), np.ones_like(x)]
    return [np.zeros_like(x), np.zeros_like(x)]


def _durrleman_simon(x: npt.NDArray, knots: npt.NDArray, deriv: int) -> list:
    # The operations, and their order, are those the Royston-Parmar
    # models have always used (``coef`` 1 for the basis, 3 for its
    # derivative), so their numbers are unchanged to the last bit.
    power, coef = _ORDERS[deriv]
    kmin, kmax = knots[0], knots[-1]
    span = kmax - kmin
    cols = []
    for kj in knots[1:-1]:
        lam = (kmax - kj) / span
        cols.append(
            coef * np.maximum(x - kj, 0.0) ** power
            - coef * lam * np.maximum(x - kmin, 0.0) ** power
            - coef * (1.0 - lam) * np.maximum(x - kmax, 0.0) ** power
        )
    return cols


def _esl(x: npt.NDArray, knots: npt.NDArray, deriv: int) -> list:
    power, coef = _ORDERS[deriv]
    lo, hi = knots[0], knots[-1]
    span = hi - lo
    u = (x - lo) / span
    # d/dx = (1 / span) d/du, so the derivative of order r carries
    # span**-r (and the antiderivative span)
    factor = coef * span ** (-float(deriv))
    starts = np.concatenate([[0.0], (knots[1:-1] - lo) / span])
    last = np.maximum(u - 1.0, 0.0) ** power

    def d(start: float) -> npt.NDArray:
        return (np.maximum(u - start, 0.0) ** power - last) / (1.0 - start)

    final = d(starts[-1])
    return [factor * (d(start) - final) for start in starts[:-1]]
