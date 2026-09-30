"""The one-line summary of a fit's data that the models print (#508).

A censoring flag read backwards (a spreadsheet's "1 = failed" column passed
as ``c``) fits without complaint and gives a plausible-looking model, so a
fitted model's printout says what data it was fitted to: how many units,
and how many of them failed or were censored or truncated. A user who
knows they had 51 failures and sees "9 failures, 51 right censored" sees
the inversion at once.

The counts are of units, weighted by ``n``: a row with ``n = 5`` is five
units. Rows are not counted because the fitters group tied rows, so the
number of rows a model holds is not the number the user passed.
"""

import numpy as np
import numpy.typing as npt

# The censoring flag's values, in the order they are listed.
_CENSORING = (
    (0, "failure", "failures"),
    (1, "right censored", "right censored"),
    (-1, "left censored", "left censored"),
    (2, "interval censored", "interval censored"),
)


def _count(k: int, singular: str, plural: str) -> str:
    return f"{k} {singular if k == 1 else plural}"


def data_summary(
    c: npt.ArrayLike,
    n: "npt.ArrayLike | None" = None,
    tl: "npt.ArrayLike | None" = None,
    tr: "npt.ArrayLike | None" = None,
    lower: float = -np.inf,
    upper: float = np.inf,
) -> str:
    """
    Summarise the censoring and truncation of a fit's data in one line.

    Parameters
    ----------
    c : array_like
        The censoring flag of each row (0 observed, 1 right, -1 left,
        2 interval).
    n : array_like, optional
        The count of each row. Default 1 for every row.
    tl, tr : array_like, optional
        The left and right truncation of each row. A unit is counted as
        left (right) truncated where its ``tl`` is above ``lower`` (its
        ``tr`` below ``upper``): a truncation at or outside the support
        truncates nothing.
    lower, upper : float, optional
        The support of the model the data were fitted to. Default the
        real line.

    Returns
    -------
    str
        For example ``"60 units: 9 failures, 51 right censored"``. The
        failures are always listed; the other kinds only when there are
        some. Truncation follows after a semicolon.

    Examples
    --------
    >>> from surpyval.utils.data_summary import data_summary
    >>> data_summary([0, 1, 1], n=[9, 50, 1])
    '60 units: 9 failures, 51 right censored'
    >>> data_summary([0, 2, 0], tl=[0, 0, 1], lower=0)
    '3 units: 2 failures, 1 interval censored; 1 left truncated'
    """
    c = np.asarray(c).ravel()
    n = np.ones_like(c) if n is None else np.asarray(n).ravel()
    total = int(np.sum(n))
    parts = []
    for flag, singular, plural in _CENSORING:
        k = int(np.sum(n[c == flag]))
        if k > 0 or flag == 0:
            parts.append(_count(k, singular, plural))
    out = f"{_count(total, 'unit', 'units')}: " + ", ".join(parts)
    truncated = []
    if tl is not None:
        k = int(np.sum(n[np.asarray(tl, dtype=float).ravel() > lower]))
        if k > 0:
            truncated.append(f"{k} left truncated")
    if tr is not None:
        k = int(np.sum(n[np.asarray(tr, dtype=float).ravel() < upper]))
        if k > 0:
            truncated.append(f"{k} right truncated")
    if truncated:
        out += "; " + ", ".join(truncated)
    return out
