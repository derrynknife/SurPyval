"""Cause (and group) labels shared by every competing-risks class.

A cause label may be any hashable value: an integer, a string, a float, a
bool, a tuple, or a mixture of these. Four operations on labels were spelt
out separately in each class, and they had drifted: only the
proportional-hazards class could order mixed ``int``/``str`` labels (the
others raised a bare ``TypeError`` from ``sorted``), and a tuple label was
split into a column of the event array by ``np.asarray`` or compared
element by element with ``e == label``. They live here once.
"""

from typing import Any, Iterable

import numpy as np
import numpy.typing as npt


def ordered_labels(labels: Iterable) -> list:
    """The distinct non-missing labels in a deterministic order.

    Labels of one sortable type keep their natural order (``[1, 2, 10]``,
    ``["a", "b"]``). Labels that do not compare with each other (an ``int``
    and a ``str``) are ordered by type name, then by their text, which is
    reproducible across runs -- a ``set``'s own iteration order is not, for
    strings, since it depends on the hash seed.

    Examples
    --------
    >>> from surpyval.univariate.competing_risks.labels import ordered_labels
    >>> ordered_labels([2, None, 1, 2])
    [1, 2]
    >>> ordered_labels(["b", 1, "a", None])
    [1, 'a', 'b']
    """
    unique = {v for v in labels if v is not None}
    try:
        return sorted(unique)
    except TypeError:
        return sorted(unique, key=lambda v: (type(v).__name__, str(v)))


def label_mask(e: npt.NDArray, label: Any) -> npt.NDArray:
    """Boolean mask of the rows of the object array ``e`` equal to ``label``.

    ``e == label`` is not safe for every label: numpy broadcasts a tuple
    label against the rows (comparing its items one by one), so each row
    is compared as a whole here.
    """
    return np.fromiter(
        (v is not None and bool(v == label) for v in e),
        dtype=bool,
        count=len(e),
    )


def label_from_native(value: Any) -> Any:
    """Undo ``to_native`` for a label read back from JSON.

    ``to_native`` writes a tuple label as a list (JSON has no tuple), and a
    list is unhashable, so it could not key the reloaded model's cause map.
    A label can never be a list, so a list is always a former tuple.
    """
    if isinstance(value, list):
        return tuple(label_from_native(v) for v in value)
    return value
