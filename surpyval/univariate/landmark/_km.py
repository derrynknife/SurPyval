"""The landmark Kaplan-Meier estimate: survival from a landmark by the
covariates' values known then."""

from __future__ import annotations

from typing import Any

import numpy as np


def landmark_kaplan_meier(
    life: Any,
    at: float,
    horizon: float,
    by: str | list[str] | None = None,
) -> dict:
    """
    The Kaplan-Meier estimate of survival from the landmark ``at``, for
    each group of units by the value(s) of their covariates ``by`` as
    known at ``at`` (the simple landmark analysis of van Houwelingen and
    Putter, 2011).

    Each estimate is of :math:`P(T > x \\mid T > L, \\text{group at } L)`
    for ``x`` from ``L`` to ``L + horizon``: the units at risk at ``L``,
    entered at ``L``, grouped by what was known about them at ``L``. Its
    bounds are the Kaplan-Meier estimate's own (``cb``), as each unit is
    counted once.

    Parameters
    ----------
    life : SurpyvalData
        The units' lifetimes and covariates, as :func:`~surpyval.landmark`
        takes them.
    at : float
        The landmark time ``L``.
    horizon : float
        The window ``w``: follow-up is cut at ``L + w``.
    by : str or list of str, optional
        The covariates to group by (names, as the landmark data's
        ``covariate_names``). Default: one estimate of every unit at risk.

    Returns
    -------
    dict
        ``{group: NonParametric}``: the group is the covariate's value
        (a tuple of values for several), or ``None`` without ``by``.

    Examples
    --------
    >>> from surpyval import SurpyvalData, landmark_kaplan_meier
    >>> life = SurpyvalData(x=[1, 3, 4, 5, 6, 8], c=[0, 0, 1, 0, 0, 1],
    ...                     Z=[[0], [1], [0], [1], [0], [1]],
    ...                     covariate_names=["z"])
    >>> km = landmark_kaplan_meier(life, at=2, horizon=4, by="z")
    >>> sorted(km)
    [0.0, 1.0]
    >>> km[1.0].sf([3, 5, 6]).round(3)
    array([0.667, 0.333, 0.333])
    """
    from surpyval.univariate.nonparametric import KaplanMeier

    from ._data import landmark

    L = np.atleast_1d(np.asarray(at, dtype=float))
    if L.size != 1:
        raise ValueError(
            "'at' must be one landmark time for landmark_kaplan_meier; "
            "got {}. Call it once per landmark.".format(L.tolist())
        )
    data = landmark(life, at=float(L[0]), horizon=horizon)
    groups: dict[Any, np.ndarray]
    if by is None:
        groups = {None: np.ones(len(data.x), dtype=bool)}
    else:
        names = [by] if isinstance(by, str) else list(by)
        have = list(data.covariate_names or [])
        missing = [v for v in names if v not in have]
        if missing:
            raise ValueError(
                "'by' names {} not among the covariates {}.".format(
                    missing, have
                )
            )
        Z = np.asarray(data.Z, dtype=float)[:, [have.index(v) for v in names]]
        groups = {}
        for row in np.unique(Z, axis=0):
            key = float(row[0]) if row.size == 1 else tuple(map(float, row))
            groups[key] = (Z == row).all(axis=1)
    return {
        key: KaplanMeier.fit(
            x=data.x[mask], c=data.c[mask], n=data.n[mask], tl=data.tl[mask]
        )
        for key, mask in groups.items()
    }
