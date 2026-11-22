"""
The scales an interval of a survival estimate is formed on.

Given a survival estimate ``R`` and the standard error of ``log R``, the
delta method gives a transform ``g(R)`` the standard error of ``log R``
times ``R g'(R)``. An interval formed on ``g`` and mapped back by its
inverse stays where ``g`` keeps it: inside ``[0, 1]`` on the log(-log),
logit and arcsine-square-root scales. The pointwise bounds (``cb``) and
the bands (``band``) of the non-parametric estimates, their test of a
difference in survival at one time (``sf_diff``), and the bound of a
non-parametric cumulative incidence all take their scales from here.

The scales are keyed by the names ``cb``'s ``bound_type`` gives them:
``"exp"`` (log(-log)), ``"logit"``, ``"arcsine"`` (arcsine square root),
``"log"`` and ``"normal"`` (the survival itself).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt


class SurvivalScale(NamedTuple):
    """A scale for the interval of a survival estimate."""

    #: The transform ``g`` of the survival ``R``.
    g: Callable[[Any], Any]
    #: ``shift(R, d)``: the change in ``g(R)`` from a change ``d`` in
    #: ``log R``, to first order, ``d R g'(R)`` (negative where ``g``
    #: falls as ``R`` rises).
    shift: Callable[[Any, Any], Any]
    #: The survival at the value ``u`` of ``g``.
    inverse: Callable[[Any], Any]


def _loglog(R: npt.NDArray) -> npt.NDArray:
    return np.log(-np.log(R))


def _loglog_shift(R: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    return d / np.log(R)


def _loglog_inverse(u: npt.NDArray) -> npt.NDArray:
    return np.exp(-np.exp(u))


def _logit(R: npt.NDArray) -> npt.NDArray:
    return np.log(R / (1.0 - R))


def _logit_shift(R: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    return d / (1.0 - R)


def _logit_inverse(u: npt.NDArray) -> npt.NDArray:
    return 1.0 / (1.0 + np.exp(-u))


def _arcsine(R: npt.NDArray) -> npt.NDArray:
    return np.arcsin(np.sqrt(R))


def _arcsine_shift(R: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    return d * 0.5 * np.sqrt(R / (1 - R))


def _arcsine_inverse(u: npt.NDArray) -> npt.NDArray:
    # The angle held within [0, pi / 2].
    return np.sin(np.clip(u, 0.0, np.pi / 2)) ** 2


def _log(R: npt.NDArray) -> npt.NDArray:
    return np.log(R)


def _exp(u: npt.NDArray) -> npt.NDArray:
    return np.exp(u)


def _log_shift(R: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    return d


def _identity(R: npt.NDArray) -> npt.NDArray:
    return R


def _plain_shift(R: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    return d * R


#: The scales, by ``cb``'s name for each.
SCALES: dict[str, SurvivalScale] = {
    "exp": SurvivalScale(_loglog, _loglog_shift, _loglog_inverse),
    "logit": SurvivalScale(_logit, _logit_shift, _logit_inverse),
    "arcsine": SurvivalScale(_arcsine, _arcsine_shift, _arcsine_inverse),
    "log": SurvivalScale(_log, _log_shift, _exp),
    "normal": SurvivalScale(_identity, _plain_shift, _identity),
}


def survival_bound(scale: str, R: Any, d: Any) -> npt.NDArray:
    """
    The survival ``R`` moved by ``d`` on the scale ``scale``
    (:data:`SCALES`): ``d`` is a normal deviate times the standard error
    of ``log R``, negative for a lower bound and positive for an upper
    one.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.univariate.nonparametric._scales import survival_bound
    >>> R = np.array([0.8, 0.8])
    >>> survival_bound("exp", R, np.array([-0.2, 0.2])).round(4)
    array([0.5788, 0.913 ])
    >>> survival_bound("arcsine", R, np.array([-0.2, 0.2])).round(4)
    array([0.6206, 0.9321])
    """
    s = SCALES[scale]
    return s.inverse(s.g(R) + s.shift(R, d))


def transformed_survival(scale: str, R: float, var_log: float) -> tuple:
    """
    The survival ``R`` on the scale ``scale`` (:data:`SCALES`) and the
    delta-method variance there, from ``var_log``, the variance of
    ``log R``.

    Examples
    --------
    >>> from surpyval.univariate.nonparametric._scales import (
    ...     transformed_survival,
    ... )
    >>> g, v = transformed_survival("logit", 0.75, 0.01)
    >>> round(float(g), 4), round(float(v), 4)
    (1.0986, 0.16)
    """
    s = SCALES[scale]
    return s.g(R), var_log * s.shift(R, 1.0) ** 2
