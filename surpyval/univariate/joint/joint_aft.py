"""The accelerated joint model fitter: :func:`JointAFT` and
``WeibullJointAFT``.

The marker's path (and the covariates) set how fast the item ages, in
place of multiplying its hazard as :func:`~.joint_ph.JointPH`'s do: an
age at a high reading uses up more than that age of life, and the life
used is not given back when the reading falls. The data, the path, the
parameters, the fit and the predictions are those of the proportional
hazards model; the likelihood's lifetime terms are :mod:`._aft`'s."""

from __future__ import annotations

from typing import Any

from .joint_ph import JointPHFitter


class JointAFTFitter(JointPHFitter):
    """
    The joint model of a marker -- a measurement read repeatedly on each
    item -- and the item's lifetime, the marker following a degradation
    path model and the survival part accelerated: the path speeds up the
    item's ageing. Built by :func:`JointAFT`, ready-made as
    ``WeibullJointAFT`` (a straight line). Its ``fit`` and
    ``fit_from_df`` are those of the proportional hazards fitter
    (:class:`~surpyval.univariate.joint.joint_ph.JointPHFitter`), and
    return a :class:`~surpyval.univariate.joint.joint_model.JointModel`
    whose ``survival`` is ``"AFT"``.
    """

    survival = "AFT"

    def __init__(
        self,
        distribution: Any,
        path_model: Any,
        varying: Any = None,
        link: Any = "value",
    ) -> None:
        try:
            named = [link] if isinstance(link, str) else list(link)
        except TypeError:
            named = []
        if "cumulative" in named:
            raise ValueError(
                "An accelerated joint model already accumulates the marker: "
                "the item's age used up is the integral of its ageing rate "
                "over its life, so the rate cannot also follow the area "
                "under the marker. Use link='value' or 'slope' (or both), "
                "or JointPH(..., link='cumulative') for a hazard that "
                "follows the area."
            )
        super().__init__(distribution, path_model, varying, link)

    def _repr_details(self) -> list[str]:
        out = super()._repr_details()
        return out[:1] + ["accelerated"] + out[1:]


def JointAFT(
    distribution: Any,
    path_model: Any = None,
    varying: Any = None,
    link: Any = "value",
) -> JointAFTFitter:
    """
    An accelerated joint model fitter of a marker and a lifetime: the
    marker follows the degradation ``path_model``
    (``surpyval.degradation.LinearPath`` by default), and its path speeds
    up the item's ageing on the baseline ``distribution`` (the Weibull
    for now). By age ``t`` the item has used up the effective age

        A(t) = int_0^t exp(gamma'z + zeta (m(s) - r)) ds,

    so a stretch at a high reading ages it faster, and the age it used
    is not given back when the reading falls; its survival is the
    baseline's at that age, ``S0(A(t))``, and its hazard ``h0(A(t))``
    times the rate it is ageing at. This is the cumulative-exposure
    model of step-stress tests, with the marker as the stress. With a
    marker that never changes and a Weibull baseline it is
    :func:`~surpyval.univariate.joint.joint_ph.JointPH` with every effect
    divided by ``beta``; once the marker moves the two differ.
    ``WeibullJointAFT`` is ``JointAFT(Weibull, LinearPath)``.

    The parameters have the proportional hazards model's names:
    ``marker_effect`` is the log of how many times faster the item ages
    for each unit of the marker (``exp(marker_effect)`` times as fast),
    and a covariate's coefficient the same for one unit of it.

    Parameters
    ----------
    distribution : ParametricFitter
        ``surpyval.Weibull``.
    path_model : PathModel, optional
        The marker's path. Default ``LinearPath``.
    varying : sequence of str, optional
        The path parameters that vary between items (their random
        effects, multivariate normal); the others are common to all
        items. Default all of them (at most three).
    link : str or sequence of str, optional
        What of the marker's path sets the ageing rate: ``"value"`` (the
        default: its current value) or ``"slope"`` (its rate of change),
        or both, each with an effect of its own (``marker_effect``,
        ``slope_effect``). ``"cumulative"`` is refused: the effective
        age already accumulates the marker.

    Returns
    -------
    JointAFTFitter
        A fitter with a ``fit(data)`` method.

    Examples
    --------
    >>> from surpyval import JointAFT, Weibull
    >>> JointAFT(Weibull)  # doctest: +NORMALIZE_WHITESPACE
    WeibullJointAFT: joint model fitter (Weibull baseline, accelerated,
    Linear path, a + b t)
    >>> JointAFT(Weibull, link=["value", "slope"])
    ... # doctest: +ELLIPSIS
    WeibullJointAFT: joint model fitter (..., link: value, slope)
    """
    if path_model is None:
        from surpyval.degradation.path_models import LinearPath

        path_model = LinearPath
    return JointAFTFitter(distribution, path_model, varying, link)
