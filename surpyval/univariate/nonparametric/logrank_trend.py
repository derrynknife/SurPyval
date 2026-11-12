"""The log-rank test for trend: k ordered groups, one degree of freedom."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .logrank import _logrank_parts, _plain, _weighting_label

#: The ``alternative`` values of :func:`trend_logrank`.
ALTERNATIVES = ("two-sided", "increasing", "decreasing")


def _finite(v: Any) -> Any:
    v = float(v)
    return v if np.isfinite(v) else None


class TrendLogRankResult:
    """
    Result of the log-rank test for trend, from :func:`trend_logrank`.

    Attributes
    ----------
    statistic : float
        The chi-squared statistic on one degree of freedom, ``z**2``.
    z : float
        The standardised trend: the score-weighted sum of the groups'
        observed-minus-expected events over its standard deviation.
        Positive when the hazard rises with the score (groups with higher
        scores have more events than expected).
    dof : int
        1, or 0 when there is nothing to compare.
    p_value : float
        The p-value for ``alternative``.
    alternative : str
        "two-sided", "increasing" or "decreasing".
    scores : dict
        The score of each group, keyed by group label.
    weighting : str
        The weighting of the event times, as for :func:`~surpyval.logrank`.
    strata : int or None
        The number of strata of a stratified test; None for an
        unstratified one.
    weighted : bool
        Whether the test was of data with sampling weights ``w`` (the
        design-based variance).

    Examples
    --------
    >>> from surpyval import trend_logrank
    >>> x = [3, 5, 7, 8, 2, 3, 4, 4, 1, 2, 2, 3]
    >>> Z = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    >>> res = trend_logrank(x, Z)
    >>> res.to_dict()["scores"][2]
    {'group': 2, 'score': 3.0}
    """

    def __init__(
        self,
        statistic: float,
        z: float,
        dof: int,
        p_value: float,
        alternative: str,
        scores: dict,
        weighting: str,
        strata: int | None = None,
        weighted: bool = False,
    ) -> None:
        self.statistic = statistic
        self.z = z
        self.dof = dof
        self.p_value = p_value
        self.alternative = alternative
        self.scores = scores
        self.weighting = weighting
        self.strata = strata
        self.weighted = weighted

    def __repr__(self) -> str:
        header = "Log-Rank Test for Trend"
        if self.strata:
            header = "Stratified " + header
        out = (
            header
            + "\n"
            + "=" * len(header)
            + f"\nWeighting        : {self.weighting}"
            + f"\nAlternative      : {self.alternative}"
            + f"\nScores           : {self.scores}"
        )
        if self.strata is not None:
            out += f"\nStrata           : {self.strata}"
        if self.weighted:
            out += "\nVariance         : robust (sampling weights w)"
        out += (
            f"\nz                : {self.z:.6g}"
            + f"\nStatistic        : {self.statistic:.6g}"
            + f"\nDoF              : {self.dof}"
            + f"\np-value          : {self.p_value:.6g}"
        )
        return out

    def to_dict(self) -> dict:
        """The result as strict JSON: the settings, each group's score,
        and the statistic, ``z``, degrees of freedom and p-value (a
        non-finite number is ``None``)."""
        return {
            "test": "log-rank trend",
            "weighting": self.weighting,
            "alternative": self.alternative,
            "strata": self.strata,
            "weighted": bool(self.weighted),
            "scores": [
                {"group": _plain(g), "score": float(s)}
                for g, s in self.scores.items()
            ],
            "z": _finite(self.z),
            "statistic": _finite(self.statistic),
            "dof": int(self.dof),
            "p_value": _finite(self.p_value),
        }


def _scores_of(scores: Any, groups: npt.NDArray) -> npt.NDArray:
    """The score of each group, in the sorted order ``groups``."""
    k = groups.size
    if scores is None:
        return np.arange(1.0, k + 1.0)
    if isinstance(scores, dict):
        labels = [_plain(g) for g in groups]
        missing = [g for g in labels if g not in scores]
        if missing:
            raise ValueError(
                "'scores' has no score for the group(s) {}; give one score "
                "per group label".format(missing)
            )
        extra = [g for g in scores if g not in labels]
        if extra:
            raise ValueError(
                f"'scores' has scores for labels not in 'Z': {extra}"
            )
        out = np.array([float(scores[g]) for g in labels])
    else:
        out = np.asarray(scores, dtype=float).ravel()
        if out.size != k:
            raise ValueError(
                "'scores' must have one score per group ({} groups, in the "
                "sorted order of their labels {}), or be a dict "
                "{{label: score}}; got {}".format(
                    k, [_plain(g) for g in groups], out.size
                )
            )
    if not np.all(np.isfinite(out)):
        raise ValueError("'scores' must be finite numbers")
    if np.ptp(out) == 0:
        raise ValueError(
            "'scores' are all equal, so they order nothing; give the groups "
            "different scores"
        )
    return out


def trend_logrank(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    scores: Any = None,
    weighting: str = "log-rank",
    rho: float = 0,
    gamma: float = 0,
    alternative: str = "two-sided",
    strata: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | float | None = None,
    t0: float | None = None,
    w: npt.ArrayLike | float | None = None,
) -> TrendLogRankResult:
    r"""
    The log-rank test for trend: whether survival worsens (or improves)
    steadily across k ordered groups.

    When the groups have a natural order -- doses, stages of a disease,
    load levels, grades of a material -- the question is often not "do
    any of them differ?" (the k-sample :func:`~surpyval.logrank` test, on
    k - 1 degrees of freedom) but "does the hazard rise with the
    order?". This test puts all its power on that one direction, so it
    finds a steady trend with fewer data than the k-sample test, at the
    cost of missing a difference that does not follow the order (a
    middle group that is worst, say). Each group is given a score (1, 2,
    ..., k by default); the test asks whether the groups' observed minus
    expected events line up with the scores.

    Parameters
    ----------
    x, Z, c, n
        As for :func:`~surpyval.logrank`: the times, the group labels,
        the censoring flags (0 observed, 1 right censored) and the counts.
    scores : array like or dict, optional
        The score of each group: a sequence with one number per group, in
        the sorted order of the group labels, or a dict ``{label:
        score}``. Defaults to 1, 2, ..., k in the sorted order of the
        labels (equally spaced). To use numeric labels as their own scores
        (doses, say), pass ``scores=sorted(set(Z))``. Only the spacing
        matters: adding a constant or multiplying by a positive one gives
        the same test.
    weighting, rho, gamma
        The weighting of the event times, as for
        :func:`~surpyval.logrank` (the default is the log-rank weight 1).
    alternative : str, optional
        "two-sided" (the default): a trend either way, the chi-squared
        statistic on one degree of freedom. "increasing": the hazard rises
        with the score (``z`` large). "decreasing": it falls (``z``
        small).
    strata, tl, t0, w
        As for :func:`~surpyval.logrank`: the strata of a stratified test
        (the observed-minus-expected vector and its covariance summed over
        the strata), the entry times, the end of the comparison window and
        sampling weights (the design-based variance).

    Returns
    -------
    TrendLogRankResult
        With the ``statistic`` (``z**2``), the signed ``z``, ``dof``, the
        ``p_value`` and the ``scores`` used.

    Raises
    ------
    ValueError
        If the scores are not one finite number per group, all equal, or
        name labels not in ``Z``; ``alternative`` is unknown; or for any
        reason :func:`~surpyval.logrank` would raise.

    Notes
    -----
    With :math:`U` the vector of the groups' (weighted) observed minus
    expected events of :func:`~surpyval.logrank`, :math:`V` its covariance
    and :math:`s` the scores,

    .. math::

        z = \frac{s^\top U}{\sqrt{s^\top V s}}, \qquad \chi^2_1 = z^2

    (Tarone 1975). With the log-rank weight and no ties it is the score
    test of a Cox model with the score as a numeric covariate (with ties,
    the Breslow score test). It is what R's
    ``survminer::surv_pvalue(..., test.for.trend = TRUE)`` reports and
    what is computed by hand from ``survdiff``'s ``obs - exp`` and
    ``var``.

    Examples
    --------
    Patients with advanced lung cancer by their ECOG performance score (0
    best, 2 worst; the one patient scored 3 is left out):

    >>> from surpyval import logrank, trend_logrank
    >>> from surpyval.datasets import load_lung
    >>> df = load_lung().dropna(subset=["ph.ecog"])
    >>> df = df[df["ph.ecog"] < 3]
    >>> x, c = df["time"], 1 - df["status"]
    >>> Z = df["ph.ecog"].astype(int)
    >>> res = trend_logrank(x, Z, c=c, scores=[0, 1, 2])
    >>> print(round(res.z, 3), round(res.p_value, 6))
    4.055 5e-05

    The trend test uses one degree of freedom where the k-sample test
    uses two, so its p-value is smaller here, where the worse score has
    the higher hazard at each step:

    >>> round(logrank(x, Z, c=c).p_value, 6)
    0.000123

    References
    ----------
    Tarone, R. E. (1975), "Tests for trend in life table analysis",
    Biometrika 62, 679-682.

    Klein, J. P. and Moeschberger, M. L. (2003), "Survival Analysis:
    Techniques for Censored and Truncated Data", 2nd ed., section 7.4.
    """
    from scipy.stats import chi2, norm

    if alternative not in ALTERNATIVES:
        raise ValueError(
            "'alternative' must be one of {}; got {!r}".format(
                list(ALTERNATIVES), alternative
            )
        )
    U, V, E, groups, n_strata, weighted = _logrank_parts(
        x, Z, c, n, weighting, rho, gamma, strata, tl, t0, True, w, 4
    )
    s = _scores_of(scores, groups)
    score_dict = {_plain(g): float(v) for g, v in zip(groups, s)}
    label = _weighting_label(weighting, rho, gamma)
    numerator = float(s @ U)
    variance = float(s @ V @ s)
    if (E > 0).sum() < 2 or not variance > 1e-12 * max(1.0, float(s @ s)):
        # Nothing to compare: no event, or every risk set holds one group.
        return TrendLogRankResult(
            0.0,
            0.0,
            0,
            1.0,
            alternative,
            score_dict,
            label,
            strata=n_strata,
            weighted=weighted,
        )
    z = numerator / np.sqrt(variance)
    if alternative == "two-sided":
        p_value = float(chi2.sf(z * z, 1))
    elif alternative == "increasing":
        p_value = float(norm.sf(z))
    else:
        p_value = float(norm.cdf(z))
    return TrendLogRankResult(
        float(z * z),
        float(z),
        1,
        p_value,
        alternative,
        score_dict,
        label,
        strata=n_strata,
        weighted=weighted,
    )
