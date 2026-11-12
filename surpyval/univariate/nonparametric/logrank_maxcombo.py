"""The max-combo test: the largest of several Fleming-Harrington weighted
log-rank statistics, for differences that are not proportional hazards."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .logrank import _checked_inputs, _plain, _risk_table, _time_weights

#: The ``alternative`` values of :func:`maxcombo_logrank`.
ALTERNATIVES = ("two-sided", "greater", "less")

#: The default ``(rho, gamma)`` pairs: the log-rank test and the
#: Fleming-Harrington weights for late, early and middle differences.
DEFAULT_RHO = (0.0, 0.0, 1.0, 1.0)
DEFAULT_GAMMA = (0.0, 1.0, 0.0, 1.0)


def _finite(v: Any) -> Any:
    v = float(v)
    return v if np.isfinite(v) else None


class MaxComboResult:
    """
    Result of the max-combo test, from :func:`maxcombo_logrank`.

    Attributes
    ----------
    statistic : float
        The combined statistic: the largest ``|z|`` of the components
        (two-sided), the largest ``z`` ("greater") or the smallest ``z``
        ("less").
    p_value : float
        The p-value of the combined statistic, from the joint normal
        distribution of the components with correlation ``correlation``.
        It allows for having taken the best of several tests.
    alternative : str
        "two-sided", "greater" or "less".
    group : label
        The group whose observed-minus-expected events the ``z`` are: the
        first of the two labels in sorted order. A positive ``z`` means
        this group had more events than expected (a higher hazard).
    rho, gamma : tuple of float
        The Fleming-Harrington parameters of the components.
    z : numpy array
        Each component's standardised statistic.
    p_values : numpy array
        Each component's own p-value (for ``alternative``), not adjusted.
    correlation : numpy array
        The estimated correlation matrix of the components.
    p_bonferroni : float
        The smallest component p-value times the number of components
        (at most 1): a cruder allowance for the multiple tests, never
        smaller than ``p_value``.
    strata : int or None
        The number of strata of a stratified test; None for an
        unstratified one.

    Examples
    --------
    >>> from surpyval import maxcombo_logrank
    >>> x = [3, 5, 7, 8, 10, 2, 3, 4, 4, 6]
    >>> Z = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    >>> res = maxcombo_logrank(x, Z)
    >>> res.to_dict()["components"][0]["rho"]
    0.0
    """

    def __init__(
        self,
        statistic: float,
        p_value: float,
        alternative: str,
        group: Any,
        rho: tuple,
        gamma: tuple,
        z: npt.NDArray,
        p_values: npt.NDArray,
        correlation: npt.NDArray,
        p_bonferroni: float,
        strata: int | None = None,
    ) -> None:
        self.statistic = statistic
        self.p_value = p_value
        self.alternative = alternative
        self.group = group
        self.rho = rho
        self.gamma = gamma
        self.z = z
        self.p_values = p_values
        self.correlation = correlation
        self.p_bonferroni = p_bonferroni
        self.strata = strata

    def __repr__(self) -> str:
        header = "Max-Combo Log-Rank Test"
        if self.strata:
            header = "Stratified " + header
        out = header + "\n" + "=" * len(header)
        out += f"\nAlternative      : {self.alternative}"
        out += f"\nz of group       : {self.group!r}"
        if self.strata is not None:
            out += f"\nStrata           : {self.strata}"
        for r, g, z, p in zip(self.rho, self.gamma, self.z, self.p_values):
            name = f"FH({r:g}, {g:g})"
            out += f"\n{name:<17}: z = {z:.6g}, p = {p:.6g}"
        out += (
            f"\nStatistic        : {self.statistic:.6g}"
            + f"\np-value          : {self.p_value:.6g}"
        )
        return out

    def to_dict(self) -> dict:
        """The result as strict JSON: the settings, each component's
        ``rho``, ``gamma``, ``z`` and p-value, the correlation matrix, and
        the combined statistic and p-value (a non-finite number is
        ``None``)."""
        return {
            "test": "max-combo log-rank",
            "alternative": self.alternative,
            "group": _plain(self.group),
            "strata": self.strata,
            "components": [
                {
                    "rho": float(r),
                    "gamma": float(g),
                    "z": _finite(z),
                    "p_value": _finite(p),
                }
                for r, g, z, p in zip(
                    self.rho, self.gamma, self.z, self.p_values
                )
            ],
            "correlation": [
                [_finite(v) for v in row] for row in self.correlation
            ],
            "statistic": _finite(self.statistic),
            "p_value": _finite(self.p_value),
            "p_bonferroni": _finite(self.p_bonferroni),
        }


def _max_normal_sf(m: float, R: npt.NDArray, alternative: str) -> float:
    """``P(max |Z_j| >= m)`` (two-sided), ``P(max Z_j >= m)`` ("greater")
    or ``P(min Z_j <= m)`` ("less") for ``Z`` normal with mean 0 and
    correlation ``R``.

    The probability of the union of the events ``A_j`` (component ``j``
    beyond ``m``) is ``sum_j P(A_j) E[1 / N]``, the expectation taken over
    ``Z`` drawn from the mixture of the normal given ``A_j`` with weights
    ``P(A_j)`` and ``N`` the number of the events that ``Z`` is in
    (Naiman and Wynn, 1997; Adler, Blanchet and Liu, 2012). ``1 / N`` lies
    between ``1 / J`` and 1, so the estimate has a small relative error
    however small the probability: about 0.1% with the 65536 draws per
    component used here, at a fixed seed so that a result does not
    change between calls. Generic multivariate normal integrators
    estimate ``1 - P(no A_j)``, whose absolute error swamps a small
    p-value (0.5% and more at p = 0.001, seen); they are used for a
    p-value above 0.2, where that error is small next to it.
    """
    from scipy.stats import norm

    J = R.shape[0]
    two_sided = alternative == "two-sided"
    if alternative == "less":
        m = -m
    if two_sided:
        m = abs(m)
    single = float(norm.sf(m)) * (2.0 if two_sided else 1.0)
    if J == 1:
        return min(single, 1.0)
    if single == 0.0:
        return 0.0
    rng = np.random.default_rng(20261113)
    size = 2**16
    inverse_counts = []
    for j in range(J):
        u = rng.random(size)
        # Z_j given A_j: its size from the normal tail beyond m.
        z_j = norm.isf(u * norm.sf(m))
        if two_sided:
            z_j = np.where(rng.random(size) < 0.5, -z_j, z_j)
        # The rest given Z_j: mean R[:, j] Z_j, covariance R - r r'.
        conditional = R - np.outer(R[:, j], R[j, :])
        values, vectors = np.linalg.eigh(conditional)
        root = vectors * np.sqrt(np.clip(values, 0.0, None))
        draws = z_j[:, None] * R[j][None, :]
        draws += rng.standard_normal((size, J)) @ root.T
        draws[:, j] = z_j
        beyond = np.abs(draws) >= m if two_sided else draws >= m
        inverse_counts.append(1.0 / beyond.sum(axis=1))
    p = J * single * float(np.mean(np.concatenate(inverse_counts)))
    if p > 0.2:
        # A large p-value is better found as 1 - P(no A_j), whose absolute
        # error (Genz's algorithm) is then small next to it; the estimate
        # above has a relative error of about 0.05% there.
        from scipy.stats import multivariate_normal

        inside = multivariate_normal.cdf(
            np.full(J, m),
            mean=np.zeros(J),
            cov=R,
            allow_singular=True,
            lower_limit=np.full(J, -m) if two_sided else None,
            maxpts=50000 * J,
            abseps=1e-5,
            releps=0.0,
            rng=np.random.default_rng(20261113),
        )
        p = 1.0 - float(inside)
    return float(min(max(p, 0.0), 1.0))


def maxcombo_logrank(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    rho: Sequence[float] = DEFAULT_RHO,
    gamma: Sequence[float] = DEFAULT_GAMMA,
    alternative: str = "two-sided",
    strata: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | float | None = None,
    t0: float | None = None,
    w: npt.ArrayLike | float | None = None,
) -> MaxComboResult:
    r"""
    The max-combo test of two groups: the largest of several
    Fleming-Harrington weighted log-rank statistics, with a p-value that
    allows for taking the largest.

    The log-rank test is the most powerful test when one group's hazard
    is a constant multiple of the other's. When the difference is not
    like that -- a treatment that takes time to work (a delayed effect),
    one that wears off, or curves that cross -- a weighted log-rank test
    that stresses early or late times can be far more powerful, but which
    one is best is not known before the data are seen, and running
    several and reporting the smallest p-value overstates the evidence.
    The max-combo test runs several weightings at once and reports the
    strongest, with a p-value from the joint distribution of all of them,
    so the overall chance of a false rejection stays at the level chosen.
    It loses little power against the log-rank test under proportional
    hazards and gains much when the hazards are not proportional.

    Parameters
    ----------
    x, Z, c, n
        As for :func:`~surpyval.logrank`: the times, the group labels
        (exactly two groups), the censoring flags (0 observed, 1 right
        censored) and the counts.
    rho, gamma : sequence of float, optional
        The Fleming-Harrington parameters of the components, one pair per
        test: weight ``S(t-)**rho * (1 - S(t-))**gamma`` with ``S`` the
        pooled Kaplan-Meier estimate. The default is the four tests of Lin
        et al. (2020): FH(0, 0), the log-rank test; FH(0, 1), late
        differences; FH(1, 0), early differences; and FH(1, 1),
        differences in the middle. ``rho`` and ``gamma`` must have the same
        length, and no pair may repeat.
    alternative : str, optional
        "two-sided" (the default): any difference, the largest ``|z|``.
        "greater": the first group (in sorted label order) has the higher
        hazard, the largest ``z``. "less": it has the lower hazard, the
        smallest ``z``.
    strata : array like, optional
        Stratum labels, as for :func:`~surpyval.logrank`: each component's
        numerator, variance and the covariances between them are summed
        over the strata, and each stratum's weights come from its own
        Kaplan-Meier estimate.
    tl : array like or scalar, optional
        Entry (left truncation) times, as for :func:`~surpyval.logrank`.
    t0 : scalar, optional
        Compare the groups up to ``t0`` only, as for
        :func:`~surpyval.logrank`.
    w : None
        Sampling weights are not supported: the covariance of the
        components would need the design-based variance of every pair of
        weightings. Passing ``w`` raises an error; use
        :func:`~surpyval.logrank` with ``w`` and one weighting chosen in
        advance.

    Returns
    -------
    MaxComboResult
        With the combined ``statistic`` and ``p_value``, each component's
        ``z`` and ``p_values``, and their ``correlation``.

    Raises
    ------
    ValueError
        If ``Z`` does not have exactly two groups, ``rho`` and ``gamma``
        differ in length, are empty, negative or repeat a pair,
        ``alternative`` is unknown, ``w`` is given, or for any reason
        :func:`~surpyval.logrank` would raise.

    Notes
    -----
    With :math:`O_t` and :math:`E_t` the observed and expected events of
    the first group at event time :math:`t`, :math:`v_t` their
    hypergeometric variance and :math:`w_j(t) = S(t-)^{\rho_j} (1 -
    S(t-))^{\gamma_j}`, the components are

    .. math::

        z_j = \frac{\sum_t w_j(t) (O_t - E_t)}
                   {\sqrt{\sum_t w_j(t)^2 v_t}},

    and their covariance is estimated by :math:`\sum_t w_j(t) w_l(t)
    v_t`. Under the null hypothesis the :math:`z_j` are jointly normal
    with that correlation, so the p-value of :math:`m = \max_j |z_j|` is
    :math:`1 - P(-m < Z_j < m \text{ for all } j)`, a multivariate normal
    probability. It is computed at a fixed seed, so a result does not
    change between calls: a small p-value by importance sampling of the
    union of the events :math:`|Z_j| \ge m` (relative error about 0.1%),
    one above 0.2 by Genz's algorithm (absolute error about 1e-4). This is
    R's ``nph::logrank.maxtest`` (Karrison 2016;
    Lin et al. 2020).

    Examples
    --------
    Patients with advanced lung cancer, by sex (R's ``survival::lung``):

    >>> from surpyval import maxcombo_logrank
    >>> from surpyval.datasets import load_lung
    >>> df = load_lung()
    >>> res = maxcombo_logrank(df["time"], df["sex"], c=1 - df["status"])
    >>> print(res.z.round(3))
    [3.214 1.86  3.566 2.769]
    >>> print(round(res.statistic, 3), round(res.p_value, 5))
    3.566 0.00098

    The early-difference weighting FH(1, 0) is the strongest, and the
    p-value allows for its having been picked from four: 0.00098, where
    FH(1, 0)'s own p-value, 0.00036, would overstate the evidence. R's
    ``nph::logrank.maxtest`` gives the same ``z`` and a p-value of
    0.000984.

    References
    ----------
    Fleming, T. R. and Harrington, D. P. (1991), "Counting Processes and
    Survival Analysis", Wiley.

    Karrison, T. G. (2016), "Versatile tests for comparing survival
    curves based on weighted log-rank statistics", The Stata Journal 16,
    678-690.

    Lin, R. S., Lin, J., Roychoudhury, S., et al. (2020), "Alternative
    analysis methods for time to event endpoints under nonproportional
    hazards: a comparative analysis", Statistics in Biopharmaceutical
    Research 12, 187-198.
    """
    from scipy.stats import norm

    if w is not None:
        raise ValueError(
            "maxcombo_logrank does not take sampling weights 'w': the "
            "covariance of the components would need the design-based "
            "variance of every pair of weightings. Choose one weighting in "
            "advance and use surpyval.logrank(x, Z, c, w=..., "
            "weighting='fleming-harrington', rho=..., gamma=...)."
        )
    if alternative not in ALTERNATIVES:
        raise ValueError(
            "'alternative' must be one of {}; got {!r}".format(
                list(ALTERNATIVES), alternative
            )
        )
    rho_t = tuple(float(r) for r in np.atleast_1d(rho))
    gamma_t = tuple(float(g) for g in np.atleast_1d(gamma))
    if len(rho_t) != len(gamma_t) or len(rho_t) == 0:
        raise ValueError(
            "'rho' and 'gamma' must be sequences of the same, positive "
            "length, one (rho, gamma) pair per component; got {} and "
            "{}".format(len(rho_t), len(gamma_t))
        )
    pairs = list(zip(rho_t, gamma_t))
    if len(set(pairs)) != len(pairs):
        raise ValueError(
            "'rho' and 'gamma' repeat a (rho, gamma) pair; give each "
            "component once"
        )
    if not all(np.isfinite(v) and v >= 0 for p in pairs for v in p):
        raise ValueError(
            "'rho' and 'gamma' must be finite and non-negative; got rho={} "
            "and gamma={}".format(list(rho_t), list(gamma_t))
        )

    x_arr, Z_arr, c_arr, n_arr, tl_arr, strata, _, groups = _checked_inputs(
        x, Z, c, n, "fleming-harrington", strata, tl, t0, None, True, 3
    )
    if groups.size != 2:
        raise ValueError(
            "maxcombo_logrank compares two groups; 'Z' has {}. Compare "
            "them in pairs with surpyval.pairwise_logrank, or all at once "
            "with surpyval.logrank and one weighting chosen in "
            "advance".format(groups.size)
        )

    J = len(pairs)
    U = np.zeros(J)
    C = np.zeros((J, J))
    E = np.zeros(2)
    labels = [None] if strata is None else list(np.unique(strata))
    for s in labels:
        mask = slice(None) if s is None else strata == s

        def rows(a: Any) -> Any:
            return None if a is None else a[mask]

        times, r_gt, d_gt = _risk_table(
            x_arr[mask],
            Z_arr[mask],
            rows(c_arr),
            rows(n_arr),
            groups,
            rows(tl_arr),
        )
        if times.size == 0:
            continue
        r_t = r_gt.sum(axis=0)
        d_t = d_gt.sum(axis=0)
        W = np.array(
            [
                _time_weights("fleming-harrington", r, g, r_t, d_t)
                for r, g in pairs
            ]
        )
        with np.errstate(all="ignore"):
            expected = np.where(r_t > 0, d_t * r_gt / r_t, 0.0)
            hyper = np.where(r_t > 1, d_t * (r_t - d_t) / (r_t - 1), 0.0)
            p0 = np.where(r_t > 0, r_gt[0] / r_t, 0.0)
        v_t = hyper * p0 * (1.0 - p0)
        U += W @ (d_gt[0] - expected[0])
        C += (W * v_t) @ W.T
        E += expected.sum(axis=1)

    n_strata = None if strata is None else len(labels)
    sd = np.sqrt(np.diag(C))
    if (E > 0).sum() < 2 or not np.all(sd > 0):
        # No event, or every risk set holds one group only (or a weighting
        # that is zero at every event time): nothing to compare.
        nothing = np.zeros(J)
        return MaxComboResult(
            0.0,
            1.0,
            alternative,
            _plain(groups[0]),
            rho_t,
            gamma_t,
            nothing,
            np.ones(J),
            np.eye(J),
            1.0,
            strata=n_strata,
        )
    z = U / sd
    R = C / np.outer(sd, sd)
    np.fill_diagonal(R, 1.0)
    if alternative == "two-sided":
        statistic = float(np.max(np.abs(z)))
        p_each = 2.0 * norm.sf(np.abs(z))
    elif alternative == "greater":
        statistic = float(np.max(z))
        p_each = norm.sf(z)
    else:
        statistic = float(np.min(z))
        p_each = norm.cdf(z)
    p_value = _max_normal_sf(statistic, R, alternative)
    p_bonferroni = float(min(1.0, J * float(np.min(p_each))))
    return MaxComboResult(
        statistic,
        p_value,
        alternative,
        _plain(groups[0]),
        rho_t,
        gamma_t,
        z,
        p_each,
        R,
        p_bonferroni,
        strata=n_strata,
    )
