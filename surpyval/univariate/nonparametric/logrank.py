import numpy as np
from scipy.stats import chi2

from surpyval.univariate.nonparametric.kaplan_meier import kaplan_meier
from surpyval.utils import xcnt_handler


class LogRankResult:
    """
    Result of a (weighted) log-rank test.

    Attributes
    ----------

    statistic : float
        The chi-squared test statistic.
    dof : int
        The degrees of freedom (number of groups - 1).
    p_value : float
        The p-value of the test.
    weighting : str
        The weighting used for the test.
    """

    def __init__(self, statistic, dof, p_value, weighting, strata=None):
        self.statistic = statistic
        self.dof = dof
        self.p_value = p_value
        self.weighting = weighting
        self.strata = strata

    def __repr__(self):
        header = "Stratified Log-Rank Test" if self.strata else "Log-Rank Test"
        out = (
            header
            + "\n"
            + "=" * len(header)
            + "\nWeighting        : {w}".format(w=self.weighting)
        )
        if self.strata is not None:
            out += "\nStrata           : {s}".format(s=self.strata)
        out += (
            "\nStatistic        : {s:.6g}".format(s=self.statistic)
            + "\nDoF              : {d}".format(d=self.dof)
            + "\np-value          : {p:.6g}".format(p=self.p_value)
        )
        return out


def _logrank_z_v(x, Z, c, n, groups, weighting, rho, gamma):
    """Per-stratum (or whole-sample) weighted log-rank ``(z, V)``.

    Returns the length-``k`` vector of weighted observed-minus-expected
    counts and its ``k x k`` covariance, using the fixed group order
    ``groups`` so contributions from different strata are aligned and can be
    summed. A group absent from this stratum simply contributes zeros.
    """
    k = groups.size
    x_g, c_g, n_g = [], [], []
    for g in groups:
        mask = Z == g
        x_i = np.atleast_1d(x)[mask]
        c_i = None if c is None else np.atleast_1d(c)[mask]
        n_i = None if n is None else np.atleast_1d(n)[mask]
        if x_i.size == 0:
            x_g.append(np.array([]))
            c_g.append(np.array([]))
            n_g.append(np.array([]))
            continue
        x_i, c_i, n_i, _ = xcnt_handler(x=x_i, c=c_i, n=n_i)
        if ((c_i != 0) & (c_i != 1)).any():
            raise ValueError(
                "Log-rank test can only be used with observed and "
                + "right censored data"
            )
        x_g.append(x_i)
        c_g.append(c_i)
        n_g.append(n_i)

    event_pool = [x_i[c_i == 0] for x_i, c_i in zip(x_g, c_g) if x_i.size > 0]
    event_times = (
        np.unique(np.concatenate(event_pool)) if event_pool else np.array([])
    )
    m = event_times.size
    if m == 0:
        return np.zeros(k), np.zeros((k, k))

    r_gt = np.zeros((k, m))
    d_gt = np.zeros((k, m))
    for j, (x_i, c_i, n_i) in enumerate(zip(x_g, c_g, n_g)):
        if x_i.size == 0:
            continue
        r_gt[j] = (n_i[:, None] * (x_i[:, None] >= event_times)).sum(axis=0)
        d_gt[j] = (
            n_i[:, None]
            * ((x_i[:, None] == event_times) & (c_i == 0)[:, None])
        ).sum(axis=0)

    r_t = r_gt.sum(axis=0)
    d_t = d_gt.sum(axis=0)

    if weighting == "log-rank":
        w = np.ones(m)
    elif weighting == "gehan":
        w = r_t
    elif weighting == "tarone-ware":
        w = np.sqrt(r_t)
    else:
        # Pooled left-continuous Kaplan-Meier, i.e. the value of the
        # estimate just prior to each event time. Weights are formed
        # within the stratum (standard for a stratified weighted test).
        S = kaplan_meier(r_t, d_t)
        S_prev = np.hstack([[1.0], S[:-1]])
        w = S_prev**rho * (1 - S_prev) ** gamma

    with np.errstate(all="ignore"):
        expected = np.where(r_t > 0, d_t * r_gt / r_t, 0.0)
    z = (w * (d_gt - expected)).sum(axis=1)

    with np.errstate(all="ignore"):
        hyper = np.where(r_t > 1, d_t * (r_t - d_t) / (r_t - 1), 0.0)
        prop = np.where(r_t > 0, r_gt / r_t, 0.0)
    V = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            delta = 1.0 if a == b else 0.0
            V[a, b] = (w**2 * hyper * prop[a] * (delta - prop[b])).sum()

    return z, V


def logrank(
    x, Z, c=None, n=None, weighting="log-rank", rho=0, gamma=0, strata=None
):
    r"""
    The k-sample (weighted) log-rank test for the equality of survival
    distributions of right censored data.

    At each distinct event time the observed number of events in each
    group is compared with the number expected under the null
    hypothesis that all groups share the same survival distribution.
    The weighted sums of these differences form a chi-squared statistic
    with k - 1 degrees of freedom.

    Parameters
    ----------

    x : array like
        Array of observations of the random variables.
    Z : array like
        Array of group labels for each observation. Any hashable values
        can be used; the test has len(unique(Z)) - 1 degrees of
        freedom.
    c : array like, optional
        Array of censoring flags. 0 is observed and 1 is right
        censored. Left or interval censored data cannot be used with
        the log-rank test. If not provided assumes all values are
        observed.
    n : array like, optional
        Array of counts for each x. If :code:`None` assumes each
        observation is 1.
    weighting : str, optional
        The weighting to use at each event time. One of:

        - "log-rank": weight 1 (the standard log-rank test; sensitive
          to proportional hazards alternatives),
        - "gehan": weight r (a.k.a. Gehan-Breslow-Wilcoxon; emphasises
          early differences),
        - "tarone-ware": weight sqrt(r),
        - "fleming-harrington": weight S(t-)**rho * (1 - S(t-))**gamma
          where S is the pooled Kaplan-Meier estimate; rho > 0
          emphasises early differences and gamma > 0 late differences.

        Defaults to "log-rank".
    rho, gamma : scalar, optional
        The parameters of the Fleming-Harrington weighting. Only used
        when weighting is "fleming-harrington". Defaults to 0, 0 (which
        is identical to the log-rank weighting).
    strata : array like, optional
        Array of stratum labels, one per observation. When supplied the
        test is *stratified*: the observed-minus-expected numerators and
        their variances are accumulated *within* each stratum (risk sets
        never cross a stratum boundary) and summed before forming the
        statistic. This removes a nuisance factor -- one whose baseline
        hazard differs across strata -- from the comparison, so groups are
        only ever compared against others in the same stratum. The degrees
        of freedom are unchanged (number of groups minus one).

    Returns
    -------

    result : LogRankResult
        Object with the chi-squared ``statistic``, the degrees of
        freedom ``dof``, and the ``p_value``.

    Examples
    --------
    >>> from surpyval import logrank
    >>> x = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161,
    ...      5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
    >>> c = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1,
    ...      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    >>> Z = [1] * 11 + [2] * 12
    >>> res = logrank(x, Z, c=c)
    >>> print(round(res.statistic, 2), round(res.p_value, 4))
    3.4 0.0653

    References
    ----------

    Klein, J. P. and Moeschberger, M. L. (2003), "Survival Analysis:
    Techniques for Censored and Truncated Data", 2nd ed., Chapter 7.
    """
    weightings = ["log-rank", "gehan", "tarone-ware", "fleming-harrington"]
    if weighting not in weightings:
        raise ValueError("'weighting' must be in {}".format(weightings))

    Z = np.asarray(Z)
    if Z.ndim != 1:
        raise ValueError("'Z' must be a 1D array of group labels")
    if len(Z) != len(np.atleast_1d(x)):
        raise ValueError("'Z' must have a label for each observation")

    groups = np.unique(Z)
    if groups.size < 2:
        raise ValueError("Test requires at least two groups")

    x = np.atleast_1d(x)
    c_arr = None if c is None else np.atleast_1d(c)
    n_arr = None if n is None else np.atleast_1d(n)

    k = groups.size
    n_strata = None
    if strata is None:
        z, V = _logrank_z_v(x, Z, c_arr, n_arr, groups, weighting, rho, gamma)
    else:
        strata = np.asarray(strata)
        if len(strata) != len(x):
            raise ValueError("'strata' must have a label for each observation")
        z = np.zeros(k)
        V = np.zeros((k, k))
        unique_strata = np.unique(strata)
        n_strata = int(unique_strata.size)
        for s in unique_strata:
            mask = strata == s
            z_s, V_s = _logrank_z_v(
                x[mask],
                Z[mask],
                None if c_arr is None else c_arr[mask],
                None if n_arr is None else n_arr[mask],
                groups,
                weighting,
                rho,
                gamma,
            )
            z += z_s
            V += V_s

    # The covariance matrix is singular (rows sum to zero); drop the
    # last group
    z_r = z[:-1]
    V_r = V[:-1, :-1]
    try:
        statistic = float(z_r @ np.linalg.solve(V_r, z_r))
    except np.linalg.LinAlgError:
        statistic = float(z_r @ np.linalg.pinv(V_r) @ z_r)

    dof = k - 1
    p_value = float(chi2.sf(statistic, dof))

    if weighting == "fleming-harrington":
        weighting = "fleming-harrington(rho={}, gamma={})".format(rho, gamma)

    return LogRankResult(statistic, dof, p_value, weighting, strata=n_strata)
