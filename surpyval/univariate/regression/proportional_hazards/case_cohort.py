"""Cox fits to case-cohort and nested case-control samples.

Both designs measure the covariates on only part of a cohort, because
measuring everyone costs too much: every case (unit with the event) and
a sample of the others.

* **Case-cohort** (Prentice 1986): a random *subcohort* is drawn from the
  whole cohort at the start, and the covariates are measured on it and on
  every case outside it. The risk set at an event time is then known
  only for the subcohort, so the partial likelihood is replaced by a
  pseudo-likelihood that compares each case with the subcohort members
  at risk, and the variance gains a second term for the sampling of the
  subcohort ("phase two").
* **Nested case-control** (Thomas 1977): at each event time a few
  controls are drawn from the units then at risk. Each case's matched set
  (the case or cases with their controls) is a small risk set, and the
  fit is the conditional logistic likelihood: a Cox model stratified by
  set, at one time.

The case-cohort estimators are those of R's ``survival::cch`` (Therneau
and Li 1999) and Barlow's (1994) weighting; see
:func:`fit_case_cohort` for each one's estimating equation and variance.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.no_maximum import warn_unverified
from surpyval.utils.warnings import caller_stacklevel

from ..semi_parametric_regression_model import SemiParametricRegressionModel
from ._weighted_cox import WeightedCox, newton, robust_covariance

#: The case-cohort estimators, by their names here (R's ``cch`` names in
#: the docstring of :func:`fit_case_cohort`).
CASE_COHORT_METHODS = (
    "prentice",
    "self-prentice",
    "lin-ying",
    "barlow",
    "borgan-i",
    "borgan-ii",
)
_STRATIFIED = ("borgan-i", "borgan-ii")


class _Sample:
    """The checked case-cohort sample: one row per unit sampled."""

    def __init__(
        self,
        x: Any,
        Z: Any,
        c: Any,
        tl: Any,
        subcohort: Any,
        cohort_size: Any,
        sampling_strata: Any,
        stratified: bool,
    ) -> None:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError(
                "x must be one time per unit sampled: a case-cohort fit "
                "takes exact and right-censored times only (no intervals)."
            )
        k = x.shape[0]
        Z = np.asarray(Z, dtype=float)
        Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
        if Z.shape[0] != k:
            raise ValueError(
                f"Z has {Z.shape[0]} rows for {k} values of x: give one row "
                "of covariates per unit sampled."
            )
        c = np.zeros(k, dtype=int) if c is None else np.asarray(c)
        if c.shape != (k,) or not np.all(np.isin(c, (0, 1))):
            raise ValueError(
                "c must be 0 (the event) or 1 (right censored), one per "
                "value of x: a case-cohort fit has no term for left- or "
                "interval-censored rows."
            )
        tl = np.zeros(k) if tl is None else np.asarray(tl, dtype=float)
        tl = np.broadcast_to(tl, (k,)).astype(float)
        for name, arr in (("x", x), ("Z", Z), ("tl", tl)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"{name} has missing or infinite values: a case-cohort "
                    "fit needs complete rows (drop the incomplete units, "
                    "or impute them, before fitting)."
                )
        if np.any(tl >= x):
            bad = int(np.sum(tl >= x))
            raise ValueError(
                f"{bad} unit(s) enter (tl) at or after their time x: each "
                "entry must come before its exit."
            )
        if subcohort is None:
            raise ValueError(
                "subcohort is required: True for each unit drawn into the "
                "subcohort, False for a case from outside it."
            )
        sub = np.asarray(subcohort)
        if sub.shape != (k,) or not np.all(np.isin(sub, (0, 1))):
            raise ValueError(
                "subcohort must be True/False (or 1/0), one per value of x."
            )
        sub = sub.astype(bool)
        outside_censored = int(np.sum(~sub & (c == 1)))
        if outside_censored:
            raise ValueError(
                f"{outside_censored} censored unit(s) are outside the "
                "subcohort: a case-cohort sample is the subcohort plus the "
                "cases, so a unit outside the subcohort must be a case "
                "(c = 0). Check subcohort and c."
            )
        if not np.any(c == 0):
            raise ValueError("The sample has no event: nothing to fit.")
        self.x, self.Z, self.c, self.tl, self.sub = x, Z, c, tl, sub
        self.case = c == 0
        if stratified:
            if sampling_strata is None:
                raise ValueError(
                    "The Borgan estimators need sampling_strata: the "
                    "stratum each unit's subcohort was drawn from, with "
                    "cohort_size a dict of each stratum's size in the "
                    "cohort."
                )
            labels = np.asarray(sampling_strata)
            if labels.shape != (k,):
                raise ValueError(
                    "sampling_strata must give one label per value of x."
                )
            levels, code = np.unique(labels, return_inverse=True)
            if not isinstance(cohort_size, dict):
                raise ValueError(
                    "cohort_size must be a dict {stratum: size in the "
                    "cohort} for the Borgan estimators."
                )
            missing = [v for v in levels.tolist() if v not in cohort_size]
            if missing:
                raise ValueError(
                    f"cohort_size has no size for the strata {missing}: "
                    "give every stratum's size in the cohort."
                )
            N = np.array([float(cohort_size[v]) for v in levels.tolist()])
        else:
            if sampling_strata is not None:
                raise ValueError(
                    "sampling_strata is for method='borgan-i' or "
                    "'borgan-ii' (a subcohort drawn within strata); the "
                    "other methods take a simple random subcohort. Drop "
                    "sampling_strata, or choose a Borgan method."
                )
            if isinstance(cohort_size, dict) or np.ndim(cohort_size) != 0:
                raise ValueError(
                    "cohort_size must be one number, the size of the "
                    "cohort the subcohort was drawn from (a dict of sizes "
                    "is for the Borgan estimators)."
                )
            levels = np.array([0])
            code = np.zeros(k, dtype=int)
            N = np.array([float(cohort_size)])
        self.levels, self.code, self.N = levels, code, N
        self.n_sub = np.bincount(code[sub], minlength=levels.size)
        self.n_sub_cases = np.bincount(
            code[sub & self.case], minlength=levels.size
        )
        self.n_cases = np.bincount(code[self.case], minlength=levels.size)
        if np.any(self.n_sub < 2):
            raise ValueError(
                "Each (sampling) stratum needs at least two subcohort "
                f"members; the counts are {self.n_sub.tolist()}."
            )
        if np.any(N < self.n_sub + self.n_cases - self.n_sub_cases):
            raise ValueError(
                "cohort_size is smaller than the sample drawn from it "
                f"({N.tolist()} for {self.n_sub.tolist()} subcohort "
                "members and the cases outside it): give the size of the "
                "whole cohort."
            )
        # Half the smallest gap between distinct event times (as cch):
        # a case from outside the subcohort enters this long before its
        # event, so that it is in its own risk set and no other.
        times = np.unique(x[self.case])
        self.delta = (
            float(np.min(np.diff(times)) / 2) if times.size > 1 else 1.0
        )


def _refuse_degenerate(s: _Sample, method: str) -> None:
    """Refuse a sample the estimator cannot use: Lin-Ying's and Borgan
    II's weights need non-cases in the subcohort (in every sampling
    stratum) to stand for the cohort's; Self-Prentice's risk sets (also
    the source of Prentice's variance, and Borgan I's) need a subcohort
    member at risk at every case's time."""
    if method in ("lin-ying", "borgan-ii"):
        non_cases = s.n_sub - s.n_sub_cases
        if np.any(non_cases < 2):
            where = (
                ""
                if method == "lin-ying"
                else " in each sampling stratum (the counts are "
                f"{dict(zip(s.levels.tolist(), non_cases.tolist()))})"
            )
            raise ValueError(
                f"method={method!r} needs at least two subcohort members "
                f"who are not cases{where}: they stand for the cohort's "
                "non-cases, and their spread gives the sampling variance. "
                "Use method='prentice' or 'self-prentice' (or 'borgan-i' "
                "for a stratified subcohort), which weight the subcohort "
                "as a whole."
            )
    if method in ("prentice", "self-prentice", "borgan-i"):
        times = s.x[s.case]
        exits = np.sort(s.x[s.sub])
        entries = np.sort(s.tl[s.sub])
        # Subcohort members with tl < t <= x at each case's time t
        at_risk = (exits.size - np.searchsorted(exits, times, "left")) - (
            entries.size - np.searchsorted(entries, times, "left")
        )
        empty = int(np.sum(at_risk < 1))
        if empty:
            raise ValueError(
                f"At {empty} case time(s) no subcohort member is at risk, "
                f"so method={method!r} (whose risk sets, and variance, are "
                "the subcohort's at risk) has nothing to compare the case "
                "with. Use method='barlow' or 'lin-ying', whose risk sets "
                "hold the cases too, or check subcohort and tl."
            )


def _rows(sample: _Sample, kind: str) -> dict:
    """The rows each estimator fits, as a Cox model with weights ``w``
    (in the deaths and the risk sets) and risk-only multipliers ``rho``.

    * ``"self-prentice"``: each case once as a death that is in no risk
      set (``rho = 0``), and each subcohort member once as at risk
      (never a death), as cch's Self-Prentice; ``"borgan-i"`` the same
      with the subcohort member weighted ``N_s / n_s``.
    * ``"prentice"``: each unit once; a case outside the subcohort enters
      ``delta`` before its event.
    * ``"lin-ying"``: each unit once, a non-case of the subcohort at risk
      ``(N - D) / (n - d)`` times over (``rho``), as cch's offset;
      ``"borgan-ii"`` the same by stratum, as a weight.
    * ``"barlow"``: a subcohort member at weight ``N / n`` until it fails
      or leaves, and each case at weight 1 from ``delta`` before its
      event (Therneau and Li 1999).
    """
    s = sample
    unit = np.arange(s.x.shape[0])
    case, sub = s.case, s.sub
    if kind in ("self-prentice", "borgan-i"):
        wt = (s.N / s.n_sub)[s.code] if kind == "borgan-i" else 1.0
        w_sub = np.broadcast_to(wt, unit.shape)[sub]
        return {
            "x": np.concatenate([s.x[case], s.x[sub]]),
            "c": np.concatenate(
                [np.zeros(case.sum(), int), np.ones(sub.sum(), int)]
            ),
            "tl": np.concatenate([s.tl[case], s.tl[sub]]),
            "Z": np.vstack([s.Z[case], s.Z[sub]]),
            "w": np.concatenate([np.ones(case.sum()), w_sub]),
            "rho": np.concatenate([np.zeros(case.sum()), np.ones(sub.sum())]),
            "unit": np.concatenate([unit[case], unit[sub]]),
            "at_risk": np.concatenate(
                [np.zeros(case.sum(), bool), np.ones(sub.sum(), bool)]
            ),
        }
    if kind == "prentice":
        tl = np.where(sub, s.tl, np.maximum(s.tl, s.x - s.delta))
        ones = np.ones(unit.shape)
        return {
            "x": s.x,
            "c": s.c,
            "tl": tl,
            "Z": s.Z,
            "w": ones,
            "rho": ones,
            "unit": unit,
            "at_risk": ~case,
        }
    if kind in ("lin-ying", "borgan-ii"):
        # The non-cases of the subcohort stand for the cohort's non-cases
        non_case_weight = (s.N - s.n_cases) / (s.n_sub - s.n_sub_cases)
        weight = np.where(case, 1.0, non_case_weight[s.code])
        ones = np.ones(unit.shape)
        lin_ying = kind == "lin-ying"
        return {
            "x": s.x,
            "c": s.c,
            "tl": s.tl,
            "Z": s.Z,
            "w": ones if lin_ying else weight,
            "rho": weight if lin_ying else ones,
            "unit": unit,
            "at_risk": ~case,
        }
    if kind == "barlow":
        # Weighted by the inverse of each one's chance of being sampled:
        # a subcohort member N / n (per stratum for Borgan I's baseline)
        # until its event, a case 1 at its event.
        wt = (s.N / s.n_sub)[s.code]
        stop = np.where(case, s.x - s.delta, s.x)
        keep = sub & (stop > s.tl)
        enter = np.maximum(s.tl[case], s.x[case] - s.delta)
        k = int(case.sum())
        return {
            "x": np.concatenate([stop[keep], s.x[case]]),
            "c": np.concatenate([np.ones(keep.sum(), int), np.zeros(k, int)]),
            "tl": np.concatenate([s.tl[keep], enter]),
            "Z": np.vstack([s.Z[keep], s.Z[case]]),
            "w": np.concatenate([wt[keep], np.ones(k)]),
            "rho": np.ones(int(keep.sum()) + k),
            "unit": np.concatenate([unit[keep], unit[case]]),
            "at_risk": np.concatenate(
                [np.ones(keep.sum(), bool), np.zeros(k, bool)]
            ),
        }
    raise AssertionError(kind)  # pragma: no cover


def _likelihood(rows: dict, mean: npt.NDArray, tie_method: str) -> WeightedCox:
    return WeightedCox(
        rows["x"],
        rows["c"],
        np.ones(rows["x"].shape[0]),
        rows["w"],
        rows["tl"],
        rows["Z"] - mean,
        tie_method,
        rho=rows["rho"],
    )


def _solve(lik: WeightedCox, p: int, tol: float, what: str) -> tuple:
    """``(beta, information, its inverse, converged)``."""
    beta, converged, iters = newton(lik, p, tol)
    _, _, info = lik.derivatives(beta)
    if not converged or not np.all(np.isfinite(beta)):
        warn_unverified(
            what,
            f"Newton-Raphson stopped after {iters} steps without converging",
            "a covariate may separate the cases from the subcohort; check "
            "the coefficients, or drop the covariate",
        )
    try:
        bread = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        raise ValueError(
            f"{what}: the information is singular, so a covariate column "
            "is constant or a combination of the others in the sample. "
            "Drop it and fit again."
        ) from None
    return beta, info, bread, bool(converged)


def _phase_two_borgan(
    sample: _Sample,
    rows: dict,
    resid: npt.NDArray,
    weight: npt.NDArray,
    size: npt.NDArray,
    sampled: npt.NDArray,
) -> npt.NDArray:
    """Borgan et al.'s (2000) phase-two term ``Delta``: in each stratum
    the sample covariance (divisor ``m_s - 1``) of the score residuals of
    the rows standing for others, times ``(wt_s - 1) * size_s``."""
    p = resid.shape[1]
    delta = np.zeros((p, p))
    stratum = sample.code[rows["unit"]]
    for j in range(sample.levels.size):
        mask = rows["at_risk"] & (stratum == j)
        L = resid[mask]
        L = L - L.mean(axis=0)
        m = sampled[j]
        delta += (weight[j] - 1) * size[j] * (L.T @ L) / (m - 1)
    return delta


def fit_case_cohort(
    x: Any,
    Z: Any,
    subcohort: Any,
    cohort_size: Any,
    c: Any = None,
    tl: Any = None,
    method: str = "prentice",
    sampling_strata: Any = None,
    robust: bool = False,
    tie_method: str = "efron",
    tol: float = 1e-10,
) -> CaseCohortModel:
    r"""
    A Cox model fitted to a case-cohort sample.

    In a case-cohort study a random *subcohort* is drawn from the whole
    cohort at the start, and the covariates are measured only on it and
    on the cases (units with the event) from outside it. The other units
    of the cohort are never measured. At each event time the case is
    compared with the subcohort members still at risk, who stand for
    everyone at risk; the estimators differ in which units they put in
    those risk sets and how they weight them. Every one estimates the
    same hazard ratios as a Cox fit to the whole cohort would, and they
    agree closely in practice. The standard errors are larger than the
    model-based ones of an ordinary fit, because the subcohort is a
    sample: each variance is the inverse information plus a term for
    that sampling.

    Notation: :math:`N` the cohort's size, :math:`n` the subcohort's,
    :math:`D` the cases, :math:`d` the cases in the subcohort,
    :math:`\alpha = n / N` the sampling fraction; :math:`S_0, S_1` the
    (weighted) sums of :math:`e^{\beta' Z}` and :math:`Z e^{\beta' Z}`
    over the risk set at an event time and :math:`\bar Z = S_1 / S_0`;
    :math:`V` the inverse of the information of the estimating equation
    at the estimate, and :math:`L_i` unit :math:`i`'s score residual
    (its derivative of the score, R's ``residuals(type = "score")``)
    with :math:`D_i = L_i V` its dfbeta.

    * ``"prentice"`` (Prentice 1986; R's ``cch`` default): the risk set
      at a case's time is the subcohort at risk plus the case itself if
      it is from outside the subcohort,
      :math:`U(\beta) = \sum_{\text{cases}} (Z_i - \bar Z_{S \cup i})`,
      solved as a Cox fit in which a case from outside the subcohort
      enters just before its event (half the smallest gap between event
      times; ties by ``tie_method``). Its variance is Self-Prentice's
      below: the two have the same asymptotic variance (Self and
      Prentice 1988), and R's ``cch`` takes it, with its information and
      dfbetas, at the Self-Prentice estimate, as here.
    * ``"self-prentice"`` (Self and Prentice 1988): the risk set is the
      subcohort at risk alone,
      :math:`U(\beta) = \sum_{\text{cases}} (Z_i - \bar Z_S)`. Variance
      (Therneau and Li 1999, eq. 6; R's ``cch``):
      :math:`V + (1 - \alpha) \sum_{i \in S} D_i' D_i`, the dfbetas of
      the subcohort members as members of the risk sets.
    * ``"lin-ying"`` (Lin and Ying 1993): every case at risk while under
      follow-up, and each non-case of the subcohort counting
      :math:`w = (N - D) / (n - d)` times in the risk sums (the cohort's
      non-cases over the subcohort's). Variance (R's ``cch``):
      :math:`V + \big(1 - \tfrac{n - d}{N - D}\big) \sum_{i}
      (D_i - \bar D)'(D_i - \bar D)`, the sum over the subcohort's
      non-cases about their mean. With ``robust=True`` the first term is
      the design-based :math:`\sum_i D_i' D_i / w_i` over every unit
      (``cch(..., robust = TRUE)``).
    * ``"barlow"`` (Barlow 1994): a weighted Cox fit in which a subcohort
      member counts :math:`1 / \alpha` until its event, and every case
      counts 1 from just before its event (Therneau and Li 1999), with
      the robust (sandwich) variance summed over each unit's rows,
      :math:`\sum_{\text{units}} \big(\sum_{\text{rows}} w_r D_r\big)'
      \big(\sum_{\text{rows}} w_r D_r\big)`, R's ``coxph(..., weights,
      cluster = id)``.
    * ``"borgan-i"`` and ``"borgan-ii"`` (Borgan et al. 2000), for a
      subcohort drawn separately within strata (``sampling_strata``,
      with each stratum's size in ``cohort_size``): Self-Prentice and
      Lin-Ying with each stratum's own weights, :math:`N_s / n_s` for
      the subcohort (I) or :math:`(N_s - D_s) / (n_s - d_s)` for its
      non-cases (II). Variance :math:`V + V \Delta V`,
      :math:`\Delta = \sum_s (w_s - 1)\, M_s\, \widehat{\mathrm{Cov}}_s(L)`,
      :math:`\widehat{\mathrm{Cov}}_s(L)` the sample covariance
      (divisor :math:`m_s - 1`) of the unweighted score residuals of the
      :math:`m_s` units standing for others in stratum :math:`s` (the
      subcohort, I; its non-cases, II), and :math:`M_s` the units they
      stand for (:math:`N_s`; :math:`N_s - D_s`), as R's ``cch``. The
      model is not stratified: one baseline, the strata only in the
      sampling.

    With the whole cohort as the subcohort (:math:`\alpha = 1`) the
    sampling terms vanish and each is the ordinary Cox fit: Prentice's,
    Lin-Ying's and Barlow's coefficients are ``CoxPH.fit``'s with
    ``tie_method``, Self-Prentice's are its Breslow fit (its risk sets
    leave a tied case out, so ties are Breslow's), and the standard
    errors are the model-based ones, with two exceptions: Barlow's are
    the robust ones, and Prentice's are those of the Self-Prentice
    (Breslow) fit at its own estimate (``cch``'s convention above), not
    those of ``CoxPH.fit`` with Efron ties where there are ties.

    Parameters
    ----------
    x : array-like
        Each sampled unit's time: its event for a case, else when its
        follow-up ended.
    Z : array-like or DataFrame
        The covariates, one row per sampled unit (no missing values).
    subcohort : array-like of bool
        True for a unit drawn into the subcohort (a case or not), False
        for a case from outside it.
    cohort_size : int or dict
        The size of the cohort the subcohort was drawn from; for the
        Borgan methods a dict of each sampling stratum's size.
    c : array-like, optional
        0 for a case, 1 for right censored. Default all cases. A unit
        outside the subcohort must be a case.
    tl : array-like, optional
        Delayed entry times (left truncation), default 0.
    method : str, optional
        One of ``"prentice"`` (default), ``"self-prentice"``,
        ``"lin-ying"``, ``"barlow"``, ``"borgan-i"``, ``"borgan-ii"``.
    sampling_strata : array-like, optional
        For the Borgan methods, the stratum each unit's subcohort was
        drawn within.
    robust : bool, optional
        ``"lin-ying"`` only: the design-based first term (above).
    tie_method : str, optional
        ``"efron"`` (default, as R's ``cch``) or ``"breslow"``, for the
        estimators whose risk sets hold a tied case (Prentice, Lin-Ying,
        Barlow, Borgan II).
    tol : float, optional
        Newton-Raphson's tolerance, in standard errors of a step.

    Returns
    -------
    CaseCohortModel
        A Cox model: ``params`` the log hazard ratios, ``covariance()``
        the design-based covariance above (``naive_covariance`` the
        inverse information alone), ``summary()`` Wald intervals on it.
        Its baseline is Breslow's, each risk set weighted to stand for
        the cohort (a subcohort member :math:`1 / \alpha`, within its
        stratum for Borgan I, a case 1 at its own event; for Lin-Ying and
        Borgan II their own weights), so ``sf``, ``Hf`` and the other
        curves are offered; their pointwise bounds (``cb``), the
        likelihood-based quantities, residuals and diagnostics refuse,
        saying why.
    """
    if method not in CASE_COHORT_METHODS:
        raise ValueError(
            f"method={method!r} is not a case-cohort estimator: use one of "
            f"{', '.join(repr(m) for m in CASE_COHORT_METHODS)}."
        )
    if tie_method not in ("efron", "breslow"):
        raise ValueError(
            f"tie_method={tie_method!r}: a case-cohort fit takes 'efron' "
            "(the default) or 'breslow'."
        )
    if robust and method != "lin-ying":
        raise ValueError(
            "robust=True is Lin-Ying's design-based variance; the other "
            "methods' variances are already design-based. Drop robust, "
            "or use method='lin-ying'."
        )
    s = _Sample(
        x,
        Z,
        c,
        tl,
        subcohort,
        cohort_size,
        sampling_strata,
        method in _STRATIFIED,
    )
    _refuse_degenerate(s, method)
    p = s.Z.shape[1]
    mean = s.Z.mean(axis=0)
    what = f"The case-cohort fit (method={method!r})"

    if method in ("self-prentice", "prentice"):
        sp_rows = _rows(s, "self-prentice")
        sp = _likelihood(sp_rows, mean, tie_method)
        beta_sp, _, V, converged = _solve(sp, p, tol, what)
        dfbeta = sp.score_residuals(beta_sp) @ V
        sub_rows = sp_rows["at_risk"]
        alpha = s.n_sub[0] / s.N[0]
        phase2 = (1 - alpha) * dfbeta[sub_rows].T @ dfbeta[sub_rows]
        cov = V + phase2
        naive = V
        beta = beta_sp
        if method == "prentice":
            p_lik = _likelihood(_rows(s, "prentice"), mean, tie_method)
            beta, _, _, converged = _solve(p_lik, p, tol, what)
    elif method == "lin-ying":
        rows = _rows(s, "lin-ying")
        lik = _likelihood(rows, mean, tie_method)
        beta, _, V, converged = _solve(lik, p, tol, what)
        dfbeta = lik.score_residuals(beta) @ V
        non_case = rows["at_risk"]
        D0 = dfbeta[non_case] - dfbeta[non_case].mean(axis=0)
        fraction = (s.n_sub[0] - s.n_sub_cases[0]) / (s.N[0] - s.n_cases[0])
        phase2 = (1 - fraction) * D0.T @ D0
        naive = V
        if robust:
            cov = (dfbeta / rows["rho"][:, None]).T @ dfbeta + phase2
        else:
            cov = V + phase2
    elif method == "barlow":
        rows = _rows(s, "barlow")
        lik = _likelihood(rows, mean, tie_method)
        beta, _, V, converged = _solve(lik, p, tol, what)
        cov = robust_covariance(lik, beta, V, rows["unit"])
        naive = V
    else:  # the Borgan estimators
        kind = "self-prentice" if method == "borgan-i" else "lin-ying"
        rows = _rows(s, "borgan-i" if method == "borgan-i" else "borgan-ii")
        lik = _likelihood(rows, mean, tie_method)
        beta, _, V, converged = _solve(lik, p, tol, what)
        resid = lik.score_residuals(beta)
        if kind == "self-prentice":
            sampled = s.n_sub
            size = s.N
        else:
            sampled = s.n_sub - s.n_sub_cases
            size = s.N - s.n_cases
        weight = size / sampled
        delta = _phase_two_borgan(s, rows, resid, weight, size, sampled)
        cov = V + V @ delta @ V
        naive = V

    # The baseline: Breslow's, on risk sets weighted to stand for the
    # cohort (see Returns).
    base_kind = method if method in ("lin-ying", "borgan-ii") else "barlow"
    base_rows = _rows(s, base_kind)
    if method == "lin-ying":
        # As a weight, not a risk-only multiplier: the same risk sums,
        # and a case's weight is 1 either way
        base_rows = {**base_rows, "w": base_rows["rho"]}
        base_rows["rho"] = np.ones_like(base_rows["w"])
    base = _likelihood(base_rows, mean, "breslow")
    times, r, d, h0 = base.baseline(beta)
    shift = float(np.exp(-beta @ mean))
    model = CaseCohortModel("Cox", "Semi-Parametric")
    model.design = "case-cohort"
    model.method = method
    model.cohort_size = (
        dict(zip(s.levels.tolist(), s.N.tolist()))
        if method in _STRATIFIED
        else float(s.N[0])
    )
    model.subcohort_size = int(s.n_sub.sum())
    model.cases = int(s.n_cases.sum())
    model.tie_method = tie_method
    model.baseline_method = "breslow"
    model.beta = np.asarray(beta, dtype=float).copy()
    model.params = model.beta.copy()
    model._covariance = cov
    model.naive_covariance = naive
    se = np.sqrt(np.diag(cov))
    model._se = se
    from scipy.stats import norm

    model.p_values = 2 * norm.sf(np.abs(model.beta / se))
    model.maximum = "verified" if converged else "unverified"
    model.center = np.zeros(p)
    model._fit_center = mean
    model.x = times
    # The risk sums and the baseline of a unit at Z = 0
    model.r = r / shift
    model.d = d
    model.h0 = h0 * shift
    model.H0 = np.cumsum(model.h0)
    model.tl = s.tl
    model._fit_data = {
        "x": s.x,
        "c": s.c,
        "n": np.ones_like(s.x),
        "Z": s.Z,
        "tl": s.tl,
    }
    model.res = None
    return model


class _DesignRefusals:
    """What a fit to a sampled cohort cannot give, refused with why and
    what to use instead."""

    design: str = ""
    method: str = ""

    def _design_refusal(self, what: str, why: str, instead: str) -> None:
        raise ValueError(
            f"{what} is not available for a Cox fit to a {self.design} "
            f"sample: {why}. {instead}"
        )

    def compute_residuals(self, kind: str = "martingale") -> Any:
        self._design_refusal(
            "compute_residuals",
            "the sample is not the cohort, so the residuals of its rows "
            "do not describe the cohort's units",
            "Check the model on the full cohort's covariates if they can "
            "be had; the coefficients' covariance() is design-based.",
        )

    def diagnose(self, *args: Any, **kwargs: Any) -> Any:
        self._design_refusal(
            "diagnose",
            "its tests assume the rows are the whole risk sets",
            "Test proportional hazards by adding a covariate's "
            "interaction with time (time_interaction) to a fit of the "
            "full cohort, where one is available.",
        )

    def robust_covariance(self, cluster: Any = None) -> Any:
        self._design_refusal(
            "robust_covariance",
            "covariance() is already the design-based variance of the "
            "sampling",
            "Use covariance() (and naive_covariance for its first term).",
        )

    def robust_summary(self, cluster: Any = None) -> Any:
        self._design_refusal(
            "robust_summary",
            "summary() is already on the design-based variance",
            "Use summary().",
        )

    def concordance(self, *args: Any, **kwargs: Any) -> Any:
        self._design_refusal(
            "concordance",
            "a sample over-represents the cases, so its pairs are not "
            "the cohort's",
            "Compute the concordance on a random sample of the cohort, "
            "such as the subcohort.",
        )

    def to_dict(self) -> dict:
        self._design_refusal(
            "to_dict",
            "the saved form of a Cox model has no field for the design "
            "and its variance",
            "Pickle the model, or save params and covariance().",
        )
        return {}  # pragma: no cover


class CaseCohortModel(_DesignRefusals, SemiParametricRegressionModel):
    """A Cox model fitted to a case-cohort sample by
    ``CoxPH.fit_case_cohort``: a
    :class:`~surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel`
    whose covariance is the design-based one of its ``method``
    (``naive_covariance`` the inverse information alone), whose baseline
    is Breslow's on risk sets weighted to stand for the cohort, and which
    refuses what a pseudo-likelihood cannot give: the information
    criteria, the likelihood-ratio test, the residuals and diagnostics,
    and the pointwise bounds of the curves."""

    design = "case-cohort"
    #: The estimator, one of ``CASE_COHORT_METHODS``.
    method = "prentice"
    #: The cohort's size (a dict by stratum for the Borgan methods).
    cohort_size: Any = None
    subcohort_size: int = 0
    cases: int = 0
    #: The inverse information of the estimating equation: the variance
    #: before the term for the subcohort's sampling.
    naive_covariance: Any = None

    def neg_ll(self) -> float:
        self._design_refusal(
            "The log-likelihood",
            "the estimating equation is a pseudo-likelihood, whose value "
            "is not that of the data, and its information criteria and "
            "likelihood-ratio tests are not chi-squared",
            "Compare models by the Wald tests of summary().",
        )
        return 0.0  # pragma: no cover

    def _refuse_ic(self, name: str) -> None:
        self.neg_ll()

    def likelihood_ratio_test(self) -> dict:
        self.neg_ll()
        return {}  # pragma: no cover

    def cb(self, *args: Any, **kwargs: Any) -> Any:
        self._design_refusal(
            "cb (pointwise bounds on the curves)",
            "the variance of the weighted baseline under the sampling of "
            "the subcohort is not built",
            "Give the coefficients' bounds from summary(); for bounds on "
            "a curve, fit the full cohort where it can be had.",
        )

    def __repr__(self) -> str:
        text = super().__repr__()
        size = self.cohort_size
        cohort = (
            f"{sum(size.values()):g}"
            if isinstance(size, dict)
            else f"{size:g}"
        )
        line = (
            f"\nDesign              : case-cohort, method {self.method!r}: "
            f"subcohort {self.subcohort_size} of a cohort of {cohort}, "
            f"{self.cases} cases; design-based covariance"
        )
        head, sep, tail = text.partition("\nCoefficients")
        return head + line + sep + tail


class NestedCaseControlModel(_DesignRefusals, SemiParametricRegressionModel):
    """A Cox model fitted to a nested case-control sample by
    ``CoxPH.fit_nested_case_control``: the conditional logistic fit,
    a Cox model stratified by matched set at one time. It gives the
    hazard ratios with their model-based covariance (R's ``clogit``),
    the log-likelihood (a true conditional likelihood, so the
    likelihood-ratio test and information criteria compare such fits to
    the same sets), and refuses every curve: the sample does not record
    how many were at risk at each event time, which a baseline hazard
    needs."""

    design = "nested case-control"
    #: The number of matched sets fitted.
    sets: int = 0

    def _no_baseline(self, what: str) -> Any:
        self._design_refusal(
            what,
            "a nested case-control sample does not record how many units "
            "were at risk at each event time, so there is no baseline "
            "hazard",
            "Use the hazard ratios exp(params) and summary(); absolute "
            "risks need the cohort's risk-set sizes at the event times, "
            "or a case-cohort design (CoxPH.fit_case_cohort).",
        )

    def hf(self, *args: Any, **kwargs: Any) -> Any:
        return self._no_baseline("hf")

    def Hf(self, *args: Any, **kwargs: Any) -> Any:
        return self._no_baseline("Hf")

    def sf(self, *args: Any, **kwargs: Any) -> Any:
        return self._no_baseline("sf")

    def ff(self, *args: Any, **kwargs: Any) -> Any:
        return self._no_baseline("ff")

    def df(self, *args: Any, **kwargs: Any) -> Any:
        return self._no_baseline("df")

    def qf(self, *args: Any, **kwargs: Any) -> Any:
        return self._no_baseline("qf")

    def random(self, *args: Any, **kwargs: Any) -> Any:
        return self._no_baseline("random")

    def cb(self, *args: Any, **kwargs: Any) -> Any:
        return self._no_baseline("cb")

    def at(self, *args: Any, **kwargs: Any) -> Any:
        return self._no_baseline("at")

    def plot_partial_effects(self, *args: Any, **kwargs: Any) -> Any:
        return self._no_baseline("plot_partial_effects")

    def __repr__(self) -> str:
        text = super().__repr__()
        line = (
            f"\nDesign              : nested case-control, {self.sets} "
            "matched sets; conditional logistic likelihood"
        )
        head, sep, tail = text.partition("\nCoefficients")
        return head + line + sep + tail


def fit_nested_case_control(
    fitter: Any,
    Z: Any,
    case: Any,
    sets: Any,
    tie_method: str = "exact",
    tol: float = 1e-10,
) -> NestedCaseControlModel:
    r"""
    A Cox model fitted to a nested case-control sample.

    In a nested case-control study, each time a unit of the cohort has
    the event, a few controls are drawn at random from the units still
    at risk then; the covariates are measured only on the cases and
    their controls. Each matched set (the case, or the cases tied at
    that time, and its controls) is a small sample of that risk set, so
    the hazard ratios are estimated by comparing each case with its own
    set: the conditional logistic likelihood,

    .. math::
        L(\beta) = \prod_{\text{sets } k}
        \frac{\exp(\beta' \sum_{i \in D_k} Z_i)}
             {\sum_{s \subseteq R_k,\, |s| = d_k}
              \exp(\beta' \sum_{i \in s} Z_i)},

    :math:`D_k` the :math:`d_k` cases of set :math:`k` and :math:`R_k` the
    whole set (Breslow et al. 1978; Goldstein and Langholz 1992). With
    one case per set it is :math:`\exp(\beta' Z_{\text{case}}) /
    \sum_{i \in R_k} \exp(\beta' Z_i)`, Cox's partial likelihood on the
    sampled risk set. It is fitted as a Cox model stratified by set with
    every unit at one time (``tie_method="exact"``, the default, the
    exact conditional likelihood above, R's ``clogit``; ``"efron"`` and
    ``"breslow"`` its approximations, which differ from it only for sets
    with more than one case).

    Parameters
    ----------
    Z : array-like or DataFrame
        The covariates, one row per sampled unit.
    case : array-like of bool
        True (1) for a case, False (0) for a control.
    sets : array-like
        The matched set of each row.
    tie_method : str, optional
        ``"exact"`` (default), ``"efron"`` or ``"breslow"``.
    tol : float, optional
        The convergence tolerance.

    Returns
    -------
    NestedCaseControlModel
        ``params`` the log hazard ratios, ``covariance()`` the inverse
        information, ``summary()``, ``log_likelihood`` and
        ``likelihood_ratio_test()``. The curves refuse: the design gives
        no baseline hazard.
    """
    names = {"exact": "kalbfleisch-prentice", "efron": "efron"}
    names["breslow"] = "breslow"
    if tie_method not in names:
        raise ValueError(
            f"tie_method={tie_method!r}: use 'exact' (the conditional "
            "likelihood, the default), 'efron' or 'breslow'."
        )
    case = np.asarray(case)
    if case.ndim != 1 or not np.all(np.isin(case, (0, 1))):
        raise ValueError("case must be True/False (or 1/0), one per row.")
    case = case.astype(bool)
    labels = np.asarray(sets)
    if labels.shape != case.shape:
        raise ValueError(
            f"sets has {labels.shape} labels for {case.shape[0]} rows: give "
            "the matched set of every row."
        )
    _, code = np.unique(labels, return_inverse=True)
    n_sets = int(code.max()) + 1
    cases_in = np.bincount(code, weights=case, minlength=n_sets)
    size = np.bincount(code, minlength=n_sets)
    idle = int(np.sum((cases_in == 0) | (cases_in == size)))
    if idle:
        warnings.warn(
            f"{idle} of {n_sets} matched sets have no case or no control, "
            "so they say nothing about the hazard ratios and are left "
            "out. Check case and sets if that is unexpected.",
            UserWarning,
            stacklevel=caller_stacklevel(),
        )
    keep = ((cases_in > 0) & (cases_in < size))[code]
    if not np.any(keep):
        raise ValueError(
            "No matched set has both a case and a control: nothing to fit."
        )
    Z_arr = np.asarray(Z, dtype=float)
    Z_arr = Z_arr.reshape(-1, 1) if Z_arr.ndim == 1 else Z_arr
    if Z_arr.shape[0] != case.shape[0]:
        raise ValueError(
            f"Z has {Z_arr.shape[0]} rows for {case.shape[0]} values of "
            "case: give one row of covariates per sampled unit."
        )
    if not np.all(np.isfinite(Z_arr)):
        bad = int(np.sum(~np.all(np.isfinite(Z_arr), axis=1)))
        raise ValueError(
            f"Z has missing or infinite values in {bad} row(s): dropping a "
            "row could leave its matched set without its case or its only "
            "control, so a nested case-control fit needs complete rows. "
            "Drop those sets, or impute the values, before fitting."
        )
    rows = int(keep.sum())
    model = fitter.fit(
        np.ones(rows),
        Z_arr[keep],
        c=np.where(case[keep], 0, 1),
        strata=code[keep],
        tie_method=names[tie_method],
        tol=tol,
    )
    out = NestedCaseControlModel("Cox", "Semi-Parametric")
    out.__dict__.update(model.__dict__)
    out.sets = n_sets - idle
    out.tie_method = tie_method
    return out
