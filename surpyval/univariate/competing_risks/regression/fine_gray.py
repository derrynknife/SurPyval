"""
This code was created for and sponsored by Cartiga (www.cartiga.com).
Cartiga makes no representations or warranties in connection with the code
and waives any and all liability in connection therewith. Your use of the
code constitutes acceptance of these terms.

Copyright 2022 Cartiga LLC

Fine-Gray subdistribution-hazard regression for competing risks.

Where cause-specific proportional hazards models the hazard of each cause
after removing subjects who fail from a competing cause, the Fine-Gray model
(Fine & Gray, 1999) keeps those subjects in a modified ("subdistribution")
risk set so that a single coefficient vector acts directly on the cumulative
incidence function (CIF) of the cause of interest,

.. math::
    F_k(t \\mid Z) = 1 - \\exp\\{-\\Lambda_{k0}(t)\\,\\exp(\\beta' Z)\\},

with :math:`\\Lambda_{k0}` a baseline cumulative subdistribution hazard. This
makes :math:`\\beta` interpretable as a (log) subdistribution hazard ratio: a
positive coefficient raises the incidence of cause :math:`k`.

Independent right-censoring is handled by inverse-probability-of-censoring
weighting (IPCW): a subject who has already failed from a competing cause
stays in the subdistribution risk set with a time-varying weight
:math:`G(t)/G(x_i)`, where :math:`G` is the Kaplan-Meier estimate of the
censoring-time survival function. Subjects who are censored, or who have
already had the event of interest, leave the risk set. The partial likelihood
is the Breslow form of this weighted risk set.
"""

import numpy as np
from autograd import grad, hessian
from autograd import numpy as anp
from scipy.optimize import minimize
from scipy.stats import norm

from surpyval.utils import validate_fine_gray_inputs


def _censoring_survival(x, c, n):
    """
    Kaplan-Meier estimate of the censoring survival ``G(t) = P(C > t)``.

    The roles are reversed relative to an ordinary survival fit: right-censored
    rows (``c == 1``) are the "events" for the censoring distribution and
    observed events (``c == 0``) are treated as censored. Returns the sorted
    unique times and the right-continuous ``G`` evaluated at each.
    """
    times = np.unique(x)
    G = np.ones(times.shape[0])
    surv = 1.0
    for k, t in enumerate(times):
        at_risk = n[x >= t].sum()
        censored = n[(x == t) & (c == 1)].sum()
        if at_risk > 0:
            surv = surv * (1.0 - censored / at_risk)
        G[k] = surv
    return times, G


def _step(times, values, query, before):
    """
    Right-continuous step function: the value carried by the largest ``times``
    entry ``<= query``; ``before`` is returned where ``query`` precedes the
    first time.
    """
    idx = np.searchsorted(times, query, side="right") - 1
    out = np.where(
        idx < 0, before, values[np.clip(idx, 0, len(values) - 1)]
    )
    return out


def _fit_cause(x, Z, e, c, n, cause):
    """
    Fit the Fine-Gray subdistribution-hazard model for a single ``cause``.

    Returns a dict with the fitted coefficients, their standard errors and
    p-values, the baseline cumulative subdistribution hazard (as sorted event
    times and the cumulative hazard at each), and the optimiser result.
    """
    is_event = (c == 0) & (e == cause)
    if not is_event.any():
        raise ValueError(f"No observed events for cause {cause!r}")
    is_competing = (c == 0) & (e != cause)

    # Censoring-survival for the IPCW weights.
    g_times, g_vals = _censoring_survival(x, c, n)
    # G is >= its last positive value; guard the ratio against division by a
    # zero tail (times beyond the last censoring-KM step).
    g_floor = g_vals[g_vals > 0].min() if np.any(g_vals > 0) else 1.0
    G_x = np.maximum(_step(g_times, g_vals, x, before=1.0), g_floor)

    event_times = x[is_event]
    G_t = _step(g_times, g_vals, event_times, before=1.0)

    # Subdistribution risk-set weight matrix W (n_events x N), independent of
    # beta: 1 for the ordinary risk set (x_i >= t_j); G(t_j)/G(x_i) for a
    # subject who already failed from a competing cause (x_i < t_j); 0 for a
    # censored subject or one that already had the event of interest.
    at_risk = x[None, :] >= event_times[:, None]
    already_competing = is_competing[None, :] & (
        x[None, :] < event_times[:, None]
    )
    W = at_risk.astype(float) + already_competing * (
        G_t[:, None] / G_x[None, :]
    )

    n_event = n[is_event]
    Z_event = Z[is_event]

    def neg_ll(beta):
        eta = anp.dot(Z, beta)
        weighted_exp = n * anp.exp(eta)
        denom = anp.dot(W, weighted_exp)
        eta_event = anp.dot(Z_event, beta)
        ll = anp.sum(n_event * eta_event) - anp.sum(
            n_event * anp.log(denom)
        )
        return -ll

    beta0 = np.zeros(Z.shape[1])
    res = minimize(neg_ll, beta0, jac=grad(neg_ll), method="BFGS")
    beta = res.x

    # Standard errors from the inverse observed information.
    H = hessian(neg_ll)(beta)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    var = np.diag(cov)
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.where(var > 0, var, np.nan))
        z_score = beta / se
    p_values = 2.0 * (1.0 - norm.cdf(np.abs(z_score)))

    # Breslow baseline cumulative subdistribution hazard: at each event-of-
    # interest time, dLambda0 = (events there) / (weighted risk set there).
    denom = W @ (n * np.exp(Z @ beta))
    order = np.argsort(event_times, kind="mergesort")
    t_sorted = event_times[order]
    d_over_r = (n_event / denom)[order]
    uniq_t, inv = np.unique(t_sorted, return_inverse=True)
    dL = np.zeros(uniq_t.shape[0])
    np.add.at(dL, inv, d_over_r)
    baseline_cumhaz = np.cumsum(dL)

    return {
        "cause": cause,
        "beta": beta,
        "se": se,
        "p_values": p_values,
        "cov": cov,
        "baseline_times": uniq_t,
        "baseline_cumhaz": baseline_cumhaz,
        "neg_ll": float(res.fun),
        "res": res,
    }


class FineGrayModel:
    """
    A fitted Fine-Gray subdistribution-hazard model for one cause of interest.

    The natural prediction is the cumulative incidence function :meth:`cif`;
    ``coefficients``/``se``/``p_values`` describe the (log) subdistribution
    hazard ratios.
    """

    def __init__(self, fit):
        self.cause = fit["cause"]
        self.coefficients = fit["beta"]
        self.beta = fit["beta"]
        self.se = fit["se"]
        self.p_values = fit["p_values"]
        self.cov = fit["cov"]
        self._times = fit["baseline_times"]
        self._cumhaz = fit["baseline_cumhaz"]
        self._neg_ll = fit["neg_ll"]
        self.res = fit["res"]

    def phi(self, Z):
        return np.exp(np.asarray(Z, dtype=float) @ self.beta)

    def cif(self, x, Z):
        """
        Cumulative incidence of the cause of interest at times ``x`` for a
        single covariate vector ``Z``: ``1 - exp(-Lambda0(x) * exp(beta'Z))``.
        The CIF is flat before the first event time and after the last (the
        baseline is a step function estimated only on the observed range).
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.asarray(Z, dtype=float).ravel()
        H0 = _step(self._times, self._cumhaz, x, before=0.0)
        return 1.0 - np.exp(-H0 * np.exp(Z @ self.beta))

    def sf(self, x, Z):
        """One minus the cumulative incidence (the cause-of-interest-free
        probability under the subdistribution)."""
        return 1.0 - self.cif(x, Z)

    def __repr__(self):
        lines = [
            "Fine-Gray Subdistribution Hazard Model",
            "======================================",
            f"Cause of interest   : {self.cause}",
            "Coefficients (beta'Z acts on the subdistribution hazard):",
        ]
        for i, (b, s, p) in enumerate(
            zip(self.beta, self.se, self.p_values)
        ):
            lines.append(
                f"   beta_{i}  :  {b: .6f}  (se {s:.6f}, p {p:.4f})"
            )
        return "\n".join(lines)


class FineGray_:
    def fit(self, x, Z, e, c=None, n=None, cause=None):
        """
        Fit the Fine-Gray model for a cause of interest.

        Parameters
        ----------
        x : array_like
            Observed times.
        Z : ndarray
            Covariate matrix, one row per observation.
        e : array_like
            Event-type (cause) labels; ``None`` for a censored observation.
        c : array_like, optional
            Censoring flags (0 observed, 1 right-censored). Defaults to all
            observed. Left/interval censoring is not supported.
        n : array_like, optional
            Counts per observation. Defaults to 1.
        cause : optional
            The cause of interest. May be omitted only when the data contains a
            single event type.

        Returns
        -------
        FineGrayModel
            The fitted model, with :meth:`~FineGrayModel.cif` prediction.
        """
        x, Z, e, c, n = validate_fine_gray_inputs(x, Z, e, c, n)

        causes = sorted(
            {ei for ei in e if ei is not None}, key=lambda v: str(v)
        )
        if cause is None:
            if len(causes) != 1:
                raise ValueError(
                    "Data has multiple event types "
                    f"({causes}); specify `cause`."
                )
            cause = causes[0]
        elif cause not in causes:
            raise ValueError(
                f"Cause {cause!r} not observed; causes are {causes}."
            )

        return FineGrayModel(_fit_cause(x, Z, e, c, n, cause))


FineGray = FineGray_()
