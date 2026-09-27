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
:math:`G(t-)/G(x_i-)`, where :math:`G` is the Kaplan-Meier estimate of the
censoring-time survival function. Subjects who are censored, or who have
already had the event of interest, leave the risk set. The partial likelihood
is the Breslow form of this weighted risk set.

:math:`G` is evaluated just before each time, as R's ``cmprsk::crr`` does
(its ``uuu`` is the censoring Kaplan-Meier at ``ftime-``): an event and a
censoring at the same instant are ordered event first, so the censorings at
:math:`t` do not yet reduce the weight at :math:`t`, nor count against a
competing failure at :math:`x_i`. :math:`G` itself is ``survfit``'s reverse
Kaplan-Meier, as in ``crr``. With no censoring time equal to an event time
the left limits equal :math:`G(t)` and :math:`G(x_i)`.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from autograd import grad, hessian
from autograd import numpy as anp
from scipy.optimize import minimize
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.competing_risks.labels import (
    label_from_native,
    label_mask,
    ordered_labels,
)
from surpyval.utils import validate_fine_gray_inputs
from surpyval.utils.ipcw import censoring_survival, step_at, step_left_limit
from surpyval.utils.linalg import safe_inv


def _fit_cause(
    x: npt.NDArray,
    Z: npt.NDArray,
    e: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    cause: Any,
) -> dict:
    """
    Fit the Fine-Gray subdistribution-hazard model for a single ``cause``.

    Returns a dict with the fitted coefficients, their standard errors and
    p-values, the baseline cumulative subdistribution hazard (as sorted event
    times and the cumulative hazard at each), and the optimiser result.
    """
    is_cause = label_mask(e, cause)
    is_event = (c == 0) & is_cause
    if not is_event.any():
        raise ValueError(f"No observed events for cause {cause!r}")
    is_competing = (c == 0) & ~is_cause

    # Censoring survival for the IPCW weights, taken just before each time,
    # G(t-), as cmprsk::crr does: at a time shared by events and censorings
    # the events come first, so the censorings there must not yet thin the
    # weights (evaluating G(t) counted them against the events they tie with).
    # G(x_i-) > 0 for every row: row i (a positive count) is at risk and
    # uncensored at every earlier censoring time, so no step before x_i can
    # reach zero and the ratio below needs no guard.
    g_times, g_vals = censoring_survival(x, c == 1, n)
    G_x = step_left_limit(g_times, g_vals, x, before=1.0)

    event_times = x[is_event]
    G_t = step_left_limit(g_times, g_vals, event_times, before=1.0)

    # Subdistribution risk-set weight matrix W (n_events x N), independent of
    # beta: 1 for the ordinary risk set (x_i >= t_j); G(t_j-)/G(x_i-) for a
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

    def neg_ll(beta: Any) -> Any:
        eta = anp.dot(Z, beta)
        weighted_exp = n * anp.exp(eta)
        denom = anp.dot(W, weighted_exp)
        eta_event = anp.dot(Z_event, beta)
        ll = anp.sum(n_event * eta_event) - anp.sum(n_event * anp.log(denom))
        return -ll

    beta0 = np.zeros(Z.shape[1])
    res = minimize(neg_ll, beta0, jac=grad(neg_ll), method="BFGS")
    beta = res.x

    # Standard errors from the inverse observed information.
    H = hessian(neg_ll)(beta)
    cov = safe_inv(H)
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


def paired_covariate_rows(Z: npt.ArrayLike, n_x: int, p: int) -> npt.NDArray:
    """
    The covariate row for each of ``n_x`` query times, shape ``(n_x, p)``.

    ``Z`` is one covariate vector (a scalar for one covariate, a 1-D array
    of ``p`` values or a single row), used at every time, or one row per
    time, paired with the times in order. Any other shape is refused: a
    ``Z`` of several rows used to be flattened into one long vector (a
    shape error from the matrix product) or broadcast against the times.
    """
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.ndim <= 1:
        Z_arr = Z_arr.reshape(1, -1)
    if Z_arr.ndim != 2 or Z_arr.shape[1] != p:
        raise ValueError(
            "Z must hold {} covariate(s) per row, got shape {}.".format(
                p, np.shape(Z)
            )
        )
    if Z_arr.shape[0] not in (1, n_x):
        raise ValueError(
            "Z has {} rows for {} times: give one covariate row, used at "
            "every time, or one row per time.".format(Z_arr.shape[0], n_x)
        )
    return np.broadcast_to(Z_arr, (n_x, p))


class FineGrayModel(SerialisableMixin):
    """
    A fitted Fine-Gray subdistribution-hazard model for one cause of interest.

    The natural prediction is the cumulative incidence function :meth:`cif`;
    ``coefficients``/``se``/``p_values`` describe the (log) subdistribution
    hazard ratios.
    """

    def __init__(self, fit: dict) -> None:
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

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted Fine-Gray model to a plain, JSON-serialisable
        dict.

        Stores the coefficients and their covariance, plus the fitted
        subdistribution baseline cumulative-hazard step arrays, so the reloaded
        model reproduces ``cif``/``sf`` exactly and can still report the
        coefficient summary. The optimiser objects are not stored.
        """
        return stamp_schema(
            {
                "model": "FineGrayModel",
                # native type: a numpy scalar label breaks JSON/BSON
                "cause": to_native(self.cause),
                "beta": np.asarray(self.beta, dtype=float).tolist(),
                "se": np.asarray(self.se, dtype=float).tolist(),
                "p_values": np.asarray(self.p_values, dtype=float).tolist(),
                "cov": np.asarray(self.cov, dtype=float).tolist(),
                "baseline_times": np.asarray(
                    self._times, dtype=float
                ).tolist(),
                "baseline_cumhaz": np.asarray(
                    self._cumhaz, dtype=float
                ).tolist(),
                "neg_ll": float(self._neg_ll),
            }
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> "FineGrayModel":
        """Rebuild a Fine-Gray model from a :meth:`to_dict` dictionary."""
        require_model_tag(model_dict, "FineGrayModel", "a Fine-Gray model")
        return cls(
            {
                "cause": label_from_native(model_dict["cause"]),
                "beta": np.array(model_dict["beta"], dtype=float),
                "se": np.array(model_dict["se"], dtype=float),
                "p_values": np.array(model_dict["p_values"], dtype=float),
                "cov": np.array(model_dict["cov"], dtype=float),
                "baseline_times": np.array(
                    model_dict["baseline_times"], dtype=float
                ),
                "baseline_cumhaz": np.array(
                    model_dict["baseline_cumhaz"], dtype=float
                ),
                "neg_ll": model_dict["neg_ll"],
                "res": None,
            }
        )

    def phi(self, Z: npt.ArrayLike) -> npt.NDArray:
        """The subdistribution hazard multiplier :math:`e^{\\beta' Z}`, one
        value per row of ``Z`` (a scalar for a single covariate vector)."""
        return np.exp(np.asarray(Z, dtype=float) @ self.beta)

    def cif(self, x: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """
        Cumulative incidence of the cause of interest at times ``x``:
        ``1 - exp(-Lambda0(x) * exp(beta'Z))``. ``Z`` is one covariate
        vector (a 1-D array or a single row), used at every time, or one
        row per time in ``x`` (row ``i`` with ``x[i]``). The CIF is flat
        before the first event time and after the last (the baseline is a
        step function estimated only on the observed range).
        """
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        rows = paired_covariate_rows(Z, x.size, np.size(self.beta))
        H0 = step_at(self._times, self._cumhaz, x, before=0.0)
        return 1.0 - np.exp(-H0 * np.exp(rows @ self.beta))

    def sf(self, x: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """One minus the cumulative incidence (the cause-of-interest-free
        probability under the subdistribution)."""
        return 1.0 - self.cif(x, Z)

    def __repr__(self) -> str:
        lines = [
            "Fine-Gray Subdistribution Hazard Model",
            "======================================",
            f"Cause of interest   : {self.cause}",
            "Coefficients (beta'Z acts on the subdistribution hazard):",
        ]
        for i, (b, s, p) in enumerate(zip(self.beta, self.se, self.p_values)):
            lines.append(f"   beta_{i}  :  {b: .6f}  (se {s:.6f}, p {p:.4f})")
        return "\n".join(lines)


class FineGray_:
    """
    The Fine-Gray subdistribution-hazards regression for one cause of a
    competing-risks problem: the covariates act proportionally on the
    *subdistribution* hazard of the cause of interest, so a coefficient
    describes its effect on that cause's cumulative incidence directly,

    .. math::
        F_k(t \\mid Z) = 1 - \\exp\\left(-\\Lambda_{k0}(t)\\,
        e^{\\beta' Z}\\right).

    Estimated by inverse-probability-of-censoring weighting (IPCW), with one
    Kaplan-Meier censoring distribution for the whole sample, so censoring
    is assumed not to depend on the covariates. A subject that failed from
    a competing cause at :math:`x_i` keeps the weight
    :math:`\\hat G(t-)/\\hat G(x_i-)` at a later event time :math:`t`: the
    censoring survival is taken just before each time, so a censoring tied
    with an event counts after it, as in R's ``cmprsk::crr``. ``FineGray``
    (from ``surpyval.univariate.competing_risks``) is an instance of this
    class; its ``fit`` returns a
    :class:`~surpyval.univariate.competing_risks.regression.fine_gray.FineGrayModel`.
    """

    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        e: npt.ArrayLike,
        c: "npt.ArrayLike | None" = None,
        n: "npt.ArrayLike | None" = None,
        cause: Any = None,
    ) -> FineGrayModel:
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
            Censoring flags (0 observed, 1 right-censored). Defaults to
            deriving them from ``e``: a missing event (``None``/``NaN``) is
            right-censored, any other is observed. Left/interval censoring is
            not supported.
        n : array_like, optional
            Counts per observation. Defaults to 1.
        cause : optional
            The cause of interest. May be omitted only when the data contains a
            single event type.

        Returns
        -------
        FineGrayModel
            The fitted model, with :meth:`~FineGrayModel.cif` prediction.

        Examples
        --------
        >>> from surpyval.univariate.competing_risks import FineGray
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * Z[:, 0])))
        >>> t_b = rng.exponential(1 / 0.05, 200)
        >>> t_c = rng.uniform(0, 20, 200)  # censoring times
        >>> x = np.minimum(np.minimum(t_a, t_b), t_c).round(3)
        >>> first = np.where(t_a < t_b, "a", "b")
        >>> e = np.where(t_c < np.minimum(t_a, t_b), None, first)
        >>> model = FineGray.fit(x, Z, e, cause="a")
        >>> model.beta.round(3)
        array([0.908])
        >>> model.cif([5, 10], [[1]]).round(4)
        array([0.5808, 0.7395])
        """
        x, Z, e, c, n = validate_fine_gray_inputs(x, Z, e, c, n)

        causes = ordered_labels(e)
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
