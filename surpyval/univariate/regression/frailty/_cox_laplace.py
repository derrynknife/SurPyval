"""
``CoxFrailty.fit(..., family="inverse_gaussian")`` and
``family="positive_stable"``: the shared frailty Cox model fitted by EM on
the full likelihood, with the baseline hazard's jumps at the event times as
parameters (the nonparametric maximum likelihood estimate; Nielsen, Gill,
Andersen and Sorensen 1992; Klein 1992).

For a given ``theta``:

* E-step: each group's posterior mean frailty ``I(D_g + 1, A_g) / I(D_g,
  A_g)`` (``families.py``), with ``D_g`` its events and ``A_g`` the sum
  over its rows of ``w exp(beta'Z) H_0(x)``;
* M-step: the Cox partial likelihood with the log-frailties as offsets
  (Breslow's ties) for ``beta``, then Breslow's baseline with the
  frailties as weights.

The expected complete-data log-likelihood depends on the frailties only
through their posterior means (it is linear in ``u``; the ``log u`` terms
do not involve ``beta`` or the baseline), so these two steps are EM's for
any frailty family. Its fixed point maximises the marginal (frailty
integrated out) log-likelihood

.. math::
    \\ell(\\beta, h, \\theta) = \\sum_k d_k \\log h_k
    + \\sum_{\\text{events}} w \\beta' Z + \\sum_g \\log I(D_g, A_g)

over ``beta`` and the jumps ``h_k`` for that ``theta``, and ``theta``
maximises the profile of it. At ``theta = 0`` it is the Breslow full
likelihood of the Cox model, whose maximum over the jumps is the Breslow
partial likelihood up to the constant ``sum_k d_k (log d_k - 1)``.

The covariance of ``(beta, theta)`` is the inverse of the profile
information: the full observed information in ``beta``, ``theta`` and
every jump, analytically (:func:`_information`), with the jumps' block
eliminated by its Schur complement.
"""

import warnings
from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import elementwise_grad, grad, jacobian
from scipy.optimize import minimize_scalar

from surpyval.utils.data_summary import data_summary
from surpyval.utils.no_maximum import warn_no_maximum, warn_unverified
from surpyval.utils.warnings import caller_stacklevel

from .._aliasing import expand
from .._fit_skeleton import covariate_center
from ..proportional_hazards.cox_likelihood import baseline_at_origin
from .families import laplace_log_integral, posterior_mean, whole_counts

# The search for theta, on its log, as for the gamma frailty: from 1e-6
# (no detectable frailty) to 100 for the inverse Gaussian's variance, and
# to 0.999 for the positive stable's 1 - alpha, which is below 1.
_LOG_THETA_BOUNDS = {
    "inverse_gaussian": (np.log(1e-6), np.log(100.0)),
    "positive_stable": (np.log(1e-6), np.log(0.999)),
}
_EM_TOL = 1e-12


def _em_class() -> Any:
    from .cox_frailty import _CoxFrailtyEM

    class _LaplaceEM(_CoxFrailtyEM):
        """EM for a family of ``families.LAPLACE_FAMILIES``: the gamma
        fit's M-step (Breslow's ties), its own E-step, and the marginal
        likelihood in place of the gamma's integrated one."""

        def __init__(self, *args: Any, family: str) -> None:
            super().__init__(*args)
            self.family = family
            # (Newton on the gamma's penalised partial likelihood is not
            # this family's)
            self.use_newton = False

        def update(
            self, theta: float, log_u: npt.NDArray, beta: npt.NDArray
        ) -> tuple[npt.NDArray, npt.NDArray]:
            beta, _ = self.m_step(beta, log_u[self.inv])
            A = self.group_hazard(beta, log_u[self.inv])
            with np.errstate(over="ignore", invalid="ignore"):
                new = np.log(posterior_mean(self.family, self.D, A, theta))
            # A group never at risk at an event time (A = 0) enters no
            # risk set, and its posterior is its prior (whose mean the
            # positive stable does not have): its offset is not used.
            return np.where(A > 0, new, 0.0), beta

        def marginal(
            self, theta: float, beta: npt.NDArray, log_u: npt.NDArray
        ) -> float:
            """The marginal log-likelihood at ``theta``, ``beta`` and the
            jumps of Breslow's baseline with the frailties ``log u`` as
            weights (the M-step's, so at EM's fixed point the maximum
            over the jumps)."""
            times, _, d, h0 = self.baseline(beta, log_u[self.inv])
            A = self.group_hazard(beta, log_u[self.inv])
            event = self.c == 0
            with np.errstate(divide="ignore"):
                jumps = float(np.sum(d[d > 0] * np.log(h0[d > 0])))
            linear = float(np.sum((self.w * (self.Z @ beta))[event]))
            groups = laplace_log_integral(self.family, self.D, A, theta)
            return jumps + linear + float(np.sum(groups))

        def profile(self, log_theta: float) -> float:
            theta = float(np.exp(log_theta))
            beta, log_u, _ = self.em(theta)
            return -self.marginal(theta, beta, log_u)

    return _LaplaceEM


def _psi_derivatives(
    family: str, D: npt.NDArray, A: npt.NDArray, theta: float
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, float]:
    """The derivatives of each group's ``psi = log I(D, A)``: in ``A``
    (``-E[u | data]``), twice in ``A`` (``Var(u | data)``), in ``A`` and
    ``theta``, and the sum's second derivative in ``theta``; by autograd
    through the closed forms."""
    if theta <= 0:
        # psi = -A
        return -np.ones(A.shape), np.zeros(A.shape), np.zeros(A.shape), 0.0

    def psi(a: Any, t: Any) -> Any:
        return laplace_log_integral(family, D, a, t)

    d_a = elementwise_grad(psi, 0)
    psi_a = np.asarray(d_a(A, theta), dtype=float)
    psi_aa = np.asarray(
        elementwise_grad(lambda a: d_a(a, theta))(A), dtype=float
    )
    psi_at = np.asarray(jacobian(lambda t: d_a(A, t))(theta), dtype=float)
    psi_tt = float(
        grad(grad(lambda t: anp.sum(laplace_log_integral(family, D, A, t))))(
            theta
        )
    )
    return psi_a, psi_aa, psi_at, psi_tt


def _information(
    em: Any,
    family: str,
    theta: float,
    beta: npt.NDArray,
    log_u: npt.NDArray,
    with_theta: bool,
) -> "npt.NDArray | None":
    """
    The covariance of ``beta`` (and ``theta``, with ``with_theta``): the
    inverse of the observed information of the marginal log-likelihood in
    ``(beta, theta)`` with the baseline's jumps ``h`` profiled out (the
    Schur complement of their block).

    With ``R_gk`` the risk weight ``sum w exp(beta'Z)`` of group ``g``'s
    rows still at risk at the ``k``-th event time, ``B_g`` the sum of ``w
    exp(beta'Z) H_0(x) Z`` over its rows and ``psi`` each group's
    ``log I(D_g, A_g)``, the Hessian's blocks are

    * ``h h``: ``-diag(d_k / h_k^2) + R' diag(psi_AA) R``;
    * ``h beta``: ``R' diag(psi_AA) B + T``, ``T_k`` the sum over the rows
      at risk at ``t_k`` of ``psi_A w exp(beta'Z) Z``;
    * ``beta beta``: ``B' diag(psi_AA) B`` plus the sum over the rows of
      ``psi_A w exp(beta'Z) H_0(x) Z Z'``;
    * ``h theta``: ``R' psi_A theta``; ``beta theta``: ``B' psi_A theta``;
      ``theta theta``: the sum of ``psi_theta theta``.

    ``None`` where the information is singular.
    """
    x, w, inv, Z = em.x, em.w, em.inv, em.Z
    G, p = em.G, em.p
    times, _, d, h0 = em.baseline(beta, log_u[inv])
    at = d > 0
    t, dk, h = times[at], d[at], h0[at]
    K = t.size
    kidx = np.searchsorted(t, x, side="right") - 1
    ok = kidx >= 0
    risk = w * np.exp(Z @ beta)
    H0x = np.where(ok, np.cumsum(h)[np.maximum(kidx, 0)], 0.0)
    A = np.bincount(inv, weights=risk * H0x, minlength=G)
    psi_a, psi_aa, psi_at, psi_tt = _psi_derivatives(family, em.D, A, theta)

    def reverse_cumsum(M: npt.NDArray) -> npt.NDArray:
        return np.cumsum(M[::-1], axis=0)[::-1]

    # R (G x K): row j adds its risk weight at every event time up to x_j
    M = np.zeros((K, G))
    np.add.at(M, (kidx[ok], inv[ok]), risk[ok])
    R = reverse_cumsum(M).T
    B = (
        np.column_stack(
            [
                np.bincount(inv, weights=risk * H0x * Z[:, j], minlength=G)
                for j in range(p)
            ]
        )
        if p
        else np.zeros((G, 0))
    )
    V = (psi_a[inv] * risk)[:, None] * Z
    Tb = np.zeros((K, p))
    np.add.at(Tb, kidx[ok], V[ok])
    T = reverse_cumsum(Tb)
    C = (Z * (psi_a[inv] * risk * H0x)[:, None]).T @ Z

    H_hh = R.T @ (psi_aa[:, None] * R)
    H_hh[np.diag_indices(K)] -= dk / h**2
    H_hb = R.T @ (psi_aa[:, None] * B) + T
    H_bb = B.T @ (psi_aa[:, None] * B) + C
    if with_theta:
        H_he = np.column_stack([H_hb, R.T @ psi_at])
        H_ee = np.block(
            [
                [H_bb, (B.T @ psi_at)[:, None]],
                [(B.T @ psi_at)[None, :], np.array([[psi_tt]])],
            ]
        )
    else:
        H_he, H_ee = H_hb, H_bb
    with np.errstate(all="ignore"):
        try:
            X = np.linalg.solve(-H_hh, -H_he)
            schur = -H_ee - (-H_he).T @ X
            schur = (schur + schur.T) / 2.0
            cov = np.linalg.inv(schur)
        except np.linalg.LinAlgError:
            return None
    if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) <= 0):
        return None
    return cov


def _check_ties(
    c: npt.NDArray,
    w: npt.NDArray,
    x: npt.NDArray,
    family: str,
    tie_method: str,
) -> None:
    """Efron's ties are not those of a full likelihood: refused where
    there are tied event times (without ties the two are the same)."""
    if tie_method != "efron":
        return
    event = c == 0
    times, counts = np.unique(x[event], return_counts=True)
    weight = np.bincount(
        np.searchsorted(times, x[event]),
        weights=w[event],
        minlength=times.size,
    )
    if np.any((counts > 1) | (weight > 1)):
        raise ValueError(
            "family={!r} is fitted on the full likelihood with the "
            "baseline's jumps as parameters, whose ties are Breslow's: "
            "these data have tied event times, and Efron's approximation "
            "(the default, tie_method='efron') has no such likelihood. "
            "Fit with tie_method='breslow'.".format(family)
        )


def laplace_cox_frailty(
    x: npt.NDArray,
    Zfull: npt.NDArray,
    Zk: npt.NDArray,
    means: npt.NDArray,
    c: npt.NDArray,
    w: npt.NDArray,
    labels: npt.NDArray,
    inv: npt.NDArray,
    kept: npt.NDArray,
    cox: Any,
    maximum: str,
    tie_method: str,
    theta: "float | None",
    center: bool,
    family: str,
) -> Any:
    """The fit of ``CoxFrailty.fit(..., family=family)`` for a family of
    ``families.LAPLACE_FAMILIES`` (see the module docstring), reported as
    a :class:`~surpyval.univariate.regression.CoxFrailtyModel`."""
    from .cox_frailty import _EM_MAX_ITER, _ORIGIN_HINT, CoxFrailtyModel

    n_obs = x.shape[0]
    n_groups = labels.shape[0]
    p_all = Zfull.shape[1]
    whole_counts(np.bincount(inv, weights=w * (c == 0)), family)
    _check_ties(c, w, x, family, tie_method)
    if theta is not None and family == "positive_stable" and not theta < 1:
        raise ValueError(
            "theta of the positive stable frailty is 1 - alpha, its index "
            "alpha in (0, 1]: give a theta in [0, 1); got {!r}.".format(theta)
        )
    em = _em_class()(
        x, Zk - means, c, w, inv, n_groups, "breslow", family=family
    )
    monotone = maximum == "no finite maximum"
    if monotone:
        em.max_iter = 20
    beta0 = (
        np.asarray(cox.beta, dtype=float)[kept]
        if cox is not None
        else np.zeros(0)
    )
    # The Cox fit (Breslow's ties) without frailty: where EM starts, and
    # the marginal likelihood at theta = 0.
    beta0, _ = em.m_step(beta0, np.zeros(n_obs))
    em.cox_beta = beta0
    no_frailty = em.marginal(0.0, beta0, np.zeros(n_groups))

    boundary = False
    if theta is None:
        bounds = _LOG_THETA_BOUNDS[family]
        res = minimize_scalar(
            em.profile,
            bounds=bounds,
            method="bounded",
            options={"xatol": 1e-7},
        )
        theta_hat = float(np.exp(res.x))
        if res.x - bounds[0] < 1e-3 and -res.fun <= no_frailty + 1e-9:
            theta_hat = 0.0
            boundary = True
        elif bounds[1] - res.x < 1e-3:
            maximum = "no finite maximum"
            warn_no_maximum(
                "the frailty's theta runs to the edge of its search, {:g}, "
                "with the marginal likelihood still increasing (groups that "
                "differ more than the family allows)".format(
                    np.exp(bounds[1])
                ),
                "The reported theta, its standard error and its bounds are "
                "meaningless",
                "check for groups whose events all come first, try another "
                "family, or fix theta with theta=",
            )
    else:
        theta_hat = float(theta)

    if theta_hat > 0:
        beta, log_u, _ = em.em(theta_hat, tol=_EM_TOL / 100)
        loglik = em.marginal(theta_hat, beta, log_u)
    else:
        beta, log_u, loglik = beta0, np.zeros(n_groups), no_frailty
    estimated = theta is None and theta_hat > 0
    cov = None
    if not monotone:
        cov = _information(em, family, theta_hat, beta, log_u, estimated)
    if boundary:
        warnings.warn(
            "The {} frailty's theta is estimated at its boundary 0: the {} "
            "groups differ no more than the Cox model without frailty "
            "allows, so the fit is that model (every frailty 1). theta has "
            "no Wald standard error or interval there (nan); test theta = 0 "
            "by twice log_likelihood - log_likelihood_no_frailty, whose null "
            "distribution on the boundary is a 50:50 mixture of 0 and "
            "chi-square(1) (Self and Liang 1987): halve chi-square(1)'s "
            "p-value.".format(family.replace("_", " "), n_groups),
            UserWarning,
            stacklevel=caller_stacklevel(),
        )
    if em.not_converged and maximum == "verified":
        maximum = "unverified"
        warn_unverified(
            "The Cox frailty fit",
            "the EM iteration over the frailties did not converge within "
            "{} iterations at {} value(s) of theta".format(
                _EM_MAX_ITER, em.not_converged
            ),
        )

    times, r, d, h0 = em.baseline(beta, log_u[inv])
    if kept.size and not center:
        r, h0 = baseline_at_origin(beta, means, Zk, r, h0, hint=_ORIGIN_HINT)

    from surpyval.utils.covariates import coefficient_names

    from .._aliasing import fit_columns

    model = CoxFrailtyModel()
    model.family = family
    model.tie_method = tie_method
    model.beta = expand(beta, kept, p_all) if p_all else np.zeros(0)
    model.center = (
        covariate_center(Zfull, w) if center and p_all else np.zeros(p_all)
    )
    model.theta = theta_hat
    model.x = times
    model.h0 = h0
    model.H0 = np.cumsum(h0)
    model.group_labels = list(labels)
    model.frailties = {
        str(lab): float(u) for lab, u in zip(labels, np.exp(log_u))
    }
    names = coefficient_names(p_all, fit_columns(), ["theta"])
    names += ["theta"]
    covariance = np.full((p_all + 1, p_all + 1), np.nan)
    if cov is not None:
        q = kept.size
        if q:
            covariance[np.ix_(kept, kept)] = cov[:q, :q]
        if estimated:
            covariance[p_all, p_all] = cov[q, q]
            if q:
                covariance[np.ix_(kept, [p_all])] = cov[:q, q:]
                covariance[np.ix_([p_all], kept)] = cov[q:, :q]
    model._covariance = covariance
    model.parameter_names = names
    model.log_likelihood = float(loglik)
    model.log_likelihood_no_frailty = float(no_frailty)
    model.k = int(kept.size) + (1 if theta is None else 0)
    model.n_obs = n_obs
    model.n_events = int((c == 0).sum())
    model.n_events_weighted = float(w[c == 0].sum())
    model.n_obs_weighted = float(w.sum())
    model.n_groups = n_groups
    model._data_summary = data_summary(c, w, x=x)
    model._fit_data = {"x": x, "c": c, "n": w, "Z": Zfull}
    model.maximum = maximum
    return model
