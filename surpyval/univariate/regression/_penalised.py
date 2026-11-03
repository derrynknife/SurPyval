"""Penalised (ridge, lasso and elastic-net) regression fits.

The penalised fit minimises

.. math::
    \\frac{1}{N} \\ell(\\theta) + \\sum_j \\lambda_j \\left(
        \\frac{1 - a}{2} b_j^2 + a |b_j| \\right),

where :math:`-\\ell` is the model's log-likelihood (the partial likelihood
of a Cox model), :math:`N` the number of units (the sum of the counts),
:math:`b_j` the covariate coefficients, :math:`\\lambda_j` the
``penalizer`` (one value, or one per coefficient; 0 leaves a coefficient
unpenalised) and :math:`a` the ``l1_ratio``: 0 is ridge, 1 the lasso.
This is glmnet's objective (Friedman, Hastie and Tibshirani 2010; Simon
et al. 2011 for the Cox model) and lifelines'. A baseline distribution's
parameters are never penalised. With ``standardise=True`` (the default,
as in both) the penalty is on the coefficients of the covariates divided
by their standard deviations (weighted by the counts, with divisor
:math:`N`, as glmnet's), and the coefficients are reported on the
covariates' own scale.

The L1 part is not differentiable at 0. The solver writes each penalised
coefficient as the difference of two non-negative parts, which makes the
objective smooth, and minimises it with L-BFGS-B, whose bounds give exact
zeros; it then polishes the answer by Newton steps on the coefficients
that are not 0 and checks the optimality (Karush-Kuhn-Tucker) conditions
of those that are: the gradient of the smooth part no larger than their
L1 weight. A coefficient that breaks them is freed and the polish
repeated.

Penalised estimates are biased towards 0 by design, and the curvature of
the penalised objective is not their covariance: Wald intervals and
p-values built from it do not have their stated coverage or size (and
after the lasso's selection there is no simple valid interval at all). So
a penalised model has no covariance, standard errors, confidence bounds
or p-values; asking for them raises an error that says so.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize


class Penalty(NamedTuple):
    """What a penalised fit penalised: the ``penalizer`` (one value per
    coefficient), the ``l1_ratio``, whether the covariates were
    ``standardise``\\ d, the ``scale`` they were divided by (ones
    without standardisation), and the fit's ``effective_df``, the
    number of parameters its ``aic`` and ``bic`` count (Gray 1992):
    :math:`\\operatorname{tr}[H (H + P)^{-1}]` over the parameters not at 0,
    :math:`H` the Hessian of the negative log-likelihood and :math:`P`
    the ridge part of the penalty's. It is the number of non-zero
    coefficients (plus the baseline's parameters) for the lasso, and
    less for a ridge penalty."""

    penalizer: npt.NDArray
    l1_ratio: float
    standardise: bool
    scale: npt.NDArray
    effective_df: float = float("nan")

    def to_dict(self) -> dict:
        """The penalty as strict JSON."""
        return {
            "penalizer": [float(v) for v in self.penalizer],
            "l1_ratio": float(self.l1_ratio),
            "standardise": bool(self.standardise),
            "scale": [float(v) for v in self.scale],
            "effective_df": float(self.effective_df),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Penalty:
        """The penalty of a :meth:`to_dict` dictionary."""
        return cls(
            np.asarray(d["penalizer"], dtype=float),
            float(d["l1_ratio"]),
            bool(d["standardise"]),
            np.asarray(d["scale"], dtype=float),
            float(d.get("effective_df", float("nan"))),
        )

    def weights(self) -> tuple[npt.NDArray, npt.NDArray]:
        """``(l1, l2)``: each coefficient's weight on ``|b|`` and on
        ``b**2 / 2``, in the standardised coefficients."""
        a = self.l1_ratio
        return self.penalizer * a, self.penalizer * (1.0 - a)


#: What a penalised model says when asked for its uncertainty.
NO_INFERENCE = (
    "This model was fitted with a penalty (penalizer={}, l1_ratio={}). "
    "Penalised estimates are biased towards 0 by design, and the "
    "curvature of the penalised objective is not their covariance, so "
    "standard errors, confidence bounds and p-values computed from it "
    "would not have their stated coverage (after the lasso's selection "
    "no simple interval does). None is given. For inference, refit the "
    "unpenalised model (fit), on the covariates you choose before seeing "
    "the data; a bootstrap of the whole penalised procedure describes "
    "its variability but is not a confidence interval."
)


def penalty_line(penalty: Penalty) -> str:
    """The printout's line for a penalised model."""
    pen = np.unique(penalty.penalizer)
    shown = f"{pen[0]:g}" if pen.size == 1 else "per column"
    return (
        "penalizer {}, l1_ratio {:g}{}; no standard errors or tests "
        "(the estimates are penalised)".format(
            shown,
            penalty.l1_ratio,
            ", standardised" if penalty.standardise else "",
        )
    )


def no_inference_error(penalty: Penalty) -> ValueError:
    """The refusal of a penalised model's covariance, bounds or tests."""
    pen = np.unique(penalty.penalizer)
    shown = float(pen[0]) if pen.size == 1 else penalty.penalizer.tolist()
    return ValueError(NO_INFERENCE.format(shown, penalty.l1_ratio))


def refuse_weights(w: Any) -> None:
    """Refuse weights ``w`` in a penalised fit (not built yet)."""
    if w is not None:
        raise ValueError(
            "fit_penalised does not take weights (w) yet: the penalty's "
            "scale (the objective is divided by the number of units) and "
            "the covariates' standardisation would both have to be "
            "weighted. Fit the unpenalised model with w (fit), or give "
            "repeated rows as counts n."
        )


def refuse_lr_test(model: Any) -> None:
    """Refuse the likelihood-ratio test of a penalised ``model``."""
    penalty = getattr(model, "penalty", None)
    if penalty is not None:
        raise ValueError(
            "likelihood_ratio_test is not available for a penalised fit: "
            "the penalised estimates do not maximise the likelihood, so "
            "twice the gain in log-likelihood over the null model is not "
            "chi-squared (and after the lasso's selection no simple test "
            "holds). Refit the unpenalised model (fit) on covariates "
            "chosen before seeing the data, and test that."
        )


def check_penalty(
    penalizer: Any, l1_ratio: Any, standardise: Any, p: int
) -> tuple[npt.NDArray, float]:
    """The ``penalizer`` as one non-negative value per coefficient and the
    ``l1_ratio`` as a float in ``[0, 1]``, or a ``ValueError`` naming the
    argument."""
    pen = np.asarray(penalizer, dtype=float)
    if pen.ndim == 0:
        pen = np.full(p, float(pen))
    if pen.shape != (p,):
        raise ValueError(
            "`penalizer` must be one value or one per covariate column "
            "({}); got shape {}.".format(p, pen.shape)
        )
    if not np.all(np.isfinite(pen)) or np.any(pen < 0):
        raise ValueError(
            "`penalizer` must be finite and at least 0 (0 leaves a "
            "coefficient unpenalised); got {}.".format(pen.tolist())
        )
    try:
        a = float(l1_ratio)
    except (TypeError, ValueError):
        a = np.nan
    if not 0.0 <= a <= 1.0:
        raise ValueError(
            "`l1_ratio` must be between 0 (ridge) and 1 (lasso); got "
            "{!r}.".format(l1_ratio)
        )
    if not isinstance(standardise, (bool, np.bool_)):
        raise ValueError(
            "`standardise` must be True or False; got {!r}.".format(
                standardise
            )
        )
    return pen, a


def covariate_scale(
    Z: npt.NDArray, n: npt.NDArray, standardise: bool
) -> npt.NDArray:
    """Each column's standard deviation, weighted by the counts ``n``
    with divisor ``sum(n)`` (glmnet's), or ones without standardisation.
    A column with no spread keeps the scale 1."""
    p = Z.shape[1]
    if not standardise:
        return np.ones(p)
    w = np.asarray(n, dtype=float)
    mean = w @ Z / w.sum()
    sd = np.sqrt(w @ (Z - mean) ** 2 / w.sum())
    return np.where(sd > 0, sd, 1.0)


class PenalisedResult(NamedTuple):
    """The solver's answer: ``x`` the parameters (in the solver's space),
    ``fun`` the penalised objective there, ``verified`` whether the
    optimality conditions hold, and ``violation`` the largest breach of
    them (0 where none), ``effective_df`` (see :class:`Penalty`) and
    ``hessian``, where the second derivatives came from: ``"exact"`` (the
    model's own) or ``"finite differences"`` (of its gradient, for a
    model that gives none)."""

    x: npt.NDArray
    fun: float
    verified: bool
    violation: float
    effective_df: float
    hessian: str = "exact"


def finite_difference_hessian(
    grad: Callable[[npt.NDArray], npt.NDArray],
) -> Callable[[npt.NDArray], npt.NDArray]:
    """The Hessian of a function with gradient ``grad`` by central
    differences of the gradient (step ``1e-5`` relative to each entry,
    about the cube root of the machine epsilon), symmetrised."""

    def hess(theta: npt.NDArray) -> npt.NDArray:
        theta = np.asarray(theta, dtype=float)
        k = theta.size
        H = np.empty((k, k))
        for j in range(k):
            h = 1e-5 * max(1.0, abs(float(theta[j])))
            up, down = theta.copy(), theta.copy()
            up[j] += h
            down[j] -= h
            H[:, j] = (
                np.asarray(grad(up), dtype=float)
                - np.asarray(grad(down), dtype=float)
            ) / (up[j] - down[j])
        return 0.5 * (H + H.T)

    return hess


def solve_penalised(
    fun: Callable[[npt.NDArray], float],
    grad: Callable[[npt.NDArray], npt.NDArray],
    hess: Callable[[npt.NDArray], npt.NDArray] | None,
    start: npt.NDArray,
    penalised: npt.NDArray,
    l1: npt.NDArray,
    l2: npt.NDArray,
    tol: float = 1e-9,
) -> PenalisedResult:
    """Minimise ``fun(theta) + sum(l2 * b**2) / 2 + sum(l1 * |b|)``.

    ``fun`` (with its gradient and Hessian) is the smooth part, already
    divided by ``N``; ``b = theta[penalised]`` are the penalised entries,
    each with its ``l1`` and ``l2`` weight; the others are free. ``tol``
    is the optimality tolerance on the gradient. ``hess`` ``None`` takes
    the Hessian (for the Newton polish, the check of the minimum and the
    effective degrees of freedom) by finite differences of ``grad``, and
    the result says so (``hessian``).
    """
    hessian = "exact"
    if hess is None:
        hess = finite_difference_hessian(grad)
        hessian = "finite differences"
    start = np.asarray(start, dtype=float)
    pen = np.asarray(penalised, dtype=int)
    split = pen[l1 > 0]  # the entries written as u - v
    l1_split = l1[l1 > 0]
    k = start.size

    def smooth(theta: npt.NDArray) -> float:
        b = theta[pen]
        return float(fun(theta)) + 0.5 * float(np.sum(l2 * b**2))

    def smooth_grad(theta: npt.NDArray) -> npt.NDArray:
        g = np.array(grad(theta), dtype=float)
        g[pen] += l2 * theta[pen]
        return g

    def smooth_hess(theta: npt.NDArray) -> npt.NDArray:
        H = np.array(hess(theta), dtype=float)
        H[pen, pen] += l2
        return H

    def total(theta: npt.NDArray) -> float:
        return smooth(theta) + float(np.sum(l1_split * np.abs(theta[split])))

    # Stage 1: the split problem, theta[split] = u - v with u, v >= 0.
    def unpack(z: npt.NDArray) -> npt.NDArray:
        theta = z[:k].copy()
        theta[split] = z[k : k + split.size] - z[k + split.size :]
        return theta

    def split_fun(z: npt.NDArray) -> tuple[float, npt.NDArray]:
        theta = unpack(z)
        f = smooth(theta) + float(
            np.sum(l1_split * (z[k : k + split.size] + z[k + split.size :]))
        )
        g_theta = smooth_grad(theta)
        g = np.concatenate(
            [g_theta, g_theta[split] + l1_split, -g_theta[split] + l1_split]
        )
        g[split] = 0.0  # those entries live in u and v
        return f, g

    z0 = np.concatenate(
        [
            start,
            np.maximum(start[split], 0.0),
            np.maximum(-start[split], 0.0),
        ]
    )
    z0[split] = 0.0
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * k
    bounds += [(0.0, None)] * (2 * split.size)
    for j in split:
        bounds[j] = (0.0, 0.0)
    with np.errstate(all="ignore"):
        res = minimize(
            split_fun,
            z0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 20000, "ftol": 1e-15, "gtol": tol * 1e-2},
        )
    theta = unpack(res.x)
    with np.errstate(all="ignore"):
        theta = _polish(
            theta, total, smooth_grad, smooth_hess, split, l1_split, tol
        )
        violation = _kkt_violation(theta, smooth_grad, split, l1_split)
        g = smooth_grad(theta)
        g[split] += l1_split * np.sign(theta[split])
        active = _active(theta, split, k)
        H = smooth_hess(theta)[np.ix_(active, active)]
        # A Hessian that is not finite cannot show a maximum (and
        # eigvalsh raises on it)
        pd = (
            bool(np.all(np.isfinite(H)) and np.all(np.linalg.eigvalsh(H) > 0))
            if active.size
            else True
        )
    scale = max(1.0, abs(total(theta)))
    stationary = (
        bool(np.max(np.abs(g[active])) <= 1e3 * tol * scale)
        if active.size
        else True
    )
    verified = (
        bool(np.isfinite(total(theta)))
        and stationary
        and pd
        and violation <= 1e3 * tol * scale
    )
    with np.errstate(all="ignore"):
        H_f = np.array(hess(theta), dtype=float)[np.ix_(active, active)]
        ridge = np.zeros(k)
        ridge[pen] = l2
        try:
            df = float(
                np.trace(np.linalg.solve(H_f + np.diag(ridge[active]), H_f))
            )
        except np.linalg.LinAlgError:
            df = float("nan")
    return PenalisedResult(
        theta, total(theta), verified, violation, df, hessian
    )


class PenalisedObjective(NamedTuple):
    """A model's smooth objective for :func:`penalised_fit`, in the
    solver's parameters with the covariates divided by their scale:
    ``fun`` (the negative log-likelihood over ``N``), its gradient
    ``grad``, its Hessian ``hess`` (``None`` for finite differences of
    ``grad``) and the ``start`` of the search."""

    fun: Callable[[npt.NDArray], float]
    grad: Callable[[npt.NDArray], npt.NDArray]
    hess: Callable[[npt.NDArray], npt.NDArray] | None
    start: npt.NDArray


class PenalisedFit(NamedTuple):
    """What :func:`penalised_fit` found: the solver's ``result``, the
    ``coefficients`` on the covariates' own scale, the :class:`Penalty`
    and ``N``, the number of units the objective was divided by."""

    result: PenalisedResult
    coefficients: npt.NDArray
    penalty: Penalty
    N: float


def penalised_fit(
    objective: Callable[[npt.NDArray, float], PenalisedObjective],
    Z: npt.NDArray,
    weights: npt.NDArray,
    coef_index: npt.ArrayLike,
    penalizer: Any,
    l1_ratio: Any,
    standardise: Any,
    tol: float,
) -> PenalisedFit:
    """The penalised fit every model's ``fit_penalised`` goes through.

    It checks the ``penalizer``, ``l1_ratio`` and ``standardise``, takes
    the scale of each column of ``Z`` (:func:`covariate_scale`, weighted
    by ``weights``, the rows' counts) and ``N``, the sum of the
    ``weights``; builds the model's smooth objective with
    ``objective(scale, N)`` (on the covariates divided by ``scale``,
    divided by ``N``); minimises it with the penalty on the entries
    ``coef_index`` (one per column of ``Z``) by :func:`solve_penalised`;
    and maps those coefficients back to the covariates' own scale,
    ``b = b_std / scale``.
    """
    Z = np.asarray(Z, dtype=float)
    p = Z.shape[1]
    pen, a = check_penalty(penalizer, l1_ratio, standardise, p)
    scale = covariate_scale(Z, weights, standardise)
    N = float(np.sum(weights))
    obj = objective(scale, N)
    l1, l2 = Penalty(pen, a, bool(standardise), scale).weights()
    index = np.asarray(coef_index, dtype=int)
    with np.errstate(all="ignore"):
        res = solve_penalised(
            obj.fun, obj.grad, obj.hess, obj.start, index, l1, l2, tol=tol
        )
    penalty = Penalty(pen, a, bool(standardise), scale, res.effective_df)
    return PenalisedFit(res, res.x[index] / scale, penalty, N)


def refuse_if_penalised(model: Any) -> None:
    """Raise the refusal of a penalised ``model``'s covariance, bounds or
    tests (:func:`no_inference_error`); nothing for an unpenalised one."""
    penalty = getattr(model, "penalty", None)
    if penalty is not None:
        raise no_inference_error(penalty)


class PenalisedModelMixin:
    """What a fitted model keeps of a penalised fit (``fit_penalised``):
    the ``penalty`` (``None`` for a maximum likelihood fit), the refusal
    of the covariance, standard errors, bounds and tests, the effective
    degrees of freedom its information criteria count, the penalty's
    entry in ``to_dict`` and its line in the printout."""

    #: What a penalised fit penalised; ``None`` for a maximum likelihood
    #: fit. A penalised model has no covariance, standard errors,
    #: confidence bounds or tests.
    penalty: Penalty | None = None

    def _refuse_if_penalised(self) -> None:
        """Raise for a penalised model, which has no covariance, standard
        errors or tests."""
        refuse_if_penalised(self)

    def _penalised_ic_k(self) -> float | None:
        """The parameter count of ``aic`` and ``bic`` of a penalised fit
        (its effective degrees of freedom), or ``None`` for an
        unpenalised one."""
        if self.penalty is None:
            return None
        return float(self.penalty.effective_df)

    def _penalty_entry(self) -> dict:
        """``{"penalty": ...}`` for ``to_dict``; empty if unpenalised."""
        if self.penalty is None:
            return {}
        return {"penalty": self.penalty.to_dict()}

    def _restore_penalty(self, model_dict: dict) -> None:
        """Set the ``penalty`` of a model restored from ``model_dict``."""
        if "penalty" in model_dict:
            self.penalty = Penalty.from_dict(model_dict["penalty"])

    def _penalised_coefficients(self) -> tuple[list, npt.NDArray]:
        """The names and values of the penalised coefficients: the last
        entries of ``params`` (one per entry of the penalty), which a
        model whose ``params`` are laid out otherwise overrides."""
        if self.penalty is None:
            raise ValueError("This model was not fitted with a penalty.")
        m = len(self.penalty.penalizer)
        params = np.asarray(getattr(self, "params"), dtype=float)
        names = list(getattr(self, "parameter_names"))
        return names[-m:], params[-m:]

    def _penalty_printout(self) -> str:
        """The printout's penalty line (with its newline), or ``""``."""
        if self.penalty is None:
            return ""
        return "\nPenalty             : " + penalty_line(self.penalty)


def _active(theta: npt.NDArray, split: npt.NDArray, k: int) -> npt.NDArray:
    """The entries not held at 0: every free one, and each L1-penalised
    one that is not 0."""
    at_zero = np.zeros(k, dtype=bool)
    at_zero[split] = theta[split] == 0.0
    return np.flatnonzero(~at_zero)


def _kkt_violation(
    theta: npt.NDArray,
    smooth_grad: Callable,
    split: npt.NDArray,
    l1: npt.NDArray,
) -> float:
    """How far the L1-penalised entries at 0 break optimality: the excess
    of the smooth part's gradient over their L1 weight (0 if none)."""
    if split.size == 0:
        return 0.0
    g = smooth_grad(theta)[split]
    zero = theta[split] == 0.0
    if not zero.any():
        return 0.0
    return float(np.max(np.maximum(np.abs(g[zero]) - l1[zero], 0.0)))


def _stationarity(
    theta: npt.NDArray,
    smooth_grad: Callable,
    split: npt.NDArray,
    l1: npt.NDArray,
) -> float:
    """How far ``theta`` is from a minimum of the penalised objective:
    the largest entry of its smallest subgradient (the gradient of the
    free entries, that plus the L1 weight times the sign for a penalised
    entry not at 0, and the excess of the gradient over the L1 weight for
    one at 0). 0 exactly at the minimum."""
    g = np.array(smooth_grad(theta), dtype=float)
    if split.size:
        b = theta[split]
        gs = g[split]
        g[split] = np.where(
            b != 0.0,
            gs + l1 * np.sign(b),
            np.maximum(np.abs(gs) - l1, 0.0),
        )
    return float(np.max(np.abs(g))) if g.size else 0.0


def _polish(
    theta: npt.NDArray,
    total: Callable,
    smooth_grad: Callable,
    smooth_hess: Callable,
    split: npt.NDArray,
    l1: npt.NDArray,
    tol: float,
) -> npt.NDArray:
    """Newton steps on the entries not at 0, each L1 term linear in its
    sign there; a step that takes an entry across 0 stops at 0, and an
    entry at 0 that breaks optimality is freed with the sign its gradient
    gives it. Never returns a point worse than it was given.

    Near the minimum a Newton step changes the objective by less than its
    rounding error (a step of 1e-8 moves it by about 1e-16), so the value
    alone cannot rank the last steps: a step that leaves the value within
    its rounding (``slack``, 1e-15 relative to the value's size) of the
    best one is kept when it brings the point closer to stationary
    (:func:`_stationarity`), and only a real increase in the value counts
    as worse."""
    k = theta.size
    best, best_f = theta.copy(), total(theta)
    slack = 1e-15 * max(1.0, abs(best_f)) if np.isfinite(best_f) else 1e-15
    best_s = _stationarity(theta, smooth_grad, split, l1)
    pos = {int(j): i for i, j in enumerate(split)}
    sign = np.zeros(k)
    sign[split] = np.sign(theta[split])
    for _ in range(200):
        active = np.flatnonzero((sign != 0) | ~np.isin(np.arange(k), split))
        if active.size == 0:
            # Every entry at 0: free the worst breach of optimality, if any.
            g = smooth_grad(theta)
            breach = np.abs(g[split]) - l1
            worst = int(np.argmax(breach))
            if breach[worst] <= tol:
                break
            sign[split[worst]] = -np.sign(g[split[worst]])
            continue
        g = smooth_grad(theta)
        g_lin = g.copy()
        g_lin[split] += l1 * sign[split]
        H = smooth_hess(theta)[np.ix_(active, active)]
        try:
            step = -np.linalg.solve(H, g_lin[active])
        except np.linalg.LinAlgError:
            break
        # Stop at the first entry that would cross 0.
        t = 1.0
        hit = -1
        for i, j in enumerate(active):
            if j in pos and step[i] != 0:
                cross = -theta[j] / step[i]
                if 0 < cross < t:
                    t, hit = cross, int(j)
        new = theta.copy()
        new[active] += t * step
        if hit >= 0:
            new[hit] = 0.0
            sign[hit] = 0.0
        f_new = total(new)
        while not (np.isfinite(f_new) and f_new <= best_f + slack):
            t /= 2
            if t < 1e-10:
                break
            new = theta.copy()
            new[active] += t * step
            if hit >= 0 and t < 1.0:
                sign[hit] = np.sign(theta[hit])
                hit = -1
            f_new = total(new)
        if not (np.isfinite(f_new) and f_new <= best_f + slack):
            break
        theta = new
        s_new = _stationarity(theta, smooth_grad, split, l1)
        if f_new < best_f - slack or s_new <= best_s:
            best, best_f, best_s = theta.copy(), min(f_new, best_f), s_new
        small = np.max(np.abs(t * step)) <= tol * max(
            1.0, float(np.max(np.abs(theta)))
        )
        if small and hit < 0:
            # Converged on this active set: free the worst breach, if any.
            g = smooth_grad(theta)
            zero = [j for j in split if sign[j] == 0]
            if not zero:
                break
            breach = np.array([abs(g[j]) - l1[pos[j]] for j in zero])
            worst = int(np.argmax(breach))
            if breach[worst] <= tol:
                break
            sign[zero[worst]] = -np.sign(g[zero[worst]])
    return best
