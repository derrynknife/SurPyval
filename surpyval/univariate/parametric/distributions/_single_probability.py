"""
The machinery shared by the two one-parameter probability models.

``Bernoulli`` and ``FixedEventProbability`` are different models -- a
coin flip over ``{0, 1}`` against a flat ``F(x) = p`` -- but their
*estimation* is the same problem: one probability ``p`` in ``(0, 1)``,
fitted from 0/1 observations by a weighted mean, with no offset,
limited-failure or zero-inflation structure. When the classes were split
in 0.20.0 that machinery was copied into both files verbatim; this mixin
is the single copy. Everything distributional -- ``sf``, ``ff``, the
supports, the docstrings that state each model's own convention -- stays
on the classes themselves.
"""

from __future__ import annotations

from typing import Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.univariate.parametric.parametric import uniform_draws
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    reject_structural_params,
)

from ..parametric import Parametric


class SingleProbabilityMixin:
    """``fit``/``from_params``/``entropy``/``random`` for a model whose
    single parameter is an event probability and whose data are 0/1."""

    # Provided by the host class's ParametricFitter initialisation.
    name: str

    def entropy(self, p: Boxable) -> Boxable:
        r"""The (Shannon) entropy of the 0/1 outcome,
        :math:`-(1 - p)\ln(1 - p) - p\ln p`, in nats."""
        return -(1 - p) * np.log1p(-p) - p * np.log(p)

    def random(
        self,
        size: int | tuple[int, ...],
        p: Boxable,
        *,
        random_state: Any = None,
    ) -> npt.NDArray:
        r"""

        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        p : float
            The probability of the ``1`` outcome
        random_state : int or numpy.random.Generator, optional
            Seed or generator for a draw of its own; ``None`` (the
            default) draws from numpy's global stream (see
            :meth:`ParametricFitter.random`).

        Returns
        -------

        random : scalar or numpy array
            Random values drawn from the distribution in shape `size`

        """
        U = uniform_draws(size, random_state)
        return (U <= p).astype(int)

    def fit(
        self, x: npt.ArrayLike, n: npt.NDArray | None = None
    ) -> Parametric:
        """
        Estimate ``p`` as the (count-weighted) proportion of ones.

        Parameters
        ----------
        x : array like
            The 0/1 outcomes; any other value raises a ``ValueError``.
        n : array like, optional
            The count of each outcome in ``x``. Defaults to one each.

        Returns
        -------
        Parametric
            The fitted model, with ``params`` holding ``p``. Its
            ``param_cb("p")`` bounds ``p`` from the number of ones and of
            outcomes: ``method="exact"`` (the default, Clopper-Pearson),
            ``"wald"`` (logit scale) or ``"lr"`` (likelihood ratio); see
            :func:`probability_bounds`. ``cb``, ``quantile_cb`` and
            ``mean_cb`` are not available: the model's uncertainty is that
            of ``p``.
        """
        x_arr = np.atleast_1d(x)
        # Each observation must be a 0 or a 1 — elementwise, for any length
        # (the previous check broadcast x against the literal [0, 1], so any
        # input of length != 2 crashed and [1, 1] was rejected, #257).
        if not np.isin(x_arr, (0, 1)).all():
            raise ValueError("'x' must be either 0 or 1")
        n_arr = np.ones_like(x_arr) if n is None else np.atleast_1d(n)
        if n_arr.shape[0] != x_arr.shape[0]:
            raise ValueError("'n' must be the same length as 'x'")

        model = Parametric(self, "MLE", None, False, False, False)
        # The proportion is the exact maximum
        model.maximum = "verified"
        p = (x_arr * n_arr).sum() / n_arr.sum()
        model.params = np.array([p])
        # The bounds on p are computed from these (#580).
        model._event_counts = (
            float((x_arr * n_arr).sum()),
            float(n_arr.sum()),
        )
        # As from_dict sets it, so a fitted and a restored model agree.
        self._set_support(model, False)  # type: ignore[attr-defined]
        return model

    def _probability_cb(
        self,
        model: Parametric,
        name: str,
        alpha_ci: float,
        bound: str,
        method: str | None,
    ) -> npt.NDArray:
        """``model.param_cb`` for the probability (#580); see
        :func:`probability_bounds`."""
        if name != "p":
            raise ValueError(
                "Unknown parameter {!r}; expected one of ['p']".format(name)
            )
        return probability_bounds(
            *event_counts(model), alpha_ci, bound, method, name
        )

    # Narrower than ParametricFitter.from_params, which takes
    # (params, gamma, p, f0). Unlike `fit`, this one is not resolved
    # by the OptimisedFitMixin split: every distribution has a
    # from_params. It is a parameter *rename* -- the base's `params`
    # became `p` -- so positional calls work and keyword calls
    # raise. Worse here: the base's `p` means the
    # limited-failure proportion, so the same keyword means two
    # unrelated things across sibling classes. Fixing it means renaming
    # back, with a deprecation alias, and is tracked separately.
    def from_params(
        self,
        params: npt.ArrayLike,
        gamma: Boxable | None = None,
        p: Boxable | None = None,
        f0: Boxable | None = None,
    ) -> Parametric:
        """Create a model from its event probability.

        Parameters
        ----------
        params : scalar
            The event probability, between 0 and 1.
        gamma, p, f0 : None
            Accepted so the signature matches
            :meth:`ParametricFitter.from_params`, and rejected: neither
            model has an offset, limited failure population or zero
            inflation. Note that the base's ``p`` is the *never-fails*
            proportion, not this distribution's parameter -- which is why
            the parameter is ``params`` and not ``p``.
        """
        reject_structural_params(self.name, gamma, p, f0)
        prob = float(np.squeeze(np.asarray(params)))

        if prob > 1:
            raise ValueError("'params' must be less than 1")

        if prob < 0:
            raise ValueError("'params' must be greater than 0")

        model = Parametric(self, "given parameters", None, False, False, False)
        model.params = np.atleast_1d(prob)
        self._set_support(model, False)  # type: ignore[attr-defined]
        return model


# -- bounds on the probability (#580) --------------------------------------


def event_counts(model: Parametric) -> tuple[float, float]:
    """``(events, trials)`` of a fitted probability model, or the one
    message for a model without them (built from its parameters, or
    restored from a dict written before they were saved)."""
    counts = getattr(model, "_event_counts", None)
    if counts is None:
        raise ValueError(
            "Bounds on the probability of a {} model come from the counts "
            "of events and trials it was fitted to, which this model does "
            "not have (it was built from its parameters, or restored from "
            "a dict saved before v0.23); refit it.".format(model.dist.name)
        )
    return counts


#: The ways ``param_cb`` bounds a probability from its counts; the first
#: is the default.
PROBABILITY_CB_METHODS = ("exact", "wald", "lr")


def probability_bounds(
    events: float,
    trials: float,
    alpha_ci: float = 0.05,
    bound: str = "two-sided",
    method: str | None = None,
    name: str = "p",
) -> npt.NDArray:
    r"""
    Confidence bound(s) on a binomial probability from ``events`` in
    ``trials``, the bounds ``param_cb`` gives the probability models.

    ``method``:

    - ``"exact"`` (the default, ``None``): Clopper and Pearson (1934), the
      ``Beta(k, N - k + 1)`` quantile at the tail probability below and
      the ``Beta(k + 1, N - k)`` quantile above; 0 below when ``k = 0`` and
      1 above when ``k = N``. It holds at least its level for any ``N``,
      and with no events its one-sided upper bound is
      :math:`1 - \alpha^{1/N}`, the complement of :func:`success_run`.
    - ``"wald"``: on the logit scale, from the information
      :math:`N / (\hat p (1 - \hat p))`; undefined (``nan``, with a
      warning) at :math:`\hat p = 0` or 1.
    - ``"lr"``: the values of :math:`p` whose deviance
      :math:`2[\ell(\hat p) - \ell(p)]` stays below the :math:`\chi^2_1`
      critical value; at :math:`\hat p = 0` the lower bound is 0 and the
      upper :math:`1 - e^{-c / (2N)}`.
    """
    from scipy.optimize import brentq
    from scipy.special import xlog1py, xlogy
    from scipy.stats import beta, norm

    from surpyval.utils.linalg import (
        bound_signs,
        param_name,
        wald_bound_on_support,
        wald_undefined,
        warn_wald_undefined,
    )
    from surpyval.utils.validation import alpha_ci_error, check_option

    method = PROBABILITY_CB_METHODS[0] if method is None else method
    check_option("method", method, PROBABILITY_CB_METHODS)
    alpha, signs = bound_signs(alpha_ci, bound)
    if not 0 < alpha_ci < 1:
        raise alpha_ci_error(alpha_ci)
    k, n = float(events), float(trials)
    p_hat = k / n

    if method == "wald":
        var = p_hat * (1.0 - p_hat) / n
        reason = wald_undefined(p_hat, var, 0, 1)
        if reason is not None:
            # -> probability_bounds -> _probability_cb -> param_cb -> caller
            warn_wald_undefined(param_name(name), reason, stacklevel=4)
            return np.full(signs.shape, np.nan)
        return wald_bound_on_support(p_hat, var, 0, 1, alpha_ci, bound, name)

    if method == "exact":
        lower = 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))
        upper = 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))
    else:
        # The signed root of the deviance, decreasing in p, set to the
        # normal quantile: a one-sided bound at alpha is the two-sided one
        # at 2 alpha, the package's likelihood-ratio convention.
        z = float(norm.ppf(1 - alpha))

        def loglik(q: float) -> float:
            return float(xlogy(k, q) + xlog1py(n - k, -q))

        def signed_root(q: float) -> float:
            dev = max(2.0 * (loglik(p_hat) - loglik(q)), 0.0)
            return float(np.sign(p_hat - q) * np.sqrt(dev))

        def solve(target: float) -> float:
            lo, hi = 1e-300, 1.0 - 1e-16
            if signed_root(lo) <= target:
                return 0.0
            if signed_root(hi) >= target:
                return 1.0
            return brentq(
                lambda q: signed_root(q) - target,
                lo,
                hi,
                xtol=1e-300,
                rtol=1e-13,
            )

        lower, upper = solve(z), solve(-z)
    out = np.where(signs < 0, lower, upper)
    return out
