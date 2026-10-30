"""Sampling, propensity and censoring weights, ``w``.

A count ``n`` says how many identical units a row records; a weight ``w``
says how many units of the population each of them *stands for* (a
sampling weight), or how much it is up- or down-weighted to undo a bias
(an inverse-probability weight). The two are different kinds of number
and are kept apart: ``n`` stays a whole-number count, and ``w`` is any
positive finite number, given beside it.

With weights the estimates are the weighted ones (each row's contribution
to a likelihood, or to the numbers at risk and of deaths, is multiplied
by ``n * w``), but the classical variances are not valid: they would treat
a weight of 2.5 as 2.5 units seen, so intervals would depend on how the
weights happen to be scaled. A weighted fit therefore always takes a
robust (sandwich, or infinitesimal-jackknife) variance, with no option of
the classical one, and refuses each bound that weights would make invalid,
saying what to use instead. This module holds what those fits share: the
check of ``w``, the effective sample size, the refusals and their wording,
the per-row scores of a likelihood and the sandwich itself.
"""

from __future__ import annotations

import copy
import inspect
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.warnings import caller_stacklevel

#: Below this effective sample size a weighted fit warns that its robust
#: variance may be poor.
SMALL_EFFECTIVE_SIZE = 20


def check_weights(w: Any, k: int) -> npt.NDArray:
    """``w`` as a float array of ``k`` positive finite weights (a scalar
    is the weight of every row), refusing anything else by name."""
    try:
        arr = np.asarray(w, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            "The weights 'w' must be numbers, one positive finite weight "
            "per row."
        ) from None
    if arr.ndim == 0:
        arr = np.full(k, float(arr))
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError("The weights 'w' must be one dimensional.")
    if arr.shape[0] != k:
        raise ValueError(
            f"'w' has {arr.shape[0]} weights for {k} rows: give one weight "
            "per row (or one weight for every row)."
        )
    if np.isnan(arr).any():
        raise ValueError(
            "The weights 'w' cannot contain NaN values: drop the rows "
            "without a weight, or give them one."
        )
    if not (np.isfinite(arr) & (arr > 0)).all():
        bad = arr[~(np.isfinite(arr) & (arr > 0))]
        raise ValueError(
            "The weights 'w' must be positive and finite; got "
            f"{bad[:5].tolist()}. A unit with weight 0 stands for nobody: "
            "leave its row out."
        )
    return arr


def effective_sample_size(n: npt.ArrayLike, w: npt.ArrayLike) -> float:
    """Kish's effective sample size of rows counted ``n`` and weighted
    ``w``: ``(sum n w)^2 / sum n w^2``. It is ``sum n`` with equal
    weights, and smaller the more uneven the weights are."""
    n_arr = np.asarray(n, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    return float(np.sum(n_arr * w_arr) ** 2 / np.sum(n_arr * w_arr**2))


def warn_if_few_effective(n: npt.ArrayLike, w: npt.ArrayLike) -> None:
    """One warning where the effective sample size is small, below
    :data:`SMALL_EFFECTIVE_SIZE`, since the robust variance is then
    poor."""
    ess = effective_sample_size(n, w)
    units = int(np.sum(np.asarray(n)))
    if ess < SMALL_EFFECTIVE_SIZE:
        warnings.warn(
            f"The weights leave an effective sample size of {ess:.3g} "
            f"(of {units} units): with so few effective units "
            "the robust (sandwich) variance, and every interval built on "
            "it, can be poor. Check whether a few units carry most of the "
            "weight.",
            UserWarning,
            stacklevel=caller_stacklevel(),
        )


#: The start of :func:`warn_if_few_effective`'s message, for
#: :func:`few_effective_said_later`.
FEW_EFFECTIVE = "The weights leave an effective sample size"


@contextmanager
def few_effective_said_later() -> Iterator[None]:
    """Silence :func:`warn_if_few_effective` inside a step whose fit
    warns of it itself afterwards (a regression's start from the
    baseline's weighted fit), so that it is said once (principle 22)."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=FEW_EFFECTIVE, category=UserWarning
        )
        yield


class WeightedRefusal(ValueError):
    """A bound or method that sampling weights would make invalid."""


def refuse_weighted(what: str, why: str, instead: str) -> WeightedRefusal:
    """The refusal of ``what`` for a fit with weights ``w``: what it is,
    why weights make it invalid, and what to use instead."""
    return WeightedRefusal(
        f"{what} is not available for a fit with weights (w): {why}. "
        f"{instead}"
    )


#: The reasons and the alternatives the refusals name, shared so that the
#: same bound is refused in the same words everywhere.
NOT_A_LIKELIHOOD = (
    "a weighted fit maximises a pseudo-likelihood (each row's term "
    "multiplied by its weight), which is not the likelihood of the data, "
    "so its value cannot be compared between models"
)
USE_WALD = (
    "Use the Wald bounds, which use the robust (sandwich) covariance of "
    "the weighted fit (method='wald', the default)."
)
LR_WHY = (
    "the likelihood-ratio statistic of a weighted (pseudo) likelihood is "
    "not chi-squared"
)
BOOT_WHY = (
    "the parametric bootstrap simulates data from the model without the "
    "sampling design the weights describe"
)


def refuse_weighted_data(data: Any, what: str, why: str, instead: str) -> None:
    """Refuse weighted ``data`` (a ``SurpyvalData`` with weights ``w``,
    or a model fitted with them) in ``what``, which does not take weights
    yet and would otherwise drop them or read them as counts: why, and
    what to use instead."""
    if getattr(data, "w", None) is None and not getattr(
        data, "weighted", False
    ):
        return
    raise WeightedRefusal(
        f"{what} does not take weights (w), which this "
        f"{type(data).__name__} holds: {why}. {instead}"
    )


#: The fitters that take weights ``w``, for the refusals that name them.
TAKES_WEIGHTS = (
    "KaplanMeier, NelsonAalen, FlemingHarrington, the parametric "
    "distributions (Weibull.fit and the others, by maximum likelihood), "
    "the parametric regressions (WeibullPH, WeibullAFT, ...) and CoxPH"
)


def refuse_unsupported_weights(
    owner: Any, qualname: str = "fit", argument: str = "w"
) -> WeightedRefusal:
    """The refusal of ``w=`` by the ``fit`` of ``owner`` (a fitter, or
    its class), which takes no weights: a ``ValueError`` naming ``w``, the
    fitter, and the fitters that take weights, in place of Python's
    ``TypeError`` for an unexpected keyword (principle 2). ``qualname``,
    the method's, names the fitter where ``owner`` is not one (the first
    argument of a fit called on its class), and ``argument`` is the one
    refused (``w``, or ``w_col`` of a ``fit_from_df``).

    Examples
    --------
    >>> from surpyval import AalenAdditive
    >>> try:
    ...     AalenAdditive.fit([1, 2, 3], [[0], [1], [0]], w=[1, 2, 1])
    ... except ValueError as error:
    ...     print(str(error)[:51])
    AalenAdditive.fit does not take weights (w) yet: we
    """
    if hasattr(owner, "fit"):
        cls = owner if isinstance(owner, type) else type(owner)
        name = cls.__name__.rstrip("_")
    else:
        owner = None
        name = (qualname.rsplit(".", 1)[0] or "This model").rstrip("_")
    if name.endswith(("Fitter", "Regression")):
        # A class behind several public fitters (WeibullDistributional,
        # RoystonParmarPH, ...): the instance's own name says which.
        given = getattr(owner, "name", None)
        if isinstance(given, str) and given.isidentifier():
            name = given
        else:
            name = name.removesuffix("Fitter")
    method = qualname.rsplit(".", 1)[-1]
    fit = getattr(owner, "fit", None)
    if method != "fit" and fit is not None:
        try:
            fit_takes_w = "w" in inspect.signature(fit).parameters
        except (TypeError, ValueError):
            fit_takes_w = False
        if fit_takes_w:
            return WeightedRefusal(
                f"{name}.{method} does not take weights ({argument}) yet, "
                f"but {name}.fit does: pass the columns as arrays, "
                f"{name}.fit(..., w=df[{argument}])."
            )
    return WeightedRefusal(
        f"{name}.{method} does not take weights ({argument}) yet: weighted "
        f"fitting is not built for this model. Fit it without "
        f"{argument} (identical units can be given as counts, n), or, for "
        f"weighted data, use one of the models that take w: "
        f"{TAKES_WEIGHTS}."
    )


#: Why a Bayesian fit refuses weighted data, and what to use instead.
BAYES_WHY = (
    "a posterior from the likelihood with each row raised to its weight "
    "treats the weights as units seen, so its credible intervals would "
    "depend on how the weights are scaled, and weighted Bayesian fitting "
    "is not built yet"
)
BAYES_INSTEAD = (
    "Fit by maximum likelihood with the weights (Weibull.fit(data)), "
    "whose robust covariance gives valid Wald intervals, or fit the data "
    "without its weights (SurpyvalData(data.x, data.c, data.n, data.t))."
)


def refuse_lr(what: str = "A likelihood-ratio bound") -> WeightedRefusal:
    return refuse_weighted(what, LR_WHY, USE_WALD)


def refuse_bootstrap(what: str = "A bootstrap bound") -> WeightedRefusal:
    return refuse_weighted(what, BOOT_WHY, USE_WALD)


def refuse_information_criterion(name: str) -> WeightedRefusal:
    return refuse_weighted(
        name,
        NOT_A_LIKELIHOOD,
        "Compare weighted models by their predictions or a weighted "
        "goodness-of-fit check instead, or choose the distribution on the "
        "unweighted data.",
    )


def with_counts(data: Any, v: Any) -> Any:
    """A shallow copy of a ``SurpyvalData`` whose likelihood arrays take
    their counts from ``v`` (one value per row of the data), each
    truncated row with its own window: the likelihoods are linear in these
    counts, so the gradient of a likelihood in ``v`` is the vector of the
    rows' own log-likelihoods. ``v`` may be traced by autograd."""
    out = copy.copy(data)
    out.n_o = v[data.mask_o]
    out.n_r = v[data.mask_r]
    out.n_l = v[data.mask_l]
    out.n_i = v[data.mask_i]
    out.n_t = v[data.truncated_mask]
    # One window per truncated row (not per distinct window), so that
    # each row's truncation term is its own.
    out.tl_unique = data.x_tl
    out.tr_unique = data.x_tr
    out.n_t_unique = out.n_t
    return out


def row_scores(
    loglik: Callable[[Any, npt.NDArray], Any],
    data: Any,
    theta: npt.ArrayLike,
) -> npt.NDArray:
    """The score of each row of ``data`` for one unit of it: ``S[i, k]``,
    the derivative in ``theta[k]`` of row ``i``'s log-likelihood, where
    ``loglik(data, theta)`` is a log-likelihood whose rows enter through
    their counts (``n_o``, ``n_r``, ... as :func:`with_counts` sets them).

    Taken by autograd as the jacobian, in the counts ``v``, of the
    gradient in ``theta`` (the gradient is linear in ``v``): one reverse
    pass per parameter, not per row."""
    from autograd import grad, jacobian

    theta = np.asarray(theta, dtype=float)
    v0 = np.ones(len(data.n))

    def gradient(v: Any) -> Any:
        return grad(lambda th: loglik(with_counts(data, v), th))(theta)

    with np.errstate(all="ignore"):
        scores_t = jacobian(gradient)(v0)
    return np.asarray(scores_t, dtype=float).T


def sandwich(
    bread: npt.NDArray,
    scores: npt.NDArray,
    n: npt.ArrayLike,
    w: npt.ArrayLike,
) -> npt.NDArray:
    """The robust covariance ``B M B``: ``bread`` the inverse of the
    weighted information (the Hessian of the weighted negative
    log-likelihood), ``M = sum_i n_i w_i^2 s_i s_i'`` the meat from the
    rows' scores ``s_i`` (one unit of each), counts ``n`` and weights
    ``w`` (Huber, 1967; Binder, 1983). Rows of ``bread`` that are zero
    (held parameters) stay zero."""
    factor = np.asarray(n, dtype=float) * np.asarray(w, dtype=float) ** 2
    meat = (scores * factor[:, None]).T @ scores
    out = bread @ meat @ bread
    return 0.5 * (out + out.T)
