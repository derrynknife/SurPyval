"""A family's side of a Bayesian fit: the *target*.

Every Bayesian fit in SurPyval is the same computation: priors on a
model's parameters times the likelihood the family's own ``fit`` uses,
turned into a posterior by one of three engines (closed form, grid or
Laplace; :mod:`surpyval.bayes.result`). What differs between families is
only what a :class:`Target` holds:

- the parameters' names and ranges;
- the negative log-likelihood at a parameter vector (the family's own,
  so there is no second likelihood to keep in step with ``fit``);
- where to start the search (the maximum-likelihood fit);
- the log-likelihood of each *unit* (a row of lives, a system's whole
  history), for WAIC and leave-one-out cross-validation;
- the model at given parameters (``with_params``), for every prediction;
- how to simulate a replicate of the data, and the statistic a
  posterior predictive check compares.

A family adds Bayesian fits by writing one ``Target``; the engines, the
priors, the result's intervals and summaries, the criteria and the
checks are shared.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.warnings import ignoring_warnings

#: The targets that can be saved and rebuilt, by ``Target.tag``.
TARGETS: dict[str, Any] = {}
#: Adapters from a fitted maximum-likelihood model to its target, tried
#: in order: ``(applies(model), build(model))``.
_ADAPTERS: list[tuple[Callable[[Any], bool], Callable[[Any], Any]]] = []


def register_target(cls: Any) -> Any:
    """Record a :class:`Target` subclass for :func:`target_from_dict`."""
    TARGETS[cls.tag] = cls
    return cls


def register_adapter(
    applies: Callable[[Any], bool], build: Callable[[Any], Any]
) -> None:
    """Say how a family's fitted model becomes its target (for
    ``prior.from_model`` and ``prior.from_fit``)."""
    _ADAPTERS.append((applies, build))


def _load_families() -> None:
    """Import the modules that register targets and adapters."""
    import surpyval.bayes.regression  # noqa: F401
    import surpyval.degradation.bayes  # noqa: F401
    import surpyval.recurrent.bayes  # noqa: F401
    import surpyval.recurrent.renewal.bayes  # noqa: F401


def target_for_model(model: Any) -> Any | None:
    """The target of a fitted model of a family that has one, else
    ``None``."""
    _load_families()
    for applies, build in _ADAPTERS:
        if applies(model):
            return build(model)
    return None


def target_from_dict(d: dict) -> Any:
    """Rebuild a target from :meth:`Target.to_dict`."""
    _load_families()
    tag = d.get("target")
    if tag not in TARGETS:
        raise ValueError(
            f"No Bayesian target {tag!r}; this version knows "
            f"{', '.join(sorted(TARGETS))}."
        )
    return TARGETS[tag].from_dict(d)


class Reparameterisation:
    """Working coordinates for the engines: a one-to-one map of the free
    parameters' unconstrained values ``phi`` to coordinates ``u`` in
    which the posterior is closer to a normal with independent axes (a
    power-law process's expected number of events and its shape, in
    place of its scale and its shape, whose posterior is a curved
    ridge). The priors, the results and every summary stay in the
    model's own parameters; only the grid's lattice and the Laplace
    normal are laid in ``u``.

    Parameters
    ----------
    names : list of str
        What each working coordinate is, for the diagnostics.
    forward, inverse : callable
        ``u = forward(phi)`` and ``phi = inverse(u)``, on rows.
    log_det : callable
        ``log |det d phi / d u|`` at rows of ``u``.
    same : list of int, optional
        The working coordinates that are ``phi``'s own.
    affine : bool, optional
        Whether ``phi`` is affine in ``u``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.bayes.target import Reparameterisation
    >>> swap = Reparameterisation(
    ...     ["b", "a"], lambda phi: phi[:, ::-1], lambda u: u[:, ::-1],
    ...     lambda u: np.zeros(len(u)))
    >>> swap.inverse(swap.forward(np.array([[1.0, 2.0]])))
    array([[1., 2.]])
    """

    def __init__(
        self,
        names: list[str],
        forward: Any,
        inverse: Any,
        log_det: Any,
        *,
        same: list[int] | None = None,
        affine: bool = False,
    ) -> None:
        self.names = list(names)
        self.forward = forward
        self.inverse = inverse
        self.log_det = log_det
        #: The working coordinates that are phi's own (the peak search
        #: then takes the target's scales for them), or None: the
        #: target's own choice of scale for every one.
        self.same = same
        #: Whether ``phi`` is an affine function of ``u``, so that a
        #: normal in ``u`` is one in ``phi``.
        self.affine = affine


class Target:
    """What a family gives the shared Bayesian layer.

    A subclass sets the attributes and overrides the methods marked
    "family"; the rest have defaults.

    Attributes
    ----------
    tag : str
        The key the target is saved under.
    family : str
        The family in words, for summaries ("Crow-AMSAA process").
    dist : object
        What the priors see as the model's family (a prior built for
        another family is refused by its name).
    names : list of str
        Every parameter, in the likelihood's order.
    bounds : list of tuple
        Each parameter's open range ``(low, high)``, ``None`` for none.
    fixed : dict
        Parameters held at a value.
    mle : object or None
        The maximum-likelihood fit to the same data.
    unit_name : str
        What one unit of the data is, for the criteria ("system").
    default_method : str
        The engine ``method="auto"`` means for this family: ``"auto"``
        is the shared rule (closed form where conjugate, the grid for
        one to three free parameters, Laplace beyond).
    laplace_draws : int
        The Laplace engine's draws.

    Examples
    --------
    The regressions' target, from a maximum-likelihood fit:

    >>> import numpy as np
    >>> import surpyval as sp
    >>> from surpyval.bayes.regression import RegressionTarget
    >>> Z = np.random.default_rng(1).normal(size=(30, 1))
    >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
    >>> target = RegressionTarget(sp.WeibullPH.fit(x, Z))
    >>> isinstance(target, sp.bayes.Target), target.names
    (True, ['alpha', 'beta', 'coef_0'])
    >>> bool(np.isfinite(target.neg_ll(target.start())))
    True
    """

    tag = "target"
    family = "model"
    dist: Any = None
    names: list[str] = []
    bounds: list[tuple] = []
    fixed: dict[str, float] = {}
    mle: Any = None
    data: Any = None
    unit_name = "unit"
    default_method = "auto"
    laplace_draws = 4096
    #: Whether a failed Laplace check points to PyMC even for three or
    #: fewer free parameters (a likelihood too slow for the grid).
    laplace_only = False

    # -- the likelihood (family) -------------------------------------------
    def neg_ll(self, theta: npt.NDArray) -> float:
        """Minus the log-likelihood at the full parameter vector."""
        raise NotImplementedError

    def log_lik(self, theta: npt.NDArray) -> npt.NDArray:
        """The log-likelihood at rows of full parameter vectors,
        ``(N, k) -> (N,)``; ``-inf`` where it is not defined. The default
        calls :meth:`neg_ll` a row at a time; a family whose likelihood
        takes arrays overrides it (the grid evaluates tens of thousands
        of points)."""
        theta = np.atleast_2d(theta)
        out = np.empty(len(theta))
        with ignoring_warnings(), np.errstate(all="ignore"):
            for i, row in enumerate(theta):
                try:
                    out[i] = -float(self.neg_ll(row))
                except (ValueError, FloatingPointError, ZeroDivisionError):
                    out[i] = -np.inf
        return np.where(np.isnan(out), -np.inf, out)

    def start(self) -> npt.NDArray | None:
        """Where the search for the posterior's peak starts: the
        maximum-likelihood estimates, or ``None`` where there are none."""
        if self.mle is None:
            return None
        return np.asarray(self.estimates(), dtype=float)

    def estimates(self) -> npt.NDArray:
        """The maximum-likelihood fit's parameters, in ``names`` order."""
        return np.asarray(self.mle.params, dtype=float)

    def covariance(self) -> npt.NDArray:
        """The maximum-likelihood fit's covariance, in ``names`` order."""
        return np.asarray(self.mle.covariance(), dtype=float)

    def maximum(self) -> str:
        """Whether the maximum-likelihood fit's maximum is verified."""
        return str(getattr(self.mle, "maximum", "unknown"))

    def mle_interval(self, name: str, alpha_ci: float) -> tuple | None:
        """The maximum-likelihood fit's confidence interval for a
        parameter, or ``None`` where it has none."""
        try:
            with ignoring_warnings(), np.errstate(all="ignore"):
                ci = np.asarray(
                    self.mle.param_cb(name, alpha_ci=alpha_ci), dtype=float
                ).reshape(-1)
        except (ValueError, TypeError, KeyError, AttributeError):
            return None
        if ci.size == 2 and np.all(np.isfinite(ci)):
            return float(ci[0]), float(ci[1])
        return None

    # -- units, for WAIC and LOO (family) ----------------------------------
    def unit_log_lik(self, theta: npt.NDArray) -> npt.NDArray:
        """The log-likelihood of each unit at rows of parameter vectors,
        ``(N, k) -> (N, U)``; the units' sum (by :meth:`unit_counts`) is
        the whole log-likelihood."""
        raise NotImplementedError

    def unit_counts(self) -> npt.NDArray:
        """How many identical units each column of :meth:`unit_log_lik`
        stands for."""
        raise NotImplementedError

    # -- models and replicates (family) -------------------------------------
    def with_params(self, theta: npt.NDArray) -> Any:
        """The family's ordinary model at a full parameter vector."""
        raise NotImplementedError

    def simulate(self, theta: npt.NDArray, rng: Any) -> Any:
        """A replicate of the data from the model at ``theta``, under the
        observation scheme of the data fitted (the same units, covariates
        and observation windows)."""
        raise NotImplementedError

    def statistics(self) -> dict[str, Callable[[Any], float]]:
        """The statistics a posterior predictive check can compare, by
        name, each a function of a data set (the observed one or a
        replicate); the first is the default. None, where the family
        has no predictive check."""
        return {}

    def observed(self) -> Any:
        """The data fitted, in the form :meth:`simulate` returns."""
        return self.data

    # -- priors -------------------------------------------------------------
    def default_priors(self, spec: Any, fixed: dict) -> dict:
        """The priors SurPyval chooses for parameters given none, by
        name (none, by default: such a parameter is flat on its
        unconstrained scale)."""
        return {}

    def check_flat(self, bound: Any) -> None:
        """Refuse flat priors the data cannot carry (the default allows
        them; the grid refuses a posterior with no finite mass)."""

    def check(self, bound: Any) -> None:
        """Refuse what the family does not cover, before the fit."""

    def conjugate(self, bound: Any) -> Any | None:
        """The closed-form posterior where the priors and the data allow
        one (a :class:`~surpyval.bayes.engines.conjugate.ConjugateResult`),
        else ``None``."""
        return None

    def reparam(self, bound: Any) -> Any | None:
        """Working coordinates for the engines (a
        :class:`Reparameterisation`) where the posterior is far from
        normal in the parameters' own, else ``None``."""
        return None

    def coordinate_scales(self, bound: Any) -> npt.NDArray | None:
        """The typical size of each free parameter on its unconstrained
        scale (in ``bound.free_names`` order), which the peak search
        divides its coordinates by (``find_mode``'s ``scale``), else
        ``None``. Only the search's steps and tolerances use it: the
        priors, the posterior and every summary are the parameters' own.

        By default the maximum-likelihood fit's standard errors on the
        unconstrained scale, each along its own axis (:func:`axis_scales`:
        with the other parameters held at their estimates), so that a
        location parameter in a time unit of millions, or of
        millionths, is searched as one of order 1. An entry that is not
        finite (no covariance, or a parameter it does not cover) is
        left to the search, which takes it from the curvature of the
        posterior at its start
        (:func:`~surpyval.bayes.engines.grid.start_scales`)."""
        if self.mle is None:
            return None
        try:
            with ignoring_warnings(), np.errstate(all="ignore"):
                theta = np.asarray(self.estimates(), dtype=float).ravel()
                cov = np.asarray(self.covariance(), dtype=float)
        except Exception:  # a fit that cannot give its covariance
            return None
        return axis_scales(theta, cov, bound)

    def collinearity(self, bound: Any) -> float | None:
        """For the Laplace engine's refusal of a fit too collinear for it
        (``result._refuse_collinear``): ``None`` (no check), except where a
        target says the check applies to it (the regressions with no
        centring, :meth:`mle_collinearity`)."""
        return None

    def mle_collinearity(self, bound: Any) -> float | None:
        """The smallest eigenvalue of the correlation matrix of the
        maximum-likelihood estimates of the free parameters, on their
        unconstrained scale (1 for independent ones, 0 for collinear
        ones), or ``None`` where the fit gives no covariance."""
        if self.mle is None or bound.d < 2:
            return None
        try:
            with ignoring_warnings(), np.errstate(all="ignore"):
                theta = np.asarray(self.estimates(), dtype=float).ravel()
                cov = np.asarray(self.covariance(), dtype=float)
        except Exception:  # a fit that cannot give its covariance
            return None
        k = len(bound.names)
        if theta.size != k or cov.shape != (k, k):
            return None
        with np.errstate(all="ignore"):
            dphi = np.array(
                [
                    np.exp(
                        -bound.transforms[i].log_jacobian(
                            bound.transforms[i].to_phi(theta[i])
                        )
                    )
                    for i in bound.free_idx
                ]
            )
            idx = np.asarray(bound.free_idx, dtype=int)
            C = cov[np.ix_(idx, idx)] * dphi[:, None] * dphi[None, :]
            sd = np.sqrt(np.diag(C))
            R = C / np.outer(sd, sd)
        if not np.all(np.isfinite(R)):
            return None
        try:
            return float(np.linalg.eigvalsh(0.5 * (R + R.T)).min())
        except np.linalg.LinAlgError:
            return None

    def quantities(self) -> dict[str, Any]:
        """The family's named quantities for ``prior.on_quantities``:
        ``{keyword: build(key) -> (label, function of the parameters'
        dict, transform kind)}``."""
        return {}

    def grid_checks(self, theta: dict[str, npt.NDArray]) -> list[npt.NDArray]:
        """Functions of the parameters whose quantiles the grid's
        self-check holds steady, beside the parameters' own."""
        return []

    # -- words ----------------------------------------------------------------
    def describe_data(self) -> str:
        """The data in a phrase, for the summary's first line."""
        return ""

    def same_data(self, other: Any) -> bool:
        """Whether another target of this family holds the same data."""
        return False

    # -- saving ---------------------------------------------------------------
    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict) -> Any:
        raise NotImplementedError


def axis_scales(
    theta: npt.NDArray, cov: npt.NDArray, bound: Any
) -> npt.NDArray | None:
    """The size of each free parameter's uncertainty on its
    unconstrained scale along its own axis (``bound.free_names``
    order), from estimates ``theta`` and their covariance ``cov``
    (every parameter, in ``bound.names`` order; a row of NaN for one it
    does not cover): the covariance is carried to the unconstrained
    scale by the delta method (``d phi / d theta``, ``1 / theta`` for a
    log scale), and each scale is ``1 / sqrt`` of the diagonal of its
    inverse, the information. That is the standard error of a parameter
    with the others held at their estimates: where two parameters are
    strongly correlated (an intercept and the coefficient of a covariate
    far from 0) each one's own standard error is many times larger, and
    a search in those units would be badly conditioned. NaN where a
    scale is not a positive finite number; ``None`` where ``theta`` or
    ``cov`` does not have every parameter's row.

    Examples
    --------
    Two independent parameters, a location in millionths and a log
    scale:

    >>> import numpy as np
    >>> from surpyval.bayes.target import axis_scales
    >>> from surpyval.bayes.transforms import Transform
    >>> class Bound:
    ...     names = free_names = ["mu", "sigma"]
    ...     free_idx = [0, 1]
    ...     d = 2
    ...     transforms = [Transform.for_bounds((None, None)),
    ...                   Transform.for_bounds((0, None))]
    >>> cov = np.diag([4e-12, 1e-14])
    >>> axis_scales(np.array([1e-5, 1e-6]), cov, Bound())
    array([2.e-06, 1.e-01])
    """
    k = len(bound.names)
    theta = np.asarray(theta, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)
    if theta.size != k or cov.shape != (k, k):
        return None
    out = np.full(bound.d, np.nan)
    with np.errstate(all="ignore"):
        dphi = np.array(
            [
                np.exp(
                    -bound.transforms[i].log_jacobian(
                        bound.transforms[i].to_phi(theta[i])
                    )
                )
                for i in bound.free_idx
            ]
        )
        idx = np.asarray(bound.free_idx, dtype=int)
        block = cov[np.ix_(idx, idx)] * dphi[:, None] * dphi[None, :]
        covered = np.all(np.isfinite(block), axis=1) & np.isfinite(dphi)
        if not covered.any():
            return out
        try:
            info = np.linalg.inv(block[np.ix_(covered, covered)])
        except np.linalg.LinAlgError:
            return out
        scale = 1.0 / np.sqrt(np.diag(info))
        good = np.isfinite(scale) & (scale > 0)
        out[np.flatnonzero(covered)[good]] = scale[good]
    return out


__all__ = [
    "Reparameterisation",
    "TARGETS",
    "Target",
    "register_adapter",
    "register_target",
    "target_for_model",
    "target_from_dict",
]
