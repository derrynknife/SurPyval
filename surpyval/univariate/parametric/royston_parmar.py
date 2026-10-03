import numpy.typing as npt

"""
Royston-Parmar flexible parametric survival models.

A Royston-Parmar model replaces the *straight line* that a Weibull draws for
its log-cumulative-hazard against log time with a **restricted cubic spline**,
giving a fully parametric model with an arbitrarily flexible baseline shape --
smooth, differentiable, and extrapolable, unlike the step baseline of a Cox
fit, yet free of the rigid single shape a Weibull or log-normal imposes.

The model is defined on one of three link scales through

.. math::
    g\\bigl(S(t)\\bigr) = s(\\log t ; \\gamma),

with ``s`` a restricted cubic spline. The link ``g`` chooses the family:

* ``scale="hazard"`` -- :math:`g = \\log(-\\log S) = \\log H`, a
  proportional-hazards flexible model (``s`` with no internal knots is exactly
  a Weibull);
* ``scale="odds"`` -- :math:`g = \\operatorname{logit}(1 - S)`, a
  proportional-odds flexible model;
* ``scale="normal"`` -- :math:`g = \\Phi^{-1}(1 - S)`, a probit flexible model
  (no internal knots is exactly a log-normal).

Knots are placed at quantiles of the uncensored (event) log-times: the boundary
knots at their extremes, the internal knots at equally-spaced centiles. Beyond
the boundary knots the spline is linear (the "restricted" part), so the model
extrapolates with a Weibull-like tail. The number of knots -- set through
``df`` -- is the modelling choice; compare a few by AIC / BIC.

The likelihood supports the full SurpyvalData censoring/truncation surface:
observed, right-, left- and interval-censored observations, with left- and/or
right-truncation. Right-censored contribute ``log S``, left-censored
``log(1 - S)``, interval-censored ``log(S(l) - S(r))``, and truncation divides
each observation's contribution by ``S(t_l) - S(t_r)``.
"""

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtri as _ndtri

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.information_criteria import (
    InformationCriteriaMixin,
    ic_sample_size,
)
from surpyval.univariate.parametric.fitters import is_local_minimum
from surpyval.utils.dataframe import UnivariateDataFrameMixin
from surpyval.utils.deprecation import ArrayMethod
from surpyval.utils.linalg import (
    numerical_gradient,
    numerical_hessian,
    standard_errors_of,
)
from surpyval.utils.no_maximum import (
    maximum_entry,
    restored_maximum,
    warn_unverified,
)
from surpyval.utils.numeric import solve_bracketed
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import (
    BOUNDS,
    check_option,
    no_covariance_error,
    option_error,
    warn_outside_unit_interval,
)

_SCALES = ("hazard", "odds", "normal")


def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted-cubic-spline design matrix (Durrleman-Simon) at ``x``.

    Columns are ``[1, x, v_1(x), ..., v_m(x)]`` for the ``m`` internal knots.
    """
    x = np.asarray(x, dtype=float)
    kmin, kmax = knots[0], knots[-1]
    cols = [np.ones_like(x), x]
    span = kmax - kmin
    for kj in knots[1:-1]:
        lam = (kmax - kj) / span
        cols.append(
            np.maximum(x - kj, 0.0) ** 3
            - lam * np.maximum(x - kmin, 0.0) ** 3
            - (1.0 - lam) * np.maximum(x - kmax, 0.0) ** 3
        )
    return np.column_stack(cols)


def _rcs_deriv(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Derivative of the RCS basis with respect to ``x``."""
    x = np.asarray(x, dtype=float)
    kmin, kmax = knots[0], knots[-1]
    cols = [np.zeros_like(x), np.ones_like(x)]
    span = kmax - kmin
    for kj in knots[1:-1]:
        lam = (kmax - kj) / span
        cols.append(
            3.0 * np.maximum(x - kj, 0.0) ** 2
            - 3.0 * lam * np.maximum(x - kmin, 0.0) ** 2
            - 3.0 * (1.0 - lam) * np.maximum(x - kmax, 0.0) ** 2
        )
    return np.column_stack(cols)


def _place_knots(x_events: np.ndarray, n_internal: int) -> np.ndarray:
    """Boundary + internal knots at quantiles of the event log-times."""
    lx = np.log(np.asarray(x_events, dtype=float))
    qs = np.linspace(0.0, 1.0, n_internal + 2)
    return np.quantile(lx, qs)


def _scale_terms(eta: np.ndarray, scale: str) -> tuple[Any, ...]:
    """``(log S, log(-dS/deta))`` at linear predictor ``eta`` for a scale."""
    from scipy.stats import norm

    if scale == "hazard":
        log_S = -np.exp(eta)
        return log_S, eta + log_S
    if scale == "odds":
        log_S = -np.logaddexp(0.0, eta)  # log(1 / (1 + e^eta))
        log_1mS = eta + log_S  # log(e^eta / (1 + e^eta))
        return log_S, log_S + log_1mS
    if scale == "normal":
        return norm.logsf(eta), norm.logpdf(eta)
    raise option_error("scale", scale, _SCALES)


def _sf_from_eta(eta: np.ndarray, scale: str) -> np.ndarray:
    from scipy.stats import norm

    if scale == "hazard":
        return np.exp(-np.exp(eta))
    if scale == "odds":
        return 1.0 / (1.0 + np.exp(eta))
    return norm.sf(eta)


def _eta_of_probability(p: npt.NDArray, scale: str) -> npt.NDArray:
    """The linear predictor at which ``ff = p`` (``sf = 1 - p``), the
    inverse of ``_sf_from_eta``: ``-inf`` at ``p = 0``, ``inf`` at
    ``p = 1``."""
    if scale == "hazard":
        return np.log(-np.log1p(-p))
    if scale == "odds":
        return np.log(p) - np.log1p(-p)
    return _ndtri(p)


def _sf_at(
    times: np.ndarray, knots: np.ndarray, gamma: np.ndarray, scale: str
) -> npt.NDArray:
    """Survival at arbitrary times, with the boundary conventions the
    censoring/truncation likelihoods need: ``S = 1`` at times ``<= 0`` (and
    ``-inf``) and ``S = 0`` at ``+inf``. Finite positive times go through the
    spline as usual.
    """
    times = np.asarray(times, dtype=float)
    out = np.empty(times.shape, dtype=float)
    pos = np.isfinite(times) & (times > 0.0)
    if np.any(pos):
        eta = _rcs_basis(np.log(times[pos]), knots) @ gamma
        out[pos] = _sf_from_eta(eta, scale)
    out[~pos] = np.where(np.isposinf(times[~pos]), 0.0, 1.0)
    return out


class RoystonParmarModel(InformationCriteriaMixin, SerialisableMixin):
    """A fitted Royston-Parmar flexible parametric model.

    Carries the spline ``knots``, the coefficients ``params`` (``gamma``), the
    link ``scale``, and (for a maximum-likelihood fit) the coefficient
    covariance for confidence bounds. Exposes the usual distribution surface:
    :meth:`sf`, :meth:`ff`, :meth:`hf`, :meth:`Hf`, :meth:`df`, :meth:`qf`,
    :meth:`random`, :meth:`mean`, and :meth:`cb`.

    Examples
    --------
    ``RoystonParmar.fit`` returns one. Here with one internal knot
    (``df=2``) on the Rossi recidivism data, where ``arrest`` is 1 for an
    arrest (so the censoring flag is ``1 - arrest``):

    >>> from surpyval import RoystonParmar
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static()
    >>> x, c = df["week"].values, 1 - df["arrest"].values
    >>> model = RoystonParmar.fit(x, c=c, df=2)
    >>> model.params.round(4)
    array([-6.9934,  1.5755,  0.0377])
    >>> model.sf([20, 52]).round(4)
    array([0.9182, 0.7365])
    >>> model.cb([20, 52]).round(4)
    array([[0.8915, 0.9386],
           [0.6923, 0.7754]])
    """

    def __init__(self) -> None:
        self.scale = "hazard"
        self.knots = np.array([])
        self.params = np.array([])
        self._covariance: "np.ndarray | None" = None
        self.support = (0.0, np.inf)
        self.n = 0
        self.n_events = 0
        self._neg_ll = 0.0
        # The sample size of bic() (see ic_sample_size), from the data at
        # fit time.
        self._ic_n = 0.0
        # What the fit reached, one of ``MAXIMUM_STATES``
        # (``surpyval.utils.no_maximum``), as its warnings say; "unknown"
        # for a model restored from a dict saved without it.
        self.maximum = "unknown"
        # The negative log-likelihood of the spline coefficients the fit
        # minimised; not saved.
        self._objective: Any = None

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``, entry by entry: the spline
        coefficients ``gamma_0``, ``gamma_1``, ..., as the summary prints
        them."""
        return ["gamma_{}".format(i) for i in range(len(self.params))]

    # -- linear predictor --------------------------------------------------

    def _eta(self, t: np.ndarray) -> np.ndarray:
        return _rcs_basis(np.log(np.asarray(t, dtype=float)), self.knots) @ (
            self.params
        )

    def _eta_deriv(self, t: np.ndarray) -> np.ndarray:
        return _rcs_deriv(np.log(np.asarray(t, dtype=float)), self.knots) @ (
            self.params
        )

    # -- distribution functions -------------------------------------------

    @keeps_query_shape
    def sf(self, x: Any) -> np.ndarray:
        """Survival function at ``x``: 1 at and before time 0 (the spline
        is in ``log x``, which does not exist there, so this came back nan)
        and 0 at infinity, as in the likelihood (see ``_sf_at``)."""
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            out = _sf_from_eta(self._eta(x), self.scale)
        out = np.where(x <= 0.0, 1.0, out)
        return np.where(np.isposinf(x), 0.0, out)

    @keeps_query_shape
    def ff(self, x: Any) -> np.ndarray:
        """Failure (CDF) function ``1 - sf(x)``."""
        return 1.0 - self.sf(x)

    @keeps_query_shape
    def Hf(self, x: Any) -> np.ndarray:
        """Cumulative hazard ``-log sf(x)``."""
        # + 0.0 turns the -0.0 of -log(1) at x <= 0 into 0.0
        return -np.log(self.sf(x)) + 0.0

    @keeps_query_shape
    def hf(self, x: Any) -> np.ndarray:
        """Hazard rate ``df(x) / sf(x)``."""
        return self.df(x) / self.sf(x)

    @keeps_query_shape
    def df(self, x: Any) -> np.ndarray:
        """Density at ``x``, from the derivative of the spline."""
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            eta = self._eta(x)
            sp = self._eta_deriv(x)
            _, log_negdS = _scale_terms(eta, self.scale)
            out = np.exp(log_negdS + np.log(sp) - np.log(x))
        # Nothing fails at or before time 0 (nan there before), nor at
        # infinity; with sf = 1 there, hf and Hf are 0 too.
        return np.where((x <= 0.0) | np.isposinf(x), 0.0, out)

    @keeps_query_shape
    def qf(self, p: Any) -> np.ndarray:
        """Quantile function: the time at which ``ff(x) = p``; 0 at
        ``p = 0`` and ``inf`` at ``p = 1``.

        Solved for every probability at once on the link scale, where
        the spline is: the linear predictor that gives ``ff = p`` is
        found in log time, in closed form beyond the boundary knots
        (where the spline is a straight line) and between them by
        ``solve_bracketed``, to a relative precision of about ``1e-15``
        in time. A ``brentq`` per probability on ``sf`` took 1-3 s for
        2000 draws (#595), and lost precision for a ``p`` near 0, where
        ``sf`` rounds to 1."""
        p = np.asarray(p, dtype=float)
        out = np.full(p.shape, np.nan)
        # As for the other parametric models: NaN, with a warning, where
        # p is outside [0, 1]; the root finder raised a bare scipy error
        # (#576). A missing probability has a missing quantile; the root
        # finder raised on it (#382).
        outside = warn_outside_unit_interval(p)
        valid = ~np.isnan(p) & ~outside
        with np.errstate(divide="ignore"):
            target = _eta_of_probability(p[valid], self.scale)
        out[valid] = np.exp(self._log_time_of_eta(target))
        return out

    def _log_time_of_eta(self, target: npt.NDArray) -> npt.NDArray:
        """The log times at which the linear predictor reaches each
        ``target``. Beyond the boundary knots the spline is linear in log
        time with the slope it has at the knot, so those are closed form
        (a slope that is not positive never reaches them: NaN); between
        the knots they are solved together, by ``solve_bracketed``."""
        k_lo, k_hi = self.knots[0], self.knots[-1]
        ends = np.array([k_lo, k_hi])
        e_lo, e_hi = _rcs_basis(ends, self.knots) @ self.params
        s_lo, s_hi = _rcs_deriv(ends, self.knots) @ self.params
        out = np.full(target.shape, np.nan)
        below = target <= e_lo
        above = ~below & (target >= e_hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            if s_lo > 0:
                out[below] = k_lo + (target[below] - e_lo) / s_lo
            if s_hi > 0:
                out[above] = k_hi + (target[above] - e_hi) / s_hi
        inside = np.flatnonzero(~below & ~above)
        if inside.size:

            def gap(lx: npt.NDArray, sel: npt.NDArray) -> npt.NDArray:
                eta = _rcs_basis(lx, self.knots) @ self.params
                return eta - target[inside[sel]]

            # Absolute in log time: relative in time.
            out[inside] = solve_bracketed(
                gap,
                np.full(inside.size, k_lo),
                np.full(inside.size, k_hi),
                e_lo - target[inside],
                e_hi - target[inside],
                xtol=4 * np.finfo(float).eps,
            )
        return out

    def random(self, size: int, *, random_state: Any = None) -> np.ndarray:
        """Draw ``size`` random lifetimes (by inverting ``ff``).

        ``random_state`` (an int or a ``numpy.random.Generator``) gives a
        draw of its own, which neither depends on nor advances numpy's
        global stream; ``None`` (the default) draws from the global
        stream, so ``np.random.seed`` reproduces it."""
        if random_state is None:
            u = np.random.uniform(0, 1, size)
        else:
            u = as_generator(random_state).uniform(0, 1, size)
        return self.qf(u)

    def mean(self) -> float:
        """Mean life, by integrating the survival function."""
        from scipy.integrate import quad

        val, _ = quad(
            lambda t: float(np.ravel(self.sf(t))[0]), 0, np.inf, limit=200
        )
        return val

    # -- confidence bounds -------------------------------------------------

    @keeps_query_shape
    def cb(
        self,
        x: Any,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """Confidence bound on a function, via the linear predictor.

        The bound is formed on the (unbounded) linear predictor ``eta`` --
        whose variance is ``B Sigma B'`` from the covariance -- and then
        pushed through the link, so ``sf`` / ``ff`` bounds stay in ``(0, 1)``.
        ``S`` is monotone decreasing in ``eta`` on every scale.

        Parameters
        ----------
        x : array like or scalar
            The times at which to bound the function.
        on : {'sf', 'ff', 'Hf'}, optional
            The function to bound (``'R'`` and ``'F'`` are aliases of
            ``'sf'`` and ``'ff'``). Default ``'sf'``.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are ``[lower, upper]`` on the last axis; a
            one-sided bound at ``alpha_ci`` is the matching end of the
            two-sided bound at ``2 * alpha_ci``.
        """
        cov = self.covariance()
        check_option("on", on, ("sf", "R", "ff", "F", "Hf"))
        # An unknown bound (say 'both') used to be taken as 'upper' (#415).
        check_option("bound", bound, BOUNDS)
        # ff = 1 - sf and Hf = -log(sf) decrease in sf, so their lower
        # bound is the transformed upper bound on sf, and vice versa. The
        # one-sided bounds used to return the other side (#415).
        if on in ("ff", "F", "Hf") and bound != "two-sided":
            bound = "upper" if bound == "lower" else "lower"
        x = np.atleast_1d(np.asarray(x, dtype=float))
        B = _rcs_basis(np.log(x), self.knots)
        eta = B @ self.params
        var = np.einsum("ij,jk,ik->i", B, cov, B)
        se = np.sqrt(np.maximum(var, 0.0))

        if bound == "two-sided":
            z = _ndtri(1.0 - alpha_ci / 2.0)
            eta_hi = eta + z * se
            eta_lo = eta - z * se
            sf_lo = _sf_from_eta(eta_hi, self.scale)  # S decreasing in eta
            sf_hi = _sf_from_eta(eta_lo, self.scale)
            band = np.column_stack([sf_lo, sf_hi])
        else:
            z = _ndtri(1.0 - alpha_ci)
            signed = eta + (z if bound == "lower" else -z) * se
            band = _sf_from_eta(signed, self.scale)

        if on in ("sf", "R"):
            return band
        if on in ("ff", "F"):
            return 1.0 - (band[:, ::-1] if band.ndim == 2 else band)
        return -np.log(band[:, ::-1] if band.ndim == 2 else band)

    # -- information criteria (InformationCriteriaMixin) -------------------
    # neg_ll(), log_likelihood, aic(), aic_c() and bic(), the last two with
    # the sample size every SurPyval BIC uses (``_ic_n``, from the data at
    # fit time; see ic_sample_size).

    @property
    def k(self) -> int:  # type: ignore[override]
        return len(self.params)

    #: The coefficients' covariance, ``covariance()`` (#605): an attribute
    #: before v0.23, which still reads it, with a DeprecationWarning.
    covariance = ArrayMethod("_covariance", no_covariance_error)

    def standard_errors(self) -> np.ndarray:
        """The spline coefficients' standard errors, the square roots of
        the diagonal of ``covariance()`` in the order of ``params``
        (``nan`` where a variance is not positive; #613).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import RoystonParmar, Weibull
        >>> x = Weibull.random(200, 10, 2, random_state=1)
        >>> model = RoystonParmar.fit(x, df=2)
        >>> model.standard_errors().shape
        (3,)
        """
        return standard_errors_of(self.covariance())

    def summary(self) -> str:
        """A text summary of the fit: link scale, knots, likelihood and
        coefficients."""
        lines = [
            "Royston-Parmar Flexible Parametric Model",
            "========================================",
            f"Scale               : {self.scale}",
            f"Knots (log-time)    : {np.round(self.knots, 4).tolist()}",
            f"Internal knots      : {len(self.knots) - 2}",
            f"Observations        : {self.n}  (events {self.n_events})",
            f"log-likelihood      : {-self._neg_ll:.4f}",
            f"AIC / BIC           : {self.aic():.2f} / {self.bic():.2f}",
            "Coefficients        :",
        ]
        for i, g in enumerate(self.params):
            lines.append(f"    gamma_{i:<3}: {g:.6g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the fitted model to a plain dictionary; restore it
        with :meth:`from_dict` or ``surpyval.from_dict``."""
        out: dict[str, Any] = {
            "model": "RoystonParmarModel",
            "scale": self.scale,
            "knots": np.asarray(self.knots, float).tolist(),
            "params": np.asarray(self.params, float).tolist(),
            "n": int(self.n),
            "n_events": int(self.n_events),
            "_neg_ll": to_native(self._neg_ll),
            **maximum_entry(self.maximum),
            "ic_n": float(self._ic_sample_size()),
        }
        if self._covariance is not None:
            out["covariance"] = np.asarray(self._covariance, float).tolist()
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "RoystonParmarModel":
        """Rebuild a model from a :meth:`to_dict` dictionary."""
        require_model_tag(
            model_dict, "RoystonParmarModel", "a Royston-Parmar model"
        )
        out = cls()
        out.scale = model_dict["scale"]
        out.knots = np.array(model_dict["knots"], dtype=float)
        out.params = np.array(model_dict["params"], dtype=float)
        out.n = int(model_dict.get("n", 0))
        out.n_events = int(model_dict.get("n_events", 0))
        out._neg_ll = float(model_dict.get("_neg_ll", 0.0))
        out.maximum = restored_maximum(model_dict)
        if "ic_n" in model_dict:
            out._ic_n = float(model_dict["ic_n"])
        else:
            # Written before the sample size was stored: the exact failures
            # are the only failures the dict records.
            out._ic_n = ic_sample_size([0], [out.n_events], n_rows=out.n)
        if "covariance" in model_dict:
            out._covariance = np.array(model_dict["covariance"], dtype=float)
        return out


class _SplineNegLL:
    """The negative log-likelihood of the spline coefficients ``g``.

    An object rather than a closure over the fit's basis matrices, so the
    fitted model, which keeps it as ``_objective``, pickles (#573). Each
    group of arrays is a kind of row (its basis matrices are ``None``
    when the data have none of that kind)."""

    def __init__(
        self,
        scale: str,
        knots: npt.NDArray,
        observed: tuple,
        right: tuple,
        left: tuple,
        interval: tuple,
        truncated: tuple,
    ) -> None:
        self.scale = scale
        self.knots = knots
        self.observed = observed
        self.right = right
        self.left = left
        self.interval = interval
        self.truncated = truncated

    def __call__(self, g: npt.NDArray) -> Any:
        scale, knots = self.scale, self.knots
        B_o, Bd_o, n_o, lx_o = self.observed
        B_r, n_r = self.right
        B_l, n_l = self.left
        B_il, B_ir, n_i = self.interval
        x_tl, x_tr, n_t = self.truncated
        ll = 0.0
        if B_o is not None:  # events: log f = log(-dS) + log s' - log t
            eta = B_o @ g
            sp = Bd_o @ g
            _, log_negdS = _scale_terms(eta, scale)
            ll += np.sum(n_o * (log_negdS + np.log(sp) - lx_o))
        if B_r is not None:  # right-censored: log S
            log_S_r, _ = _scale_terms(B_r @ g, scale)
            ll += np.sum(n_r * log_S_r)
        if B_l is not None:  # left-censored: log F = log(1 - S)
            log_S_l, _ = _scale_terms(B_l @ g, scale)
            ll += np.sum(n_l * np.log1p(-np.exp(log_S_l)))
        if B_il is not None:  # interval-censored: log(S(l) - S(r))
            S_il = _sf_from_eta(B_il @ g, scale)
            S_ir = _sf_from_eta(B_ir @ g, scale)
            ll += np.sum(n_i * np.log(S_il - S_ir))
        if x_tl.size:  # truncation: divide by P(entry <= T <= exit)
            S_tl = _sf_at(x_tl, knots, g, scale)
            S_tr = _sf_at(x_tr, knots, g, scale)
            ll -= np.sum(n_t * np.log(S_tl - S_tr))
        return -ll


class RoystonParmar_(UnivariateDataFrameMixin):
    """Fitter for :class:`RoystonParmarModel`. Use the singleton
    :data:`RoystonParmar`.

    The Royston-Parmar model is a restricted cubic spline in log time on
    the log cumulative hazard (``scale="hazard"``), log cumulative odds
    or probit scale: a smooth parametric survival curve whose flexibility
    is set by ``df``. ``df=1`` is a Weibull (on the hazard scale).

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import RoystonParmar
    >>> rng = np.random.default_rng(0)
    >>> x = 10 * rng.weibull(2, 50)
    >>> model = RoystonParmar.fit(x, df=3)
    >>> model.sf([5, 10]).round(4)
    array([0.8029, 0.4131])
    """

    def fit(
        self,
        x: Any = None,
        c: Any = None,
        n: Any = None,
        t: Any = None,
        xl: Any = None,
        xr: Any = None,
        tl: Any = None,
        tr: Any = None,
        df: int = 3,
        scale: str = "hazard",
        knots: Any = None,
    ) -> RoystonParmarModel:
        """Fit a Royston-Parmar model by maximum likelihood.

        Accepts the full SurpyvalData censoring/truncation surface: observed,
        right-, left- and interval-censored observations, with left- and/or
        right-truncation (delayed entry and right-truncated sampling).

        Parameters
        ----------
        x : array_like
            Observed times (strictly positive). For interval-censored rows the
            entry is a 2-element ``[left, right]`` pair (see ``c``).
        c : array_like, optional
            Censoring flags: ``0`` event, ``1`` right-censored, ``-1``
            left-censored, ``2`` interval-censored. Default all events.
        n : array_like, optional
            Observation weights / counts. Default 1.
        t : array_like, optional
            ``(m, 2)`` array of ``[left, right]`` truncation bounds per row.
            Mutually exclusive with ``tl`` / ``tr``.
        xl, xr : array_like, optional
            Left/right interval bounds for interval-censored data, as an
            alternative to passing 2-element ``x`` rows.
        tl, tr : array_like or scalar, optional
            Left- and right-truncation bounds. A scalar truncates every
            observation at that value.
        df : int, optional
            Degrees of freedom = number of spline terms beyond the intercept;
            ``df - 1`` internal knots. ``df = 1`` is a Weibull (``scale`` =
            hazard) or log-normal (``scale`` = normal). Default 3.
        scale : {"hazard", "odds", "normal"}, optional
            The link scale (proportional hazards, proportional odds, probit).
        knots : array_like, optional
            Explicit knot locations *on the log-time scale* (including the two
            boundary knots), overriding the quantile-based default.
        """
        check_option("scale", scale, _SCALES)

        from surpyval.utils.surpyval_data import SurpyvalData

        data = SurpyvalData(
            x=x,
            c=c,
            n=n,
            t=t,
            xl=xl,
            xr=xr,
            tl=tl,
            tr=tr,
            group_and_sort=True,
        )

        if np.any(data.x_min <= 0):
            raise ValueError(
                "Royston-Parmar requires strictly positive times."
            )

        # Per-type times and weights.
        x_o, n_o = data.x_o, data.n_o.astype(float)
        x_r, n_r = data.x_r, data.n_r.astype(float)
        x_l, n_l = data.x_l, data.n_l.astype(float)
        x_il, x_ir, n_i = data.x_il, data.x_ir, data.n_i.astype(float)
        x_tl, x_tr, n_t = data.x_tl, data.x_tr, data.n_t.astype(float)

        # Right-censored-only data carries no finite MLE: at least one
        # observed event, left- or interval-censored row is needed to identify
        # the baseline.
        if (x_o.size + x_l.size + x_il.size) == 0:
            raise ValueError(
                "Royston-Parmar requires at least one event (or left- or "
                "interval-censored) observation; right-censored-only data is "
                "not identifiable."
            )

        # Knots go on the exactly-observed event log-times when there are any,
        # else on whatever finite failure information the data provide.
        event_times = x_o
        if event_times.size == 0:
            event_times = np.concatenate([x_il, x_ir, x_l])
            event_times = event_times[np.isfinite(event_times)]

        if knots is None:
            n_internal = max(int(df) - 1, 0)
            n_distinct = np.unique(event_times).size
            if n_distinct < n_internal + 2:
                raise ValueError(
                    f"Royston-Parmar with df={df} needs at least "
                    f"{n_internal + 2} distinct event times to place "
                    f"{n_internal + 2} knots; the data have {n_distinct}. "
                    "Reduce df (df=1 fits a two-parameter model with no "
                    "internal knots) or pass explicit knots."
                )
            knots = _place_knots(event_times, n_internal)
        else:
            knots = np.asarray(knots, dtype=float)
        if np.unique(knots).size != len(knots):
            raise ValueError(
                "Royston-Parmar knots must be distinct; quantile placement "
                "over tied event times produced coincident knots "
                f"({np.exp(knots).tolist()} in time units). Reduce df or "
                "pass explicit knots."
            )
        n_params = len(knots)  # [1, x] + (len(knots) - 2) internal terms

        lx_o = np.log(x_o) if x_o.size else x_o
        B_o = _rcs_basis(lx_o, knots) if x_o.size else None
        Bd_o = _rcs_deriv(lx_o, knots) if x_o.size else None
        B_r = _rcs_basis(np.log(x_r), knots) if x_r.size else None
        B_l = _rcs_basis(np.log(x_l), knots) if x_l.size else None
        B_il = _rcs_basis(np.log(x_il), knots) if x_il.size else None
        B_ir = _rcs_basis(np.log(x_ir), knots) if x_ir.size else None

        neg_ll = _SplineNegLL(
            scale,
            knots,
            (B_o, Bd_o, n_o, lx_o),
            (B_r, n_r),
            (B_l, n_l),
            (B_il, B_ir, n_i),
            (x_tl, x_tr, n_t),
        )

        # Initialise from the Weibull/log-normal that the no-knot model is.
        init = np.zeros(n_params)
        init[1] = 1.0
        with np.errstate(all="ignore"):
            res = minimize(
                neg_ll,
                init,
                method="Nelder-Mead",
                options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-9},
            )
            # The BFGS polish can diverge (e.g. from a boundary point on
            # doubly-truncated data); keep it only if it produced a finite
            # improvement over the Nelder-Mead result (#274).
            res_polish = minimize(neg_ll, res.x, method="BFGS")
            if np.isfinite(res_polish.fun) and res_polish.fun <= res.fun:
                res = res_polish
        if not np.isfinite(res.fun):
            raise ValueError(
                "Royston-Parmar optimisation failed to find a finite "
                "likelihood; the model may be unidentifiable for these data "
                "(check truncation bounds and df)."
            )
        gamma = res.x

        covariance = None
        n_obs = float(n_o.sum() + n_r.sum() + n_l.sum() + n_i.sum())
        with np.errstate(all="ignore"):
            steps = 1e-05 * np.maximum(np.abs(gamma), 1.0)
            information = numerical_hessian(neg_ll, gamma, step=steps)
            try:
                cov = np.linalg.inv(information)
                if np.all(np.isfinite(cov)):
                    covariance = cov
            except np.linalg.LinAlgError:
                covariance = None
            # Accepted as a maximum only where it is one: the gradient ~0
            # and the information (the Hessian the covariance inverts)
            # positive definite, per observation (principle 13).
            verified = is_local_minimum(
                neg_ll,
                lambda g: numerical_gradient(neg_ll, g, 1e-2 * steps),
                lambda g: information,
                gamma,
                obj_scale=max(n_obs, 1.0),
            )
        if not verified:
            warn_unverified("The Royston-Parmar fit")

        model = RoystonParmarModel()
        model.scale = scale
        model.knots = knots
        model.params = gamma
        model._covariance = covariance
        model.n = int(
            round(float(n_o.sum() + n_r.sum() + n_l.sum() + n_i.sum()))
        )
        model.n_events = int(round(float(n_o.sum())))
        model._neg_ll = float(res.fun)
        model._ic_n = ic_sample_size(data.c, data.n)
        model.maximum = "verified" if verified else "unverified"
        model._objective = neg_ll
        return model


RoystonParmar = RoystonParmar_()
