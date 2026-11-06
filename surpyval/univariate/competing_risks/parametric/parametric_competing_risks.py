"""
Parametric competing-risks model.

Where :class:`~surpyval.univariate.competing_risks.CompetingRisks` estimates
each cause's cumulative incidence non-parametrically (a step function),
``ParametricCompetingRisks`` fits a *parametric distribution* to each cause's
cause-specific hazard and assembles smooth, extrapolatable cumulative-incidence
functions from them.

The key fact that makes this simple and exact is that the parametric
cause-specific likelihood **factorises across causes**: the contribution of an
observation is the cause-specific density if it failed from that cause and the
cause-specific survival otherwise, so maximising the joint likelihood is the
same as fitting each cause's distribution independently with the *other*
causes' events treated as right-censored. The cumulative incidence of cause
:math:`k` is then

.. math::
    \\mathrm{CIF}_k(t) = \\int_0^t h_k(u)\\,S(u)\\,du
                       = \\int_0^t f_k(u)\\!\\!\\prod_{j\\neq k} S_j(u)\\,du,

with the all-cause survival :math:`S(u) = \\prod_j S_j(u) = \\exp(-\\sum_j
H_j(u))`. The cause CIFs sum to the all-cause failure probability
:math:`1 - S(t)`.

A left- or interval-censored row (a unit found failed from a known cause
at an inspection, some time since the one before) contributes
:math:`\\mathrm{CIF}_k(b) - \\mathrm{CIF}_k(a)`, which holds every cause's
parameters at once, so with such rows the causes are fitted together (see
:meth:`ParametricCompetingRisks.fit`).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import IntegrationWarning, quad, quad_vec
from scipy.optimize import brentq
from scipy.special import ndtri as z

from surpyval.diagnostics.mixin import DiagnoseMixin
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
from surpyval.univariate.competing_risks.parametric import (
    _interval_likelihood as joint,
)
from surpyval.univariate.information_criteria import corrected_aic
from surpyval.univariate.parametric import Weibull
from surpyval.univariate.parametric.parametric import Parametric
from surpyval.utils import (
    check_e_and_x,
    is_missing_event,
    resolve_cr_censoring,
    xcnt_handler,
)
from surpyval.utils.dataframe import frame_column
from surpyval.utils.linalg import (
    bound_signs,
    standard_errors_of,
    warn_wald_undefined,
)
from surpyval.utils.no_maximum import combined_maximum
from surpyval.utils.removed_names import column_arguments
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import (
    BOUNDS,
    check_alpha_ci,
    check_option,
    no_covariance_error,
    unknown_cause_error,
    validate_cif_event,
    warn_outside_unit_interval,
)
from surpyval.utils.warnings import caller_stacklevel

_CB_ON = ("cif", "sf", "ff", "Hf", "R", "F")
_CB_METHODS = ("wald", "lr", "bootstrap")


def _validate(
    x: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    e: npt.ArrayLike,
    tl: npt.ArrayLike | float | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Wrangle ``x``/``c``/``n``/``tl`` and check the event labels: ``e``
    is the per-observation cause. A missing event (``None`` / ``NaN``)
    marks a censored observation; if ``c`` is not given it is derived from
    the events. The truncation is checked as every fitter checks it (each
    row's time above its ``tl``) and returned as the ``(N, 2)`` ``t``.
    With a left- or interval-censored row, ``x`` is ``(N, 2)``: each
    row's ``[xl, xr]``.
    """
    if c is None:
        # A window given as [xl, xr] without flags: its flag from the times
        # (2, or -1 with no lower end), as every fit derives it, and 1 where
        # the row has no cause and the times say nothing more.
        x_derived, derived, _, _ = xcnt_handler(
            x, None, n, tl=tl, group_and_sort=False
        )
        if np.isin(derived, (-1, 2)).any():
            missing = np.fromiter(
                (is_missing_event(v) for v in np.asarray(e, dtype=object)),
                dtype=bool,
            )
            x = x_derived
            c = np.where(missing & (derived == 0), 1, derived)
    e, c = resolve_cr_censoring(e, c)
    x, c, n, t = xcnt_handler(x, c, n, tl=tl, group_and_sort=False)
    x, c, n = (np.asarray(a, dtype=float) for a in (x, c, n))
    e = np.asarray(e, dtype=object)
    check_e_and_x(e, x[:, 0] if x.ndim == 2 else x)
    if any(ev is not None for ev in e[c == 1]) or any(
        ev is None for ev in e[c != 1]
    ):
        raise ValueError(
            "A missing event type (None / NaN) is allowed only for a "
            "censored observation (c = 1), and every censored observation "
            "must have one: a failure, exact (c = 0) or in a window (c = -1 "
            "or 2), needs its cause in 'e'. (A window with no upper end, "
            "[xl, inf], is a unit right censored at xl.)"
        )
    return x, c, n, e, t


class ParametricCompetingRisks(
    DiagnoseMixin,
    SerialisableMixin,
):
    """
    A parametric competing-risks model: one distribution per cause, combined
    into cumulative-incidence functions. Build it in one step from data with
    :meth:`fit` / :meth:`fit_from_df`, or assemble it from already-fitted
    per-cause models -- each of any distribution family -- with
    :meth:`from_fitted`.
    """

    causes: list
    models: dict
    # The joint fit of data with left- or interval-censored rows: its
    # covariance (``None`` where the Hessian was not positive definite),
    # negative log-likelihood, parameter count and BIC sample size. ``None``
    # for a model fitted a cause at a time or assembled from fitted models.
    _joint: dict | None = None

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted parametric competing-risks model to a plain,
        JSON-serialisable dict: the list of causes and each cause's fitted
        distribution (via its own ``to_dict``). The reloaded model reproduces
        every cumulative-incidence / hazard function exactly. A model fitted
        to left- or interval-censored rows also stores its joint fit under
        ``"joint"``: the covariance over every cause's parameters, the
        negative log-likelihood, the number of parameters and the BIC's
        sample size. (A SurPyval before v0.27 reads such a dict without
        it: each cause's own block of the covariance, and none between
        causes.)
        """
        out: dict = {
            "model": "ParametricCompetingRisks",
            "causes": to_native(list(self.causes)),
            "models": [self.models[cause].to_dict() for cause in self.causes],
        }
        if self._joint is not None:
            fitted = {
                "neg_ll": float(self._joint["neg_ll"]),
                "k": int(self._joint["k"]),
                "n": float(self._joint["n"]),
            }
            if self._joint["covariance"] is not None:
                fitted["covariance"] = np.asarray(
                    self._joint["covariance"], dtype=float
                ).tolist()
            out["joint"] = fitted
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> ParametricCompetingRisks:
        """Rebuild a parametric competing-risks model from a dict."""
        require_model_tag(
            model_dict,
            "ParametricCompetingRisks",
            "a parametric competing-risks model",
        )
        out = cls()
        out.causes = [label_from_native(k) for k in model_dict["causes"]]
        out.models = {
            cause: Parametric.from_dict(sub)
            for cause, sub in zip(out.causes, model_dict["models"])
        }
        fitted = model_dict.get("joint")
        if fitted is not None:
            cov = fitted.get("covariance")
            out._joint = {
                "covariance": (
                    None if cov is None else np.asarray(cov, dtype=float)
                ),
                "neg_ll": float(fitted["neg_ll"]),
                "k": int(fitted["k"]),
                "n": float(fitted["n"]),
            }
        return out

    def __repr__(self) -> str:
        dists = ", ".join(
            f"{k}: {self.models[k].dist.name}" for k in self.causes
        )
        return (
            "Parametric Competing Risks SurPyval Model\n"
            "=========================================\n"
            "Causes              : {causes}\n"
            "Cause distributions : {dists}".format(
                causes=list(self.causes), dists=dists
            )
        )

    # -- cause-specific and all-cause functions ---------------------------

    def Hf(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """Cumulative hazard. ``event=None`` gives the all-cause cumulative
        hazard :math:`\\sum_k H_k`; ``event=k`` gives cause ``k``'s."""
        if event is not None:
            self._check_event(event)
            return self.models[event].Hf(x)
        return sum(self.models[k].Hf(x) for k in self.causes)

    def hf(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """Hazard rate. ``event=None`` is the all-cause hazard
        :math:`\\sum_k h_k`; ``event=k`` is the cause-specific hazard."""
        if event is not None:
            self._check_event(event)
            return self.models[event].hf(x)
        return sum(self.models[k].hf(x) for k in self.causes)

    def sf(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """
        Survival. ``event=None`` (the default) gives the all-cause
        survival :math:`S(t) = \\prod_k S_k(t)`, the probability of having
        failed from no cause by ``t``; ``event=k`` gives the net survival
        :math:`S_k(t)` from cause ``k`` alone, as if the other causes were
        removed. The net survival is *not* the probability of escaping
        cause ``k`` while the others still act -- use :meth:`cif` for
        that.

        Parameters
        ----------
        x : array_like or float
            Times at which to evaluate the survival.
        event : optional
            The cause; ``None`` for all causes combined.

        Returns
        -------
        numpy array or float
            The survival at each time, in the shape of ``x``.

        Examples
        --------
        >>> from surpyval import Exponential
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> model = ParametricCompetingRisks.from_fitted(
        ...     {"a": Exponential.from_params([0.1]),
        ...      "b": Exponential.from_params([0.3])}
        ... )
        >>> round(float(model.sf(2.0)), 6)  # exp(-0.4 * 2)
        0.449329
        >>> round(float(model.sf(2.0, event="a")), 6)  # exp(-0.1 * 2)
        0.818731
        """
        if event is not None:
            self._check_event(event)
            return self.models[event].sf(x)
        return np.exp(-self.Hf(x))

    def ff(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """
        Failure probability, ``1 - sf``. ``event=None`` (the default) gives
        the all-cause failure probability :math:`1 - S(t)`, the total
        cumulative incidence over all causes; ``event=k`` gives the net
        failure probability :math:`F_k(t) = 1 - S_k(t)` from cause ``k``
        with the others removed. For the probability of failing *from*
        cause ``k`` while the others still act, use :meth:`cif`.

        Parameters
        ----------
        x : array_like or float
            Times at which to evaluate the failure probability.
        event : optional
            The cause; ``None`` for all causes combined.

        Returns
        -------
        numpy array or float
            The failure probability at each time, in the shape of ``x``.

        Examples
        --------
        >>> from surpyval import Exponential
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> model = ParametricCompetingRisks.from_fitted(
        ...     {"a": Exponential.from_params([0.1]),
        ...      "b": Exponential.from_params([0.3])}
        ... )
        >>> round(float(model.ff(2.0)), 6)
        0.550671
        >>> round(float(model.ff(2.0, event="a")), 6)
        0.181269
        >>> round(float(model.cif(2.0, "a")), 6)  # a quarter of ff(2)
        0.137668
        """
        if event is not None:
            self._check_event(event)
            return self.models[event].ff(x)
        return -np.expm1(-self.Hf(x))

    def df(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """
        Probability density. ``event=None`` (the default) gives the density
        of the all-cause failure time (the time of the first failure from
        any cause),

        .. math::
            f(t) = S(t) \\sum_k h_k(t),

        which is also the sum of the causes' :meth:`iif`; ``event=k`` gives
        cause ``k``'s net density :math:`f_k(t)`, the density of its own
        model, matching ``sf(x, event=k)``.

        Parameters
        ----------
        x : array_like or float
            Times at which to evaluate the density.
        event : optional
            The cause; ``None`` for all causes combined.

        Returns
        -------
        numpy array or float
            The density at each time, in the shape of ``x``.

        Examples
        --------
        >>> from surpyval import Exponential
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> model = ParametricCompetingRisks.from_fitted(
        ...     {"a": Exponential.from_params([0.1]),
        ...      "b": Exponential.from_params([0.3])}
        ... )
        >>> model.df([0.0, 2.0]).round(6)  # 0.4 exp(-0.4 t)
        array([0.4     , 0.179732])
        """
        if event is not None:
            self._check_event(event)
            return self.models[event].df(x)
        with np.errstate(all="ignore"):
            s = self.sf(x)
            h = self.hf(x)
            # Where the survival has reached 0 the density is 0, even if
            # a hazard there is infinite (0 * inf).
            out = np.where(s == 0, 0.0, s * h)
        return out[()]

    def qf(self, p: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """
        Quantile function. ``event=None`` (the default) gives the time by
        which a fraction ``p`` of units has failed from any cause: the
        inverse of the all-cause :meth:`ff`, so the B10 life is
        ``qf(0.1)`` and the median ``qf(0.5)``. ``event=k`` gives cause
        ``k``'s net quantile (its own model's ``qf``, the inverse of
        ``ff(x, event=k)``).

        The all-cause failure function has no closed-form inverse, so the
        quantile is the root of :math:`H(t) = -\\log(1 - p)`, with
        :math:`H = \\sum_k H_k` the all-cause cumulative hazard, found by
        bracketed root finding (Brent's method) on :math:`\\log t`. It is
        the same equation as ``ff(t) = p``, since :math:`F = 1 -
        e^{-H}`, but on the cumulative-hazard scale it keeps every digit
        of a small survival, so ``qf(1 - 1e-12)`` is as precise as
        ``qf(0.5)``. The root lies between the causes' net quantiles: no
        later than the earliest cause's quantile at ``p``, and no earlier
        than the earliest at :math:`1 - (1 - p)^{1/K}` for ``K`` causes.

        Parameters
        ----------
        p : array_like or float
            Probabilities, in [0, 1].
        event : optional
            The cause; ``None`` for all causes combined.

        Returns
        -------
        numpy array or float
            The quantile at each probability, in the shape of ``p``.

        Notes
        -----
        If the causes carry a cure (limited-failure) fraction, the
        all-cause failure probability only reaches its ceiling
        :math:`1 - \\prod_k (1 - p_k)` in the limit, so a quantile at or
        above the ceiling is infinite, as a limited-failure ``Parametric``
        model's ``qf`` gives. ``qf(0)`` and ``qf(1)`` are the ends of the
        support. A probability outside [0, 1] gives NaN with one warning,
        as every other ``qf`` (it is most often a percentage given for a
        probability: ``qf(10)`` for the B10 life); NaN gives NaN.

        Examples
        --------
        >>> from surpyval import Exponential
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> model = ParametricCompetingRisks.from_fitted(
        ...     {"a": Exponential.from_params([0.1]),
        ...      "b": Exponential.from_params([0.3])}
        ... )
        >>> round(float(model.qf(0.1)), 6)  # -log(0.9) / 0.4
        0.263401
        >>> round(float(model.qf(0.1, event="a")), 6)  # -log(0.9) / 0.1
        1.053605
        """
        if event is not None:
            self._check_event(event)
            return self.models[event].qf(p)
        u = np.asarray(p, dtype=float)
        flat = np.atleast_1d(u).ravel()
        outside = warn_outside_unit_interval(flat)
        q = np.full(flat.shape, np.nan)
        ok = ~outside & ~np.isnan(flat)
        ceiling = self._ff_ceiling()
        q[ok & (flat == 0)] = self._support_end(0.0)
        if ceiling < 1:
            q[ok & (flat >= ceiling)] = np.inf
        else:
            q[ok & (flat == 1)] = self._support_end(1.0)
        for i in np.flatnonzero(ok & (flat > 0) & (flat < min(ceiling, 1))):
            q[i] = self._invert_Hf(float(flat[i]))
        return q.reshape(u.shape)[()]

    def mean(self, defective: bool = False) -> float:
        """
        The mean time to the first failure from any cause (the all-cause
        MTTF),

        .. math::
            \\mathbb{E}[T] = \\int_0^\\infty S(t)\\,dt,

        the area under the all-cause survival curve, by adaptive quadrature
        on a log-time axis, to a relative accuracy of about
        :math:`10^{-12}`.

        Parameters
        ----------
        defective : bool, optional
            Only matters when a cause has a limited-failure (cure)
            fraction, so that the all-cause failure probability levels off
            below 1. If :code:`False` (the default), the mean lifetime,
            which is infinite there, since some units never fail. If
            :code:`True`, the *defective* mean :math:`\\int t\\,dF(t)`,
            the integral over the units that fail, as
            ``Parametric.mean(defective=True)``; that is not the mean life
            of the units that fail, which divides it by the ceiling
            :math:`F(\\infty)`.

        Returns
        -------
        mean : float
            The mean of the all-cause failure time; ``inf`` where it is
            infinite.

        Notes
        -----
        With a single cause this is that cause's ``Parametric.mean()``;
        with exponential causes it is :math:`1 / \\sum_k \\lambda_k`. A net
        mean of one cause, with the others removed, is
        ``model.models[k].mean()``. A tail too heavy for the integral to
        converge (a survival falling no faster than :math:`1/t` at very
        long times, as a LogLogistic with shape at most 1 alone) gives
        ``inf``. Over a support reaching below 0 (a Normal cause), the
        negative part :math:`-\\int_{-\\infty}^0 F(t)\\,dt` is added.

        Examples
        --------
        >>> from surpyval import Exponential
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> model = ParametricCompetingRisks.from_fitted(
        ...     {"a": Exponential.from_params([0.1]),
        ...      "b": Exponential.from_params([0.3])}
        ... )
        >>> round(model.mean(), 10)  # 1 / (0.1 + 0.3)
        2.5
        """
        ceiling = self._ff_ceiling()
        if ceiling < 1 and not defective:
            # Some units never fail: E[T] is infinite, as Parametric.mean.
            return np.inf
        # S(inf), as sf computes it (exp(-H(inf))), so S - S(inf) is 0
        # in the cured tail rather than a rounding residue.
        s_end = 0.0
        if ceiling < 1:
            with np.errstate(all="ignore"):
                s_end = float(np.exp(-np.ravel(self.Hf(np.inf))[0]))
            if not np.isfinite(s_end):
                s_end = 1.0 - ceiling
        scale = float(self.qf(min(0.5, ceiling / 2)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        def upper(y: float) -> float:
            with np.errstate(all="ignore"):
                t = scale * np.exp(y)
                s = float(np.ravel(self.sf(t))[0]) - s_end
            if not np.isfinite(s) or s <= 8 * np.finfo(float).eps * s_end:
                return 0.0
            return t * s

        if self._heavy_tail(upper):
            return np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            total = sum(
                quad(upper, a, b, epsabs=0.0, epsrel=1e-12, limit=500)[0]
                for a, b in ((-np.inf, 0.0), (0.0, np.inf))
            )
            if self._support_end(0.0) < 0:
                # The part of the support below 0 counts negatively.
                def lower(t: float) -> float:
                    with np.errstate(all="ignore"):
                        f = float(np.ravel(self.ff(t))[0])
                    return f if np.isfinite(f) else 0.0

                total -= quad(lower, -np.inf, 0.0, epsrel=1e-12, limit=500)[0]
        return float(total)

    def _ff_ceiling(self) -> float:
        """``lim F(t)`` as ``t -> inf`` of the all-cause failure function:
        1, or ``1 - prod_k (1 - p_k)`` with cure fractions."""
        limits = [min(_ff_limit(m), 1.0) for m in self.models.values()]
        if max(limits) >= 1.0:
            return 1.0
        return float(-np.expm1(sum(np.log1p(-v) for v in limits)))

    def _support_end(self, p: float) -> float:
        """The all-cause support's lower (``p = 0``) or upper (``p = 1``)
        end: the earliest of the causes' ends (the first failure ends it)."""
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ends = [float(np.ravel(m.qf(p))[0]) for m in self.models.values()]
        return min(ends)

    def _invert_Hf(self, p: float) -> float:
        """The all-cause quantile at ``p`` in (0, ceiling): the root of
        ``Hf(t) = -log(1 - p)``, bracketed by the causes' quantiles (see
        :meth:`qf`)."""
        target = -np.log1p(-p)
        k = len(self.causes)
        p_low = float(-np.expm1(np.log1p(-p) / k))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lows = [
                float(np.ravel(m.qf(p_low))[0]) for m in self.models.values()
            ]
            highs = [float(np.ravel(m.qf(p))[0]) for m in self.models.values()]
        lo = min(v for v in lows if not np.isnan(v))
        hi = min(v for v in highs if not np.isnan(v))

        def resid(t: float) -> float:
            with np.errstate(all="ignore"):
                return float(np.ravel(self.Hf(t))[0]) - target

        if not np.isfinite(hi):
            # Every cause's own quantile is infinite (cure fractions below
            # p), yet together they pass p: widen until H reaches target.
            hi = max(lo, 1.0) * 2.0
            while resid(hi) < 0 and hi < 1e300:
                hi *= 10.0
        r_lo, r_hi = resid(lo), resid(hi)
        if r_lo >= 0:
            return lo
        if r_hi <= 0:
            return hi
        if lo > 0:
            y = brentq(
                lambda v: resid(np.exp(v)),
                np.log(lo),
                np.log(hi),
                xtol=1e-14,
                rtol=4 * np.finfo(float).eps,
                maxiter=500,
            )
            return float(np.exp(y))
        return float(
            brentq(
                resid,
                lo,
                hi,
                xtol=1e-14 * max(abs(lo), abs(hi), 1e-300),
                rtol=4 * np.finfo(float).eps,
                maxiter=500,
            )
        )

    @staticmethod
    def _heavy_tail(upper: Any) -> bool:
        """Whether ``int t S(t) dlog t`` diverges: the integrand ``t S(t)``
        (``upper`` on the log axis) still not falling at very long times,
        where a survival decaying as ``t^-a`` gives the slope ``1 - a``."""
        y1, y2 = np.log(1e12), np.log(1e14)
        g1, g2 = upper(y1), upper(y2)
        if g1 <= 0 or g2 <= 0:
            return False
        return bool((np.log(g2) - np.log(g1)) / (y2 - y1) > -1e-3)

    def iif(self, x: npt.ArrayLike, event: Any) -> npt.NDArray:
        """
        Instantaneous incidence function (the subdistribution density) of a
        cause: :math:`f_k^{\\mathrm{sub}}(t) = h_k(t) S(t) = f_k(t)
        \\prod_{j\\neq k} S_j(t)`.
        """
        self._check_event(event)
        x = np.asarray(x, dtype=float)
        others = np.ones_like(x, dtype=float)
        for j in self.causes:
            if j != event:
                others = others * self.models[j].sf(x)
        # A density may be singular at 0 (e.g. LogNormal); the grid includes
        # 0, so silence the harmless evaluation there.
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.models[event].df(x) * others

    def cif(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray | float:
        """
        Cumulative incidence function. ``event=k`` returns
        :math:`\\int_0^t f_k^{\\mathrm{sub}}(u)\\,du`, the probability of
        having failed from cause ``k`` by ``t``; ``event=None`` gives the
        all-cause incidence :math:`1 - S(t) = \\sum_k \\mathrm{CIF}_k(t)`.

        The integral is taken over the cause's own probability scale,

        .. math::
            \\mathrm{CIF}_k(t) = \\int_0^{F_k(t)}
            \\prod_{j \\neq k} S_j\\left(F_k^{-1}(p)\\right) dp,

        (the substitution :math:`p = F_k(u)`), by adaptive quadrature to a
        relative accuracy of about :math:`10^{-10}`. The integrand is
        bounded by 1 and monotone, so the result is accurate for every
        requested time independently of the others, over any span of
        times, and also where a cause's density is infinite (a Weibull
        shape below 1) or very heavy-tailed. The causes' CIFs sum to the
        all-cause ``ff`` to that accuracy. Each cause's model needs a
        quantile function ``qf``.

        Parameters
        ----------
        x : array_like or float
            Times at which to evaluate the incidence; ``inf`` gives the
            eventual probability of the cause (see
            :meth:`probability_of_cause`).
        event : optional
            The cause; ``None`` for all causes combined.

        Returns
        -------
        numpy array or float
            The cumulative incidence at each time, in the shape of ``x`` (a
            float for a scalar ``x``).
        """
        if event is None:
            return self.ff(x)
        self._check_event(event)
        x_arr = np.asarray(x, dtype=float)
        flat = np.atleast_1d(x_arr).ravel()
        model = self.models[event]
        others = [self.models[j] for j in self.causes if j != event]

        # F_k(x): the upper limit of the integral on the probability scale.
        # A time outside a model's support can give NaN (a Weibull's
        # log(-5)); no probability has accrued there. ``inf`` is the
        # model's limit, which a few families' formulas cannot evaluate
        # (inf / inf).
        with np.errstate(all="ignore"):
            top = np.asarray(model.ff(flat), dtype=float).ravel()
        top = np.where(np.isinf(flat) & (flat > 0), _ff_limit(model), top)
        top = np.where(np.isfinite(top), np.clip(top, 0.0, 1.0), 0.0)
        out = np.where(np.isnan(flat), np.nan, 0.0)
        todo = (top > 0) & ~np.isnan(flat)
        if todo.any():
            upper = top[todo]
            # The survival of the other causes at the far end of a cause's
            # support, used where its quantile is infinite.
            s_inf = [1.0 - _ff_limit(m) for m in others]

            def integrand(t: float) -> npt.NDArray:
                with np.errstate(all="ignore"):
                    s = np.asarray(model.qf(t * upper), dtype=float).ravel()
                val = np.ones_like(s)
                far = np.isposinf(s)
                for m, s_end in zip(others, s_inf):
                    with np.errstate(all="ignore"):
                        sj = np.asarray(m.sf(s), dtype=float).ravel()
                    val = val * np.where(far, s_end, sj)
                # Below a model's support its survival formula can give
                # NaN; it is 1 there.
                return np.where(np.isfinite(val), val, 1.0)

            integral, _ = quad_vec(
                integrand,
                0.0,
                1.0,
                epsabs=1e-13,
                epsrel=1e-11,
                norm="max",
                limit=2000,
            )
            out[todo] = upper * np.asarray(integral, dtype=float)
        if np.ndim(x) == 0:
            return float(out[0])
        return out.reshape(x_arr.shape)

    def probability_of_cause(self, event: Any) -> Any:
        """
        The eventual probability that a unit fails from ``event``,
        :math:`\\mathrm{CIF}_k(\\infty)`. These sum to one over all causes
        unless a cause has a cure (limited-failure) fraction, in which case
        they sum to the all-cause probability of ever failing.

        It is :meth:`cif` at ``inf``: the integral runs over the whole of
        the cause's probability scale, so no finite horizon is chosen and
        a very heavy-tailed cause (a LogNormal with a large :math:`\\sigma`)
        is as accurate as any other.
        """
        self._check_event(event)
        return self.cif(np.inf, event)

    def random(
        self, size: int, random_state: int | None = None
    ) -> npt.NDArray:
        """
        Draw ``size`` samples of ``(time, cause)`` from the model, using the
        latent-failure-time representation: draw a latent time from each cause
        and take the earliest, recording its cause.

        Each latent time is drawn by inverse-transform sampling through the
        cause's quantile function, so it works for *any* per-cause model,
        including limited-failure (cure) models -- there the quantile is
        infinite above the cure ceiling, so a draw in the cure region yields an
        infinite latent time. A unit whose every latent time is infinite never
        fails; it is returned with ``x = inf`` and cause ``None``.

        Returns a structured array with fields ``x`` and ``e``.
        """
        rng = as_generator(random_state)
        latent = np.column_stack(
            [
                np.ravel(self.models[k].qf(rng.uniform(size=size)))
                for k in self.causes
            ]
        )
        idx = np.argmin(latent, axis=1)
        x = latent[np.arange(size), idx]
        e = np.array([self.causes[i] for i in idx], dtype=object)
        # A unit with no finite latent time never fails from any cause.
        never = ~np.isfinite(x)
        if never.any():
            e[never] = None
        out = np.empty(size, dtype=[("x", float), ("e", object)])
        out["x"] = x
        out["e"] = e
        return out

    # -- parameters, their covariance and the model at other values -------

    @property
    def params(self) -> npt.NDArray:
        """
        Every cause's parameters in one vector: each cause's ``params``
        (its distribution's parameters), cause by cause in the order of
        :attr:`causes`. :attr:`parameter_names` names them. The rows of
        :meth:`covariance` are the longer :attr:`covariance_names`, which
        add a cause's ``lfp_p`` and ``f0`` where it has them.

        Examples
        --------
        >>> from surpyval import Exponential
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> model = ParametricCompetingRisks.from_fitted(
        ...     {"a": Exponential.from_params([0.1]),
        ...      "b": Exponential.from_params([0.3])}
        ... )
        >>> model.params
        array([0.1, 0.3])
        """
        return np.concatenate(
            [
                np.ravel(np.asarray(self.models[k].params, dtype=float))
                for k in self.causes
            ]
        )

    @property
    def parameter_names(self) -> list[str]:
        """
        The names of :attr:`params`, entry by entry: ``"<cause>: <name>"``
        for each cause's ``parameter_names``.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> model = ParametricCompetingRisks.from_fitted(
        ...     {"a": Weibull.from_params([10, 2]),
        ...      "b": Weibull.from_params([15, 1])}
        ... )
        >>> model.parameter_names
        ['a: alpha', 'a: beta', 'b: alpha', 'b: beta']
        """
        return [
            f"{k}: {name}"
            for k in self.causes
            for name in self.models[k].parameter_names
        ]

    @property
    def covariance_names(self) -> list[str]:
        """
        The names of the rows (and columns) of :meth:`covariance`, in its
        order: ``"<cause>: <name>"`` for each cause's own
        ``covariance_names`` (its distribution's parameters, then
        ``lfp_p`` and ``f0`` where it has them), cause by cause. A
        cause's offset ``gamma`` is not a row, as it is not in the cause's
        own covariance; it is held at its estimate.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> model = ParametricCompetingRisks.from_fitted(
        ...     {"a": Weibull.from_params([10, 2], lfp_p=0.6),
        ...      "b": Weibull.from_params([15, 1])}
        ... )
        >>> model.covariance_names
        ['a: alpha', 'a: beta', 'a: lfp_p', 'b: alpha', 'b: beta']
        """
        return [
            f"{k}: {name}"
            for k in self.causes
            for name in _cause_covariance_names(self.models[k])
        ]

    def _estimates(self) -> npt.NDArray:
        """The fitted parameters in the order of :attr:`covariance_names`:
        each cause's in turn."""
        return np.concatenate(
            [_cause_estimates(self.models[k]) for k in self.causes]
        )

    def covariance(self) -> npt.NDArray:
        """
        The covariance of every cause's fitted parameters, in the order of
        :attr:`covariance_names`: the causes' own ``covariance()`` on the
        diagonal, and zeros between causes.

        The zeros are exact, not an approximation: the competing-risks
        likelihood is a product of one factor per cause (each cause's
        density where it failed, its survival elsewhere), so each cause's
        parameters are estimated from their own factor and the
        information matrix is block diagonal. With left- or
        interval-censored rows it is not: those rows tie the causes
        together, the causes are fitted jointly (see :meth:`fit`), and
        the covariance is the inverse of the joint Hessian, with the
        terms between causes; each cause's ``covariance()`` is then its
        block of it. A cause fitted with
        ``lfp=True`` or ``zi=True`` adds its ``lfp_p`` or ``f0`` row; an
        offset ``gamma`` has no row (see :attr:`covariance_names`).

        Returns
        -------
        numpy array
            The ``(P, P)`` covariance, ``P`` the number of
            :attr:`covariance_names`.

        Raises
        ------
        ValueError
            Where a cause has no covariance (fitted by ``how="MPS"``,
            ``"MSE"`` or ``"MPP"``, built from parameters, or with a
            singular Hessian), naming the cause; or where a joint fit's
            Hessian was not positive definite at its optimum.

        Examples
        --------
        >>> from surpyval import Exponential
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> e = ['a', 'b', 'a', None, 'a', 'b', 'a', None, 'b', 'a']
        >>> model = ParametricCompetingRisks.fit(x, e, dist=Exponential)
        >>> model.covariance().round(6)  # rate^2 / failures, per cause
        array([[0.001653, 0.      ],
               [0.      , 0.000992]])
        """
        if self._joint is not None:
            cov = self._joint["covariance"]
            if cov is None:
                raise no_covariance_error(
                    "the joint fit's Hessian was not positive definite at "
                    "its optimum"
                )
            return np.array(cov, dtype=float)
        blocks = [self._cause_covariance(k) for k in self.causes]
        size = sum(b.shape[0] for b in blocks)
        out = np.zeros((size, size))
        start = 0
        for b in blocks:
            stop = start + b.shape[0]
            out[start:stop, start:stop] = b
            start = stop
        return out

    def _cause_covariance(self, k: Any) -> npt.NDArray:
        """Cause ``k``'s covariance, or the refusal naming it."""
        model = self.models[k]
        names = _cause_covariance_names(model)
        try:
            cov = np.atleast_2d(np.asarray(model.covariance(), dtype=float))
        except (ValueError, AttributeError) as err:
            raise no_covariance_error(
                f"cause {k!r} has none: its model was not fitted by MLE, "
                "was built from parameters, or its Hessian was singular at "
                "the optimum"
            ) from err
        if cov.shape != (len(names), len(names)):
            raise no_covariance_error(
                f"cause {k!r}'s covariance has shape {cov.shape}, not one "
                f"row per name in {names}"
            )
        return cov

    def standard_errors(self) -> npt.NDArray:
        """
        The standard errors of every cause's fitted parameters, the square
        roots of the diagonal of :meth:`covariance`, in the order of
        :attr:`covariance_names`; ``nan`` where a variance is not positive.
        Raises the ``ValueError`` of :meth:`covariance` where a cause has
        none.

        Examples
        --------
        >>> from surpyval import Exponential
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> e = ['a', 'b', 'a', None, 'a', 'b', 'a', None, 'b', 'a']
        >>> model = ParametricCompetingRisks.fit(x, e, dist=Exponential)
        >>> model.standard_errors().round(4)
        array([0.0407, 0.0315])
        """
        return standard_errors_of(self.covariance())

    def with_params(
        self, params: npt.ArrayLike, gamma: dict | None = None
    ) -> ParametricCompetingRisks:
        """
        The same model -- its causes, each cause's distribution, offset,
        limited-failure and zero-inflation options and unit -- at other
        parameter values: for propagating the fit's uncertainty by drawing
        parameters from :meth:`covariance`, say.

        Parameters
        ----------
        params : array like
            A flat vector in the order of :attr:`covariance_names` (each
            cause's distribution parameters, then its ``lfp_p`` and ``f0``
            where it has them, cause by cause), as a draw from
            :meth:`covariance` comes; or, shorter, in the order of
            :attr:`parameter_names` (the distributions' parameters only),
            keeping each cause's ``lfp_p`` and ``f0``.
        gamma : dict, optional
            ``{cause: offset}`` for causes fitted with an offset; a cause
            left out keeps its fitted ``gamma``, which no covariance row
            holds.

        Returns
        -------
        ParametricCompetingRisks
            A new model built from parameters, as :meth:`from_fitted` of
            ``from_params`` models: it has no data, covariance or fitted
            likelihood, so its ``cb``, ``covariance`` and ``aic`` refuse.
            A fit to data with counts is just the distributions at those
            parameters. The model itself is not changed.

        Raises
        ------
        ValueError
            If ``params`` has neither length, naming both and the names;
            if a value is outside its parameter's range (as
            ``from_params`` checks it: a draw can give an ``lfp_p`` above
            1); or if a cause's model cannot be rebuilt from parameters.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> model = ParametricCompetingRisks.from_fitted(
        ...     {"a": Weibull.from_params([10, 2], lfp_p=0.6),
        ...      "b": Weibull.from_params([15, 1])}
        ... )
        >>> other = model.with_params([12, 2, 0.5, 15, 1])
        >>> other.models["a"].params, other.models["a"].lfp_p
        (array([12.,  2.]), 0.5)
        >>> model.with_params([12, 2, 15, 1]).models["a"].lfp_p
        0.6
        """
        values = np.asarray(params, dtype=float)
        full = [
            len(_cause_covariance_names(self.models[k])) for k in self.causes
        ]
        short = [len(self.models[k].parameter_names) for k in self.causes]
        if values.ndim != 1 or values.size not in (sum(full), sum(short)):
            expected = (
                "{} values in the order of covariance_names ({})".format(
                    sum(full), ", ".join(self.covariance_names)
                )
            )
            if sum(short) != sum(full):
                expected += (
                    ", or {} in the order of parameter_names ({})".format(
                        sum(short), ", ".join(self.parameter_names)
                    )
                )
            raise ValueError(
                "'params' must be a flat vector of {}; got shape {}".format(
                    expected, np.shape(params)
                )
            )
        sizes = full if values.size == sum(full) else short
        offsets = {} if gamma is None else dict(gamma)
        for k in offsets:
            self._check_event(k)
        models = {}
        start = 0
        for k, size in zip(self.causes, sizes):
            models[k] = _rebuild_cause(
                k, self.models[k], values[start : start + size], offsets.get(k)
            )
            start += size
        return type(self).from_fitted(models)

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        event: Any = None,
        on: str = "cif",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        r"""
        Pointwise confidence bounds on the cumulative incidence of cause
        ``event`` (``on="cif"``, the default), or on the survival, failure
        probability or cumulative hazard (``on="sf"``, ``"ff"``, ``"Hf"``)
        of all causes (``event=None``) or one cause's net ones, as
        ``CompetingRisks.cb`` bounds them for the non-parametric fit.

        The Wald bounds carry the uncertainty of every cause's parameters
        (:meth:`covariance`) to the function by the delta method, on a
        scale that keeps them in range:

        - the all-cause ``sf``, ``ff`` and ``Hf`` share one bound, on
          :math:`\log H(t)`, the log of the all-cause cumulative hazard
          (``log(-log sf)``, the scale of the univariate bounds). As
          :math:`H = \sum_k H_k` and the causes' parameters are
          independent, :math:`\mathrm{Var}\,H = \sum_k \mathrm{Var}\,H_k`,
          each from its cause's analytic gradient. A model fitted to
          left- or interval-censored rows has covariance between causes:
          there the gradient is over every parameter at once, by central
          differences as for the cumulative incidence below;
        - the cumulative incidence on the logit scale,
          :math:`\log(\mathrm{CIF}/(1 - \mathrm{CIF}))`. It depends on
          every cause's parameters through an integral, so its gradient
          is taken by central differences on :attr:`covariance_names`
          (each step :math:`10^{-5}` of the parameter's value or standard
          error, whichever is larger);
        - one cause's net ``sf``, ``ff`` and ``Hf`` (``event=k``) are
          functions of that cause's parameters alone: the bound is the
          cause's own ``models[k].cb``, which also offers
          ``method="lr"`` and ``"bootstrap"`` (not for a model fitted to
          left- or interval-censored rows, whose causes were fitted
          jointly: there the cause's model carries its block of the
          joint covariance and its Wald bound only).

        The bounds hold at each ``x`` separately.

        Parameters
        ----------
        x : array_like
            The times at which to bound the function.
        event : optional
            The cause; required for ``on="cif"``. For ``sf``, ``ff`` and
            ``Hf``, ``None`` (the default) is all causes and a cause its
            net function.
        on : ('cif', 'sf', 'ff', 'Hf'), str, optional
            The function to bound ('R' and 'F' are accepted for 'sf' and
            'ff'). Defaults to 'cif'.
        alpha_ci : float, optional
            The significance level: 0.05 (the default) gives a 95%
            interval.
        bound : ('two-sided', 'lower', 'upper'), str, optional
            The two-sided interval (the default), with one row per ``x``
            and the columns ``[lower, upper]``, or one side, which puts all
            of ``alpha_ci`` on that side.
        method : ('wald', 'lr', 'bootstrap'), str, optional
            ``"wald"`` (the default). ``"lr"`` and ``"bootstrap"`` are
            available for a cause's net function (``event=k``, ``on`` not
            ``"cif"``), from its own model; elsewhere they raise a
            ``ValueError`` saying so.
        n_boot : int, optional
            The number of bootstrap refits, with ``method="bootstrap"``.
        random_state : int or numpy Generator, optional
            The seed of the bootstrap's simulations.

        Returns
        -------
        numpy array
            The bounds at each ``x``: ``[lower, upper]`` rows for a
            two-sided bound, one value per ``x`` for one side, in the shape
            of ``x``.

        Raises
        ------
        ValueError
            If ``on``, ``bound`` or ``method`` is not one of the values
            above, ``alpha_ci`` is not between 0 and 1, ``event`` is not a
            cause of the model, ``on="cif"`` is asked without an
            ``event``, or a cause has no covariance (naming it).

        Notes
        -----
        Where the function is 0 (before any failure can happen) or its
        variance is 0, both bounds are the estimate. Where the
        delta-method variance is negative or not finite the bound is
        ``nan``, with a warning. A cause with an offset ``gamma`` holds it
        at its estimate, with that cause's warning, as
        ``Parametric.cb`` does. A simulation study
        (``scripts/studies/parametric_competing_risks_cb_coverage.py``)
        checks the coverage.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        >>> e = ['a', 'b', 'a', None, 'a', 'b', 'a', 'b'] * 2
        >>> model = ParametricCompetingRisks.fit(x, e, dist=Weibull)
        >>> round(model.cif(5, 'a'), 4)
        0.1598
        >>> model.cb(5, 'a').round(4)
        array([0.0566, 0.3762])
        >>> model.cb(5, 'a', bound='upper').round(4)
        np.float64(0.3338)

        The all-cause survival:

        >>> model.cb([5, 10], on='sf').round(4)
        array([[0.5365, 0.8956],
               [0.1963, 0.576 ]])
        """
        check_alpha_ci(alpha_ci)
        check_option("on", on, _CB_ON)
        check_option("bound", bound, BOUNDS)
        check_option("method", str(method).lower(), _CB_METHODS)
        method = str(method).lower()
        if event is not None:
            self._check_event(event)
        on = {"R": "sf", "F": "ff"}.get(on, on)
        t = np.atleast_1d(np.asarray(x, dtype=float))
        if on != "cif" and event is not None:
            if self._joint is not None and method != "wald":
                raise ValueError(
                    f"method={method!r} refits one cause on its own, but "
                    "this model was fitted to left- or interval-censored "
                    "rows, whose likelihood holds every cause at once: "
                    "use method='wald', which carries the joint "
                    "covariance."
                )
            # A net function: the cause's own model and its own bounds.
            try:
                return np.asarray(
                    self.models[event].cb(
                        t,
                        on=on,
                        alpha_ci=alpha_ci,
                        bound=bound,
                        method=method,
                        n_boot=n_boot,
                        random_state=random_state,
                    ),
                    dtype=float,
                )
            except ValueError as err:
                raise ValueError(f"cause {event!r}: {err}") from err
        if on == "cif":
            validate_cif_event(event)
        if method != "wald":
            raise ValueError(
                f"method={method!r} is available for one cause's net sf, "
                "ff and Hf (cb(x, event=k, on='sf', method=...)), from that "
                "cause's model; the cumulative incidence and the all-cause "
                "functions depend on every cause's parameters at once and "
                "have Wald bounds only: use method='wald', or the "
                "non-parametric CompetingRisks.cb on the same data."
            )
        if t.size == 0:
            return np.empty((0, 2) if bound == "two-sided" else (0,))
        cov = self.covariance()
        for k in self.causes:
            warn_offset = getattr(self.models[k], "_warn_offset_wald", None)
            if callable(warn_offset):
                warn_offset("cb")
        with np.errstate(all="ignore"):
            if on == "cif":
                return self._cb_cif(t, event, cov, alpha_ci, bound)
            return self._cb_all_cause(t, on, alpha_ci, bound)

    def _cb_all_cause(
        self, t: npt.NDArray, on: str, alpha_ci: float, bound: str
    ) -> npt.NDArray:
        """The Wald bound on the all-cause ``sf``, ``ff`` or ``Hf``: one
        bound on ``log H``, with ``Var H`` the sum of the causes' (over
        the joint covariance where the causes were fitted together)."""
        H = np.asarray(self.Hf(t), dtype=float)
        if self._joint is not None:
            # Covariance between causes: the gradient over every one.
            cov = self.covariance()

            def H_of(values: npt.NDArray) -> npt.NDArray:
                return np.asarray(self.with_params(values).Hf(t), float)

            grad = _central_differences(H_of, self._estimates(), cov, t.size)
            var = np.einsum("jt,jk,kt->t", grad, cov, grad)
        else:
            var = np.zeros(t.shape)
            for k in self.causes:
                var = var + self._var_Hf_cause(k, t)
        ok = np.isfinite(H) & (H > 0)
        sd = np.where(ok, _checked_sd(var, t, "Hf", ok), 0.0) / np.where(
            ok, H, 1.0
        )
        alpha, signs = bound_signs(alpha_ci, bound)
        if on == "sf":
            # sf falls as H rises: its lower bound is at H's upper one.
            signs = -signs
        H_b = H[:, None] * np.exp(signs[None, :] * z(1 - alpha) * sd[:, None])
        if on == "Hf":
            out = H_b
        elif on == "sf":
            out = np.exp(-H_b)
        else:
            out = -np.expm1(-H_b)
        return out if bound == "two-sided" else out[:, 0]

    def _var_Hf_cause(self, k: Any, t: npt.NDArray) -> npt.NDArray:
        """The delta-method variance of cause ``k``'s cumulative hazard at
        ``t``: from the analytic (autograd) gradient of a ``Parametric``
        cause, as its own ``cb`` takes it, else by central differences."""
        model = self.models[k]
        if isinstance(model, Parametric):
            ctx = model._cb_context()
            finite = np.isfinite(t)
            var = np.full(t.shape, np.nan)
            if finite.any():
                tf = t[finite]

                def func_at_f(idx: Any) -> Any:
                    ti = tf[idx]
                    return lambda phi: model._cb_full_Hf(ti, phi, ctx)

                var[finite] = model._cb_delta_var_each(
                    func_at_f, ctx, int(tf.size)
                )
            return var
        cov = self._cause_covariance(k)
        est = _cause_estimates(model)

        def H_of(values: npt.NDArray) -> npt.NDArray:
            rebuilt = _rebuild_cause(k, model, values, None)
            return np.asarray(rebuilt.Hf(t), dtype=float)

        grad = _central_differences(H_of, est, cov, t.size)
        return np.einsum("jt,jk,kt->t", grad, cov, grad)

    def _cb_cif(
        self,
        t: npt.NDArray,
        event: Any,
        cov: npt.NDArray,
        alpha_ci: float,
        bound: str,
    ) -> npt.NDArray:
        """The Wald bound on cause ``event``'s cumulative incidence, on
        the logit scale, from its numerical gradient."""
        F = np.asarray(self.cif(t, event), dtype=float)

        def cif_of(values: npt.NDArray) -> npt.NDArray:
            return np.asarray(
                self.with_params(values).cif(t, event), dtype=float
            )

        grad = _central_differences(cif_of, self._estimates(), cov, t.size)
        var = np.einsum("jt,jk,kt->t", grad, cov, grad)
        ok = (F > 0) & (F < 1) & np.isfinite(F)
        sd = np.where(ok, _checked_sd(var, t, "cif", ok), 0.0)
        sd_logit = sd / np.where(ok, F * (1 - F), 1.0)
        alpha, signs = bound_signs(alpha_ci, bound)
        logit_F = np.log(np.where(ok, F, 0.5)) - np.log1p(
            -np.where(ok, F, 0.5)
        )
        moved = logit_F[:, None] + signs[None, :] * z(1 - alpha) * (
            sd_logit[:, None]
        )
        out = np.where(ok[:, None], 1.0 / (1.0 + np.exp(-moved)), F[:, None])
        return out if bound == "two-sided" else out[:, 0]

    # -- goodness of fit (the joint likelihood factorises over causes) ----

    @property
    def maximum(self) -> str:
        """What the causes' fits reached, one of ``MAXIMUM_STATES``
        (``surpyval.utils.no_maximum``): the worst of the causes' own
        ``maximum``, since the joint likelihood is maximised a cause at a
        time (principles 12 and 13); with left- or interval-censored rows,
        what the joint fit reached, which every cause's model carries.

        Examples
        --------
        >>> from surpyval import Exponential
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> e = ["a", "b", "a", "b", "a", "b", "a", "b"]
        >>> ParametricCompetingRisks.fit(x, e, dist=Exponential).maximum
        'verified'
        """
        return combined_maximum(
            getattr(self.models[k], "maximum", "unknown") for k in self.causes
        )

    def neg_ll(self) -> float:
        """Total negative log-likelihood: the sum over the per-cause fits,
        or the joint fit's where there are left- or interval-censored
        rows."""
        if self._joint is not None:
            return float(self._joint["neg_ll"])
        return float(sum(self.models[k].neg_ll() for k in self.causes))

    def aic(self) -> float:
        """Akaike information criterion of the joint model,
        ``2 K + 2 neg_ll`` with ``K`` the number of parameters estimated
        over all causes: the sum of the causes' AICs, since both terms
        add over causes."""
        if self._joint is not None:
            return float(2 * self._joint["k"] + 2 * self._joint["neg_ll"])
        return float(sum(self.models[k].aic() for k in self.causes))

    def bic(self) -> float:
        """Bayesian information criterion of the joint model.

        ``2 neg_ll + K ln(n)``, with ``K`` the number of parameters
        estimated over all causes and ``n`` the sample size every SurPyval
        BIC uses, counted on the whole data: the observed failures of any
        cause, weighted by their counts (right-censored units add
        nothing). That is the sum of the causes' own sample sizes, since
        each counts the failures of its cause; for a model assembled with
        :meth:`from_fitted` it is that sum, which is the whole data's count
        when each model was fitted to the cause-specific view of the same
        data.

        It is not the sum of the causes' BICs, which charged each cause's
        parameters ``ln`` of its own cause's failures (and was what this
        method returned before v0.21), a smaller penalty.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Exponential
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> e = ['a', 'b', 'a', None, 'a', 'b', 'a', None, 'b', 'a']
        >>> model = ParametricCompetingRisks.fit(x, e, dist=Exponential)
        >>> round(model.bic(), 4)
        61.5902
        >>> round(float(2 * model.neg_ll() + 2 * np.log(8)), 4)
        61.5902
        """
        k_total, n_total = self._ic_terms()
        return float(2 * self.neg_ll() + k_total * np.log(n_total))

    def aic_c(self) -> float:
        """The small-sample corrected AIC of the joint model, ``aic() +
        (2K^2 + 2K) / (n - K - 1)``, with the ``K`` and ``n`` of
        :meth:`bic`; ``nan`` where ``n <= K + 1`` (``corrected_aic``), as
        on every other model (#605)."""
        k_total, n_total = self._ic_terms()
        return corrected_aic(self.aic(), k_total, n_total)

    @property
    def log_likelihood(self) -> float:
        """The maximised log-likelihood of the joint model, ``-neg_ll()``:
        the sum of the causes' (#605)."""
        return -self.neg_ll()

    def _ic_terms(self) -> tuple[int, float]:
        # The parameters estimated over every cause, and the sample size of
        # the whole data (see bic): every failure, in a window or not.
        if self._joint is not None:
            return int(self._joint["k"]), float(self._joint["n"])
        k_total, n_total = 0, 0.0
        for k in self.causes:
            k_cause, n_cause = _ic_terms(self.models[k])
            k_total += k_cause
            n_total += n_cause
        return k_total, n_total

    # -- helpers ----------------------------------------------------------

    def _check_event(self, event: Any) -> None:
        if event not in self.models:
            raise unknown_cause_error(event, self.causes)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_fitted(cls, models: dict) -> ParametricCompetingRisks:
        """
        Assemble a competing-risks model from already-fitted single-cause
        models -- one per cause -- instead of fitting them here.

        Each cause's model may be of a completely different family: a Weibull
        with a limited-failure (cure) fraction for one cause, a LogNormal for
        another, a discrete distribution for a third, and so on. The only
        requirement is that every model exposes the standard surpyval model
        interface (``sf`` / ``ff`` / ``df`` / ``hf`` / ``Hf``, and the
        quantile function ``qf``, over which the cumulative incidence is
        integrated and the samples are drawn); the cumulative incidence,
        all-cause survival and sampling are then assembled from them exactly
        as for a :meth:`fit` model.

        This is the right entry point when each cause has been modelled
        separately -- for example fitted with its own distribution, offset,
        limited-failure or zero-inflated options -- and you want to combine
        them into one competing-risks object.

        Parameters
        ----------
        models : dict or sequence
            Either a ``{cause: model}`` mapping, or a sequence of models whose
            causes are taken to be their positions ``0, 1, 2, ...``.

        Returns
        -------
        ParametricCompetingRisks
            The assembled model.

        Notes
        -----
        Each per-cause model should be fitted to the *cause-specific* view of
        the data (that cause's events observed, every other cause's events and
        every censored unit treated as right-censored) for the assembled CIFs
        to be the competing-risks quantities. If the causes carry a cure
        fraction the all-cause survival need not fall to zero, so the cause
        probabilities need not sum to one -- some units never fail.
        """
        if isinstance(models, dict):
            mapping = dict(models)
        else:
            mapping = {i: m for i, m in enumerate(models)}
        if len(mapping) == 0:
            raise ValueError("At least one cause model is required.")
        for k, m in mapping.items():
            missing = [
                a
                for a in ("sf", "ff", "df", "hf", "Hf", "qf")
                if not callable(getattr(m, a, None))
            ]
            if missing:
                raise ValueError(
                    "Model for cause {!r} is missing the method(s) {}; it "
                    "does not look like a fitted surpyval model.".format(
                        k, missing
                    )
                )
        causes = ordered_labels(mapping)

        model = cls()
        model.causes = causes
        model.models = mapping
        return model

    @classmethod
    @takes_unit
    @accepts_surpyval_data
    def fit(
        cls,
        x: npt.ArrayLike,
        e: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        dist: Any = Weibull,
        how: str = "MLE",
        tl: npt.ArrayLike | float | None = None,
    ) -> ParametricCompetingRisks:
        """
        Fit a parametric distribution to each cause's cause-specific hazard.

        Parameters
        ----------
        x : array_like
            Observed times; for an interval-censored row, its ``[xl, xr]``
            (a row of two values, as every SurPyval fit takes it).
        e : array_like
            The cause of each observation: any hashable labels (integers,
            strings, tuples, or a mix), kept in sorted order in ``causes``.
            A missing value (``None`` / ``NaN``) marks a censored
            observation with no attributed cause.
        c : array_like, optional
            Censoring flag: 0 a failure at ``x``, 1 right censored (still
            working at ``x``), -1 left censored (found failed from its
            cause at ``x``, at some time before, or since ``tl``), 2
            interval censored (failed from its cause in ``(xl, xr]``). If
            omitted it is derived from ``e`` -- a missing event is
            censored, an event present is observed -- so data can be
            passed as ``(x, e)`` alone. A left- or interval-censored row
            needs its cause; with any such row the causes are fitted
            jointly (see Notes).
        n : array_like, optional
            Counts per observation.
        dist : ParametricFitter or dict, optional
            The distribution fitted to each cause (default ``Weibull``). Pass a
            ``{cause: distribution}`` mapping, with an entry for every cause,
            to use a different distribution per cause.
        how : str, optional
            Estimation method passed to each distribution's ``fit`` (default
            ``"MLE"``).
        tl : array_like or scalar, optional
            Left truncation (delayed entry): the time each unit came under
            observation, a scalar for every unit, as the univariate
            ``fit`` takes it. A unit seen only from ``tl`` (say, when the
            maintenance records start) counts as at risk only after it.

        Returns
        -------
        ParametricCompetingRisks
            The fitted model.

        Notes
        -----
        Left truncation keeps the likelihood factorising across causes:
        a unit entering at :math:`t_l` contributes
        :math:`\\prod_k h_k(x)^{\\delta_k} S_k(x) / S_k(t_l)`, so each cause's
        distribution is fitted with the same ``tl``.

        Left and interval censoring do not factorise. A unit found failed
        from cause :math:`k` in :math:`(x_l, x_r]` contributes the
        probability of that, :math:`(\\mathrm{CIF}_k(x_r) -
        \\mathrm{CIF}_k(x_l)) / S(t_l)`, an integral of the all-cause
        survival (a left-censored row is the window from its entry
        ``tl``, or from the start). So with any such row the causes are
        fitted together, by maximum likelihood over every cause's
        parameters at once (``how="MLE"`` only, continuous
        distributions); the integral is taken by Gauss-Legendre
        quadrature on the cause's probability scale. The covariance is
        then the inverse of the joint Hessian, with terms between causes
        (see :meth:`covariance`), and each cause's model in ``models``
        carries the parameters and its block of it, but not the data, so
        its own ``neg_ll``, ``aic``, likelihood-ratio and bootstrap
        bounds are not available: use this model's. Fitting each cause as
        a univariate model with the other causes' events right censored
        would be an approximation, not this model's likelihood (it is the
        starting point of the search). Without such rows the fit is a
        cause at a time, as before. Right truncation is not taken.

        Examples
        --------
        >>> from surpyval import Exponential
        >>> from surpyval.univariate.competing_risks import (
        ...     ParametricCompetingRisks,
        ... )
        >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> e = ['a', 'b', 'a', None, 'a', 'b', 'a', None, 'b', 'a']
        >>> model = ParametricCompetingRisks.fit(x, e, dist=Exponential)
        >>> model.cif([5, 10], 'a').round(4)
        array([0.323 , 0.4791])
        >>> round(model.probability_of_cause('a'), 4)
        0.625

        With failures found at inspections -- a window ``[xl, xr]``
        flagged 2, or one found at the first inspection flagged -1 -- the
        causes are fitted together, and their estimates are correlated:

        >>> x = [[2, 2], [2, 4], [4, 6], [5, 5], [6, 8], [7, 7], [8, 10],
        ...      [9, 9], [10, 10], [12, 12]]
        >>> e = ['a', 'b', 'a', 'b', 'a', 'a', 'b', None, 'a', None]
        >>> c = [-1, 2, 2, 0, 2, 0, 2, 1, 0, 1]
        >>> joint = ParametricCompetingRisks.fit(x, e, c=c, dist=Exponential)
        >>> joint.params.round(4)
        array([0.0737, 0.0442])
        >>> joint.covariance().round(6)
        array([[1.09e-03, 1.00e-06],
               [1.00e-06, 6.53e-04]])
        """
        x, c, n, e, t = _validate(x, c, n, e, tl)
        truncated = tl is not None
        if ((c == -1) | (c == 2)).any():
            return cls._fit_joint(x, c, n, e, t, dist, how)

        causes = ordered_labels(e[c == 0])
        if not causes:
            raise ValueError("No observed events to fit a cause to.")
        if isinstance(dist, dict):
            missing = [k for k in causes if k not in dist]
            if missing:
                raise ValueError(
                    "`dist` has no distribution for the cause(s) {}; give "
                    "one for every cause {}.".format(missing, causes)
                )

        models = {}
        for k in causes:
            # Cause k observed where its event occurred; every other event and
            # every censored row is right-censored for cause k.
            c_k = np.where(label_mask(e, k) & (c == 0), 0, 1).astype(int)
            distribution = dist[k] if isinstance(dist, dict) else dist
            # The same entry time for every cause (left truncation
            # factorises; see Notes).
            extra = {"t": t} if truncated else {}
            models[k] = distribution.fit(x=x, c=c_k, n=n, how=how, **extra)

        model = cls()
        model.causes = causes
        model.models = models
        return model

    @classmethod
    def _fit_joint(
        cls,
        x: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        e: npt.NDArray,
        t: npt.NDArray,
        dist: Any,
        how: str,
    ) -> ParametricCompetingRisks:
        """The fit of data with left- or interval-censored rows: every
        cause's parameters at once (see the Notes of :meth:`fit`)."""
        if how != "MLE":
            raise ValueError(
                f"how={how!r}: data with left- or interval-censored rows "
                "(c = -1 or 2) are fitted by maximum likelihood over every "
                "cause at once; use how='MLE'."
            )
        causes = ordered_labels(e[c != 1])
        if isinstance(dist, dict):
            missing = [k for k in causes if k not in dist]
            if missing:
                raise ValueError(
                    "`dist` has no distribution for the cause(s) {}; give "
                    "one for every cause {}.".format(missing, causes)
                )
        dists = [dist[k] if isinstance(dist, dict) else dist for k in causes]
        data = joint.prepare(causes, dists, x, c, n, e, t)
        init = joint.start(causes, dists, x, c, n, e, t)
        found = joint.fit(data, init)
        models = {}
        start = 0
        for k, d, p in zip(causes, dists, found.params):
            stop = start + p.size
            model = d.from_params(p)
            # The cause's parameters and its block of the joint covariance:
            # its Wald bounds hold; it has no data of its own.
            model.method = "MLE"
            model.maximum = found.maximum
            if found.covariance is not None:
                block = np.array(found.covariance[start:stop, start:stop])
                model.hess_inv = block
                model._covariance = block
            models[k] = model
            start = stop
        out = cls()
        out.causes = causes
        out.models = models
        out._joint = {
            "covariance": found.covariance,
            "neg_ll": found.neg_ll,
            "k": found.k,
            "n": found.n_ic,
        }
        return out

    @classmethod
    @column_arguments("x", "c", "n", "tl")
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str | None,
        e_col: str,
        c_col: str | None = None,
        n_col: str | None = None,
        dist: Any = Weibull,
        how: str = "MLE",
        tl_col: str | float | None = None,
        xl_col: str | None = None,
        xr_col: str | None = None,
    ) -> ParametricCompetingRisks:
        """
        Fit from the columns of a :class:`pandas.DataFrame`; see :meth:`fit`.

        Parameters
        ----------
        df : DataFrame
            The data.
        x_col, e_col : str
            The time and cause columns. ``x_col`` is ``None`` where the
            times are given as windows, by ``xl_col`` and ``xr_col``.
        c_col, n_col : str, optional
            The censoring-flag and count columns.
        dist, how : optional
            As for :meth:`fit`.
        tl_col : str or scalar, optional
            The column of each unit's left truncation (entry) time, or a
            number for every unit; the ``tl`` of :meth:`fit` (#571).
        xl_col, xr_col : str, optional
            In place of ``x_col``: the columns of each row's lower and
            upper time, ``[xl, xr]`` in :meth:`fit` (equal for a failure
            or censoring at one time; a blank lower end is a row left
            censored at the upper one, a blank upper end one right
            censored at the lower one).

        Returns
        -------
        ParametricCompetingRisks
            The fitted model.
        """
        if xl_col is not None or xr_col is not None:
            if x_col is not None or xl_col is None or xr_col is None:
                raise ValueError(
                    "Give the times as x_col, or as xl_col and xr_col "
                    "(with x_col=None), not both."
                )
            lower = frame_column(df, xl_col, "xl_col", time=True)
            upper = frame_column(df, xr_col, "xr_col", time=True)
            lower = np.asarray(lower, dtype=float)
            upper = np.asarray(upper, dtype=float)
            x = np.column_stack(
                [
                    np.where(np.isnan(lower), -np.inf, lower),
                    np.where(np.isnan(upper), np.inf, upper),
                ]
            )
        elif x_col is None:
            raise ValueError(
                "Give the times: x_col, or xl_col and xr_col for windows."
            )
        else:
            x = df[x_col].to_numpy()
        e = df[e_col].to_numpy(dtype=object)
        c = None if c_col is None else df[c_col].to_numpy()
        n = None if n_col is None else df[n_col].to_numpy()
        tl = (
            frame_column(df, tl_col, "tl_col", time=True)
            if isinstance(tl_col, str)
            else tl_col
        )
        model = cls.fit(x, e, c=c, n=n, dist=dist, how=how, tl=tl)
        return model


def _ic_terms(model: Any) -> tuple[int, float]:
    """``(k, n)`` of one cause's model: the parameters it estimated and
    the sample size of its BIC (its observed failures). A parametric
    model counts only the parameters it estimated (``_ic_k``); a
    Royston-Parmar model carries both as attributes."""
    if hasattr(model, "_ic_k"):
        return int(model._ic_k()), float(model._ic_sample_size())
    return int(model.k), float(model._ic_n)


def _ff_limit(model: Any) -> float:
    """``lim F(t)`` as ``t -> inf``: 1, or the cure ceiling of a
    limited-failure model.

    ``ff(inf)`` gives it for most families; a few formulas evaluate
    ``inf / inf`` there (the LogLogistic), so fall back to the last finite
    value of ``ff`` on a geometric grid of large times (``ff`` is monotone).
    """
    with np.errstate(all="ignore"):
        val = float(np.ravel(model.ff(np.inf))[0])
        if np.isfinite(val):
            return min(max(val, 0.0), 1.0)
        grid = np.logspace(0, 300, 301)
        vals = np.asarray(model.ff(grid), dtype=float).ravel()
    finite = vals[np.isfinite(vals)]
    return float(min(max(finite[-1], 0.0), 1.0)) if finite.size else 1.0


def _cause_covariance_names(model: Any) -> list[str]:
    """The names of a cause's covariance rows: its ``covariance_names``,
    or its ``parameter_names`` for a model without them."""
    names = getattr(model, "covariance_names", None)
    if names is None:
        names = model.parameter_names
    return list(names)


def _cause_estimates(model: Any) -> npt.NDArray:
    """A cause's fitted values in the order of its covariance names: its
    own ``_estimates()`` where it has one, else (a ``Parametric``) its
    parameters, then ``lfp_p`` and ``f0`` where it has them."""
    own = getattr(model, "_estimates", None)
    if callable(own):
        return np.ravel(np.asarray(own(), dtype=float))
    values = list(np.ravel(np.asarray(model.params, dtype=float)))
    if isinstance(model, Parametric):
        if model.lfp:
            values.append(float(model.lfp_p))
        if model.zi:
            values.append(float(model.f0))
    if len(values) != len(_cause_covariance_names(model)):
        raise ValueError(
            "{}'s parameters do not line up with its covariance_names {}, "
            "so it cannot be part of the competing-risks covariance".format(
                type(model).__name__, _cause_covariance_names(model)
            )
        )
    return np.asarray(values, dtype=float)


def _rebuild_cause(k: Any, model: Any, values: npt.NDArray, gamma: Any) -> Any:
    """Cause ``k``'s model at ``values`` (its covariance names' order, or
    its distribution's parameters only), keeping its options. A
    ``Parametric`` is rebuilt by its distribution's ``from_params``, with
    its offset, ``lfp_p`` and ``f0`` kept unless ``values`` or ``gamma``
    give them; any other model by its own ``with_params``."""
    if isinstance(model, Parametric):
        n_core = len(model.parameter_names)
        extras = dict(model.extras)
        names = _cause_covariance_names(model)
        for name, value in zip(names[n_core:], values[n_core:]):
            extras[name] = float(value)
        if gamma is not None:
            if not model.offset:
                raise ValueError(
                    f"cause {k!r} was not fitted with an offset, so "
                    "'gamma' has no offset of it to replace; leave the "
                    "cause out of 'gamma'"
                )
            extras["gamma"] = float(gamma)
        return model.dist.from_params(
            values[:n_core], **extras, unit=model.unit
        )
    if gamma is not None:
        raise ValueError(
            f"cause {k!r} is not a Parametric model with an offset; leave "
            "it out of 'gamma'"
        )
    rebuild = getattr(model, "with_params", None)
    if not callable(rebuild):
        raise ValueError(
            f"cause {k!r}'s model ({type(model).__name__}) has no "
            "with_params, so it cannot be rebuilt from parameters"
        )
    return rebuild(values)


def _central_differences(
    func: Any, est: npt.NDArray, cov: npt.NDArray, size: int
) -> npt.NDArray:
    """The gradient of ``func`` (a vector of ``size`` values) at ``est``
    by central differences, one row per parameter: each step ``1e-5`` of
    the parameter's value or standard error, whichever is larger. A
    parameter with no variance (fixed) gets a zero row; where a step
    leaves the parameter's range (``from_params`` refuses it), the
    one-sided difference on the other side is used."""
    grad = np.zeros((est.size, size))
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    centre: list = []

    def at(values: npt.NDArray) -> Any:
        try:
            return np.ravel(np.asarray(func(values), dtype=float))
        except ValueError:
            return None

    def mid() -> npt.NDArray:
        if not centre:
            centre.append(np.ravel(np.asarray(func(est), dtype=float)))
        return centre[0]

    for j in range(est.size):
        if not sd[j] > 0:
            continue
        h = 1e-5 * max(abs(float(est[j])), float(sd[j]))
        up, down = est.copy(), est.copy()
        up[j] += h
        down[j] -= h
        f_up, f_down = at(up), at(down)
        if f_up is not None and f_down is not None:
            grad[j] = (f_up - f_down) / (2 * h)
        elif f_up is not None:
            grad[j] = (f_up - mid()) / h
        elif f_down is not None:
            grad[j] = (mid() - f_down) / h
        else:
            grad[j] = np.nan
    return grad


def _checked_sd(
    var: npt.NDArray, t: npt.NDArray, on: str, where: npt.NDArray
) -> npt.NDArray:
    """``sqrt(var)``, ``nan`` with one warning where (among ``where``) the
    delta-method variance is negative or not finite. A variance a hair
    below 0 by rounding (a flat direction) is 0."""
    var = np.where((var < 0) & (var > -1e-14), 0.0, var)
    bad = where & ~(var >= 0) & ~np.isnan(t)
    if bad.any():
        warn_wald_undefined(
            f"{on} at x = {t[bad].tolist()}",
            "its delta-method variance is negative or not finite (the "
            "parameter covariance is not positive definite, or the "
            "function's derivatives overflow there)",
            stacklevel=caller_stacklevel(),
        )
    return np.sqrt(np.where(bad | ~(var >= 0), np.nan, var))
