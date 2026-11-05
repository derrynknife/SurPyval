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
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import IntegrationWarning, quad, quad_vec
from scipy.optimize import brentq

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
from surpyval.univariate.information_criteria import corrected_aic
from surpyval.univariate.parametric import Weibull
from surpyval.univariate.parametric.parametric import Parametric
from surpyval.utils import (
    check_e_and_x,
    resolve_cr_censoring,
    xcnt_handler,
)
from surpyval.utils.dataframe import frame_column
from surpyval.utils.no_maximum import combined_maximum
from surpyval.utils.removed_names import column_arguments
from surpyval.utils.rng import as_generator
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import (
    unknown_cause_error,
    warn_outside_unit_interval,
)


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
    """
    e, c = resolve_cr_censoring(e, c)
    x, c, n, t = xcnt_handler(x, c, n, tl=tl, group_and_sort=False)
    x, c, n = (np.asarray(a, dtype=float) for a in (x, c, n))
    e = np.asarray(e, dtype=object)
    check_e_and_x(e, x)
    if (-1 in c) or (2 in c):
        raise ValueError(
            "Left or interval censoring is not supported by competing risks."
        )
    if any(ev is not None for ev in e[c == 1]) or any(
        ev is None for ev in e[c != 1]
    ):
        raise ValueError(
            "A missing event type (None / NaN) is allowed only for a "
            "censored observation (c = 1), and every censored observation "
            "must have one."
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

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted parametric competing-risks model to a plain,
        JSON-serialisable dict: the list of causes and each cause's fitted
        distribution (via its own ``to_dict``). The reloaded model reproduces
        every cumulative-incidence / hazard function exactly.
        """
        return stamp_schema(
            {
                "model": "ParametricCompetingRisks",
                "causes": to_native(list(self.causes)),
                "models": [
                    self.models[cause].to_dict() for cause in self.causes
                ],
            }
        )

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

    # -- goodness of fit (the joint likelihood factorises over causes) ----

    @property
    def maximum(self) -> str:
        """What the causes' fits reached, one of ``MAXIMUM_STATES``
        (``surpyval.utils.no_maximum``): the worst of the causes' own
        ``maximum``, since the joint likelihood is maximised a cause at a
        time (principles 12 and 13).

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
        """Total negative log-likelihood: the sum over the per-cause fits."""
        return float(sum(self.models[k].neg_ll() for k in self.causes))

    def aic(self) -> float:
        """Akaike information criterion of the joint model,
        ``2 K + 2 neg_ll`` with ``K`` the number of parameters estimated
        over all causes: the sum of the causes' AICs, since both terms
        add over causes."""
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
        # the whole data (see bic).
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
            Observed times.
        e : array_like
            The cause of each observation: any hashable labels (integers,
            strings, tuples, or a mix), kept in sorted order in ``causes``.
            A missing value (``None`` / ``NaN``) marks a censored
            observation with no attributed cause.
        c : array_like, optional
            Censoring flag (0 observed, 1 right-censored). If omitted it is
            derived from ``e`` -- a missing event is censored, an event present
            is observed -- so data can be passed as ``(x, e)`` alone.
            Left/interval censoring is not supported.
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
        distribution is fitted with the same ``tl``. Right truncation and
        interval censoring do not factorise (their terms are integrals of
        the all-cause survival, :math:`\\mathrm{CIF}_k(x_r) -
        \\mathrm{CIF}_k(x_l)` for an interval), so this fit takes neither;
        fitting each cause as a univariate model with the other causes'
        events right censored is then an approximation, not this model's
        likelihood.

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
        """
        x, c, n, e, t = _validate(x, c, n, e, tl)
        truncated = tl is not None

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
    @column_arguments("x", "c", "n", "tl")
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        e_col: str,
        c_col: str | None = None,
        n_col: str | None = None,
        dist: Any = Weibull,
        how: str = "MLE",
        tl_col: str | float | None = None,
    ) -> ParametricCompetingRisks:
        """
        Fit from the columns of a :class:`pandas.DataFrame`; see :meth:`fit`.

        Parameters
        ----------
        df : DataFrame
            The data.
        x_col, e_col : str
            The time and cause columns.
        c_col, n_col : str, optional
            The censoring-flag and count columns.
        dist, how : optional
            As for :meth:`fit`.
        tl_col : str or scalar, optional
            The column of each unit's left truncation (entry) time, or a
            number for every unit; the ``tl`` of :meth:`fit` (#571).

        Returns
        -------
        ParametricCompetingRisks
            The fitted model.
        """
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
