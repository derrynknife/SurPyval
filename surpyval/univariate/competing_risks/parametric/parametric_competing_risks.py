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

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad_vec

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
from surpyval.univariate.parametric import Weibull
from surpyval.univariate.parametric.parametric import Parametric
from surpyval.utils import (
    check_e_and_x,
    resolve_cr_censoring,
    xcnt_handler,
)


def _validate(
    x: npt.ArrayLike,
    c: "npt.ArrayLike | None",
    n: "npt.ArrayLike | None",
    e: npt.ArrayLike,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Wrangle ``x``/``c``/``n`` and check the event labels: ``e`` is the
    per-observation cause. A missing event (``None`` / ``NaN``) marks a
    censored observation; if ``c`` is not given it is derived from the events.
    """
    e, c = resolve_cr_censoring(e, c)
    x, c, n, _ = xcnt_handler(x, c, n, group_and_sort=False)
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
    return x, c, n, e


class ParametricCompetingRisks(SerialisableMixin):
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
    def from_dict(cls, model_dict: dict) -> "ParametricCompetingRisks":
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
            "{}: {}".format(k, self.models[k].dist.name) for k in self.causes
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

    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        """All-cause survival :math:`S(t) = \\prod_k S_k(t)`."""
        return np.exp(-self.Hf(x))

    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        """All-cause failure probability :math:`1 - S(t)` (the total
        cumulative incidence over all causes)."""
        return -np.expm1(-self.Hf(x))

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

    def cif(
        self, x: npt.ArrayLike, event: Any = None
    ) -> "npt.NDArray | float":
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
        self, size: int, random_state: "int | None" = None
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
        rng = np.random.default_rng(random_state)
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

    def neg_ll(self) -> float:
        """Total negative log-likelihood: the sum over the per-cause fits."""
        return float(sum(self.models[k].neg_ll() for k in self.causes))

    def aic(self) -> float:
        """Akaike information criterion of the joint model."""
        return float(sum(self.models[k].aic() for k in self.causes))

    def bic(self) -> float:
        """Bayesian information criterion of the joint model."""
        return float(sum(self.models[k].bic() for k in self.causes))

    # -- helpers ----------------------------------------------------------

    def _check_event(self, event: Any) -> None:
        if event not in self.models:
            raise ValueError(
                "Unknown cause {!r}; fitted causes are {}.".format(
                    event, list(self.causes)
                )
            )

    # -- construction -----------------------------------------------------

    @classmethod
    def from_fitted(cls, models: dict) -> "ParametricCompetingRisks":
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
    def fit(
        cls,
        x: npt.ArrayLike,
        e: npt.ArrayLike,
        c: "npt.ArrayLike | None" = None,
        n: "npt.ArrayLike | None" = None,
        dist: Any = Weibull,
        how: str = "MLE",
    ) -> "ParametricCompetingRisks":
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

        Returns
        -------
        ParametricCompetingRisks
            The fitted model.

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
        x, c, n, e = _validate(x, c, n, e)

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
            models[k] = distribution.fit(x=x, c=c_k, n=n, how=how)

        model = cls()
        model.causes = causes
        model.models = models
        return model

    @classmethod
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        e_col: str,
        c_col: "str | None" = None,
        n_col: "str | None" = None,
        dist: Any = Weibull,
        how: str = "MLE",
    ) -> "ParametricCompetingRisks":
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

        Returns
        -------
        ParametricCompetingRisks
            The fitted model.
        """
        x = df[x_col].to_numpy()
        e = df[e_col].to_numpy(dtype=object)
        c = None if c_col is None else df[c_col].to_numpy()
        n = None if n_col is None else df[n_col].to_numpy()
        model = cls.fit(x, e, c=c, n=n, dist=dist, how=how)
        return model


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
